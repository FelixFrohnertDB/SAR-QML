"""Capacity-matched reduced-state classifier for unfocused SAR phase histories.

Supporting code for ``notebooks/fig_07_sar_geometry_classification.ipynb``.
A ``Q x Q`` complex phase-history patch (azimuth index ``a``, range index
``d``) is amplitude-encoded on two ``n``-qubit registers, multiplied by the
fixed diagonal query ``exp(-i f a~ d~)``, and the range register is traced
out. The reduced azimuth state is read out by a shallow trainable residual
circuit followed by a fixed inverse Fourier transform; the class is the
measured azimuth cell.

Contents: the scene generator, the query and partial trace, the classifier,
Adam training over paired restarts, the three experiments (query scan,
geometry tracking, sample efficiency), the validity checks and the draw
bootstrap.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .common import balanced_accuracy, balanced_labels, seed_from
from .gates import fourier_readout, ry, rz
from .geometry import mixed_derivative_relative_error, register_z_signs

TWO_PI = 2 * np.pi

RAW_COLUMNS = [
    "run_id",
    "task_id",
    "experiment",
    "draw",
    "range_m",
    "f_geom",
    "f",
    "f_over_star",
    "arm",
    "n_train",
    "train_loss",
    "best_restart",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_nll",
    "compact_qft_accuracy",
]


# --------------------------------------------------------------------------
# scenes and query
# --------------------------------------------------------------------------


def scenes(
    key,
    batch_size,
    f_geom,
    *,
    h_exact,
    sensor_phase,
    snr_db,
    n_clutter,
    clutter_rms,
    azimuth_jitter,
    target_range_jitter,
    clutter_range_ratio,
):
    """Balanced target-cell labels and normalized unfocused complex histories.

    The labelled target sits in azimuth cell ``label`` (with a continuous
    sub-cell offset of at most ``azimuth_jitter`` cell widths) at coupling
    ``f_geom`` times ``1 + target_range_jitter * N(0, 1)``. ``n_clutter``
    off-range scatterers have couplings ``f_geom * U(clutter_range_ratio)``.
    Complex Gaussian noise at ``snr_db`` and the fixed azimuth-channel phase
    ``sensor_phase`` are added, and each patch is normalized to unit norm.
    Returns ``(signal, labels)`` with ``signal`` of shape ``(batch, Q, Q)``.
    """
    q = h_exact.shape[0]
    cell_width = TWO_PI / q
    index = jnp.asarray(np.arange(q, dtype=float))
    h_exact = jnp.asarray(h_exact)
    sensor_phase = jnp.asarray(sensor_phase)

    keys = jax.random.split(key, 13)
    labels = balanced_labels(keys[0], batch_size, q)

    u_center = TWO_PI * labels / q
    u_target = jnp.mod(
        u_center
        + azimuth_jitter
        * cell_width
        * jax.random.uniform(keys[1], (batch_size,), minval=-1.0, maxval=1.0),
        TWO_PI,
    )
    v_target = jax.random.uniform(keys[2], (batch_size,), minval=0.0, maxval=TWO_PI)
    f_target = f_geom * (
        1.0 + target_range_jitter * jax.random.normal(keys[3], (batch_size,))
    )
    target_reflectivity = jnp.exp(
        1j * jax.random.uniform(keys[4], (batch_size,), minval=0.0, maxval=TWO_PI)
    )

    target = target_reflectivity[:, None, None] * jnp.exp(
        -1j
        * (
            u_target[:, None, None] * index[None, :, None]
            + v_target[:, None, None] * index[None, None, :]
        )
        + 1j * f_target[:, None, None] * h_exact[None, :, :]
    )

    if n_clutter:
        u_clutter = jax.random.uniform(
            keys[5], (batch_size, n_clutter), minval=0.0, maxval=TWO_PI
        )
        v_clutter = jax.random.uniform(
            keys[6], (batch_size, n_clutter), minval=0.0, maxval=TWO_PI
        )
        range_ratio = jax.random.uniform(
            keys[7],
            (batch_size, n_clutter),
            minval=clutter_range_ratio[0],
            maxval=clutter_range_ratio[1],
        )
        clutter_reflectivity = (
            clutter_rms
            / np.sqrt(2 * n_clutter)
            * (
                jax.random.normal(keys[8], (batch_size, n_clutter))
                + 1j * jax.random.normal(keys[9], (batch_size, n_clutter))
            )
        )
        clutter_phase = (
            -1j
            * (
                u_clutter[:, :, None, None] * index[None, None, :, None]
                + v_clutter[:, :, None, None] * index[None, None, None, :]
            )
            + 1j
            * (f_geom * range_ratio)[:, :, None, None]
            * h_exact[None, None, :, :]
        )
        clutter = jnp.sum(
            clutter_reflectivity[:, :, None, None] * jnp.exp(clutter_phase),
            axis=1,
        )
    else:
        clutter = jnp.zeros_like(target)

    signal = target + clutter
    rms = jnp.sqrt(jnp.mean(jnp.abs(signal) ** 2, axis=(1, 2), keepdims=True))
    sigma = rms * 10 ** (-snr_db / 20) / np.sqrt(2)
    noise = sigma * (
        jax.random.normal(keys[10], signal.shape)
        + 1j * jax.random.normal(keys[11], signal.shape)
    )
    signal = (signal + noise) * sensor_phase[None, :, None]
    signal = signal / jnp.linalg.norm(signal.reshape(batch_size, -1), axis=1)[:, None, None]
    return signal, labels


def rho_azimuth(signal, f, generator):
    """Apply the diagonal query ``exp(-i f generator)`` and trace out range.

    ``generator`` is the compact ``a~ d~`` for the circuit, or the exact
    interaction for reference computations. Returns the reduced azimuth
    states, shape ``(batch, Q, Q)``.
    """
    generator = jnp.asarray(generator)
    queried = signal * jnp.exp(-1j * f * generator)[None, :, :]
    # Contract d, the second register. Contracting a here would be a silent bug.
    return jnp.einsum("bad,bAd->baA", queried, jnp.conj(queried))


def purity(rho):
    """Purity ``tr(rho^2) / tr(rho)^2`` of a batch of density matrices."""
    trace = jnp.real(jnp.trace(rho, axis1=1, axis2=2))
    return jnp.real(jnp.einsum("bij,bji->b", rho, rho)) / trace**2


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------


def _kron_all(matrices):
    result = jnp.eye(1, dtype=jnp.complex128)
    for matrix in matrices:
        result = jnp.kron(result, matrix)
    return result


def _cz_ring_matrix(n):
    diagonal = np.ones(2**n)
    for state in range(2**n):
        bits = [(state >> (n - 1 - q)) & 1 for q in range(n)]
        phase = 1
        for q in range(n):
            if bits[q] and bits[(q + 1) % n]:
                phase *= -1
        diagonal[state] = phase
    return jnp.diag(jnp.asarray(diagonal, dtype=jnp.complex128))


@dataclass(frozen=True, eq=False)
class ReadoutClassifier:
    """Trainable residual circuit plus fixed Fourier readout on ``n`` qubits.

    Each of the ``layers`` layers applies ``R_Y`` then ``R_Z`` to every qubit
    in turn, followed by a CZ ring. Parameters have shape ``(layers, n, 2)``
    with the last axis ordered (``R_Y``, ``R_Z``); at zero parameters the
    residual circuit is the identity.
    """

    n_qubits: int
    layers: int
    identity: jax.Array = field(init=False, repr=False)
    cz_ring: jax.Array = field(init=False, repr=False)
    fourier: jax.Array = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "identity", jnp.eye(2, dtype=jnp.complex128))
        object.__setattr__(self, "cz_ring", _cz_ring_matrix(self.n_qubits))
        object.__setattr__(self, "fourier", fourier_readout(2**self.n_qubits))

    def residual_unitary(self, parameters):
        """Unitary of the trainable residual circuit."""
        n = self.n_qubits
        parameters = jnp.asarray(parameters).reshape(self.layers, n, 2)
        unitary = jnp.eye(2**n, dtype=jnp.complex128)
        for layer in range(self.layers):
            for qubit in range(n):
                for gate_index, gate in enumerate((ry, rz)):
                    matrices = [self.identity] * n
                    matrices[qubit] = gate(parameters[layer, qubit, gate_index])
                    unitary = _kron_all(matrices) @ unitary
            unitary = self.cz_ring @ unitary
        return unitary

    def readout_unitary(self, parameters):
        """Fourier readout after the residual circuit."""
        return self.fourier @ self.residual_unitary(parameters)

    def probabilities(self, parameters, rho):
        """Class probabilities: basis populations after the readout."""
        unitary = self.readout_unitary(parameters)
        probabilities_ = jnp.real(
            jnp.einsum("ja,baA,jA->bj", unitary, rho, jnp.conj(unitary))
        )
        probabilities_ = jnp.clip(probabilities_, 1e-12, None)
        return probabilities_ / probabilities_.sum(axis=1, keepdims=True)

    def cross_entropy(self, parameters, rho, labels):
        """Mean negative log-likelihood of the labels."""
        p = self.probabilities(parameters, rho)
        return -jnp.mean(jnp.log(p[jnp.arange(len(labels)), labels]))

    def calibration_parameters(self, sensor_z_phases):
        """Parameters whose first ``R_Z`` layer inverts the channel phase."""
        parameters = np.zeros((self.layers, self.n_qubits, 2))
        parameters[0, :, 1] = -sensor_z_phases
        return parameters


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def make_adam_runner(loss, *, steps, learning_rate):
    """Jitted Adam over a batch of restarts of ``loss(parameters, rho, labels)``.

    Gradients are clipped to norm 10 per restart, the learning rate follows a
    cosine schedule from ``learning_rate`` to ``0.1 * learning_rate``, and each
    restart keeps its lowest-loss parameters. The returned function maps
    ``(initial_parameters, rho, labels)`` to ``(best_parameters, best_losses)``.
    """
    loss_and_gradient = jax.vmap(
        jax.value_and_grad(loss),
        in_axes=(0, None, None),
    )
    loss_batch = jax.vmap(loss, in_axes=(0, None, None))

    @jax.jit
    def run(initial_parameters, rho, labels):
        parameters = jnp.asarray(initial_parameters)
        first = jnp.zeros_like(parameters)
        second = jnp.zeros_like(parameters)
        best_parameters = parameters
        best_losses = loss_batch(parameters, rho, labels)

        def body(step, state):
            parameters, first, second, best_parameters, best_losses = state
            _, gradient = loss_and_gradient(parameters, rho, labels)
            gradient_norm = jnp.linalg.norm(
                gradient.reshape(gradient.shape[0], -1), axis=1
            )
            scale = jnp.minimum(1.0, 10.0 / (gradient_norm + 1e-12))
            gradient = gradient * scale.reshape((-1,) + (1,) * (gradient.ndim - 1))

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            t = step + 1
            first_hat = first / (1.0 - 0.9**t)
            second_hat = second / (1.0 - 0.999**t)
            fraction = step / jnp.maximum(1, steps - 1)
            rate = learning_rate * (
                0.1 + 0.9 * 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
            )
            parameters = parameters - rate * first_hat / (
                jnp.sqrt(second_hat) + 1e-8
            )
            new_losses = loss_batch(parameters, rho, labels)
            improved = new_losses < best_losses
            best_losses = jnp.where(improved, new_losses, best_losses)
            best_parameters = jnp.where(
                improved.reshape((-1,) + (1,) * (parameters.ndim - 1)),
                parameters,
                best_parameters,
            )
            return parameters, first, second, best_parameters, best_losses

        _, _, _, best_parameters, best_losses = jax.lax.fori_loop(
            0,
            steps,
            body,
            (parameters, first, second, best_parameters, best_losses),
        )
        return best_parameters, best_losses

    return run


def paired_initializations(seed, *, n_restarts, layers, n_qubits):
    """Restart initializations ``0.15 N(0, 1)``; restart 0 is all zeros."""
    rng = np.random.default_rng(seed)
    shape = (n_restarts, layers, n_qubits, 2)
    parameters = 0.15 * rng.normal(size=shape)
    parameters[0] = 0.0
    return parameters


def make_readout_fitter(classifier, *, n_restarts, steps, learning_rate):
    """Return ``fit(rho_train, labels_train, rho_test=None, labels_test=None, *, seed)``.

    ``fit`` trains all paired restarts with Adam, selects the restart with
    the lowest training loss and, if test states are given, evaluates it.
    The result dict holds ``train_loss`` and ``best_restart`` and optionally
    ``test_accuracy``, ``test_balanced_accuracy`` and ``test_nll``.
    """
    runner = make_adam_runner(
        classifier.cross_entropy, steps=steps, learning_rate=learning_rate
    )
    n_classes = 2**classifier.n_qubits

    def fit(rho_train, labels_train, rho_test=None, labels_test=None, *, seed):
        initial = paired_initializations(
            seed,
            n_restarts=n_restarts,
            layers=classifier.layers,
            n_qubits=classifier.n_qubits,
        )
        candidates, losses = runner(
            jnp.asarray(initial), jnp.asarray(rho_train), jnp.asarray(labels_train)
        )
        losses = np.asarray(losses)
        best_index = int(np.argmin(losses))
        best_parameters = np.asarray(candidates[best_index])
        result = {
            "train_loss": float(losses[best_index]),
            "best_restart": best_index,
        }
        if rho_test is not None:
            p = np.asarray(classifier.probabilities(best_parameters, rho_test))
            labels = np.asarray(labels_test, dtype=int)
            prediction = np.argmax(p, axis=1)
            result.update(
                test_accuracy=float(np.mean(prediction == labels)),
                test_balanced_accuracy=balanced_accuracy(labels, prediction, n_classes),
                test_nll=float(-np.mean(np.log(p[np.arange(len(labels)), labels]))),
            )
        return result

    return fit


def fixed_fourier_accuracy(classifier, rho, labels, parameters):
    """Balanced accuracy of the untrained readout at fixed ``parameters``."""
    p = np.asarray(classifier.probabilities(parameters, rho))
    prediction = np.argmax(p, axis=1)
    return balanced_accuracy(labels, prediction, 2**classifier.n_qubits)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


def run_query_scan(
    log,
    *,
    fit,
    scenes,
    h_compact,
    f_star,
    reference_range,
    scales,
    queries,
    n_draws,
    n_train,
    n_test,
    base_seed,
):
    """Retrain at every query ``f = scale * f_star`` on identical scenes."""
    for draw in range(n_draws):
        train_signal, train_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 100, draw)), n_train, f_star
        )
        test_signal, test_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 101, draw)), n_test, f_star
        )
        paired_seed = seed_from(base_seed, 102, draw)
        for index, (scale, f) in enumerate(zip(scales, queries)):
            task_id = f"scan-d{draw:03d}-f{index:03d}"
            if task_id in log:
                continue
            result = fit(
                rho_azimuth(train_signal, f, h_compact),
                train_labels,
                rho_azimuth(test_signal, f, h_compact),
                test_labels,
                seed=paired_seed,
            )
            log.record(
                task_id,
                {
                    "experiment": "scan",
                    "draw": draw,
                    "range_m": reference_range,
                    "f_geom": f_star,
                    "f": float(f),
                    "f_over_star": float(scale),
                    "arm": "scan",
                    "n_train": n_train,
                    **result,
                },
            )
            print(
                f"scan draw={draw:02d} f/f*={scale:.3f} "
                f"acc={result['test_balanced_accuracy']:.3f}",
                flush=True,
            )


def run_geometry_tracking(
    log,
    *,
    fit,
    scenes,
    h_compact,
    ranges,
    f_geoms,
    f_grid,
    n_draws,
    n_train,
    base_seed,
):
    """Training-loss scans over ``f_grid`` for data at several reference ranges."""
    n_before = len(log)
    for geometry_index, (range_m, f_geom) in enumerate(zip(ranges, f_geoms)):
        for draw in range(n_draws):
            train_signal, train_labels = scenes(
                jax.random.PRNGKey(seed_from(base_seed, 200, geometry_index, draw)),
                n_train,
                float(f_geom),
            )
            paired_seed = seed_from(base_seed, 201, geometry_index, draw)
            for f_index, f in enumerate(f_grid):
                task_id = f"recovery-g{geometry_index:02d}-d{draw:03d}-f{f_index:03d}"
                if task_id in log:
                    continue
                result = fit(
                    rho_azimuth(train_signal, f, h_compact),
                    train_labels,
                    seed=paired_seed,
                )
                log.record(
                    task_id,
                    {
                        "experiment": "recovery",
                        "draw": draw,
                        "range_m": float(range_m),
                        "f_geom": float(f_geom),
                        "f": float(f),
                        "f_over_star": float(f / f_geom),
                        "arm": "scan",
                        "n_train": n_train,
                        **result,
                    },
                )
            if len(log) > n_before:
                print(
                    f"recovery range={range_m:.0f} m draw={draw:02d} complete",
                    flush=True,
                )
            n_before = len(log)


def run_sample_efficiency(
    log,
    *,
    fit,
    scenes,
    classifier,
    calibration,
    h_compact,
    h_exact,
    arms,
    f_star,
    reference_range,
    n_train_values,
    n_draws,
    n_test,
    base_seed,
):
    """Compare the query arms on nested prefixes of one training pool.

    For every draw the calibrated, untrained Fourier readout after the exact
    and the compact query is also recorded on the same test scenes.
    """
    n_max = max(n_train_values)
    n_before = len(log)
    for draw in range(n_draws):
        pool_signal, pool_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 300, draw)), n_max, f_star
        )
        test_signal, test_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 301, draw)), n_test, f_star
        )

        # Fixed Fourier references are properties of the same test scenes.
        reference_task = f"reference-d{draw:03d}"
        if reference_task not in log:
            exact_accuracy = fixed_fourier_accuracy(
                classifier,
                rho_azimuth(test_signal, f_star, h_exact),
                test_labels,
                calibration,
            )
            compact_accuracy = fixed_fourier_accuracy(
                classifier,
                rho_azimuth(test_signal, f_star, h_compact),
                test_labels,
                calibration,
            )
            log.record(
                reference_task,
                {
                    "experiment": "reference",
                    "draw": draw,
                    "range_m": reference_range,
                    "f_geom": f_star,
                    "f": f_star,
                    "f_over_star": 1.0,
                    "arm": "exact_qft",
                    "n_train": 0,
                    "train_loss": np.nan,
                    "best_restart": np.nan,
                    "test_accuracy": exact_accuracy,
                    "test_balanced_accuracy": exact_accuracy,
                    "test_nll": np.nan,
                    "compact_qft_accuracy": compact_accuracy,
                },
            )

        for n_train in n_train_values:
            paired_seed = seed_from(base_seed, 302, draw, n_train)
            for arm, f in arms.items():
                task_id = f"efficiency-d{draw:03d}-n{n_train:05d}-{arm}"
                if task_id in log:
                    continue
                result = fit(
                    rho_azimuth(pool_signal[:n_train], f, h_compact),
                    pool_labels[:n_train],
                    rho_azimuth(test_signal, f, h_compact),
                    test_labels,
                    seed=paired_seed,
                )
                log.record(
                    task_id,
                    {
                        "experiment": "efficiency",
                        "draw": draw,
                        "range_m": reference_range,
                        "f_geom": f_star,
                        "f": float(f),
                        "f_over_star": float(f / f_star),
                        "arm": arm,
                        "n_train": int(n_train),
                        **result,
                    },
                )
            if len(log) > n_before:
                print(
                    f"efficiency draw={draw:02d} N={n_train:04d} complete",
                    flush=True,
                )
            n_before = len(log)


# --------------------------------------------------------------------------
# validity checks
# --------------------------------------------------------------------------


def geometry_checks(
    *,
    h_exact,
    kx_center,
    kr_center,
    kappa,
    bilinear_coefficient,
    bilinear_r2,
    maximum_aligned_residual,
):
    """Mixed derivative, removed separable terms and compact approximation."""
    checks = {}
    checks["mixed derivative relative error"] = mixed_derivative_relative_error(
        kx_center, kr_center, kappa
    )
    assert checks["mixed derivative relative error"] < 2e-5

    checks["interaction row mean"] = float(np.max(np.abs(h_exact.mean(axis=1))))
    checks["interaction column mean"] = float(np.max(np.abs(h_exact.mean(axis=0))))
    checks["bilinear coefficient"] = bilinear_coefficient
    checks["bilinear R2"] = bilinear_r2
    checks["aligned residual max rad"] = maximum_aligned_residual
    assert checks["interaction row mean"] < 1e-8
    assert checks["interaction column mean"] < 1e-8
    assert abs(bilinear_coefficient - 1.0) < 0.01
    assert bilinear_r2 > 0.995
    return checks


def compiled_query_phase(f, n):
    """Diagonal of the product of the ``n^2`` cross-register ``R_ZZ`` gates."""
    q = 2**n
    weights = 2 ** np.arange(n - 1, -1, -1)
    z = register_z_signs(n)
    exponent = np.zeros((q, q))
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            theta = f * wi * wj / 2.0
            exponent += -0.5 * theta * np.outer(z[:, i], z[:, j])
    return np.exp(1j * exponent)


def circuit_checks(classifier, *, h_compact, sensor_phase, sensor_z_phases):
    """``R_ZZ`` compilation, identity at zero and exact channel calibration."""
    n = classifier.n_qubits
    checks = {}
    checks["RZZ decomposition"] = float(
        np.max(np.abs(compiled_query_phase(0.731, n) - np.exp(-1j * 0.731 * h_compact)))
    )
    checks["two-register RZZ gates"] = n * n
    assert checks["RZZ decomposition"] < 1e-12

    zero_parameters = np.zeros((classifier.layers, n, 2))
    checks["zero residual identity"] = float(
        np.max(
            np.abs(np.asarray(classifier.residual_unitary(zero_parameters)) - np.eye(2**n))
        )
    )
    assert checks["zero residual identity"] < 1e-12

    calibration = classifier.calibration_parameters(sensor_z_phases)
    checks["sensor calibration exactness"] = float(
        np.max(
            np.abs(
                np.asarray(classifier.residual_unitary(calibration))
                - np.diag(np.conj(sensor_phase))
            )
        )
    )
    assert checks["sensor calibration exactness"] < 1e-12
    return checks


def scene_checks(scenes, *, h_compact, f_star, declared_sizes, base_seed):
    """Reduced states are density matrices; declared sizes are balanced."""
    q = h_compact.shape[0]
    checks = {}
    check_signal, _ = scenes(
        jax.random.PRNGKey(seed_from(base_seed, 10)), 8 * q, f_star
    )
    check_rho = np.asarray(rho_azimuth(check_signal, f_star, h_compact))
    checks["trace error"] = float(
        np.max(np.abs(np.trace(check_rho, axis1=1, axis2=2) - 1.0))
    )
    checks["Hermiticity error"] = float(
        np.max(np.abs(check_rho - np.conj(np.swapaxes(check_rho, 1, 2))))
    )
    checks["minimum eigenvalue"] = float(np.linalg.eigvalsh(check_rho).min())
    assert checks["trace error"] < 1e-11
    assert checks["Hermiticity error"] < 1e-11
    assert checks["minimum eigenvalue"] > -1e-10

    for size in set(declared_sizes):
        _, labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 11, size)), size, f_star
        )
        counts = np.bincount(np.asarray(labels), minlength=q)
        assert counts.max() == counts.min()
    checks["balanced declared sizes"] = True
    return checks


def contrast_checks(
    classifier, scenes, *, h_exact, h_compact, f_star, arms, calibration, base_seed
):
    """Clean-target purities and the calibrated untrained contrast of the arms."""
    q = h_exact.shape[0]
    index = np.arange(q, dtype=float)
    u0, v0 = 1.231, 2.137
    clean = (
        np.exp(-1j * (u0 * index[:, None] + v0 * index[None, :]))
        * np.exp(1j * f_star * h_exact)
    )[None] / q
    purities = {}
    for name, f, generator in [
        ("exact", f_star, h_exact),
        ("compact", f_star, h_compact),
        ("wrong", arms["wrong"], h_compact),
        ("none", arms["none"], h_compact),
    ]:
        purities[name] = float(purity(rho_azimuth(jnp.asarray(clean), f, generator))[0])
    checks = {f"clean purity {name}": value for name, value in purities.items()}
    assert abs(purities["exact"] - 1.0) < 1e-12
    assert purities["compact"] > 0.98
    assert purities["none"] < 0.2

    # known inverse sensor phase, used only to audit the generator
    probe_signal, probe_labels = scenes(
        jax.random.PRNGKey(seed_from(base_seed, 12)), 800, f_star
    )
    oracle_accuracy = {}
    for name, f in arms.items():
        oracle_accuracy[name] = fixed_fourier_accuracy(
            classifier,
            rho_azimuth(probe_signal, f, h_compact),
            probe_labels,
            calibration,
        )
    checks.update(
        {f"probe accuracy {name}": value for name, value in oracle_accuracy.items()}
    )
    assert oracle_accuracy["aligned"] > oracle_accuracy["wrong"] + 0.25
    assert oracle_accuracy["aligned"] > 0.60
    assert abs(oracle_accuracy["none"] - 1.0 / q) < 0.08
    return checks


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap_matrix(values, *, n_samples, base_seed, seed):
    """Draw bootstrap of the column means of ``values`` (draws x points).

    Returns the mean, the 2.5% and 97.5% quantiles of the bootstrap means
    (``lower``, ``upper``) and the bootstrap means themselves (``draws``).
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_from(base_seed, 500, seed))
    indices = rng.integers(0, values.shape[0], (n_samples, values.shape[0]))
    draws = values[indices].mean(axis=1)
    return {
        "mean": values.mean(axis=0),
        "lower": np.quantile(draws, 0.025, axis=0),
        "upper": np.quantile(draws, 0.975, axis=0),
        "draws": draws,
    }


def interpolated_argmin(x, y):
    """Argmin of sampled ``y(x)``, refined by a local quadratic.

    The vertex of the parabola through the minimum and its two neighbours is
    used only if the parabola is convex and the vertex lies between the
    neighbours. Returns ``(x_min, method)`` with method ``"quadratic"``,
    ``"sampled"`` or ``"grid_edge"``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    index = int(np.argmin(y))
    if index == 0 or index == len(x) - 1:
        return float(x[index]), "grid_edge"
    x_local = x[index - 1 : index + 2]
    y_local = y[index - 1 : index + 2]
    a, b, _ = np.polyfit(x_local, y_local, 2)
    if a <= 0:
        return float(x[index]), "sampled"
    vertex = -b / (2 * a)
    if not (x_local[0] <= vertex <= x_local[-1]):
        return float(x[index]), "sampled"
    return float(vertex), "quadratic"
