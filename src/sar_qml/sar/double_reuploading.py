"""Double data re-uploading classifier with a geometry-fixed query.

Supporting code for ``notebooks/fig_08_double_reuploading_geometry.ipynb``.
Two ``n``-qubit registers ``A`` (azimuth) and ``D`` (range) start in the
uniform superposition. Each of the ``L`` layers applies a trainable product
block ``W_A x W_D``, the phase-only data upload ``E_x = diag(exp(i x / L))``,
a second product block ``V_A x V_D`` and the fixed query
``E_f = diag(exp(-i (f / L) a~ d~))``. A fixed inverse Fourier transform on
``A`` follows, ``D`` is traced out and the class is the measured azimuth cell.

Contents: the scene generator and phase encoding, the circuit, the query
arms, Adam training over paired restarts, the three experiments (query scan,
validation profiles across acquisitions, sample efficiency), the validity
checks, the draw bootstrap and two plotting helpers.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from .common import balanced_accuracy, balanced_labels, seed_from
from .gates import fourier_readout, ry, rz
from .geometry import (
    compact_interaction,
    mixed_derivative_relative_error,
    register_z_signs,
)

TWO_PI = 2 * np.pi

ARMS = ("aligned", "mismatched", "none", "learned")

RAW_COLUMNS = [
    "run_id",
    "task_id",
    "experiment",
    "draw",
    "geometry",
    "f_geometry",
    "arm",
    "query_value",
    "query_ratio",
    "fitted_query",
    "n_train",
    "train_loss",
    "best_restart",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_nll",
    "validation_balanced_accuracy",
    "validation_nll",
]


# --------------------------------------------------------------------------
# scenes and encoding
# --------------------------------------------------------------------------


def scenes(
    key,
    batch_size,
    *,
    geometry,
    sensor_phase,
    snr_db,
    n_clutter,
    clutter_rms,
    azimuth_jitter,
    clutter_range_ratio,
):
    """Complex phase-history patches and labels at one acquisition geometry.

    The labelled target sits in azimuth cell ``label`` (with a continuous
    sub-cell offset of at most ``azimuth_jitter`` cell widths) at the fixed
    coupling ``geometry["f_geometry"]``. ``n_clutter`` off-range scatterers
    have couplings ``f_geometry * U(clutter_range_ratio)``. Complex Gaussian
    noise at ``snr_db`` and the fixed azimuth-channel phase ``sensor_phase``
    are added. Returns ``(signal, labels)`` with ``signal`` of shape
    ``(batch, Q, Q)``; the patches are not normalized.
    """
    h_exact = geometry["h_exact_jax"]
    f_geometry = geometry["f_geometry"]
    q = h_exact.shape[0]
    cell_width = TWO_PI / q
    index = jnp.asarray(np.arange(q, dtype=float))
    sensor_phase = jnp.asarray(sensor_phase)

    keys = jax.random.split(key, 11)
    labels = balanced_labels(keys[0], batch_size, q)
    u_target = jnp.mod(
        TWO_PI * labels / q
        + azimuth_jitter
        * cell_width
        * jax.random.uniform(keys[1], (batch_size,), minval=-1.0, maxval=1.0),
        TWO_PI,
    )
    v_target = jax.random.uniform(
        keys[2], (batch_size,), minval=0.0, maxval=TWO_PI
    )
    reflectivity = jnp.exp(
        1j * jax.random.uniform(keys[3], (batch_size,), minval=0.0, maxval=TWO_PI)
    )

    target = reflectivity[:, None, None] * jnp.exp(
        -1j
        * (
            u_target[:, None, None] * index[None, :, None]
            + v_target[:, None, None] * index[None, None, :]
        )
        + 1j * f_geometry * h_exact[None, :, :]
    )

    if n_clutter:
        u_clutter = jax.random.uniform(
            keys[4], (batch_size, n_clutter), minval=0.0, maxval=TWO_PI
        )
        v_clutter = jax.random.uniform(
            keys[5], (batch_size, n_clutter), minval=0.0, maxval=TWO_PI
        )
        range_ratio = jax.random.uniform(
            keys[6],
            (batch_size, n_clutter),
            minval=clutter_range_ratio[0],
            maxval=clutter_range_ratio[1],
        )
        clutter_reflectivity = (
            clutter_rms
            / np.sqrt(2 * n_clutter)
            * (
                jax.random.normal(keys[7], (batch_size, n_clutter))
                + 1j * jax.random.normal(keys[8], (batch_size, n_clutter))
            )
        )
        clutter_phase = (
            -1j
            * (
                u_clutter[:, :, None, None] * index[None, None, :, None]
                + v_clutter[:, :, None, None] * index[None, None, None, :]
            )
            + 1j
            * (f_geometry * range_ratio)[:, :, None, None]
            * h_exact[None, None, :, :]
        )
        clutter = jnp.sum(
            clutter_reflectivity[:, :, None, None] * jnp.exp(clutter_phase), axis=1
        )
    else:
        clutter = jnp.zeros_like(target)

    signal = target + clutter
    rms = jnp.sqrt(jnp.mean(jnp.abs(signal) ** 2, axis=(1, 2), keepdims=True))
    sigma = rms * 10 ** (-snr_db / 20) / np.sqrt(2)
    noise = sigma * (
        jax.random.normal(keys[9], signal.shape)
        + 1j * jax.random.normal(keys[10], signal.shape)
    )
    signal = (signal + noise) * sensor_phase[None, :, None]
    return signal, labels


def encode(signal):
    """Phase-only variables ``x = arg s`` uploaded by ``E_x``, flattened over (a, d)."""
    return jnp.angle(signal).reshape(signal.shape[0], -1)


# --------------------------------------------------------------------------
# circuit
# --------------------------------------------------------------------------


def _register_cz_diagonal(n):
    """CZ cycles within A and D, with no cross-register coupling."""
    num_qubits = 2 * n
    diagonal = np.ones(2**num_qubits)
    edges = []
    for offset in (0, n):
        edges.extend((offset + qubit, offset + (qubit + 1) % n) for qubit in range(n))
    for state in range(2**num_qubits):
        bits = [(state >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        phase = 1
        for left, right in edges:
            if bits[left] and bits[right]:
                phase *= -1
        diagonal[state] = phase
    return jnp.asarray(diagonal, dtype=jnp.complex128)


@dataclass(frozen=True, eq=False)
class DoubleReuploadingCircuit:
    """Double data re-uploading circuit on two ``n``-qubit registers.

    The parameter vector holds, for every layer, the ``(R_Y, R_Z)`` angles of
    all ``2n`` qubits for ``W`` and then for ``V``, followed by one raw
    global query that is used only when ``learn_query`` is nonzero and is
    mapped to ``query_bound * tanh(raw)``.
    """

    n_qubits: int
    layers: int
    query_bound: float
    register_cz: jax.Array = field(init=False, repr=False)
    fourier: jax.Array = field(init=False, repr=False)
    h_compact_flat: jax.Array = field(init=False, repr=False)

    def __post_init__(self):
        q = 2**self.n_qubits
        object.__setattr__(self, "register_cz", _register_cz_diagonal(self.n_qubits))
        object.__setattr__(self, "fourier", fourier_readout(q))
        object.__setattr__(
            self, "h_compact_flat", jnp.asarray(compact_interaction(q).reshape(-1))
        )

    @property
    def q(self):
        """Register dimension, equal to the number of classes."""
        return 2**self.n_qubits

    @property
    def num_qubits(self):
        return 2 * self.n_qubits

    @property
    def dim(self):
        return self.q * self.q

    @property
    def parameters_per_layer(self):
        """``W`` (2 angles x 2n qubits) plus ``V`` (the same)."""
        return 4 * self.num_qubits

    @property
    def n_variational_parameters(self):
        return self.layers * self.parameters_per_layer

    @property
    def n_parameters(self):
        """Variational parameters plus the optional learned global query."""
        return self.n_variational_parameters + 1

    def single_qubit_layer(self, angles):
        """Tensor product of ``R_Z R_Y`` on all ``2n`` qubits."""
        unitary = jnp.eye(1, dtype=jnp.complex128)
        for qubit in range(self.num_qubits):
            unitary = jnp.kron(unitary, rz(angles[qubit, 1]) @ ry(angles[qubit, 0]))
        return unitary

    def trainable_block(self, angles):
        """A product ``U_A x U_D`` with entanglers only within each register."""
        return self.register_cz[:, None] * self.single_qubit_layer(angles)

    def unpack(self, parameters):
        """Per-layer ``(W, V)`` angle arrays and the raw global query."""
        parameters = jnp.asarray(parameters)
        n = self.num_qubits
        layers = []
        for layer in range(self.layers):
            offset = layer * self.parameters_per_layer
            layers.append(
                (
                    parameters[offset : offset + 2 * n].reshape(n, 2),
                    parameters[offset + 2 * n : offset + 4 * n].reshape(n, 2),
                )
            )
        return layers, parameters[-1]

    def resolved_query(self, parameters, base_query=0.0, learn_query=0.0):
        """Applied query: ``base_query + learn_query * bound * tanh(raw)``."""
        _, query_raw = self.unpack(parameters)
        learned = self.query_bound * jnp.tanh(query_raw)
        return base_query + learn_query * learned

    def state(self, parameters, phases, base_query=0.0, learn_query=0.0):
        """Amplitudes after the fixed azimuth Fourier readout, shape ``(b, Q, Q)``."""
        layers, _ = self.unpack(parameters)
        query = self.resolved_query(parameters, base_query, learn_query)
        psi = jnp.ones((phases.shape[0], self.dim), dtype=jnp.complex128) / np.sqrt(
            self.dim
        )
        x_upload = jnp.exp(1j * phases / self.layers)
        f_upload = jnp.exp(-1j * (query / self.layers) * self.h_compact_flat)
        for w_angles, v_angles in layers:
            psi = psi @ self.trainable_block(w_angles).T
            psi = psi * x_upload
            psi = psi @ self.trainable_block(v_angles).T
            psi = psi * f_upload[None, :]
        return jnp.einsum("ja,bad->bjd", self.fourier, psi.reshape(-1, self.q, self.q))

    def probabilities(self, parameters, phases, base_query=0.0, learn_query=0.0):
        """Class probabilities: readout populations summed over the range register."""
        amplitudes = self.state(parameters, phases, base_query, learn_query)
        p = jnp.sum(jnp.abs(amplitudes) ** 2, axis=2)
        p = jnp.clip(p, 1e-12, None)
        return p / p.sum(axis=1, keepdims=True)

    def cross_entropy(self, parameters, phases, labels, base_query, learn_query):
        """Mean negative log-likelihood of the labels."""
        p = self.probabilities(parameters, phases, base_query, learn_query)
        return -jnp.mean(jnp.log(p[jnp.arange(labels.shape[0]), labels]))

    def calibration_parameters(self, sensor_z_phases):
        """Known inverse channel phase in the first ``R_Z`` layer of ``A``.

        Used only for pre-training audits and the mechanism figure.
        """
        parameters = np.zeros(self.n_parameters)
        for qubit in range(self.n_qubits):
            parameters[2 * qubit + 1] = -sensor_z_phases[qubit]
        return parameters


def arm_setting(arm, f_geometry):
    """``(base_query, learn_query)`` of a query arm.

    ``aligned``: ``f*_geom``; ``mismatched``: ``-f*_geom``; ``none``: 0;
    ``learned``: a trainable global query starting from 0.
    """
    if arm == "aligned":
        return float(f_geometry), 0.0
    if arm == "mismatched":
        return float(-f_geometry), 0.0
    if arm == "none":
        return 0.0, 0.0
    if arm == "learned":
        return 0.0, 1.0
    raise ValueError(f"unknown arm={arm!r}")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def make_adam_runner(loss, *, steps, learning_rate):
    """Jitted Adam over a batch of restarts of
    ``loss(parameters, phases, labels, base_query, learn_query)``.

    Gradients are clipped to norm 10 per restart, the learning rate follows a
    cosine schedule from ``learning_rate`` to ``0.1 * learning_rate``, and each
    restart keeps its lowest-loss parameters. The returned function maps
    ``(initial_parameters, phases, labels, base_query, learn_query)`` to
    ``(best_parameters, best_losses)``.
    """
    loss_and_gradient = jax.vmap(
        jax.value_and_grad(loss), in_axes=(0, None, None, None, None)
    )
    loss_batch = jax.vmap(loss, in_axes=(0, None, None, None, None))

    @jax.jit
    def run(initial_parameters, phases, labels, base_query, learn_query):
        parameters = jnp.asarray(initial_parameters)
        first = jnp.zeros_like(parameters)
        second = jnp.zeros_like(parameters)
        best_parameters = parameters
        best_losses = loss_batch(parameters, phases, labels, base_query, learn_query)

        def body(step, state):
            parameters, first, second, best_parameters, best_losses = state
            _, gradient = loss_and_gradient(
                parameters, phases, labels, base_query, learn_query
            )
            norm = jnp.linalg.norm(gradient, axis=1)
            gradient = gradient * jnp.minimum(1.0, 10.0 / (norm + 1e-12))[:, None]
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            t = step + 1
            first_hat = first / (1.0 - 0.9**t)
            second_hat = second / (1.0 - 0.999**t)
            fraction = step / jnp.maximum(1, steps - 1)
            rate = learning_rate * (
                0.1 + 0.9 * 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
            )
            parameters = parameters - rate * first_hat / (jnp.sqrt(second_hat) + 1e-8)
            new_losses = loss_batch(parameters, phases, labels, base_query, learn_query)
            improved = new_losses < best_losses
            best_losses = jnp.where(improved, new_losses, best_losses)
            best_parameters = jnp.where(improved[:, None], parameters, best_parameters)
            return parameters, first, second, best_parameters, best_losses

        _, _, _, best_parameters, best_losses = jax.lax.fori_loop(
            0,
            steps,
            body,
            (parameters, first, second, best_parameters, best_losses),
        )
        return best_parameters, best_losses

    return run


def paired_initializations(seed, *, n_restarts, n_parameters, learned_query):
    """Restart initializations shared by all arms.

    Variational parameters are ``0.15 N(0, 1)`` with restart 0 at zero. The
    raw global query is ``U(-1, 1)`` (0 for restart 0) for the trainable-query
    arm and 0 otherwise.
    """
    rng = np.random.default_rng(seed)
    parameters = 0.15 * rng.normal(size=(n_restarts, n_parameters))
    parameters[0] = 0.0
    parameters[:, -1] = (
        rng.uniform(-1.0, 1.0, size=n_restarts) if learned_query else 0.0
    )
    if learned_query:
        parameters[0, -1] = 0.0
    return parameters


def make_circuit_fitter(circuit, *, n_restarts, steps, learning_rate):
    """Return ``fit(train_phases, train_labels, *, seed, base_query=0.0,
    learn_query=0.0, test_phases=None, test_labels=None)``.

    ``fit`` trains all paired restarts with Adam, selects the restart with
    the lowest training loss and, if held-out data are given, evaluates it.
    The result dict holds ``train_loss``, ``best_restart`` and
    ``fitted_query`` and optionally ``test_accuracy``,
    ``test_balanced_accuracy`` and ``test_nll``.
    """
    runner = make_adam_runner(
        circuit.cross_entropy, steps=steps, learning_rate=learning_rate
    )

    def fit(
        train_phases,
        train_labels,
        *,
        seed,
        base_query=0.0,
        learn_query=0.0,
        test_phases=None,
        test_labels=None,
    ):
        initial = paired_initializations(
            seed,
            n_restarts=n_restarts,
            n_parameters=circuit.n_parameters,
            learned_query=bool(learn_query),
        )
        candidates, losses = runner(
            jnp.asarray(initial),
            train_phases,
            train_labels,
            float(base_query),
            float(learn_query),
        )
        losses = np.asarray(losses)
        best_index = int(np.argmin(losses))
        best = np.asarray(candidates[best_index])
        fitted_query = float(
            circuit.resolved_query(best, float(base_query), float(learn_query))
        )
        result = {
            "train_loss": float(losses[best_index]),
            "best_restart": best_index,
            "fitted_query": fitted_query,
        }
        if test_phases is not None:
            p = np.asarray(
                circuit.probabilities(best, test_phases, base_query, learn_query)
            )
            labels = np.asarray(test_labels, dtype=int)
            prediction = np.argmax(p, axis=1)
            result.update(
                test_accuracy=float(np.mean(prediction == labels)),
                test_balanced_accuracy=balanced_accuracy(labels, prediction, circuit.q),
                test_nll=float(-np.mean(np.log(p[np.arange(len(labels)), labels]))),
            )
        return result

    return fit


def calibrated_accuracy(circuit, parameters, phases, labels, base_query, learn_query):
    """Balanced accuracy of the circuit at fixed, untrained ``parameters``."""
    return balanced_accuracy(
        labels,
        np.argmax(
            np.asarray(circuit.probabilities(parameters, phases, base_query, learn_query)),
            axis=1,
        ),
        circuit.q,
    )


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


def run_query_scan(
    log,
    *,
    fit,
    scenes,
    geometry,
    ratios,
    n_draws,
    n_train,
    n_test,
    base_seed,
):
    """Fit every fixed query ``f = ratio * f*_geom`` on identical samples."""
    f_geometry = geometry["f_geometry"]
    for draw in range(n_draws):
        train_signal, train_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 100, draw)),
            n_train,
            geometry=geometry,
        )
        test_signal, test_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 101, draw)),
            n_test,
            geometry=geometry,
        )
        train_phases, test_phases = encode(train_signal), encode(test_signal)
        paired_seed = seed_from(base_seed, 102, draw)
        for index, ratio in enumerate(ratios):
            task_id = f"scan-d{draw:03d}-r{index:03d}"
            if task_id in log:
                continue
            query = float(ratio * f_geometry)
            result = fit(
                train_phases,
                train_labels,
                seed=paired_seed,
                base_query=query,
                test_phases=test_phases,
                test_labels=test_labels,
            )
            log.record(
                task_id,
                {
                    "experiment": "scan",
                    "draw": draw,
                    "geometry": geometry["name"],
                    "f_geometry": f_geometry,
                    "arm": "scan",
                    "query_value": query,
                    "query_ratio": float(ratio),
                    "n_train": n_train,
                    **result,
                },
            )
            print(
                f"scan draw={draw:02d} f/f*={ratio:+.3f} "
                f"acc={result['test_balanced_accuracy']:.3f}",
                flush=True,
            )


def run_geometry_profiles(
    log,
    *,
    fit,
    scenes,
    geometries,
    ratios,
    n_draws,
    n_train,
    n_validation,
    base_seed,
):
    """Fixed-query validation profiles for data from every acquisition.

    The held-out set is a validation set, so its metrics are stored in the
    validation columns.
    """
    for geometry_index, geometry in enumerate(geometries):
        query_grid = ratios * geometry["f_geometry"]
        for draw in range(n_draws):
            train_signal, train_labels = scenes(
                jax.random.PRNGKey(seed_from(base_seed, 200, geometry_index, draw)),
                n_train,
                geometry=geometry,
            )
            validation_signal, validation_labels = scenes(
                jax.random.PRNGKey(seed_from(base_seed, 201, geometry_index, draw)),
                n_validation,
                geometry=geometry,
            )
            train_phases = encode(train_signal)
            validation_phases = encode(validation_signal)
            paired_seed = seed_from(base_seed, 202, geometry_index, draw)
            for query_index, query in enumerate(query_grid):
                task_id = f"profile-g{geometry_index:02d}-d{draw:03d}-q{query_index:03d}"
                if task_id in log:
                    continue
                result = fit(
                    train_phases,
                    train_labels,
                    seed=paired_seed,
                    base_query=float(query),
                    test_phases=validation_phases,
                    test_labels=validation_labels,
                )
                validation_accuracy = result.pop("test_balanced_accuracy")
                result.pop("test_accuracy")
                validation_nll = result.pop("test_nll")
                log.record(
                    task_id,
                    {
                        "experiment": "geometry_profile",
                        "draw": draw,
                        "geometry": geometry["name"],
                        "f_geometry": geometry["f_geometry"],
                        "arm": "profile",
                        "query_value": float(query),
                        "query_ratio": float(query / geometry["f_geometry"]),
                        "n_train": n_train,
                        "validation_balanced_accuracy": validation_accuracy,
                        "validation_nll": validation_nll,
                        **result,
                    },
                )
                print(
                    f"profile {geometry['name']:19s} draw={draw:02d} "
                    f"f={query:.3f} val_nll={validation_nll:.3f}",
                    flush=True,
                )


def run_sample_efficiency(
    log,
    *,
    fit,
    scenes,
    geometry,
    arms,
    n_train_values,
    n_draws,
    n_test,
    base_seed,
):
    """Compare the query arms on nested prefixes of one training pool."""
    f_geometry = geometry["f_geometry"]
    n_max = max(n_train_values)
    for draw in range(n_draws):
        pool_signal, pool_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 300, draw)),
            n_max,
            geometry=geometry,
        )
        test_signal, test_labels = scenes(
            jax.random.PRNGKey(seed_from(base_seed, 301, draw)),
            n_test,
            geometry=geometry,
        )
        pool_phases, test_phases = encode(pool_signal), encode(test_signal)
        for n_train in n_train_values:
            paired_seed = seed_from(base_seed, 302, draw, n_train)
            for arm in arms:
                task_id = f"efficiency-d{draw:03d}-n{n_train:05d}-{arm}"
                if task_id in log:
                    continue
                base_query, learn_query = arm_setting(arm, f_geometry)
                result = fit(
                    pool_phases[:n_train],
                    pool_labels[:n_train],
                    seed=paired_seed,
                    base_query=base_query,
                    learn_query=learn_query,
                    test_phases=test_phases,
                    test_labels=test_labels,
                )
                log.record(
                    task_id,
                    {
                        "experiment": "efficiency",
                        "draw": draw,
                        "geometry": geometry["name"],
                        "f_geometry": f_geometry,
                        "arm": arm,
                        "query_value": base_query if not learn_query else np.nan,
                        "query_ratio": (
                            base_query / f_geometry if not learn_query else np.nan
                        ),
                        "n_train": n_train,
                        **result,
                    },
                )
            print(f"efficiency draw={draw:02d} N={n_train} complete", flush=True)


# --------------------------------------------------------------------------
# validity checks
# --------------------------------------------------------------------------


def geometry_checks(geometry, profile_geometries):
    """Mixed derivative and bilinear quality for every acquisition.

    Because ``R^2`` alone can hide a sizable phase error when ``f`` is large,
    the largest residual phase ``f*_geom |H_exact - a~ d~|`` is bounded too.
    """
    h_compact = compact_interaction(geometry["h_exact"].shape[0])
    checks = {}
    checks["mixed derivative relative error"] = mixed_derivative_relative_error(
        geometry["kx_center"], geometry["kr_center"], geometry["kappa_xr"]
    )
    assert checks["mixed derivative relative error"] < 2e-5

    for g in profile_geometries:
        h = g["h_exact"]
        assert np.max(np.abs(h.mean(axis=1))) < 1e-8
        assert np.max(np.abs(h.mean(axis=0))) < 1e-8
        assert abs(g["bilinear_coefficient"] - 1.0) < 0.01
        assert g["bilinear_r2"] > 0.995
    checks["worst bilinear R2"] = float(min(g["bilinear_r2"] for g in profile_geometries))
    checks["worst aligned residual rad"] = float(
        max(
            np.max(np.abs(g["f_geometry"] * (g["h_exact"] - h_compact)))
            for g in profile_geometries
        )
    )
    assert checks["worst aligned residual rad"] < 0.35
    return checks


def rzz_checks(circuit):
    """The compact query equals ``n^2`` cross-register ``R_ZZ`` rotations."""
    n = circuit.n_qubits
    z = register_z_signs(n)
    z_cross = np.einsum("ai,dj->adij", z, z).reshape(circuit.dim, n * n)
    weights = np.outer(2.0 ** np.arange(n - 1, -1, -1), 2.0 ** np.arange(n - 1, -1, -1))
    h_compact = compact_interaction(circuit.q)

    theta = 0.731
    compiled = np.zeros(circuit.dim)
    for index, weight in enumerate(weights.reshape(-1)):
        compiled += -0.5 * (theta * weight / 2.0) * z_cross[:, index]
    checks = {
        "RZZ decomposition": float(
            np.max(
                np.abs(np.exp(1j * compiled) - np.exp(-1j * theta * h_compact).reshape(-1))
            )
        ),
        "RZZ gates per upload": n * n,
    }
    assert checks["RZZ decomposition"] < 1e-12
    return checks


def diagonal_operator_schmidt_spectrum(query, h_compact):
    """Operator-Schmidt spectrum across A|D of ``diag(exp(-i query a~ d~))``."""
    matrix = np.exp(-1j * query * h_compact)
    return np.linalg.svd(matrix, compute_uv=False)


def unitary_operator_schmidt_spectrum(unitary, q):
    """Operator-Schmidt spectrum across A|D of a ``Q^2 x Q^2`` unitary."""
    tensor = np.asarray(unitary).reshape(q, q, q, q)
    reshaped = tensor.transpose(0, 2, 1, 3).reshape(q * q, q * q)
    return np.linalg.svd(reshaped, compute_uv=False)


def schmidt_checks(circuit, profile_geometries, *, base_seed):
    """Trainable blocks are products; every geometry query is not.

    A random trainable block must have operator-Schmidt rank one, every
    per-upload query at ``f*_geom / L`` must have rank above one with a
    non-negligible product residual, and the aligned and wrong-sign queries
    must have identical spectra.
    """
    q = circuit.q
    h_compact = compact_interaction(q)
    checks = {}
    rng = np.random.default_rng(seed_from(base_seed, 20))
    probe_angles = rng.normal(size=(circuit.num_qubits, 2))
    trainable_s = unitary_operator_schmidt_spectrum(
        circuit.trainable_block(probe_angles), q
    )
    checks["trainable block Schmidt residual"] = float(
        np.sqrt(np.sum(trainable_s[1:] ** 2) / np.sum(trainable_s**2))
    )
    assert checks["trainable block Schmidt residual"] < 1e-10

    minimum_query_rank = circuit.dim
    minimum_query_residual = 1.0
    maximum_sign_spectrum_error = 0.0
    for geometry in profile_geometries:
        per_upload = geometry["f_geometry"] / circuit.layers
        positive = diagonal_operator_schmidt_spectrum(per_upload, h_compact)
        negative = diagonal_operator_schmidt_spectrum(-per_upload, h_compact)
        rank = int(np.sum(positive > 1e-10 * positive[0]))
        residual = float(np.sqrt(np.sum(positive[1:] ** 2) / np.sum(positive**2)))
        minimum_query_rank = min(minimum_query_rank, rank)
        minimum_query_residual = min(minimum_query_residual, residual)
        maximum_sign_spectrum_error = max(
            maximum_sign_spectrum_error, float(np.max(np.abs(positive - negative)))
        )
    checks["minimum query operator-Schmidt rank"] = minimum_query_rank
    checks["minimum query product residual"] = minimum_query_residual
    checks["sign-control Schmidt-spectrum error"] = maximum_sign_spectrum_error
    assert checks["minimum query operator-Schmidt rank"] > 1
    assert checks["minimum query product residual"] > 0.05
    assert checks["sign-control Schmidt-spectrum error"] < 1e-12
    return checks


def zero_parameter_check(circuit, scenes, *, f_geometry, base_seed):
    """At zero parameters the circuit is the product of its two encodings."""
    q = circuit.q
    probe_signal, _ = scenes(jax.random.PRNGKey(seed_from(base_seed, 21)), 8 * q)
    probe_phases = encode(probe_signal)
    analytic = np.abs(
        np.einsum(
            "ja,bad->bjd",
            np.asarray(circuit.fourier),
            (
                np.exp(1j * np.asarray(probe_phases))
                * np.exp(-1j * f_geometry * circuit.h_compact_flat)[None, :]
            ).reshape(-1, q, q)
            / np.sqrt(circuit.dim),
        )
    ) ** 2
    analytic = analytic.sum(axis=2)
    checks = {
        "zero-parameter circuit is encoding product": float(
            np.max(
                np.abs(
                    np.asarray(
                        circuit.probabilities(
                            np.zeros(circuit.n_parameters), probe_phases, f_geometry, 0.0
                        )
                    )
                    - analytic / analytic.sum(axis=1, keepdims=True)
                )
            )
        )
    }
    assert checks["zero-parameter circuit is encoding product"] < 1e-10
    return checks


def calibration_checks(circuit, scenes, *, calibration, f_geometry, base_seed):
    """The calibrated, untrained circuit shows the intended contrast.

    Clean scenes are classified almost perfectly at the aligned query; on
    noisy scenes the aligned query beats the zero and wrong-sign queries.
    """
    q = circuit.q
    clean_signal, clean_labels = scenes(
        jax.random.PRNGKey(seed_from(base_seed, 22)), 8 * q, snr_db=60.0, n_clutter=0
    )
    checks = {
        "clean calibrated aligned accuracy": calibrated_accuracy(
            circuit, calibration, encode(clean_signal), clean_labels, f_geometry, 0.0
        )
    }
    assert checks["clean calibrated aligned accuracy"] > 0.95

    audit_signal, audit_labels = scenes(jax.random.PRNGKey(seed_from(base_seed, 23)), 800)
    audit_phases = encode(audit_signal)
    untrained = {}
    for arm in ("aligned", "mismatched", "none"):
        query, learn = arm_setting(arm, f_geometry)
        untrained[arm] = calibrated_accuracy(
            circuit, calibration, audit_phases, audit_labels, query, learn
        )
    checks.update({f"calibrated accuracy {arm}": value for arm, value in untrained.items()})
    assert untrained["aligned"] > untrained["none"] + 0.20
    assert untrained["aligned"] > untrained["mismatched"] + 0.15
    return checks


def balance_checks(scenes, *, declared_sizes, n_classes, base_seed):
    """Every declared training, validation and test size is exactly balanced."""
    for size in set(declared_sizes):
        _, labels = scenes(jax.random.PRNGKey(seed_from(base_seed, 24, size)), size)
        counts = np.bincount(np.asarray(labels), minlength=n_classes)
        assert counts.max() == counts.min()
    return {
        "balanced declared sizes": True,
        "query fixed within each dataset": True,
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap_matrix(values, *, n_samples, base_seed, seed):
    """Draw bootstrap of the column means of ``values`` (draws x points).

    Returns the mean and the 2.5% and 97.5% quantiles of the bootstrap means
    (``lower``, ``upper``).
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_from(base_seed, 500, seed))
    indices = rng.integers(0, values.shape[0], (n_samples, values.shape[0]))
    draws = values[indices].mean(axis=1)
    return {
        "mean": values.mean(axis=0),
        "lower": np.quantile(draws, 0.025, axis=0),
        "upper": np.quantile(draws, 0.975, axis=0),
    }


def bootstrap_peak_interval(x, values, *, n_samples, base_seed, seed):
    """Bootstrap the argmax of a paired set of response curves.

    Returns the 95% interval of the peak position and the fraction of
    bootstrap peaks at ``x = 1`` (the geometry query).
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_from(base_seed, 510, seed))
    indices = rng.integers(0, values.shape[0], (n_samples, values.shape[0]))
    means = values[indices].mean(axis=1)
    peaks = np.asarray(x)[np.argmax(means, axis=1)]
    return {
        "lower": float(np.quantile(peaks, 0.025)),
        "upper": float(np.quantile(peaks, 0.975)),
        "probability_at_geometry": float(np.mean(np.isclose(peaks, 1.0))),
    }


def profile_minimum(query, loss):
    """Grid minimum, refined by a local quadratic when interior and convex.

    The vertex is clipped to the two neighbouring grid points.
    """
    query = np.asarray(query, dtype=float)
    loss = np.asarray(loss, dtype=float)
    index = int(np.argmin(loss))
    if index == 0 or index == len(query) - 1:
        return float(query[index])
    local_x = query[index - 1 : index + 2]
    local_y = loss[index - 1 : index + 2]
    quadratic, linear, _ = np.polyfit(local_x, local_y, 2)
    if quadratic <= 0:
        return float(query[index])
    vertex = -linear / (2 * quadratic)
    return float(np.clip(vertex, local_x[0], local_x[-1]))


def bootstrap_profile_minimum(query, losses, *, n_samples, base_seed, seed):
    """Profile estimate and draw-bootstrap interval for ``argmin E[L(f)]``.

    The full mean profile is re-bootstrapped before each minimum is taken.
    """
    query = np.asarray(query, dtype=float)
    losses = np.asarray(losses, dtype=float)
    point = profile_minimum(query, losses.mean(axis=0))
    rng = np.random.default_rng(seed_from(base_seed, 520, seed))
    indices = rng.integers(0, losses.shape[0], (n_samples, losses.shape[0]))
    mean_profiles = losses[indices].mean(axis=1)
    minima = np.asarray([profile_minimum(query, profile) for profile in mean_profiles])
    return {
        "f_hat": point,
        "lower": float(np.quantile(minima, 0.025)),
        "upper": float(np.quantile(minima, 0.975)),
    }


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------


def phase_rgb(patch, cmap=None):
    """RGB image of a complex patch: hue from the phase, brightness from |patch|."""
    if cmap is None:
        cmap = matplotlib.colormaps["twilight_shifted"]
    amplitude = np.abs(patch)
    amplitude = amplitude / max(amplitude.max(), 1e-30)
    rgb = cmap((np.angle(patch) + np.pi) / (2 * np.pi))[..., :3]
    return 1.0 - amplitude[..., None] * (1.0 - rgb)


def style_tile(axis, first, xlabel, ylabel, q):
    """Ticks and limits of one ``q x q`` image tile; labels only on the first."""
    axis.set_xticks([0, 3, 6])
    axis.set_yticks([0, 3, 6])
    axis.set_xlim(-0.5, q - 0.5)
    axis.set_ylim(-0.5, q - 0.5)
    if first:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    else:
        axis.set_yticklabels([])
