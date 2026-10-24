"""Query-aligned teacher-student circuits: model, teachers and labelled data.

The circuit is a data re-uploading model on ``n_qubits`` qubits (grouped into
dimers of two) with ``n_layers`` layers. Each layer applies input phases
``E_x``, a trainable block ``W``, the nonlocal query layer ``Q(s f)`` built from
commuting ``Y⊗Z`` and ``Z⊗Y`` rotations, and a trainable block ``V``; the score
is the parity expectation ``<Z...Z>``. The same circuit serves as teacher
(at the query ``f*``) and as student (at a fixed query ``f``).

This module holds everything that is shared by the teacher-student notebooks:
the circuit configuration and forward pass, the parameter prior, the Fourier
nonseparability measures, teacher selection and the labelled data sets.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd

_I = np.eye(2, dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)


def rng_from(*parts):
    """Independent random generator seeded by a tuple of integers."""
    return np.random.default_rng(
        np.random.SeedSequence([int(part) for part in parts])
    )


@dataclass(frozen=True)
class Cfg:
    """Circuit size and type of the trainable single-qubit blocks.

    ``block="ry"`` keeps only the R_Y angle of every single-qubit factor;
    ``block="yzy"`` uses full R_Z R_Y R_Z rotations.
    """

    n_qubits: int = 2
    n_layers: int = 2
    block: str = "ry"

    def __post_init__(self):
        if self.n_qubits % 2:
            raise ValueError("the YZ/ZY dimer construction needs an even n_qubits")
        if self.block not in {"ry", "yzy"}:
            raise ValueError("block must be 'ry' or 'yzy'")

    @property
    def n_dimers(self):
        """Number of qubit pairs carrying a query layer."""
        return self.n_qubits // 2

    @property
    def dimers(self):
        """Qubit index pairs (2d, 2d + 1) of every dimer."""
        return [(2 * d, 2 * d + 1) for d in range(self.n_dimers)]

    @property
    def scales(self):
        """Query scales s_l = 1, ..., n_layers."""
        return np.arange(1, self.n_layers + 1, dtype=float)

    @property
    def n_params(self):
        """Number of free (trainable) parameters."""
        per_single_qubit = 1 if self.block == "ry" else 3
        return self.n_layers * 2 * self.n_qubits * per_single_qubit

    def expand(self, free_parameters):
        """Map free parameters to the full (layer, W/V, qubit, ZYZ angle) tensor."""
        free_parameters = jnp.asarray(free_parameters).ravel()
        shape = (self.n_layers, 2, self.n_qubits)
        if self.block == "yzy":
            return free_parameters.reshape(*shape, 3)
        return jnp.zeros((*shape, 3)).at[:, :, :, 1].set(
            free_parameters.reshape(*shape)
        )


def make_f_star(f_star_pair, n_qubits):
    """Repeat the two-qubit query ``f_star_pair`` on every dimer."""
    return np.tile(f_star_pair, n_qubits // 2).astype(float)


# ---------------------------------------------------------------------------
# Matrices and gates
# ---------------------------------------------------------------------------


def _kron_numpy(matrices):
    """Kronecker product of a list of matrices."""
    out = np.array([[1.0]], dtype=complex)
    for matrix in matrices:
        out = np.kron(out, matrix)
    return out


def query_generators(n_qubits):
    """YZ and ZY query generators of every dimer as full matrices."""
    generators = []
    for i, j in [(2 * d, 2 * d + 1) for d in range(n_qubits // 2)]:
        for left, right in ((_Y, _Z), (_Z, _Y)):
            matrices = [_I] * n_qubits
            matrices[i], matrices[j] = left, right
            generators.append(_kron_numpy(matrices))
    return generators


def maximum_generator_commutator(n_qubits):
    """Largest entry of any pairwise commutator of the query generators."""
    generators = query_generators(n_qubits)
    return max(
        float(np.max(np.abs(a @ b - b @ a)))
        for i, a in enumerate(generators)
        for b in generators[i + 1 :]
    )


def basis_signs(n_qubits):
    """Z eigenvalues (+1 or -1) of every qubit for each computational basis state."""
    indices = np.arange(2**n_qubits)
    bits = (
        indices[:, None]
        >> np.arange(n_qubits - 1, -1, -1)[None, :]
    ) & 1
    return 1.0 - 2.0 * bits


def _su2_jax(parameters):
    """Single-qubit rotation RZ(c) RY(b) RZ(a)."""
    a, b, c = parameters
    rz = lambda angle: jnp.diag(
        jnp.array(
            [jnp.exp(-0.5j * angle), jnp.exp(0.5j * angle)],
            dtype=jnp.complex128,
        )
    )
    cosine = jnp.cos(b / 2).astype(jnp.complex128)
    sine = jnp.sin(b / 2).astype(jnp.complex128)
    ry = jnp.array([[cosine, -sine], [sine, cosine]], dtype=jnp.complex128)
    return rz(c) @ ry @ rz(a)


def _kron_jax(matrices):
    """Kronecker product of a list of matrices (JAX version)."""
    out = jnp.eye(1, dtype=jnp.complex128)
    for matrix in matrices:
        out = jnp.kron(out, matrix)
    return out


def query_layer(f, scale, cfg):
    """Query layer Q at angles scale * f: a YZ and then a ZY rotation on each dimer."""
    yz = jnp.asarray(np.kron(_Y, _Z))
    zy = jnp.asarray(np.kron(_Z, _Y))
    identity = jnp.eye(4, dtype=jnp.complex128)

    def rotation(angle, generator):
        return (
            jnp.cos(angle / 2).astype(jnp.complex128) * identity
            - 1j * jnp.sin(angle / 2).astype(jnp.complex128) * generator
        )

    dimer_gates = []
    for dimer in range(cfg.n_dimers):
        f0, f1 = f[2 * dimer], f[2 * dimer + 1]
        dimer_gates.append(
            rotation(scale * f1, zy) @ rotation(scale * f0, yz)
        )
    return _kron_jax(dimer_gates)


# ---------------------------------------------------------------------------
# Forward pass and prior
# ---------------------------------------------------------------------------


def forward(full_parameters, x, f, cfg):
    """Parity score <Z...Z> of the circuit for a batch of inputs.

    Parameters
    ----------
    full_parameters : array of shape (n_layers, 2, n_qubits, 3)
        Full parameter tensor, e.g. ``cfg.expand(free_parameters)``.
    x : array of shape (n_points, n_qubits)
        Input angles.
    f : array of shape (n_qubits,)
        Fixed query value.

    Returns the score of every input as a JAX array of shape (n_points,).
    """
    dimension = 2**cfg.n_qubits
    signs = basis_signs(cfg.n_qubits)
    parity = jnp.asarray(np.prod(signs, axis=1))
    initial = jnp.full(
        (dimension,), 1.0 / np.sqrt(dimension), dtype=jnp.complex128
    )

    theta = jnp.asarray(full_parameters).reshape(
        cfg.n_layers, 2, cfg.n_qubits, 3
    )
    x = jnp.atleast_2d(jnp.asarray(x))
    f = jnp.asarray(f)
    data_phase = jnp.exp(-0.5j * (x @ jnp.asarray(signs).T))

    state = jnp.broadcast_to(initial, (x.shape[0], dimension))
    for layer in range(cfg.n_layers):
        state = state * data_phase
        w_layer = _kron_jax(
            [_su2_jax(theta[layer, 0, q]) for q in range(cfg.n_qubits)]
        )
        state = state @ w_layer.T
        state = state @ query_layer(f, cfg.scales[layer], cfg).T
        v_layer = _kron_jax(
            [_su2_jax(theta[layer, 1, q]) for q in range(cfg.n_qubits)]
        )
        state = state @ v_layer.T

    return jnp.real(jnp.sum(jnp.abs(state) ** 2 * parity, axis=1))


def sample_free_prior(rng, cfg):
    """Draw free parameters from the prior used for teachers and initialization.

    For ``block="ry"`` the angles are uniform on [-pi, pi); for ``"yzy"`` the
    rotations are Haar distributed.
    """
    if cfg.block == "ry":
        return rng.uniform(-np.pi, np.pi, cfg.n_params)

    shape = (cfg.n_layers, 2, cfg.n_qubits)
    a = rng.uniform(0.0, 2 * np.pi, shape)
    c = rng.uniform(0.0, 2 * np.pi, shape)
    b = np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0, shape))
    return np.stack([a, b, c], axis=-1).ravel()


# ---------------------------------------------------------------------------
# Nonseparability
# ---------------------------------------------------------------------------


def fourier_matrix(full_parameters, f, cfg, dimer=0):
    """Fourier-coefficient matrix of the score restricted to one dimer.

    The score is sampled on a regular (2L+1) x (2L+1) grid and transformed by
    a 2D FFT, which gives the coefficients of the modes -L, ..., L per input.
    """
    modes_per_axis = 2 * cfg.n_layers + 1
    sub_cfg = Cfg(2, cfg.n_layers, cfg.block)
    i, j = cfg.dimers[dimer]
    theta = np.asarray(full_parameters).reshape(
        cfg.n_layers, 2, cfg.n_qubits, 3
    )[:, :, [i, j], :]
    f_sub = np.asarray(f)[[i, j]]
    grid = 2 * np.pi * np.arange(modes_per_axis) / modes_per_axis
    x0, x1 = np.meshgrid(grid, grid, indexing="ij")
    points = np.stack([x0.ravel(), x1.ravel()], axis=1)
    values = np.asarray(forward(theta, points, f_sub, sub_cfg)).reshape(
        modes_per_axis, modes_per_axis
    )
    return np.fft.fftshift(
        np.fft.fft2(values) / modes_per_axis**2
    )


def coefficient_nonseparability(full_parameters, f, cfg):
    """Nonseparability 1 - s_1^2 / sum_k s_k^2 of the coefficient matrix.

    Returns a dict with one entry ``"dimer<d>"`` per dimer and their minimum
    under ``"min"``.
    """
    result = {}
    for dimer in range(cfg.n_dimers):
        singular_values = np.linalg.svd(
            fourier_matrix(full_parameters, f, cfg, dimer),
            compute_uv=False,
        )
        result[f"dimer{dimer}"] = float(
            1.0
            - singular_values[0] ** 2
            / np.sum(singular_values**2)
        )
    result["min"] = min(
        result[f"dimer{d}"] for d in range(cfg.n_dimers)
    )
    return result


def operator_nonseparable_weight(gate):
    """Operator-Schmidt nonseparability of a two-qubit gate."""
    reshuffled = (
        np.asarray(gate)
        .reshape(2, 2, 2, 2)
        .transpose(0, 2, 1, 3)
        .reshape(4, 4)
    )
    singular_values = np.linalg.svd(reshuffled, compute_uv=False)
    weights = singular_values**2
    return float(1.0 - weights[0] / np.sum(weights))


def check_circuit(cfg, f_star, *, base_seed):
    """Assert the basic circuit properties; raise AssertionError otherwise.

    Checks that the query generators commute, that scores lie in [-1, 1],
    that the score at f = 0 is separable and that (for two qubits) the query
    gate at f* is nonseparable.
    """
    assert maximum_generator_commutator(cfg.n_qubits) < 1e-12
    rng = rng_from(base_seed, 1)
    free_parameters = sample_free_prior(rng, cfg)
    theta = np.asarray(cfg.expand(free_parameters))
    x = rng.uniform(0, 2 * np.pi, (64, cfg.n_qubits))
    values = np.asarray(forward(theta, x, f_star, cfg))
    assert np.max(np.abs(values)) <= 1.0 + 1e-10
    assert coefficient_nonseparability(
        theta, np.zeros_like(f_star), cfg
    )["min"] < 1e-10
    generic_query = np.asarray(query_layer(f_star, 1.0, cfg))
    if cfg.n_qubits == 2:
        assert operator_nonseparable_weight(generic_query) > 1e-3


# ---------------------------------------------------------------------------
# Teachers and labelled data
# ---------------------------------------------------------------------------


def select_teacher(
    cfg,
    f_star,
    teacher_seed,
    *,
    n_probe,
    balance,
    min_nonsep,
    max_draws,
    base_seed,
):
    """Return the first prior draw that passes the balance and nonseparability criteria.

    A draw is accepted if the fraction of positive scores on an independent
    probe set lies strictly inside ``balance`` and its coefficient
    nonseparability exceeds ``min_nonsep``. The returned record holds the
    free and full parameters, the number of draws, the label balance, the
    nonseparability and the probe fraction with |v_T| < 0.05.
    """
    probe_rng = rng_from(base_seed, 100, teacher_seed)
    parameter_rng = rng_from(base_seed, 101, teacher_seed)
    x_probe = probe_rng.uniform(
        0, 2 * np.pi, (n_probe, cfg.n_qubits)
    )

    for draw in range(1, max_draws + 1):
        free_parameters = sample_free_prior(parameter_rng, cfg)
        full_parameters = np.asarray(cfg.expand(free_parameters))
        score = np.asarray(forward(full_parameters, x_probe, f_star, cfg))
        positive_fraction = float(np.mean(score > 0))
        if not (balance[0] < positive_fraction < balance[1]):
            continue

        nonseparability = coefficient_nonseparability(
            full_parameters, f_star, cfg
        )
        if nonseparability["min"] <= min_nonsep:
            continue

        return {
            "teacher": int(teacher_seed),
            "draws": int(draw),
            "free_parameters": np.asarray(free_parameters),
            "full_parameters": full_parameters,
            "balance": positive_fraction,
            "nonseparability": nonseparability,
            "boundary_mass_005": float(np.mean(np.abs(score) < 0.05)),
        }

    raise RuntimeError(
        f"teacher {teacher_seed} did not pass the predeclared criteria "
        f"in {max_draws} draws"
    )


def teacher_table(teachers):
    """One row per teacher: draws, label balance, nonseparability, boundary mass."""
    return pd.DataFrame(
        [
            {
                "teacher": t["teacher"],
                "draws": t["draws"],
                "balance": t["balance"],
                "nonseparability": t["nonseparability"]["min"],
                "boundary_mass_005": t["boundary_mass_005"],
            }
            for t in teachers
        ]
    )


def labelled_inputs(rng, n_points, teacher, f_star, cfg):
    """Draw inputs uniformly from [0, 2 pi)^n and label them 1[v_T(x) > 0].

    Returns ``(x, y)`` with float labels.
    """
    x = rng.uniform(0, 2 * np.pi, (n_points, cfg.n_qubits))
    score = np.asarray(forward(teacher["full_parameters"], x, f_star, cfg))
    return x, (score > 0).astype(float)


def teacher_test_set(teacher, f_star, cfg, *, n_test, base_seed):
    """Held-out test set of a teacher, shared by all folds, arms and queries."""
    rng = rng_from(base_seed, 400, teacher["teacher"])
    return labelled_inputs(rng, n_test, teacher, f_star, cfg)


def teacher_training_pool(teacher, fold, f_star, cfg, *, n_points, base_seed):
    """Training pool of one teacher and fold; training sets are its prefixes."""
    rng = rng_from(base_seed, 401, teacher["teacher"], fold)
    return labelled_inputs(rng, n_points, teacher, f_star, cfg)


def paired_seed(teacher_id, fold, n_train):
    """Seed of the initial-parameter ensemble, shared by all arms and queries."""
    return 1_000_000 * teacher_id + 10_000 * fold + int(n_train)
