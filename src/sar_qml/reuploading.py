"""Two-qubit data-reuploading circuit shared by the query-encoding and query-value sweeps.

The circuit acts on two qubits, qubit 0 being the left tensor factor. Each
layer applies the data encoding ``E_x``, a trainable block ``W``, a query
layer and a trainable block ``V``; the initial state is ``|++>`` and the
observable is ``Z_0 Z_1``. This module holds what both sweeps share:

* Pauli operators and the CNOT used by the robustness variant,
* single-qubit rotations, the ZYZ Euler parameterization of SU(2) and the
  trainable blocks built from it,
* the Haar and narrow-Gaussian priors over the Euler angles,
* Fourier-grid helpers, the operator-Schmidt test for product gates and the
  joint query spectrum used by both structural validations,
* saving and loading of the swept curves.

The query layers themselves differ between the two sweeps and live in
:mod:`sar_qml.reuploading_encoding` and :mod:`sar_qml.reuploading_query`.
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def kron(*operators):
    """Kronecker product of an arbitrary number of operators."""
    result = np.array([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


I4 = np.eye(4, dtype=complex)
X0 = kron(X, I2)
X1 = kron(I2, X)
Y0 = kron(Y, I2)
Y1 = kron(I2, Y)
Z0 = kron(Z, I2)
Z1 = kron(I2, Z)
ZZ = Z0 @ Z1

CNOT_01 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=complex,
)

PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
INITIAL_STATE = kron(PLUS, PLUS).reshape(-1)


# --- gates -----------------------------------------------------------------


def pauli_rotation(pauli, angle):
    r"""Return exp(-i angle pauli / 2) for a Hermitian involution ``pauli``."""
    dimension = pauli.shape[0]
    return (
        np.cos(angle / 2.0) * np.eye(dimension, dtype=complex)
        - 1j * np.sin(angle / 2.0) * pauli
    )


def single_qubit_su2(euler_angles):
    r"""Euler parameterization RZ(alpha) RY(beta) RZ(gamma) of SU(2)."""
    alpha, beta, gamma = euler_angles
    return (
        pauli_rotation(Z, alpha)
        @ pauli_rotation(Y, beta)
        @ pauli_rotation(Z, gamma)
    )


def trainable_block(block_parameters, block_type):
    r"""
    Return the trainable block U^(0) tensor U^(1).

    ``block_parameters`` has shape (2, 3), one row of Euler angles per
    qubit. With ``block_type="cnot"`` a fixed CNOT follows the local
    rotations in circuit order.
    """
    local = kron(
        single_qubit_su2(block_parameters[0]),
        single_qubit_su2(block_parameters[1]),
    )
    if block_type == "product":
        return local
    if block_type == "cnot":
        return CNOT_01 @ local
    raise ValueError("Unknown trainable block type.")


def trainable_blocks(parameters, block_type):
    """Return the list of trainable blocks W_1, V_1, W_2, V_2, ... in circuit order."""
    return [
        trainable_block(parameters[index], block_type)
        for index in range(len(parameters))
    ]


def data_encoding(x):
    r"""E_x(x) = RZ(x_0) tensor RZ(x_1), one component per qubit."""
    return pauli_rotation(Z0, x[0]) @ pauli_rotation(Z1, x[1])


# --- parameter priors ------------------------------------------------------


def sample_haar_euler_parameters(rng):
    r"""Sample ZYZ Euler angles from the Haar measure on SU(2)."""
    alpha = rng.uniform(0.0, 2.0 * np.pi)
    gamma = rng.uniform(0.0, 2.0 * np.pi)
    beta = np.arccos(rng.uniform(-1.0, 1.0))
    return np.array([alpha, beta, gamma])


def sample_narrow_euler_parameters(rng, sigma):
    r"""Sample ZYZ Euler angles from zero-mean Gaussians of width ``sigma``."""
    return rng.normal(0.0, sigma, size=3)


def sample_parameters(rng, distribution, number_of_blocks, narrow_sigma=None):
    """
    Sample the Euler angles of all trainable blocks.

    ``distribution`` is ``"haar"`` or ``"narrow"``; the latter needs
    ``narrow_sigma``. Angles are drawn block by block and qubit by qubit, so
    the random stream is consumed in that order. Returns an array of shape
    (number_of_blocks, 2, 3).
    """
    if distribution not in ("haar", "narrow"):
        raise ValueError("distribution must be either 'haar' or 'narrow'.")
    if distribution == "narrow" and narrow_sigma is None:
        raise ValueError("The narrow prior needs narrow_sigma.")

    parameters = np.empty((number_of_blocks, 2, 3), dtype=float)
    for block in range(number_of_blocks):
        for qubit in range(2):
            if distribution == "haar":
                parameters[block, qubit] = sample_haar_euler_parameters(rng)
            else:
                parameters[block, qubit] = sample_narrow_euler_parameters(
                    rng, narrow_sigma
                )
    return parameters


# --- Fourier grids and spectra ---------------------------------------------


def fourier_grid(size):
    """Equispaced sample points 2 pi k / size, k = 0, ..., size - 1."""
    return 2.0 * np.pi * np.arange(size) / size


def fourier_modes(size):
    """Integer frequencies of a length-``size`` FFT, in fftshift order."""
    return np.fft.fftshift(np.fft.fftfreq(size, d=1.0 / size)).astype(int)


def data_encoding_grid(scalar_grid):
    """Return E_x on the product grid, indexed [i_x0, i_x1]."""
    return np.asarray([
        [data_encoding(np.array([x0, x1])) for x1 in scalar_grid]
        for x0 in scalar_grid
    ])


def operator_nonseparable_weight(gate):
    r"""
    Operator-Schmidt nonseparable weight of a two-qubit gate.

    Zero exactly for a product unitary A tensor B, positive otherwise.
    """
    reshuffled = (
        gate.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    )
    singular_values = np.linalg.svd(reshuffled, compute_uv=False)
    weights = singular_values**2
    return float(1.0 - weights[0] / np.sum(weights))


def joint_query_spectrum(
    parameters,
    block_type,
    query_layer,
    data_point,
    number_of_layers,
    query_grid_size,
):
    r"""
    Return the joint query-Fourier coefficients C[omega_f0, omega_f1].

    The data point and the parameters are held fixed and the circuit is
    evaluated on a ``query_grid_size``^2 grid in (f_0, f_1).
    ``query_layer(query_vector, layer)`` returns the 4 x 4 query layer of
    the sweep in question. The coefficients are in fftshift order.
    """
    blocks = trainable_blocks(parameters, block_type)
    encoding = data_encoding(data_point)
    grid = fourier_grid(query_grid_size)

    query_gates = np.asarray([
        [
            query_layer(np.array([f0, f1]), layer)
            for f0 in grid
            for f1 in grid
        ]
        for layer in range(number_of_layers)
    ])

    states = np.broadcast_to(
        INITIAL_STATE,
        (query_grid_size**2, INITIAL_STATE.size),
    ).copy()

    for layer in range(number_of_layers):
        W_layer = blocks[2 * layer]
        V_layer = blocks[1 + 2 * layer]
        states = np.einsum("ij,fj->fi", encoding, states, optimize=True)
        states = np.einsum("ij,fj->fi", W_layer, states, optimize=True)
        states = np.einsum(
            "fij,fj->fi", query_gates[layer], states, optimize=True
        )
        states = np.einsum("ij,fj->fi", V_layer, states, optimize=True)

    expectations = np.einsum(
        "fi,ij,fj->f", states.conj(), ZZ, states, optimize=True
    )
    values = expectations.real.reshape(query_grid_size, query_grid_size)
    return np.fft.fftshift(np.fft.fft2(values) / query_grid_size**2)


# --- results files ---------------------------------------------------------

#: Ensemble curves stored per series, next to the fixed-parameter shape change.
STATISTICS_KEYS = (
    "nonseparable_weight",
    "prior_distance",
    "prior_distance_error",
    "mean_schmidt_rank",
)


def save_sweep_results(path, header, series, shape_change, statistics):
    """
    Store the plotted curves of one sweep in an ``.npz`` file.

    ``header`` maps the configuration entries (abscissa, query, noise
    ceiling, block type) to their values and is written first, followed by
    the series names and, per series, the shape change and the
    ``STATISTICS_KEYS`` entries of ``statistics[series]``.
    """
    arrays = dict(header)
    arrays["series"] = np.asarray(series)
    for name in series:
        arrays[f"shape_change__{name}"] = shape_change[name]
        for key in STATISTICS_KEYS:
            arrays[f"{key}__{name}"] = statistics[name][key]
    np.savez(path, **arrays)
    print(f"Saved results to {path}")


def load_sweep_results(path, expected, block_type, series, shape_change):
    """
    Read the curves written by :func:`save_sweep_results` and check them.

    ``expected`` maps header keys to ``(value, description)``; every stored
    array must equal its expected value exactly, the stored block type must
    match, and the stored shape change must agree with the recomputed
    ``shape_change`` to 1e-12. Returns ``(statistics, noise_ceiling)`` with
    ``statistics[series][key]`` for every key in ``STATISTICS_KEYS``.
    """
    with np.load(path) as data:
        for key, (value, description) in expected.items():
            if not np.array_equal(data[key], value):
                raise RuntimeError(
                    f"Cached {description} does not match the configuration."
                )
        if str(data["trainable_blocks"]) != block_type:
            raise RuntimeError("Cached block type does not match the configuration.")
        for name in series:
            if not np.allclose(
                data[f"shape_change__{name}"], shape_change[name], rtol=0, atol=1e-12
            ):
                raise RuntimeError(
                    f"Recomputed shape change for {name} differs from the cache."
                )
        statistics = {
            name: {key: data[f"{key}__{name}"] for key in STATISTICS_KEYS}
            for name in series
        }
        noise_ceiling = float(data["noise_ceiling"])
    return statistics, noise_ceiling


def null_noise_ceiling(prior_distance, prior_distance_error):
    """Upper 95% envelope max [D + 1.96 SE] of the null series' prior distance."""
    return float(np.max(prior_distance + 1.96 * prior_distance_error))
