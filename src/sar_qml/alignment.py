"""Task-spectrum alignment of the query branch of a re-uploading circuit.

Supporting code for ``notebooks/fig_03_approx_spectrum.ipynb``:

* reachability: generator scales, the accessible frequency set of one query
  coordinate and the coverage error of a task mode set;
* a batched NumPy statevector toolkit (Pauli matrices, tensor-product
  rotation layers, the CNOT ring, the ``Z_0`` observable);
* the spectral-prior distance ``D_Sigma`` between second-moment matrices of
  Fourier coefficient vectors, with a delete-a-block jackknife estimator;
* the consistency checks printed by the notebook.

The circuit itself is defined in the notebook and passed to the functions
here as a callable ``circuit(parameters, signal_points, query, scales,
generator_name)`` that returns ``<Z_0>`` with shape
``(number of parameter vectors, number of signal points)``. Parameter arrays
have shape ``(batch, 2, number_of_layers, number_of_qubits)``.
"""

import numpy as np

PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)


# ---------------------------------------------------------------------------
# Reachability of the task modes
# ---------------------------------------------------------------------------


def generator_scales(number_of_layers, scheme, coordinate_extent):
    """Generator scales ``s_{l,n}`` for ``l = 1..L_f`` (identical for every n).

    ``scheme`` is ``"dyadic"`` (``R 2^{-(l+1)}``) or ``"uniform"``
    (``R / L_f``), with ``R = coordinate_extent``. Returns shape ``(L_f,)``.
    """
    layers = np.arange(1, number_of_layers + 1)
    if scheme == "dyadic":
        return coordinate_extent * 2.0 ** (-(layers + 1))
    if scheme == "uniform":
        return np.full(number_of_layers, coordinate_extent / number_of_layers)
    raise ValueError("Unknown scheme.")


def accessible_coordinate_spectrum(scales):
    """All frequencies ``sum_l c_l 2 s_l`` (``c_l in {-1, 0, 1}``) of one coordinate.

    Redundant signed-digit sums are deduplicated with a tolerance relative to
    the smallest scale. Returns the sorted distinct frequencies.
    """
    values = np.array([0.0])
    for scale in scales:
        values = np.concatenate(
            [values - 2.0 * scale, values, values + 2.0 * scale]
        )
    resolution = 2.0 * np.min(np.abs(scales))
    keys = np.round(values / resolution, 6)
    _, first = np.unique(keys, return_index=True)
    return np.sort(values[first])


def coverage_error(number_of_layers, scheme, task_modes, coordinate_extent):
    """``max_{nu in task_modes} dist(nu, Omega^{(f)})`` for the given scale scheme."""
    spectrum = accessible_coordinate_spectrum(
        generator_scales(number_of_layers, scheme, coordinate_extent)
    )
    distances = []
    for mode in task_modes:
        squared = 0.0
        for value in mode:
            squared += np.min((spectrum - value) ** 2)
        distances.append(np.sqrt(squared))
    return float(np.max(distances))


# ---------------------------------------------------------------------------
# Batched statevector toolkit
# ---------------------------------------------------------------------------


def pauli_z_diagonal(qubit, number_of_qubits):
    """Diagonal of ``Z`` acting on ``qubit`` (qubit 0 is the most significant bit)."""
    bits = (np.arange(2**number_of_qubits) >> (number_of_qubits - 1 - qubit)) & 1
    return np.where(bits == 0, PAULI_Z[0, 0].real, PAULI_Z[1, 1].real)


def _cnot(control, target, number_of_qubits):
    """Permutation matrix of a CNOT gate."""
    dimension = 2**number_of_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for state in range(dimension):
        bit = (state >> (number_of_qubits - 1 - control)) & 1
        image = state ^ (1 << (number_of_qubits - 1 - target)) if bit else state
        matrix[image, state] = 1.0
    return matrix


def cnot_ring(number_of_qubits):
    """``CNOT_{n-1,0} ... CNOT_{1,2} CNOT_{0,1}``, the entangling ring."""
    ring = np.eye(2**number_of_qubits, dtype=complex)
    for qubit in range(number_of_qubits):
        ring = _cnot(qubit, (qubit + 1) % number_of_qubits, number_of_qubits) @ ring
    return ring


def rotation_layer(angles, generator):
    """Batched ``tensor_n exp(-i angles[:, n] generator / 2)``.

    ``angles`` has shape ``(batch, number_of_qubits)``; returns
    ``(batch, 2**number_of_qubits, 2**number_of_qubits)``.
    """
    batch = angles.shape[0]
    result = np.ones((batch, 1, 1), dtype=complex)
    for index in range(angles.shape[1]):
        half = 0.5 * angles[:, index]
        block = (
            np.cos(half)[:, None, None] * np.eye(2)[None]
            - 1j * np.sin(half)[:, None, None] * generator[None]
        )
        result = np.einsum("bij,bkl->bikjl", result, block).reshape(
            batch, result.shape[1] * 2, result.shape[2] * 2
        )
    return result


# ---------------------------------------------------------------------------
# Spectral-prior distance and its jackknife estimator
# ---------------------------------------------------------------------------


def signal_grid(number_of_qubits, grid_size):
    """The ``grid_size**number_of_qubits`` points of the uniform grid on ``[0, 2 pi)^N``."""
    grid_1d = 2.0 * np.pi * np.arange(grid_size) / grid_size
    return np.stack(
        np.meshgrid(*([grid_1d] * number_of_qubits), indexing="ij"), axis=-1
    ).reshape(-1, number_of_qubits)


def coefficient_vectors(circuit, parameters, query, scales, generator_name,
                        grid, grid_size):
    """Exact Fourier coefficients over ``x`` of the circuit, one row per parameter vector.

    ``grid`` is ``signal_grid(number_of_qubits, grid_size)`` with
    ``grid_size = 2 L + 1``. Returns shape ``(batch, grid_size**N)``.
    """
    number_of_qubits = grid.shape[1]
    values = circuit(parameters, grid, query, scales, generator_name)
    axes = tuple(range(1, number_of_qubits + 1))
    spectrum = np.fft.fftn(
        values.reshape(parameters.shape[0], *([grid_size] * number_of_qubits)),
        axes=axes,
    ) / grid_size**number_of_qubits
    return spectrum.reshape(parameters.shape[0], -1)


def _alignment_distance(first, second):
    """1 - normalized Frobenius alignment of two second-moment matrices."""
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-30:
        return 0.0
    return 1.0 - np.clip(
        np.real(np.vdot(first, second)) / denominator, 0.0, 1.0
    )


def _gram_blocks(vectors, number_of_blocks):
    """Total Gram matrix, per-block Gram matrices and block sizes."""
    blocks = np.array_split(np.arange(vectors.shape[0]), number_of_blocks)
    partial = [
        vectors[index].conj().T @ vectors[index] for index in blocks
    ]
    total = sum(partial)
    sizes = np.array([len(index) for index in blocks])
    return total, partial, sizes


def jackknife_prior_distance(circuit, scale_parameter, task_query, scales,
                             generator_name, rng, number_of_samples,
                             number_of_layers, number_of_blocks=10,
                             batch_size=100):
    """Jackknife bias-corrected ``D_Sigma`` along the ray ``f = t f^*``.

    Draws ``number_of_samples`` parameter vectors uniformly from
    ``[0, 2 pi)`` with ``rng``; the same ensemble (and the same deleted block)
    is used for ``Sigma(t f^*)`` and ``Sigma(0)``. ``scale_parameter`` holds
    the values of ``t``. Returns the corrected estimate, its jackknife
    standard error and the raw estimate, each of shape
    ``(len(scale_parameter),)``.
    """
    number_of_qubits = len(task_query)
    grid_size = 2 * number_of_layers + 1
    grid = signal_grid(number_of_qubits, grid_size)
    parameters = rng.uniform(
        0.0, 2.0 * np.pi,
        (number_of_samples, 2, number_of_layers, number_of_qubits),
    )

    def gram_at(query):
        vectors = []
        for start in range(0, number_of_samples, batch_size):
            vectors.append(coefficient_vectors(
                circuit, parameters[start:start + batch_size], query,
                scales, generator_name, grid, grid_size,
            ))
        return _gram_blocks(
            np.concatenate(vectors, axis=0), number_of_blocks
        )

    reference_total, reference_partial, sizes = gram_at(
        np.zeros(number_of_qubits)
    )

    corrected, errors, raw = [], [], []
    for value in scale_parameter:
        total, partial, _ = gram_at(value * task_query)
        full = _alignment_distance(
            total / number_of_samples, reference_total / number_of_samples
        )
        replicates = []
        for index in range(number_of_blocks):
            retained = number_of_samples - sizes[index]
            replicates.append(_alignment_distance(
                (total - partial[index]) / retained,
                (reference_total - reference_partial[index]) / retained,
            ))
        replicates = np.array(replicates)
        mean = replicates.mean()
        corrected.append(
            number_of_blocks * full - (number_of_blocks - 1) * mean
        )
        errors.append(np.sqrt(
            (number_of_blocks - 1) / number_of_blocks
            * np.sum((replicates - mean) ** 2)
        ))
        raw.append(full)
    return np.array(corrected), np.array(errors), np.array(raw)


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def report(name, passed, detail=""):
    """Print ``[PASS]``/``[FAIL]`` with an optional detail; raise if it failed."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")
    assert passed, name


def check_task_modes(task_modes, coordinate_extent):
    """Size, sparsity, extent and zero mode of the task mode set."""
    print("\nTask modes")

    report("41 distinct task modes (10 nonzero per coordinate, plus zero)",
           len(task_modes) == 41, f"got {len(task_modes)}")
    report(
        "every task mode has at most one active query coordinate",
        all(np.count_nonzero(mode) <= 1 for mode in task_modes),
    )
    report(
        "task modes lie within the coordinate extent",
        np.abs(task_modes).max() <= coordinate_extent + 1e-12,
        f"max |nu| = {np.abs(task_modes).max():.4f}",
    )
    report("the zero mode is present", any(
        np.allclose(mode, 0.0) for mode in task_modes
    ))


def check_query_spectrum(coordinate_extent):
    """Size, lattice spacing, span and symmetry of the dyadic accessible spectrum."""
    print("\nAccessible query spectrum")

    def dyadic_spectrum(layers):
        return accessible_coordinate_spectrum(
            generator_scales(layers, "dyadic", coordinate_extent)
        )

    for layers in (1, 2, 3, 4):
        spectrum = dyadic_spectrum(layers)
        report(
            f"L_f = {layers}: at most 3^L_f = {3**layers} distinct frequencies",
            len(spectrum) <= 3**layers,
            f"got {len(spectrum)}",
        )

    for layers in (2, 4, 6, 8):
        spectrum = dyadic_spectrum(layers)
        gap = np.diff(spectrum).max()
        predicted = coordinate_extent * 2.0 ** (-layers)
        report(
            f"L_f = {layers}: spacing equals R 2^(-L_f)",
            abs(gap - predicted) < 1e-9,
            f"gap {gap:.3e}, predicted {predicted:.3e}",
        )

    spectrum = dyadic_spectrum(6)
    report(
        "spectrum spans +- R (1 - 2^-L_f)",
        abs(spectrum.max() - coordinate_extent * (1.0 - 2.0 ** (-6))) < 1e-9,
        f"max {spectrum.max():.4f}",
    )
    report(
        "dyadic scales give a uniform lattice of 2^(L_f+1)-1 frequencies",
        len(spectrum) == 2**7 - 1,
        f"got {len(spectrum)}",
    )

    spectrum = dyadic_spectrum(4)
    report(
        "spectrum is symmetric under omega -> -omega",
        np.allclose(spectrum, -spectrum[::-1]),
    )


def check_reachability_bound(task_modes, coordinate_extent, number_of_queries):
    """Dyadic coverage error against ``sqrt(N) R 2^{-L_f}``, and against uniform scales."""
    print("\nTheorem 1")

    layer_range = np.arange(1, 11)
    errors = np.array([
        coverage_error(layers, "dyadic", task_modes, coordinate_extent)
        for layers in layer_range
    ])
    bounds = np.sqrt(number_of_queries) * coordinate_extent * 2.0 ** (-layer_range)

    report("dyadic coverage error respects the bound", np.all(errors <= bounds),
           f"max ratio {np.max(errors / bounds):.4f}")
    ratios = errors[:-1] / errors[1:]
    report("dyadic coverage error halves each layer",
           np.allclose(ratios, 2.0, atol=1e-9),
           f"ratios in [{ratios.min():.4f}, {ratios.max():.4f}]")

    uniform_errors = np.array([
        coverage_error(layers, "uniform", task_modes, coordinate_extent)
        for layers in layer_range
    ])
    report("mismatched scales converge only polynomially",
           uniform_errors[-1] / errors[-1] > 10.0,
           f"L_f=10: uniform {uniform_errors[-1]:.3e} vs "
           f"dyadic {errors[-1]:.3e}, ratio {uniform_errors[-1]/errors[-1]:.1f}")
    report("the advantage of the dyadic scales grows with depth",
           (uniform_errors / errors)[-1] > 4.0 * (uniform_errors / errors)[3],
           f"ratio L_f=4 {(uniform_errors/errors)[3]:.1f} -> "
           f"L_f=10 {(uniform_errors/errors)[-1]:.1f}")


def _check_ensemble(number_of_layers, number_of_qubits):
    """Small fixed parameter and signal sample (seed 0) for the circuit checks."""
    rng = np.random.default_rng(0)
    parameters = rng.uniform(
        0.0, 4.0 * np.pi,
        (4, 2, number_of_layers, number_of_qubits),
    )
    signal = rng.uniform(0.0, 2.0 * np.pi, (40, number_of_qubits))
    return parameters, signal


def check_absorbability(circuit, scales, query):
    """The ``Y_n`` query is absorbed by shifting ``theta_1`` by ``2 s f``; ``Z_n`` is not."""
    print("\nAbsorbability (Proposition 7)")

    number_of_layers = len(scales)
    parameters, signal = _check_ensemble(number_of_layers, len(query))
    shifted = parameters.copy()
    for layer in range(number_of_layers):
        shifted[:, 1, layer, :] = (
            parameters[:, 1, layer, :] + 2.0 * scales[layer] * query
        )

    control_query = circuit(parameters, signal, query, scales, "Y")
    control_zero = circuit(shifted, signal, np.zeros_like(query), scales, "Y")
    residual = np.abs(control_query - control_zero).max()
    report(
        "Y_n query is exactly absorbed by shifting the adjacent R_y block",
        residual < 1e-12, f"max residual {residual:.2e}",
    )

    aligned_query = circuit(parameters, signal, query, scales, "Z")
    aligned_zero = circuit(shifted, signal, np.zeros_like(query), scales, "Z")
    residual = np.abs(aligned_query - aligned_zero).max()
    report(
        "Z_n query is not absorbed by the same shift",
        residual > 0.1, f"max residual {residual:.3f}",
    )

    report(
        "Y_n and Z_n generators share their eigenvalues, hence Omega^(f)",
        np.allclose(
            np.sort(np.linalg.eigvalsh(PAULI_Y)),
            np.sort(np.linalg.eigvalsh(PAULI_Z)),
        ),
    )


def check_parameter_periodicity(circuit, scales, query):
    """``R_y(theta + 2 pi) = -R_y(theta)`` leaves expectation values unchanged."""
    print("\nParameter ensemble periodicity")

    def rotation(angle):
        return np.cos(0.5 * angle) * np.eye(2) - 1j * np.sin(0.5 * angle) * PAULI_Y

    report(
        "R_y(theta + 2 pi) = - R_y(theta)",
        np.abs(rotation(1.234 + 2.0 * np.pi) + rotation(1.234)).max() < 1e-14,
    )

    parameters, signal = _check_ensemble(len(scales), len(query))
    shifted_by_period = parameters.copy()
    shifted_by_period[:, 1, 0, 0] += 2.0 * np.pi
    report(
        "expectation values are invariant under theta -> theta + 2 pi",
        np.abs(
            circuit(parameters, signal, query, scales, "Y")
            - circuit(shifted_by_period, signal, query, scales, "Y")
        ).max() < 1e-12,
    )
