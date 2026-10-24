"""Query-value sweep: the query moves along a ray at a fixed query encoding.

Used by ``fig_02_vary_query.ipynb``. The query f(t) = t d is swept along a
ray with direction d, and the query layer uses one of two isospectral
generator pairs: the local pair (X_0, X_1), which is absorbable into product
trainable blocks, or the nonlocal pair (Y_0 Z_1, Z_0 Y_1). The module
contains

* the query layer E_f^(l)(f) and the accessible ray frequencies,
* the precomputed Fourier grid of the sweep (:class:`RaySweep`) and the
  coefficient tensor C[omega_x0, omega_x1, omega_t] of one parameter vector,
* the structural validations (absorbability, equal joint query spectra),
* the diagnostics: shape change, separability measures and the
  block-jackknife spectral-prior distance over a sampled ensemble of
  coefficient tensors.
"""

from dataclasses import dataclass

import numpy as np

from .reuploading import (
    INITIAL_STATE,
    X0,
    X1,
    Y,
    Z,
    ZZ,
    data_encoding_grid,
    fourier_grid,
    fourier_modes,
    joint_query_spectrum,
    kron,
    operator_nonseparable_weight,
    pauli_rotation,
    sample_parameters,
    trainable_blocks,
)

#: Query generator pairs (G_0, G_1), one generator per qubit.
QUERY_GENERATORS = {
    "local": (X0, X1),
    "nonlocal": (kron(Y, Z), kron(Z, Y)),
}
QUERY_TYPES = ("local", "nonlocal")


# --- query layer and ray spectrum ------------------------------------------


def query_encoding(query_vector, layer, query_type, query_weights):
    r"""
    Return E_f^(l)(f), the query layer of the reuploading circuit.

    Qubit q is rotated by s_l f_q about G_q, with s_l =
    ``query_weights[layer]``; the two generators of a pair commute, so the
    order of the factors does not matter.
    """
    if query_type not in QUERY_GENERATORS:
        raise ValueError("query_type must be either 'local' or 'nonlocal'.")

    generator_0, generator_1 = QUERY_GENERATORS[query_type]
    weight = query_weights[layer]
    return pauli_rotation(
        generator_1, weight * query_vector[1]
    ) @ pauli_rotation(generator_0, weight * query_vector[0])


def query_vector_at(ray_parameter, query_direction):
    """Point on the query ray f(t) = t * query_direction."""
    return ray_parameter * query_direction


def accessible_ray_modes(query_direction, query_weights):
    r"""
    Frequencies of t accessible to the circuit along f(t) = t d.

    Layer l contributes the halved eigenvalue differences of
    d_0 G_0 + d_1 G_1 scaled by s_l, and the accessible modes are their
    Minkowski sum over layers. Both generator pairs must give the same
    integer set, which is returned sorted.
    """
    per_query_type = {}
    for query_type in QUERY_TYPES:
        generator_0, generator_1 = QUERY_GENERATORS[query_type]
        layer_generator = (
            query_direction[0] * generator_0 + query_direction[1] * generator_1
        )
        eigenvalues = np.linalg.eigvalsh(layer_generator)
        differences = {
            (left - right) / 2.0 for left in eigenvalues for right in eigenvalues
        }

        modes = {0.0}
        for weight in query_weights:
            layer_modes = {weight * difference for difference in differences}
            modes = {left + right for left in modes for right in layer_modes}

        if any(abs(mode - round(mode)) > 1e-9 for mode in modes):
            raise ValueError(
                "QUERY_DIRECTION must give integer ray frequencies so "
                "that the ray spectrum lies on the Fourier grid; use "
                "integer components."
            )
        per_query_type[query_type] = frozenset(round(mode) for mode in modes)

    # Isospectral generator pairs must predict the same ray spectrum;
    # validate_query_spectrum() confirms this numerically.
    distinct = set(per_query_type.values())
    if len(distinct) != 1:
        raise RuntimeError(
            "The query types predict different ray spectra: "
            + ", ".join(
                f"{query_type} has {len(modes)} modes"
                for query_type, modes in per_query_type.items()
            )
        )
    return np.asarray(sorted(distinct.pop()), dtype=int)


# --- Fourier grid of the sweep ---------------------------------------------


@dataclass(frozen=True)
class RaySweep:
    """
    Input-independent objects of the ray sweep.

    ``data_encodings[i_x0, i_x1]`` is E_x on the x grid and
    ``query_encodings[query_type][layer][j_t]`` the query layer on the ray
    grid. Ray modes with |omega_t| > ``maximum_ray_frequency`` must carry no
    weight.
    """

    grid_size: int
    ray_grid_size: int
    number_of_layers: int
    maximum_ray_frequency: int
    data_encodings: np.ndarray
    query_encodings: dict


def build_ray_sweep(
    grid_size,
    ray_grid_size,
    query_direction,
    query_weights,
    number_of_layers,
    maximum_ray_frequency,
):
    """Precompute the data encodings on the x grid and the query layers on the ray grid."""
    ray_grid = fourier_grid(ray_grid_size)
    query_encodings = {
        query_type: [
            [
                query_encoding(
                    query_vector_at(ray_parameter, query_direction),
                    layer,
                    query_type,
                    query_weights,
                )
                for ray_parameter in ray_grid
            ]
            for layer in range(number_of_layers)
        ]
        for query_type in QUERY_TYPES
    }
    return RaySweep(
        grid_size=grid_size,
        ray_grid_size=ray_grid_size,
        number_of_layers=number_of_layers,
        maximum_ray_frequency=maximum_ray_frequency,
        data_encodings=data_encoding_grid(fourier_grid(grid_size)),
        query_encodings=query_encodings,
    )


def coefficient_tensor(sweep, parameters, query_type, block_type):
    r"""
    Return C[omega_x0, omega_x1, omega_t] of one parameter vector from a 3D FFT.

    The circuit is evaluated on the (x_0, x_1, t) grid of ``sweep``. Weight
    outside the predicted ray band raises, since it would mean the ray grid
    aliases. Returns the coefficients over the retained modes together with
    the x modes and the ray modes.
    """
    blocks = trainable_blocks(parameters, block_type)
    grid_size = sweep.grid_size
    ray_grid_size = sweep.ray_grid_size
    values = np.empty((grid_size, grid_size, ray_grid_size), dtype=float)

    for ix0 in range(grid_size):
        for ix1 in range(grid_size):
            Ex = sweep.data_encodings[ix0][ix1]

            for jf in range(ray_grid_size):
                total_unitary = np.eye(4, dtype=complex)

                for layer in range(sweep.number_of_layers):
                    W_layer = blocks[2 * layer]
                    V_layer = blocks[1 + 2 * layer]
                    total_unitary = (
                        V_layer
                        @ sweep.query_encodings[query_type][layer][jf]
                        @ W_layer
                        @ Ex
                        @ total_unitary
                    )

                state = total_unitary @ INITIAL_STATE
                expectation = state.conj() @ ZZ @ state
                values[ix0, ix1, jf] = expectation.real

    full_coefficients = np.fft.fftshift(
        np.fft.fftn(values) / (grid_size * grid_size * ray_grid_size)
    )
    x_full_modes = fourier_modes(grid_size)
    ray_full_modes = fourier_modes(ray_grid_size)

    x_keep = np.abs(x_full_modes) <= sweep.number_of_layers
    f_keep = np.abs(ray_full_modes) <= sweep.maximum_ray_frequency

    if np.any(~f_keep):
        outside = float(np.max(np.abs(full_coefficients[..., ~f_keep])))
        if outside > 1e-10:
            raise RuntimeError(
                "Ray-Fourier weight appeared outside the predicted "
                f"band (max {outside:.3e}); RAY_GRID_SIZE is too small."
            )

    x_indices = np.flatnonzero(x_keep)
    f_indices = np.flatnonzero(f_keep)
    coefficients = full_coefficients[np.ix_(x_indices, x_indices, f_indices)]
    return coefficients, x_full_modes[x_keep], ray_full_modes[f_keep]


def coefficient_matrix_at_query(C, query_modes, query_value):
    r"""Return A(f(t)) = sum_omega C[..., omega] e^{i omega t}, indexed by the x modes."""
    phases = np.exp(1j * query_modes * query_value)
    return np.einsum("ijk,k->ij", C, phases)


def coefficient_vectors_at_query(coefficient_samples, query_modes, query_value):
    """Evaluate all sampled x-coefficient vectors at one ray parameter; shape (n, n_x**2)."""
    phases = np.exp(1j * query_modes * query_value)
    matrices = np.einsum("nijk,k->nij", coefficient_samples, phases)
    return matrices.reshape(len(coefficient_samples), -1)


def coefficient_rank(C, relative_tolerance):
    """Rank of C grouped as all x modes against the ray modes."""
    matrix = C.reshape(-1, C.shape[-1])
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(
        np.count_nonzero(singular_values > relative_tolerance * singular_values[0])
    )


# --- structural validation -------------------------------------------------


def validate_absorbability(query_values, query_direction, query_weights, number_of_layers):
    """
    Check that the local query layer is a product unitary and the nonlocal one is not.

    The operator nonseparable weight is evaluated at every layer and every
    ray parameter in ``query_values``. Returns the maximum per query type.
    """
    weights = {query_type: [] for query_type in QUERY_TYPES}
    for query_type in QUERY_TYPES:
        for ray_parameter in query_values:
            for layer in range(number_of_layers):
                weights[query_type].append(
                    operator_nonseparable_weight(
                        query_encoding(
                            query_vector_at(ray_parameter, query_direction),
                            layer,
                            query_type,
                            query_weights,
                        )
                    )
                )

    local_maximum = float(np.max(weights["local"]))
    nonlocal_maximum = float(np.max(weights["nonlocal"]))
    if local_maximum > 1e-12:
        raise RuntimeError(
            "The local query gate is not a product unitary "
            f"(nonseparable weight {local_maximum:.3e}); the "
            "absorption argument for Cases I and II fails."
        )
    if nonlocal_maximum <= 0.05:
        raise RuntimeError(
            "The nonlocal query gate is nearly a product unitary "
            f"(maximum nonseparable weight {nonlocal_maximum:.3e}); "
            "it would be effectively absorbable."
        )
    return {"local_maximum": local_maximum, "nonlocal_maximum": nonlocal_maximum}


def validate_query_spectrum(
    query_weights,
    query_grid_size,
    data_point,
    seed,
    number_of_samples,
    occupation_tolerance,
    block_type,
):
    r"""
    Check that both generator pairs occupy the same joint (f_0, f_1) query modes.

    For each pair the occupied set holds the modes whose coefficient
    magnitude, maximized over ``number_of_samples`` Haar parameter vectors,
    exceeds ``occupation_tolerance``. No weight may lie outside
    |omega| <= sum_l s_l, and the two sets must be equal; for
    ``block_type="cnot"`` a mismatch only warns. Returns, per query type,
    the out-of-support weight, the support size and whether it matches the
    other pair.
    """
    number_of_layers = len(query_weights)
    modes = fourier_modes(query_grid_size)
    per_coordinate_maximum = int(np.sum(query_weights))
    inside = np.abs(modes) <= per_coordinate_maximum
    inside_mask = np.outer(inside, inside)
    mode_grid = np.stack(np.meshgrid(modes, modes, indexing="ij"), axis=-1)

    rng = np.random.default_rng(seed)
    parameter_samples = [
        sample_parameters(rng, "haar", 2 * number_of_layers)
        for _ in range(number_of_samples)
    ]

    supports = {}
    report = {}
    for query_type in QUERY_TYPES:
        def query_layer(query_vector, layer):
            return query_encoding(query_vector, layer, query_type, query_weights)

        magnitudes = np.zeros_like(inside_mask, dtype=float)
        for parameters in parameter_samples:
            np.maximum(
                magnitudes,
                np.abs(
                    joint_query_spectrum(
                        parameters,
                        block_type,
                        query_layer,
                        data_point,
                        number_of_layers,
                        query_grid_size,
                    )
                ),
                out=magnitudes,
            )

        leakage = float(np.max(magnitudes[~inside_mask]))
        if leakage > 1e-10:
            raise RuntimeError(
                f"The {query_type} query leaks Fourier weight outside "
                f"the predicted support (max {leakage:.3e})."
            )

        occupied = magnitudes > occupation_tolerance
        supports[query_type] = frozenset(map(tuple, mode_grid[occupied].tolist()))
        report[query_type] = {
            "out_of_support_weight": leakage,
            "support_size": int(occupied.sum()),
        }

    difference = supports["local"] ^ supports["nonlocal"]
    identical = not difference
    for query_type in QUERY_TYPES:
        report[query_type]["support_matches_other"] = identical
    if not identical:
        message = (
            "The absorbable and nonabsorbable series do not have the "
            f"same joint query spectrum with TRAINABLE_BLOCKS="
            f"{block_type}: the occupied mode sets differ by "
            f"{len(difference)} modes "
            f"({report['local']['support_size']} versus "
            f"{report['nonlocal']['support_size']} modes). The "
            "comparison is then not spectrum controlled."
        )
        if block_type == "product":
            raise RuntimeError(message)
        # The CNOT variant is a robustness check outside the figure; its
        # spectrum depends on the blocks, so we warn instead of failing.
        print(f"Warning: {message}")

    return report


# --- diagnostics -----------------------------------------------------------


def fixed_parameter_shape_change(C, query_modes, query_values):
    r"""
    Shape change Delta(t) relative to t = 0 for one fixed parameter vector.

    ``C`` is a coefficient tensor from :func:`coefficient_tensor`; the
    shape change is evaluated at every ray parameter in ``query_values``.
    Values below 1e-12 are set to zero.
    """
    reference = coefficient_matrix_at_query(C, query_modes, 0.0).reshape(-1)
    reference_norm = np.linalg.norm(reference)

    changes = []
    for query_value in query_values:
        vector = coefficient_matrix_at_query(C, query_modes, query_value).reshape(-1)
        similarity = np.abs(np.vdot(vector, reference)) / (
            np.linalg.norm(vector) * reference_norm
        )
        changes.append(1.0 - np.clip(similarity, 0.0, 1.0))

    changes = np.asarray(changes)
    changes[np.abs(changes) < 1e-12] = 0.0
    return changes


def normalized_matrix_distance(matrix, reference):
    r"""One minus the normalized real Frobenius alignment."""
    alignment = np.vdot(matrix, reference).real / (
        np.linalg.norm(matrix) * np.linalg.norm(reference)
    )
    return 1.0 - np.clip(alignment, 0.0, 1.0)


def prior_statistics(coefficient_samples, query_modes, query_values, number_of_jackknife_blocks):
    r"""
    Bias-corrected prior distance D_Sigma(t) from t = 0 and its standard error.

    Sigma_f(t) is estimated from the sampled coefficient tensors at every
    ray parameter. Returns the jackknife-corrected distances (clipped at
    zero), their block-jackknife standard errors, the raw distances, the
    normalized Fourier power diag(Sigma) / tr(Sigma) and the list of
    second-moment matrices.
    """
    number_of_samples = len(coefficient_samples)
    number_of_blocks = min(number_of_jackknife_blocks, number_of_samples)
    blocks = np.array_split(np.arange(number_of_samples), number_of_blocks)

    reference_vectors = coefficient_vectors_at_query(
        coefficient_samples, query_modes, 0.0
    )
    reference_sum = reference_vectors.conj().T @ reference_vectors
    reference = reference_sum / number_of_samples
    reference_block_sums = [
        reference_vectors[index].conj().T @ reference_vectors[index]
        for index in blocks
    ]

    prior_matrices = []
    normalized_powers = []
    raw_distances = []
    corrected_distances = []
    standard_errors = []

    for query_value in query_values:
        vectors = coefficient_vectors_at_query(
            coefficient_samples, query_modes, query_value
        )
        sigma_sum = vectors.conj().T @ vectors
        sigma = sigma_sum / number_of_samples
        prior_matrices.append(sigma)

        power = np.real(np.diag(sigma))
        normalized_powers.append(power / np.sum(power))

        raw_distance = normalized_matrix_distance(sigma, reference)
        raw_distances.append(raw_distance)

        leave_one_block_out = []
        for index, reference_block_sum in zip(blocks, reference_block_sums):
            retained = number_of_samples - len(index)
            sigma_block_sum = vectors[index].conj().T @ vectors[index]
            sigma_loo = (sigma_sum - sigma_block_sum) / retained
            reference_loo = (reference_sum - reference_block_sum) / retained
            leave_one_block_out.append(
                normalized_matrix_distance(sigma_loo, reference_loo)
            )

        leave_one_block_out = np.asarray(leave_one_block_out)
        jackknife_mean = np.mean(leave_one_block_out)
        corrected = (
            number_of_blocks * raw_distance
            - (number_of_blocks - 1) * jackknife_mean
        )
        corrected_distances.append(max(0.0, corrected))
        standard_errors.append(
            np.sqrt(
                (number_of_blocks - 1)
                / number_of_blocks
                * np.sum((leave_one_block_out - jackknife_mean) ** 2)
            )
        )

    return (
        np.asarray(corrected_distances),
        np.asarray(standard_errors),
        np.asarray(raw_distances),
        np.asarray(normalized_powers),
        prior_matrices,
    )


def separability_measures(matrix, zero_tolerance=1e-15):
    r"""
    Nonseparable weight and entropy effective Schmidt rank of one coefficient matrix.

    Returns (0, 0) for a matrix with vanishing total weight.
    """
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    weights = singular_values**2
    total = np.sum(weights)
    if total <= zero_tolerance:
        return 0.0, 0.0

    weights = weights / total
    nonseparable = float(1.0 - weights[0])

    retained = weights[weights > zero_tolerance]
    effective_rank = float(np.exp(-np.sum(retained * np.log(retained))))
    return nonseparable, effective_rank


def separability_statistics(
    coefficient_samples, query_modes, query_values, number_of_schmidt_samples
):
    r"""
    Class witness S(t) and the mean separability measures at every ray parameter.

    Uses the first ``number_of_schmidt_samples`` coefficient tensors. Returns
    the maximum nonseparable weight, the mean nonseparable weight and the
    mean effective Schmidt rank.
    """
    number_used = min(number_of_schmidt_samples, len(coefficient_samples))
    selected = coefficient_samples[:number_used]

    maximum_weights = []
    mean_weights = []
    mean_ranks = []
    for query_value in query_values:
        measures = np.asarray([
            separability_measures(
                coefficient_matrix_at_query(C, query_modes, query_value)
            )
            for C in selected
        ])
        maximum_weights.append(np.max(measures[:, 0]))
        mean_weights.append(np.mean(measures[:, 0]))
        mean_ranks.append(np.mean(measures[:, 1]))

    return (
        np.asarray(maximum_weights),
        np.asarray(mean_weights),
        np.asarray(mean_ranks),
    )


def sample_coefficient_ensemble(
    sweep,
    series,
    query_type_of_series,
    distribution_of_series,
    number_of_samples,
    seeds,
    narrow_sigma,
    block_type,
    x_modes,
    query_modes,
):
    """
    Coefficient tensors of a parameter ensemble, one array per series.

    For each draw, one parameter vector per prior is sampled from
    ``np.random.default_rng(seeds[prior])`` and every series evaluates its
    query type with the parameters of its prior. The returned arrays have
    shape (number_of_samples, n_x, n_x, n_t); the mode sets must agree with
    ``x_modes`` and ``query_modes``.
    """
    rngs = {
        distribution: np.random.default_rng(seed)
        for distribution, seed in seeds.items()
    }
    number_of_trainable_blocks = 2 * sweep.number_of_layers
    coefficient_samples = {name: [] for name in series}

    for _ in range(number_of_samples):
        parameters = {
            distribution: sample_parameters(
                rng, distribution, number_of_trainable_blocks, narrow_sigma
            )
            for distribution, rng in rngs.items()
        }
        for name in series:
            C, current_x_modes, current_query_modes = coefficient_tensor(
                sweep,
                parameters[distribution_of_series[name]],
                query_type_of_series[name],
                block_type,
            )
            if not np.array_equal(current_x_modes, x_modes):
                raise RuntimeError("Inconsistent x-mode sets.")
            if not np.array_equal(current_query_modes, query_modes):
                raise RuntimeError("Inconsistent query-mode sets.")
            coefficient_samples[name].append(C)

    return {name: np.asarray(samples) for name, samples in coefficient_samples.items()}


def ensemble_statistics(
    coefficient_samples,
    query_modes,
    query_values,
    number_of_jackknife_blocks,
    number_of_schmidt_samples,
):
    """
    Prior distance and separability statistics of every series.

    Returns, per series, the jackknife prior distance and its error, the raw
    distance, the normalized power, the second-moment matrices, the class
    witness S(t) and the mean nonseparable weight and effective Schmidt rank.
    """
    statistics = {}
    for name, samples in coefficient_samples.items():
        (
            prior_distance,
            prior_distance_error,
            prior_distance_raw,
            normalized_power,
            prior_matrices,
        ) = prior_statistics(
            samples, query_modes, query_values, number_of_jackknife_blocks
        )
        (
            nonseparable_weight,
            mean_nonseparable_weight,
            mean_schmidt_rank,
        ) = separability_statistics(
            samples, query_modes, query_values, number_of_schmidt_samples
        )
        statistics[name] = {
            "prior_distance": prior_distance,
            "prior_distance_error": prior_distance_error,
            "prior_distance_raw": prior_distance_raw,
            "normalized_power": normalized_power,
            "prior_matrices": prior_matrices,
            "nonseparable_weight": nonseparable_weight,
            "mean_nonseparable_weight": mean_nonseparable_weight,
            "mean_schmidt_rank": mean_schmidt_rank,
        }
    return statistics
