"""Query-encoding sweep: the query generator is interpolated at a fixed query value.

Used by ``fig_01_vary_encoding.ipynb``. Each query generator interpolates out
of the local Pauli X_q, either towards Y_q (local, absorbable path) or towards
the nonlocal target Y_0 Z_1 / Z_0 Y_1 (nonlocal path), through the angle
alpha. The module contains

* the interpolated generators G_q(alpha) and the query layer E_f^(l)(f; alpha),
* the precomputed Fourier grid of the sweep (:class:`EncodingSweep`) and the
  x-coefficient matrices A(theta, alpha) on it,
* the structural validations (generator spectra, endpoints, absorbability,
  invariance of the joint query spectrum),
* the diagnostics: shape change, separability measures and the
  block-jackknife spectral-prior distance, accumulated over a parameter
  ensemble without storing per-sample coefficients.
"""

from dataclasses import dataclass

import numpy as np

from .reuploading import (
    I4,
    INITIAL_STATE,
    X0,
    X1,
    Y0,
    Y1,
    Z0,
    Z1,
    ZZ,
    data_encoding_grid,
    fourier_grid,
    fourier_modes,
    joint_query_spectrum,
    operator_nonseparable_weight,
    pauli_rotation,
    sample_parameters,
    trainable_blocks,
)

#: Local query generator on each qubit and its two spectrum-preserving targets.
LOCAL_GENERATORS = (X0, X1)
ROTATED_LOCAL_GENERATORS = (Y0, Y1)
NONLOCAL_GENERATORS = (Y0 @ Z1, Z0 @ Y1)
PATHS = ("local", "nonlocal")

# Norms and singular-value totals below this are treated as zero.
_ZERO_TOLERANCE = 1e-14


# --- query layer -----------------------------------------------------------


def _involution_rotation(generator, angle):
    r"""exp(-i angle G / 2) after checking that G squares to the identity."""
    identity = np.eye(generator.shape[0], dtype=complex)
    if not np.allclose(generator @ generator, identity, atol=1e-12):
        raise ValueError("The supplied generator does not square to identity.")
    return pauli_rotation(generator, angle)


def query_generator(encoding_angle, qubit, path):
    r"""Return the spectrum-preserving generator G_q(alpha) on ``path``."""
    local_generator = LOCAL_GENERATORS[qubit]
    if path == "local":
        second_generator = ROTATED_LOCAL_GENERATORS[qubit]
    elif path == "nonlocal":
        second_generator = NONLOCAL_GENERATORS[qubit]
    else:
        raise ValueError("path must be either 'local' or 'nonlocal'.")

    generator = (
        np.cos(encoding_angle) * local_generator
        + np.sin(encoding_angle) * second_generator
    )
    if not np.allclose(generator @ generator, I4, atol=1e-12):
        raise RuntimeError(
            "The interpolated query generator does not square to identity."
        )
    return generator


def query_encoding(query_vector, encoding_angle, layer, path, query_weights):
    r"""
    Return E_f^(l)(f; alpha), the query layer of the reuploading circuit.

    Qubit q is rotated by s_l f_q about G_q(alpha), in the order q = 0 then
    q = 1, with layer weight s_l = ``query_weights[layer]``.
    """
    weight = query_weights[layer]
    gate = I4
    for qubit in range(2):
        gate = (
            _involution_rotation(
                query_generator(encoding_angle, qubit, path),
                weight * query_vector[qubit],
            )
            @ gate
        )
    return gate


def accessible_query_modes(query_weights):
    """Minkowski sum of the layerwise query frequency sets {-s_l, 0, s_l}."""
    modes = {0}
    for weight in query_weights:
        layer_modes = {-int(weight), 0, int(weight)}
        modes = {left + right for left in modes for right in layer_modes}
    return np.asarray(sorted(modes), dtype=int)


# --- Fourier grid of the sweep ---------------------------------------------


@dataclass(frozen=True)
class EncodingSweep:
    """
    Input-independent objects of the encoding sweep.

    ``data_encodings`` holds E_x on the flattened x grid, shape
    (grid_size**2, 4, 4); ``query_encodings[path]`` holds the query layers,
    shape (number of angles, number of layers, 4, 4); ``x_mode_indices``
    selects the retained x modes ``x_modes`` from the fftshifted FFT axis.
    """

    encoding_angles: np.ndarray
    grid_size: int
    data_encodings: np.ndarray
    query_encodings: dict
    x_mode_indices: np.ndarray
    x_modes: np.ndarray

    @property
    def number_of_layers(self):
        return self.query_encodings[PATHS[0]].shape[1]


def build_encoding_sweep(
    fixed_query_vector, encoding_angles, grid_size, number_of_layers, query_weights
):
    """
    Precompute the data encodings on the x grid and the query layers at every alpha.

    The retained x modes are |omega| <= number_of_layers per coordinate.
    """
    data_encodings = data_encoding_grid(fourier_grid(grid_size)).reshape(
        grid_size**2, 4, 4
    )
    query_encodings = {
        path: np.asarray([
            [
                query_encoding(
                    fixed_query_vector, encoding_angle, layer, path, query_weights
                )
                for layer in range(number_of_layers)
            ]
            for encoding_angle in encoding_angles
        ])
        for path in PATHS
    }
    full_modes = fourier_modes(grid_size)
    x_mode_indices = np.flatnonzero(np.abs(full_modes) <= number_of_layers)
    return EncodingSweep(
        encoding_angles=encoding_angles,
        grid_size=grid_size,
        data_encodings=data_encodings,
        query_encodings=query_encodings,
        x_mode_indices=x_mode_indices,
        x_modes=full_modes[x_mode_indices],
    )


def coefficient_matrices(sweep, parameters, path, block_type):
    r"""
    Return the x-coefficient matrices A(theta, alpha) for every encoding angle.

    The circuit is evaluated on the x grid at every alpha of ``sweep``; a 2D
    FFT gives an array of shape (number of angles, n_x, n_x) over the
    retained modes (omega_x0, omega_x1).
    """
    blocks = trainable_blocks(parameters, block_type)
    number_of_angles = len(sweep.encoding_angles)
    grid_size = sweep.grid_size

    states = np.broadcast_to(
        INITIAL_STATE,
        (number_of_angles, grid_size**2, INITIAL_STATE.size),
    ).copy()

    for layer in range(sweep.number_of_layers):
        W_layer = blocks[2 * layer]
        V_layer = blocks[1 + 2 * layer]

        states = np.einsum(
            "xij,axj->axi", sweep.data_encodings, states, optimize=True
        )
        states = np.einsum("ij,axj->axi", W_layer, states, optimize=True)
        states = np.einsum(
            "aij,axj->axi",
            sweep.query_encodings[path][:, layer],
            states,
            optimize=True,
        )
        states = np.einsum("ij,axj->axi", V_layer, states, optimize=True)

    expectations = np.einsum(
        "axi,ij,axj->ax", states.conj(), ZZ, states, optimize=True
    )
    maximum_imaginary_part = np.max(np.abs(expectations.imag))
    if maximum_imaginary_part > 1e-10:
        raise RuntimeError(
            "Expectation values have unexpectedly large imaginary parts: "
            f"{maximum_imaginary_part:.3e}."
        )

    values = expectations.real.reshape(number_of_angles, grid_size, grid_size)
    full_coefficients = np.fft.fftshift(
        np.fft.fft2(values, axes=(1, 2)) / grid_size**2,
        axes=(1, 2),
    )
    indices = sweep.x_mode_indices
    return full_coefficients[:, indices][:, :, indices]


# --- structural validation -------------------------------------------------


def validate_encoding_paths(
    encoding_angles, fixed_query_vector, query_weights, number_of_layers
):
    """
    Check the generators and the absorbability of the query layers along both paths.

    Every generator must be Hermitian with spectrum {-1, -1, 1, 1}; both
    paths must start at X_q and end at their targets, the nonlocal targets
    must differ, the local query layer must be a product unitary for every
    alpha and the nonlocal one only at alpha = 0. Returns the largest
    local-path and the smallest nonlocal-endpoint operator nonseparable weight.
    """
    target_spectrum = np.array([-1.0, -1.0, 1.0, 1.0])
    for path in PATHS:
        for encoding_angle in encoding_angles:
            for qubit in range(2):
                generator = query_generator(encoding_angle, qubit, path)
                if not np.allclose(generator, generator.conj().T, atol=1e-12):
                    raise RuntimeError("A query generator is not Hermitian.")
                eigenvalues = np.linalg.eigvalsh(generator)
                if not np.allclose(eigenvalues, target_spectrum, atol=1e-12):
                    raise RuntimeError(
                        "The query-generator spectrum changes along a path."
                    )

        # Both paths start at the same local encoding.
        for qubit in range(2):
            if not np.allclose(
                query_generator(0.0, qubit, path),
                LOCAL_GENERATORS[qubit],
                atol=1e-12,
            ):
                raise RuntimeError("An alpha=0 generator is not the local Pauli.")

    for qubit in range(2):
        if not np.allclose(
            query_generator(np.pi / 2.0, qubit, "local"),
            ROTATED_LOCAL_GENERATORS[qubit],
            atol=1e-12,
        ):
            raise RuntimeError(
                "The local-path endpoint is not the rotated local Pauli."
            )
        if not np.allclose(
            query_generator(np.pi / 2.0, qubit, "nonlocal"),
            NONLOCAL_GENERATORS[qubit],
            atol=1e-12,
        ):
            raise RuntimeError(
                "The nonlocal-path endpoint is not the nonlocal target."
            )

    # Coinciding nonlocal targets would turn the query gate into a single
    # two-qubit rotation and collapse the joint query spectrum onto its
    # diagonal at the endpoint.
    if np.allclose(NONLOCAL_GENERATORS[0], NONLOCAL_GENERATORS[1], atol=1e-12):
        raise RuntimeError(
            "The nonlocal query generators coincide, which would make "
            "the query gate depend on f_0 + f_1 alone."
        )

    def layer_weight(encoding_angle, layer, path):
        return operator_nonseparable_weight(
            query_encoding(
                fixed_query_vector, encoding_angle, layer, path, query_weights
            )
        )

    # Absorbability: the local path stays a product unitary for every
    # alpha, the nonlocal path leaves that set for alpha > 0.
    local_defect = 0.0
    nonlocal_defect_at_zero = 0.0
    for encoding_angle in encoding_angles:
        for layer in range(number_of_layers):
            local_defect = max(
                local_defect, layer_weight(encoding_angle, layer, "local")
            )
    for layer in range(number_of_layers):
        nonlocal_defect_at_zero = max(
            nonlocal_defect_at_zero, layer_weight(0.0, layer, "nonlocal")
        )
    if local_defect > 1e-12:
        raise RuntimeError(
            "A local-path query gate is not a product unitary "
            f"(nonseparable weight {local_defect:.3e})."
        )
    if nonlocal_defect_at_zero > 1e-12:
        raise RuntimeError(
            "The alpha=0 nonlocal query gate is not a product unitary."
        )
    nonlocal_defect_at_end = min(
        layer_weight(np.pi / 2.0, layer, "nonlocal")
        for layer in range(number_of_layers)
    )
    if nonlocal_defect_at_end < 1e-3:
        raise RuntimeError("The nonlocal-path endpoint is still a product unitary.")

    return {
        "local_operator_nonseparable_weight": local_defect,
        "nonlocal_endpoint_nonseparable_weight": nonlocal_defect_at_end,
    }


def validate_query_spectrum(
    query_weights,
    query_grid_size,
    check_angles,
    data_point,
    seed,
    number_of_samples,
    occupation_tolerance,
    block_type,
):
    r"""
    Check that the joint query spectrum is the same at every alpha on both paths.

    At each of ``check_angles`` the occupied (f_0, f_1) modes are the modes
    whose coefficient magnitude, maximized over ``number_of_samples`` Haar
    parameter vectors, exceeds ``occupation_tolerance``. No weight may lie
    outside the accessible band, and the occupied set must equal the one at
    the first angle; for ``block_type="cnot"`` a changing set only warns.
    Returns, per path, the out-of-support weight, the support size and
    whether the support is invariant.
    """
    number_of_layers = len(query_weights)
    query_modes = accessible_query_modes(query_weights)
    full_modes = fourier_modes(query_grid_size)
    inside = np.abs(full_modes) <= int(np.max(query_modes))
    inside_mask = np.outer(inside, inside)
    mode_grid = np.stack(
        np.meshgrid(full_modes, full_modes, indexing="ij"),
        axis=-1,
    )
    rng = np.random.default_rng(seed)
    parameter_samples = [
        sample_parameters(rng, "haar", 2 * number_of_layers)
        for _ in range(number_of_samples)
    ]

    report = {}
    for path in PATHS:
        leakage = 0.0
        support_sets = {}
        for encoding_angle in check_angles:
            def query_layer(query_vector, layer):
                return query_encoding(
                    query_vector, encoding_angle, layer, path, query_weights
                )

            magnitudes = np.zeros_like(inside_mask, dtype=float)
            for parameters in parameter_samples:
                coefficients = joint_query_spectrum(
                    parameters,
                    block_type,
                    query_layer,
                    data_point,
                    number_of_layers,
                    query_grid_size,
                )
                np.maximum(magnitudes, np.abs(coefficients), out=magnitudes)
            leakage = max(leakage, float(np.max(magnitudes[~inside_mask])))
            occupied = magnitudes > occupation_tolerance
            support_sets[float(encoding_angle)] = frozenset(
                map(tuple, mode_grid[occupied].tolist())
            )

        if leakage > 1e-10:
            raise RuntimeError(
                f"The {path} path leaks query-Fourier weight outside the "
                f"predicted support (max {leakage:.3e})."
            )

        reference_support = support_sets[float(check_angles[0])]
        changed_angles = [
            angle
            for angle, support in support_sets.items()
            if support != reference_support
        ]
        invariant = not changed_angles
        if not invariant:
            largest_difference = max(
                len(support_sets[angle] ^ reference_support)
                for angle in changed_angles
            )
            message = (
                f"The query spectrum of the {path} path changes along the "
                f"encoding interpolation with TRAINABLE_BLOCKS="
                f"{block_type}: the occupied mode set differs from its "
                f"alpha=0 value at {len(changed_angles)} of "
                f"{len(support_sets)} sampled angles, by up to "
                f"{largest_difference} modes."
            )
            if block_type == "product":
                raise RuntimeError(message)
            # The CNOT variant is a robustness check outside the figure;
            # its spectrum is not invariant, so we warn instead of failing.
            print(f"Warning: {message}")

        report[path] = {
            "out_of_support_weight": leakage,
            "support_size": len(reference_support),
            "support_invariant": invariant,
        }
    return report


# --- diagnostics -----------------------------------------------------------


def fixed_parameter_shape_change(coefficient_path):
    r"""
    Shape change Delta(alpha) of the coefficient direction relative to alpha = 0.

    ``coefficient_path`` is the output of :func:`coefficient_matrices` for
    one parameter vector. Values below 1e-12 are set to zero.
    """
    vectors = coefficient_path.reshape(coefficient_path.shape[0], -1)
    reference = vectors[0]
    norms = np.linalg.norm(vectors, axis=1)
    reference_norm = np.linalg.norm(reference)
    if reference_norm <= _ZERO_TOLERANCE or np.any(norms <= _ZERO_TOLERANCE):
        raise RuntimeError("Encountered a zero Fourier coefficient vector.")

    overlaps = np.einsum("ai,i->a", vectors.conj(), reference, optimize=True)
    similarities = np.abs(overlaps) / (norms * reference_norm)
    changes = 1.0 - np.clip(similarities, 0.0, 1.0)
    changes[np.abs(changes) < 1e-12] = 0.0
    return changes


def normalized_matrix_distance(matrix, reference):
    r"""One minus the normalized real Frobenius alignment; fails on a zero matrix."""
    denominator = np.linalg.norm(matrix) * np.linalg.norm(reference)
    if denominator <= _ZERO_TOLERANCE:
        raise RuntimeError("Cannot normalize a zero second-moment matrix.")
    alignment = np.vdot(matrix, reference).real / denominator
    return 1.0 - np.clip(alignment, 0.0, 1.0)


def separability_measures(coefficient_path):
    r"""
    Nonseparable weight and entropy effective Schmidt rank of every A(theta, alpha).

    Returns two arrays over the encoding angles: 1 - sigma_1^2 / sum sigma_i^2
    and exp of the entropy of the normalized squared singular values.
    """
    singular_values = np.linalg.svd(coefficient_path, compute_uv=False)
    weights = singular_values**2
    totals = np.sum(weights, axis=1, keepdims=True)
    if np.any(totals <= _ZERO_TOLERANCE):
        raise RuntimeError("Encountered a zero coefficient matrix.")

    probabilities = weights / totals
    nonseparable = 1.0 - probabilities[:, 0]

    positive = probabilities > _ZERO_TOLERANCE
    safe_probabilities = np.where(positive, probabilities, 1.0)
    entropies = -np.sum(
        np.where(positive, probabilities * np.log(safe_probabilities), 0.0),
        axis=1,
    )
    return nonseparable, np.exp(entropies)


def jackknife_prior_statistics(total_moments, block_moments, block_sizes):
    r"""
    Bias-corrected prior distance D_Sigma(alpha) from alpha = 0 and its standard error.

    ``total_moments[a]`` is the summed a a^dagger at angle index a and
    ``block_moments[b, a]`` the same sum over jackknife block b. Returns the
    corrected distances (clipped at zero), their block-jackknife standard
    errors, the raw distances and the second-moment matrices Sigma(alpha).
    """
    number_of_samples = int(np.sum(block_sizes))
    number_of_blocks = len(block_sizes)
    prior_matrices = total_moments / number_of_samples
    reference = prior_matrices[0]

    raw_distances = np.asarray([
        normalized_matrix_distance(matrix, reference)
        for matrix in prior_matrices
    ])
    corrected_distances = np.empty_like(raw_distances)
    standard_errors = np.empty_like(raw_distances)

    for alpha_index in range(len(total_moments)):
        leave_one_out = []
        for block_index, block_size in enumerate(block_sizes):
            retained = number_of_samples - int(block_size)
            sigma_loo = (
                total_moments[alpha_index]
                - block_moments[block_index, alpha_index]
            ) / retained
            reference_loo = (
                total_moments[0] - block_moments[block_index, 0]
            ) / retained
            leave_one_out.append(
                normalized_matrix_distance(sigma_loo, reference_loo)
            )

        leave_one_out = np.asarray(leave_one_out)
        jackknife_mean = np.mean(leave_one_out)
        corrected = (
            number_of_blocks * raw_distances[alpha_index]
            - (number_of_blocks - 1) * jackknife_mean
        )
        corrected_distances[alpha_index] = max(0.0, corrected)
        standard_errors[alpha_index] = np.sqrt(
            (number_of_blocks - 1)
            / number_of_blocks
            * np.sum((leave_one_out - jackknife_mean) ** 2)
        )

    corrected_distances[0] = 0.0
    standard_errors[0] = 0.0
    return corrected_distances, standard_errors, raw_distances, prior_matrices


def ensemble_statistics(
    sweep,
    series,
    path_of_series,
    distribution_of_series,
    number_of_samples,
    number_of_schmidt_samples,
    number_of_jackknife_blocks,
    seeds,
    narrow_sigma,
    block_type,
):
    r"""
    Accumulate the spectral second moments and separability statistics per series.

    For each of ``number_of_samples`` draws, one parameter vector per prior
    is sampled from ``np.random.default_rng(seeds[prior])`` and every series
    evaluates the coefficient matrices of its path with the parameters of its
    prior. Nothing is stored per sample: the moments a a^dagger are summed in
    place (also per jackknife block), and the class witness is a running
    maximum over the first ``number_of_schmidt_samples`` draws. Returns, per
    series, the jackknife prior distance and its error, the raw distance,
    Sigma(alpha), the class witness S(alpha) and the mean nonseparable
    weight and effective Schmidt rank.
    """
    number_of_angles = len(sweep.encoding_angles)
    number_of_x_coefficients = len(sweep.x_modes) ** 2
    number_of_blocks = min(number_of_jackknife_blocks, number_of_samples)
    sample_blocks = np.array_split(np.arange(number_of_samples), number_of_blocks)
    block_assignment = np.empty(number_of_samples, dtype=int)
    block_sizes = np.empty(number_of_blocks, dtype=int)
    for block_index, indices in enumerate(sample_blocks):
        block_assignment[indices] = block_index
        block_sizes[block_index] = len(indices)

    moment_shape = (
        number_of_angles,
        number_of_x_coefficients,
        number_of_x_coefficients,
    )
    total_moments = {
        name: np.zeros(moment_shape, dtype=complex) for name in series
    }
    block_moments = {
        name: np.zeros((number_of_blocks,) + moment_shape, dtype=complex)
        for name in series
    }

    number_of_schmidt_samples = min(number_of_schmidt_samples, number_of_samples)
    maximum_nonseparable = {name: np.zeros(number_of_angles) for name in series}
    nonseparable_sum = {name: np.zeros(number_of_angles) for name in series}
    schmidt_sum = {name: np.zeros(number_of_angles) for name in series}

    rngs = {
        distribution: np.random.default_rng(seed)
        for distribution, seed in seeds.items()
    }
    number_of_trainable_blocks = 2 * sweep.number_of_layers
    progress_interval = max(1, number_of_samples // 10)

    for sample_index in range(number_of_samples):
        parameters = {
            distribution: sample_parameters(
                rng, distribution, number_of_trainable_blocks, narrow_sigma
            )
            for distribution, rng in rngs.items()
        }

        for name in series:
            coefficient_path = coefficient_matrices(
                sweep,
                parameters[distribution_of_series[name]],
                path_of_series[name],
                block_type,
            )
            vectors = coefficient_path.reshape(
                number_of_angles, number_of_x_coefficients
            )

            # Outer products a a^dagger, matching the definition of Sigma.
            moments = np.einsum(
                "ai,aj->aij", vectors, vectors.conj(), optimize=True
            )
            total_moments[name] += moments
            block_moments[name][block_assignment[sample_index]] += moments

            if sample_index < number_of_schmidt_samples:
                nonseparable, ranks = separability_measures(coefficient_path)
                np.maximum(
                    maximum_nonseparable[name],
                    nonseparable,
                    out=maximum_nonseparable[name],
                )
                nonseparable_sum[name] += nonseparable
                schmidt_sum[name] += ranks

        completed = sample_index + 1
        if completed % progress_interval == 0 or completed == number_of_samples:
            print(f"Processed {completed}/{number_of_samples} parameter samples.")

    statistics = {}
    for name in series:
        (
            prior_distance,
            prior_distance_error,
            prior_distance_raw,
            prior_matrices,
        ) = jackknife_prior_statistics(
            total_moments[name], block_moments[name], block_sizes
        )
        statistics[name] = {
            "prior_distance": prior_distance,
            "prior_distance_error": prior_distance_error,
            "prior_distance_raw": prior_distance_raw,
            "prior_matrices": prior_matrices,
            "nonseparable_weight": maximum_nonseparable[name],
            "mean_nonseparable_weight": (
                nonseparable_sum[name] / number_of_schmidt_samples
            ),
            "mean_schmidt_rank": schmidt_sum[name] / number_of_schmidt_samples,
        }
    return statistics
