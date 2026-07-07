from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from sar_qml.pqc import PQCClassifier, PQCConfig


DEFAULT_OUTPUT_ROOT = Path("outputs")


def make_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits / feature dimension")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of re-uploading layers")
    parser.add_argument(
        "--n-x-layers",
        type=int,
        default=None,
        help="Optional number of x-branch layers (defaults to n-layers)",
    )
    parser.add_argument(
        "--n-f-layers",
        type=int,
        default=None,
        help="Optional number of f-branch layers for double re-uploading (defaults to n-layers)",
    )
    parser.add_argument("--n-epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--n-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help=(
            "Optional number of training samples to use per fold. "
            "When unset, use the full fold training split."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help=(
            "Number of validation samples to use per fold. "
            "Set <=0 to disable subsampling and use full fold validation split."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Compute validation metrics every N epochs (reduces runtime when >1)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable tqdm progress bars (disabled by default for faster runs)",
    )
    parser.add_argument(
        "--catalyst-batch-size",
        type=int,
        default=64,
        help=(
            "AD minibatch size for Catalyst qjit-compiled optimizer steps. "
            "Use 0 for full-batch JAX jit mode (scales to ~10k samples; "
            "avoids Catalyst MLIR compile-time blow-up)."
        ),
    )
    parser.add_argument(
        "--readout-mode",
        "--readout_mode",
        dest="readout_mode",
        type=str,
        default="single",
        choices=["single", "linear"],
        help=(
            "Readout head type: 'single' keeps the original single-observable output; "
            "'linear' uses multiple Z expectations with a trainable linear logit."
        ),
    )
    parser.add_argument(
        "--readout-wires",
        "--readout_wires",
        dest="readout_wires",
        type=int,
        default=None,
        help=(
            "Number of Pauli-Z observables for linear readout. "
            "Ignored when --readout-mode=single; defaults to all qubits in linear mode."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--fixed-f",
        action="store_true",
        help=(
            "Treat the frequency configuration f as a dataset-level HYPERPARAMETER: "
            "draw a single f once and share it across all samples, so labels depend "
            "on x alone and the standard (x-only) and aligned models are directly "
            "comparable. Default (off) reproduces the original per-sample-f design."
        ),
    )
    parser.add_argument(
        "--f-encoding-axis",
        type=str,
        default="Y",
        choices=["X", "Y", "Z"],
        help=(
            "Rotation axis for encoding f in the aligned circuit's f-branch. "
            "'Y' = original additive scheme (a fixed f is absorbed into the "
            "adjacent trainable RY, i.e. inert under --fixed-f). 'Z' or 'X' make "
            "the fixed f non-absorbable, so it acts as a genuine spectral "
            "hyperparameter. Use 'Z' together with --fixed-f."
        ),
    )
    parser.add_argument(
        "--trainable-block",
        type=str,
        default="yzy",
        choices=["yzy"],
        help=(
            "Trainable single-qubit block used in every variational layer of BOTH "
            "models. 'yzy' = universal RY.RZ.RY block (3 params/qubit) whose RY boundaries "
            "keep a fixed RZ(f) encoding non-absorbable; this is the enforced default."
        ),
    )
    parser.add_argument(
        "--init-scale-mode",
        type=str,
        default="system",
        choices=["system", "legacy"],
        help=(
            "Initialization policy: 'system' uses split quantum/readout scales with "
            "a mild small-system quantum boost; 'legacy' keeps the original single scale."
        ),
    )
    parser.add_argument(
        "--quantum-init-scale",
        type=float,
        default=None,
        help="Optional override for quantum-parameter initialization scale.",
    )
    parser.add_argument(
        "--readout-init-scale",
        type=float,
        default=None,
        help="Optional override for readout-parameter initialization scale in linear mode.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay coefficient (applied inside loss; useful to curb overfitting in aligned model)",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory")
    return parser


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def required_dataset_size_for_fold_subsampling(
    n_folds: int,
    train_size: Optional[int],
    val_size: Optional[int],
    safety_margin: float = 0.05,
    minimum_padding: int = 20,
) -> int:
    """Compute a practical dataset size so each CV fold can be subsampled safely.

    For K-fold CV, each fold has about N/K validation samples and N*(K-1)/K
    training samples. This helper inverts those relations and adds a small
    padding to reduce failures from integer rounding and class stratification.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    needed_for_val = 0
    if val_size is not None and val_size > 0:
        needed_for_val = int(val_size) * int(n_folds)

    needed_for_train = 0
    if train_size is not None and train_size > 0:
        needed_for_train = int(math.ceil(train_size * n_folds / (n_folds - 1)))

    base = max(needed_for_train, needed_for_val)
    if base <= 0:
        return 0

    padding = max(int(math.ceil(base * safety_margin)), int(minimum_padding))
    return base + padding


def make_experiment_dir(output_root: str, experiment_name: str, run_name: Optional[str] = None) -> Path:
    name = run_name or timestamp_slug()
    out_dir = Path(output_root) / experiment_name / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    return out_dir


def config_from_args(args: argparse.Namespace) -> PQCConfig:
    return PQCConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_x_layers=args.n_x_layers,
        n_f_layers=args.n_f_layers,
        n_epochs=args.n_epochs,
        n_folds=args.n_folds,
        random_state=args.seed,
        learning_rate=args.lr,
        init_scale_mode=args.init_scale_mode,
        quantum_init_scale=args.quantum_init_scale,
        readout_init_scale=args.readout_init_scale,
        weight_decay=args.weight_decay,
        patience=args.patience,
        eval_every=args.eval_every,
        show_progress=args.show_progress,
        catalyst_batch_size=args.catalyst_batch_size,
        readout_mode=args.readout_mode,
        readout_wires=args.readout_wires,
        f_encoding_axis=getattr(args, "f_encoding_axis", "Y"),
        trainable_block=getattr(args, "trainable_block", "yzy"),
    )


def compress_ordered_features(x: np.ndarray, target_dim: int) -> np.ndarray:
    """Deterministic 1D compression that preserves feature order better than random projection."""
    if x.shape[1] == target_dim:
        return x.astype(np.float64)
    if target_dim <= 1:
        return x.mean(axis=1, keepdims=True).astype(np.float64)

    src = np.linspace(0.0, 1.0, x.shape[1], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_dim, dtype=np.float64)
    compressed = np.vstack([np.interp(dst, src, row) for row in x])
    return compressed.astype(np.float64)


def make_fixed_frequency_inputs(n_samples: int, n_qubits: int, jitter_scale: float = 0.0, seed: int = 0) -> np.ndarray:
    base = np.linspace(-np.pi, np.pi, n_qubits, dtype=np.float64)
    f = np.tile(base, (n_samples, 1))
    if jitter_scale > 0:
        rng = np.random.default_rng(seed)
        f = f + jitter_scale * rng.normal(size=f.shape)
    return f.astype(np.float64)


def make_random_frequency_inputs(n_samples: int, n_qubits: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits)).astype(np.float64)


def balanced_subsample(x: np.ndarray, y: np.ndarray, max_samples: Optional[int], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return balanced subset indices and corresponding arrays when max_samples is set."""
    indices = np.arange(len(y))
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return indices, x, y

    rng = np.random.default_rng(seed)
    unique = np.unique(y)
    per_class = max_samples // len(unique)
    chosen: List[np.ndarray] = []
    for cls in unique:
        cls_idx = indices[y == cls]
        take = min(per_class, len(cls_idx))
        chosen.append(rng.choice(cls_idx, size=take, replace=False))

    subset = np.concatenate(chosen)
    rng.shuffle(subset)
    return subset, x[subset], y[subset]


def linear_mnudft_dataset(
    n_samples: int,
    n_positions: int,
    n_frequencies: int,
    seed: int,
    noise_scale: float = 0.0,
    margin: float = 0.2,
    fixed_f: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    seed_seq = np.random.SeedSequence(seed)
    rng_task, rng_x, rng_f, rng_noise = [np.random.default_rng(s) for s in seed_seq.spawn(4)]

    # When f is treated as a dataset-level hyperparameter, draw the frequency
    # configuration ONCE and share it across every sample. The label then depends
    # on x alone, making the x-only (standard) and aligned models comparable.
    shared_f_pairs = None
    if fixed_f:
        f_vals0 = rng_f.uniform(low=-1.0, high=1.0, size=n_frequencies)
        shared_f_pairs = np.stack([f_vals0, f_vals0], axis=1)

    # ------------------------------------------------------------
    # Fixed 2D spatial geometry p_d ∈ R^2
    # ------------------------------------------------------------
    angles = np.linspace(0.0, 2.0 * np.pi, n_positions, endpoint=False)

    positions = np.stack(
        [np.cos(angles), np.sin(angles)],
        axis=1
    ).astype(np.float64)

    # ------------------------------------------------------------
    # Complex linear separator in spectral space
    # The input signal remains strictly real-valued.
    # Only the MNUDFT coefficients become complex.
    # ------------------------------------------------------------
    w = rng_task.normal(size=n_frequencies) + 1j * rng_task.normal(size=n_frequencies)

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    x_list = []
    f_list = []
    score_list = []

    # ------------------------------------------------------------
    # Rejection sampling loop
    # ------------------------------------------------------------
    while len(x_list) < n_samples:

        # --------------------------------------------------------
        # 1. Generate REAL-valued reflectivity signal
        # x(p_d) ∈ R
        # --------------------------------------------------------
        x = rng_x.normal(size=n_positions)

        # Normalize
        x /= np.max(np.abs(x)) + 1e-12

        # --------------------------------------------------------
        # 2. Generate nonuniform 2D frequency queries
        # f_k ∈ R^2
        # --------------------------------------------------------
        # f_pairs = rng.uniform(
        #     low=-1.0,
        #     high=1.0,
        #     size=(n_frequencies, 2)
        # )

        if shared_f_pairs is not None:
            f_pairs = shared_f_pairs
        else:
            f_vals = rng_f.uniform(low=-1.0, high=1.0, size=n_frequencies)
            f_pairs = np.stack([f_vals, f_vals], axis=1)

        # --------------------------------------------------------
        # 3. Evaluate complex MNUDFT
        #
        # X_k = Σ_d x_d exp(-i 2π p_d · f_k)
        #
        # The signal x is real-valued.
        # The Fourier response X is complex-valued.
        # --------------------------------------------------------
        X = np.zeros(n_frequencies, dtype=np.complex128)

        for k in range(n_frequencies):

            phase = 2.0 * np.pi * (positions @ f_pairs[k])

            kernel = np.exp(-1j * phase)

            X[k] = np.sum(x * kernel)

        # --------------------------------------------------------
        # 4. Linear separator in spectral space
        # --------------------------------------------------------
        score = np.real(np.sum(w * X))

        if noise_scale > 0.0:
            score += noise_scale * rng_noise.normal()

        # --------------------------------------------------------
        # 5. Margin filtering
        # --------------------------------------------------------
        if np.abs(score) >= margin:

            # x is already strictly real-valued
            x_list.append(x.astype(np.float64))

            # Flatten frequency pairs:
            # [fx0, fy0, fx1, fy1, ...]
            f_flat = np.zeros(2 * n_frequencies)

            f_flat[0::2] = f_pairs[:, 0]
            f_flat[1::2] = f_pairs[:, 1]

            f_list.append(f_flat.astype(np.float64))

            score_list.append(score)

    # ------------------------------------------------------------
    # Final arrays
    # ------------------------------------------------------------
    x_data = np.array(x_list, dtype=np.float64)

    f_data = np.array(f_list, dtype=np.float64)

    scores = np.array(score_list, dtype=np.float64)

    # ------------------------------------------------------------
    # Balanced binary labels
    # ------------------------------------------------------------
    threshold = np.median(scores)

    y = (scores > threshold).astype(np.int64)

    return x_data, f_data, y


def mnudft_teacher_dataset(
    n_samples: int,
    n_positions: int,
    n_frequencies: int,
    seed: int,
    noise_scale: float = 0.0,
    margin: float = 0.2,
    fixed_f: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """MNUDFT teacher with a KNOWN analytic decision function.

    Identical data-generating process to ``linear_mnudft_dataset`` (same ring
    geometry, same complex spectral separator, same interleaved f-encoding), but
    it ALSO returns a ``teacher`` spec so downstream analysis can compute the
    exact reference decision function and its spectrum:

        s*(x; f) = Re( sum_k w_k * sum_d x_d * exp(-2 pi i f_k . p_d) )

    with complex weights ``w``, 2-D positions ``p`` (unit ring), and — under
    ``fixed_f`` — the single shared query configuration ``f_true`` used to label
    every sample. The frequency vector fed to the circuit is the interleaved
    ``[fx0, fy0, fx1, fy1, ...]`` (length ``2 * n_frequencies``) with
    ``fx_k = fy_k`` (scalar-per-frequency), matching ``linear_mnudft_dataset``.

    Returns ``(x_data, f_data, y, teacher)`` where ``teacher`` holds
    ``w`` (complex), ``positions`` (n_positions, 2), ``f_true`` (or None),
    ``scores``, ``threshold`` and bookkeeping fields. Use it with the analytic
    helpers in :mod:`sar_qml.spectral` to obtain the teacher's Fourier
    fingerprint and to measure how well a trained model recovers it.
    """
    seed_seq = np.random.SeedSequence(seed)
    rng_task, rng_x, rng_f, rng_noise = [np.random.default_rng(s) for s in seed_seq.spawn(4)]

    shared_f_pairs = None
    f_true_flat = None
    if fixed_f:
        f_vals0 = rng_f.uniform(low=-1.0, high=1.0, size=n_frequencies)
        shared_f_pairs = np.stack([f_vals0, f_vals0], axis=1)
        f_true_flat = np.zeros(2 * n_frequencies)
        f_true_flat[0::2] = shared_f_pairs[:, 0]
        f_true_flat[1::2] = shared_f_pairs[:, 1]

    angles = np.linspace(0.0, 2.0 * np.pi, n_positions, endpoint=False)
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)

    w = rng_task.normal(size=n_frequencies) + 1j * rng_task.normal(size=n_frequencies)

    x_list, f_list, score_list = [], [], []
    while len(x_list) < n_samples:
        x = rng_x.normal(size=n_positions)
        x /= np.max(np.abs(x)) + 1e-12

        if shared_f_pairs is not None:
            f_pairs = shared_f_pairs
        else:
            f_vals = rng_f.uniform(low=-1.0, high=1.0, size=n_frequencies)
            f_pairs = np.stack([f_vals, f_vals], axis=1)

        X = np.zeros(n_frequencies, dtype=np.complex128)
        for k in range(n_frequencies):
            phase = 2.0 * np.pi * (positions @ f_pairs[k])
            X[k] = np.sum(x * np.exp(-1j * phase))

        score = np.real(np.sum(w * X))
        if noise_scale > 0.0:
            score += noise_scale * rng_noise.normal()

        if np.abs(score) >= margin:
            x_list.append(x.astype(np.float64))
            f_flat = np.zeros(2 * n_frequencies)
            f_flat[0::2] = f_pairs[:, 0]
            f_flat[1::2] = f_pairs[:, 1]
            f_list.append(f_flat.astype(np.float64))
            score_list.append(score)

    x_data = np.array(x_list, dtype=np.float64)
    f_data = np.array(f_list, dtype=np.float64)
    scores = np.array(score_list, dtype=np.float64)

    threshold = float(np.median(scores))
    y = (scores > threshold).astype(np.int64)

    teacher = {
        "w_real": np.real(w).astype(np.float64),
        "w_imag": np.imag(w).astype(np.float64),
        "positions": positions,
        "n_positions": int(n_positions),
        "n_frequencies": int(n_frequencies),
        "threshold": threshold,
        "scores": scores,
        "fixed_f": bool(fixed_f),
        "f_true_flat": f_true_flat,  # interleaved [fx0,fy0,...] or None
        "margin": float(margin),
        "noise_scale": float(noise_scale),
    }
    return x_data, f_data, y, teacher


def raw_sar_synthetic_dataset(
    n_samples: int,
    n_qubits: int,
    seed: int,
    noise_scale: float = 0.03,
    dominance_ratio: float = 0.55,
    fixed_f: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic raw-SAR task with labels defined after image reconstruction.

    The raw input x stores real/imag parts of complex measurements at K=n_qubits/2
    non-uniform query locations. The paired input f stores the flattened 2D query
    coordinates [fx0, fy0, fx1, fy1, ...]. Labels are extracted from a simple
    inverse reconstruction by counting dominant reflectors in the reconstructed image.
    """
    if n_qubits < 2 or (n_qubits % 2) != 0:
        raise ValueError("n_qubits must be an even integer >= 2")

    seed_seq = np.random.SeedSequence(seed)
    rng_scene, rng_query, rng_noise, rng_perm = [np.random.default_rng(s) for s in seed_seq.spawn(4)]
    n_measurements = n_qubits // 2
    n_scene_positions = min(3, n_measurements)

    if n_scene_positions == 1:
        counts_by_label = {0: np.array([0], dtype=np.int64), 1: np.array([1], dtype=np.int64)}
    elif n_scene_positions == 2:
        counts_by_label = {0: np.array([1], dtype=np.int64), 1: np.array([2], dtype=np.int64)}
    else:
        counts_by_label = {0: np.array([2], dtype=np.int64), 1: np.array([3], dtype=np.int64)}

    label_by_count = {
        int(count): int(label)
        for label, counts in counts_by_label.items()
        for count in counts
    }

    if n_scene_positions == 1:
        scene_positions = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        axis = np.linspace(-0.75, 0.75, n_scene_positions, dtype=np.float64)
        scene_positions = np.stack([axis, 0.35 * np.sin(np.pi * axis)], axis=1).astype(np.float64)

    x_list: List[np.ndarray] = []
    f_list: List[np.ndarray] = []
    y_list: List[int] = []

    if n_measurements == 1:
        base_query_pairs = np.array([[0.35, -0.25]], dtype=np.float64)
    else:
        base_axis = np.linspace(-0.8, 0.8, n_measurements, dtype=np.float64)
        base_query_pairs = np.stack([base_axis, base_axis[::-1]], axis=1).astype(np.float64)

    while len(x_list) < n_samples:
        desired_label = len(x_list) % 2
        active_count = int(rng_scene.choice(counts_by_label[desired_label]))

        reflectivity = np.zeros(n_scene_positions, dtype=np.float64)
        if active_count > 0:
            support = rng_scene.choice(n_scene_positions, size=active_count, replace=False)
            reflectivity[support] = rng_scene.uniform(0.8, 1.2, size=active_count)

        if fixed_f:
            # Fixed acquisition geometry: the same query coordinates for every
            # scene, so f is a dataset-level hyperparameter and the label depends
            # only on the reflectivity encoded in x.
            query_pairs = base_query_pairs.astype(np.float64)
        else:
            query_pairs = np.clip(
                base_query_pairs + 0.08 * rng_query.normal(size=(n_measurements, 2)),
                -1.0,
                1.0,
            ).astype(np.float64)
        phases = 2.0 * np.pi * (scene_positions @ query_pairs.T)
        sensing_matrix = np.exp(-1j * phases).T
        if np.linalg.cond(sensing_matrix) > 20.0:
            continue

        measurements = sensing_matrix @ reflectivity

        if noise_scale > 0.0 and active_count > 0:
            noise = rng_noise.normal(size=n_measurements) + 1j * rng_noise.normal(size=n_measurements)
            measurements = measurements + noise_scale * noise / np.sqrt(2.0)

        reconstruction = np.abs(np.linalg.pinv(sensing_matrix) @ measurements)
        peak = float(np.max(reconstruction))
        if peak <= 1e-8:
            dominant_count = 0
        else:
            dominant_count = int(np.sum(reconstruction >= dominance_ratio * peak))
        label = label_by_count.get(dominant_count)
        if label is None:
            continue
        if label != desired_label:
            continue

        x_features = np.concatenate([measurements.real, measurements.imag]).astype(np.float64)
        f_features = np.empty(n_qubits, dtype=np.float64)
        f_features[0::2] = query_pairs[:, 0]
        f_features[1::2] = query_pairs[:, 1]

        x_list.append(x_features)
        f_list.append(f_features)
        y_list.append(label)

    x_data = np.asarray(x_list, dtype=np.float64)
    x_data = (x_data - x_data.mean(axis=0)) / (x_data.std(axis=0) + 1e-8)

    f_data = np.asarray(f_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)

    order = rng_perm.permutation(n_samples)
    return x_data[order], f_data[order], y[order]

def simulate_reflector_dataset(
    n_samples: int,
    n_qubits: int,
    class_a_targets: int,
    class_b_targets: int,
    noise_std: float,
    seed: int,
    freq_jitter: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labels from a real-projected spectral score that depends on both x and f."""
    rng = np.random.default_rng(seed)
    x_data = np.zeros((n_samples, n_qubits), dtype=np.float64)
    scores = np.zeros(n_samples, dtype=np.float64)

    max_terms = max(class_a_targets, class_b_targets)
    support_vectors = rng.uniform(-1.0, 1.0, size=(max_terms, n_qubits))
    coefficient_vectors = rng.normal(size=(max_terms, n_qubits)) / np.sqrt(n_qubits)
    phase_offsets = rng.uniform(0.0, 2.0 * np.pi, size=max_terms)
    f_data = make_random_frequency_inputs(n_samples, n_qubits, seed=seed + 1)

    freqs = np.linspace(0.5, 2.5, n_qubits)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, max(class_a_targets, class_b_targets)))

    for i in range(n_samples):
        # Alternate between two structural regimes so robustness studies still vary task complexity,
        # but define the label through a score that genuinely depends on both x and f.
        n_targets = class_b_targets if (i % 2 == 1) else class_a_targets

        amps = rng.uniform(0.7, 1.3, size=n_targets)
        signal = np.zeros(n_qubits, dtype=np.float64)
        for t in range(n_targets):
            signal += amps[t] * np.cos(freqs * (t + 1) + phases[i, t])

        signal += noise_std * rng.normal(size=n_qubits)
        x_data[i] = signal

        score = 0.0
        for t in range(n_targets):
            coeff = np.tanh(signal @ coefficient_vectors[t])
            score += coeff * np.cos(f_data[i] @ support_vectors[t] + phase_offsets[t])
        scores[i] = score

    x_data = (x_data - x_data.mean(axis=0)) / (x_data.std(axis=0) + 1e-8)
    if freq_jitter > 0:
        f_data = f_data + freq_jitter * rng.normal(size=f_data.shape)
    threshold = float(np.median(scores))
    y = (scores > threshold).astype(np.int64)
    return x_data.astype(np.float64), f_data.astype(np.float64), y


# def teacher_student_dataset(
#     n_samples: int,
#     config: PQCConfig,
#     seed: int,
#     threshold: float = 0.5,
#     teacher_weight_init_scale: float = 1.0,
#     use_soft_labels: bool = True,
#     fixed_f: bool = False,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Generate labels from a fixed aligned teacher model with SAR-like x/f semantics.

#     x_data emulates flattened reflectivity descriptors (derived from synthetic complex returns),
#     while f_data is sampled as real-valued probing-frequency vectors.

#     teacher_weight_init_scale should be significantly larger than the student's
#     weight_init_scale (default 0.01) so the teacher encodes a non-trivial function
#     that the student must genuinely learn rather than accidentally matching at
#     initialisation.

#     use_soft_labels=True returns raw teacher probabilities as targets. BCE handles
#     soft targets (y in [0,1]) naturally and removes label noise near the decision
#     boundary. Set to False to recover hard 0/1 labels via threshold.
#     """
#     seed_seq = np.random.SeedSequence(seed)
#     rng_x, rng_f = [np.random.default_rng(s) for s in seed_seq.spawn(2)]

#     # Build structured reflectivity-style features from synthetic complex returns,
#     # then compress to the model input dimension.
#     latent_dim = max(4 * config.n_qubits, 16)
#     refl_real = rng_x.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
#     refl_imag = rng_x.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
#     magnitude = np.sqrt(refl_real**2 + refl_imag**2)
#     phase = np.arctan2(refl_imag, refl_real)
#     reflectivity_features = np.concatenate([magnitude, phase], axis=1)
#     x_data = compress_ordered_features(reflectivity_features, config.n_qubits)
#     x_data = (x_data - x_data.mean(axis=0)) / (x_data.std(axis=0) + 1e-8)

#     # Real probing frequencies in R: monotone grids with sample-specific bandwidth/offset.
#     if config.n_qubits == 1:
#         base_axis = np.array([0.0], dtype=np.float64)
#     else:
#         base_axis = np.linspace(0.0, 1.0, config.n_qubits, dtype=np.float64)

#     if fixed_f:
#         # f as a dataset-level hyperparameter: one probing-frequency vector shared
#         # by all samples, so the teacher's label depends only on x.
#         start_freq = rng_f.uniform(0.15, 0.9, size=(1, 1))
#         bandwidth = rng_f.uniform(0.2, 1.4, size=(1, 1))
#         f_row = 2.0 * np.pi * (start_freq + bandwidth * base_axis[None, :])
#         f_data = np.tile(f_row, (n_samples, 1)).astype(np.float64)
#     else:
#         start_freq = rng_f.uniform(0.15, 0.9, size=(n_samples, 1))
#         bandwidth = rng_f.uniform(0.2, 1.4, size=(n_samples, 1))
#         f_data = 2.0 * np.pi * (start_freq + bandwidth * base_axis[None, :])
#         f_data += 0.03 * rng_f.normal(size=f_data.shape)
#         f_data = f_data.astype(np.float64)

#     # Teacher uses a dedicated seed offset so its weights are independent of the
#     # student's initialisation seed, and a larger init scale so it encodes a
#     # genuinely non-trivial function.
#     teacher_config = PQCConfig(
#         **{k: v for k, v in config.__dict__.items() if k != "weight_init_scale"},
#         weight_init_scale=teacher_weight_init_scale,
#     )
#     teacher = PQCClassifier(teacher_config, use_double_reuploading=True)
#     teacher_weights = teacher.init_weights()
#     probs = np.asarray(teacher.predict((x_data, f_data), teacher_weights), dtype=np.float64)

#     if use_soft_labels:
#         # Soft labels: raw probabilities in [0, 1].  BCE handles these naturally and
#         # avoids injecting label noise near the teacher's decision boundary.
#         y = probs.astype(np.float64)
#     else:
#         y = (probs > threshold).astype(np.int64)
#     return x_data, f_data, y

def teacher_student_dataset(
    n_samples: int,
    config: PQCConfig,
    seed: int,
    threshold: float = 0.5,
    teacher_weight_init_scale: float = 1.0,
    use_soft_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labels from a fixed aligned teacher model with simplified x/f semantics.

    x_data is sampled directly from N(0, 1).
    f_data is always a dataset-level fixed vector drawn with the provided seed,
    non-uniformly distributed in [-1, 1], and shared across all samples.

    teacher_weight_init_scale should be significantly larger than the student's
    weight_init_scale (default 0.01) so the teacher encodes a non-trivial function
    that the student must genuinely learn rather than accidentally matching at
    initialisation.

    use_soft_labels=True returns raw teacher probabilities as targets. BCE handles
    soft targets (y in [0,1]) naturally and removes label noise near the decision
    boundary. Set to False to recover hard 0/1 labels via threshold.
    """
    seed_seq = np.random.SeedSequence(seed)
    rng_x, rng_f = [np.random.default_rng(s) for s in seed_seq.spawn(2)]

    x_data = rng_x.normal(loc=0.0, scale=1.0, size=(n_samples, config.n_qubits)).astype(np.float64)

    # Fixed, seeded and non-uniform in [-1, 1]. tanh(N(0,1)) gives a smooth,
    # center-heavy non-uniform distribution with strict bounds.
    f_row = np.tanh(rng_f.normal(loc=0.0, scale=1.0, size=(1, config.n_qubits))).astype(np.float64)
    f_data = np.tile(f_row, (n_samples, 1))

    # Teacher uses a dedicated seed offset so its weights are independent of the
    # student's initialisation seed, and a larger init scale so it encodes a
    # genuinely non-trivial function.
    teacher_config = PQCConfig(
        **{k: v for k, v in config.__dict__.items() if k != "weight_init_scale"},
        weight_init_scale=teacher_weight_init_scale,
    )
    teacher = PQCClassifier(teacher_config, use_double_reuploading=True)
    teacher_weights = teacher.init_weights()
    probs = np.asarray(teacher.predict((x_data, f_data), teacher_weights), dtype=np.float64)

    if use_soft_labels:
        # Soft labels: raw probabilities in [0, 1].  BCE handles these naturally and
        # avoids injecting label noise near the teacher's decision boundary.
        y = probs.astype(np.float64)
    else:
        y = (probs > threshold).astype(np.int64)
    return x_data, f_data, y


def load_sar_csv(
    csv_path: Path,
    prefix: str,
    n_qubits: int,
    label_threshold: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    sample_ids = np.array([int(r["sample_id"]) for r in rows], dtype=np.int64)
    num_targets = np.array([int(r["num_targets"]) for r in rows], dtype=np.int64)

    real_cols = [k for k in rows[0].keys() if k.startswith(f"{prefix}_real_")]
    imag_cols = [k for k in rows[0].keys() if k.startswith(f"{prefix}_imag_")]
    real_cols.sort(key=lambda c: int(c.split("_")[-1]))
    imag_cols.sort(key=lambda c: int(c.split("_")[-1]))

    real = np.array([[float(r[c]) for c in real_cols] for r in rows], dtype=np.float64)
    imag = np.array([[float(r[c]) for c in imag_cols] for r in rows], dtype=np.float64)
    features = np.concatenate([real, imag], axis=1)
    x_data = compress_ordered_features(features, n_qubits)
    x_data = (x_data - x_data.mean(axis=0)) / (x_data.std(axis=0) + 1e-8)

    y = (num_targets >= label_threshold).astype(np.int64)
    # Use a fixed, non-uniform evaluation geometry shared across samples.
    f_data = make_fixed_frequency_inputs(len(rows), n_qubits, jitter_scale=0.15, seed=seed + 3)
    return x_data, f_data, y, sample_ids


def _train_single_fold(
    config: PQCConfig,
    use_double_reuploading: bool,
    train_payload: Any,
    y_train: np.ndarray,
    val_payload: Any,
    y_val: np.ndarray,
    fold_id: int,
) -> Tuple[np.ndarray, Dict[str, List[float]], float]:
    model = PQCClassifier(config, use_double_reuploading=use_double_reuploading)
    best_weights, history, best_val_acc = model.train_fold(
        train_payload,
        np.asarray(y_train),
        val_payload,
        np.asarray(y_val),
        fold_id,
    )
    return np.asarray(best_weights), history, float(best_val_acc)


def _subsample_indices(
    indices: np.ndarray,
    y_strat: np.ndarray,
    target_size: Optional[int],
    seed: int,
    split_name: str,
) -> np.ndarray:
    if target_size is None or target_size <= 0:
        return np.asarray(indices)

    indices = np.asarray(indices)
    if target_size > len(indices):
        raise ValueError(
            f"Requested {split_name}_size={target_size}, but fold has only {len(indices)} "
            f"{split_name} samples. Increase --n-samples or reduce --n-folds/{split_name}_size."
        )
    if target_size == len(indices):
        return indices

    labels = np.asarray(y_strat)[indices]
    unique = np.unique(labels)

    # Use stratified subsampling whenever possible to preserve class balance.
    if len(unique) >= 2 and target_size >= len(unique):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=target_size, random_state=seed)
        rel_idx, _ = next(splitter.split(np.zeros(len(indices)), labels))
        return indices[np.asarray(rel_idx)]

    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(indices, size=target_size, replace=False))


def run_crossval(
    config: PQCConfig,
    x_data: np.ndarray,
    y: np.ndarray,
    f_data: Optional[np.ndarray] = None,
    use_double_reuploading: bool = False,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
) -> Dict[str, Any]:
    # StratifiedKFold requires integer class labels; threshold soft targets for splitting only.
    y_strat = (np.asarray(y) > 0.5).astype(int)
    kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    fold_metrics: List[Dict[str, Any]] = []
    fold_weights: List[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data, y_strat), start=1):
        train_idx = _subsample_indices(
            train_idx,
            y_strat=y_strat,
            target_size=train_size,
            seed=config.random_state + 1000 * fold + 11,
            split_name="train",
        )
        val_idx = _subsample_indices(
            val_idx,
            y_strat=y_strat,
            target_size=val_size,
            seed=config.random_state + 1000 * fold + 29,
            split_name="val",
        )

        y_train = y[train_idx]
        y_val = y[val_idx]

        if use_double_reuploading:
            assert f_data is not None
            train_payload = (np.asarray(x_data[train_idx]), np.asarray(f_data[train_idx]))
            val_payload = (np.asarray(x_data[val_idx]), np.asarray(f_data[val_idx]))
        else:
            if f_data is None:
                x_single = x_data
            
            train_payload = np.asarray(x_single[train_idx])
            val_payload = np.asarray(x_single[val_idx])

        best_w, history, best_val_acc = _train_single_fold(
            config,
            use_double_reuploading,
            train_payload,
            y_train,
            val_payload,
            y_val,
            fold,
        )

        fold_weights.append(best_w)
        fold_metrics.append(
            {
                "fold": fold,
                "best_val_acc": best_val_acc,
                "history": history,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
        )

    best_accs = [m["best_val_acc"] for m in fold_metrics]
    return {
        "mean_best_val_acc": float(np.mean(best_accs)),
        "std_best_val_acc": float(np.std(best_accs)),
        "fold_metrics": fold_metrics,
        "fold_weights": fold_weights,
    }


def run_crossval_single_fold(
    config: PQCConfig,
    x_data: np.ndarray,
    y: np.ndarray,
    fold_index: int,
    f_data: Optional[np.ndarray] = None,
    use_double_reuploading: bool = False,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run exactly one StratifiedKFold split (1-based fold_index)."""
    if fold_index < 1 or fold_index > config.n_folds:
        raise ValueError(f"fold_index must be in [1, {config.n_folds}], got {fold_index}")

    y_strat = (np.asarray(y) > 0.5).astype(int)
    kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    splits = list(kf.split(x_data, y_strat))
    train_idx, val_idx = splits[fold_index - 1]
    train_idx = _subsample_indices(
        train_idx,
        y_strat=y_strat,
        target_size=train_size,
        seed=config.random_state + 1000 * fold_index + 11,
        split_name="train",
    )
    val_idx = _subsample_indices(
        val_idx,
        y_strat=y_strat,
        target_size=val_size,
        seed=config.random_state + 1000 * fold_index + 29,
        split_name="val",
    )

    y_train = y[train_idx]
    y_val = y[val_idx]

    if use_double_reuploading:
        assert f_data is not None
        train_payload = (np.asarray(x_data[train_idx]), np.asarray(f_data[train_idx]))
        val_payload = (np.asarray(x_data[val_idx]), np.asarray(f_data[val_idx]))
    else:
        train_payload = np.asarray(x_data[train_idx])
        val_payload = np.asarray(x_data[val_idx])

    best_w, history, best_val_acc = _train_single_fold(
        config,
        use_double_reuploading,
        train_payload,
        y_train,
        val_payload,
        y_val,
        fold_index,
    )

    fold_metric = {
        "fold": fold_index,
        "best_val_acc": best_val_acc,
        "history": history,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }

    return {
        "mean_best_val_acc": float(best_val_acc),
        "std_best_val_acc": 0.0,
        "fold_metrics": [fold_metric],
        "fold_weights": [best_w],
        "selected_fold": int(fold_index),
    }


def run_paired_models(
    config: PQCConfig,
    x_data: np.ndarray,
    f_data: np.ndarray,
    y: np.ndarray,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
) -> Dict[str, Any]:
    # The aligned circuit interleaves x-blocks and f-blocks, so its total number of
    # variational sub-layers is n_x_layers + n_f_layers.  To keep the parameter count
    # equal the standard (single-branch) circuit must have the same total depth.
    from dataclasses import replace as dc_replace
    x_depth = config.n_x_layers if config.n_x_layers is not None else config.n_layers
    f_depth = config.n_f_layers if config.n_f_layers is not None else config.n_layers
    standard_n_layers = x_depth + f_depth
    standard_config = dc_replace(config, n_layers=standard_n_layers, n_x_layers=None, n_f_layers=None)

    aligned = run_crossval(
        config=config,
        x_data=x_data,
        f_data=f_data,
        y=y,
        use_double_reuploading=True,
        train_size=train_size,
        val_size=val_size,
    )
    # Fair baseline for alignment tests: standard model gets only x input,
    # with total depth matched to aligned (x_depth + f_depth).
    standard = run_crossval(
        config=standard_config,
        x_data=x_data,
        f_data=None,
        y=y,
        use_double_reuploading=False,
        train_size=train_size,
        val_size=val_size,
    )
    return {
        "aligned": aligned,
        "standard": standard,
        "_depth_info": {
            "aligned_x_depth": x_depth,
            "aligned_f_depth": f_depth,
            "standard_n_layers": standard_n_layers,
            "standard_input": "x_only",
        },
    }


def run_paired_models_single_fold(
    config: PQCConfig,
    x_data: np.ndarray,
    f_data: np.ndarray,
    y: np.ndarray,
    fold_index: int,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run aligned/standard comparison on one fixed fold only."""
    from dataclasses import replace as dc_replace

    x_depth = config.n_x_layers if config.n_x_layers is not None else config.n_layers
    f_depth = config.n_f_layers if config.n_f_layers is not None else config.n_layers
    standard_n_layers = x_depth + f_depth
    standard_config = dc_replace(config, n_layers=standard_n_layers, n_x_layers=None, n_f_layers=None)

    aligned = run_crossval_single_fold(
        config=config,
        x_data=x_data,
        f_data=f_data,
        y=y,
        fold_index=fold_index,
        use_double_reuploading=True,
        train_size=train_size,
        val_size=val_size,
    )
    standard = run_crossval_single_fold(
        config=standard_config,
        x_data=x_data,
        f_data=None,
        y=y,
        fold_index=fold_index,
        use_double_reuploading=False,
        train_size=train_size,
        val_size=val_size,
    )
    return {
        "aligned": aligned,
        "standard": standard,
        "_depth_info": {
            "aligned_x_depth": x_depth,
            "aligned_f_depth": f_depth,
            "standard_n_layers": standard_n_layers,
            "standard_input": "x_only",
            "selected_fold": int(fold_index),
        },
    }


def _save_nested_weights(weights_dir: Path, node: Any, prefix: str) -> Any:
    """Recursively save fold weights and return JSON-serializable payload without arrays."""
    if isinstance(node, dict):
        if "fold_weights" in node:
            cleaned = dict(node)
            fold_weights = cleaned.pop("fold_weights", [])
            for fold_idx, w in enumerate(fold_weights, start=1):
                np.save(weights_dir / f"{prefix}_fold{fold_idx}.npy", np.asarray(w))

            out: Dict[str, Any] = {}
            for k, v in cleaned.items():
                out[k] = _save_nested_weights(weights_dir, v, f"{prefix}_{k}")
            return out

        out = {}
        for k, v in node.items():
            child_prefix = f"{prefix}_{k}" if prefix else k
            out[k] = _save_nested_weights(weights_dir, v, child_prefix)
        return out

    if isinstance(node, list):
        return [_save_nested_weights(weights_dir, v, f"{prefix}_{i}") for i, v in enumerate(node)]

    if isinstance(node, np.ndarray):
        return node.tolist()

    return node


def save_results_bundle(
    out_dir: Path,
    experiment_name: str,
    config: PQCConfig,
    payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    weights_dir = out_dir / "weights"
    serializable_payload = _save_nested_weights(weights_dir, payload, prefix="")

    results = {
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "config": asdict(config),
        "metadata": metadata or {},
        "results": serializable_payload,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)


def print_summary(payload: Dict[str, Any], title: str) -> None:
    print(f"\n=== {title} ===")
    if "aligned" in payload and "standard" in payload:
        print(
            f"aligned: {payload['aligned']['mean_best_val_acc']:.4f} +/- "
            f"{payload['aligned']['std_best_val_acc']:.4f}"
        )
        print(
            f"standard: {payload['standard']['mean_best_val_acc']:.4f} +/- "
            f"{payload['standard']['std_best_val_acc']:.4f}"
        )
    else:
        for k, v in payload.items():
            if isinstance(v, dict) and "mean_best_val_acc" in v:
                print(f"{k}: {v['mean_best_val_acc']:.4f} +/- {v['std_best_val_acc']:.4f}")
