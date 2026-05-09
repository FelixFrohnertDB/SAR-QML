from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pennylane import numpy as pp
from sklearn.model_selection import StratifiedKFold

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
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
        weight_decay=args.weight_decay,
        patience=args.patience,
        eval_every=args.eval_every,
        show_progress=args.show_progress,
    )


# def reduce_features(x: np.ndarray, target_dim: int, seed: int = 0) -> np.ndarray:
#     if x.shape[1] == target_dim:
#         return x.astype(np.float64)
#     if x.shape[1] < target_dim:
#         pad = np.zeros((x.shape[0], target_dim - x.shape[1]), dtype=np.float64)
#         return np.concatenate([x, pad], axis=1)

#     rng = np.random.default_rng(seed)
#     proj = rng.normal(size=(x.shape[1], target_dim)) / np.sqrt(x.shape[1])
#     return (x @ proj).astype(np.float64)


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



# def linear_mnudft_dataset(
#     n_samples: int,
#     n_qubits: int,
#     seed: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Generate a dataset that is linearly separable after MNUDFT processing.

#     Encoding budget:
#     - n_qubits = 2 * n_positions
#     - x packs 2 strictly real numbers per position (2D value channels)
#     - f packs 2 real numbers per frequency query (2D frequency vectors)
#     - number of frequencies K = n_positions

#     Labels are generated by a linear rule in transformed features:
#         score = sum_k alpha_k * X_k
#     where X_k is a real-valued MNUDFT-style projection at frequency f_k.
#     """
#     if n_qubits < 2 or (n_qubits % 2) != 0:
#         raise ValueError("linear_mnudft_dataset requires an even n_qubits >= 2")

#     rng = np.random.default_rng(seed)
#     n_positions = n_qubits // 2
#     K = n_positions

#     # Deterministic 2D geometry for the position coordinates p_d.
#     # Coordinates lie on a unit circle to avoid collinearity artifacts.
#     angles = np.linspace(0.0, 2.0 * np.pi, n_positions, endpoint=False)
#     positions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)

#     # Linear separator in transformed feature space.
#     alpha_k = rng.normal(size=K)

#     x_data = np.zeros((n_samples, n_qubits), dtype=np.float64)
#     f_data = np.zeros((n_samples, n_qubits), dtype=np.float64)
#     scores = np.zeros(n_samples, dtype=np.float64)

#     for i in range(n_samples):
#         # Fill all X dimensions: for each position d, store two real channels (x0_d, x1_d).
#         x0 = rng.normal(size=n_positions)
#         x1 = rng.normal(size=n_positions)
#         x_data[i, 0::2] = x0
#         x_data[i, 1::2] = x1

#         # Fill all F dimensions: for each query k, store 2D frequency vector (f_x, f_y).
#         f_pairs = rng.uniform(-1.0, 1.0, size=(K, 2))
#         f_data[i, 0::2] = f_pairs[:, 0]
#         f_data[i, 1::2] = f_pairs[:, 1]

#         # Real-valued MNUDFT-style feature extraction at K query frequencies:
#         # X_k = sum_d [x0_d * cos(phi_kd) + x1_d * sin(phi_kd)],
#         # with phi_kd = 2*pi * <f_k, p_d>.
#         xk = np.zeros(K, dtype=np.float64)
#         for k in range(K):
#             phase = 2.0 * np.pi * (positions @ f_pairs[k])
#             xk[k] = np.sum(x0 * np.cos(phase) + x1 * np.sin(phase))

#         scores[i] = float(np.sum(alpha_k * xk))

#     x_data = (x_data - x_data.mean(axis=0)) / (x_data.std(axis=0) + 1e-8)
#     threshold = float(np.median(scores))
#     y = (scores > threshold).astype(np.int64)
#     return x_data.astype(np.float64), f_data.astype(np.float64), y

import numpy as np
from typing import Tuple


import numpy as np
from typing import Tuple

def linear_mnudft_dataset(
    n_samples: int,
    n_positions: int,
    n_frequencies: int,
    seed: int,
    noise_scale: float = 0.0,
    margin: float = 0.2, # Added margin parameter
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)

    # --- Fixed spatial geometry p_d in R^2 ---
    angles = np.linspace(0.0, 2.0 * np.pi, n_positions, endpoint=False)
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)

    # --- Complex separator in spectral space ---
    w = rng.normal(size=n_frequencies) + 1j * rng.normal(size=n_frequencies)

    # --- Storage for valid samples ---
    x_list = []
    f_list = []
    score_list = []

    # --- Rejection Sampling Loop ---
    # We continue until we have exactly n_samples that pass the margin filter
    while len(x_list) < n_samples:
        # 1. Generate candidate real-valued input channels
        x0 = rng.normal(size=n_positions)
        x1 = rng.normal(size=n_positions)
        
        # Normalization
        x0 /= np.max(np.abs(x0)) + 1e-12
        x1 /= np.max(np.abs(x1)) + 1e-12

        # 2. Generate non-uniform frequency queries
        f_pairs = rng.uniform(low=-1.0, high=1.0, size=(n_frequencies, 2))

        # 3. Complex MNUDFT evaluation
        x_complex = x0 + 1j * x1
        X = np.zeros(n_frequencies, dtype=np.complex128)
        for k in range(n_frequencies):
            phase = 2.0 * np.pi * (positions @ f_pairs[k])
            kernel = np.exp(-1j * phase)
            X[k] = np.sum(x_complex * kernel)

        # 4. Calculate Score
        score = np.real(np.sum(w * X))
        if noise_scale > 0.0:
            score += noise_scale * rng.normal()

        # 5. MARGIN FILTER
        # Only keep the point if it is outside the decision boundary "radius"
        # We assume the boundary is at 0 (the expected mean of the scores)
        if np.abs(score) >= margin:
            # Flatten x and f for storage
            x_flat = np.zeros(2 * n_positions)
            x_flat[0::2] = x0
            x_flat[1::2] = x1
            
            f_flat = np.zeros(2 * n_frequencies)
            f_flat[0::2] = f_pairs[:, 0]
            f_flat[1::2] = f_pairs[:, 1]

            x_list.append(x_flat)
            f_list.append(f_flat)
            score_list.append(score)

    # --- Finalize Arrays ---
    x_data = np.array(x_list)
    f_data = np.array(f_list)
    scores = np.array(score_list)

    # Balanced binary labels based on the scores
    # Since we already filtered around 0, the median will still be near 0
    threshold = np.median(scores)
    y = (scores > threshold).astype(np.int64)

    return x_data, f_data, y

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


def teacher_student_dataset(
    n_samples: int,
    config: PQCConfig,
    seed: int,
    threshold: float = 0.5,
    teacher_weight_init_scale: float = 1.0,
    use_soft_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labels from a fixed aligned teacher model with SAR-like x/f semantics.

    x_data emulates flattened reflectivity descriptors (derived from synthetic complex returns),
    while f_data is sampled as real-valued probing-frequency vectors.

    teacher_weight_init_scale should be significantly larger than the student's
    weight_init_scale (default 0.01) so the teacher encodes a non-trivial function
    that the student must genuinely learn rather than accidentally matching at
    initialisation.

    use_soft_labels=True returns raw teacher probabilities as targets. BCE handles
    soft targets (y in [0,1]) naturally and removes label noise near the decision
    boundary. Set to False to recover hard 0/1 labels via threshold.
    """
    rng = np.random.default_rng(seed)

    # Build structured reflectivity-style features from synthetic complex returns,
    # then compress to the model input dimension.
    latent_dim = max(4 * config.n_qubits, 16)
    refl_real = rng.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
    refl_imag = rng.normal(loc=0.0, scale=1.0, size=(n_samples, latent_dim))
    magnitude = np.sqrt(refl_real**2 + refl_imag**2)
    phase = np.arctan2(refl_imag, refl_real)
    reflectivity_features = np.concatenate([magnitude, phase], axis=1)
    x_data = compress_ordered_features(reflectivity_features, config.n_qubits)
    x_data = (x_data - x_data.mean(axis=0)) / (x_data.std(axis=0) + 1e-8)

    # Real probing frequencies in R: monotone grids with sample-specific bandwidth/offset.
    if config.n_qubits == 1:
        base_axis = np.array([0.0], dtype=np.float64)
    else:
        base_axis = np.linspace(0.0, 1.0, config.n_qubits, dtype=np.float64)

    start_freq = rng.uniform(0.15, 0.9, size=(n_samples, 1))
    bandwidth = rng.uniform(0.2, 1.4, size=(n_samples, 1))
    f_data = 2.0 * np.pi * (start_freq + bandwidth * base_axis[None, :])
    f_data += 0.03 * rng.normal(size=f_data.shape)
    f_data = f_data.astype(np.float64)

    # Teacher uses a dedicated seed offset so its weights are independent of the
    # student's initialisation seed, and a larger init scale so it encodes a
    # genuinely non-trivial function.
    teacher_config = PQCConfig(
        **{k: v for k, v in config.__dict__.items() if k != "weight_init_scale"},
        weight_init_scale=teacher_weight_init_scale,
    )
    pp.random.seed(seed + 99999)
    teacher = PQCClassifier(teacher_config, use_double_reuploading=True)
    teacher_weights = teacher.init_weights()
    probs = np.asarray(teacher.predict((pp.array(x_data), pp.array(f_data)), teacher_weights), dtype=np.float64)

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
        pp.array(y_train),
        val_payload,
        pp.array(y_val),
        fold_id,
    )
    return np.asarray(best_weights), history, float(best_val_acc)


def run_crossval(
    config: PQCConfig,
    x_data: np.ndarray,
    y: np.ndarray,
    f_data: Optional[np.ndarray] = None,
    use_double_reuploading: bool = False,
) -> Dict[str, Any]:
    # StratifiedKFold requires integer class labels; threshold soft targets for splitting only.
    y_strat = (np.asarray(y) > 0.5).astype(int)
    kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    fold_metrics: List[Dict[str, Any]] = []
    fold_weights: List[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data, y_strat), start=1):
        y_train = y[train_idx]
        y_val = y[val_idx]

        if use_double_reuploading:
            assert f_data is not None
            train_payload = (pp.array(x_data[train_idx]), pp.array(f_data[train_idx]))
            val_payload = (pp.array(x_data[val_idx]), pp.array(f_data[val_idx]))
        else:
            if f_data is None:
                x_single = x_data
            
            train_payload = pp.array(x_single[train_idx])
            val_payload = pp.array(x_single[val_idx])

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
) -> Dict[str, Any]:
    """Run exactly one StratifiedKFold split (1-based fold_index)."""
    if fold_index < 1 or fold_index > config.n_folds:
        raise ValueError(f"fold_index must be in [1, {config.n_folds}], got {fold_index}")

    y_strat = (np.asarray(y) > 0.5).astype(int)
    kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    splits = list(kf.split(x_data, y_strat))
    train_idx, val_idx = splits[fold_index - 1]

    y_train = y[train_idx]
    y_val = y[val_idx]

    if use_double_reuploading:
        assert f_data is not None
        train_payload = (pp.array(x_data[train_idx]), pp.array(f_data[train_idx]))
        val_payload = (pp.array(x_data[val_idx]), pp.array(f_data[val_idx]))
    else:
        train_payload = pp.array(x_data[train_idx])
        val_payload = pp.array(x_data[val_idx])

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
    )
    # Fair baseline for alignment tests: standard model gets only x input,
    # with total depth matched to aligned (x_depth + f_depth).
    standard = run_crossval(
        config=standard_config,
        x_data=x_data,
        f_data=None,
        y=y,
        use_double_reuploading=False,
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
    )
    standard = run_crossval_single_fold(
        config=standard_config,
        x_data=x_data,
        f_data=None,
        y=y,
        fold_index=fold_index,
        use_double_reuploading=False,
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
