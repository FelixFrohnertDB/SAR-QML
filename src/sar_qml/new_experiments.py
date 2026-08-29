"""Rich experiment runners for the redesigned aligned-vs-standard study.

Three experiments, restructured as an existence -> causation -> consequence arc:

  E1  spectral_recovery   : does an aligned student recover the analytic MNUDFT
                            teacher's decision function AND its Fourier fingerprint,
                            where an x-only standard student cannot?
  E2  alignment_sweep     : is the *value* of the fixed query geometry causal?
                            Hold gate set / blocks / params fixed and vary ONLY the
                            fixed f (interpolate toward random, or rotate it),
                            tracing accuracy vs misalignment and locating the
                            crossover where a misaligned prior does WORSE than the
                            standard baseline.
  E3  sample_efficiency   : learning curves vs #training scenes. The aligned bias
                            should reach a target with fewer scenes and converge with
                            the standard model at high data (bias-not-capacity).

Every runner returns a plain dict of arrays / nested history, and each has a
``save_*`` companion that writes ``results.json`` (metadata + curves + spectra as
lists) plus ``.npy`` sidecars for the large numeric arrays, under a run directory.
The functions are deliberately thin wrappers over :mod:`sar_qml.common` so the
notebooks stay short and all heavy logic is unit-testable here.

Recommended configuration for a fair, parameter-matched comparison:
    trainable_block="yzy"   (universal SU(2) block in BOTH models)
    f_encoding_axis="Z"     (fixed f is non-absorbable)
    fixed_f=True            (f is a dataset-level hyperparameter; labels depend on x)
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace as dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .pqc import PQCConfig, PQCClassifier
from . import common
from . import spectral as sp


# ============================================================================
# small shared helpers
# ============================================================================
def _standard_config_matched(config: PQCConfig) -> PQCConfig:
    """Standard (x-only) config with total depth matched to the aligned model.

    The aligned circuit interleaves x-blocks and f-blocks (x_depth + f_depth
    variational sub-layers). To match trainable-parameter count exactly, the
    single-branch standard model must have that same total number of layers.
    """
    x_depth = config.n_x_layers if config.n_x_layers is not None else config.n_layers
    f_depth = config.n_f_layers if config.n_f_layers is not None else config.n_layers
    return dc_replace(config, n_layers=x_depth + f_depth, n_x_layers=None, n_f_layers=None)


def _best_fold(payload: Dict[str, Any]) -> int:
    accs = [m["best_val_acc"] for m in payload["fold_metrics"]]
    return int(np.argmax(accs))


def _centered_predict_double(model: PQCClassifier, weights, probe_x: np.ndarray):
    """Closure: batch of interleaved f-vectors -> centred scalar output (expval)."""
    def fn(f_points: np.ndarray) -> np.ndarray:
        X = np.tile(np.asarray(probe_x, dtype=np.float64), (f_points.shape[0], 1))
        probs = np.asarray(model.predict((X, np.asarray(f_points)), weights))
        return 2.0 * probs.reshape(-1) - 1.0
    return fn


def _centered_predict_x(model: PQCClassifier, weights, use_double: bool, fixed_f_row):
    """Closure: batch of x-vectors -> centred scalar output, at the fixed f (aligned)."""
    def fn(x_points: np.ndarray) -> np.ndarray:
        x_points = np.asarray(x_points, dtype=np.float64)
        if use_double:
            F = np.tile(np.asarray(fixed_f_row, dtype=np.float64), (x_points.shape[0], 1))
            probs = np.asarray(model.predict((x_points, F), weights))
        else:
            probs = np.asarray(model.predict(x_points, weights))
        return 2.0 * probs.reshape(-1) - 1.0
    return fn


def _param_count(config: PQCConfig, use_double: bool) -> int:
    m = PQCClassifier(config, use_double_reuploading=use_double)
    return int(m._quantum_param_size())


def _now() -> str:
    return datetime.utcnow().isoformat()


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, complex):
            return {"re": o.real, "im": o.imag}
        raise TypeError(f"not serialisable: {type(o)}")
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


# ============================================================================
# E1 : spectral recovery
# ============================================================================
def run_e1_spectral_recovery(
    config: PQCConfig,
    n_samples: int = 400,
    margin: float = 0.2,
    noise_scale: float = 0.0,
    n_grid: int = 128,
    n_freqs: int = 16,
    n_probes: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train aligned + standard on an analytic MNUDFT teacher and measure recovery.

    Saves per-epoch train/val curves (via cross-validation), and — from the best
    fold of each model — the learned Fourier fingerprint along the f-scan and
    along each x-coordinate scan, compared against the analytic teacher spectrum.

    n_positions = n_frequencies = n_qubits // 2 (matching linear_mnudft_dataset).
    """
    nq = config.n_qubits
    if nq < 2 or nq % 2 != 0:
        raise ValueError("n_qubits must be an even integer >= 2")
    n_pos = nq // 2
    n_freq = nq // 2

    x_data, f_data, y, teacher = common.mnudft_teacher_dataset(
        n_samples=n_samples,
        n_positions=n_pos,
        n_frequencies=n_freq,
        seed=seed,
        noise_scale=noise_scale,
        margin=margin,
        fixed_f=True,
    )
    f_true_flat = np.asarray(teacher["f_true_flat"], dtype=np.float64)  # length nq

    std_config = _standard_config_matched(config)

    aligned = common.run_crossval(config, x_data, y, f_data=f_data, use_double_reuploading=True)
    standard = common.run_crossval(std_config, x_data, y, f_data=None, use_double_reuploading=False)

    aligned_model = PQCClassifier(config, use_double_reuploading=True)
    standard_model = PQCClassifier(std_config, use_double_reuploading=False)

    wa = np.asarray(aligned["fold_weights"][_best_fold(aligned)])
    ws = np.asarray(standard["fold_weights"][_best_fold(standard)])

    # ---- probe points: real training samples spanning both classes ----
    rng = np.random.default_rng(seed + 101)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    take = max(1, n_probes // 2)
    probe_idx = np.concatenate([
        rng.choice(idx0, size=min(take, len(idx0)), replace=False),
        rng.choice(idx1, size=min(take, len(idx1)), replace=False),
    ])
    probes = x_data[probe_idx]

    # ---- (a) f-direction spectrum (aligned + teacher; standard has no f input) ----
    f_dir = np.ones(nq, dtype=np.float64)  # sweep all interleaved f entries in lockstep
    f_spectra_aligned, f_spectra_teacher, f_overlaps = [], [], []
    for px in probes:
        al = sp.spectrum_of_predict(
            _centered_predict_double(aligned_model, wa, px),
            base_point=f_true_flat, direction=f_dir, n_grid=n_grid, mode="set", n_freqs=n_freqs,
        )
        teach_fn = lambda fpts, _px=px: sp.mnudft_teacher_score(
            np.tile(_px, (fpts.shape[0], 1)), fpts, teacher
        )
        te = sp.spectrum_of_predict(teach_fn, base_point=f_true_flat, direction=f_dir,
                                    n_grid=n_grid, mode="set", n_freqs=n_freqs)
        f_spectra_aligned.append(np.abs(al["coeffs"]))
        f_spectra_teacher.append(np.abs(te["coeffs"]))
        f_overlaps.append(sp.spectral_overlap(al["coeffs"], te["coeffs"]))

    # ---- (b) x-coordinate spectra at fixed f_true (aligned, standard, teacher) ----
    x_spec = {"aligned": [], "standard": [], "teacher": [], "overlap_aligned": [], "overlap_standard": []}
    fn_al_x = _centered_predict_x(aligned_model, wa, use_double=True, fixed_f_row=f_true_flat)
    fn_st_x = _centered_predict_x(standard_model, ws, use_double=False, fixed_f_row=None)
    for d in range(n_pos):
        x_dir = np.zeros(n_pos); x_dir[d] = 1.0
        base_x = np.zeros(n_pos)
        al = sp.spectrum_of_predict(fn_al_x, base_x, x_dir, n_grid=n_grid, mode="set", n_freqs=n_freqs)
        st = sp.spectrum_of_predict(fn_st_x, base_x, x_dir, n_grid=n_grid, mode="set", n_freqs=n_freqs)
        teach_fn = lambda xpts, _d=d: sp.mnudft_teacher_score(
            xpts, np.tile(f_true_flat, (xpts.shape[0], 1)), teacher
        )
        te = sp.spectrum_of_predict(teach_fn, base_x, x_dir, n_grid=n_grid, mode="set", n_freqs=n_freqs)
        x_spec["aligned"].append(np.abs(al["coeffs"]))
        x_spec["standard"].append(np.abs(st["coeffs"]))
        x_spec["teacher"].append(np.abs(te["coeffs"]))
        x_spec["overlap_aligned"].append(sp.spectral_overlap(al["coeffs"], te["coeffs"]))
        x_spec["overlap_standard"].append(sp.spectral_overlap(st["coeffs"], te["coeffs"]))

    # ---- teacher-recovery accuracy on a held-out probe set ----
    x_test, f_test, y_test, _ = common.mnudft_teacher_dataset(
        n_samples=max(200, n_samples // 2), n_positions=n_pos, n_frequencies=n_freq,
        seed=seed + 999, noise_scale=noise_scale, margin=margin, fixed_f=True,
    )
    # relabel test set with the ORIGINAL teacher so both share the same target fn
    y_test_ref = sp.mnudft_teacher_label(x_test, np.tile(f_true_flat, (len(x_test), 1)), teacher)
    pa = np.asarray(aligned_model.predict((x_test, np.tile(f_true_flat, (len(x_test), 1))), wa)) > 0.5
    ps = np.asarray(standard_model.predict(x_test, ws)) > 0.5
    recovery = {
        "aligned_acc_vs_teacher": float(np.mean(pa.astype(int) == y_test_ref)),
        "standard_acc_vs_teacher": float(np.mean(ps.astype(int) == y_test_ref)),
    }

    return {
        "experiment": "E1_spectral_recovery",
        "config": asdict(config),
        "standard_config": asdict(std_config),
        "meta": {
            "n_samples": n_samples, "n_qubits": nq, "n_positions": n_pos, "n_frequencies": n_freq,
            "margin": margin, "noise_scale": noise_scale, "seed": seed,
            "n_grid": n_grid, "n_freqs": n_freqs, "n_probes": int(len(probe_idx)),
            "trainable_block": config.trainable_block, "f_encoding_axis": config.f_encoding_axis,
            "param_count_aligned": _param_count(config, True),
            "param_count_standard": _param_count(std_config, False),
        },
        "teacher": {k: v for k, v in teacher.items() if k != "scores"},
        "cv": {"aligned": aligned, "standard": standard},
        "recovery": recovery,
        "freqs": np.arange(0, n_freqs + 1),
        "f_scan": {
            "spectra_aligned": np.array(f_spectra_aligned),
            "spectra_teacher": np.array(f_spectra_teacher),
            "overlap": np.array(f_overlaps),
        },
        "x_scan": {k: np.array(v) for k, v in x_spec.items()},
        "probe_idx": probe_idx,
    }


# ============================================================================
# E2 : alignment sweep (causation)
# ============================================================================
def run_e2_alignment_sweep(
    config: PQCConfig,
    n_samples: int = 400,
    margin: float = 0.2,
    mode: str = "interpolate",      # "interpolate" | "rotate"
    alphas: Optional[List[float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Vary ONLY the fixed query geometry; keep gate set / blocks / params fixed.

    The dataset (x and its labels) is generated ONCE from the true geometry
    f_true. We then train the aligned model told a *perturbed* geometry
    f(alpha) — either interpolated toward a random geometry or rigidly rotated —
    while the labels stay defined by f_true. The standard (x-only) model is the
    alpha-independent baseline. This isolates the causal effect of *which*
    geometry the prior encodes; it should show a crossover where a badly
    misaligned prior underperforms the agnostic baseline.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0] if mode == "interpolate" \
            else list(np.linspace(0.0, np.pi / 2, 7))
    nq = config.n_qubits
    n_pos = nq // 2
    n_freq = nq // 2

    x_data, f_data, y, teacher = common.mnudft_teacher_dataset(
        n_samples=n_samples, n_positions=n_pos, n_frequencies=n_freq,
        seed=seed, margin=margin, fixed_f=True,
    )
    f_true_flat = np.asarray(teacher["f_true_flat"], dtype=np.float64)

    rng = np.random.default_rng(seed + 202)
    f_rand = rng.uniform(-1.0, 1.0, size=n_freq)
    f_other_flat = np.zeros(nq); f_other_flat[0::2] = f_rand; f_other_flat[1::2] = f_rand

    std_config = _standard_config_matched(config)
    standard = common.run_crossval(std_config, x_data, y, f_data=None, use_double_reuploading=False)
    standard_acc = standard["mean_best_val_acc"]

    rows = []
    for alpha in alphas:
        if mode == "interpolate":
            f_used = sp.interpolate_f(f_true_flat, f_other_flat, alpha)
        else:
            f_used = sp.rotate_query_frequencies(f_true_flat, alpha)
        F_used = sp.make_shared_f(f_used, n_samples)
        res = common.run_crossval(config, x_data, y, f_data=F_used, use_double_reuploading=True)
        rows.append({
            "alpha": float(alpha),
            "aligned_mean_acc": res["mean_best_val_acc"],
            "aligned_std_acc": res["std_best_val_acc"],
            "misalignment_l2": float(np.linalg.norm(f_used - f_true_flat)),
            "fold_metrics": res["fold_metrics"],
        })

    return {
        "experiment": "E2_alignment_sweep",
        "config": asdict(config),
        "standard_config": asdict(std_config),
        "meta": {
            "n_samples": n_samples, "n_qubits": nq, "margin": margin, "seed": seed,
            "mode": mode, "trainable_block": config.trainable_block,
            "f_encoding_axis": config.f_encoding_axis,
            "param_count_aligned": _param_count(config, True),
            "param_count_standard": _param_count(std_config, False),
        },
        "f_true_flat": f_true_flat,
        "f_other_flat": f_other_flat,
        "standard_baseline": {"mean_acc": standard_acc, "std_acc": standard["std_best_val_acc"],
                              "cv": standard},
        "alphas": np.asarray(alphas, dtype=np.float64),
        "sweep": rows,
        "aligned_acc": np.array([r["aligned_mean_acc"] for r in rows]),
        "aligned_acc_std": np.array([r["aligned_std_acc"] for r in rows]),
        "misalignment_l2": np.array([r["misalignment_l2"] for r in rows]),
    }


# ============================================================================
# E3 : sample efficiency (consequence)
# ============================================================================
def run_e3_sample_efficiency(
    config: PQCConfig,
    dataset: str = "mnudft",         # "mnudft" | "raw_sar"
    train_sizes: Optional[List[int]] = None,
    n_test: int = 300,
    n_repeats: int = 3,
    margin: float = 0.2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Learning curves (accuracy vs #training scenes) for aligned vs standard.

    For each training size and repeat, draw a fresh train set of that size and a
    fixed held-out test set (shared geometry, fixed_f), train both models, and
    record test accuracy. Aligned should reach a target with fewer scenes and
    converge with standard at large N (a bias-not-capacity signature).
    """
    nq = config.n_qubits
    n_pos = nq // 2
    n_freq = nq // 2
    if train_sizes is None:
        train_sizes = [20, 40, 80, 160, 320]

    std_config = _standard_config_matched(config)

    def gen(n, s):
        if dataset == "mnudft":
            x, f, yy, _ = common.mnudft_teacher_dataset(
                n_samples=n, n_positions=n_pos, n_frequencies=n_freq,
                seed=s, margin=margin, fixed_f=True)
            return x, f, yy
        elif dataset == "raw_sar":
            x, f, yy = common.raw_sar_synthetic_dataset(
                n_samples=n, n_qubits=nq, seed=s, fixed_f=True)
            return x, f, yy
        raise ValueError("dataset must be 'mnudft' or 'raw_sar'")

    # fixed held-out test set
    x_test, f_test, y_test = gen(n_test, seed + 5000)

    curves = {"aligned": [], "standard": []}
    for n_tr in train_sizes:
        acc_al, acc_st = [], []
        for r in range(n_repeats):
            x_tr, f_tr, y_tr = gen(n_tr, seed + 1000 * (r + 1) + n_tr)

            am = PQCClassifier(config, use_double_reuploading=True)
            wa, _, _ = am.train_fold((x_tr, f_tr), y_tr, (x_test, f_test), y_test, fold_num=r)
            pa = np.asarray(am.predict((x_test, f_test), wa)) > 0.5
            acc_al.append(float(np.mean(pa.astype(int) == (np.asarray(y_test) > 0.5).astype(int))))

            sm = PQCClassifier(std_config, use_double_reuploading=False)
            ws, _, _ = sm.train_fold(x_tr, y_tr, x_test, y_test, fold_num=r)
            ps = np.asarray(sm.predict(x_test, ws)) > 0.5
            acc_st.append(float(np.mean(ps.astype(int) == (np.asarray(y_test) > 0.5).astype(int))))

        curves["aligned"].append({"n_train": int(n_tr), "acc": acc_al,
                                  "mean": float(np.mean(acc_al)), "std": float(np.std(acc_al))})
        curves["standard"].append({"n_train": int(n_tr), "acc": acc_st,
                                   "mean": float(np.mean(acc_st)), "std": float(np.std(acc_st))})

    return {
        "experiment": "E3_sample_efficiency",
        "config": asdict(config),
        "standard_config": asdict(std_config),
        "meta": {
            "dataset": dataset, "n_qubits": nq, "n_test": n_test, "n_repeats": n_repeats,
            "margin": margin, "seed": seed, "trainable_block": config.trainable_block,
            "f_encoding_axis": config.f_encoding_axis,
            "param_count_aligned": _param_count(config, True),
            "param_count_standard": _param_count(std_config, False),
        },
        "train_sizes": np.asarray(train_sizes),
        "curves": curves,
        "aligned_mean": np.array([c["mean"] for c in curves["aligned"]]),
        "aligned_std": np.array([c["std"] for c in curves["aligned"]]),
        "standard_mean": np.array([c["mean"] for c in curves["standard"]]),
        "standard_std": np.array([c["std"] for c in curves["standard"]]),
    }


# ============================================================================
# saving
# ============================================================================
def save_experiment(out_dir: Path, result: Dict[str, Any]) -> Path:
    """Write results.json + .npy sidecars for large arrays under out_dir.

    Large numeric arrays (spectra, curves) are stored both as JSON (for portability)
    and as .npy (for exact reload). Returns the results.json path.
    """
    out_dir = Path(out_dir)
    npy_dir = out_dir / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # dump any top-level ndarray as .npy for exact reload
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            np.save(npy_dir / f"{key}.npy", val)
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, np.ndarray):
                    np.save(npy_dir / f"{key}__{k2}.npy", v2)

    payload = dict(result)
    payload["saved_utc"] = _now()
    _dump_json(out_dir / "results.json", payload)
    return out_dir / "results.json"
