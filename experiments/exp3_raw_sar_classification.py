from __future__ import annotations

from pathlib import Path

from src.sar_qml.common import (
    balanced_subsample,
    config_from_args,
    load_sar_csv,
    make_common_parser,
    make_experiment_dir,
    run_paired_models,
    save_results_bundle,
)


def run_case(config, raw_csv: Path, focused_csv: Path | None, label_threshold: int, seed: int):
    x_raw, f_raw, y_raw, _ = load_sar_csv(
        raw_csv,
        prefix="raw",
        n_qubits=config.n_qubits,
        label_threshold=label_threshold,
        seed=seed,
    )
    raw_result = run_paired_models(config=config, x_data=x_raw, f_data=f_raw, y=y_raw)

    focused_result = None
    if focused_csv is not None and focused_csv.exists():
        x_foc, f_foc, y_foc, _ = load_sar_csv(
            focused_csv,
            prefix="focused",
            n_qubits=config.n_qubits,
            label_threshold=label_threshold,
            seed=seed + 11,
        )
        focused_result = run_paired_models(config=config, x_data=x_foc, f_data=f_foc, y=y_foc)

    return raw_result, focused_result


def main() -> None:
    parser = make_common_parser("Experiment 5: Final raw SAR classification benchmark")
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="noise",
        choices=["noise", "nonoise"],
        help="Which SAR CSV set to use",
    )
    parser.add_argument("--label-threshold", type=int, default=3, help="Positive class if num_targets >= threshold")
    parser.add_argument("--include-focused", action="store_true", help="Also evaluate focused counterpart as reference")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Optional balanced subsample size for practical full-batch training on SAR CSVs",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name")
    args = parser.parse_args()

    config = config_from_args(args)

    base = Path("data")
    raw_csv = base / f"sar_onedim_{args.noise_mode}_raw_complex_realimag.csv"
    focused_csv = base / f"sar_onedim_{args.noise_mode}_focused_complex_realimag.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw dataset: {raw_csv}")

    x_raw, f_raw, y_raw, raw_ids = load_sar_csv(
        raw_csv,
        prefix="raw",
        n_qubits=config.n_qubits,
        label_threshold=args.label_threshold,
        seed=args.seed,
    )
    subset_idx, x_raw_sub, y_raw_sub = balanced_subsample(x_raw, y_raw, args.max_samples, args.seed)
    f_raw_sub = f_raw[subset_idx]
    raw_result = run_paired_models(config=config, x_data=x_raw_sub, f_data=f_raw_sub, y=y_raw_sub)

    focused_result = None
    focused_subset_size = None
    if args.include_focused and focused_csv.exists():
        x_foc, f_foc, y_foc, _ = load_sar_csv(
            focused_csv,
            prefix="focused",
            n_qubits=config.n_qubits,
            label_threshold=args.label_threshold,
            seed=args.seed + 11,
        )
        foc_idx, x_foc_sub, y_foc_sub = balanced_subsample(x_foc, y_foc, args.max_samples, args.seed + 11)
        f_foc_sub = f_foc[foc_idx]
        focused_subset_size = int(len(y_foc_sub))
        focused_result = run_paired_models(config=config, x_data=x_foc_sub, f_data=f_foc_sub, y=y_foc_sub)

    payload = {"raw": raw_result}
    if focused_result is not None:
        payload["focused_reference"] = focused_result

    out_dir = make_experiment_dir(args.output_root, "exp5_raw_sar_classification", run_name=args.run_name)
    save_results_bundle(
        out_dir,
        "exp5_raw_sar_classification",
        config,
        payload=payload,
        metadata={
            "noise_mode": args.noise_mode,
            "label_threshold": args.label_threshold,
            "max_samples": args.max_samples,
            "raw_dataset_size": int(len(y_raw)),
            "raw_subset_size": int(len(y_raw_sub)),
            "raw_csv": str(raw_csv),
            "focused_csv": str(focused_csv) if args.include_focused else None,
            "focused_subset_size": focused_subset_size,
        },
    )

    print("\n=== Experiment 5 complete ===")
    print(
        f"raw/aligned: {raw_result['aligned']['mean_best_val_acc']:.4f} +/- "
        f"{raw_result['aligned']['std_best_val_acc']:.4f}"
    )
    print(
        f"raw/standard: {raw_result['standard']['mean_best_val_acc']:.4f} +/- "
        f"{raw_result['standard']['std_best_val_acc']:.4f}"
    )
    if focused_result is not None:
        print(
            f"focused/aligned: {focused_result['aligned']['mean_best_val_acc']:.4f} +/- "
            f"{focused_result['aligned']['std_best_val_acc']:.4f}"
        )
        print(
            f"focused/standard: {focused_result['standard']['mean_best_val_acc']:.4f} +/- "
            f"{focused_result['standard']['std_best_val_acc']:.4f}"
        )
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
