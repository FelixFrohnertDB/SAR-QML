from __future__ import annotations

from sar_qml.common import (
    config_from_args,
    make_common_parser,
    make_experiment_dir,
    print_summary,
    raw_sar_synthetic_dataset,
    run_paired_models,
    run_paired_models_single_fold,
    save_results_bundle,
)

def main() -> None:
    parser = make_common_parser("Experiment 3: Synthetic raw SAR classification")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of synthetic samples")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.03,
        help="Complex measurement noise scale added before reconstruction",
    )
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=0.55,
        help="Relative reconstruction threshold used to count dominant reflectors",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="1-based fold index to run a single fold (useful for array jobs)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name")
    args = parser.parse_args()

    if args.n_qubits < 2 or (args.n_qubits % 2) != 0:
        raise ValueError("--n-qubits must be an even integer >= 2")
    if args.fold_index is not None and (args.fold_index < 1 or args.fold_index > args.n_folds):
        raise ValueError(f"--fold-index must be in [1, {args.n_folds}]")

    config = config_from_args(args)

    if args.n_qubits == 2:
        class_pair = (0, 1)
    elif args.n_qubits == 4:
        class_pair = (1, 2)
    elif args.n_qubits == 6:
        class_pair = (2, 3)
    else:
        raise ValueError("Experiment 3 currently expects --n-qubits in {2, 4, 6}")

    x_data, f_data, y = raw_sar_synthetic_dataset(
        n_samples=args.n_samples,
        n_qubits=args.n_qubits,
        seed=args.seed,
        noise_scale=args.noise_scale,
        dominance_ratio=args.dominance_ratio,
    )

    if args.fold_index is None:
        payload = run_paired_models(config=config, x_data=x_data, f_data=f_data, y=y)
    else:
        payload = run_paired_models_single_fold(
            config=config,
            x_data=x_data,
            f_data=f_data,
            y=y,
            fold_index=args.fold_index,
        )

    out_dir = make_experiment_dir(args.output_root, "exp3_raw_sar_classification", run_name=args.run_name)
    save_results_bundle(
        out_dir,
        "exp3_raw_sar_classification",
        config,
        payload,
        metadata={
            "n_samples": args.n_samples,
            "n_qubits": args.n_qubits,
            "n_measurements": args.n_qubits // 2,
            "n_scene_positions": min(3, args.n_qubits // 2),
            "class_pair": {
                "class_0_reflectors": class_pair[0],
                "class_1_reflectors": class_pair[1],
            },
            "noise_scale": args.noise_scale,
            "dominance_ratio": args.dominance_ratio,
            "fold_index": args.fold_index,
            "label_balance": {
                "class_0": int((y == 0).sum()),
                "class_1": int((y == 1).sum()),
            },
        },
    )

    summary_title = f"E3 synthetic raw SAR complete (N={args.n_qubits})"
    if args.fold_index is not None:
        summary_title = f"E3 synthetic raw SAR complete (N={args.n_qubits}, fold={args.fold_index})"
    print_summary(payload, summary_title)

    print("\n=== Experiment 3 complete ===")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
