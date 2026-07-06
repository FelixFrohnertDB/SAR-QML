from __future__ import annotations

from sar_qml.common import (
    config_from_args,
    make_common_parser,
    make_experiment_dir,
    print_summary,
    run_paired_models,
    run_paired_models_single_fold,
    save_results_bundle,
    linear_mnudft_dataset,
)


def main() -> None:
    parser = make_common_parser("Experiment 2: MNUDFT linearity")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of synthetic samples per case")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name")
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="1-based fold index to run a single fold (useful for array jobs)",
    )

    args = parser.parse_args()

    if args.n_qubits < 2 or (args.n_qubits % 2) != 0:
        raise ValueError("--n-qubits must be an even integer >= 2")
    if args.fold_index is not None and (args.fold_index < 1 or args.fold_index > args.n_folds):
        raise ValueError(f"--fold-index must be in [1, {args.n_folds}]")

    # for n_qubits in qubit_cases:
    print(f"\n{'='*60}")
    print(f"Testing N={args.n_qubits} qubits")
    print(f"{'='*60}")

    config = config_from_args(args)

    # Generate linearly-separable-in-MNUDFT-space dataset
    x_data, f_data, y = linear_mnudft_dataset(
        n_samples=args.n_samples,
        n_positions=args.n_qubits // 2,
        n_frequencies=args.n_qubits // 2,
        seed=args.seed,
        fixed_f=args.fixed_f,
    )

    # Run paired models (aligned vs standard X-only)
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

    # Prepare metadata
    y_hard = (y > 0.5).astype(int)
    metadata = {
        "n_samples": args.n_samples,
        "n_qubits": args.n_qubits,
        "n_positions": args.n_qubits // 2,
        "n_frequencies": args.n_qubits // 2,
        "fold_index": args.fold_index,
        "fixed_f": args.fixed_f,
        "f_encoding_axis": args.f_encoding_axis,
        "label_balance": {
            "class_0": int((y_hard == 0).sum()),
            "class_1": int((y_hard == 1).sum()),
        },
    }

    # Print summary for this case
    summary_title = f"N={args.n_qubits} case complete"
    if args.fold_index is not None:
        summary_title = f"N={args.n_qubits} fold={args.fold_index} complete"
    print_summary(payload, summary_title)

    # Save all results to single directory
    out_dir = make_experiment_dir(args.output_root, "exp2_mnudft_linearity", run_name=args.run_name)

    # Save with a unique subdirectory label per case
    case_out_dir = out_dir / f"case_n{args.n_qubits}"
    if args.fold_index is not None:
        case_out_dir = case_out_dir / f"fold_{args.fold_index}"
    case_out_dir.mkdir(parents=True, exist_ok=True)
    (case_out_dir / "weights").mkdir(parents=True, exist_ok=True)

    save_results_bundle(
        case_out_dir,
        "exp2_mnudft_linearity",
        config,
        payload,
        metadata=metadata,
    )

    print(f"\n{'='*60}")
    print(f"All cases complete. Saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
