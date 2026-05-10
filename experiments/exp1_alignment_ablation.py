from __future__ import annotations

from sar_qml.common import (
    config_from_args,
    make_common_parser,
    make_experiment_dir,
    print_summary,
    run_paired_models,
    run_paired_models_single_fold,
    save_results_bundle,
    teacher_student_dataset,
)


def main() -> None:
    parser = make_common_parser("Experiment 1: Alignment ablation (teacher-student)")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of synthetic samples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Teacher label threshold (hard labels only)")
    parser.add_argument(
        "--teacher-weight-init-scale",
        type=float,
        default=1.0,
        help="Init scale for teacher weights. Should be >> student init (0.01) so the teacher encodes a non-trivial function.",
    )
    parser.add_argument(
        "--hard-labels",
        action="store_true",
        help="Use hard 0/1 labels via threshold instead of soft teacher probabilities.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="1-based fold index to run a single fold (useful for array jobs)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name")
    args = parser.parse_args()

    if args.fold_index is not None and (args.fold_index < 1 or args.fold_index > args.n_folds):
        raise ValueError(f"--fold-index must be in [1, {args.n_folds}]")

    config = config_from_args(args)
    use_soft_labels = not args.hard_labels
    x_data, f_data, y = teacher_student_dataset(
        n_samples=args.n_samples,
        config=config,
        seed=args.seed,
        threshold=args.threshold,
        teacher_weight_init_scale=args.teacher_weight_init_scale,
        use_soft_labels=use_soft_labels,
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

    out_dir = make_experiment_dir(args.output_root, "exp1_alignment_ablation", run_name=args.run_name)

    y_hard = (y > 0.5).astype(int) if use_soft_labels else y
    metadata = {
        "n_samples": args.n_samples,
        "fold_index": args.fold_index,
        "use_soft_labels": use_soft_labels,
        "teacher_weight_init_scale": args.teacher_weight_init_scale,
        "threshold": args.threshold,
        "label_balance": {
            "class_0": int((y_hard == 0).sum()),
            "class_1": int((y_hard == 1).sum()),
        },
    }
    save_results_bundle(out_dir, "exp1_alignment_ablation", config, payload, metadata=metadata)
    print_summary(payload, "Experiment 1 complete")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
