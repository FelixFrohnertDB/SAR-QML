from __future__ import annotations

from sar_qml.common import (
    config_from_args,
    make_common_parser,
    make_experiment_dir,
    print_summary,
    required_dataset_size_for_fold_subsampling,
    run_paired_models,
    run_paired_models_single_fold,
    save_results_bundle,
    teacher_student_dataset,
)


def main() -> None:
    parser = make_common_parser("Experiment 1: Alignment ablation (teacher-student)")
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
    if args.train_size is None or args.train_size <= 0:
        raise ValueError("--train-size must be > 0")

    n_samples = required_dataset_size_for_fold_subsampling(
        n_folds=args.n_folds,
        train_size=args.train_size,
        val_size=args.val_size,
    )

    config = config_from_args(args)
    use_soft_labels = not args.hard_labels
    x_data, f_data, y = teacher_student_dataset(
        n_samples=n_samples,
        config=config,
        seed=args.seed,
        teacher_weight_init_scale=args.teacher_weight_init_scale,
        use_soft_labels=use_soft_labels,
    )

    if args.fold_index is None:
        payload = run_paired_models(
            config=config,
            x_data=x_data,
            f_data=f_data,
            y=y,
            train_size=args.train_size,
            val_size=args.val_size,
        )
    else:
        payload = run_paired_models_single_fold(
            config=config,
            x_data=x_data,
            f_data=f_data,
            y=y,
            fold_index=args.fold_index,
            train_size=args.train_size,
            val_size=args.val_size,
        )

    size_tag = f"tr{args.train_size or 'full'}_va{args.val_size if args.val_size and args.val_size > 0 else 'full'}"
    run_name = f"{args.run_name}_{size_tag}" if args.run_name else size_tag
    out_dir = make_experiment_dir(args.output_root, "exp1_alignment_ablation", run_name=run_name)

    case_out_dir = out_dir / f"case_n{args.n_qubits}"
    fold_dir_name = f"fold_{args.fold_index}" if args.fold_index is not None else "fold_all"
    case_out_dir = case_out_dir / fold_dir_name
    case_out_dir.mkdir(parents=True, exist_ok=True)
    (case_out_dir / "weights").mkdir(parents=True, exist_ok=True)

    y_hard = (y > 0.5).astype(int) if use_soft_labels else y
    metadata = {
        "n_samples_generated": n_samples,
        "n_qubits": args.n_qubits,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "fold_index": args.fold_index,
        "fixed_f": True,
        "f_encoding_axis": args.f_encoding_axis,
        "use_soft_labels": use_soft_labels,
        "label_threshold_mode": "median_rank_balanced",
        "teacher_weight_init_scale": args.teacher_weight_init_scale,
        "label_balance": {
            "class_0": int((y_hard == 0).sum()),
            "class_1": int((y_hard == 1).sum()),
        },
    }
    save_results_bundle(case_out_dir, "exp1_alignment_ablation", config, payload, metadata=metadata)
    print_summary(payload, "Experiment 1 complete")
    print(f"saved to: {case_out_dir}")


if __name__ == "__main__":
    main()
