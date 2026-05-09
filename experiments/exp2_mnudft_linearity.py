from __future__ import annotations

from sar_qml.common import (
    config_from_args,
    make_common_parser,
    make_experiment_dir,
    print_summary,
    run_paired_models,
    save_results_bundle,
    linear_mnudft_dataset,
)


def main() -> None:
    parser = make_common_parser("Experiment 2: MNUDFT linearity (3 qubit circuits)")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of synthetic samples per case")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name")
    args = parser.parse_args()

    # Test three cases: N=2, 4, 6 qubits
    qubit_cases = [2]
    all_results = {}

    for n_qubits in qubit_cases:
        print(f"\n{'='*60}")
        print(f"Testing N={n_qubits} qubits")
        print(f"{'='*60}")

        # Override n_qubits for this case
        args.n_qubits = n_qubits
        config = config_from_args(args)

        # Generate linearly-separable-in-MNUDFT-space dataset
        x_data, f_data, y = linear_mnudft_dataset(
            n_samples=args.n_samples,
            n_positions=n_qubits // 2,
            n_frequencies=n_qubits // 2,
            seed=args.seed,
        )

        # Run paired models (aligned vs standard X-only)
        payload = run_paired_models(config=config, x_data=x_data, f_data=f_data, y=y)

        # Prepare metadata
        y_hard = (y > 0.5).astype(int)
        metadata = {
            "n_samples": args.n_samples,
            "n_qubits": n_qubits,
            "n_positions": n_qubits // 2,
            "n_frequencies": n_qubits // 2,
            "label_balance": {
                "class_0": int((y_hard == 0).sum()),
                "class_1": int((y_hard == 1).sum()),
            },
        }

        all_results[f"n_qubits_{n_qubits}"] = payload
        all_results[f"metadata_{n_qubits}"] = metadata

        # Print summary for this case
        print_summary(payload, f"N={n_qubits} case complete")

    # Save all results to single directory
    out_dir = make_experiment_dir(args.output_root, "exp2_mnudft_linearity", run_name=args.run_name)

    # Create a comprehensive metadata
    comprehensive_metadata = {
        "n_samples": args.n_samples,
        "test_cases": qubit_cases,
        "descriptions": {
            "n_qubits_2": "2D spatial grid (1 positions) × 1 frequency query",
            "n_qubits_4": "2D spatial grid (2 positions) × 2 frequency queries",
            "n_qubits_6": "2D spatial grid (3 positions) × 3 frequency queries",
        },
    }
    for n_qubits in qubit_cases:
        comprehensive_metadata[f"metadata_{n_qubits}"] = all_results.get(f"metadata_{n_qubits}", {})

    # Save results for each case separately
    for n_qubits in qubit_cases:
        payload = all_results[f"n_qubits_{n_qubits}"]
        meta = all_results[f"metadata_{n_qubits}"]

        # Save with a unique subdirectory label per case
        args.n_qubits = n_qubits
        config = config_from_args(args)
        case_out_dir = out_dir / f"case_n{n_qubits}"
        case_out_dir.mkdir(parents=True, exist_ok=True)
        (case_out_dir / "weights").mkdir(parents=True, exist_ok=True)

        save_results_bundle(
            case_out_dir,
            "exp2_mnudft_linearity",
            config,
            payload,
            metadata=meta,
        )

    print(f"\n{'='*60}")
    print(f"All cases complete. Saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
