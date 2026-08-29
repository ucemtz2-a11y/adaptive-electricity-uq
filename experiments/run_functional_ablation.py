# Module purpose: Run context, function-space, and kernel ablation experiments.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ablation_experiments import (  # noqa: E402
    CONTEXT_ABLATIONS,
    run_context_ablation_market,
    run_kernel_ablation_market,
)
from src.ablation_reporting import (  # noqa: E402
    add_context_degradation,
    aggregate_context_ablation,
    aggregate_kernel_ablation,
    build_function_space_tables,
    plot_context_ablation,
    plot_function_space_ablation,
    plot_kernel_ablation,
)
from src.protocol import DEVELOPMENT_PROTOCOL, PAPER_MARKETS  # noqa: E402


# Parse args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run v12 context, function-space, and kernel-approximation "
            "ablations for Functional ACI."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "processed" / "multi_market"
        ),
    )

    parser.add_argument(
        "--v10-results",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/versions/results_v10_multi_market_functional"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/versions/results_v12_functional_ablation"
        ),
    )

    parser.add_argument("--markets", nargs="*", default=list(PAPER_MARKETS))

    parser.add_argument("--alpha", type=float, default=DEVELOPMENT_PROTOCOL.alpha)
    parser.add_argument("--train-frac", type=float, default=DEVELOPMENT_PROTOCOL.train_fraction)
    parser.add_argument("--validation-frac", type=float, default=DEVELOPMENT_PROTOCOL.validation_fraction)
    parser.add_argument("--random-state", type=int, default=DEVELOPMENT_PROTOCOL.random_state)

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use only the frozen v10 learning rates and a reduced kernel grid."
        ),
    )

    return parser.parse_args()


# Main.
def main() -> None:
    args = parse_args()

    # Read runtime arguments and prepare experiment output directories.
    tables_dir = (
        args.output / "tables"
    )

    # Generate and save figures for headline results and diagnostics.
    figures_dir = (
        args.output / "figures"
    )

    diagnostics_dir = (
        args.output / "diagnostics"
    )

    tables_dir.mkdir(parents=True, exist_ok=True)

    figures_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Configure hyperparameter candidates for quick or full execution.
    configuration = {
        "markets": args.markets,
        "alpha": args.alpha,
        "train_frac": args.train_frac,
        "validation_frac": (
            args.validation_frac
        ),
        "context_ablations": (
            CONTEXT_ABLATIONS
        ),
        "kernel_components": (
            [64, 128] if args.quick else [64, 128, 256]
        ),
        "kernel_length_scales": (
            [1.0, 2.0] if args.quick else [1.0, 2.0, 4.0]
        ),
        "quick": args.quick,
    }

    # Save result tables, prediction details, and reproducible diagnostics.
    (
        diagnostics_dir / "ablation_configuration.json"
    ).write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    context_results = []
    kernel_results = []

    for market_prefix in args.markets:
        (
            context_table,
            _,
            prepared,
        ) = run_context_ablation_market(
            market_prefix=market_prefix,
            args=args,
            tables_dir=tables_dir,
            diagnostics_dir=(
                diagnostics_dir
            ),
        )

        context_results.append(context_table)

        kernel_results.append(
            run_kernel_ablation_market(
                market_prefix=market_prefix,
                prepared=prepared,
                args=args,
            )
        )

    context_results = pd.concat(context_results, ignore_index=True)

    # Aggregate results across markets, scenarios, or seeds for comparison.
    context_summary = (
        aggregate_context_ablation(context_results)
    )

    # Compute coverage, interval-efficiency, and conditional-coverage metrics.
    context_degradation = (
        add_context_degradation(context_results)
    )

    kernel_results = pd.concat(kernel_results, ignore_index=True)

    kernel_summary = (
        aggregate_kernel_ablation(kernel_results)
    )

    (
        function_space_results,
        function_space_average,
        function_space_pairwise,
    ) = build_function_space_tables(args.v10_results)

    context_results.to_csv(tables_dir / "context_ablation_results.csv", index=False)

    context_summary.to_csv(tables_dir / "context_ablation_average.csv", index=False)

    context_degradation.to_csv(
        tables_dir / "context_ablation_degradation.csv",
        index=False,
    )

    function_space_results.to_csv(
        tables_dir / "function_space_ablation_results.csv",
        index=False,
    )

    function_space_average.to_csv(
        tables_dir / "function_space_ablation_average.csv",
        index=False,
    )

    function_space_pairwise.to_csv(
        tables_dir / "function_space_pairwise_deltas.csv",
        index=False,
    )

    kernel_results.to_csv(tables_dir / "kernel_ablation_results.csv", index=False)

    kernel_summary.to_csv(tables_dir / "kernel_ablation_average.csv", index=False)

    plot_context_ablation(context_summary, figures_dir)

    plot_function_space_ablation(function_space_average, figures_dir)

    plot_kernel_ablation(kernel_summary, figures_dir)

    print("\n" + "=" * 90)
    print("V12 CONTEXT ABLATION AVERAGE")
    print("=" * 90)

    context_display_columns = [
        "ablation",
        "method",
        "coverage_mean",
        "coverage_error_mean",
        "avg_width_mean",
        "winkler_mean",
        "functional_error_mean",
        "worst_group_error_mean",
    ]

    print(context_summary[context_display_columns].to_string(index=False))

    print("\n" + "=" * 90)
    print("V12 FUNCTION-SPACE ABLATION")
    print("=" * 90)

    function_display_columns = [
        "method",
        "function_space",
        "coverage_mean",
        "coverage_error_mean",
        "avg_width_mean",
        "winkler_mean",
        "functional_error_mean",
        "worst_group_error_mean",
    ]

    print(function_space_average[function_display_columns].to_string(index=False))

    print("\n" + "=" * 90)
    print("V12 KERNEL APPROXIMATION ABLATION")
    print("=" * 90)

    kernel_display_columns = [
        "n_components",
        "length_scale",
        "coverage_mean",
        "avg_width_mean",
        "winkler_mean",
        "functional_error_mean",
        "worst_group_error_mean",
        "runtime_seconds_mean",
    ]

    print(kernel_summary[kernel_display_columns].to_string(index=False))

    print("\nSaved results to: " f"{args.output}")


if __name__ == "__main__":
    main()
