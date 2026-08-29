# Collect the earlier experiment outputs and turn them into the final dissertation summary.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.final_summary_data import (
    METHOD_ORDER,
    SCENARIO_ORDER,
    SYNTHETIC_METHODS,
    derive_headline_findings,
    derive_synthetic_summary,
    prepare_table,
    read_csv,
    save_table,
    select_columns,
)
from src.final_summary_reporting import (
    make_figures,
    write_readme,
    write_results_section,
)


# Read where each versioned result folder is stored.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-ready v14 tables, figures and a LaTeX results "
            "section from the frozen v9-v13 outputs."
        )
    )
    parser.add_argument(
        "--v9",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v9_synthetic_functional_aci",
    )
    parser.add_argument(
        "--v10",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v10_multi_market_functional",
    )
    parser.add_argument(
        "--v11",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v11_functional_perturbation",
    )
    parser.add_argument(
        "--v12",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v12_functional_ablation",
    )
    parser.add_argument(
        "--v13",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v13_strong_baselines_unified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v14_final_summary",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser.parse_args()


# Build all v14 tables, figures, and short written findings in one run.
def main() -> None:
    args = parse_args()

    # Keep tables, figures, LaTeX, and small diagnostic files in separate folders.
    tables_dir = args.output / "tables"
    figures_dir = args.output / "figures"
    latex_dir = args.output / "latex"
    diagnostics_dir = args.output / "diagnostics"

    for directory in [tables_dir, figures_dir, latex_dir, diagnostics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Rebuild the synthetic averages from seed-level results before making the table.
    synthetic = derive_synthetic_summary(args.v9, args.alpha)

    # Load the saved v10--v13 tables; this script does not rerun any model.
    v10_results = prepare_table(
        read_csv(args.v10 / "tables" / "multi_market_results.csv"),
        args.alpha,
    )

    perturbation = prepare_table(
        read_csv(args.v11 / "tables" / "perturbation_cross_market_summary.csv"),
        args.alpha,
    )

    context_ablation = prepare_table(
        read_csv(args.v12 / "tables" / "context_ablation_average.csv"),
        args.alpha,
    )

    function_space = prepare_table(
        read_csv(args.v12 / "tables" / "function_space_ablation_average.csv"),
        args.alpha,
    )

    kernel_ablation = prepare_table(
        read_csv(args.v12 / "tables" / "kernel_ablation_average.csv"),
        args.alpha,
    )

    unified_results = prepare_table(
        read_csv(args.v13 / "tables" / "unified_all_method_results.csv"),
        args.alpha,
    )

    unified_average = prepare_table(
        read_csv(args.v13 / "tables" / "unified_all_method_average.csv"),
        args.alpha,
    )

    # Table 1 keeps only the columns needed for the synthetic comparison.
    synthetic_main = select_columns(
        synthetic,
        [
            "scenario",
            "method",
            "coverage_mean",
            "coverage_error_mean",
            "avg_width_mean",
            "winkler_mean",
            "functional_error_mean",
            "worst_group_error_mean",
            "adaptation_delay_mean",
        ],
    )

    if "scenario" in synthetic_main.columns:
        synthetic_main["scenario"] = (
            pd.Categorical(
                synthetic_main["scenario"],
                categories=SCENARIO_ORDER,
                ordered=True,
            )
        )

    if "method" in synthetic_main.columns:
        synthetic_main["method"] = (
            pd.Categorical(
                synthetic_main["method"],
                categories=SYNTHETIC_METHODS,
                ordered=True,
            )
        )

    synthetic_main = synthetic_main.sort_values(
        [
            column
            for column in ["scenario", "method"]
            if column in synthetic_main.columns
        ]
    )

    # Save both CSV and LaTeX versions so the numbers have one shared source.
    save_table(
        synthetic_main,
        tables_dir / "table_1_synthetic_main.csv",
        latex_dir / "table_1_synthetic_main.tex",
        (
            "Synthetic experiments across linear, nonlinear "
            "and non-stationary scenarios."
        ),
        "tab:synthetic-main",
    )

    # Table 2 shows each method separately for all four real markets.
    market_main = select_columns(
        v10_results,
        [
            "market",
            "method",
            "coverage",
            "coverage_error",
            "avg_width",
            "winkler",
            "functional_error",
            "worst_group_error",
            "mean_abs_linear_component",
            "mean_abs_functional_component",
        ],
    )

    market_main["method"] = pd.Categorical(
        market_main["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    market_main = market_main.sort_values(["market", "method"])

    save_table(
        market_main,
        tables_dir / "table_2_multi_market_main.csv",
        latex_dir / "table_2_multi_market_main.tex",
        (
            "Multi-market comparison on DE--LU, DK1, "
            "DK2 and SE3."
        ),
        "tab:multi-market-main",
    )

    # Table 3 averages the methods that use the same raw prediction intervals.
    unified_main = select_columns(
        unified_average,
        [
            "method",
            "coverage_mean",
            "coverage_error_mean",
            "avg_width_mean",
            "winkler_mean",
            "functional_error_mean",
            "worst_group_error_mean",
        ],
    )

    unified_main["method"] = pd.Categorical(
        unified_main["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    unified_main = unified_main.sort_values("method")

    save_table(
        unified_main,
        tables_dir / "table_3_unified_baselines_average.csv",
        latex_dir / "table_3_unified_baselines_average.tex",
        (
            "Unified comparison using identical raw quantile "
            "predictions and test observations."
        ),
        "tab:unified-baselines",
    )

    # Table 4 follows performance as the test features are perturbed more strongly.
    perturbation_main = select_columns(
        perturbation,
        [
            "rho",
            "method",
            "coverage_mean",
            "coverage_error_mean",
            "avg_width_mean",
            "winkler_mean",
            "functional_error_mean",
            "worst_group_error_mean",
        ],
    )

    perturbation_main["method"] = pd.Categorical(
        perturbation_main["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    perturbation_main = perturbation_main.sort_values(["rho", "method"])

    save_table(
        perturbation_main,
        tables_dir / "table_4_perturbation_main.csv",
        latex_dir / "table_4_perturbation_main.tex",
        (
            "Cross-market stochastic-feature "
            "perturbation results."
        ),
        "tab:perturbation-main",
    )

    # Table 5 shows what happens when one group of context variables is removed.
    context_main = select_columns(
        context_ablation,
        [
            "ablation",
            "method",
            "coverage_mean",
            "coverage_error_mean",
            "avg_width_mean",
            "winkler_mean",
            "functional_error_mean",
            "worst_group_error_mean",
        ],
    )

    save_table(
        context_main,
        tables_dir / "table_5_context_ablation.csv",
        latex_dir / "table_5_context_ablation.tex",
        (
            "Context-variable ablation averaged "
            "across markets."
        ),
        "tab:context-ablation",
    )

    # Table 6 compares scalar, linear, nonlinear, and hybrid function spaces.
    function_main = select_columns(
        function_space,
        [
            "method",
            "function_space",
            "coverage_mean",
            "coverage_error_mean",
            "avg_width_mean",
            "winkler_mean",
            "functional_error_mean",
            "worst_group_error_mean",
        ],
    )

    function_main["method"] = pd.Categorical(
        function_main["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    function_main = function_main.sort_values("method")

    save_table(
        function_main,
        tables_dir / "table_6_function_space_ablation.csv",
        latex_dir / "table_6_function_space_ablation.tex",
        (
            "Function-space ablation averaged "
            "across markets."
        ),
        "tab:function-space-ablation",
    )

    # Table 7 checks whether the random-feature size changes accuracy or runtime.
    kernel_main = select_columns(
        kernel_ablation,
        [
            "n_components",
            "length_scale",
            "coverage_mean",
            "coverage_error_mean",
            "avg_width_mean",
            "winkler_mean",
            "functional_error_mean",
            "worst_group_error_mean",
            "runtime_seconds_mean",
        ],
    )

    kernel_main = kernel_main.sort_values(
        [
            column
            for column in ["length_scale", "n_components"]
            if column in kernel_main.columns
        ]
    )

    save_table(
        kernel_main,
        tables_dir / "table_7_kernel_ablation.csv",
        latex_dir / "table_7_kernel_ablation.tex",
        "Kernel approximation sensitivity and runtime.",
        "tab:kernel-ablation",
    )

    # Derive headline percentages directly from tables instead of typing them by hand.
    headline_table, findings = (
        derive_headline_findings(
            synthetic=synthetic,
            v10_results=v10_results,
            perturbation=perturbation,
            context_ablation=context_ablation,
            unified_average=unified_average,
        )
    )

    save_table(
        headline_table,
        tables_dir / "table_8_headline_findings.csv",
        latex_dir / "table_8_headline_findings.tex",
        "Automatically derived headline comparisons.",
        "tab:headline-findings",
    )

    (
        diagnostics_dir / "key_findings.json"
    ).write_text(
        json.dumps(findings, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # Record exactly which result folders were used to build this summary.
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": args.alpha,
        "sources": {
            "v9": str(args.v9),
            "v10": str(args.v10),
            "v11": str(args.v11),
            "v12": str(args.v12),
            "v13": str(args.v13),
        },
    }

    (
        diagnostics_dir / "source_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # Use the same prepared tables for the figures and the written results section.
    make_figures(
        synthetic=synthetic,
        v10_results=v10_results,
        perturbation=perturbation,
        context_ablation=context_ablation,
        kernel_ablation=kernel_ablation,
        unified_results=unified_results,
        unified_average=unified_average,
        figures_dir=figures_dir,
    )

    write_results_section(latex_dir / "experimental_results_summary.tex", findings)

    # Create one LaTeX file that includes all eight generated tables in order.
    table_files = [
        "table_1_synthetic_main.tex",
        "table_2_multi_market_main.tex",
        "table_3_unified_baselines_average.tex",
        "table_4_perturbation_main.tex",
        "table_5_context_ablation.tex",
        "table_6_function_space_ablation.tex",
        "table_7_kernel_ablation.tex",
        "table_8_headline_findings.tex",
    ]

    (
        latex_dir / "all_final_tables.tex"
    ).write_text(
        "\n\n".join(rf"\input{{{name}}}" for name in table_files) + "\n",
        encoding="utf-8",
    )

    write_readme(args.output / "README.md")

    print("\n" + "=" * 90)
    print("V14 FINAL SUMMARY")
    print("=" * 90)

    print("\nUnified baseline average:")
    print(unified_main.to_string(index=False))

    print("\nHeadline findings:")
    print(headline_table.to_string(index=False))

    print("\nSaved final summary to: " f"{args.output}")


if __name__ == "__main__":
    main()
