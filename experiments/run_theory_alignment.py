# Module purpose: Test theoretical regret, functional-error, and perturbation-budget relationships.

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.theory_alignment import (
    aggregate_results,
    run_single_path,
    safe_log_slope,
)


# Parse args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V15 theory-alignment experiment for robust adaptive "
            "functional calibration under drift and feature perturbations."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/versions/results_v15_theory_alignment"
        ),
    )

    parser.add_argument(
        "--t-grid",
        nargs="*",
        type=int,
        default=[1000, 2000, 5000, 10000],
    )

    parser.add_argument(
        "--drift-grid",
        nargs="*",
        type=float,
        default=[0.0, 0.25, 0.50, 1.00],
    )

    parser.add_argument(
        "--rho-grid",
        nargs="*",
        type=float,
        default=[0.0, 0.05, 0.10, 0.20, 0.30],
    )

    parser.add_argument("--grid-t", type=int, default=5000)

    parser.add_argument("--scaling-drift", type=float, default=0.50)

    parser.add_argument("--scaling-rho", type=float, default=0.0)

    parser.add_argument("--seeds", type=int, default=20)

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use 3 seeds, a reduced drift/rho grid, and shorter "
            "horizon scaling for a pipeline check."
        ),
    )

    parser.add_argument("--alpha", type=float, default=0.10)

    parser.add_argument("--context-dim", type=int, default=4)

    parser.add_argument("--rff-components", type=int, default=24)

    parser.add_argument("--length-scale", type=float, default=1.5)

    parser.add_argument("--residual-ridge", type=float, default=1e-3)

    parser.add_argument("--train-fraction-map", type=float, default=0.20)

    parser.add_argument("--parameter-radius", type=float, default=6.0)

    parser.add_argument("--eta-scale", type=float, default=2.0)

    parser.add_argument("--score-sensitivity", type=float, default=0.35)

    parser.add_argument("--random-state", type=int, default=20260810)

    # Read runtime arguments and prepare experiment output directories.
    return parser.parse_args()


# Save figures.
def save_figures(
    grid_summary: pd.DataFrame,
    scaling_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    # Generate and save figures for headline results and diagnostics.
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test the relationship between dynamic regret and the unified theoretical driver.
    plt.figure(figsize=(8, 5))

    # Aggregate results across markets, scenarios, or seeds for comparison.
    plt.scatter(
        grid_summary["unified_regret_driver_mean"],
        grid_summary["dynamic_regret_clean_avg_mean"],
        s=55,
    )

    plt.xlabel(r"$(1+V_T)/\sqrt{T}+B_T/T$")

    plt.ylabel("Mean clean dynamic regret per step")

    plt.title("Dynamic regret versus drift-perturbation driver")

    plt.tight_layout()

    # Save result tables, prediction details, and reproducible diagnostics.
    plt.savefig(figures_dir / "regret_vs_unified_driver.png", dpi=240)

    plt.close()

    # 2. Test clean-path functional error against the calibration driver.
    plt.figure(figsize=(8, 5))

    plt.scatter(
        grid_summary["unified_calibration_driver_mean"],
        grid_summary["conditional_functional_error_clean_mean"],
        s=55,
    )

    plt.xlabel(r"$T^{-1/2}+B_T/T$")

    plt.ylabel("Mean clean conditional functional error")

    plt.title("Functional calibration versus perturbation driver")

    plt.tight_layout()

    plt.savefig(figures_dir / "calibration_vs_unified_driver.png", dpi=240)

    plt.close()

    # 3. Test the relationship between drift intensity and dynamic regret.
    zero_rho = grid_summary[np.isclose(grid_summary["rho"], 0.0)].sort_values(
        "drift_intensity"
    )

    if not zero_rho.empty:
        plt.figure(figsize=(8, 5))

        plt.plot(
            zero_rho["drift_intensity"],
            zero_rho["dynamic_regret_clean_avg_mean"],
            marker="o",
        )

        plt.xlabel("Drift intensity")

        plt.ylabel("Mean clean dynamic regret per step")

        plt.title("Drift primarily affects tracking efficiency")

        plt.tight_layout()

        plt.savefig(figures_dir / "drift_vs_dynamic_regret.png", dpi=240)

        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(
            zero_rho["drift_intensity"],
            zero_rho["conditional_functional_error_clean_mean"],
            marker="o",
        )

        plt.xlabel("Drift intensity")

        plt.ylabel("Mean clean conditional functional error")

        plt.title("Functional calibration under increasing drift")

        plt.tight_layout()

        plt.savefig(figures_dir / "drift_vs_functional_calibration.png", dpi=240)

        plt.close()

    # 4. Test coverage-state flips against the one-step perturbation budget.
    plt.figure(figsize=(8, 5))

    plt.scatter(
        grid_summary["budget_per_step_mean"],
        grid_summary["flip_rate_mean"],
        s=55,
    )

    plt.xlabel(r"$B_T/T$")

    plt.ylabel("Mean coverage-status flip rate")

    plt.title("Coverage flips versus perturbation budget")

    plt.tight_layout()

    plt.savefig(figures_dir / "flip_rate_vs_budget.png", dpi=240)

    plt.close()

    # 5. Test how cumulative error scales with the time horizon.
    scaling_sorted = (
        scaling_summary.sort_values("T")
    )

    plt.figure(figsize=(8, 5))

    plt.loglog(
        scaling_sorted["T"],
        scaling_sorted["conditional_functional_error_clean_mean"],
        marker="o",
    )

    reference = (
        scaling_sorted["conditional_functional_error_clean_mean"].iloc[0]
        * np.sqrt(scaling_sorted["T"].iloc[0] / scaling_sorted["T"])
    )

    plt.loglog(
        scaling_sorted["T"],
        reference,
        linestyle="--",
        label=r"$T^{-1/2}$ reference",
    )

    plt.xlabel("Horizon T")
    plt.ylabel("Conditional functional error")

    plt.title("Finite-horizon functional-calibration scaling")

    plt.legend()
    plt.tight_layout()

    plt.savefig(figures_dir / "horizon_scaling_functional_error.png", dpi=240)

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.loglog(
        scaling_sorted["T"],
        scaling_sorted["dynamic_regret_clean_avg_mean"],
        marker="o",
    )

    regret_reference = (
        max(scaling_sorted["dynamic_regret_clean_avg_mean"].iloc[0], 1e-8)
        * np.sqrt(scaling_sorted["T"].iloc[0] / scaling_sorted["T"])
    )

    plt.loglog(
        scaling_sorted["T"],
        regret_reference,
        linestyle="--",
        label=r"$T^{-1/2}$ reference",
    )

    plt.xlabel("Horizon T")
    plt.ylabel("Mean clean dynamic regret per step")

    plt.title("Finite-horizon dynamic-regret scaling")

    plt.legend()
    plt.tight_layout()

    plt.savefig(figures_dir / "horizon_scaling_dynamic_regret.png", dpi=240)

    plt.close()


# Main.
def main() -> None:
    args = parse_args()

    # Configure hyperparameter candidates for quick or full execution.
    if args.quick:
        args.seeds = 3
        args.drift_grid = [0.0, 0.5, 1.0]
        args.rho_grid = [0.0, 0.10, 0.30]
        args.t_grid = [1000, 3000, 5000]
        args.grid_t = 3000

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

    for directory in [tables_dir, figures_dir, diagnostics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    configuration = {
        "alpha": args.alpha,
        "context_dim": (
            args.context_dim
        ),
        "rff_components": (
            args.rff_components
        ),
        "length_scale": (
            args.length_scale
        ),
        "residual_ridge": (
            args.residual_ridge
        ),
        "parameter_radius": (
            args.parameter_radius
        ),
        "eta_scale": (
            args.eta_scale
        ),
        "score_sensitivity": (
            args.score_sensitivity
        ),
        "grid_T": (
            args.grid_t
        ),
        "drift_grid": (
            args.drift_grid
        ),
        "rho_grid": (
            args.rho_grid
        ),
        "horizon_grid": (
            args.t_grid
        ),
        "horizon_scaling_drift": (
            args.scaling_drift
        ),
        "horizon_scaling_rho": (
            args.scaling_rho
        ),
        "seeds": args.seeds,
        "quick": args.quick,
        "score_noise": (
            "Uniform(-0.9, 0.1), "
            "whose 0.9 quantile is zero"
        ),
        "oracle_definition": (
            "time-varying parameter path "
            "inside the same direct-sum feature space"
        ),
    }

    # Save result tables, prediction details, and reproducible diagnostics.
    (
        diagnostics_dir / "theory_alignment_configuration.json"
    ).write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    start = time.perf_counter()

    grid_rows = []

    total_grid = (
        len(args.drift_grid) * len(args.rho_grid) * args.seeds
    )

    counter = 0

    for (
        drift_intensity,
        rho,
        seed_index,
    ) in product(args.drift_grid, args.rho_grid, range(args.seeds)):
        counter += 1

        seed = (
            args.random_state + 100_000 * seed_index + int(10_000 * drift_intensity)
            + int(1_000 * rho)
        )

        print(
            f"[grid {counter}/{total_grid}] "
            f"T={args.grid_t}, "
            f"drift={drift_intensity:g}, "
            f"rho={rho:g}, "
            f"seed={seed_index}"
        )

        grid_rows.append(
            run_single_path(
                t_horizon=(
                    args.grid_t
                ),
                drift_intensity=(
                    drift_intensity
                ),
                rho=rho,
                seed=seed,
                args=args,
            )
        )

    grid_results = pd.DataFrame(grid_rows)

    # Aggregate results across markets, scenarios, or seeds for comparison.
    grid_summary = aggregate_results(grid_results, ["T", "drift_intensity", "rho"])

    scaling_rows = []

    total_scaling = (
        len(args.t_grid) * args.seeds
    )

    counter = 0

    for (
        t_horizon,
        seed_index,
    ) in product(args.t_grid, range(args.seeds)):
        counter += 1

        seed = (
            args.random_state + 50_000_000 + 100_000 * seed_index + t_horizon
        )

        print(
            f"[scaling {counter}/{total_scaling}] "
            f"T={t_horizon}, "
            f"drift={args.scaling_drift:g}, "
            f"rho={args.scaling_rho:g}, "
            f"seed={seed_index}"
        )

        scaling_rows.append(
            run_single_path(
                t_horizon=(
                    t_horizon
                ),
                drift_intensity=(
                    args.scaling_drift
                ),
                rho=(
                    args.scaling_rho
                ),
                seed=seed,
                args=args,
            )
        )

    scaling_results = (
        pd.DataFrame(scaling_rows)
    )

    scaling_summary = (
        aggregate_results(scaling_results, ["T"])
    )

    calibration_slope = safe_log_slope(
        scaling_summary["T"].to_numpy(dtype=float),
        scaling_summary["conditional_functional_error_clean_mean"].to_numpy(
            dtype=float
        ),
    )

    empirical_calibration_slope = safe_log_slope(
        scaling_summary["T"].to_numpy(dtype=float),
        scaling_summary["functional_error_clean_mean"].to_numpy(dtype=float),
    )

    regret_slope = safe_log_slope(
        scaling_summary["T"].to_numpy(dtype=float),
        np.maximum(
            scaling_summary["dynamic_regret_clean_avg_mean"].to_numpy(dtype=float),
            1e-12,
        ),
    )

    slope_table = pd.DataFrame(
        [
            {
                "metric": (
                    "functional_error_clean"
                ),
                "log_log_slope_vs_T": (
                    empirical_calibration_slope
                ),
                "theoretical_reference": (
                    -0.5
                ),
            },
            {
                "metric": (
                    "conditional_functional_error_clean"
                ),
                "log_log_slope_vs_T": (
                    calibration_slope
                ),
                "theoretical_reference": (
                    -0.5
                ),
            },
            {
                "metric": (
                    "dynamic_regret_clean_avg"
                ),
                "log_log_slope_vs_T": (
                    regret_slope
                ),
                "theoretical_reference": (
                    -0.5
                ),
            },
        ]
    )

    grid_results.to_csv(tables_dir / "theory_alignment_grid_results.csv", index=False)

    grid_summary.to_csv(tables_dir / "theory_alignment_grid_summary.csv", index=False)

    scaling_results.to_csv(tables_dir / "horizon_scaling_results.csv", index=False)

    scaling_summary.to_csv(tables_dir / "horizon_scaling_summary.csv", index=False)

    slope_table.to_csv(tables_dir / "horizon_scaling_slopes.csv", index=False)

    save_figures(
        grid_summary=(
            grid_summary
        ),
        scaling_summary=(
            scaling_summary
        ),
        figures_dir=(
            figures_dir
        ),
    )

    elapsed = (
        time.perf_counter() - start
    )

    print("\n" + "=" * 90)

    print("V15 THEORY ALIGNMENT: GRID SUMMARY")

    print("=" * 90)

    display_columns = [
        "drift_intensity",
        "rho",
        "oracle_path_variation_mean",
        "budget_per_step_mean",
        "flip_rate_mean",
        "conditional_functional_error_clean_mean",
        "functional_error_clean_mean",
        "dynamic_regret_clean_avg_mean",
        "projection_correction_rate_mean",
    ]

    print(grid_summary[display_columns].to_string(index=False))

    print("\n" + "=" * 90)

    print("V15 HORIZON SCALING")

    print("=" * 90)

    scaling_display = [
        "T",
        "conditional_functional_error_clean_mean",
        "functional_error_clean_mean",
        "dynamic_regret_clean_avg_mean",
        "oracle_path_variation_mean",
        "projection_correction_rate_mean",
    ]

    print(scaling_summary[scaling_display].to_string(index=False))

    print("\nLog-log slopes:")

    print(slope_table.to_string(index=False))

    print(f"\nRuntime: {elapsed:.2f} seconds")

    print("Saved results to: " f"{args.output}")


if __name__ == "__main__":
    main()
