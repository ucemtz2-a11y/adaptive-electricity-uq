# Module purpose: Generate perturbation diagnostics from standardized summary tables.

"""Plotting-only helpers for stochastic-feature perturbation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Plot metric.
def plot_metric(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 5))

    for method_name in table["method"].unique():
        part = (
            table[table["method"] == method_name].sort_values("rho")
        )

        plt.plot(part["rho"], part[f"{metric}_mean"], marker="o", label=method_name)

        lower = (
            part[f"{metric}_mean"] - 1.96 * part[f"{metric}_se"].fillna(0.0)
        )

        upper = (
            part[f"{metric}_mean"] + 1.96 * part[f"{metric}_se"].fillna(0.0)
        )

        plt.fill_between(part["rho"], lower, upper, alpha=0.12)

    plt.xlabel(r"Perturbation intensity $\rho$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} under stochastic-feature perturbations")
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


# Make figures.
def make_figures(
    cross_market_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        cross_market_summary,
        "coverage",
        "Coverage",
        figures_dir / "perturbation_coverage.png",
    )

    plot_metric(
        cross_market_summary,
        "coverage_error",
        "Absolute coverage error",
        figures_dir / "perturbation_coverage_error.png",
    )

    plot_metric(
        cross_market_summary,
        "functional_error",
        "Functional coverage error",
        figures_dir / "perturbation_functional_error.png",
    )

    plot_metric(
        cross_market_summary,
        "worst_group_error",
        "Worst-group coverage error",
        figures_dir / "perturbation_worst_group_error.png",
    )

    plot_metric(
        cross_market_summary,
        "winkler",
        "Winkler score",
        figures_dir / "perturbation_winkler.png",
    )

    plot_metric(
        cross_market_summary,
        "avg_width",
        "Average interval width",
        figures_dir / "perturbation_width.png",
    )
