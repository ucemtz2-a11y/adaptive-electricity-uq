# Module purpose: Aggregate ablation results and plot context, function-space, and kernel comparisons.

"""Aggregation and plotting helpers for Functional ACI ablations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FUNCTION_SPACE_ORDER = [
    "Raw quantile",
    "Scalar ACI",
    "Linear contextual ACI",
    "Functional ACI",
    "Hybrid Functional ACI",
]


# Aggregate context ablation.
def aggregate_context_ablation(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "coverage",
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
        "mean_adjustment",
        "mean_abs_linear_component",
        "mean_abs_functional_component",
    ]

    grouped = (
        results.groupby(["ablation", "method"])[metrics].agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    summary = grouped.reset_index()

    for metric in metrics:
        summary[
            f"{metric}_se"
        ] = (
            summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
        )

    return summary.sort_values(["method", "functional_error_mean"])


# Add context degradation.
def add_context_degradation(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    full = (
        results[results["ablation"] == "full"][["market", "method", *metrics]]
        .rename(
            columns={metric: f"{metric}_full" for metric in metrics}
        )
    )

    merged = results.merge(full, on=["market", "method"], how="left")

    for metric in metrics:
        merged[
            f"delta_{metric}"
        ] = (
            merged[metric] - merged[f"{metric}_full"]
        )

    return merged


# Build function space tables.
def build_function_space_tables(
    v10_results: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = (
        v10_results / "tables" / "multi_market_results.csv"
    )

    if not source.exists():
        raise FileNotFoundError(f"v10 result table not found: {source}")

    # Load inputs or existing results and normalize them for downstream processing.
    results = pd.read_csv(source)

    function_space_map = {
        "Raw quantile": (
            "No online calibration"
        ),
        "Scalar ACI": (
            "Global scalar"
        ),
        "Linear contextual ACI": (
            "Linear context space"
        ),
        "Functional ACI": (
            "Centred RBF space"
        ),
        "Hybrid Functional ACI": (
            "Linear + residual RBF"
        ),
    }

    results["function_space"] = (
        results["method"].map(function_space_map)
    )

    results["method_order"] = (
        results["method"].map(
            {method: index for index, method in enumerate(FUNCTION_SPACE_ORDER)}
        )
    )

    results = results.sort_values(["market", "method_order"])

    # Compute coverage, interval-efficiency, and conditional-coverage metrics.
    metrics = [
        "coverage",
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    grouped = (
        results.groupby(["method", "function_space"])[metrics]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    # Aggregate results across markets, scenarios, or seeds for comparison.
    average = grouped.reset_index()

    pivot = results.pivot(index="market", columns="method", values=metrics)

    pairwise_rows = []

    comparisons = [
        ("Hybrid Functional ACI", "Linear contextual ACI"),
        ("Hybrid Functional ACI", "Functional ACI"),
        ("Hybrid Functional ACI", "Scalar ACI"),
        ("Linear contextual ACI", "Scalar ACI"),
    ]

    for first, second in comparisons:
        for market in results["market"].unique():
            first_row = results[
                (results["market"] == market) & (results["method"] == first)
            ].iloc[0]

            second_row = results[
                (results["market"] == market) & (results["method"] == second)
            ].iloc[0]

            row = {"market": market, "first_method": first, "second_method": second}

            for metric in metrics:
                row[
                    f"delta_{metric}"
                ] = (
                    first_row[metric] - second_row[metric]
                )

            pairwise_rows.append(row)

    pairwise = pd.DataFrame(pairwise_rows)

    return results, average, pairwise


# Aggregate kernel ablation.
def aggregate_kernel_ablation(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "coverage",
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
        "runtime_seconds",
        "mean_abs_linear_component",
        "mean_abs_functional_component",
    ]

    grouped = (
        results.groupby(["n_components", "length_scale"])[metrics]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    return (
        grouped.reset_index()
        .sort_values(
            ["functional_error_mean", "runtime_seconds_mean"]
        )
    )


# Plot context ablation.
def plot_context_ablation(
    summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in [
        (
            "functional_error_mean",
            "Mean functional coverage error",
            "context_ablation_functional_error.png",
        ),
        (
            "worst_group_error_mean",
            "Mean worst-group coverage error",
            "context_ablation_worst_group_error.png",
        ),
        ("winkler_mean", "Mean Winkler score", "context_ablation_winkler.png"),
    ]:
        pivot = summary.pivot(index="ablation", columns="method", values=metric)

        ax = pivot.plot(kind="bar", figsize=(12, 5))

        ax.set_xlabel("Context specification")

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " under context ablations")

        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")

        plt.xticks(rotation=25, ha="right")

        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=220)
        plt.close()


# Plot function space ablation.
def plot_function_space_ablation(
    average: pd.DataFrame,
    figures_dir: Path,
) -> None:
    ordered = (
        average.set_index("method").reindex(FUNCTION_SPACE_ORDER).reset_index()
    )

    for metric, ylabel, filename in [
        (
            "functional_error_mean",
            "Mean functional coverage error",
            "function_space_functional_error.png",
        ),
        (
            "worst_group_error_mean",
            "Mean worst-group coverage error",
            "function_space_worst_group_error.png",
        ),
        ("winkler_mean", "Mean Winkler score", "function_space_winkler.png"),
    ]:
        plt.figure(figsize=(11, 5))

        plt.bar(ordered["method"], ordered[metric])

        plt.xlabel("Function space")
        plt.ylabel(ylabel)
        plt.title(ylabel + " across function spaces")

        plt.xticks(rotation=25, ha="right")

        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=220)
        plt.close()


# Plot kernel ablation.
def plot_kernel_ablation(
    summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    for metric, ylabel, filename in [
        (
            "functional_error_mean",
            "Mean functional coverage error",
            "kernel_ablation_functional_error.png",
        ),
        ("winkler_mean", "Mean Winkler score", "kernel_ablation_winkler.png"),
        (
            "runtime_seconds_mean",
            "Mean runtime (seconds)",
            "kernel_ablation_runtime.png",
        ),
    ]:
        plt.figure(figsize=(10, 5))

        for length_scale in sorted(summary["length_scale"].unique()):
            part = (
                summary[summary["length_scale"] == length_scale]
                .sort_values(
                    "n_components"
                )
            )

            plt.plot(
                part["n_components"],
                part[metric],
                marker="o",
                label=(
                    f"length scale = "
                    f"{length_scale:g}"
                ),
            )

        plt.xlabel("Random Fourier components")

        plt.ylabel(ylabel)

        plt.title(ylabel + " for kernel approximation")

        plt.legend()
        plt.tight_layout()

        plt.savefig(figures_dir / filename, dpi=220)

        plt.close()
