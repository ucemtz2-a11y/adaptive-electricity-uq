# Turn the prepared summary tables into dissertation figures and short text files.

"""Figures and text outputs for the final paper summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_summary_data import (
    CORE_METHODS,
    METHOD_ORDER,
    SCENARIO_ORDER,
    SYNTHETIC_METHODS,
    categorical_sort,
)


# Draw the repeated grouped-bar layout used by several comparison figures.
def grouped_bar(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    ylabel: str,
    title: str,
    path: Path,
    method_order: list[str] | None = None,
) -> None:
    pivot = data.pivot(index=index, columns=columns, values=values)

    if method_order is not None:
        keep = [method for method in method_order if method in pivot.columns]
        pivot = pivot[keep]

    ax = pivot.plot(kind="bar", figsize=(12, 5))

    ax.set_xlabel(index.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        title=columns.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


# Create the seven final figures from already prepared result tables.
def make_figures(
    synthetic: pd.DataFrame,
    v10_results: pd.DataFrame,
    perturbation: pd.DataFrame,
    context_ablation: pd.DataFrame,
    kernel_ablation: pd.DataFrame,
    unified_results: pd.DataFrame,
    unified_average: pd.DataFrame,
    figures_dir: Path,
) -> None:
    # Make sure the destination exists before any plot is saved.
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Keep the same method order in every figure so colours and positions are consistent.
    average = categorical_sort(unified_average, "method", METHOD_ORDER)

    plt.figure(figsize=(12, 5))
    plt.bar(average["method"].astype(str), average["functional_error_mean"])
    plt.xlabel("Method")
    plt.ylabel("Mean functional coverage error")
    plt.title("Unified strong-baseline comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    # Figure 1 gives the overall functional-error comparison across all markets.
    plt.savefig(figures_dir / "fig_1_unified_functional_error.png", dpi=240)
    plt.close()

    # Figure 2 keeps market differences visible instead of averaging them away.
    market_methods = [
        "Raw quantile",
        "Scalar ACI",
        "Adaptive conformal score",
        "Linear contextual ACI",
        "Hybrid Functional ACI",
    ]

    grouped_bar(
        unified_results[unified_results["method"].isin(market_methods)],
        index="market",
        columns="method",
        values="functional_error",
        ylabel="Functional coverage error",
        title=(
            "State-dependent calibration across markets"
        ),
        path=(
            figures_dir / "fig_2_market_functional_error.png"
        ),
        method_order=market_methods,
    )

    # Figure 3 puts the four controlled synthetic scenarios side by side.
    if (
        "scenario" in synthetic.columns and "functional_error_mean" in synthetic.columns
    ):
        synthetic_plot = synthetic[synthetic["method"].isin(SYNTHETIC_METHODS)].copy()

        synthetic_plot["scenario"] = (
            pd.Categorical(
                synthetic_plot["scenario"],
                categories=SCENARIO_ORDER,
                ordered=True,
            )
        )

        synthetic_plot = synthetic_plot.sort_values("scenario")

        grouped_bar(
            synthetic_plot,
            index="scenario",
            columns="method",
            values="functional_error_mean",
            ylabel=(
                "Mean functional coverage error"
            ),
            title=(
                "Synthetic linear, nonlinear and drift scenarios"
            ),
            path=(
                figures_dir / "fig_3_synthetic_functional_error.png"
            ),
            method_order=SYNTHETIC_METHODS,
        )

    plt.figure(figsize=(11, 5))

    # Figure 4 draws one line per method as perturbation strength increases.
    for method in CORE_METHODS:
        part = (
            perturbation[perturbation["method"] == method].sort_values("rho")
        )

        if part.empty:
            continue

        plt.plot(part["rho"], part["functional_error_mean"], marker="o", label=method)

    plt.xlabel(r"Perturbation intensity $\rho$")
    plt.ylabel("Mean functional coverage error")
    plt.title("Robustness to stochastic feature perturbations")
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_4_perturbation_functional_error.png", dpi=240)
    plt.close()

    # Figure 5 shows which removed context group changes functional error most.
    grouped_bar(
        context_ablation[
            context_ablation["method"].isin(
                ["Linear contextual ACI", "Hybrid Functional ACI"]
            )
        ],
        index="ablation",
        columns="method",
        values="functional_error_mean",
        ylabel=(
            "Mean functional coverage error"
        ),
        title="Context-variable ablation",
        path=(
            figures_dir / "fig_5_context_ablation.png"
        ),
        method_order=["Linear contextual ACI", "Hybrid Functional ACI"],
    )

    # Figure 6 plots the direct hybrid-minus-linear gain for each market.
    hybrid_market = (
        v10_results[v10_results["method"] == "Hybrid Functional ACI"][
            ["market", "functional_error", "mean_abs_functional_component"]
        ]
        .rename(
            columns={"functional_error": ("hybrid_functional_error")}
        )
    )

    linear_market = (
        v10_results[v10_results["method"] == "Linear contextual ACI"][
            ["market", "functional_error"]
        ]
        .rename(
            columns={"functional_error": ("linear_functional_error")}
        )
    )

    gain = hybrid_market.merge(linear_market, on="market", how="inner")

    if not gain.empty:
        gain["hybrid_gain"] = (
            gain["linear_functional_error"] - gain["hybrid_functional_error"]
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(gain["mean_abs_functional_component"], gain["hybrid_gain"], s=70)

        for _, row in gain.iterrows():
            plt.annotate(
                str(row["market"]),
                (row["mean_abs_functional_component"], row["hybrid_gain"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("Mean absolute nonlinear component")
        plt.ylabel("Linear error minus Hybrid error")
        plt.title("Market calibration complexity and Hybrid gain")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_6_market_nonlinearity_gain.png", dpi=240)
        plt.close()

    # Figure 7 shows the accuracy/runtime trade-off for random-feature settings.
    plt.figure(figsize=(9, 5))
    plt.scatter(
        kernel_ablation["runtime_seconds_mean"],
        kernel_ablation["functional_error_mean"],
        s=70,
    )

    # Label each kernel point with its feature count to make the trade-off readable.
    for _, row in kernel_ablation.iterrows():
        plt.annotate(
            (
                f"D={int(row['n_components'])}, "
                f"l={row['length_scale']:g}"
            ),
            (row["runtime_seconds_mean"], row["functional_error_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Mean runtime (seconds)")
    plt.ylabel("Mean functional coverage error")
    plt.title("Kernel approximation accuracy-runtime trade-off")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_7_kernel_accuracy_runtime.png", dpi=240)
    plt.close()


# Fill a short LaTeX results section using values calculated from the tables.
def write_results_section(
    path: Path,
    findings: dict,
) -> None:
    unified = findings.get("unified_baselines", {})
    market = findings.get("market_heterogeneity", {})
    perturbation = findings.get("perturbation", {})
    context = findings.get("context_ablation", {})
    synthetic = findings.get("synthetic_nonlinear", {})

    hybrid_linear = unified.get("hybrid_vs_linear_functional_reduction_percent", np.nan)
    hybrid_aci = unified.get("hybrid_vs_aci_functional_reduction_percent", np.nan)
    hybrid_aci_worst = unified.get(
        "hybrid_vs_aci_worst_group_reduction_percent",
        np.nan,
    )

    nonlinear_market = market.get(
        "largest_nonlinear_market",
        "the market with the largest nonlinear component",
    )
    nonlinear_component = market.get("largest_mean_abs_functional_component", np.nan)

    perturbation_changes = perturbation.get(
        "functional_error_relative_change_percent",
        {},
    )
    hybrid_perturbation = (
        perturbation_changes.get("Hybrid Functional ACI", np.nan)
    )

    context_changes = context.get("no_past_miscoverage_increase_percent", {})
    hybrid_context = context_changes.get("Hybrid Functional ACI", np.nan)

    synthetic_reduction = synthetic.get("hybrid_vs_linear_reduction_percent", np.nan)

    text = rf"""
\section{{Experimental Results}}
\label{{sec:experimental-results}}

\subsection{{Synthetic evidence}}
The synthetic experiments separate linear state dependence, nonlinear state
dependence, abrupt distribution shifts, and gradually changing stochastic
features. In the nonlinear scenario, Hybrid Functional ACI reduces the
functional coverage error by approximately
\({synthetic_reduction:.1f}\%\) relative to linear contextual calibration.
In the linear scenario, the hybrid method remains close to its linear special
case, showing that the residual nonlinear component need not dominate when
the calibration structure is approximately linear.

\subsection{{Multi-market evaluation}}
Across DE--LU, DK1, DK2, and SE3, Hybrid Functional ACI achieves the lowest
average functional coverage error. Relative to linear contextual calibration,
the reduction is approximately \({hybrid_linear:.1f}\%\). The largest learned
nonlinear component occurs in {nonlinear_market}, where its mean absolute
magnitude is approximately \({nonlinear_component:.3f}\). This supports the
interpretation that electricity markets exhibit heterogeneous calibration
complexity.

\subsection{{Comparison with strong baselines}}
All strong baselines are re-evaluated using exactly the same raw quantile
predictions, chronological split, test observations, functional witness map,
and state groups. Relative to adaptive conformal score calibration, the hybrid
method lowers average functional coverage error by approximately
\({hybrid_aci:.1f}\%\) and worst-group coverage error by approximately
\({hybrid_aci_worst:.1f}\%\). Adaptive conformal score calibration obtains a
slightly lower average Winkler score, whereas Hybrid Functional ACI provides
coverage closer to the nominal level, narrower average intervals, and better
state-dependent calibration.

\subsection{{Stochastic-feature robustness}}
Under increasing perturbations of lagged load, wind, and solar features, all
methods incur wider intervals and higher Winkler scores. The functional
coverage error of Hybrid Functional ACI changes by only
\({hybrid_perturbation:.1f}\%\) between the unperturbed setting and the largest
reported perturbation level. This is interpreted as finite-noise stability,
not as evidence that the cumulative perturbation budget is sublinear.

\subsection{{Ablation results}}
Removing historical miscoverage from the context increases the hybrid
functional coverage error by approximately \({hybrid_context:.1f}\%\), making
it the most influential state variable in the reported ablation. Rolling
volatility is the second most important state descriptor. Function-space
ablations show that the linear-plus-residual-RBF representation outperforms
scalar, linear-only, and pure-RBF alternatives on average. Kernel ablations
indicate that the useful nonlinear structure is smooth and can be represented
with a moderate number of random Fourier features.

\subsection{{Overall interpretation}}
The results support three conclusions. First, marginal long-run coverage alone
does not prevent persistent state-dependent errors. Second, linear contextual
calibration captures a substantial share of the available structure. Third, a
residual nonlinear function space provides additional gains when calibration
complexity is genuinely nonlinear, while remaining close to the linear model
where the nonlinear signal is weak.
"""

    # The text is generated so quoted percentages stay in sync with the CSV files.
    path.write_text(text.strip() + "\n", encoding="utf-8")


# Add a small guide beside the generated v14 tables and figures.
def write_readme(
    path: Path,
) -> None:
    path.write_text(
        """# v14 Final Summary

This directory is generated from the frozen v9-v13 outputs.

## Tables

1. Synthetic scenario summary.
2. Multi-market method comparison.
3. Unified strong-baseline comparison.
4. Stochastic-feature perturbation summary.
5. Context ablation.
6. Function-space ablation.
7. Kernel approximation ablation.
8. Automatically derived headline findings.

Matching LaTeX tables are stored in `latex/`.

## Interpretation rule

Average width must not be ranked without considering coverage. Raw intervals
can be narrow because they under-cover. Marginal coverage, functional error,
worst-group error, width, and Winkler score should be reported together.
""",
        encoding="utf-8",
    )
