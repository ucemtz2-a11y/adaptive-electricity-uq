# Prepare the saved experiment tables used in the final dissertation summary.

"""Data preparation and headline findings for the final paper summary."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "Raw quantile",
    "Rolling historical quantile",
    "Split CQR",
    "Scalar ACI",
    "Adaptive conformal score",
    "Functional ACI",
    "Linear contextual ACI",
    "Hybrid Functional ACI",
]


CORE_METHODS = [
    "Raw quantile",
    "Scalar ACI",
    "Adaptive conformal score",
    "Functional ACI",
    "Linear contextual ACI",
    "Hybrid Functional ACI",
]


SYNTHETIC_METHODS = [
    "Raw quantile",
    "Scalar ACI",
    "Functional ACI",
    "Linear contextual ACI",
    "Hybrid Functional ACI",
]


SCENARIO_ORDER = ["linear", "nonlinear", "abrupt_drift", "gradual_stochastic"]


# Give a clear error when one of the earlier experiment files is missing.
def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found:\n{path}")
    return path


# Read a required CSV after checking that it exists.
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


# Add coverage error when an older table contains coverage but not the gap itself.
def ensure_coverage_error(
    df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    df = df.copy()
    target = 1.0 - alpha

    if (
        "coverage_error" not in df.columns and "coverage" in df.columns
    ):
        df["coverage_error"] = (df["coverage"] - target).abs()

    if (
        "coverage_error_mean" not in df.columns and "coverage_mean" in df.columns
    ):
        df["coverage_error_mean"] = (df["coverage_mean"] - target).abs()

    return df


# Apply the small amount of shared cleanup needed by final tables.
def prepare_table(
    df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    return ensure_coverage_error(df, alpha)


# Accept the few column names used by different saved synthetic result versions.
def find_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    lookup = {str(column).lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# Recalculate synthetic means and standard errors from the saved seed-level rows.
def derive_synthetic_summary(
    v9: Path,
    alpha: float,
) -> pd.DataFrame:
    summary_path = (
        v9 / "tables" / "synthetic_summary.csv"
    )

    if summary_path.exists():
        return prepare_table(pd.read_csv(summary_path), alpha)

    seed_results = prepare_table(
        read_csv(v9 / "tables" / "synthetic_seed_results.csv"),
        alpha,
    )

    scenario_col = find_column(seed_results.columns, ["scenario", "data_scenario"])

    if scenario_col is None:
        raise KeyError("No scenario column found in synthetic_seed_results.csv.")

    if scenario_col != "scenario":
        seed_results = seed_results.rename(columns={scenario_col: "scenario"})

    metrics = [
        column
        for column in [
            "coverage",
            "coverage_error",
            "avg_width",
            "winkler",
            "functional_error",
            "worst_group_error",
            "adaptation_delay",
        ]
        if column in seed_results.columns
    ]

    # Grouping by scenario and method keeps every seed as one repeated experiment.
    grouped = (
        seed_results.groupby(["scenario", "method"])[metrics]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    summary = grouped.reset_index()

    for metric in metrics:
        summary[f"{metric}_se"] = (
            summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
        )

    return summary


# Keep requested columns that are actually available in the source table.
def select_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


# Use the dissertation's method order instead of alphabetical order.
def categorical_sort(
    df: pd.DataFrame,
    column: str,
    order: list[str],
) -> pd.DataFrame:
    df = df.copy()

    if column in df.columns:
        df[column] = pd.Categorical(df[column], categories=order, ordered=True)
        df = df.sort_values(column)

    return df


# Write one prepared table as both CSV and a matching LaTeX table.
def save_table(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
) -> None:
    df.to_csv(csv_path, index=False)

    tex = df.to_latex(
        index=False,
        escape=True,
        float_format="%.4f",
        caption=caption,
        label=label,
    )

    tex_path.write_text(tex, encoding="utf-8")


# Look up one method/metric value while returning NaN for a missing row.
def get_value(
    table: pd.DataFrame,
    method: str,
    metric: str,
) -> float:
    part = table[table["method"] == method]

    if part.empty or metric not in part.columns:
        return np.nan

    return float(part.iloc[0][metric])


# Express how much smaller a new value is than its reference value.
def reduction_percent(
    baseline: float,
    improved: float,
) -> float:
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan

    return (
        100.0 * (baseline - improved) / baseline
    )


# Calculate the comparisons quoted in the final results section from saved numbers.
def derive_headline_findings(
    synthetic: pd.DataFrame,
    v10_results: pd.DataFrame,
    perturbation: pd.DataFrame,
    context_ablation: pd.DataFrame,
    unified_average: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    payload: dict = {}

    # Start with the main cross-market comparison between hybrid and baseline methods.
    hybrid_fe = get_value(
        unified_average,
        "Hybrid Functional ACI",
        "functional_error_mean",
    )
    linear_fe = get_value(
        unified_average,
        "Linear contextual ACI",
        "functional_error_mean",
    )
    aci_fe = get_value(
        unified_average,
        "Adaptive conformal score",
        "functional_error_mean",
    )

    hybrid_wg = get_value(
        unified_average,
        "Hybrid Functional ACI",
        "worst_group_error_mean",
    )
    aci_wg = get_value(
        unified_average,
        "Adaptive conformal score",
        "worst_group_error_mean",
    )

    hybrid_winkler = get_value(unified_average, "Hybrid Functional ACI", "winkler_mean")
    aci_winkler = get_value(unified_average, "Adaptive conformal score", "winkler_mean")

    # Keep related numbers together so the reporting code can use simple keys.
    unified_payload = {
        "hybrid_functional_error": hybrid_fe,
        "linear_functional_error": linear_fe,
        "adaptive_conformal_functional_error": aci_fe,
        "hybrid_worst_group_error": hybrid_wg,
        "adaptive_conformal_worst_group_error": aci_wg,
        "hybrid_winkler": hybrid_winkler,
        "adaptive_conformal_winkler": aci_winkler,
        "hybrid_vs_linear_functional_reduction_percent": (
            reduction_percent(linear_fe, hybrid_fe)
        ),
        "hybrid_vs_aci_functional_reduction_percent": (
            reduction_percent(aci_fe, hybrid_fe)
        ),
        "hybrid_vs_aci_worst_group_reduction_percent": (
            reduction_percent(aci_wg, hybrid_wg)
        ),
    }

    payload["unified_baselines"] = unified_payload

    rows.extend(
        [
            {
                "finding": (
                    "Hybrid vs Linear functional-error reduction"
                ),
                "value": unified_payload[
                    "hybrid_vs_linear_functional_reduction_percent"
                ],
                "unit": "percent",
            },
            {
                "finding": (
                    "Hybrid vs Adaptive conformal "
                    "functional-error reduction"
                ),
                "value": unified_payload["hybrid_vs_aci_functional_reduction_percent"],
                "unit": "percent",
            },
            {
                "finding": (
                    "Hybrid vs Adaptive conformal "
                    "worst-group-error reduction"
                ),
                "value": unified_payload["hybrid_vs_aci_worst_group_reduction_percent"],
                "unit": "percent",
            },
            {
                "finding": (
                    "Hybrid minus Adaptive conformal Winkler"
                ),
                "value": hybrid_winkler - aci_winkler,
                "unit": "score points",
            },
        ]
    )

    # Compare hybrid and linear methods market by market to show where nonlinearity helps.
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

    market_gain = hybrid_market.merge(linear_market, on="market", how="inner")

    if not market_gain.empty:
        market_gain["hybrid_gain"] = (
            market_gain["linear_functional_error"]
            - market_gain["hybrid_functional_error"]
        )

        largest = market_gain.sort_values(
            "mean_abs_functional_component",
            ascending=False,
        ).iloc[0]

        # This small correlation is descriptive only; it is not used for model selection.
        correlation = (
            market_gain[["mean_abs_functional_component", "hybrid_gain"]].corr()
            .iloc[0, 1]
            if len(market_gain) > 1
            else np.nan
        )

        payload["market_heterogeneity"] = {
            "largest_nonlinear_market": str(largest["market"]),
            "largest_mean_abs_functional_component": float(
                largest["mean_abs_functional_component"]
            ),
            "component_gain_correlation": float(correlation),
        }

        rows.append(
            {
                "finding": (
                    "Largest Hybrid nonlinear component"
                ),
                "value": float(largest["mean_abs_functional_component"]),
                "unit": str(largest["market"]),
            }
        )

    # Compare the clean path with the strongest saved perturbation for each method.
    if not perturbation.empty:
        maximum_rho = float(perturbation["rho"].max())

        changes = {}

        for method in CORE_METHODS:
            part = perturbation[perturbation["method"] == method]

            baseline = part[part["rho"] == 0.0]

            maximum = part[part["rho"] == maximum_rho]

            if baseline.empty or maximum.empty:
                continue

            start = float(baseline.iloc[0]["functional_error_mean"])

            end = float(maximum.iloc[0]["functional_error_mean"])

            change = (
                100.0 * (end - start) / start if start != 0 else np.nan
            )

            changes[method] = change

            rows.append(
                {
                    "finding": (
                        f"{method} functional-error change "
                        f"at rho={maximum_rho:g}"
                    ),
                    "value": change,
                    "unit": "percent",
                }
            )

        payload["perturbation"] = {
            "maximum_rho": maximum_rho,
            "functional_error_relative_change_percent": (
                changes
            ),
        }

    # Measure the effect of removing past miscoverage from the context vector.
    context_changes = {}

    for method in ["Hybrid Functional ACI", "Linear contextual ACI"]:
        full = context_ablation[
            (context_ablation["method"] == method)
            & (context_ablation["ablation"] == "full")
        ]

        removed = context_ablation[
            (context_ablation["method"] == method)
            & (
                context_ablation["ablation"] == "no_past_miscoverage"
            )
        ]

        if full.empty or removed.empty:
            continue

        full_value = float(full.iloc[0]["functional_error_mean"])

        removed_value = float(removed.iloc[0]["functional_error_mean"])

        increase = (
            100.0 * (removed_value - full_value) / full_value
        )

        context_changes[method] = increase

        rows.append(
            {
                "finding": (
                    f"{method} functional-error increase "
                    "without past miscoverage"
                ),
                "value": increase,
                "unit": "percent",
            }
        )

    payload["context_ablation"] = {
        "no_past_miscoverage_increase_percent": (
            context_changes
        )
    }

    # Finish with the nonlinear synthetic scenario where hybrid structure is most relevant.
    if (
        "scenario" in synthetic.columns and "functional_error_mean" in synthetic.columns
    ):
        nonlinear = synthetic[synthetic["scenario"].astype(str) == "nonlinear"]

        if not nonlinear.empty:
            hybrid = get_value(
                nonlinear,
                "Hybrid Functional ACI",
                "functional_error_mean",
            )
            linear = get_value(
                nonlinear,
                "Linear contextual ACI",
                "functional_error_mean",
            )
            scalar = get_value(nonlinear, "Scalar ACI", "functional_error_mean")

            payload["synthetic_nonlinear"] = {
                "hybrid_functional_error": hybrid,
                "linear_functional_error": linear,
                "scalar_functional_error": scalar,
                "hybrid_vs_linear_reduction_percent": (
                    reduction_percent(linear, hybrid)
                ),
                "hybrid_vs_scalar_reduction_percent": (
                    reduction_percent(scalar, hybrid)
                ),
            }

            rows.extend(
                [
                    {
                        "finding": (
                            "Synthetic nonlinear: Hybrid vs Linear "
                            "functional-error reduction"
                        ),
                        "value": reduction_percent(linear, hybrid),
                        "unit": "percent",
                    },
                    {
                        "finding": (
                            "Synthetic nonlinear: Hybrid vs Scalar "
                            "functional-error reduction"
                        ),
                        "value": reduction_percent(scalar, hybrid),
                        "unit": "percent",
                    },
                ]
            )

    return pd.DataFrame(rows), payload
