# Module purpose: Load frozen v10 artefacts and verify exact raw-prediction reproduction.

"""Frozen v10 artefact loading and exact-reproduction checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Load v10 selected parameters.
def load_v10_selected_parameters(
    v10_results: Path,
    market_prefix: str,
) -> dict:
    path = (
        v10_results / "diagnostics" / f"{market_prefix}_selected_hyperparameters.json"
    )

    if not path.exists():
        raise FileNotFoundError("Missing frozen v10 hyperparameters:\n" f"{path}")

    return json.loads(path.read_text(encoding="utf-8"))


# Load v10 test predictions.
def load_v10_test_predictions(
    v10_results: Path,
    market_prefix: str,
) -> pd.DataFrame:
    path = (
        v10_results / "tables" / f"{market_prefix}_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError("Missing v10 test predictions:\n" f"{path}")

    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    df = (
        df.dropna(subset=["datetime"]).sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .set_index("datetime")
    )

    required = ["y_true", "lower_raw_quantile", "upper_raw_quantile"]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")

    return df


# Assert raw prediction match.
def assert_raw_prediction_match(
    generated_predictions: pd.DataFrame,
    test_slice: slice,
    stored_v10: pd.DataFrame,
    market: str,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    # Train base quantile models and generate uncalibrated prediction intervals.
    generated = (
        generated_predictions.iloc[test_slice][["y_true", "lower_raw", "upper_raw"]]
        .copy()
    )

    generated.index = pd.to_datetime(generated.index, utc=True)

    merged = generated.join(
        stored_v10[["y_true", "lower_raw_quantile", "upper_raw_quantile"]],
        how="inner",
        lsuffix="_generated",
        rsuffix="_stored",
    )

    if len(merged) != len(stored_v10):
        raise RuntimeError(
            f"{market}: generated and stored v10 timestamps do not align. "
            f"Common={len(merged)}, stored={len(stored_v10)}."
        )

    target_diff = np.abs(merged["y_true_generated"] - merged["y_true_stored"])

    lower_diff = np.abs(merged["lower_raw"] - merged["lower_raw_quantile"])

    upper_diff = np.abs(merged["upper_raw"] - merged["upper_raw_quantile"])

    diagnostics = {
        "target_max_abs_difference": float(target_diff.max()),
        "lower_max_abs_difference": float(lower_diff.max()),
        "upper_max_abs_difference": float(upper_diff.max()),
        "lower_mean_abs_difference": float(lower_diff.mean()),
        "upper_mean_abs_difference": float(upper_diff.mean()),
    }

    if (
        diagnostics["target_max_abs_difference"] > tolerance
        or diagnostics["lower_max_abs_difference"] > tolerance
        or diagnostics["upper_max_abs_difference"] > tolerance
    ):
        raise RuntimeError(
            f"{market}: regenerated raw intervals do not exactly match v10.\n"
            f"Diagnostics: {diagnostics}\n"
            "Do not merge the results until the v10 model configuration "
            "and random seed are reproduced exactly."
        )

    return diagnostics


# Interval result.
def interval_result(
    y_true: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lower_array = np.asarray(lower, dtype=float).reshape(-1)
    upper_array = np.asarray(upper, dtype=float).reshape(-1)

    if not (len(y) == len(lower_array) == len(upper_array)):
        raise ValueError("y_true, lower and upper must have equal length.")

    miscoverage = ((y < lower_array) | (y > upper_array)).astype(int)

    zeros = np.zeros(len(y), dtype=float)

    return {
        "lower": lower_array,
        "upper": upper_array,
        "adjustment": zeros.copy(),
        "miscoverage": miscoverage,
        "global_component": zeros.copy(),
        "linear_component": zeros.copy(),
        "functional_component": zeros.copy(),
    }
