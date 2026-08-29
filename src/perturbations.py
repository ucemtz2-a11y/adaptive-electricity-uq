# Module purpose: Implement perturbations, prediction replay, and degradation summaries.

"""Computation-only helpers for stochastic-feature perturbation experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate
from src.functional_pipeline import fit_quantile_model, make_raw_result, run_final_models


PERTURBED_FEATURES = [
    "load_lag_24",
    "wind_lag_24",
    "solar_lag_24",
    "residual_load_lag_24",
]


PRIMITIVE_FEATURES = ["load_lag_24", "wind_lag_24", "solar_lag_24"]


# Load selected parameters.
def load_selected_parameters(
    v10_results: Path,
    market_prefix: str,
) -> dict:
    path = (
        v10_results / "diagnostics" / f"{market_prefix}_selected_hyperparameters.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Frozen v10 hyperparameters were not found:\n"
            f"{path}\n"
            "Run v10 first or pass --v10-results."
        )

    return json.loads(path.read_text(encoding="utf-8"))


# Fit base models.
def fit_base_models(
    df: pd.DataFrame,
    target_col: str,
    model_features: list[str],
    train_slice: slice,
    alpha: float,
    random_state: int,
):
    x = df[model_features]
    y = df[target_col]

    lower_model = fit_quantile_model(
        x.iloc[train_slice],
        y.iloc[train_slice],
        alpha / 2.0,
        random_state,
    )

    median_model = fit_quantile_model(
        x.iloc[train_slice],
        y.iloc[train_slice],
        0.50,
        random_state + 1,
    )

    upper_model = fit_quantile_model(
        x.iloc[train_slice],
        y.iloc[train_slice],
        1.0 - alpha / 2.0,
        random_state + 2,
    )

    return lower_model, median_model, upper_model


# Make clean predictions.
def make_clean_predictions(
    df: pd.DataFrame,
    target_col: str,
    model_features: list[str],
    lower_model,
    median_model,
    upper_model,
) -> pd.DataFrame:
    x = df[model_features]

    predictions = pd.DataFrame(index=df.index)
    predictions["y_true"] = df[target_col].to_numpy()
    predictions["lower_raw"] = lower_model.predict(x)
    predictions["median_raw"] = median_model.predict(x)
    predictions["upper_raw"] = upper_model.predict(x)

    crossing = (
        predictions["lower_raw"] > predictions["upper_raw"]
    )

    if crossing.any():
        lower_old = predictions.loc[crossing, "lower_raw"].copy()

        upper_old = predictions.loc[crossing, "upper_raw"].copy()

        predictions.loc[
            crossing, "lower_raw",
        ] = np.minimum(lower_old, upper_old)

        predictions.loc[
            crossing, "upper_raw",
        ] = np.maximum(lower_old, upper_old)

    return predictions


# Draw noise.
def draw_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    noise: str,
    clip_z: float,
) -> np.ndarray:
    if noise == "gaussian":
        epsilon = rng.normal(loc=0.0, scale=1.0, size=shape)

        if clip_z > 0:
            epsilon = np.clip(epsilon, -clip_z, clip_z)

        return epsilon

    return rng.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=shape)


# Perturb test features.
def perturb_test_features(
    x_test: pd.DataFrame,
    train_std: pd.Series,
    rho: float,
    rng: np.random.Generator,
    mode: str,
    noise: str,
    clip_z: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    perturbed = x_test.copy()

    if rho == 0.0:
        delta = np.zeros((len(x_test), len(PERTURBED_FEATURES)), dtype=float)
    elif mode == "coherent":
        epsilon = draw_noise(
            rng=rng,
            shape=(len(x_test), len(PRIMITIVE_FEATURES)),
            noise=noise,
            clip_z=clip_z,
        )

        for column_index, column in enumerate(PRIMITIVE_FEATURES):
            scale = float(max(train_std[column], 1e-12))

            perturbed[column] = (
                x_test[column].to_numpy() + rho * scale * epsilon[:, column_index]
            )

            # Enforce nonnegative physical bounds for load, wind, and solar generation.
            perturbed[column] = np.maximum(perturbed[column].to_numpy(), 0.0)

        perturbed[
            "residual_load_lag_24"
        ] = (
            perturbed["load_lag_24"] - perturbed["wind_lag_24"]
            - perturbed["solar_lag_24"]
        )

        delta = np.column_stack(
            [
                perturbed[column].to_numpy() - x_test[column].to_numpy()
                for column in PERTURBED_FEATURES
            ]
        )

    else:
        epsilon = draw_noise(
            rng=rng,
            shape=(len(x_test), len(PERTURBED_FEATURES)),
            noise=noise,
            clip_z=clip_z,
        )

        for column_index, column in enumerate(PERTURBED_FEATURES):
            scale = float(max(train_std[column], 1e-12))

            perturbed[column] = (
                x_test[column].to_numpy() + rho * scale * epsilon[:, column_index]
            )

            if column in PRIMITIVE_FEATURES:
                perturbed[column] = np.maximum(perturbed[column].to_numpy(), 0.0)

        delta = np.column_stack(
            [
                perturbed[column].to_numpy() - x_test[column].to_numpy()
                for column in PERTURBED_FEATURES
            ]
        )

    scales = np.array(
        [max(float(train_std[column]), 1e-12) for column in PERTURBED_FEATURES],
        dtype=float,
    )

    standardized_delta = delta / scales[None, :]

    standardized_l2 = np.linalg.norm(standardized_delta, axis=1)

    raw_l2 = np.linalg.norm(delta, axis=1)

    diagnostics = {
        "mean_standardized_l2": float(standardized_l2.mean()),
        "median_standardized_l2": float(np.median(standardized_l2)),
        "max_standardized_l2": float(standardized_l2.max()),
        "cumulative_standardized_l2": float(standardized_l2.sum()),
        "mean_raw_l2": float(raw_l2.mean()),
        "max_raw_l2": float(raw_l2.max()),
    }

    return perturbed, diagnostics


# Replace test predictions.
def replace_test_predictions(
    clean_predictions: pd.DataFrame,
    test_slice: slice,
    x_test_perturbed: pd.DataFrame,
    lower_model,
    median_model,
    upper_model,
) -> pd.DataFrame:
    predictions = clean_predictions.copy()

    lower_test = lower_model.predict(x_test_perturbed)
    median_test = median_model.predict(x_test_perturbed)
    upper_test = upper_model.predict(x_test_perturbed)

    crossing = lower_test > upper_test

    if np.any(crossing):
        old_lower = lower_test.copy()
        old_upper = upper_test.copy()

        lower_test[crossing] = np.minimum(old_lower[crossing], old_upper[crossing])

        upper_test[crossing] = np.maximum(old_lower[crossing], old_upper[crossing])

    predictions.iloc[
        test_slice, predictions.columns.get_loc("lower_raw"),
    ] = lower_test

    predictions.iloc[
        test_slice, predictions.columns.get_loc("median_raw"),
    ] = median_test

    predictions.iloc[
        test_slice, predictions.columns.get_loc("upper_raw"),
    ] = upper_test

    return predictions


# Evaluate perturbed path.
def evaluate_perturbed_path(
    market: str,
    selected: dict,
    alpha: float,
    market_seed: int,
    train_context: np.ndarray,
    validation_context: np.ndarray,
    test_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    y_test: np.ndarray,
    raw_test_predictions: pd.DataFrame,
    evaluation_map_test: np.ndarray,
    test_group_masks: dict[str, np.ndarray],
) -> list[dict]:
    calibrated_results = run_final_models(
        selected=selected,
        alpha=alpha,
        max_adjustment=float(selected["max_adjustment"]),
        random_state=market_seed,
        train_context=train_context,
        validation_context=validation_context,
        test_context=test_context,
        lower_validation=lower_validation,
        upper_validation=upper_validation,
        y_validation=y_validation,
        lower_test=lower_test,
        upper_test=upper_test,
        y_test=y_test,
    )

    method_results = {
        "Raw quantile": make_raw_result(raw_test_predictions),
        **calibrated_results,
    }

    rows = []

    for method_name, result in method_results.items():
        metrics = evaluate(
            method_name=method_name,
            y=y_test,
            result=result,
            evaluation_map=evaluation_map_test,
            group_masks=test_group_masks,
            alpha=alpha,
        )

        metrics["market"] = market
        rows.append(metrics)

    return rows


# Summarise seed results.
def summarise_seed_results(
    seed_results: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "coverage",
        "coverage_error",
        "avg_width",
        "median_width",
        "winkler",
        "functional_error",
        "worst_group_error",
        "mean_adjustment",
        "mean_abs_linear_component",
        "mean_abs_functional_component",
    ]

    grouped = (
        seed_results.groupby(
            ["market", "rho", "method", "perturbation_mode", "noise"]
        )[metric_columns]
        .agg(
            ["mean", "std", "count"]
        )
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    summary = grouped.reset_index()

    for metric in metric_columns:
        summary[
            f"{metric}_se"
        ] = (
            summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
        )

    return summary.sort_values(["market", "rho", "method"])


# Compute degradation.
def compute_degradation(
    seed_results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    baseline = (
        seed_results[seed_results["rho"] == 0.0][["market", "seed", "method", *metrics]]
        .rename(
            columns={metric: f"{metric}_rho0" for metric in metrics}
        )
    )

    merged = seed_results.merge(baseline, on=["market", "seed", "method"], how="left")

    for metric in metrics:
        merged[
            f"delta_{metric}"
        ] = (
            merged[metric] - merged[f"{metric}_rho0"]
        )

    columns = [
        "market",
        "seed",
        "rho",
        "method",
        "perturbation_mode",
        "noise",
        *[f"delta_{metric}" for metric in metrics],
    ]

    return merged[columns]


# Aggregate degradation.
def aggregate_degradation(
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    delta_columns = [
        column
        for column in degradation.columns
        if column.startswith("delta_")
    ]

    grouped = (
        degradation.groupby(
            ["market", "rho", "method", "perturbation_mode", "noise"]
        )[delta_columns]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    return grouped.reset_index()


# Make cross market summary.
def make_cross_market_summary(
    seed_results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "coverage",
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    market_seed_means = (
        seed_results.groupby(["market", "rho", "method"])[metrics].mean().reset_index()
    )

    grouped = (
        market_seed_means.groupby(["rho", "method"])[metrics]
        .agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    summary = grouped.reset_index()

    for metric in metrics:
        summary[
            f"{metric}_se"
        ] = (
            summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
        )

    return summary.sort_values(["rho", "functional_error_mean"])
