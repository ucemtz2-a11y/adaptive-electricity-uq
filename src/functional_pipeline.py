# Keep the data preparation and model steps shared by the real-market experiments.

"""Shared, frozen helpers for the paper's Functional ACI experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.calibration.functional_aci import (
    CenteredRandomFourierFeatures,
    FunctionalACI,
    HybridFunctionalACI,
    LinearContextualACI,
    ScalarACI,
)
TARGET_CANDIDATES = ["price", "target", "y", "y_true", "day_ahead_price", "Price"]


DATETIME_CANDIDATES = ["datetime", "timestamp", "time", "date", "Unnamed: 0"]


MODEL_FEATURES = [
    "price_lag_1",
    "price_lag_24",
    "price_lag_168",
    "load_lag_24",
    "wind_lag_24",
    "solar_lag_24",
    "residual_load_lag_24",
    "hour",
    "weekday",
    "month",
]


CONTEXT_COLUMNS = [
    "raw_width",
    "rolling_price_std_24",
    "rolling_price_std_168",
    "rolling_raw_miscoverage_168",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "weekend",
]


# Match common CSV column names without making every dataset use identical labels.
def find_column(
    columns: Iterable[str],
    candidates: list[str],
    kind: str,
) -> str:
    columns = list(columns)
    lookup = {str(column).lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"Could not identify the {kind} column. "
        f"Available columns: {columns}"
    )


# Build model inputs from past prices and yesterday's electricity-system values.
def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    required = ["load", "wind", "solar"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Missing raw columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "residual_load" not in df.columns:
        df["residual_load"] = df["load"] - df["wind"] - df["solar"]

    # Every shift is positive, so these features cannot use the price being predicted.
    df["price_lag_1"] = df[target_col].shift(1)
    df["price_lag_24"] = df[target_col].shift(24)
    df["price_lag_168"] = df[target_col].shift(168)

    df["load_lag_24"] = df["load"].shift(24)
    df["wind_lag_24"] = df["wind"].shift(24)
    df["solar_lag_24"] = df["solar"].shift(24)
    df["residual_load_lag_24"] = df["residual_load"].shift(24)

    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month

    df = df.dropna(subset=[target_col, *MODEL_FEATURES])

    return df, MODEL_FEATURES.copy()


# Load one processed market, sort it by time, and create its modelling features.
def load_market_dataset(
    path: Path,
) -> tuple[pd.DataFrame, str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    datetime_col = find_column(df.columns, DATETIME_CANDIDATES, "datetime")
    target_col = find_column(df.columns, TARGET_CANDIDATES, "target")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)

    # Keeping the last duplicate gives one unambiguous observation per UTC timestamp.
    df = (
        df.dropna(subset=[datetime_col]).sort_values(datetime_col)
        .drop_duplicates(subset=[datetime_col], keep="last")
        .set_index(datetime_col)
    )

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[target_col])
    df, model_features = create_lag_features(df, target_col)
    df = df[[target_col, *model_features]].copy()

    if len(df) < 2000:
        raise ValueError(f"Only {len(df)} usable observations remain for {path}.")

    return df, target_col, model_features


# Build the standard processed filename for one market.
def market_dataset_path(data_dir: Path, market: str) -> Path:
    return Path(data_dir) / f"{market}_dataset.csv"


# Load a market by name using the standard directory layout.
def load_market_data(
    market: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, str, list[str]]:
    return load_market_dataset(market_dataset_path(data_dir, market))


# Try the saved learning rate and its nearby half/double values during a full run.
def scaled_candidates(value: float, quick: bool = False) -> list[float]:
    value = float(value)
    if quick:
        return [value]
    return sorted({max(0.0001, 0.5 * value), value, min(0.20, 2.0 * value)})


# Give coverage the largest weight while still considering conditional error and width.
def selection_objective(metrics: dict, raw_width: float) -> float:
    width_ratio = metrics["avg_width"] / max(raw_width, 1e-12)
    return float(
        3.0 * metrics["coverage_error"]
        + metrics["functional_error"]
        + 0.50 * metrics["worst_group_error"]
        + 0.02 * width_ratio
    )


# Return train, validation, and test slices without shuffling the time series.
def chronological_split(
    n: int,
    train_frac: float,
    validation_frac: float,
) -> tuple[slice, slice, slice]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie in (0, 1).")
    if not 0.0 < validation_frac < 1.0:
        raise ValueError("validation_frac must lie in (0, 1).")
    if train_frac + validation_frac >= 1.0:
        raise ValueError("train_frac + validation_frac must be below 1.")

    train_end = int(n * train_frac)
    validation_end = int(n * (train_frac + validation_frac))

    return (
        slice(0, train_end),
        slice(train_end, validation_end),
        slice(validation_end, n),
    )


# Fit one LightGBM model for a requested conditional quantile.
def fit_quantile_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
    random_state: int,
) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        colsample_bytree=0.9,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model


# Train the lower, median, and upper models on the training period only.
def make_raw_predictions(
    df: pd.DataFrame,
    target_col: str,
    features: list[str],
    train_slice: slice,
    alpha: float,
    random_state: int,
) -> pd.DataFrame:
    x = df[features]
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

    predictions = pd.DataFrame(index=df.index)
    predictions["y_true"] = y
    predictions["lower_raw"] = lower_model.predict(x)
    predictions["median_raw"] = median_model.predict(x)
    predictions["upper_raw"] = upper_model.predict(x)

    # Swap the rare crossed bounds so every saved interval has lower <= upper.
    crossing = predictions["lower_raw"] > predictions["upper_raw"]
    if crossing.any():
        old_lower = predictions.loc[crossing, "lower_raw"].copy()
        old_upper = predictions.loc[crossing, "upper_raw"].copy()
        predictions.loc[crossing, "lower_raw"] = np.minimum(old_lower, old_upper)
        predictions.loc[crossing, "upper_raw"] = np.maximum(old_lower, old_upper)

    return predictions


# Create the information used to change interval width during online calibration.
def build_context(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    index = predictions.index
    y = predictions["y_true"]
    lower = predictions["lower_raw"]
    upper = predictions["upper_raw"]

    raw_miss = ((y < lower) | (y > upper)).astype(float)

    context = pd.DataFrame(index=index)
    context["raw_width"] = upper - lower

    # shift(1) is important: rolling statistics stop before the current outcome.
    context["rolling_price_std_24"] = (
        y.rolling(24, min_periods=12).std().shift(1)
    )
    context["rolling_price_std_168"] = (
        y.rolling(168, min_periods=48).std().shift(1)
    )
    context["rolling_raw_miscoverage_168"] = (
        raw_miss.rolling(168, min_periods=48).mean().shift(1)
    )

    context["hour_sin"] = np.sin(2.0 * np.pi * index.hour / 24.0)
    context["hour_cos"] = np.cos(2.0 * np.pi * index.hour / 24.0)
    context["weekday_sin"] = np.sin(2.0 * np.pi * index.weekday / 7.0)
    context["weekday_cos"] = np.cos(2.0 * np.pi * index.weekday / 7.0)
    context["weekend"] = (index.weekday >= 5).astype(float)

    return context


# Learn missing-value replacements and scaling from the training rows only.
def preprocess_context(
    context: pd.DataFrame,
    train_slice: slice,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Validation and test rows reuse these training values to avoid data leakage.
    train_imputed = imputer.fit_transform(context.iloc[train_slice][CONTEXT_COLUMNS])
    scaler.fit(train_imputed)

    all_imputed = imputer.transform(context[CONTEXT_COLUMNS])
    return scaler.transform(all_imputed)


# Convert a calibrator output object to the dictionary used by evaluation code.
def convert_result(output) -> dict[str, np.ndarray]:
    return {
        "lower": output.lower,
        "upper": output.upper,
        "adjustment": output.adjustment,
        "miscoverage": output.miscoverage,
        "global_component": output.global_component,
        "linear_component": output.linear_component,
        "functional_component": output.functional_component,
    }


# Put uncalibrated quantile intervals in the same format as calibrated methods.
def make_raw_result(
    predictions: pd.DataFrame,
) -> dict[str, np.ndarray]:
    y = predictions["y_true"].to_numpy()
    lower = predictions["lower_raw"].to_numpy()
    upper = predictions["upper_raw"].to_numpy()

    miscoverage = ((y < lower) | (y > upper)).astype(int)
    zeros = np.zeros(len(y), dtype=float)

    return {
        "lower": lower,
        "upper": upper,
        "adjustment": zeros.copy(),
        "miscoverage": miscoverage,
        "global_component": zeros.copy(),
        "linear_component": zeros.copy(),
        "functional_component": zeros.copy(),
    }


# Fit a separate feature map used only to measure functional coverage error.
def fit_evaluation_map(
    train_context: np.ndarray,
    random_state: int,
) -> CenteredRandomFourierFeatures:
    feature_map = CenteredRandomFourierFeatures(
        n_components=256,
        length_scale=2.0,
        random_state=random_state,
    )
    feature_map.fit(train_context)
    return feature_map


# Combine scaled linear context with nonlinear features for the final metric.
def transform_evaluation_map(
    feature_map: CenteredRandomFourierFeatures,
    context: np.ndarray,
) -> np.ndarray:
    nonlinear_features = feature_map.transform(context)
    linear_features = context / np.sqrt(context.shape[1])

    return np.column_stack([linear_features, nonlinear_features])


# Define readable test groups using thresholds learned from training context.
def create_group_masks(
    index: pd.Index,
    context_part: pd.DataFrame,
    context_train: pd.DataFrame,
) -> dict[str, np.ndarray]:
    # Training medians keep group definitions independent of the test outcomes.
    volatility_cut = float(context_train["rolling_price_std_168"].median())
    width_cut = float(context_train["raw_width"].median())
    miscoverage_cut = float(context_train["rolling_raw_miscoverage_168"].median())

    return {
        "all": np.ones(len(index), dtype=bool),
        "high_volatility": (
            context_part["rolling_price_std_168"].to_numpy() >= volatility_cut
        ),
        "low_volatility": (
            context_part["rolling_price_std_168"].to_numpy() < volatility_cut
        ),
        "peak_hours": (
            (index.hour >= 8) & (index.hour <= 19)
        ),
        "off_peak_hours": ~((index.hour >= 8) & (index.hour <= 19)),
        "weekend": index.weekday >= 5,
        "weekday": index.weekday < 5,
        "wide_raw_interval": (
            context_part["raw_width"].to_numpy() >= width_cut
        ),
        "narrow_raw_interval": (
            context_part["raw_width"].to_numpy() < width_cut
        ),
        "high_past_miscoverage": (
            context_part["rolling_raw_miscoverage_168"].to_numpy() >= miscoverage_cut
        ),
        "low_past_miscoverage": (
            context_part["rolling_raw_miscoverage_168"].to_numpy() < miscoverage_cut
        ),
    }






# Calculate coverage and width separately for every method and test group.
def make_group_table(
    market: str,
    y: np.ndarray,
    method_results: dict,
    group_masks: dict[str, np.ndarray],
    target_coverage: float,
) -> pd.DataFrame:
    rows = []

    for method_name, result in method_results.items():
        for group_name, mask in group_masks.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue

            coverage = float(
                np.mean(
                    (y[mask] >= result["lower"][mask])
                    & (y[mask] <= result["upper"][mask])
                )
            )

            rows.append(
                {
                    "market": market,
                    "method": method_name,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "coverage": coverage,
                    "coverage_error": abs(coverage - target_coverage),
                    "avg_width": float(
                        np.mean(result["upper"][mask] - result["lower"][mask])
                    ),
                }
            )

    return pd.DataFrame(rows)


# Warm each selected model on validation data, then continue into the test period.
def run_final_models(
    selected: dict,
    alpha: float,
    max_adjustment: float,
    random_state: int,
    train_context: np.ndarray,
    validation_context: np.ndarray,
    test_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    scalar_model = ScalarACI(
        alpha=alpha,
        eta=selected["scalar"]["eta"],
        max_adjustment=max_adjustment,
    )
    # reset=False carries the learned validation state forward instead of restarting.
    scalar_model.run(lower_validation, upper_validation, y_validation, reset=True)
    scalar_test = scalar_model.run(lower_test, upper_test, y_test, reset=False)

    contextual_model = LinearContextualACI(
        alpha=alpha,
        eta_global=selected["contextual"]["eta_global"],
        eta_linear=selected["contextual"]["eta_linear"],
        linear_radius=selected["contextual"]["linear_radius"],
        max_adjustment=max_adjustment,
    )
    contextual_model.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )
    contextual_test = contextual_model.run(
        lower_test,
        upper_test,
        y_test,
        test_context,
        reset=False,
    )

    functional_model = FunctionalACI(
        alpha=alpha,
        eta_global=selected["functional"]["eta_global"],
        eta_functional=selected["functional"]["eta_functional"],
        n_components=selected["functional"]["n_components"],
        length_scale=selected["functional"]["length_scale"],
        functional_radius=selected["functional"]["functional_radius"],
        max_adjustment=max_adjustment,
        random_state=random_state,
    )
    functional_model.fit_feature_map(train_context)
    functional_model.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )
    functional_test = functional_model.run(
        lower_test,
        upper_test,
        y_test,
        test_context,
        reset=False,
    )

    hybrid_model = HybridFunctionalACI(
        alpha=alpha,
        eta_global=selected["hybrid"]["eta_global"],
        eta_linear=selected["hybrid"]["eta_linear"],
        eta_functional=selected["hybrid"]["eta_functional"],
        n_components=selected["hybrid"]["n_components"],
        length_scale=selected["hybrid"]["length_scale"],
        residual_ridge=selected["hybrid"]["residual_ridge"],
        linear_radius=selected["hybrid"]["linear_radius"],
        functional_radius=selected["hybrid"]["functional_radius"],
        max_adjustment=max_adjustment,
        random_state=random_state,
    )
    hybrid_model.fit_feature_map(train_context)
    hybrid_model.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )
    hybrid_test = hybrid_model.run(
        lower_test,
        upper_test,
        y_test,
        test_context,
        reset=False,
    )

    return {
        "Scalar ACI": convert_result(scalar_test),
        "Linear contextual ACI": convert_result(contextual_test),
        "Functional ACI": convert_result(functional_test),
        "Hybrid Functional ACI": convert_result(hybrid_test),
    }
