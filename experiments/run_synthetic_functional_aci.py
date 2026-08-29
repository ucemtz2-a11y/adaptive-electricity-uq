# Create simple simulated datasets where nonlinearity and drift are known in advance.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.functional_aci import (  # noqa: E402
    CenteredRandomFourierFeatures,
    FunctionalACI,
    HybridFunctionalACI,
    LinearContextualACI,
    ScalarACI,
)
from src.evaluation.metrics import (  # noqa: E402
    functional_coverage_error,
    interval_metrics,
)
from src.functional_pipeline import convert_result as convert  # noqa: E402

SCENARIOS = ("linear", "nonlinear", "abrupt_drift", "gradual_stochastic")
METHODS = (
    "Raw quantile",
    "Scalar ACI",
    "Linear contextual ACI",
    "Functional ACI",
    "Hybrid Functional ACI",
)


# Keep all arrays from one simulated scenario together so they cannot become misaligned.
@dataclass
class Data:
    y: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    context_raw: np.ndarray
    latent: np.ndarray
    sigma: np.ndarray
    drift_index: int | None


# Read scenario size, seeds, output folder, and the optional quick flag.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/versions/results_v9_synthetic_functional_aci",
    )
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.40)
    p.add_argument("--validation-frac", type=float, default=0.20)
    p.add_argument("--rolling-window", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


# Divide each simulated sequence into train, validation, and test slices in time order.
def splits(n: int, train_frac: float, val_frac: float) -> tuple[slice, slice, slice]:
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid chronological split.")
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


# Generate a smooth AR(1) feature whose current value depends on its previous value.
def ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    scale = np.sqrt(max(1.0 - rho**2, 1e-8))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + scale * e[t]
    return x


# Shift an array without wrapping future values back to the start.
def roll_shift(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    return (
        pd.Series(x).rolling(window, min_periods=min_periods).mean().shift(1).to_numpy()
    )


# Generate one controlled scenario and its deliberately imperfect raw intervals.
def generate(
    scenario: str,
    n: int,
    seed: int,
    train_end: int,
    alpha: float,
) -> Data:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    u = t / max(n - 1, 1)

    z1 = ar1(n, 0.92, rng)
    z2 = ar1(n, 0.75, rng)
    z3 = np.sin(2 * np.pi * t / 24)
    z4 = np.cos(2 * np.pi * t / 168)
    latent = np.column_stack([z1, z2, z3, z4])
    observed = latent.copy()
    drift_index: int | None = None

    if scenario == "linear":
        log_sigma = 0.05 + 0.38 * z1 + 0.28 * z2 + 0.18 * z3
    elif scenario == "nonlinear":
        log_sigma = (
            0.02 + 0.48 * np.sin(1.4 * z1) + 0.34 * (z2**2 - 1.0) + 0.30 * z1 * z3
            + 0.12 * np.cos(2.0 * z4)
        )
    elif scenario == "abrupt_drift":
        drift_index = int(0.80 * n)
        before = 0.02 + 0.42 * z1 + 0.25 * z2 + 0.16 * z3
        after = (
            0.15 - 0.25 * z1 + 0.48 * np.sin(1.7 * z2) + 0.35 * z1 * z3
        )
        log_sigma = np.where(t < drift_index, before, after)
    elif scenario == "gradual_stochastic":
        drift = np.column_stack(
            [1.2 * u, 0.6 * np.sin(2 * np.pi * u), np.zeros(n), np.zeros(n)]
        )
        latent = latent + drift
        feature_noise = 0.05 + 0.45 * u
        observed = latent + feature_noise[:, None] * rng.normal(size=latent.shape)
        log_sigma = (
            0.02 + 0.30 * latent[:, 0] + 0.30 * np.sin(1.5 * latent[:, 1])
            + 0.25 * latent[:, 0] * latent[:, 2]
            + 0.30 * u
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    sigma = np.exp(np.clip(log_sigma, -0.9, 0.9))
    mean = (
        0.55 * latent[:, 0] - 0.35 * latent[:, 1] + 0.25 * latent[:, 2]
        + 0.15 * latent[:, 3]
    )
    y = mean + sigma * rng.normal(size=n)

    quantile = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    base_sigma = float(np.median(sigma[:train_end]))
    half_width = quantile * base_sigma
    lower = mean - half_width
    upper = mean + half_width

    raw_miss = ((y < lower) | (y > upper)).astype(float)
    context_raw = np.column_stack(
        [
            observed,
            roll_shift(np.abs(y - mean), 50, 20),
            roll_shift(raw_miss, 100, 30),
            np.sin(2 * np.pi * t / 48),
            np.cos(2 * np.pi * t / 48),
        ]
    )
    return Data(y, lower, upper, context_raw, latent, sigma, drift_index)


# Fit context scaling on training rows and apply it to the full simulated path.
def preprocess(x: np.ndarray, train: slice) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(x[train])
    scaler.fit(train_imputed)
    return scaler.transform(imputer.transform(x))


# Build the fixed feature map used to measure functional coverage error.
def witness_map(train_x: np.ndarray, x: np.ndarray, seed: int) -> np.ndarray:
    rff = CenteredRandomFourierFeatures(
        n_components=256,
        length_scale=2.0,
        random_state=seed,
    )
    rff.fit(train_x)
    linear = x / np.sqrt(x.shape[1])
    return np.column_stack([linear, rff.transform(x)])


# Define simple test groups from context values and time positions.
def masks(data: Data, target: slice, train: slice) -> dict[str, np.ndarray]:
    train_sigma = data.sigma[train]
    target_sigma = data.sigma[target]
    z = data.latent[target]
    q1, q2 = np.quantile(train_sigma, [0.33, 0.67])

    out = {
        "all": np.ones(len(target_sigma), dtype=bool),
        "low_sigma": target_sigma <= q1,
        "medium_sigma": (target_sigma > q1) & (target_sigma < q2),
        "high_sigma": target_sigma >= q2,
        "z1_positive": z[:, 0] >= 0,
        "z1_negative": z[:, 0] < 0,
        "z2_positive": z[:, 1] >= 0,
        "z2_negative": z[:, 1] < 0,
    }
    if data.drift_index is not None:
        idx = np.arange(target.start, target.stop)
        out["pre_drift"] = idx < data.drift_index
        out["post_drift"] = idx >= data.drift_index
    return out


# Put uncalibrated synthetic intervals in the common result format.
def raw_result(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, np.ndarray]:
    zero = np.zeros(len(y))
    return {
        "lower": lower,
        "upper": upper,
        "adjustment": zero.copy(),
        "miscoverage": ((y < lower) | (y > upper)).astype(int),
        "global_component": zero.copy(),
        "linear_component": zero.copy(),
        "functional_component": zero.copy(),
    }


# Calculate the largest coverage gap across the synthetic test groups.
def worst_group(
    y: np.ndarray,
    result: dict[str, np.ndarray],
    group_masks: dict[str, np.ndarray],
    target: float,
) -> float:
    values: list[float] = []
    for name, mask in group_masks.items():
        if name == "all" or mask.sum() < 20:
            continue
        coverage = np.mean(
            (y[mask] >= result["lower"][mask]) & (y[mask] <= result["upper"][mask])
        )
        values.append(abs(float(coverage) - target))
    return float(max(values)) if values else np.nan


# Calculate all reported metrics for one synthetic method path.
def evaluate(
    method: str,
    y: np.ndarray,
    result: dict[str, np.ndarray],
    witness: np.ndarray,
    group_masks: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, float | str]:
    out = interval_metrics(y, result["lower"], result["upper"], alpha)
    out.update(
        {
            "method": method,
            "functional_error": functional_coverage_error(
                result["miscoverage"], witness, alpha
            ),
            "worst_group_error": worst_group(y, result, group_masks, 1.0 - alpha),
            "mean_adjustment": float(np.mean(result["adjustment"])),
            "mean_abs_linear_component": float(
                np.mean(np.abs(result["linear_component"]))
            ),
            "mean_abs_functional_component": float(
                np.mean(np.abs(result["functional_component"]))
            ),
        }
    )
    return out


# Combine validation metrics into the same style of tuning score used elsewhere.
def score(metrics: dict[str, float | str], raw_width: float, alpha: float) -> float:
    return float(
        3.0 * abs(float(metrics["coverage"]) - (1.0 - alpha))
        + float(metrics["functional_error"])
        + 0.50 * float(metrics["worst_group_error"])
        + 0.02 * float(metrics["avg_width"]) / max(raw_width, 1e-12)
    )


# Return readable hyperparameter candidates for each calibration method.
def grids(quick: bool) -> dict[str, list[dict[str, float | int]]]:
    if quick:
        scalar_eta = [0.02, 0.05]
        eta_g = [0.02, 0.05]
        eta_l = [0.005, 0.02]
        eta_f = [0.005, 0.02]
        dims = [128]
        radii = [2.0, 5.0]
        scales = [1.0, 2.0]
        ridges = [1e-3]
    else:
        scalar_eta = [0.01, 0.02, 0.05, 0.10]
        eta_g = [0.01, 0.02, 0.05, 0.10]
        eta_l = [0.001, 0.005, 0.01, 0.02, 0.05]
        eta_f = [0.001, 0.005, 0.01, 0.02, 0.05]
        dims = [128, 256]
        radii = [2.0, 5.0, 10.0]
        scales = [0.5, 1.0, 2.0, 4.0]
        ridges = [1e-3, 1e-2]

    return {
        "Scalar ACI": [{"eta": x} for x in scalar_eta],
        "Linear contextual ACI": [
            {"eta_global": a, "eta_linear": b, "linear_radius": r}
            for a, b, r in product(eta_g, eta_l, radii)
        ],
        "Functional ACI": [
            {
                "eta_global": a,
                "eta_functional": b,
                "n_components": d,
                "functional_radius": r,
                "length_scale": s,
            }
            for a, b, d, r, s in product(eta_g, eta_f, dims, radii, scales)
        ],
    }


# Build the hybrid candidate list from the smaller component grids.
def hybrid_grid(
    quick: bool,
    best_linear: dict[str, float | int],
) -> list[dict[str, float | int]]:
    if quick:
        eta_functional = [0.005, 0.02]
        dims = [128]
        radii = [2.0, 5.0]
        scales = [1.0, 2.0]
        ridges = [1e-3]
    else:
        eta_functional = [0.001, 0.005, 0.01, 0.02, 0.05]
        dims = [128, 256]
        radii = [2.0, 5.0, 10.0]
        scales = [0.5, 1.0, 2.0, 4.0]
        ridges = [1e-3, 1e-2]

    return [
        {
            "eta_global": float(best_linear["eta_global"]),
            "eta_linear": float(best_linear["eta_linear"]),
            "eta_functional": ef,
            "n_components": d,
            "linear_radius": float(best_linear["linear_radius"]),
            "functional_radius": radius,
            "length_scale": scale,
            "residual_ridge": ridge,
        }
        for ef, d, radius, scale, ridge in product(
            eta_functional,
            dims,
            radii,
            scales,
            ridges,
        )
    ]


# Create one calibrator from a method name and one parameter dictionary.
def new_model(
    method: str,
    params: dict[str, float | int],
    alpha: float,
    max_adjustment: float,
    seed: int,
) -> Any:
    if method == "Scalar ACI":
        return ScalarACI(
            alpha=alpha,
            eta=float(params["eta"]),
            max_adjustment=max_adjustment,
        )
    if method == "Linear contextual ACI":
        return LinearContextualACI(
            alpha=alpha,
            eta_global=float(params["eta_global"]),
            eta_linear=float(params["eta_linear"]),
            linear_radius=float(params["linear_radius"]),
            max_adjustment=max_adjustment,
        )
    if method == "Functional ACI":
        return FunctionalACI(
            alpha=alpha,
            eta_global=float(params["eta_global"]),
            eta_functional=float(params["eta_functional"]),
            n_components=int(params["n_components"]),
            length_scale=float(params["length_scale"]),
            functional_radius=float(params["functional_radius"]),
            max_adjustment=max_adjustment,
            random_state=seed,
        )
    if method == "Hybrid Functional ACI":
        return HybridFunctionalACI(
            alpha=alpha,
            eta_global=float(params["eta_global"]),
            eta_linear=float(params["eta_linear"]),
            eta_functional=float(params["eta_functional"]),
            n_components=int(params["n_components"]),
            length_scale=float(params["length_scale"]),
            residual_ridge=float(params["residual_ridge"]),
            linear_radius=float(params["linear_radius"]),
            functional_radius=float(params["functional_radius"]),
            max_adjustment=max_adjustment,
            random_state=seed,
        )
    raise ValueError(method)


# Fit nonlinear feature maps only for methods that actually use them.
def fit_map(model: Any, method: str, train_x: np.ndarray) -> None:
    if method in {"Functional ACI", "Hybrid Functional ACI"}:
        model.fit_feature_map(train_x)


# Run one method on one continuous interval path.
def run_once(
    method: str,
    params: dict[str, float | int],
    train_x: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_adjustment: float,
    seed: int,
) -> dict[str, np.ndarray]:
    model = new_model(method, params, alpha, max_adjustment, seed)
    fit_map(model, method, train_x)
    if method == "Scalar ACI":
        return convert(model.run(lower, upper, y, reset=True))
    return convert(model.run(lower, upper, y, x, reset=True))


# Learn state on validation rows, then continue into test rows without resetting.
def run_warm_test(
    method: str,
    params: dict[str, float | int],
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    lower_val: np.ndarray,
    upper_val: np.ndarray,
    y_val: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    max_adjustment: float,
    seed: int,
) -> dict[str, np.ndarray]:
    model = new_model(method, params, alpha, max_adjustment, seed)
    fit_map(model, method, train_x)

    if method == "Scalar ACI":
        model.run(lower_val, upper_val, y_val, reset=True)
        return convert(model.run(lower_test, upper_test, y_test, reset=False))

    model.run(lower_val, upper_val, y_val, val_x, reset=True)
    return convert(model.run(lower_test, upper_test, y_test, test_x, reset=False))


# Select each method's settings using validation rows only.
def tune(
    data: Data,
    x: np.ndarray,
    train: slice,
    val: slice,
    alpha: float,
    quick: bool,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    train_x, val_x = x[train], x[val]
    y, lower, upper = data.y[val], data.lower[val], data.upper[val]
    # Every candidate is judged with one fixed evaluation map and group definition.
    witness = witness_map(train_x, val_x, seed + 10_000)
    group_masks = masks(data, val, train)
    raw_width = float(np.mean(upper - lower))
    max_adjustment = max(10.0, 4.0 * raw_width)

    selected: dict[str, dict[str, float | int]] = {
        "meta": {"max_adjustment": max_adjustment}
    }

    # Quick mode keeps only the first candidate for a fast pipeline check.
    for method, candidates in grids(quick).items():
        best_value = np.inf
        best: dict[str, float | int] | None = None

        for params in candidates:
            result = run_once(
                method,
                params,
                train_x,
                val_x,
                lower,
                upper,
                y,
                alpha,
                max_adjustment,
                seed,
            )
            value = score(
                evaluate(method, y, result, witness, group_masks, alpha),
                raw_width,
                alpha,
            )
            if value < best_value:
                best_value, best = value, params

        if best is None:
            raise RuntimeError(f"No best config for {method}.")

        selected[method] = best

    # Store the full tuning table so the winning candidate can be checked later.
    hybrid_best_value = np.inf
    hybrid_best: dict[str, float | int] | None = None

    for params in hybrid_grid(quick, selected["Linear contextual ACI"]):
        result = run_once(
            "Hybrid Functional ACI",
            params,
            train_x,
            val_x,
            lower,
            upper,
            y,
            alpha,
            max_adjustment,
            seed,
        )
        value = score(
            evaluate("Hybrid Functional ACI", y, result, witness, group_masks, alpha),
            raw_width,
            alpha,
        )
        if value < hybrid_best_value:
            hybrid_best_value, hybrid_best = value, params

    if hybrid_best is None:
        raise RuntimeError("No best config for Hybrid Functional ACI.")

    selected["Hybrid Functional ACI"] = hybrid_best
    return selected


# Measure how long coverage takes to recover after the known drift point.
def delay(
    miscoverage: np.ndarray,
    drift_step: int | None,
    alpha: float,
    window: int,
) -> float:
    if drift_step is None:
        return np.nan
    rolling = (
        1.0
        - pd.Series(miscoverage).rolling(window, min_periods=window).mean().to_numpy()
    )
    good = np.abs(rolling - (1.0 - alpha)) <= 0.03
    start = max(drift_step + window - 1, window - 1)
    for idx in range(start, len(good) - 2):
        if np.all(good[idx : idx + 3]):
            return float(idx - drift_step)
    return np.nan


# Generate, tune, and evaluate all methods for one scenario and random seed.
def run_seed(
    scenario: str,
    seed: int,
    n: int,
    train: slice,
    val: slice,
    test: slice,
    selected: dict[str, dict[str, float | int]],
    alpha: float,
    window: int,
) -> tuple[list[dict[str, float | str | int]], pd.DataFrame]:
    data = generate(scenario, n, seed, train.stop, alpha)
    x = preprocess(data.context_raw, train)

    train_x, val_x, test_x = x[train], x[val], x[test]
    y_val, y_test = data.y[val], data.y[test]
    lower_val, lower_test = data.lower[val], data.lower[test]
    upper_val, upper_test = data.upper[val], data.upper[test]

    witness = witness_map(train_x, test_x, seed + 10_000)
    group_masks = masks(data, test, train)
    max_adjustment = float(selected["meta"]["max_adjustment"])

    drift_step: int | None = None
    if data.drift_index is not None:
        candidate = data.drift_index - test.start
        if 0 <= candidate < len(y_test):
            drift_step = candidate

    rows: list[dict[str, float | str | int]] = []
    curves = pd.DataFrame(
        {
            "scenario": scenario,
            "seed": seed,
            "test_step": np.arange(len(y_test)),
            "drift_step": drift_step if drift_step is not None else np.nan,
        }
    )

    for method in METHODS:
        if method == "Raw quantile":
            result = raw_result(y_test, lower_test, upper_test)
        else:
            result = run_warm_test(
                method,
                selected[method],
                train_x,
                val_x,
                test_x,
                lower_val,
                upper_val,
                y_val,
                lower_test,
                upper_test,
                y_test,
                alpha,
                max_adjustment,
                seed,
            )

        metrics = evaluate(method, y_test, result, witness, group_masks, alpha)
        metrics.update(
            {
                "scenario": scenario,
                "seed": seed,
                "adaptation_delay": delay(
                    result["miscoverage"], drift_step, alpha, window
                ),
            }
        )
        rows.append(metrics)
        curves[method] = (
            1.0
            - pd.Series(result["miscoverage"]).rolling(window, min_periods=window)
            .mean()
            .to_numpy()
        )
    return rows, curves


# Average repeated seeds for each scenario and method.
def summarise(results: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "coverage",
        "miscoverage",
        "avg_width",
        "median_width",
        "winkler",
        "functional_error",
        "worst_group_error",
        "mean_adjustment",
        "mean_abs_linear_component",
        "mean_abs_functional_component",
        "adaptation_delay",
    )
    rows = []
    for (scenario, method), group in results.groupby(
        ["scenario", "method"], sort=False
    ):
        row: dict[str, float | str | int] = {
            "scenario": scenario,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


# Draw one grouped bar chart for a selected synthetic metric.
def metric_plot(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    scenarios = list(summary["scenario"].drop_duplicates())
    methods = list(summary["method"].drop_duplicates())
    x = np.arange(len(scenarios))
    total = 0.84
    width = total / len(methods)

    plt.figure(figsize=(13, 6))
    for idx, method in enumerate(methods):
        frame = (
            summary[summary["method"] == method].set_index("scenario")
            .reindex(scenarios)
        )
        pos = x - total / 2 + width / 2 + idx * width
        plt.bar(
            pos,
            frame[f"{metric}_mean"],
            width=width,
            yerr=frame[f"{metric}_se"],
            capsize=2,
            label=method,
        )
    plt.xticks(x, scenarios)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# Show the rolling miscoverage paths around the simulated drift point.
def drift_plot(curves: pd.DataFrame, alpha: float, path: Path) -> None:
    frame = curves[curves["scenario"] == "abrupt_drift"]
    drift_values = frame["drift_step"].dropna()
    if frame.empty or drift_values.empty:
        return
    drift_step = int(drift_values.iloc[0])

    plt.figure(figsize=(12, 6))
    for method in METHODS:
        curve = frame.groupby("test_step")[method].mean()
        plt.plot(curve.index, curve.values, label=method, linewidth=1.2)
    plt.axhline(1.0 - alpha, linestyle="--", linewidth=1.0, label="Target")
    plt.axvline(drift_step, linestyle=":", linewidth=1.2, label="Drift")
    plt.xlabel("Test step")
    plt.ylabel("Rolling coverage")
    plt.title("Adaptation after abrupt drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# Run every requested scenario and seed, then save detailed and averaged results.
def main() -> None:
    args = parse_args()
    # Quick mode reduces sample size and repetitions but leaves the formulas unchanged.
    n = min(args.n_samples, 2500) if args.quick else args.n_samples
    n_seeds = min(args.n_seeds, 5) if args.quick else args.n_seeds
    # Use the same time-ordered fractions for every simulated scenario.
    train, val, test = splits(n, args.train_frac, args.validation_frac)

    # Create separate output folders for results, plots, and tuning details.
    tables = args.output / "tables"
    # Figures are written separately from the CSV tables used to build them.
    figures = args.output / "figures"
    diagnostics = args.output / "diagnostics"
    for directory in (tables, figures, diagnostics):
        directory.mkdir(parents=True, exist_ok=True)

    selected_all: dict[str, dict[str, dict[str, float | int]]] = {}
    print("Tuning one development seed per scenario...")

    for idx, scenario in enumerate(SCENARIOS):
        seed = args.random_state + 1000 + idx
        data = generate(scenario, n, seed, train.stop, args.alpha)
        x = preprocess(data.context_raw, train)
        print(f"  {scenario}")
        selected_all[scenario] = tune(data, x, train, val, args.alpha, args.quick, seed)

    # Save selected settings before evaluating them across fresh random seeds.
    (diagnostics / "selected_hyperparameters.json").write_text(
        json.dumps(selected_all, indent=2),
        encoding="utf-8",
    )

    all_rows: list[dict[str, float | str | int]] = []
    all_curves: list[pd.DataFrame] = []
    print("Evaluating frozen parameters across seeds...")

    for scenario_idx, scenario in enumerate(SCENARIOS):
        for seed_idx in range(n_seeds):
            seed = args.random_state + 10_000 + 100 * scenario_idx + seed_idx
            print(f"  {scenario}: seed {seed_idx + 1}/{n_seeds}")
            rows, curves = run_seed(
                scenario,
                seed,
                n,
                train,
                val,
                test,
                selected_all[scenario],
                args.alpha,
                args.rolling_window,
            )
            all_rows.extend(rows)
            all_curves.append(curves)

    seed_results = pd.DataFrame(all_rows)
    curve_table = pd.concat(all_curves, ignore_index=True)
    # Average seeds only after keeping all seed-level results in a separate CSV.
    summary = summarise(seed_results)

    seed_results.to_csv(tables / "synthetic_seed_results.csv", index=False)
    summary.to_csv(tables / "synthetic_summary.csv", index=False)
    curve_table.to_csv(tables / "synthetic_adaptation_curves.csv", index=False)

    for metric, label, filename in (
        ("coverage", "Marginal coverage", "synthetic_coverage.png"),
        ("winkler", "Winkler score", "synthetic_winkler.png"),
        (
            "functional_error",
            "Functional coverage error",
            "synthetic_functional_error.png",
        ),
        (
            "worst_group_error",
            "Worst-group coverage error",
            "synthetic_worst_group_error.png",
        ),
    ):
        metric_plot(summary, metric, label, figures / filename)

    drift_plot(curve_table, args.alpha, figures / "synthetic_drift_adaptation.png")

    columns = [
        "scenario",
        "method",
        "coverage_mean",
        "avg_width_mean",
        "winkler_mean",
        "functional_error_mean",
        "worst_group_error_mean",
        "adaptation_delay_mean",
    ]
    print("\nSynthetic summary:")
    print(
        summary[columns].sort_values(["scenario", "functional_error_mean"])
        .to_string(index=False)
    )
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
