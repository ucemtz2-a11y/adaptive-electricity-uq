# Module purpose: Provide the authoritative interval and conditional-coverage metrics.

"""Authoritative metric implementations for all paper experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.calibration.functional_aci import Array, _as_1d, _as_2d


# Interval metrics.
def interval_metrics(
    y_true: Array,
    lower: Array,
    upper: Array,
    alpha: float = 0.10,
) -> dict[str, float]:
    y = _as_1d(y_true)
    lower = _as_1d(lower)
    upper = _as_1d(upper)

    covered = (y >= lower) & (y <= upper)
    width = upper - lower

    winkler = width.copy()

    below = y < lower
    above = y > upper

    winkler[below] += (
        2.0 / alpha * (lower[below] - y[below])
    )

    winkler[above] += (
        2.0 / alpha * (y[above] - upper[above])
    )

    return {
        "coverage": float(covered.mean()),
        "miscoverage": float(1.0 - covered.mean()),
        "avg_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "winkler": float(winkler.mean()),
    }


# Functional coverage error.
def functional_coverage_error(
    miscoverage: Array,
    feature_map: Array,
    alpha: float = 0.10,
) -> float:
    e = _as_1d(miscoverage)
    psi = _as_2d(feature_map)

    if len(e) != len(psi):
        raise ValueError("miscoverage and feature_map must align.")

    discrepancy = np.mean((e - alpha)[:, None] * psi, axis=0)

    return float(np.linalg.norm(discrepancy))


# Rolling functional coverage error.
def rolling_functional_coverage_error(
    miscoverage: Array,
    feature_map: Array,
    alpha: float = 0.10,
    window: int = 168,
    min_periods: Optional[int] = None,
) -> Array:
    e = _as_1d(miscoverage)
    psi = _as_2d(feature_map)

    if len(e) != len(psi):
        raise ValueError("miscoverage and feature_map must align.")

    if min_periods is None:
        min_periods = window

    weighted = (e - alpha)[:, None] * psi

    cumulative = np.vstack([np.zeros((1, psi.shape[1])), np.cumsum(weighted, axis=0)])

    output = np.full(len(e), np.nan)

    for end in range(1, len(e) + 1):
        start = max(0, end - window)
        count = end - start

        if count >= min_periods:
            mean_vector = (cumulative[end] - cumulative[start]) / count

            output[end - 1] = float(np.linalg.norm(mean_vector))

    return output


# Compute worst group error.
def compute_worst_group_error(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    group_masks: dict[str, np.ndarray],
    target_coverage: float,
) -> float:
    errors = []

    for group_name, mask in group_masks.items():
        if group_name == "all":
            continue

        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue

        coverage = float(np.mean((y[mask] >= lower[mask]) & (y[mask] <= upper[mask])))
        errors.append(abs(coverage - target_coverage))

    return float(max(errors)) if errors else np.nan


# Evaluate.
def evaluate(
    method_name: str,
    y: np.ndarray,
    result: dict[str, np.ndarray],
    evaluation_map: np.ndarray,
    group_masks: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, float | str]:
    metrics = interval_metrics(y, result["lower"], result["upper"], alpha)

    metrics["method"] = method_name
    metrics["coverage_error"] = abs(metrics["coverage"] - (1.0 - alpha))
    metrics["functional_error"] = (
        functional_coverage_error(result["miscoverage"], evaluation_map, alpha)
    )
    metrics["worst_group_error"] = (
        compute_worst_group_error(
            y=y,
            lower=result["lower"],
            upper=result["upper"],
            group_masks=group_masks,
            target_coverage=1.0 - alpha,
        )
    )
    metrics["mean_adjustment"] = float(np.mean(result["adjustment"]))
    metrics["mean_abs_linear_component"] = float(
        np.mean(np.abs(result["linear_component"]))
    )
    metrics["mean_abs_functional_component"] = float(
        np.mean(np.abs(result["functional_component"]))
    )

    return metrics
