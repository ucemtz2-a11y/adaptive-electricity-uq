# Module purpose: Implement the rolling, Split CQR, and ACS baselines used in the paper.

"""Strong interval-calibration baselines used in the paper experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Conformal quantile.
def conformal_quantile(scores, alpha: float = 0.1) -> float:
    """Return the finite-sample corrected conformal quantile."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]

    n = len(scores)
    if n == 0:
        return 0.0

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q_level, interpolation="higher"))


# Rolling historical interval.
def rolling_historical_interval(
    y: pd.Series,
    test_index,
    window: int = 168,
    alpha: float = 0.1,
):
    """Construct a model-free interval from past realised prices only."""
    lower = y.shift(1).rolling(window=window, min_periods=window).quantile(alpha / 2)
    upper = y.shift(1).rolling(window=window, min_periods=window).quantile(1 - alpha / 2)

    lower_test = lower.loc[test_index]
    upper_test = upper.loc[test_index]

    # Fall back to expanding-window empirical quantiles before the rolling window fills.
    exp_lower = y.shift(1).expanding(min_periods=24).quantile(alpha / 2)
    exp_upper = y.shift(1).expanding(min_periods=24).quantile(1 - alpha / 2)

    lower_test = lower_test.fillna(exp_lower.loc[test_index])
    upper_test = upper_test.fillna(exp_upper.loc[test_index])

    return lower_test, upper_test


# Split CQR interval.
def split_cqr_interval(
    y_cal: pd.Series,
    lower_cal: pd.Series,
    upper_cal: pd.Series,
    lower_test: pd.Series,
    upper_test: pd.Series,
    alpha: float = 0.1,
):
    """Apply split conformal/CQR calibration to raw test intervals."""
    scores = np.maximum.reduce(
        [
            lower_cal.values - y_cal.values,
            y_cal.values - upper_cal.values,
            np.zeros(len(y_cal)),
        ]
    )

    qhat = conformal_quantile(scores, alpha=alpha)

    lower_cqr = lower_test - qhat
    upper_cqr = upper_test + qhat

    return lower_cqr, upper_cqr, qhat


# Adaptive conformal score interval.
def adaptive_conformal_score_interval(
    y_test: pd.Series,
    lower_test: pd.Series,
    upper_test: pd.Series,
    q_init: float,
    alpha: float = 0.1,
    eta: float = 1.0,
    q_min: float = 0.0,
    q_max: float = 100.0,
):
    """Update a scalar conformal-score threshold after each observation."""
    records = []
    q = float(q_init)

    for idx in y_test.index:
        y_t = float(y_test.loc[idx])
        l_raw = float(lower_test.loc[idx])
        u_raw = float(upper_test.loc[idx])

        l = l_raw - q
        u = u_raw + q

        covered = int((y_t >= l) and (y_t <= u))
        err = 1 - covered

        records.append(
            {
                "datetime": idx,
                "y_true": y_t,
                "lower_adaptive_conformal": l,
                "upper_adaptive_conformal": u,
                "q": q,
                "covered": covered,
            }
        )

        q = q + eta * (err - alpha)
        q = float(np.clip(q, q_min, q_max))

    out = pd.DataFrame(records)
    out = out.set_index("datetime")

    return out
