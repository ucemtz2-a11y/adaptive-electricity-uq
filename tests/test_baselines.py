# Module purpose: Test baseline definitions, online update order, and chronological splitting.

import unittest

import numpy as np
import pandas as pd

from src.calibration.baselines import (
    adaptive_conformal_score_interval,
    conformal_quantile,
    rolling_historical_interval,
    split_cqr_interval,
)
from src.functional_pipeline import chronological_split


# Implement BaselineTests.
class BaselineTests(unittest.TestCase):
    # Test chronological split matches paper protocol.
    def test_chronological_split_matches_paper_protocol(self):
        train, validation, test = chronological_split(100, 0.60, 0.20)
        self.assertEqual((train.start, train.stop), (0, 60))
        self.assertEqual((validation.start, validation.stop), (60, 80))
        self.assertEqual((test.start, test.stop), (80, 100))

    # Test conformal quantile uses finite sample correction.
    def test_conformal_quantile_uses_finite_sample_correction(self):
        scores = np.arange(10, dtype=float)
        self.assertEqual(conformal_quantile(scores, alpha=0.1), 9.0)
        self.assertEqual(conformal_quantile([], alpha=0.1), 0.0)

    # Test split CQR applies the same scalar expansion.
    def test_split_cqr_applies_the_same_scalar_expansion(self):
        index = pd.RangeIndex(4)
        y_cal = pd.Series([0.0, 2.0, 4.0, 8.0], index=index)
        lower_cal = pd.Series([0.0, 1.0, 3.0, 5.0], index=index)
        upper_cal = pd.Series([1.0, 3.0, 5.0, 6.0], index=index)
        lower_test = pd.Series([10.0, 20.0])
        upper_test = pd.Series([12.0, 22.0])

        lower, upper, qhat = split_cqr_interval(
            y_cal,
            lower_cal,
            upper_cal,
            lower_test,
            upper_test,
            alpha=0.25,
        )

        np.testing.assert_allclose(lower.to_numpy(), lower_test - qhat)
        np.testing.assert_allclose(upper.to_numpy(), upper_test + qhat)

    # Test adaptive score predicts before updating.
    def test_adaptive_score_predicts_before_updating(self):
        index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        y = pd.Series([5.0, 0.5], index=index)
        lower = pd.Series([0.0, 0.0], index=index)
        upper = pd.Series([1.0, 1.0], index=index)

        result = adaptive_conformal_score_interval(
            y,
            lower,
            upper,
            q_init=1.0,
            alpha=0.1,
            eta=0.2,
        )

        self.assertEqual(result.iloc[0]["q"], 1.0)
        self.assertAlmostEqual(result.iloc[1]["q"], 1.18)

    # Test rolling interval does not use current outcome.
    def test_rolling_interval_does_not_use_current_outcome(self):
        index = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
        y = pd.Series(np.arange(40, dtype=float), index=index)
        test_index = index[30:]

        lower_before, upper_before = rolling_historical_interval(
            y,
            test_index,
            window=24,
            alpha=0.1,
        )
        changed = y.copy()
        changed.loc[test_index[-1]] = 1_000_000.0
        lower_after, upper_after = rolling_historical_interval(
            changed,
            test_index,
            window=24,
            alpha=0.1,
        )

        self.assertEqual(lower_before.iloc[-1], lower_after.iloc[-1])
        self.assertEqual(upper_before.iloc[-1], upper_after.iloc[-1])


if __name__ == "__main__":
    unittest.main()
