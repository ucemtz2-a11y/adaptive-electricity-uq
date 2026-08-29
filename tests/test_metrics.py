# Check the metric formulas with small examples whose answers are easy to calculate.

import unittest

import numpy as np

from src.evaluation.metrics import (
    compute_worst_group_error,
    evaluate,
    functional_coverage_error,
    interval_metrics,
)


# Keep all metric checks in one unittest class.
class MetricTests(unittest.TestCase):
    # Compare coverage, width, and Winkler score with a hand calculation.
    def test_interval_metrics_match_hand_calculation(self):
        y = np.array([0.0, 2.0, 5.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 4.0])

        metrics = interval_metrics(y, lower, upper, alpha=0.1)

        self.assertAlmostEqual(metrics["coverage"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["miscoverage"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["avg_width"], 2.0)
        self.assertAlmostEqual(metrics["median_width"], 1.0)
        self.assertAlmostEqual(metrics["winkler"], 46.0 / 3.0)

    # Check that functional error follows its vector formula exactly.
    def test_functional_error_matches_vector_definition(self):
        miscoverage = np.array([0.0, 1.0, 0.0])
        feature_map = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, -1.0]])
        expected = np.linalg.norm(
            np.mean((miscoverage - 0.1)[:, None] * feature_map, axis=0)
        )

        self.assertEqual(
            functional_coverage_error(miscoverage, feature_map, alpha=0.1),
            expected,
        )

    # Empty groups should be ignored when the worst coverage gap is calculated.
    def test_worst_group_error_uses_nonempty_conditional_groups(self):
        y = np.array([0.0, 0.0, 2.0, 2.0])
        lower = np.zeros(4)
        upper = np.ones(4)
        masks = {
            "all": np.ones(4, dtype=bool),
            "first_half": np.array([1, 1, 0, 0], dtype=bool),
            "second_half": np.array([0, 0, 1, 1], dtype=bool),
            "empty": np.zeros(4, dtype=bool),
        }

        self.assertAlmostEqual(
            compute_worst_group_error(y, lower, upper, masks, 0.9),
            0.9,
        )

    # All experiments rely on evaluate returning the same set of result fields.
    def test_evaluate_returns_standard_result_schema(self):
        y = np.array([0.0, 2.0])
        lower = np.array([-1.0, 0.0])
        upper = np.array([1.0, 1.0])
        result = {
            "lower": lower,
            "upper": upper,
            "miscoverage": np.array([0, 1]),
            "adjustment": np.array([0.0, 0.2]),
            "linear_component": np.array([0.0, 0.1]),
            "functional_component": np.array([0.0, -0.1]),
        }
        metrics = evaluate(
            "method",
            y,
            result,
            np.ones((2, 1)),
            {"all": np.ones(2, dtype=bool), "tail": np.array([0, 1], dtype=bool)},
            0.1,
        )

        expected_fields = {
            "method",
            "coverage",
            "coverage_error",
            "avg_width",
            "winkler",
            "functional_error",
            "worst_group_error",
            "mean_adjustment",
            "mean_abs_linear_component",
            "mean_abs_functional_component",
        }
        self.assertTrue(expected_fields.issubset(metrics))


if __name__ == "__main__":
    unittest.main()
