# Run small deterministic examples through calibration, tuning, and evaluation.

import unittest

import numpy as np

from experiments import run_multi_market_functional_aci as multi_market_experiment
from src.evaluation.metrics import evaluate
from src.functional_pipeline import convert_result, run_final_models


# These tests connect several modules instead of checking one helper at a time.
class EndToEndTests(unittest.TestCase):
    # Exercise the exact call that previously failed when convert_result was not imported.
    def test_multi_market_tuning_uses_shared_result_converter(self):
        self.assertIs(multi_market_experiment.convert_result, convert_result)

        y = np.linspace(-1.0, 1.0, 8)
        lower = y - 0.25
        upper = y + 0.25
        evaluation_map = np.ones((len(y), 1))
        group_masks = {
            "all": np.ones(len(y), dtype=bool),
            "first_half": np.arange(len(y)) < len(y) // 2,
        }

        selected, table = multi_market_experiment.tune_scalar(
            lower=lower,
            upper=upper,
            y=y,
            alpha=0.1,
            max_adjustment=10.0,
            evaluation_map=evaluation_map,
            group_masks=group_masks,
            eta_values=[0.02],
        )

        self.assertEqual(len(table), 1)
        self.assertEqual(selected["eta"], 0.02)

    # Run every ACI version on a tiny dataset and compare the final metrics.
    def test_small_calibration_and_evaluation_pipeline(self):
        rng = np.random.default_rng(11)
        train_context = rng.normal(size=(30, 3))
        validation_context = rng.normal(size=(12, 3))
        test_context = rng.normal(size=(16, 3))
        y_validation = rng.normal(size=12)
        y_test = rng.normal(size=16)

        selected = {
            "scalar": {"eta": 0.02},
            "contextual": {
                "eta_global": 0.02,
                "eta_linear": 0.01,
                "linear_radius": 2.0,
            },
            "functional": {
                "eta_global": 0.02,
                "eta_functional": 0.01,
                "n_components": 8,
                "functional_radius": 2.0,
                "length_scale": 1.0,
            },
            "hybrid": {
                "eta_global": 0.02,
                "eta_linear": 0.01,
                "eta_functional": 0.01,
                "n_components": 8,
                "linear_radius": 2.0,
                "functional_radius": 2.0,
                "length_scale": 1.0,
                "residual_ridge": 0.01,
            },
        }
        results = run_final_models(
            selected=selected,
            alpha=0.1,
            max_adjustment=10.0,
            random_state=42,
            train_context=train_context,
            validation_context=validation_context,
            test_context=test_context,
            lower_validation=y_validation - 0.35,
            upper_validation=y_validation + 0.35,
            y_validation=y_validation,
            lower_test=y_test - 0.35,
            upper_test=y_test + 0.35,
            y_test=y_test,
        )

        evaluation_map = rng.normal(size=(len(y_test), 5))
        group_masks = {
            "all": np.ones(len(y_test), dtype=bool),
            "even": np.arange(len(y_test)) % 2 == 0,
            "odd": np.arange(len(y_test)) % 2 == 1,
        }
        observed = {}
        for method, result in results.items():
            metrics = evaluate(
                method_name=method,
                y=y_test,
                result=result,
                evaluation_map=evaluation_map,
                group_masks=group_masks,
                alpha=0.1,
            )
            observed[method] = np.array(
                [
                    metrics["coverage"],
                    metrics["avg_width"],
                    metrics["winkler"],
                    metrics["functional_error"],
                    metrics["worst_group_error"],
                ]
            )

        expected = {
            "Scalar ACI": np.array([1.0, 0.7, 0.7, 0.04685166, 0.1]),
            "Linear contextual ACI": np.array(
                [1.0, 0.70376740, 0.70376740, 0.04685166, 0.1]
            ),
            "Functional ACI": np.array([1.0, 0.70095334, 0.70095334, 0.04685166, 0.1]),
            "Hybrid Functional ACI": np.array(
                [1.0, 0.70055142, 0.70055142, 0.04685166, 0.1]
            ),
        }
        self.assertEqual(set(observed), set(expected))
        for method in expected:
            np.testing.assert_allclose(
                observed[method],
                expected[method],
                rtol=0.0,
                atol=1e-8,
            )


if __name__ == "__main__":
    unittest.main()
