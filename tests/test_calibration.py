# Module purpose: Test online calibration order and fixed-seed determinism.

import unittest

import numpy as np

from src.calibration.functional_aci import ScalarACI
from src.functional_pipeline import run_final_models


# Implement CalibrationTests.
class CalibrationTests(unittest.TestCase):
    # Test scalar ACI predicts before updating.
    def test_scalar_aci_predicts_before_updating(self):
        model = ScalarACI(alpha=0.1, eta=0.2, max_adjustment=10.0)
        output = model.run(
            lower_raw=np.array([0.0, 0.0, 0.0]),
            upper_raw=np.array([1.0, 1.0, 1.0]),
            y_true=np.array([2.0, 0.5, 2.0]),
        )

        np.testing.assert_allclose(output.adjustment, [0.0, 0.18, 0.16])
        np.testing.assert_array_equal(output.miscoverage, [1, 0, 1])

    # Test all calibrators are deterministic under fixed seed.
    def test_all_calibrators_are_deterministic_under_fixed_seed(self):
        rng = np.random.default_rng(7)
        train_context = rng.normal(size=(60, 3))
        validation_context = rng.normal(size=(20, 3))
        test_context = rng.normal(size=(25, 3))
        y_validation = rng.normal(size=20)
        y_test = rng.normal(size=25)
        lower_validation = y_validation - 0.4
        upper_validation = y_validation + 0.4
        lower_test = y_test - 0.4
        upper_test = y_test + 0.4
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

        arguments = dict(
            selected=selected,
            alpha=0.1,
            max_adjustment=10.0,
            random_state=42,
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
        first = run_final_models(**arguments)
        second = run_final_models(**arguments)

        self.assertEqual(first.keys(), second.keys())
        for method in first:
            self.assertEqual(first[method].keys(), second[method].keys())
            for field in first[method]:
                np.testing.assert_array_equal(first[method][field], second[method][field])


if __name__ == "__main__":
    unittest.main()
