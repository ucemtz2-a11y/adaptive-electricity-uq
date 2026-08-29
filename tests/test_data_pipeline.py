# Module purpose: Test market loading, chronological ordering, and lag construction.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.functional_pipeline import MODEL_FEATURES, load_market_data


# Implement DataPipelineTests.
class DataPipelineTests(unittest.TestCase):
    # Test market loader builds expected chronological features.
    def test_market_loader_builds_expected_chronological_features(self):
        n_rows = 2300
        index = pd.date_range("2022-01-01", periods=n_rows, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "datetime": index,
                "price": np.linspace(10.0, 100.0, n_rows),
                "load": np.linspace(500.0, 800.0, n_rows),
                "wind": np.linspace(50.0, 80.0, n_rows),
                "solar": np.linspace(0.0, 40.0, n_rows),
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DK_1_dataset.csv"
            frame.iloc[::-1].to_csv(path, index=False)
            loaded, target, features = load_market_data("DK_1", Path(directory))

        self.assertEqual(target, "price")
        self.assertEqual(features, MODEL_FEATURES)
        self.assertEqual(len(loaded), n_rows - 168)
        self.assertTrue(loaded.index.is_monotonic_increasing)
        self.assertEqual(loaded.index.tz, index.tz)
        self.assertEqual(loaded.iloc[0]["price_lag_168"], frame.iloc[0]["price"])
        expected_residual_lag = (
            frame.iloc[144]["load"] - frame.iloc[144]["wind"] - frame.iloc[144]["solar"]
        )
        self.assertAlmostEqual(
            loaded.iloc[0]["residual_load_lag_24"],
            expected_residual_lag,
        )


if __name__ == "__main__":
    unittest.main()
