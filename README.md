# Hybrid Functional Adaptive Conformal Inference

This repository contains the code for my dissertation. 

The main method is Hybrid Functional Adaptive Conformal Inference (HF-ACI). I
use it to study prediction intervals for electricity prices when the data and
the input features change over time. The experiments include synthetic data,
four European electricity markets, comparisons with other conformal methods,
feature perturbations, ablation studies, and a final test on 2024 data.

## Setup

I used Python 3.12 for the final experiments. The easiest way to install the
project is to create a virtual environment and install the packages in
`requirements.txt`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Preparing the data

The experiments use four electricity markets: DE-LU, DK1, DK2, and SE3.

To download data from ENTSO-E, first copy `.env.example` to `.env` and add your
API key as `ENTSOE_API_KEY`. Then download and process the 2022--2023 data:

```bash
python src/data/download_entsoe_multi_market.py
python src/data/preprocess_multi_market.py
```

The dates for this part are stored in `config/config.yaml`.

The 2024 data is downloaded separately so that it does not overwrite the
earlier data:

```bash
python src/data/download_entsoe_multi_2024.py
python src/data/preprocess_entsoe_multi_2024.py
```

After preprocessing, the files should be in:

```text
data/processed/multi_market/{DE_LU,DK_1,DK_2,SE_3}_dataset.csv
data/processed/multi_market_2024/{DE_LU,DK_1,DK_2,SE_3}_dataset.csv
```

The data and generated outputs are not uploaded to Git because they are large
and may contain local file paths.

## Running the experiments

The dissertation experiments are numbered v9 to v16. Run the development
experiments in this order:

```bash
# v9: synthetic experiments
python experiments/run_synthetic_functional_aci.py

# v10: main experiment on four electricity markets
python experiments/run_multi_market_functional_aci.py

# v11: feature perturbation experiment
python experiments/run_functional_perturbation.py

# v12: ablation experiments
python experiments/run_functional_ablation.py

# v13: comparison with strong baseline methods
python experiments/run_strong_baselines_v10.py

# v14: create the final development tables and figures
python experiments/run_final_summary.py

# v15: simulations comparing the method with the theory
python experiments/run_theory_alignment.py
```

Some experiments take a while. You can add `--quick` to the longer commands to
check that the code runs, for example:

```bash
python experiments/run_multi_market_functional_aci.py --quick
```

Quick runs use smaller parameter grids, so their results are only for checking
the code and are not the final dissertation results.

## Final 2024 experiment (v16)

The 2024 data is used as a final test. It must not be used to tune the models.
Before running the final test, use this command to check that the historical
2022--2023 pipeline still reproduces the saved v10 predictions:

```bash
python experiments/run_final_untouched_2024.py --check-only
```

If the check passes, the final evaluation can be run once with:

```bash
python experiments/run_final_untouched_2024.py --execute-once
```

The script uses a lock file to help prevent the final test from being run by
accident more than once. In the original one-shot run, HF-ACI had a
cross-market functional coverage error of 0.0142 and a mean Winkler score of
69.58.

The main experimental settings are:

- 2022--2023 data is split in time order: 60% training, 20% validation, and
  20% development test data.
- The final dataset contains all 8,784 UTC hours in 2024.
- Model settings, context transformations, evaluation features, group
  definitions, and random seeds are fixed before the 2024 evaluation.
- Each online method makes its prediction before it sees the true value for
  that time step.

## Results

Tables, figures, and diagnostic files are saved under:

```text
outputs/versions/
```

Each experiment has its own versioned folder, such as
`results_v10_multi_market_functional` or `results_v14_final_summary`.

## Main folders and files

```text
config/                         experiment dates and configuration
data/                           downloaded and processed data
experiments/                    scripts used to run each experiment
outputs/versions/               generated tables, figures, and diagnostics
src/calibration/                HF-ACI and baseline calibration methods
src/data/                       ENTSO-E download and preprocessing scripts
src/evaluation/metrics.py       evaluation metrics used in the dissertation
src/functional_pipeline.py      shared electricity-market experiment code
src/protocol.py                 shared market names, seeds, and default settings
tests/                          small regression and end-to-end tests
```

`MIGRATION.md` contains more detail about the earlier code cleanup and the
checks used to make sure the results did not change.

## Checking the code

Run the tests with:

```bash
python -m unittest discover -s tests
```

You can also check that all Python files compile:

```bash
python -m compileall -q src experiments tests
```

At the time of writing, the repository has 14 tests. The most important final
check is the `--check-only` command above because it verifies all four markets
without calculating any 2024 performance results.
