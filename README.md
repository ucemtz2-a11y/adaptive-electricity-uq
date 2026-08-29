# Hybrid Functional Adaptive Conformal Inference

Code for the dissertation **Adaptive Uncertainty Quantification for
Non-stationary Stochastic Features**. The repository implements Hybrid
Functional Adaptive Conformal Inference (HF-ACI) and the experiments reported
in the paper: synthetic benchmarks, four-market development experiments,
strong baselines, stochastic-feature perturbations, ablations,
theory-alignment simulations, and the frozen one-shot 2024 evaluation.

## Reproducibility contract

The empirical pipeline follows the paper exactly:

- Markets: DE-LU, DK1, DK2, and SE3.
- Development data: hourly observations from 2022--2023.
- Chronological development split: 60% training, 20% validation, and 20%
  development holdout.
- Final data: the 8,784 UTC hours of calendar year 2024.
- Base forecasts, context transformations, hyperparameters, evaluation maps,
  group definitions, and random-feature seeds are frozen before the 2024 run.
- Online methods predict first and update only after observing the corresponding
  outcome.

Do not tune models using the 2024 outputs. The `--check-only` mode verifies the
frozen historical reproduction without computing 2024 performance.

## Installation

Python 3.12 was used for the final experiments. Create an isolated environment
and install the pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For ENTSO-E downloads, copy `.env.example` to `.env` and set
`ENTSOE_API_KEY`. Raw data, processed data, and generated outputs are ignored
by Git because they are large or contain local paths.

## Data preparation

The multi-market download uses the date range in `config/config.yaml` for the
2022--2023 development data:

```bash
python src/data/download_entsoe_multi_market.py
python src/data/preprocess_multi_market.py
```

The final-year scripts write to separate directories so the historical files
cannot be overwritten:

```bash
python src/data/download_entsoe_multi_2024.py
python src/data/preprocess_entsoe_multi_2024.py
```

Expected processed files are:

```text
data/processed/multi_market/{DE_LU,DK_1,DK_2,SE_3}_dataset.csv
data/processed/multi_market_2024/{DE_LU,DK_1,DK_2,SE_3}_dataset.csv
```

## Paper experiments

Run the development-stage experiments in this order:

```bash
# Synthetic benchmark (v9)
python experiments/run_synthetic_functional_aci.py

# Four-market HF-ACI experiment and frozen hyperparameter selection (v10)
python experiments/run_multi_market_functional_aci.py

# Feature perturbation robustness analysis (v11)
python experiments/run_functional_perturbation.py

# Context, function-space, and kernel ablations (v12)
python experiments/run_functional_ablation.py

# Unified strong-baseline comparison (v13)
python experiments/run_strong_baselines_v10.py

# Development-result tables and figures (v14)
python experiments/run_final_summary.py

# Theory-alignment simulations (v15)
python experiments/run_theory_alignment.py
```

The longer development scripts provide `--quick` modes for smoke checks. These
reduced runs are not the paper results.

Verify the frozen historical pipeline before opening the final labels:

```bash
python experiments/run_final_untouched_2024.py --check-only
```

The confirmatory command is intentionally guarded by a lock file:

```bash
python experiments/run_final_untouched_2024.py --execute-once
```

The paper's first one-shot run produced a cross-market functional coverage
error of 0.0142 and a mean Winkler score of 69.58 for HF-ACI.

## Repository structure

```text
config/                         ENTSO-E date and market configuration
src/calibration/functional_aci.py
                                S-ACI, L-ACI, F-ACI, and HF-ACI
src/calibration/baselines.py    rolling, Split CQR, and ACS baselines
src/evaluation/metrics.py       authoritative paper metric implementations
src/functional_pipeline.py      shared market loader and frozen experiment pipeline
src/frozen_v10.py               frozen artefact loading and reproduction checks
src/protocol.py                 shared market metadata, seeds, and frozen defaults
src/theory_alignment.py         theory-alignment simulation and regret core
src/ablation_experiments.py     context/kernel ablation execution core
src/ablation_reporting.py       ablation aggregation and figures
src/perturbations.py            stochastic-feature perturbation computation
src/perturbation_reporting.py   perturbation figures
src/final_summary_data.py       paper-table preparation and headline findings
src/final_summary_reporting.py  final figures and LaTeX/text outputs
src/data/entsoe_config.py       shared ENTSO-E configuration/API-key loading
src/data/                       multi-market download and preprocessing
experiments/                    paper experiment entry points only
```

See `MIGRATION.md` for old-to-new path mappings, removed legacy files,
compatibility notes, equivalence checks, and remaining technical debt.

Generated tables and figures are written under `outputs/versions/` and are not
tracked by default.

## Verification

Run the lightweight unit tests and import checks with:

```bash
python -m unittest discover -s tests
python -m compileall -q src experiments
```

The most important end-to-end check is the frozen historical preflight shown
above; it regenerates the old raw intervals and verifies agreement with the
stored v10 predictions before any final-year metric is computed.