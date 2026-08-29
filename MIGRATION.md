# Refactoring and migration log

This log records structural changes made while preserving the numerical
behaviour of the dissertation experiments. No loss, update equation, feature,
split, seed, hyperparameter, metric formula, or output schema was intentionally
changed.

## Current authoritative modules

| Responsibility | Current path | Previous location |
|---|---|---|
| S-ACI, L-ACI, F-ACI, HF-ACI | `src/calibration/functional_aci.py` | unchanged; evaluation helpers moved out |
| Rolling, Split CQR, ACS baselines | `src/calibration/baselines.py` | helper functions in `experiments/run_strong_baselines.py` |
| All paper metrics | `src/evaluation/metrics.py` | split between `src/calibration/functional_aci.py` and `src/functional_pipeline.py` |
| Shared real-market loader and pipeline | `src/functional_pipeline.py` | repeated market paths, candidate scaling, selection objective, and pipeline logic |
| Frozen v10 artefact checks | `src/frozen_v10.py` | `experiments/run_strong_baselines_v10.py` |
| Shared market metadata, seeds, and protocol defaults | `src/protocol.py` | repeated market lists, display names, seed derivation, and argparse defaults |
| ENTSO-E configuration and API-key loading | `src/data/entsoe_config.py` | duplicated in both retained download scripts |
| Theory-alignment core | `src/theory_alignment.py` | `experiments/run_theory_alignment.py` |
| Ablation computation | `src/ablation_experiments.py` | `experiments/run_functional_ablation.py` |
| Ablation reporting | `src/ablation_reporting.py` | `experiments/run_functional_ablation.py` |
| Perturbation computation | `src/perturbations.py` | `experiments/run_functional_perturbation.py` |
| Perturbation reporting | `src/perturbation_reporting.py` | `experiments/run_functional_perturbation.py` |
| Final-summary preparation | `src/final_summary_data.py` | `experiments/run_final_summary.py` |
| Final-summary reporting | `src/final_summary_reporting.py` | `experiments/run_final_summary.py` |

The original experiment command paths remain valid. Output directory and file
names are intentionally unchanged for compatibility with the frozen results.

## Removed superseded code

The following files were empty, development-only, or superseded by the final
v9--v16 paper pipeline and had no remaining imports from retained code:

```text
experiments/run_baseline.py
experiments/run_contextual_calibration.py
experiments/run_functional_aci.py
experiments/run_multi_market_contextual.py
experiments/run_online_calibration.py
experiments/run_perturbation_experiment.py
experiments/run_quantile_model.py
experiments/run_strong_baselines.py
src/calibration/contextual_calibration.py
src/calibration/loss.py
src/calibration/online_calibration.py
src/evaluation/drift_analysis.py
src/evaluation/plot_perturbation.py
src/evaluation/plot_strong_baselines.py
src/evaluation/plots.py
src/features/build_features.py
src/models/baseline_models.py
src/models/quantile_lgbm.py
src/models/quantile_xgb.py
src/utils/time_utils.py
src/data/download_entsoe.py
src/data/preprocess.py
```

No datasets or generated results were deleted. A temporary pre-refactor source
backup was retained at `/tmp/electricity-uq-project-before-comments` during the
refactoring session; `/tmp` should not be treated as permanent storage.

## Repository scale after migration

- Python source and tests: 11,762 lines across 33 files.
- Production Python under `src/` and `experiments/`: 11,288 lines.
- The temporary pre-refactor snapshot contained 18,622 Python lines.
- No retained Python file is above 2,000 lines; the largest entry point is now
  1,054 lines. Line count was treated as a diagnostic rather
  than the optimization objective.
- Python comments are English-only. Short literal collections and expressions
  are kept on one line when they fit within an 88-character limit.

## Second-pass simplification

- Merged the one-function `src/model_selection.py` module into the shared
  functional pipeline, which is its only consumer context.
- Centralized paper/download market order, display names, and deterministic
  market seed derivation in `src/protocol.py`.
- Consolidated the duplicated local learning-rate candidate helper.
- Removed directory/result wrappers that only forwarded one call.
- Removed result-column aliases used only by superseded v0--v6 outputs; active
  v9--v16 result schemas already use the canonical names.
- Removed unused legacy YAML sections and generated caches left by deleted
  modules. Download/preprocessing variants remain separate because their date
  guards and failure behaviour intentionally differ.

## Scientific-equivalence checks

- Every mechanically moved function was compared by AST before and after the
  move.
- Comment translation and expression compaction preserved the semantic AST of
  every Python file exactly.
- The metric module was compared numerically against a saved pre-migration
  baseline.
- Theory-alignment quick-run CSV outputs were exactly equal before and after.
- Final-summary CSV, JSON, LaTeX, and generated README text were exactly equal.
- Perturbation quick-run CSV and JSON outputs were exactly equal.
- Synthetic quick-run CSV, JSON, and figure files were byte-for-byte equal.
- Ablation quick-run outputs were exactly equal except for wall-clock runtime
  fields, which are inherently variable.
- The frozen historical preflight reproduced v10 raw predictions for all four
  markets without computing 2024 metrics.
- Fourteen lightweight tests cover metrics, calibration updates, fixed-seed
  determinism, market loading, baselines, and a small end-to-end path.

## Post-refactor bug fix

After the two behaviour-preserving refactoring passes were complete, a separate
bug-fix phase restored the multi-market quick entry point. The script called the
canonical `src.functional_pipeline.convert_result` helper during tuning but had
not imported it, so execution stopped with `NameError` in `tune_scalar`.

- The production fix is one added import in
  `experiments/run_multi_market_functional_aci.py`; the shared implementation
  was reused without modification or duplication.
- A lightweight regression test verifies that the entry point uses the shared
  converter and executes the previously failing scalar-tuning path.
- The single-market quick run now completes, all 14 tests pass, all retained
  modules compile and import, perturbation rows match the frozen outputs,
  repeated ablation outputs match except for runtime fields, and the four-market
  frozen historical preflight passes without computing 2024 metrics.

This was a behaviour-restoring engineering fix, not part of the earlier
equivalence-preserving refactor. No scientific logic or experiment setting was
changed.

## Compatibility and intentional differences

- Versioned output names such as `results_v10_*` and the historical script name
  `run_strong_baselines_v10.py` are retained because later stages reference
  those frozen artefacts.
- Synthetic and theory-alignment experiments intentionally keep their own
  split and random-seed defaults. `src/protocol.py` applies only to the shared
  real-market protocol.
- Functional error uses a smooth fixed evaluation map, while worst-group error
  uses hard prespecified group indicators. Their differing behaviour is
  intentional and follows the paper.
- The synthetic method construction remains explicit rather than using a
  factory: its calibrators require materially different constructor arguments,
  and the explicit branches make those experiment settings easier to audit.
- Runtime columns are not deterministic and are excluded only when checking
  numerical equivalence of model results.

## Remaining technical debt

- `run_multi_market_functional_aci.py` still contains experiment-specific
  hyperparameter tuning and plotting in one entry point. It is explicit and
  scientifically readable, but could be split further after another output
  regression check.
- Synthetic plotting remains in the synthetic experiment script.
- The v14 final-summary generator consumes v9--v13 development outputs; v15
  theory-alignment and v16 final-2024 reporting remain separate by design.
- ENTSO-E historical download and final-year download have slightly different
  failure handling. This behaviour was preserved rather than silently changed.
- The calibration classes share a small number of similarly named projection
  methods, but their state and mathematical roles differ. They were not merged
  into an inheritance hierarchy solely to reduce line count.

No confirmed scientific bug was found during refactoring. Potential behavioural
inconsistencies listed above were documented and intentionally left unchanged.
