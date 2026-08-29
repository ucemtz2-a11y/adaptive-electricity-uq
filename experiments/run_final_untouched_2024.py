# Check the frozen historical pipeline and run the separate final 2024 evaluation.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Reuse the checked v10 method code so nothing is retuned from 2024 results.
from src.functional_pipeline import (  # noqa: E402
    build_context,
    convert_result,
    create_group_masks,
    create_lag_features,
    fit_evaluation_map,
    make_group_table,
    make_raw_predictions,
    make_raw_result,
    preprocess_context,
    transform_evaluation_map,
)
from src.evaluation.metrics import evaluate  # noqa: E402

from src.calibration.functional_aci import (  # noqa: E402
    FunctionalACI,
    HybridFunctionalACI,
    LinearContextualACI,
    ScalarACI,
)

# Use the same baseline formulas as v13 for a fair final comparison.
from src.calibration.baselines import (  # noqa: E402
    adaptive_conformal_score_interval,
    rolling_historical_interval,
    split_cqr_interval,
)

# Share the existing loader and v10 reproduction checks instead of copying them here.
from src.frozen_v10 import (  # noqa: E402
    assert_raw_prediction_match,
    interval_result,
    load_v10_selected_parameters,
    load_v10_test_predictions,
)
from src.protocol import (  # noqa: E402
    DEVELOPMENT_PROTOCOL,
    MARKET_NAME_MAP,
    PAPER_MARKETS,
    market_random_state,
)


FINAL_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
FINAL_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

LOCK_NAME = "FINAL_2024_EVALUATION_COMPLETED.lock"


# Require the user to choose either a safe preflight or the one-shot final run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot untouched-2024 evaluation using the exact v10/v13 "
            "pipeline, frozen hyperparameters and evaluation definitions."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Verify data, frozen hyperparameters and exact v10 reproduction "
            "using only the already-inspected 2022-2023 segment. "
            "No 2024 performance metric is computed."
        ),
    )
    mode.add_argument(
        "--execute-once",
        action="store_true",
        help="Run the untouched 2024 final evaluation exactly once.",
    )

    parser.add_argument(
        "--historical-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/multi_market",
    )
    parser.add_argument(
        "--final-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/multi_market_2024",
    )
    parser.add_argument(
        "--v10-results",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v10_multi_market_functional",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v16_final_untouched_2024",
    )
    parser.add_argument("--markets", nargs="*", default=list(PAPER_MARKETS))
    parser.add_argument("--alpha", type=float, default=DEVELOPMENT_PROTOCOL.alpha)
    parser.add_argument("--train-frac", type=float, default=DEVELOPMENT_PROTOCOL.train_fraction)
    parser.add_argument("--validation-frac", type=float, default=DEVELOPMENT_PROTOCOL.validation_fraction)
    parser.add_argument("--rolling-window", type=int, default=DEVELOPMENT_PROTOCOL.rolling_window)
    parser.add_argument("--random-state", type=int, default=DEVELOPMENT_PROTOCOL.random_state)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Allow a rerun after the lock exists. This should not be used "
            "if you want to retain a strict untouched-final-test claim."
        ),
    )

    return parser.parse_args()


# Hash each input file so the manifest records exactly which data was used.
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Read one processed CSV, sort UTC timestamps, and remove duplicate hours.
def read_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "datetime" not in df.columns:
        raise KeyError(
            f"{path} must contain a 'datetime' column. "
            f"Available: {list(df.columns)}"
        )

    required = ["price", "load", "wind", "solar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    df = (
        df.dropna(subset=["datetime"]).sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .set_index("datetime")
    )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "residual_load" not in df.columns:
        df["residual_load"] = (
            df["load"] - df["wind"] - df["solar"]
        )

    return df


# Join verified historical rows with 2024 rows while keeping their boundary visible.
def load_combined_market(
    historical_path: Path,
    final_path: Path,
) -> tuple[pd.DataFrame, str, list[str], int]:
    """
    Concatenate the original 2022-2023 processed data with 2024 FIRST,
    then call the original v10 lag constructor.

    This is important because price_lag_168 and the 24-hour physical lags at
    the start of 2024 must use the final hours of 2023.
    """
    old = read_processed(historical_path)
    new = read_processed(final_path)

    if (new.index < FINAL_START).any():
        raise ValueError(f"{final_path} contains timestamps before 2024.")

    combined = pd.concat([old, new], axis=0)
    combined = (
        combined.sort_index().loc[~combined.index.duplicated(keep="last")].copy()
    )

    target_col = "price"

    combined = combined.dropna(subset=[target_col])
    combined, model_features = create_lag_features(combined, target_col)

    combined = combined[[target_col, *model_features]].copy()

    # Count historical rows first, then rebuild the original 60/20/20 v10 slices.
    # This stops the appended 2024 rows from changing any development boundary.
    n_dev = int((combined.index < FINAL_START).sum())

    if n_dev < 2000:
        raise ValueError(f"Only {n_dev} usable pre-2024 observations remain.")

    final_mask = (
        (combined.index >= FINAL_START) & (combined.index < FINAL_END)
    )

    if int(final_mask.sum()) < 8000:
        raise ValueError(
            f"Only {int(final_mask.sum())} usable 2024 observations remain."
        )

    return combined, target_col, model_features, n_dev


# Return slices for the historical validation/test periods and the separate 2024 period.
def old_slices(
    n_dev: int,
    train_frac: float,
    validation_frac: float,
) -> tuple[slice, slice, slice]:
    train_end = int(n_dev * train_frac)
    validation_end = int(n_dev * (train_frac + validation_frac))

    if validation_end >= n_dev:
        raise ValueError("Invalid development split.")

    return (
        slice(0, train_end),
        slice(train_end, validation_end),
        slice(validation_end, n_dev),
    )


# Warm the ACI methods on historical data, then continue their state through 2024.
def run_online_methods_through_2024(
    selected: dict,
    alpha: float,
    random_state: int,
    train_context: np.ndarray,
    validation_context: np.ndarray,
    devtest_context: np.ndarray,
    final_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    lower_devtest: np.ndarray,
    upper_devtest: np.ndarray,
    y_devtest: np.ndarray,
    lower_final: np.ndarray,
    upper_final: np.ndarray,
    y_final: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Exact continuation of the frozen v10 online algorithms.

    Chronology:
      - validation: reset=True (same warm-up used by v10);
      - old 2023 development holdout: reset=False;
      - untouched 2024: reset=False, and ONLY this block is evaluated.

    No 2024 value is used to choose any hyperparameter.
    """
    B = float(selected["max_adjustment"])

    scalar = ScalarACI(alpha=alpha, eta=selected["scalar"]["eta"], max_adjustment=B)
    scalar.run(lower_validation, upper_validation, y_validation, reset=True)
    scalar.run(lower_devtest, upper_devtest, y_devtest, reset=False)
    scalar_final = scalar.run(lower_final, upper_final, y_final, reset=False)

    contextual = LinearContextualACI(
        alpha=alpha,
        eta_global=selected["contextual"]["eta_global"],
        eta_linear=selected["contextual"]["eta_linear"],
        linear_radius=selected["contextual"]["linear_radius"],
        max_adjustment=B,
    )
    contextual.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )
    contextual.run(
        lower_devtest,
        upper_devtest,
        y_devtest,
        devtest_context,
        reset=False,
    )
    contextual_final = contextual.run(
        lower_final,
        upper_final,
        y_final,
        final_context,
        reset=False,
    )

    functional = FunctionalACI(
        alpha=alpha,
        eta_global=selected["functional"]["eta_global"],
        eta_functional=selected["functional"]["eta_functional"],
        n_components=selected["functional"]["n_components"],
        length_scale=selected["functional"]["length_scale"],
        functional_radius=selected["functional"]["functional_radius"],
        max_adjustment=B,
        random_state=random_state,
    )
    functional.fit_feature_map(train_context)
    functional.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )
    functional.run(
        lower_devtest,
        upper_devtest,
        y_devtest,
        devtest_context,
        reset=False,
    )
    functional_final = functional.run(
        lower_final,
        upper_final,
        y_final,
        final_context,
        reset=False,
    )

    hybrid = HybridFunctionalACI(
        alpha=alpha,
        eta_global=selected["hybrid"]["eta_global"],
        eta_linear=selected["hybrid"]["eta_linear"],
        eta_functional=selected["hybrid"]["eta_functional"],
        n_components=selected["hybrid"]["n_components"],
        length_scale=selected["hybrid"]["length_scale"],
        residual_ridge=selected["hybrid"]["residual_ridge"],
        linear_radius=selected["hybrid"]["linear_radius"],
        functional_radius=selected["hybrid"]["functional_radius"],
        max_adjustment=B,
        random_state=random_state,
    )
    hybrid.fit_feature_map(train_context)
    hybrid.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )
    hybrid.run(lower_devtest, upper_devtest, y_devtest, devtest_context, reset=False)
    hybrid_final = hybrid.run(
        lower_final,
        upper_final,
        y_final,
        final_context,
        reset=False,
    )

    return {
        "Scalar ACI": convert_result(scalar_final),
        "Linear contextual ACI": convert_result(contextual_final),
        "Functional ACI": convert_result(functional_final),
        "Hybrid Functional ACI": convert_result(hybrid_final),
    }


# Reproduce one market's saved v10 raw intervals without evaluating any 2024 label.
def preflight_one_market(
    prefix: str,
    args: argparse.Namespace,
) -> dict:
    """
    Strict preflight using only the already-inspected development period.

    It verifies:
      - 2024 data exist and have the right rough time coverage;
      - the frozen market-specific v10 JSON exists;
      - regenerating the OLD v10 raw test intervals is numerically identical
        to the stored v10 prediction file.

    It does NOT compute a 2024 coverage/Winkler/functional metric.
    """
    historical_path = (
        args.historical_data_dir / f"{prefix}_dataset.csv"
    )
    final_path = (
        args.final_data_dir / f"{prefix}_dataset.csv"
    )

    if not historical_path.exists():
        raise FileNotFoundError(historical_path)
    if not final_path.exists():
        raise FileNotFoundError(final_path)

    # Read historical data only during preflight.
    final_raw = read_processed(final_path)

    final_2024 = final_raw.loc[
        (final_raw.index >= FINAL_START) & (final_raw.index < FINAL_END)
    ]

    if len(final_2024) < 8000:
        raise ValueError(f"{prefix}: only {len(final_2024)} raw 2024 rows.")

    selected = load_v10_selected_parameters(args.v10_results, prefix)

    required_top = {
        "alpha",
        "max_adjustment",
        "scalar",
        "contextual",
        "functional",
        "hybrid",
    }
    missing = sorted(required_top - set(selected.keys()))
    if missing:
        raise KeyError(f"{prefix}: frozen v10 JSON missing {missing}")

    # Load the saved v10 rows that the regenerated predictions must match.
    from src.functional_pipeline import (
        chronological_split,
        load_market_dataset,
    )

    old_df, target_col, features = load_market_dataset(historical_path)
    # Rebuild the original historical train, validation, and test slices.
    train_slice, _, test_slice = chronological_split(
        len(old_df),
        args.train_frac,
        args.validation_frac,
    )

    seed = market_random_state(prefix, args.random_state)

    # Regenerate raw bounds using the same v10 models and market seed.
    old_predictions = make_raw_predictions(
        df=old_df,
        target_col=target_col,
        features=features,
        train_slice=train_slice,
        alpha=args.alpha,
        random_state=seed,
    )

    stored_v10 = load_v10_test_predictions(args.v10_results, prefix)

    match = assert_raw_prediction_match(
        generated_predictions=old_predictions,
        test_slice=test_slice,
        stored_v10=stored_v10,
        market=MARKET_NAME_MAP.get(prefix, prefix),
    )

    return {
        "market": MARKET_NAME_MAP.get(prefix, prefix),
        "historical_file": str(historical_path),
        "historical_sha256": sha256_file(historical_path),
        "final_2024_file": str(final_path),
        "final_2024_sha256": sha256_file(final_path),
        "raw_2024_rows": int(len(final_2024)),
        "raw_2024_start": str(final_2024.index.min()),
        "raw_2024_end": str(final_2024.index.max()),
        "selected_hyperparameters_sha256": sha256_file(
            args.v10_results / "diagnostics" / f"{prefix}_selected_hyperparameters.json"
        ),
        "old_v10_raw_match": match,
    }


# Run the full frozen pipeline for one market after preflight has passed.
def execute_one_market(
    prefix: str,
    args: argparse.Namespace,
    tables_dir: Path,
    predictions_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    market = MARKET_NAME_MAP.get(prefix, prefix)
    print("\n" + "=" * 90)
    print(f"FINAL UNTOUCHED 2024: {market}")
    print("=" * 90)

    historical_path = (
        args.historical_data_dir / f"{prefix}_dataset.csv"
    )
    final_path = (
        args.final_data_dir / f"{prefix}_dataset.csv"
    )

    # Load historical and final-year rows without changing their UTC order.
    df, target_col, model_features, n_dev = (
        load_combined_market(historical_path, final_path)
    )

    # Keep historical splits fixed and place all 2024 rows in one final slice.
    train_slice, validation_slice, devtest_slice = (
        old_slices(n_dev, args.train_frac, args.validation_frac)
    )

    final_slice = slice(n_dev, len(df))

    seed = market_random_state(prefix, args.random_state)

    # Base quantile models still train only on the original historical training slice.
    predictions = make_raw_predictions(
        df=df,
        target_col=target_col,
        features=model_features,
        train_slice=train_slice,
        alpha=args.alpha,
        random_state=seed,
    )

    # Stop before final evaluation if historical raw predictions have changed at all.
    stored_v10 = load_v10_test_predictions(args.v10_results, prefix)
    # Once reproduction passes, use those same models to predict the appended 2024 rows.
    raw_match = assert_raw_prediction_match(
        generated_predictions=predictions,
        test_slice=devtest_slice,
        stored_v10=stored_v10,
        market=market,
    )

    # Fit context preprocessing on historical training rows only.
    context = build_context(predictions)
    context_scaled = preprocess_context(context, train_slice)

    train_context = context_scaled[train_slice]
    validation_context = context_scaled[validation_slice]
    devtest_context = context_scaled[devtest_slice]
    final_context = context_scaled[final_slice]

    final_index = predictions.index[final_slice]

    if not ((final_index >= FINAL_START) & (final_index < FINAL_END)).all():
        raise RuntimeError(f"{market}: final slice is not exactly inside 2024.")

    # Reuse the v10 evaluation-feature seed and training context.
    evaluation_feature_map = fit_evaluation_map(train_context, seed + 10_000)
    # Transform 2024 context with the already fitted evaluation map.
    evaluation_map_final = transform_evaluation_map(
        evaluation_feature_map,
        final_context,
    )

    # Group thresholds come from historical training context, never from 2024 outcomes.
    final_group_masks = create_group_masks(
        index=final_index,
        context_part=context.iloc[final_slice],
        context_train=context.iloc[train_slice],
    )

    y_validation = (
        predictions["y_true"].iloc[validation_slice].to_numpy()
    )
    lower_validation = (
        predictions["lower_raw"].iloc[validation_slice].to_numpy()
    )
    upper_validation = (
        predictions["upper_raw"].iloc[validation_slice].to_numpy()
    )

    y_devtest = (
        predictions["y_true"].iloc[devtest_slice].to_numpy()
    )
    lower_devtest = (
        predictions["lower_raw"].iloc[devtest_slice].to_numpy()
    )
    upper_devtest = (
        predictions["upper_raw"].iloc[devtest_slice].to_numpy()
    )

    y_final_series = (
        predictions["y_true"].iloc[final_slice]
    )
    y_final = y_final_series.to_numpy()
    lower_final_series = (
        predictions["lower_raw"].iloc[final_slice]
    )
    upper_final_series = (
        predictions["upper_raw"].iloc[final_slice]
    )
    lower_final = lower_final_series.to_numpy()
    upper_final = upper_final_series.to_numpy()

    selected = load_v10_selected_parameters(args.v10_results, prefix)

    if abs(float(selected["alpha"]) - args.alpha) > 1e-12:
        raise RuntimeError(f"{market}: frozen alpha differs from requested alpha.")

    method_results = {"Raw quantile": make_raw_result(predictions.iloc[final_slice])}

    # Replay the verified historical test period first, then enter 2024 without resetting.
    # This carries forward online state but never changes the frozen hyperparameters.
    method_results.update(
        run_online_methods_through_2024(
            selected=selected,
            alpha=args.alpha,
            random_state=seed,
            train_context=train_context,
            validation_context=validation_context,
            devtest_context=devtest_context,
            final_context=final_context,
            lower_validation=lower_validation,
            upper_validation=upper_validation,
            y_validation=y_validation,
            lower_devtest=lower_devtest,
            upper_devtest=upper_devtest,
            y_devtest=y_devtest,
            lower_final=lower_final,
            upper_final=upper_final,
            y_final=y_final,
        )
    )

    # Run the three v13 baselines with their original definitions and update order.

    # Rolling intervals use only prices observed before the hour being predicted.
    lower_roll, upper_roll = rolling_historical_interval(
        y=df[target_col],
        test_index=final_index,
        window=args.rolling_window,
        alpha=args.alpha,
    )
    method_results["Rolling historical quantile"] = (
        interval_result(y_final_series, lower_roll, upper_roll)
    )

    # Split CQR learns its one expansion from the historical validation intervals.
    y_cal = predictions["y_true"].iloc[validation_slice]
    lower_cal = predictions["lower_raw"].iloc[validation_slice]
    upper_cal = predictions["upper_raw"].iloc[validation_slice]

    lower_cqr, upper_cqr, qhat = split_cqr_interval(
        y_cal=y_cal,
        lower_cal=lower_cal,
        upper_cal=upper_cal,
        lower_test=lower_final_series,
        upper_test=upper_final_series,
        alpha=args.alpha,
    )
    method_results["Split CQR"] = interval_result(
        y_final_series,
        lower_cqr,
        upper_cqr,
    )

    # Adaptive scores keep the v13 initial value and learning-rate formulas.
    # They also replay historical test feedback before continuing into 2024.
    theta_max = float(selected["max_adjustment"])

    # eta uses only the known 2024 horizon length and is fixed before seeing performance.
    eta_aci = (
        theta_max / (10.0 * np.sqrt(len(y_final_series)))
    )

    continuation_index = predictions.index[slice(devtest_slice.start, final_slice.stop)]
    y_cont = predictions["y_true"].loc[continuation_index]
    lo_cont = predictions["lower_raw"].loc[continuation_index]
    hi_cont = predictions["upper_raw"].loc[continuation_index]

    aci_cont = adaptive_conformal_score_interval(
        y_test=y_cont,
        lower_test=lo_cont,
        upper_test=hi_cont,
        q_init=float(qhat),
        alpha=args.alpha,
        eta=eta_aci,
        q_min=0.0,
        q_max=theta_max,
    )

    aci_final = aci_cont.loc[final_index]
    method_results["Adaptive conformal score"] = (
        interval_result(
            y_final_series,
            aci_final["lower_adaptive_conformal"],
            aci_final["upper_adaptive_conformal"],
        )
    )

    # Evaluate all final methods with the same v10 metrics and group definitions.
    summary_rows = []

    for method_name, result in method_results.items():
        metrics = evaluate(
            method_name=method_name,
            y=y_final,
            result=result,
            evaluation_map=evaluation_map_final,
            group_masks=final_group_masks,
            alpha=args.alpha,
        )
        metrics["market"] = market
        metrics["qhat"] = (
            float(qhat)
            if method_name in {"Split CQR", "Adaptive conformal score"}
            else np.nan
        )
        metrics["eta"] = (
            float(eta_aci) if method_name == "Adaptive conformal score" else np.nan
        )
        summary_rows.append(metrics)

    # Keep one summary row per method and market before cross-market averaging.
    summary = pd.DataFrame(summary_rows)

    group_table = make_group_table(
        market=market,
        y=y_final,
        method_results=method_results,
        group_masks=final_group_masks,
        target_coverage=1.0 - args.alpha,
    )

    # Save hourly intervals and online components so individual results can be checked.
    pred = pd.DataFrame(
        {
            "datetime": final_index,
            "y_true": y_final,
            "lower_raw_quantile": lower_final,
            "upper_raw_quantile": upper_final,
        }
    )

    for method_name, result in method_results.items():
        safe = (
            method_name.lower().replace(" ", "_").replace("-", "_")
        )
        pred[f"lower_{safe}"] = result["lower"]
        pred[f"upper_{safe}"] = result["upper"]
        pred[f"adjustment_{safe}"] = result["adjustment"]
        pred[f"linear_component_{safe}"] = result["linear_component"]
        pred[f"functional_component_{safe}"] = result["functional_component"]

    # Include data hashes and reproduction differences in the market diagnostics.
    pred.to_csv(predictions_dir / f"{prefix}_final_2024_predictions.csv", index=False)

    summary.to_csv(tables_dir / f"{prefix}_final_2024_summary.csv", index=False)

    group_table.to_csv(tables_dir / f"{prefix}_final_2024_groups.csv", index=False)

    meta = {
        "market": market,
        "n_train": train_slice.stop - train_slice.start,
        "n_validation": (
            validation_slice.stop - validation_slice.start
        ),
        "n_old_development_holdout": (
            devtest_slice.stop - devtest_slice.start
        ),
        "n_final_2024": len(y_final),
        "train_start": str(df.index[train_slice.start]),
        "train_end": str(df.index[train_slice.stop - 1]),
        "validation_start": str(df.index[validation_slice.start]),
        "validation_end": str(df.index[validation_slice.stop - 1]),
        "old_development_holdout_start": str(df.index[devtest_slice.start]),
        "old_development_holdout_end": str(df.index[devtest_slice.stop - 1]),
        "final_start": str(final_index[0]),
        "final_end": str(final_index[-1]),
        "split_cqr_qhat": float(qhat),
        "adaptive_conformal_eta": float(eta_aci),
        "adaptive_conformal_theta_max": float(theta_max),
        "old_v10_raw_match": raw_match,
    }

    print(
        summary[
            [
                "method",
                "coverage",
                "coverage_error",
                "avg_width",
                "winkler",
                "functional_error",
                "worst_group_error",
            ]
        ].to_string(index=False)
    )

    return summary, group_table, meta


# Run safe preflight for every market, then stop or continue according to the chosen flag.
def main() -> None:
    args = parse_args()

    # Read the requested mode and prepare the output folder without touching source data.
    args.output.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output / "tables"
    # Record input file hashes before any model or metric is run.
    predictions_dir = tables_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    lock_path = args.output / LOCK_NAME

    if (
        args.execute_once and lock_path.exists() and not args.force_rerun
    ):
        raise RuntimeError(
            "\nFINAL 2024 EVALUATION HAS ALREADY BEEN RUN.\n"
            f"Lock file: {lock_path}\n\n"
            "Do not rerun/tune on 2024 if you want to retain the "
            "untouched-final-test claim."
        )

    # Always reproduce all requested historical markets before opening final results.
    manifest = {
        "protocol": (
            "Frozen v10/v13 pipeline; old 60/20/20 development split "
            "reconstructed exactly; online methods continue through the "
            "already-inspected old development holdout; metrics are reported "
            "only on 2024."
        ),
        "alpha": args.alpha,
        "train_frac": args.train_frac,
        "validation_frac": args.validation_frac,
        "rolling_window": args.rolling_window,
        "random_state": args.random_state,
        "markets": {},
    }

    for prefix in args.markets:
        print(f"\nPreflight: {prefix}")
        manifest["markets"][prefix] = (
            preflight_one_market(prefix, args)
        )
        print(
            "  exact old v10 raw prediction reproduction: PASS"
        )

    manifest_path = (
        args.output / "protocol_manifest_PRE_2024_EVALUATION.json"
    )
    # Save a PRE manifest even when the command stops after the safe check.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    if args.check_only:
        print("\n" + "=" * 90)
        print("PRE-FLIGHT PASSED.")
        print("No 2024 performance metric was computed.")
        print(f"Manifest: {manifest_path}")
        print("=" * 90)
        return

    if args.force_rerun:
        (
            args.output / "WARNING_FORCED_RERUN.txt"
        ).write_text(
            "This is a forced rerun after the final 2024 test was already "
            "opened. Any model decision made after the first run must not "
            "be described as based on an untouched 2024 test.",
            encoding="utf-8",
        )

    # The execute-once path starts only after every historical preflight has passed.
    all_summaries = []
    all_groups = []
    metadata = {}

    for prefix in args.markets:
        summary, groups, meta = execute_one_market(
            prefix,
            args,
            tables_dir,
            predictions_dir,
        )
        all_summaries.append(summary)
        all_groups.append(groups)
        metadata[prefix] = meta

    # Average market results at the end, while keeping all market-level rows.
    full_summary = pd.concat(all_summaries, ignore_index=True)
    full_groups = pd.concat(all_groups, ignore_index=True)

    full_summary.to_csv(tables_dir / "final_2024_all_market_summary.csv", index=False)
    full_groups.to_csv(tables_dir / "final_2024_all_market_groups.csv", index=False)

    numeric_cols = [
        "coverage",
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
        "mean_adjustment",
        "mean_abs_linear_component",
        "mean_abs_functional_component",
    ]

    average = (
        full_summary.groupby("method", as_index=False)[numeric_cols].mean()
        .sort_values(
            ["functional_error", "worst_group_error"]
        )
        .reset_index(drop=True)
    )

    average.to_csv(tables_dir / "final_2024_cross_market_average.csv", index=False)

    (
        args.output / "final_2024_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    lock_payload = {
        "status": "FINAL_2024_EVALUATION_COMPLETED",
        "strict_untouched_claim": (
            not args.force_rerun
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "note": (
            "The 2024 labels were used only for the final sequential "
            "evaluation. Online updates within 2024 are part of the "
            "pre-specified deployed algorithm."
        ),
    }

    lock_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("FINAL 2024 CROSS-MARKET AVERAGE")
    print("=" * 90)
    print(average.to_string(index=False))
    print(f"\nSaved to: {args.output}")
    print(f"LOCK WRITTEN: {lock_path}")


if __name__ == "__main__":
    main()
