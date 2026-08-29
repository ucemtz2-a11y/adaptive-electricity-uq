# Module purpose: Reproduce frozen v10 predictions and add standardized strong baselines.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.functional_pipeline import (  # noqa: E402
    build_context,
    chronological_split,
    create_group_masks,
    fit_evaluation_map,
    load_market_data,
    make_raw_predictions,
    preprocess_context,
    transform_evaluation_map,
)
from src.evaluation.metrics import evaluate  # noqa: E402

# Reuse the exact v6 strong-baseline definitions to avoid version drift.
from src.calibration.baselines import (  # noqa: E402
    adaptive_conformal_score_interval,
    rolling_historical_interval,
    split_cqr_interval,
)
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


# Parse args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the v6 strong baselines using exactly the same raw "
            "quantile predictions, chronological splits and evaluation "
            "framework as v10."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "processed" / "multi_market"
        ),
    )

    parser.add_argument(
        "--v10-results",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/versions/results_v10_multi_market_functional"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/versions/results_v13_strong_baselines_unified"
        ),
    )

    parser.add_argument("--markets", nargs="*", default=list(PAPER_MARKETS))

    parser.add_argument("--alpha", type=float, default=DEVELOPMENT_PROTOCOL.alpha)
    parser.add_argument("--train-frac", type=float, default=DEVELOPMENT_PROTOCOL.train_fraction)
    parser.add_argument("--validation-frac", type=float, default=DEVELOPMENT_PROTOCOL.validation_fraction)
    parser.add_argument("--rolling-window", type=int, default=DEVELOPMENT_PROTOCOL.rolling_window)
    parser.add_argument("--random-state", type=int, default=DEVELOPMENT_PROTOCOL.random_state)

    return parser.parse_args()










# Evaluate baseline.
def evaluate_baseline(
    market: str,
    method_name: str,
    y_test: pd.Series,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
    evaluation_map_test: np.ndarray,
    test_group_masks: dict[str, np.ndarray],
    alpha: float,
    eta: float | None = None,
    qhat: float | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    result = interval_result(y_true=y_test, lower=lower, upper=upper)

    metrics = evaluate(
        method_name=method_name,
        y=y_test.to_numpy(),
        result=result,
        evaluation_map=evaluation_map_test,
        group_masks=test_group_masks,
        alpha=alpha,
    )

    metrics["market"] = market
    metrics["eta"] = (
        np.nan if eta is None else float(eta)
    )
    metrics["qhat"] = (
        np.nan if qhat is None else float(qhat)
    )

    return metrics, result


# Run one market.
def run_one_market(
    market_prefix: str,
    args: argparse.Namespace,
    tables_dir: Path,
    diagnostics_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = MARKET_NAME_MAP.get(market_prefix, market_prefix)

    print("\n" + "=" * 90)
    print(f"Unified strong baselines: {market}")

    # Load inputs or existing results and normalize them for downstream processing.
    (
        df,
        target_col,
        model_features,
    ) = load_market_data(market_prefix, args.data_dir)

    # Split chronologically into explicit training, validation, and test periods.
    (
        train_slice,
        validation_slice,
        test_slice,
    ) = chronological_split(len(df), args.train_frac, args.validation_frac)

    market_seed = market_random_state(market_prefix, args.random_state)

    # Reproduce the v10 raw quantile models under frozen settings.
    predictions = make_raw_predictions(
        df=df,
        target_col=target_col,
        features=model_features,
        train_slice=train_slice,
        alpha=args.alpha,
        random_state=market_seed,
    )

    # Train base quantile models and generate uncalibrated prediction intervals.
    stored_v10 = load_v10_test_predictions(args.v10_results, market_prefix)

    match_diagnostics = (
        assert_raw_prediction_match(
            generated_predictions=predictions,
            test_slice=test_slice,
            stored_v10=stored_v10,
            market=market,
        )
    )

    # Save result tables, prediction details, and reproducible diagnostics.
    (
        diagnostics_dir / f"{market_prefix}_raw_match.json"
    ).write_text(
        json.dumps(match_diagnostics, indent=2),
        encoding="utf-8",
    )

    print("Verified exact v10 raw-interval match.")

    # Build and scale context features for conditional calibration.
    context = build_context(predictions)

    context_scaled = preprocess_context(context, train_slice)

    train_context = context_scaled[train_slice]

    test_context = context_scaled[test_slice]

    test_index = predictions.index[test_slice]

    # Build the functional-error map and conditional groups for common evaluation.
    evaluation_feature_map = fit_evaluation_map(train_context, market_seed + 10_000)

    evaluation_map_test = (
        transform_evaluation_map(evaluation_feature_map, test_context)
    )

    test_group_masks = create_group_masks(
        index=test_index,
        context_part=context.iloc[test_slice],
        context_train=context.iloc[train_slice],
    )

    y_cal = predictions["y_true"].iloc[validation_slice]

    lower_cal = predictions["lower_raw"].iloc[validation_slice]

    upper_cal = predictions["upper_raw"].iloc[validation_slice]

    y_test = predictions["y_true"].iloc[test_slice]

    lower_test = predictions["lower_raw"].iloc[test_slice]

    upper_test = predictions["upper_raw"].iloc[test_slice]

    selected_v10 = load_v10_selected_parameters(args.v10_results, market_prefix)

    theta_max = float(selected_v10["max_adjustment"])

    records = []
    prediction_columns = {
        "datetime": test_index,
        "y_true": y_test.to_numpy(),
        "lower_raw": lower_test.to_numpy(),
        "upper_raw": upper_test.to_numpy(),
    }

    # 1. Rolling historical quantiles use only realized prices.
    lower_roll, upper_roll = (
        rolling_historical_interval(
            y=df[target_col],
            test_index=y_test.index,
            window=args.rolling_window,
            alpha=args.alpha,
        )
    )

    # Compute coverage, interval-efficiency, and conditional-coverage metrics.
    rolling_metrics, _ = evaluate_baseline(
        market=market,
        method_name=(
            "Rolling historical quantile"
        ),
        y_test=y_test,
        lower=lower_roll,
        upper=upper_roll,
        evaluation_map_test=(
            evaluation_map_test
        ),
        test_group_masks=(
            test_group_masks
        ),
        alpha=args.alpha,
    )

    records.append(rolling_metrics)

    prediction_columns[
        "lower_rolling_historical"
    ] = np.asarray(lower_roll, dtype=float)

    prediction_columns[
        "upper_rolling_historical"
    ] = np.asarray(upper_roll, dtype=float)

    # 2. Split CQR.
    (
        lower_cqr,
        upper_cqr,
        qhat,
    ) = split_cqr_interval(
        y_cal=y_cal,
        lower_cal=lower_cal,
        upper_cal=upper_cal,
        lower_test=lower_test,
        upper_test=upper_test,
        alpha=args.alpha,
    )

    cqr_metrics, _ = evaluate_baseline(
        market=market,
        method_name="Split CQR",
        y_test=y_test,
        lower=lower_cqr,
        upper=upper_cqr,
        evaluation_map_test=(
            evaluation_map_test
        ),
        test_group_masks=(
            test_group_masks
        ),
        alpha=args.alpha,
        qhat=float(qhat),
    )

    records.append(cqr_metrics)

    prediction_columns[
        "lower_split_cqr"
    ] = np.asarray(lower_cqr, dtype=float)

    prediction_columns[
        "upper_split_cqr"
    ] = np.asarray(upper_cqr, dtype=float)

    # 3. Adaptive conformal scores update online from observed feedback.
    eta_aci = (
        theta_max / (10.0 * np.sqrt(len(y_test)))
    )

    aci_df = (
        adaptive_conformal_score_interval(
            y_test=y_test,
            lower_test=lower_test,
            upper_test=upper_test,
            q_init=float(qhat),
            alpha=args.alpha,
            eta=eta_aci,
            q_min=0.0,
            q_max=theta_max,
        )
    )

    aci_metrics, _ = evaluate_baseline(
        market=market,
        method_name=(
            "Adaptive conformal score"
        ),
        y_test=y_test,
        lower=aci_df["lower_adaptive_conformal"],
        upper=aci_df["upper_adaptive_conformal"],
        evaluation_map_test=(
            evaluation_map_test
        ),
        test_group_masks=(
            test_group_masks
        ),
        alpha=args.alpha,
        eta=eta_aci,
        qhat=float(qhat),
    )

    records.append(aci_metrics)

    prediction_columns[
        "lower_adaptive_conformal"
    ] = aci_df["lower_adaptive_conformal"].to_numpy()

    prediction_columns[
        "upper_adaptive_conformal"
    ] = aci_df["upper_adaptive_conformal"].to_numpy()

    if "q" in aci_df.columns:
        prediction_columns[
            "q_adaptive_conformal"
        ] = aci_df["q"].to_numpy()

    elif "theta" in aci_df.columns:
        prediction_columns[
            "q_adaptive_conformal"
        ] = aci_df["theta"].to_numpy()

    prediction_table = pd.DataFrame(prediction_columns)

    prediction_table.to_csv(
        tables_dir / (f"{market_prefix}_" "unified_strong_baseline_" "predictions.csv"),
        index=False,
    )

    result_table = pd.DataFrame(records)

    result_table = result_table[
        [
            "market",
            "method",
            "coverage",
            "coverage_error",
            "avg_width",
            "median_width",
            "winkler",
            "functional_error",
            "worst_group_error",
            "eta",
            "qhat",
        ]
    ]

    print(result_table.to_string(index=False))

    return result_table, prediction_table


# Append v10 methods.
def append_v10_methods(
    strong_results: pd.DataFrame,
    v10_results: Path,
) -> pd.DataFrame:
    path = (
        v10_results / "tables" / "multi_market_results.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing v10 result table: {path}")

    v10 = pd.read_csv(path)

    keep = [
        "market",
        "method",
        "coverage",
        "coverage_error",
        "avg_width",
        "median_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    v10 = v10[keep].copy()
    v10["eta"] = np.nan
    v10["qhat"] = np.nan
    v10["source"] = "v10 frozen methods"

    strong = strong_results.copy()
    strong["source"] = (
        "re-run on exact v10 raw intervals"
    )

    return pd.concat([v10, strong], ignore_index=True)


# Make average summary.
def make_average_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "coverage",
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    grouped = (
        results.groupby("method")[metrics].agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    return (
        grouped.reset_index().sort_values(["functional_error_mean", "winkler_mean"])
    )


# Make pairwise hybrid comparison.
def make_pairwise_hybrid_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    hybrid = (
        results[results["method"] == "Hybrid Functional ACI"][
            [
                "market",
                "coverage_error",
                "avg_width",
                "winkler",
                "functional_error",
                "worst_group_error",
            ]
        ]
        .rename(
            columns={
                "coverage_error": (
                    "hybrid_coverage_error"
                ),
                "avg_width": (
                    "hybrid_avg_width"
                ),
                "winkler": (
                    "hybrid_winkler"
                ),
                "functional_error": (
                    "hybrid_functional_error"
                ),
                "worst_group_error": (
                    "hybrid_worst_group_error"
                ),
            }
        )
    )

    competitors = results[results["method"] != "Hybrid Functional ACI"].copy()

    merged = competitors.merge(hybrid, on="market", how="left")

    for metric in [
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]:
        merged[
            f"hybrid_minus_competitor_{metric}"
        ] = (
            merged[f"hybrid_{metric}"] - merged[metric]
        )

    return merged


# Make figures.
def make_figures(
    results: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in [
        (
            "functional_error",
            "Functional coverage error",
            "unified_baselines_functional_error.png",
        ),
        (
            "worst_group_error",
            "Worst-group coverage error",
            "unified_baselines_worst_group_error.png",
        ),
        ("winkler", "Winkler score", "unified_baselines_winkler.png"),
        (
            "coverage_error",
            "Absolute coverage error",
            "unified_baselines_coverage_error.png",
        ),
    ]:
        pivot = results.pivot(index="market", columns="method", values=metric)

        ax = pivot.plot(kind="bar", figsize=(14, 6))

        ax.set_xlabel("Market")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " under a unified raw-prediction pipeline")

        ax.legend(title="Method", bbox_to_anchor=(1.02, 1.0), loc="upper left")

        plt.tight_layout()

        plt.savefig(figures_dir / filename, dpi=220)

        plt.close()


# Main.
def main() -> None:
    args = parse_args()

    # Read runtime arguments and prepare experiment output directories.
    tables_dir = (
        args.output / "tables"
    )

    # Generate and save figures for headline results and diagnostics.
    figures_dir = (
        args.output / "figures"
    )

    diagnostics_dir = (
        args.output / "diagnostics"
    )

    tables_dir.mkdir(parents=True, exist_ok=True)

    figures_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_strong_results = []

    for market_prefix in args.markets:
        market_results, _ = run_one_market(
            market_prefix=market_prefix,
            args=args,
            tables_dir=tables_dir,
            diagnostics_dir=diagnostics_dir,
        )

        all_strong_results.append(market_results)

    strong_results = pd.concat(all_strong_results, ignore_index=True)

    # Save result tables, prediction details, and reproducible diagnostics.
    strong_results.to_csv(
        tables_dir / "unified_strong_baseline_results.csv",
        index=False,
    )

    all_results = append_v10_methods(
        strong_results=strong_results,
        v10_results=args.v10_results,
    )

    all_results.to_csv(tables_dir / "unified_all_method_results.csv", index=False)

    # Aggregate results across markets, scenarios, or seeds for comparison.
    average_summary = make_average_summary(all_results)

    average_summary.to_csv(tables_dir / "unified_all_method_average.csv", index=False)

    pairwise = (
        make_pairwise_hybrid_comparison(all_results)
    )

    pairwise.to_csv(tables_dir / "hybrid_pairwise_comparison.csv", index=False)

    make_figures(results=all_results, figures_dir=figures_dir)

    print("\n" + "=" * 90)
    print("UNIFIED STRONG-BASELINE RESULTS")
    print("=" * 90)

    print(strong_results.to_string(index=False))

    print("\n" + "=" * 90)
    print("UNIFIED ALL-METHOD AVERAGE")
    print("=" * 90)

    display_columns = [
        "method",
        "coverage_mean",
        "coverage_error_mean",
        "avg_width_mean",
        "winkler_mean",
        "functional_error_mean",
        "worst_group_error_mean",
    ]

    print(average_summary[display_columns].to_string(index=False))

    print("\nSaved results to: " f"{args.output}")


if __name__ == "__main__":
    main()
