# Module purpose: Run Functional ACI methods and cross-market summaries on multiple markets.

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.calibration.functional_aci import (  # noqa: E402
    FunctionalACI,
    HybridFunctionalACI,
    LinearContextualACI,
    ScalarACI,
)
from src.evaluation.metrics import evaluate, rolling_functional_coverage_error  # noqa: E402
from src.protocol import (  # noqa: E402
    DEVELOPMENT_PROTOCOL,
    MARKET_NAME_MAP,
    PAPER_MARKETS,
    market_random_state,
)
from src.functional_pipeline import (  # noqa: E402
    build_context,
    chronological_split,
    convert_result,
    create_group_masks,
    fit_evaluation_map,
    load_market_data,
    market_dataset_path,
    make_group_table,
    make_raw_predictions,
    make_raw_result,
    preprocess_context,
    run_final_models,
    scaled_candidates,
    selection_objective,
    transform_evaluation_map,
)

# Parse args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Raw, Scalar, Linear Contextual, Functional, and Hybrid "
            "Functional ACI on multiple electricity markets."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/multi_market",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/versions/results_v10_multi_market_functional",
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=list(PAPER_MARKETS),
        help="Dataset prefixes, e.g. DE_LU DK_1 DK_2 SE_3.",
    )
    parser.add_argument("--alpha", type=float, default=DEVELOPMENT_PROTOCOL.alpha)
    parser.add_argument("--train-frac", type=float, default=DEVELOPMENT_PROTOCOL.train_fraction)
    parser.add_argument("--validation-frac", type=float, default=DEVELOPMENT_PROTOCOL.validation_fraction)
    parser.add_argument("--rolling-window", type=int, default=DEVELOPMENT_PROTOCOL.rolling_window)
    parser.add_argument("--random-state", type=int, default=DEVELOPMENT_PROTOCOL.random_state)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced hyperparameter grids for a pipeline check.",
    )
    return parser.parse_args()
# Tune scalar.
def tune_scalar(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_adjustment: float,
    evaluation_map: np.ndarray,
    group_masks: dict[str, np.ndarray],
    eta_values: list[float],
) -> tuple[dict, pd.DataFrame]:
    rows = []
    raw_width = float(np.mean(upper - lower))

    for eta in eta_values:
        model = ScalarACI(alpha=alpha, eta=eta, max_adjustment=max_adjustment)
        result = convert_result(model.run(lower, upper, y, reset=True))
        metrics = evaluate("Scalar ACI", y, result, evaluation_map, group_masks, alpha)
        rows.append(
            {
                "eta": eta,
                "objective": selection_objective(metrics, raw_width),
                **metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values("objective")
    return table.iloc[0].to_dict(), table


# Tune contextual.
def tune_contextual(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    alpha: float,
    max_adjustment: float,
    evaluation_map: np.ndarray,
    group_masks: dict[str, np.ndarray],
    eta_global_values: list[float],
    eta_linear_values: list[float],
    radius_values: list[float],
) -> tuple[dict, pd.DataFrame]:
    rows = []
    raw_width = float(np.mean(upper - lower))

    for eta_global, eta_linear, radius in product(
        eta_global_values,
        eta_linear_values,
        radius_values,
    ):
        model = LinearContextualACI(
            alpha=alpha,
            eta_global=eta_global,
            eta_linear=eta_linear,
            linear_radius=radius,
            max_adjustment=max_adjustment,
        )
        result = convert_result(model.run(lower, upper, y, context, reset=True))
        metrics = evaluate(
            "Linear contextual ACI",
            y,
            result,
            evaluation_map,
            group_masks,
            alpha,
        )
        rows.append(
            {
                "eta_global": eta_global,
                "eta_linear": eta_linear,
                "linear_radius": radius,
                "objective": selection_objective(metrics, raw_width),
                **metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values("objective")
    return table.iloc[0].to_dict(), table


# Tune functional.
def tune_functional(
    train_context: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    alpha: float,
    max_adjustment: float,
    evaluation_map: np.ndarray,
    group_masks: dict[str, np.ndarray],
    eta_global_values: list[float],
    eta_functional_values: list[float],
    component_values: list[int],
    radius_values: list[float],
    length_scale_values: list[float],
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    raw_width = float(np.mean(upper - lower))

    # Configure hyperparameter candidates for quick or full execution.
    for (
        eta_global,
        eta_functional,
        n_components,
        radius,
        length_scale,
    ) in product(
        eta_global_values,
        eta_functional_values,
        component_values,
        radius_values,
        length_scale_values,
    ):
        model = FunctionalACI(
            alpha=alpha,
            eta_global=eta_global,
            eta_functional=eta_functional,
            n_components=n_components,
            length_scale=length_scale,
            functional_radius=radius,
            max_adjustment=max_adjustment,
            random_state=random_state,
        )
        model.fit_feature_map(train_context)

        result = convert_result(model.run(lower, upper, y, context, reset=True))
        metrics = evaluate(
            "Functional ACI",
            y,
            result,
            evaluation_map,
            group_masks,
            alpha,
        )
        rows.append(
            {
                "eta_global": eta_global,
                "eta_functional": eta_functional,
                "n_components": n_components,
                "functional_radius": radius,
                "length_scale": length_scale,
                "objective": selection_objective(metrics, raw_width),
                **metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values("objective")
    return table.iloc[0].to_dict(), table


# Tune hybrid.
def tune_hybrid(
    train_context: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    alpha: float,
    max_adjustment: float,
    evaluation_map: np.ndarray,
    group_masks: dict[str, np.ndarray],
    contextual_best: dict,
    quick: bool,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    raw_width = float(np.mean(upper - lower))

    base_eta_global = float(contextual_best["eta_global"])
    base_eta_linear = float(contextual_best["eta_linear"])
    base_linear_radius = float(contextual_best["linear_radius"])

    # Configure hyperparameter candidates for quick or full execution.
    if quick:
        eta_functional_values = [0.005, 0.02, 0.05]
        component_values = [128]
        functional_radius_values = [2.0, 5.0]
        length_scale_values = [1.0, 2.0]
        ridge_values = [1e-3]
    else:
        eta_functional_values = [0.001, 0.005, 0.01, 0.02, 0.05]
        component_values = [128, 256]
        functional_radius_values = [2.0, 5.0, 10.0]
        length_scale_values = [0.5, 1.0, 2.0, 4.0]
        ridge_values = [1e-4, 1e-2]

    for (
        eta_functional,
        n_components,
        functional_radius,
        length_scale,
        residual_ridge,
    ) in product(
        eta_functional_values,
        component_values,
        functional_radius_values,
        length_scale_values,
        ridge_values,
    ):
        model = HybridFunctionalACI(
            alpha=alpha,
            eta_global=base_eta_global,
            eta_linear=base_eta_linear,
            eta_functional=eta_functional,
            n_components=n_components,
            length_scale=length_scale,
            residual_ridge=residual_ridge,
            linear_radius=base_linear_radius,
            functional_radius=functional_radius,
            max_adjustment=max_adjustment,
            random_state=random_state,
        )
        model.fit_feature_map(train_context)

        result = convert_result(model.run(lower, upper, y, context, reset=True))
        metrics = evaluate(
            "Hybrid Functional ACI",
            y,
            result,
            evaluation_map,
            group_masks,
            alpha,
        )
        rows.append(
            {
                "stage": 1,
                "eta_global": base_eta_global,
                "eta_linear": base_eta_linear,
                "eta_functional": eta_functional,
                "n_components": n_components,
                "linear_radius": base_linear_radius,
                "functional_radius": functional_radius,
                "length_scale": length_scale,
                "residual_ridge": residual_ridge,
                "objective": selection_objective(metrics, raw_width),
                **metrics,
            }
        )

    stage_one = pd.DataFrame(rows).sort_values("objective")
    # Compare candidates on validation data and select each method's hyperparameters.
    best_stage_one = stage_one.iloc[0].to_dict()

    stage_two_rows = []

    for eta_global, eta_linear, eta_functional in product(
        scaled_candidates(base_eta_global),
        scaled_candidates(base_eta_linear),
        scaled_candidates(float(best_stage_one["eta_functional"])),
    ):
        model = HybridFunctionalACI(
            alpha=alpha,
            eta_global=eta_global,
            eta_linear=eta_linear,
            eta_functional=eta_functional,
            n_components=int(best_stage_one["n_components"]),
            length_scale=float(best_stage_one["length_scale"]),
            residual_ridge=float(best_stage_one["residual_ridge"]),
            linear_radius=float(best_stage_one["linear_radius"]),
            functional_radius=float(best_stage_one["functional_radius"]),
            max_adjustment=max_adjustment,
            random_state=random_state,
        )
        model.fit_feature_map(train_context)

        result = convert_result(model.run(lower, upper, y, context, reset=True))
        metrics = evaluate(
            "Hybrid Functional ACI",
            y,
            result,
            evaluation_map,
            group_masks,
            alpha,
        )

        stage_two_rows.append(
            {
                "stage": 2,
                "eta_global": eta_global,
                "eta_linear": eta_linear,
                "eta_functional": eta_functional,
                "n_components": int(best_stage_one["n_components"]),
                "linear_radius": float(best_stage_one["linear_radius"]),
                "functional_radius": float(best_stage_one["functional_radius"]),
                "length_scale": float(best_stage_one["length_scale"]),
                "residual_ridge": float(best_stage_one["residual_ridge"]),
                "objective": selection_objective(metrics, raw_width),
                **metrics,
            }
        )

    full_table = pd.concat(
        [stage_one, pd.DataFrame(stage_two_rows)],
        ignore_index=True,
    ).sort_values("objective")

    return full_table.iloc[0].to_dict(), full_table


# Get parameter grids.
def get_parameter_grids(
    quick: bool,
) -> dict[str, list]:
    if quick:
        return {
            "scalar_eta": [0.02, 0.05, 0.10],
            "context_eta_global": [0.02, 0.05],
            "context_eta_linear": [0.005, 0.02, 0.05],
            "context_radius": [5.0, 10.0],
            "functional_eta_global": [0.02, 0.05],
            "functional_eta": [0.005, 0.02, 0.05],
            "functional_components": [128],
            "functional_radius": [2.0, 5.0],
            "functional_length_scale": [1.0, 2.0],
        }

    return {
        "scalar_eta": [0.005, 0.01, 0.02, 0.05, 0.10],
        "context_eta_global": [0.01, 0.02, 0.05, 0.10],
        "context_eta_linear": [0.001, 0.005, 0.01, 0.02, 0.05],
        "context_radius": [2.0, 5.0, 10.0],
        "functional_eta_global": [0.01, 0.02, 0.05, 0.10],
        "functional_eta": [0.001, 0.005, 0.01, 0.02, 0.05],
        "functional_components": [128, 256],
        "functional_radius": [2.0, 5.0],
        "functional_length_scale": [0.5, 1.0, 2.0, 4.0],
    }






# Run market.
def run_market(
    market_prefix: str,
    args: argparse.Namespace,
    tables_dir: Path,
    diagnostics_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = MARKET_NAME_MAP.get(market_prefix, market_prefix)
    data_path = market_dataset_path(args.data_dir, market_prefix)

    print("\n" + "=" * 90)
    print(f"Running market: {market}")
    print(f"Dataset: {data_path}")

    # Load inputs or existing results and normalize them for downstream processing.
    df, target_col, model_features = load_market_data(market_prefix, args.data_dir)

    # Split chronologically into explicit training, validation, and test periods.
    train_slice, validation_slice, test_slice = (
        chronological_split(len(df), args.train_frac, args.validation_frac)
    )

    print(
        f"Usable observations: {len(df):,}\n"
        f"Train: {train_slice.stop - train_slice.start:,}, "
        f"Validation: {validation_slice.stop - validation_slice.start:,}, "
        f"Test: {test_slice.stop - test_slice.start:,}"
    )

    split_table = pd.DataFrame(
        [
            {
                "market": market,
                "split": "train",
                "start": df.index[train_slice.start],
                "end": df.index[train_slice.stop - 1],
                "n": train_slice.stop - train_slice.start,
            },
            {
                "market": market,
                "split": "validation",
                "start": df.index[validation_slice.start],
                "end": df.index[validation_slice.stop - 1],
                "n": validation_slice.stop - validation_slice.start,
            },
            {
                "market": market,
                "split": "test",
                "start": df.index[test_slice.start],
                "end": df.index[test_slice.stop - 1],
                "n": test_slice.stop - test_slice.start,
            },
        ]
    )
    # Save result tables, prediction details, and reproducible diagnostics.
    split_table.to_csv(tables_dir / f"{market_prefix}_splits.csv", index=False)

    market_seed = market_random_state(market_prefix, args.random_state)

    # Train base quantile models and generate uncalibrated prediction intervals.
    predictions = make_raw_predictions(
        df=df,
        target_col=target_col,
        features=model_features,
        train_slice=train_slice,
        alpha=args.alpha,
        random_state=market_seed,
    )

    # Build and scale context features for conditional calibration.
    context = build_context(predictions)
    context_scaled = preprocess_context(context, train_slice)

    train_context = context_scaled[train_slice]
    validation_context = context_scaled[validation_slice]
    test_context = context_scaled[test_slice]

    lower_validation = (
        predictions["lower_raw"].iloc[validation_slice].to_numpy()
    )
    upper_validation = (
        predictions["upper_raw"].iloc[validation_slice].to_numpy()
    )
    y_validation = (
        predictions["y_true"].iloc[validation_slice].to_numpy()
    )

    lower_test = (
        predictions["lower_raw"].iloc[test_slice].to_numpy()
    )
    upper_test = (
        predictions["upper_raw"].iloc[test_slice].to_numpy()
    )
    y_test = (
        predictions["y_true"].iloc[test_slice].to_numpy()
    )

    validation_index = predictions.index[validation_slice]
    test_index = predictions.index[test_slice]

    train_width = (
        predictions["upper_raw"].iloc[train_slice]
        - predictions["lower_raw"].iloc[train_slice]
    )
    max_adjustment = max(20.0, float(2.0 * np.nanquantile(train_width, 0.95)))

    # Build the functional-error map and conditional groups for common evaluation.
    evaluation_feature_map = fit_evaluation_map(train_context, market_seed + 10_000)
    evaluation_map_validation = (
        transform_evaluation_map(evaluation_feature_map, validation_context)
    )
    evaluation_map_test = transform_evaluation_map(evaluation_feature_map, test_context)

    validation_group_masks = create_group_masks(
        index=validation_index,
        context_part=context.iloc[validation_slice],
        context_train=context.iloc[train_slice],
    )
    test_group_masks = create_group_masks(
        index=test_index,
        context_part=context.iloc[test_slice],
        context_train=context.iloc[train_slice],
    )

    # Configure hyperparameter candidates for quick or full execution.
    grids = get_parameter_grids(args.quick)

    print("Tuning Scalar ACI...")
    # Compare candidates on validation data and select each method's hyperparameters.
    best_scalar, scalar_table = tune_scalar(
        lower=lower_validation,
        upper=upper_validation,
        y=y_validation,
        alpha=args.alpha,
        max_adjustment=max_adjustment,
        evaluation_map=evaluation_map_validation,
        group_masks=validation_group_masks,
        eta_values=grids["scalar_eta"],
    )

    print("Tuning Linear contextual ACI...")
    best_contextual, contextual_table = (
        tune_contextual(
            lower=lower_validation,
            upper=upper_validation,
            y=y_validation,
            context=validation_context,
            alpha=args.alpha,
            max_adjustment=max_adjustment,
            evaluation_map=evaluation_map_validation,
            group_masks=validation_group_masks,
            eta_global_values=grids["context_eta_global"],
            eta_linear_values=grids["context_eta_linear"],
            radius_values=grids["context_radius"],
        )
    )

    print("Tuning pure Functional ACI...")
    best_functional, functional_table = (
        tune_functional(
            train_context=train_context,
            lower=lower_validation,
            upper=upper_validation,
            y=y_validation,
            context=validation_context,
            alpha=args.alpha,
            max_adjustment=max_adjustment,
            evaluation_map=evaluation_map_validation,
            group_masks=validation_group_masks,
            eta_global_values=grids["functional_eta_global"],
            eta_functional_values=grids["functional_eta"],
            component_values=grids["functional_components"],
            radius_values=grids["functional_radius"],
            length_scale_values=grids["functional_length_scale"],
            random_state=market_seed,
        )
    )

    print("Tuning Hybrid Functional ACI...")
    best_hybrid, hybrid_table = tune_hybrid(
        train_context=train_context,
        lower=lower_validation,
        upper=upper_validation,
        y=y_validation,
        context=validation_context,
        alpha=args.alpha,
        max_adjustment=max_adjustment,
        evaluation_map=evaluation_map_validation,
        group_masks=validation_group_masks,
        contextual_best=best_contextual,
        quick=args.quick,
        random_state=market_seed,
    )

    scalar_table.to_csv(
        diagnostics_dir / f"{market_prefix}_scalar_tuning.csv",
        index=False,
    )
    contextual_table.to_csv(
        diagnostics_dir / f"{market_prefix}_contextual_tuning.csv",
        index=False,
    )
    functional_table.to_csv(
        diagnostics_dir / f"{market_prefix}_functional_tuning.csv",
        index=False,
    )
    hybrid_table.to_csv(
        diagnostics_dir / f"{market_prefix}_hybrid_tuning.csv",
        index=False,
    )

    selected = {
        "alpha": float(args.alpha),
        "max_adjustment": float(max_adjustment),
        "scalar": {"eta": float(best_scalar["eta"])},
        "contextual": {
            "eta_global": float(best_contextual["eta_global"]),
            "eta_linear": float(best_contextual["eta_linear"]),
            "linear_radius": float(best_contextual["linear_radius"]),
        },
        "functional": {
            "eta_global": float(best_functional["eta_global"]),
            "eta_functional": float(best_functional["eta_functional"]),
            "n_components": int(best_functional["n_components"]),
            "functional_radius": float(best_functional["functional_radius"]),
            "length_scale": float(best_functional["length_scale"]),
        },
        "hybrid": {
            "eta_global": float(best_hybrid["eta_global"]),
            "eta_linear": float(best_hybrid["eta_linear"]),
            "eta_functional": float(best_hybrid["eta_functional"]),
            "n_components": int(best_hybrid["n_components"]),
            "linear_radius": float(best_hybrid["linear_radius"]),
            "functional_radius": float(best_hybrid["functional_radius"]),
            "length_scale": float(best_hybrid["length_scale"]),
            "residual_ridge": float(best_hybrid["residual_ridge"]),
        },
    }

    (
        diagnostics_dir / f"{market_prefix}_selected_hyperparameters.json"
    ).write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )

    # Run the final path with selected parameters to produce test predictions.
    final_results = run_final_models(
        selected=selected,
        alpha=args.alpha,
        max_adjustment=max_adjustment,
        random_state=market_seed,
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

    method_results = {
        "Raw quantile": make_raw_result(predictions.iloc[test_slice]),
        **final_results,
    }

    # Aggregate results across markets, scenarios, or seeds for comparison.
    summary_rows = []

    for method_name, result in method_results.items():
        metrics = evaluate(
            method_name=method_name,
            y=y_test,
            result=result,
            evaluation_map=evaluation_map_test,
            group_masks=test_group_masks,
            alpha=args.alpha,
        )
        metrics["market"] = market
        metrics["test_start"] = test_index.min()
        metrics["test_end"] = test_index.max()
        metrics["test_n"] = len(test_index)
        summary_rows.append(metrics)

    market_summary = pd.DataFrame(summary_rows)
    market_summary = market_summary[
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
            "mean_adjustment",
            "mean_abs_linear_component",
            "mean_abs_functional_component",
            "test_start",
            "test_end",
            "test_n",
        ]
    ].sort_values(
        ["functional_error", "worst_group_error"]
    )

    group_table = make_group_table(
        market=market,
        y=y_test,
        method_results=method_results,
        group_masks=test_group_masks,
        target_coverage=1.0 - args.alpha,
    )

    prediction_table = pd.DataFrame(index=test_index)
    prediction_table["market"] = market
    prediction_table["y_true"] = y_test

    for method_name, result in method_results.items():
        safe_name = (
            method_name.lower().replace(" ", "_").replace("-", "_")
        )

        for column in [
            "lower",
            "upper",
            "adjustment",
            "miscoverage",
            "global_component",
            "linear_component",
            "functional_component",
        ]:
            prediction_table[
                f"{column}_{safe_name}"
            ] = result[column]

    prediction_table.to_csv(
        tables_dir / f"{market_prefix}_test_predictions.csv",
        index_label="datetime",
    )

    rolling_rows = []

    for method_name, result in method_results.items():
        rolling_coverage = (
            1.0
            - pd.Series(result["miscoverage"], index=test_index)
            .rolling(
                args.rolling_window,
                min_periods=args.rolling_window,
            )
            .mean()
        )

        rolling_functional = (
            rolling_functional_coverage_error(
                result["miscoverage"],
                evaluation_map_test,
                alpha=args.alpha,
                window=args.rolling_window,
                min_periods=args.rolling_window,
            )
        )

        for timestamp, coverage_value, functional_value in zip(
            test_index,
            rolling_coverage.to_numpy(),
            rolling_functional,
        ):
            rolling_rows.append(
                {
                    "market": market,
                    "datetime": timestamp,
                    "method": method_name,
                    "rolling_coverage": coverage_value,
                    "rolling_functional_error": functional_value,
                }
            )

    rolling_table = pd.DataFrame(rolling_rows)

    print("\nMarket summary:")
    print(market_summary.to_string(index=False))

    return market_summary, group_table, rolling_table


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
        "mean_adjustment",
        "mean_abs_linear_component",
        "mean_abs_functional_component",
    ]

    grouped = (
        results.groupby("method")[metrics].agg(["mean", "std"])
    )
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    return (
        grouped.reset_index()
        .sort_values(
            ["functional_error_mean", "worst_group_error_mean"]
        )
    )


# Make rank summary.
def make_rank_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    ranking_metrics = [
        "coverage_error",
        "avg_width",
        "winkler",
        "functional_error",
        "worst_group_error",
    ]

    rank_frames = []

    for metric in ranking_metrics:
        temp = results[["market", "method", metric]].copy()

        temp["rank"] = temp.groupby("market")[metric].rank(method="min", ascending=True)

        temp["metric"] = metric
        temp["win"] = (temp["rank"] == 1.0).astype(int)

        rank_frames.append(temp)

    ranks = pd.concat(rank_frames, ignore_index=True)

    summary = (
        ranks.groupby(["method", "metric"])
        .agg(
            average_rank=("rank", "mean"),
            wins=("win", "sum"),
        )
        .reset_index()
    )

    overall = (
        ranks.groupby("method")
        .agg(
            overall_average_rank=("rank", "mean"),
            total_metric_wins=("win", "sum"),
        )
        .reset_index()
    )

    return summary.merge(overall, on="method", how="left").sort_values(
        ["overall_average_rank", "metric"]
    )


# Make plots.
def make_plots(
    results: pd.DataFrame,
    rolling: pd.DataFrame,
    figures_dir: Path,
    alpha: float,
) -> None:
    # Generate and save figures for headline results and diagnostics.
    figures_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in [
        ("coverage_error", "Absolute coverage error", "multi_market_coverage_error.png"),
        (
            "functional_error",
            "Functional coverage error",
            "multi_market_functional_error.png",
        ),
        (
            "worst_group_error",
            "Worst-group coverage error",
            "multi_market_worst_group_error.png",
        ),
        ("winkler", "Winkler score", "multi_market_winkler.png"),
    ]:
        pivot = results.pivot(index="market", columns="method", values=metric)

        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.set_xlabel("Market")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " across markets")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1.0), loc="upper left")

        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=220)
        plt.close()

    # Aggregate results across markets, scenarios, or seeds for comparison.
    average = (
        results.groupby("method")[["functional_error", "worst_group_error", "winkler"]]
        .mean()
        .sort_values(
            "functional_error"
        )
    )

    ax = average[["functional_error", "worst_group_error"]].plot(
        kind="bar",
        figsize=(11, 5),
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean error across markets")
    ax.set_title("Average state-dependent calibration error")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    # Save result tables, prediction details, and reproducible diagnostics.
    plt.savefig(figures_dir / "multi_market_average_calibration_error.png", dpi=220)
    plt.close()

    # Plot rolling coverage separately per market to avoid overlapping curves.
    for market in rolling["market"].unique():
        part = rolling[rolling["market"] == market]

        plt.figure(figsize=(12, 5))

        for method_name in part["method"].unique():
            method_part = part[part["method"] == method_name]
            plt.plot(
                pd.to_datetime(method_part["datetime"]),
                method_part["rolling_coverage"],
                label=method_name,
                linewidth=1.0,
            )

        plt.axhline(1.0 - alpha, linestyle="--", linewidth=1.0, label="Target")
        plt.xlabel("Time")
        plt.ylabel("Rolling coverage")
        plt.title(f"{market}: rolling coverage")
        plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()

        safe_market = (
            market.replace("-", "_").replace(" ", "_")
        )
        plt.savefig(figures_dir / f"{safe_market}_rolling_coverage.png", dpi=220)
        plt.close()


# Main.
def main() -> None:
    args = parse_args()

    # Read runtime arguments and prepare experiment output directories.
    tables_dir = args.output / "tables"
    # Generate and save figures for headline results and diagnostics.
    figures_dir = args.output / "figures"
    diagnostics_dir = args.output / "diagnostics"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_groups = []
    all_rolling = []

    for market_prefix in args.markets:
        market_results, group_table, rolling_table = (
            run_market(
                market_prefix=market_prefix,
                args=args,
                tables_dir=tables_dir,
                diagnostics_dir=diagnostics_dir,
            )
        )

        all_results.append(market_results)
        all_groups.append(group_table)
        all_rolling.append(rolling_table)

    results = pd.concat(all_results, ignore_index=True)
    groups = pd.concat(all_groups, ignore_index=True)
    rolling = pd.concat(all_rolling, ignore_index=True)

    # Save result tables, prediction details, and reproducible diagnostics.
    results.to_csv(tables_dir / "multi_market_results.csv", index=False)
    groups.to_csv(tables_dir / "multi_market_group_coverage.csv", index=False)
    rolling.to_csv(tables_dir / "multi_market_rolling_metrics.csv", index=False)

    # Aggregate results across markets, scenarios, or seeds for comparison.
    average_summary = make_average_summary(results)
    average_summary.to_csv(tables_dir / "multi_market_average_summary.csv", index=False)

    rank_summary = make_rank_summary(results)
    rank_summary.to_csv(tables_dir / "multi_market_rank_summary.csv", index=False)

    make_plots(
        results=results,
        rolling=rolling,
        figures_dir=figures_dir,
        alpha=args.alpha,
    )

    print("\n" + "=" * 90)
    print("MULTI-MARKET RESULTS")
    print("=" * 90)
    print(results.to_string(index=False))

    print("\nAVERAGE SUMMARY")
    print(average_summary.to_string(index=False))

    print("\nRANK SUMMARY")
    print(rank_summary.to_string(index=False))

    print("\nSaved results to: " f"{args.output}")


if __name__ == "__main__":
    main()
