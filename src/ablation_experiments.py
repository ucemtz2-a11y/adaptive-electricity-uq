# Module purpose: Run tuning, path replay, and per-market evaluation for context and kernel ablations.

"""Experiment core for the paper's Functional ACI ablations."""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.calibration.functional_aci import HybridFunctionalACI, LinearContextualACI
from src.functional_pipeline import (
    CONTEXT_COLUMNS,
    build_context,
    chronological_split,
    convert_result,
    create_group_masks,
    fit_evaluation_map,
    load_market_data,
    make_raw_predictions,
    preprocess_context,
    scaled_candidates,
    selection_objective,
    transform_evaluation_map,
)
from src.evaluation.metrics import evaluate
from src.protocol import MARKET_NAME_MAP, market_random_state


CONTEXT_ABLATIONS = {
    "full": list(CONTEXT_COLUMNS),
    "no_rolling_volatility": [
        column
        for column in CONTEXT_COLUMNS
        if column not in {"rolling_price_std_24", "rolling_price_std_168"}
    ],
    "no_past_miscoverage": [
        column
        for column in CONTEXT_COLUMNS
        if column != "rolling_raw_miscoverage_168"
    ],
    "no_raw_width": [column for column in CONTEXT_COLUMNS if column != "raw_width"],
    "no_calendar": [
        column
        for column in CONTEXT_COLUMNS
        if column
        not in {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend"}
    ],
}


# Load selected parameters.
def load_selected_parameters(
    v10_results: Path,
    market_prefix: str,
) -> dict:
    path = (
        v10_results / "diagnostics" / f"{market_prefix}_selected_hyperparameters.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Frozen v10 hyperparameters were not found:\n"
            f"{path}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


# Preprocess selected context.
def preprocess_selected_context(
    context: pd.DataFrame,
    columns: list[str],
    train_slice: slice,
) -> np.ndarray:
    if not columns:
        raise ValueError("At least one context column is required.")

    missing = [column for column in columns if column not in context.columns]

    if missing:
        raise KeyError(f"Missing context columns: {missing}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(context.iloc[train_slice][columns])
    scaler.fit(train_imputed)

    all_imputed = imputer.transform(context[columns])

    return scaler.transform(all_imputed)


# Run linear path.
def run_linear_path(
    parameters: dict,
    alpha: float,
    max_adjustment: float,
    validation_context: np.ndarray,
    test_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, np.ndarray]:
    model = LinearContextualACI(
        alpha=alpha,
        eta_global=float(parameters["eta_global"]),
        eta_linear=float(parameters["eta_linear"]),
        linear_radius=float(parameters["linear_radius"]),
        max_adjustment=max_adjustment,
    )

    model.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )

    return convert_result(
        model.run(lower_test, upper_test, y_test, test_context, reset=False)
    )


# Run hybrid path.
def run_hybrid_path(
    parameters: dict,
    alpha: float,
    max_adjustment: float,
    random_state: int,
    train_context: np.ndarray,
    validation_context: np.ndarray,
    test_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, np.ndarray]:
    model = HybridFunctionalACI(
        alpha=alpha,
        eta_global=float(parameters["eta_global"]),
        eta_linear=float(parameters["eta_linear"]),
        eta_functional=float(parameters["eta_functional"]),
        n_components=int(parameters["n_components"]),
        length_scale=float(parameters["length_scale"]),
        residual_ridge=float(parameters["residual_ridge"]),
        linear_radius=float(parameters["linear_radius"]),
        functional_radius=float(parameters["functional_radius"]),
        max_adjustment=max_adjustment,
        random_state=random_state,
    )

    model.fit_feature_map(train_context)

    model.run(
        lower_validation,
        upper_validation,
        y_validation,
        validation_context,
        reset=True,
    )

    return convert_result(
        model.run(lower_test, upper_test, y_test, test_context, reset=False)
    )


# Tune linear local.
def tune_linear_local(
    base_parameters: dict,
    quick: bool,
    alpha: float,
    max_adjustment: float,
    validation_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    evaluation_map_validation: np.ndarray,
    validation_group_masks: dict[str, np.ndarray],
) -> tuple[dict, pd.DataFrame]:
    rows = []

    raw_width = float(np.mean(upper_validation - lower_validation))

    # Configure hyperparameter candidates for quick or full execution.
    for eta_global, eta_linear in product(
        scaled_candidates(base_parameters["eta_global"], quick),
        scaled_candidates(base_parameters["eta_linear"], quick),
    ):
        model = LinearContextualACI(
            alpha=alpha,
            eta_global=eta_global,
            eta_linear=eta_linear,
            linear_radius=float(base_parameters["linear_radius"]),
            max_adjustment=max_adjustment,
        )

        result = convert_result(
            model.run(
                lower_validation,
                upper_validation,
                y_validation,
                validation_context,
                reset=True,
            )
        )

        metrics = evaluate(
            method_name=(
                "Linear contextual ACI"
            ),
            y=y_validation,
            result=result,
            evaluation_map=(
                evaluation_map_validation
            ),
            group_masks=(
                validation_group_masks
            ),
            alpha=alpha,
        )

        rows.append(
            {
                "eta_global": eta_global,
                "eta_linear": eta_linear,
                "linear_radius": float(base_parameters["linear_radius"]),
                "objective": (
                    selection_objective(metrics, raw_width)
                ),
                **metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values("objective")

    return table.iloc[0].to_dict(), table


# Tune hybrid local.
def tune_hybrid_local(
    base_parameters: dict,
    quick: bool,
    alpha: float,
    max_adjustment: float,
    random_state: int,
    train_context: np.ndarray,
    validation_context: np.ndarray,
    lower_validation: np.ndarray,
    upper_validation: np.ndarray,
    y_validation: np.ndarray,
    evaluation_map_validation: np.ndarray,
    validation_group_masks: dict[str, np.ndarray],
) -> tuple[dict, pd.DataFrame]:
    rows = []

    raw_width = float(np.mean(upper_validation - lower_validation))

    # Configure hyperparameter candidates for quick or full execution.
    grid = product(
        scaled_candidates(base_parameters["eta_global"], quick),
        scaled_candidates(base_parameters["eta_linear"], quick),
        scaled_candidates(base_parameters["eta_functional"], quick),
    )

    for (
        eta_global,
        eta_linear,
        eta_functional,
    ) in grid:
        parameters = {
            **base_parameters,
            "eta_global": eta_global,
            "eta_linear": eta_linear,
            "eta_functional": (
                eta_functional
            ),
        }

        model = HybridFunctionalACI(
            alpha=alpha,
            eta_global=eta_global,
            eta_linear=eta_linear,
            eta_functional=(
                eta_functional
            ),
            n_components=int(parameters["n_components"]),
            length_scale=float(parameters["length_scale"]),
            residual_ridge=float(parameters["residual_ridge"]),
            linear_radius=float(parameters["linear_radius"]),
            functional_radius=float(parameters["functional_radius"]),
            max_adjustment=max_adjustment,
            random_state=random_state,
        )

        model.fit_feature_map(train_context)

        result = convert_result(
            model.run(
                lower_validation,
                upper_validation,
                y_validation,
                validation_context,
                reset=True,
            )
        )

        metrics = evaluate(
            method_name=(
                "Hybrid Functional ACI"
            ),
            y=y_validation,
            result=result,
            evaluation_map=(
                evaluation_map_validation
            ),
            group_masks=(
                validation_group_masks
            ),
            alpha=alpha,
        )

        rows.append(
            {
                "eta_global": eta_global,
                "eta_linear": eta_linear,
                "eta_functional": (
                    eta_functional
                ),
                "n_components": int(parameters["n_components"]),
                "linear_radius": float(parameters["linear_radius"]),
                "functional_radius": float(parameters["functional_radius"]),
                "length_scale": float(parameters["length_scale"]),
                "residual_ridge": float(parameters["residual_ridge"]),
                "objective": (
                    selection_objective(metrics, raw_width)
                ),
                **metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values("objective")

    return table.iloc[0].to_dict(), table


# Run context ablation market.
def run_context_ablation_market(
    market_prefix: str,
    args: argparse.Namespace,
    tables_dir: Path,
    diagnostics_dir: Path,
) -> tuple[pd.DataFrame, dict, dict]:
    market = MARKET_NAME_MAP.get(market_prefix, market_prefix)

    print("\n" + "=" * 90)
    print(f"Context ablation: {market}")

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

    full_context_scaled = preprocess_context(context, train_slice)

    full_train_context = (
        full_context_scaled[train_slice]
    )

    full_validation_context = (
        full_context_scaled[validation_slice]
    )

    full_test_context = (
        full_context_scaled[test_slice]
    )

    validation_index = (
        predictions.index[validation_slice]
    )

    test_index = (
        predictions.index[test_slice]
    )

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

    train_width = (
        predictions["upper_raw"].iloc[train_slice]
        - predictions["lower_raw"].iloc[train_slice]
    )

    max_adjustment = max(20.0, float(2.0 * np.nanquantile(train_width, 0.95)))

    # Build the functional-error map and conditional groups for common evaluation.
    witness_map = fit_evaluation_map(full_train_context, market_seed + 10_000)

    evaluation_map_validation = (
        transform_evaluation_map(witness_map, full_validation_context)
    )

    evaluation_map_test = (
        transform_evaluation_map(witness_map, full_test_context)
    )

    validation_group_masks = (
        create_group_masks(
            index=validation_index,
            context_part=(
                context.iloc[validation_slice]
            ),
            context_train=(
                context.iloc[train_slice]
            ),
        )
    )

    test_group_masks = create_group_masks(
        index=test_index,
        context_part=(
            context.iloc[test_slice]
        ),
        context_train=(
            context.iloc[train_slice]
        ),
    )

    selected_v10 = load_selected_parameters(args.v10_results, market_prefix)

    rows = []
    # Read runtime arguments and prepare experiment output directories.
    selected_output = {}
    tuning_output = {}

    for (
        ablation_name,
        context_columns,
    ) in CONTEXT_ABLATIONS.items():
        print(
            f"  {ablation_name}: "
            f"{len(context_columns)} context features"
        )

        reduced_context_scaled = (
            preprocess_selected_context(
                context=context,
                columns=context_columns,
                train_slice=train_slice,
            )
        )

        train_context = (
            reduced_context_scaled[train_slice]
        )

        validation_context = (
            reduced_context_scaled[validation_slice]
        )

        test_context = (
            reduced_context_scaled[test_slice]
        )

        (
            best_linear,
            linear_tuning,
        ) = tune_linear_local(
            base_parameters=(
                selected_v10["contextual"]
            ),
            quick=args.quick,
            alpha=args.alpha,
            max_adjustment=max_adjustment,
            validation_context=(
                validation_context
            ),
            lower_validation=(
                lower_validation
            ),
            upper_validation=(
                upper_validation
            ),
            y_validation=y_validation,
            evaluation_map_validation=(
                evaluation_map_validation
            ),
            validation_group_masks=(
                validation_group_masks
            ),
        )

        (
            best_hybrid,
            hybrid_tuning,
        ) = tune_hybrid_local(
            base_parameters=(
                selected_v10["hybrid"]
            ),
            quick=args.quick,
            alpha=args.alpha,
            max_adjustment=max_adjustment,
            random_state=market_seed,
            train_context=train_context,
            validation_context=(
                validation_context
            ),
            lower_validation=(
                lower_validation
            ),
            upper_validation=(
                upper_validation
            ),
            y_validation=y_validation,
            evaluation_map_validation=(
                evaluation_map_validation
            ),
            validation_group_masks=(
                validation_group_masks
            ),
        )

        selected_linear = {
            "eta_global": float(best_linear["eta_global"]),
            "eta_linear": float(best_linear["eta_linear"]),
            "linear_radius": float(best_linear["linear_radius"]),
        }

        selected_hybrid = {
            "eta_global": float(best_hybrid["eta_global"]),
            "eta_linear": float(best_hybrid["eta_linear"]),
            "eta_functional": float(best_hybrid["eta_functional"]),
            "n_components": int(best_hybrid["n_components"]),
            "linear_radius": float(best_hybrid["linear_radius"]),
            "functional_radius": float(best_hybrid["functional_radius"]),
            "length_scale": float(best_hybrid["length_scale"]),
            "residual_ridge": float(best_hybrid["residual_ridge"]),
        }

        linear_result = run_linear_path(
            parameters=selected_linear,
            alpha=args.alpha,
            max_adjustment=max_adjustment,
            validation_context=(
                validation_context
            ),
            test_context=test_context,
            lower_validation=(
                lower_validation
            ),
            upper_validation=(
                upper_validation
            ),
            y_validation=y_validation,
            lower_test=lower_test,
            upper_test=upper_test,
            y_test=y_test,
        )

        hybrid_result = run_hybrid_path(
            parameters=selected_hybrid,
            alpha=args.alpha,
            max_adjustment=max_adjustment,
            random_state=market_seed,
            train_context=train_context,
            validation_context=(
                validation_context
            ),
            test_context=test_context,
            lower_validation=(
                lower_validation
            ),
            upper_validation=(
                upper_validation
            ),
            y_validation=y_validation,
            lower_test=lower_test,
            upper_test=upper_test,
            y_test=y_test,
        )

        for method_name, result in {
            "Linear contextual ACI": (
                linear_result
            ),
            "Hybrid Functional ACI": (
                hybrid_result
            ),
        }.items():
            metrics = evaluate(
                method_name=method_name,
                y=y_test,
                result=result,
                evaluation_map=(
                    evaluation_map_test
                ),
                group_masks=(
                    test_group_masks
                ),
                alpha=args.alpha,
            )

            rows.append(
                {
                    "market": market,
                    "market_prefix": (
                        market_prefix
                    ),
                    "ablation": (
                        ablation_name
                    ),
                    "method": method_name,
                    "n_context_features": (
                        len(context_columns)
                    ),
                    "context_columns": "|".join(context_columns),
                    **metrics,
                }
            )

        selected_output[
            ablation_name
        ] = {
            "context_columns": (
                context_columns
            ),
            "linear": selected_linear,
            "hybrid": selected_hybrid,
        }

        tuning_output[
            ablation_name
        ] = {"linear": (linear_tuning), "hybrid": (hybrid_tuning)}

        linear_tuning.to_csv(
            diagnostics_dir
            / (
                f"{market_prefix}_"
                f"{ablation_name}_"
                "linear_tuning.csv"
            ),
            index=False,
        )

        hybrid_tuning.to_csv(
            diagnostics_dir
            / (
                f"{market_prefix}_"
                f"{ablation_name}_"
                "hybrid_tuning.csv"
            ),
            index=False,
        )

    # Save result tables, prediction details, and reproducible diagnostics.
    (
        diagnostics_dir
        / (
            f"{market_prefix}_"
            "context_ablation_"
            "selected_parameters.json"
        )
    ).write_text(
        json.dumps(selected_output, indent=2),
        encoding="utf-8",
    )

    return (
        pd.DataFrame(rows),
        selected_output,
        {
            "market": market,
            "full_context_scaled": (
                full_context_scaled
            ),
            "context": context,
            "predictions": predictions,
            "train_slice": train_slice,
            "validation_slice": (
                validation_slice
            ),
            "test_slice": test_slice,
            "full_train_context": (
                full_train_context
            ),
            "full_validation_context": (
                full_validation_context
            ),
            "full_test_context": (
                full_test_context
            ),
            "evaluation_map_test": (
                evaluation_map_test
            ),
            "test_group_masks": (
                test_group_masks
            ),
            "lower_validation": (
                lower_validation
            ),
            "upper_validation": (
                upper_validation
            ),
            "y_validation": y_validation,
            "lower_test": lower_test,
            "upper_test": upper_test,
            "y_test": y_test,
            "max_adjustment": (
                max_adjustment
            ),
            "market_seed": market_seed,
            "selected_v10": selected_v10,
        },
    )


# Run kernel ablation market.
def run_kernel_ablation_market(
    market_prefix: str,
    prepared: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    market = prepared["market"]

    print(f"Kernel approximation ablation: " f"{market}")

    # Configure hyperparameter candidates for quick or full execution.
    if args.quick:
        component_values = [64, 128]
        length_scale_values = [1.0, 2.0]
    else:
        component_values = [64, 128, 256]
        length_scale_values = [1.0, 2.0, 4.0]

    base = prepared["selected_v10"]["hybrid"]

    rows = []

    for (
        n_components,
        length_scale,
    ) in product(component_values, length_scale_values):
        parameters = {
            **base,
            "n_components": (
                n_components
            ),
            "length_scale": (
                length_scale
            ),
        }

        start_time = time.perf_counter()

        result = run_hybrid_path(
            parameters=parameters,
            alpha=args.alpha,
            max_adjustment=(
                prepared["max_adjustment"]
            ),
            random_state=(
                prepared["market_seed"]
            ),
            train_context=(
                prepared["full_train_context"]
            ),
            validation_context=(
                prepared["full_validation_context"]
            ),
            test_context=(
                prepared["full_test_context"]
            ),
            lower_validation=(
                prepared["lower_validation"]
            ),
            upper_validation=(
                prepared["upper_validation"]
            ),
            y_validation=(
                prepared["y_validation"]
            ),
            lower_test=(
                prepared["lower_test"]
            ),
            upper_test=(
                prepared["upper_test"]
            ),
            y_test=prepared["y_test"],
        )

        runtime_seconds = (
            time.perf_counter() - start_time
        )

        metrics = evaluate(
            method_name=(
                "Hybrid Functional ACI"
            ),
            y=prepared["y_test"],
            result=result,
            evaluation_map=(
                prepared["evaluation_map_test"]
            ),
            group_masks=(
                prepared["test_group_masks"]
            ),
            alpha=args.alpha,
        )

        rows.append(
            {
                "market": market,
                "market_prefix": (
                    market_prefix
                ),
                "n_components": (
                    n_components
                ),
                "length_scale": (
                    length_scale
                ),
                "runtime_seconds": (
                    runtime_seconds
                ),
                "eta_global": float(base["eta_global"]),
                "eta_linear": float(base["eta_linear"]),
                "eta_functional": float(base["eta_functional"]),
                "linear_radius": float(base["linear_radius"]),
                "functional_radius": float(base["functional_radius"]),
                "residual_ridge": float(base["residual_ridge"]),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
