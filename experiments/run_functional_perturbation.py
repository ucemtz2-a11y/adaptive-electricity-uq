# Module purpose: Evaluate Functional ACI under controlled test-feature perturbations.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    preprocess_context,
    transform_evaluation_map,
)
from src.perturbation_reporting import make_figures  # noqa: E402
from src.protocol import (  # noqa: E402
    DEVELOPMENT_PROTOCOL,
    MARKET_NAME_MAP,
    PAPER_MARKETS,
    market_random_state,
)
from src.perturbations import (  # noqa: E402
    PERTURBED_FEATURES,
    PRIMITIVE_FEATURES,
    aggregate_degradation,
    compute_degradation,
    evaluate_perturbed_path,
    fit_base_models,
    load_selected_parameters,
    make_clean_predictions,
    make_cross_market_summary,
    perturb_test_features,
    replace_test_predictions,
    summarise_seed_results,
)

# Parse args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v11 stochastic-feature perturbation experiment "
            "using the frozen v10 multi-market methods and validation-selected "
            "hyperparameters."
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
        help=(
            "Directory containing the frozen v10 selected hyperparameters."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/versions/results_v11_functional_perturbation"
        ),
    )

    parser.add_argument("--markets", nargs="*", default=list(PAPER_MARKETS))

    parser.add_argument(
        "--rhos",
        nargs="*",
        type=float,
        default=[0.0, 0.05, 0.10, 0.20, 0.30],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of perturbation seeds.",
    )

    parser.add_argument(
        "--perturbation-mode",
        choices=["coherent", "independent"],
        default="coherent",
        help=(
            "coherent: perturb load/wind/solar and recompute residual load; "
            "independent: perturb all four features separately."
        ),
    )

    parser.add_argument("--noise", choices=["gaussian", "uniform"], default="gaussian")

    parser.add_argument(
        "--clip-z",
        type=float,
        default=3.0,
        help=(
            "For Gaussian noise, clip standard-normal draws to +/- clip-z. "
            "Set a non-positive value to disable clipping."
        ),
    )

    parser.add_argument("--alpha", type=float, default=DEVELOPMENT_PROTOCOL.alpha)
    parser.add_argument("--train-frac", type=float, default=DEVELOPMENT_PROTOCOL.train_fraction)
    parser.add_argument("--validation-frac", type=float, default=DEVELOPMENT_PROTOCOL.validation_fraction)
    parser.add_argument("--random-state", type=int, default=DEVELOPMENT_PROTOCOL.random_state)

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run 3 seeds and rho in {0, 0.10, 0.30} for a pipeline check."
        ),
    )

    # Read runtime arguments and prepare experiment output directories.
    return parser.parse_args()




























# Main.
def main() -> None:
    args = parse_args()

    # Configure hyperparameter candidates for quick or full execution.
    if args.quick:
        rhos = [0.0, 0.10, 0.30]
        number_of_seeds = 3
    else:
        rhos = sorted(set(float(rho) for rho in args.rhos))
        number_of_seeds = int(args.seeds)

    # Read runtime arguments and prepare experiment output directories.
    tables_dir = args.output / "tables"
    # Generate and save figures for headline results and diagnostics.
    figures_dir = args.output / "figures"
    diagnostics_dir = args.output / "diagnostics"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_seed_rows = []
    all_budget_rows = []

    configuration = {
        "markets": args.markets,
        "rhos": rhos,
        "number_of_seeds": number_of_seeds,
        "perturbation_mode": args.perturbation_mode,
        "noise": args.noise,
        "clip_z": args.clip_z,
        "alpha": args.alpha,
        "train_frac": args.train_frac,
        "validation_frac": args.validation_frac,
        "perturbed_features": PERTURBED_FEATURES,
        "primitive_features": PRIMITIVE_FEATURES,
    }

    # Save result tables, prediction details, and reproducible diagnostics.
    (
        diagnostics_dir / "perturbation_configuration.json"
    ).write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    for market_prefix in args.markets:
        market = MARKET_NAME_MAP.get(market_prefix, market_prefix)

        print("\n" + "=" * 90)
        print(f"Preparing market: {market}")

        df, target_col, model_features = (
            load_market_data(market_prefix, args.data_dir)
        )

        (
            train_slice,
            validation_slice,
            test_slice,
        ) = chronological_split(len(df), args.train_frac, args.validation_frac)

        market_seed_offset = market_random_state(market_prefix, 0)
        market_seed = market_random_state(market_prefix, args.random_state)

        selected = load_selected_parameters(args.v10_results, market_prefix)

        (
            lower_model,
            median_model,
            upper_model,
        ) = fit_base_models(
            df=df,
            target_col=target_col,
            model_features=model_features,
            train_slice=train_slice,
            alpha=args.alpha,
            random_state=market_seed,
        )

        clean_predictions = (
            make_clean_predictions(
                df=df,
                target_col=target_col,
                model_features=model_features,
                lower_model=lower_model,
                median_model=median_model,
                upper_model=upper_model,
            )
        )

        clean_context = build_context(clean_predictions)

        clean_context_scaled = (
            preprocess_context(clean_context, train_slice)
        )

        train_context = clean_context_scaled[train_slice]

        validation_context = (
            clean_context_scaled[validation_slice]
        )

        lower_validation = (
            clean_predictions["lower_raw"].iloc[validation_slice].to_numpy()
        )

        upper_validation = (
            clean_predictions["upper_raw"].iloc[validation_slice].to_numpy()
        )

        y_validation = (
            clean_predictions["y_true"].iloc[validation_slice].to_numpy()
        )

        y_test = (
            clean_predictions["y_true"].iloc[test_slice].to_numpy()
        )

        test_index = clean_predictions.index[test_slice]

        evaluation_feature_map = (
            fit_evaluation_map(train_context, market_seed + 10_000)
        )

        x_test_clean = df[model_features].iloc[test_slice].copy()

        train_std = (
            df[model_features].iloc[train_slice].std(ddof=0)
        )

        for seed_index in range(number_of_seeds):
            perturbation_seed = (
                args.random_state + 100_000 + 10_000 * market_seed_offset + seed_index
            )

            for rho in rhos:
                rng = np.random.default_rng(
                    perturbation_seed + int(round(rho * 1_000_000))
                )

                (
                    x_test_perturbed,
                    budget_diagnostics,
                ) = perturb_test_features(
                    x_test=x_test_clean,
                    train_std=train_std,
                    rho=rho,
                    rng=rng,
                    mode=args.perturbation_mode,
                    noise=args.noise,
                    clip_z=args.clip_z,
                )

                perturbed_predictions = (
                    replace_test_predictions(
                        clean_predictions=(
                            clean_predictions
                        ),
                        test_slice=test_slice,
                        x_test_perturbed=(
                            x_test_perturbed
                        ),
                        lower_model=lower_model,
                        median_model=median_model,
                        upper_model=upper_model,
                    )
                )

                perturbed_context = build_context(perturbed_predictions)

                perturbed_context_scaled = (
                    preprocess_context(perturbed_context, train_slice)
                )

                test_context = (
                    perturbed_context_scaled[test_slice]
                )

                lower_test = (
                    perturbed_predictions["lower_raw"].iloc[test_slice].to_numpy()
                )

                upper_test = (
                    perturbed_predictions["upper_raw"].iloc[test_slice].to_numpy()
                )

                evaluation_map_test = (
                    transform_evaluation_map(evaluation_feature_map, test_context)
                )

                test_group_masks = (
                    create_group_masks(
                        index=test_index,
                        context_part=(
                            perturbed_context.iloc[test_slice]
                        ),
                        context_train=(
                            clean_context.iloc[train_slice]
                        ),
                    )
                )

                metric_rows = (
                    evaluate_perturbed_path(
                        market=market,
                        selected=selected,
                        alpha=args.alpha,
                        market_seed=market_seed,
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
                        raw_test_predictions=(
                            perturbed_predictions.iloc[test_slice]
                        ),
                        evaluation_map_test=(
                            evaluation_map_test
                        ),
                        test_group_masks=(
                            test_group_masks
                        ),
                    )
                )

                for row in metric_rows:
                    row["market_prefix"] = (
                        market_prefix
                    )
                    row["seed"] = seed_index
                    row[
                        "perturbation_seed"
                    ] = perturbation_seed
                    row["rho"] = float(rho)
                    row[
                        "perturbation_mode"
                    ] = args.perturbation_mode
                    row["noise"] = args.noise
                    all_seed_rows.append(row)

                all_budget_rows.append(
                    {
                        "market": market,
                        "market_prefix": (
                            market_prefix
                        ),
                        "seed": seed_index,
                        "perturbation_seed": (
                            perturbation_seed
                        ),
                        "rho": float(rho),
                        "perturbation_mode": (
                            args.perturbation_mode
                        ),
                        "noise": args.noise,
                        **budget_diagnostics,
                    }
                )

                print(
                    f"{market}: seed "
                    f"{seed_index + 1}/"
                    f"{number_of_seeds}, "
                    f"rho={rho:.2f}"
                )

    seed_results = pd.DataFrame(all_seed_rows)

    budget_results = pd.DataFrame(all_budget_rows)

    # Aggregate results across markets, scenarios, or seeds for comparison.
    summary = summarise_seed_results(seed_results)

    # Compute coverage, interval-efficiency, and conditional-coverage metrics.
    degradation = compute_degradation(seed_results)

    degradation_summary = (
        aggregate_degradation(degradation)
    )

    cross_market_summary = (
        make_cross_market_summary(seed_results)
    )

    seed_results.to_csv(tables_dir / "perturbation_seed_results.csv", index=False)

    summary.to_csv(tables_dir / "perturbation_summary.csv", index=False)

    degradation.to_csv(tables_dir / "perturbation_degradation_by_seed.csv", index=False)

    degradation_summary.to_csv(
        tables_dir / "perturbation_degradation_summary.csv",
        index=False,
    )

    cross_market_summary.to_csv(
        tables_dir / "perturbation_cross_market_summary.csv",
        index=False,
    )

    budget_results.to_csv(tables_dir / "perturbation_budget.csv", index=False)

    make_figures(cross_market_summary=(cross_market_summary), figures_dir=figures_dir)

    print("\n" + "=" * 90)
    print("V11 CROSS-MARKET PERTURBATION SUMMARY")
    print("=" * 90)

    display_columns = [
        "rho",
        "method",
        "coverage_mean",
        "coverage_error_mean",
        "avg_width_mean",
        "winkler_mean",
        "functional_error_mean",
        "worst_group_error_mean",
    ]

    print(cross_market_summary[display_columns].to_string(index=False))

    print("\nSaved results to: " f"{args.output}")


if __name__ == "__main__":
    main()
