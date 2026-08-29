# Build the simulated oracle paths and regret values used to check the theory.

"""Mathematical core of the theory-alignment experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Create nonlinear features after removing the part already explained by linear context.
@dataclass
class ResidualRFFMap:
    omega: np.ndarray
    phase: np.ndarray
    rff_mean: np.ndarray
    residual_coef: np.ndarray

    # Apply the original random Fourier map before the linear projection is removed.
    def raw_rff(self, z: np.ndarray) -> np.ndarray:
        d_features = self.omega.shape[1]
        return (
            np.sqrt(2.0 / d_features) * np.cos(z @ self.omega + self.phase)
        )

    # Transform new rows with the map and projection learned from training context.
    def transform(self, z: np.ndarray) -> np.ndarray:
        rff_centered = (
            self.raw_rff(z) - self.rff_mean
        )

        linear_design = np.column_stack([np.ones(len(z)), z])

        residual = (
            rff_centered - linear_design @ self.residual_coef
        )

        return np.column_stack([np.ones(len(z)), z, residual])


# Keep a parameter vector inside a fixed Euclidean-radius constraint.
def project_l2_ball(
    u: np.ndarray,
    radius: float,
) -> np.ndarray:
    norm = float(np.linalg.norm(u))

    if norm <= radius:
        return u.copy()

    return (
        u * (radius / norm)
    )


# Calculate quantile loss for a vector of predictions and targets.
def pinball_loss(
    residual: float | np.ndarray,
    tau: float,
) -> float | np.ndarray:
    residual = np.asarray(residual)

    return np.where(residual >= 0.0, tau * residual, (tau - 1.0) * residual)


# Evaluate the simple uniform CDF used by the known simulation noise.
def uniform_cdf(
    x: np.ndarray,
    low: float = -0.9,
    high: float = 0.1,
) -> np.ndarray:
    return np.clip((x - low) / (high - low), 0.0, 1.0)


# Generate correlated context variables for one simulated time series.
def generate_context(
    t_horizon: int,
    context_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    z = np.zeros((t_horizon, context_dim), dtype=float)

    innovation = rng.normal(size=(t_horizon, context_dim))

    phi_ar = 0.72

    for t in range(1, t_horizon):
        z[t] = (
            phi_ar * z[t - 1] + np.sqrt(1.0 - phi_ar**2) * innovation[t]
        )

    z = np.clip(z, -2.5, 2.5)

    return z


# Fit the random nonlinear basis using only the supplied training context.
def fit_residual_rff(
    z_train: np.ndarray,
    n_components: int,
    length_scale: float,
    ridge: float,
    rng: np.random.Generator,
) -> ResidualRFFMap:
    context_dim = z_train.shape[1]

    omega = rng.normal(
        loc=0.0,
        scale=1.0 / length_scale,
        size=(context_dim, n_components),
    )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_components)

    raw = (
        np.sqrt(2.0 / n_components) * np.cos(z_train @ omega + phase)
    )

    rff_mean = raw.mean(axis=0, keepdims=True)

    centered = (
        raw - rff_mean
    )

    design = np.column_stack([np.ones(len(z_train)), z_train])

    gram = (
        design.T @ design
    )

    penalty = (
        ridge * np.eye(gram.shape[0])
    )

    penalty[0, 0] = 0.0

    residual_coef = np.linalg.solve(gram + penalty, design.T @ centered)

    return ResidualRFFMap(
        omega=omega,
        phase=phase,
        rff_mean=rff_mean,
        residual_coef=residual_coef,
    )


# Create the changing ideal quantile function that the online learner follows.
def make_oracle_path(
    phi_clean: np.ndarray,
    drift_intensity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    t_horizon, feature_dim = (
        phi_clean.shape
    )

    u_base = np.zeros(feature_dim, dtype=float)

    u_base[0] = 1.85

    # Linear effects form the easy-to-read base of the simulated oracle.
    linear_end = min(5, feature_dim)

    linear_pattern = np.array([0.15, -0.11, 0.08, 0.06], dtype=float)

    u_base[
        1:linear_end
    ] = linear_pattern[: linear_end - 1]

    # A few nonlinear effects test whether the functional component adds useful capacity.
    nonlinear_start = linear_end

    nonlinear_pattern = np.array([0.18, -0.15, 0.12, -0.10, 0.08, -0.06], dtype=float)

    nonlinear_count = min(len(nonlinear_pattern), feature_dim - nonlinear_start)

    if nonlinear_count > 0:
        u_base[
            nonlinear_start:
            nonlinear_start + nonlinear_count
        ] = nonlinear_pattern[:nonlinear_count]

    direction_smooth = rng.normal(size=feature_dim)

    direction_jump = rng.normal(size=feature_dim)

    # Limit intercept drift so the ideal interval expansion stays positive.
    direction_smooth[0] *= 0.10
    direction_jump[0] *= 0.10

    direction_smooth /= max(np.linalg.norm(direction_smooth), 1e-12)

    direction_jump /= max(np.linalg.norm(direction_jump), 1e-12)

    time_fraction = (
        np.arange(t_horizon) / max(t_horizon - 1, 1)
    )

    smooth_scalar = (
        0.34 * np.sin(2.0 * np.pi * time_fraction)
    )

    jump_scalar = (
        0.24 * (time_fraction >= 0.55).astype(float)
    )

    u_star = np.empty((t_horizon, feature_dim), dtype=float)

    for t in range(t_horizon):
        u_star[t] = (
            u_base
            + drift_intensity
            * (
                smooth_scalar[t] * direction_smooth + jump_scalar[t] * direction_jump
            )
        )

    q_star = np.einsum("td,td->t", u_star, phi_clean)

    # Keep scores positive because they represent an amount added to interval width.
    minimum_required = 1.05

    if q_star.min() < minimum_required:
        shift = (
            minimum_required - q_star.min() + 0.05
        )

        u_star[:, 0] += shift

        q_star = np.einsum("td,td->t", u_star, phi_clean)

    return u_star, q_star


# Measure how much the oracle parameters move from one step to the next.
def path_variation(
    u_star: np.ndarray,
) -> float:
    if len(u_star) <= 1:
        return 0.0

    return float(np.linalg.norm(np.diff(u_star, axis=0), axis=1).sum())


# Run one online learner against one known oracle path.
def run_single_path(
    t_horizon: int,
    drift_intensity: float,
    rho: float,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    rng = np.random.default_rng(seed)

    z_clean = generate_context(
        t_horizon=t_horizon,
        context_dim=args.context_dim,
        rng=rng,
    )

    map_train_size = max(100, int(args.train_fraction_map * t_horizon))

    map_train_size = min(map_train_size, t_horizon)

    feature_map = fit_residual_rff(
        z_train=z_clean[:map_train_size],
        n_components=(
            args.rff_components
        ),
        length_scale=(
            args.length_scale
        ),
        ridge=(
            args.residual_ridge
        ),
        rng=rng,
    )

    phi_clean = (
        feature_map.transform(z_clean)
    )

    (
        u_star,
        q_star,
    ) = make_oracle_path(
        phi_clean=phi_clean,
        drift_intensity=(
            drift_intensity
        ),
        rng=rng,
    )

    oracle_path_variation = (
        path_variation(u_star)
    )

    # This uniform noise has zero as its 0.9 quantile, matching 90% target coverage.
    score_noise = rng.uniform(-0.9, 0.1, size=t_horizon)

    score_clean = (
        q_star + score_noise
    )

    raw_delta = (
        rho * rng.normal(size=z_clean.shape)
    )

    raw_delta = np.clip(raw_delta, -3.0 * max(rho, 1e-12), 3.0 * max(rho, 1e-12))

    z_perturbed = np.clip(z_clean + raw_delta, -3.0, 3.0)

    delta = (
        z_perturbed - z_clean
    )

    d_t = np.linalg.norm(delta, axis=1)

    budget = float(d_t.sum())

    phi_perturbed = (
        feature_map.transform(z_perturbed)
    )

    score_shift = (
        args.score_sensitivity * delta[:, 0]
    )

    score_perturbed = np.maximum(0.0, score_clean + score_shift)

    tau = 1.0 - args.alpha

    eta = (
        args.eta_scale / np.sqrt(t_horizon)
    )

    feature_dim = (
        phi_clean.shape[1]
    )

    u = np.zeros(feature_dim, dtype=float)

    # A short warm-up gives the learner a sensible nonnegative starting expansion.
    u[0] = 1.20

    clean_moment_sum = np.zeros(feature_dim, dtype=float)

    perturbed_moment_sum = np.zeros(feature_dim, dtype=float)

    clean_conditional_moment_sum = np.zeros(feature_dim, dtype=float)

    perturbed_conditional_moment_sum = np.zeros(feature_dim, dtype=float)

    dynamic_regret_clean = 0.0
    dynamic_regret_perturbed = 0.0

    projection_correction = 0.0

    flip_count = 0

    min_q_clean = np.inf
    min_q_perturbed = np.inf

    max_feature_norm = 0.0

    q_clean_series = np.empty(t_horizon, dtype=float)

    q_perturbed_series = np.empty(t_horizon, dtype=float)

    e_clean_series = np.empty(t_horizon, dtype=int)

    e_perturbed_series = np.empty(t_horizon, dtype=int)

    for t in range(t_horizon):
        psi_clean_t = (
            phi_clean[t]
        )

        psi_perturbed_t = (
            phi_perturbed[t]
        )

        max_feature_norm = max(
            max_feature_norm,
            float(np.linalg.norm(psi_clean_t)),
            float(np.linalg.norm(psi_perturbed_t)),
        )

        q_clean_t = float(u @ psi_clean_t)

        q_perturbed_t = float(u @ psi_perturbed_t)

        q_clean_series[t] = (
            q_clean_t
        )

        q_perturbed_series[t] = (
            q_perturbed_t
        )

        min_q_clean = min(min_q_clean, q_clean_t)

        min_q_perturbed = min(min_q_perturbed, q_perturbed_t)

        e_clean = int(score_clean[t] > q_clean_t)

        e_perturbed = int(score_perturbed[t] > q_perturbed_t)

        e_clean_series[t] = e_clean
        e_perturbed_series[t] = (
            e_perturbed
        )

        s_clean = (
            e_clean - args.alpha
        )

        s_perturbed = (
            e_perturbed - args.alpha
        )

        clean_moment_sum += (
            s_clean * psi_clean_t
        )

        perturbed_moment_sum += (
            s_perturbed * psi_perturbed_t
        )

        # Known simulation noise lets us calculate conditional miscoverage exactly.
        clean_threshold = (
            q_clean_t - q_star[t]
        )

        p_miss_clean = (
            1.0 - uniform_cdf(np.array([clean_threshold]))[0]
        )

        mu_clean = (
            p_miss_clean - args.alpha
        )

        clean_conditional_moment_sum += (
            mu_clean * psi_clean_t
        )

        # Perturbed scores add the chosen shift to the clean oracle score and noise.
        perturbed_threshold = (
            q_perturbed_t - q_star[t] - score_shift[t]
        )

        p_miss_perturbed = (
            1.0 - uniform_cdf(np.array([perturbed_threshold]))[0]
        )

        mu_perturbed = (
            p_miss_perturbed - args.alpha
        )

        perturbed_conditional_moment_sum += (
            mu_perturbed * psi_perturbed_t
        )

        if e_clean != e_perturbed:
            flip_count += 1

        oracle_q_clean = (
            q_star[t]
        )

        clean_online_loss = float(pinball_loss(score_clean[t] - q_clean_t, tau))

        clean_oracle_loss = float(pinball_loss(score_clean[t] - oracle_q_clean, tau))

        dynamic_regret_clean += (
            clean_online_loss - clean_oracle_loss
        )

        # This diagnostic changes features but keeps the same oracle parameters.
        oracle_q_perturbed = float(u_star[t] @ psi_perturbed_t)

        perturbed_online_loss = float(
            pinball_loss(score_perturbed[t] - q_perturbed_t, tau)
        )

        perturbed_oracle_loss = float(
            pinball_loss(score_perturbed[t] - oracle_q_perturbed, tau)
        )

        dynamic_regret_perturbed += (
            perturbed_online_loss - perturbed_oracle_loss
        )

        unprojected = (
            u + eta * s_perturbed * psi_perturbed_t
        )

        u_next = project_l2_ball(unprojected, args.parameter_radius)

        projection_correction += float(np.linalg.norm(unprojected - u_next))

        u = u_next

    functional_error_clean = float(np.linalg.norm(clean_moment_sum / t_horizon))

    functional_error_perturbed = float(np.linalg.norm(perturbed_moment_sum / t_horizon))

    conditional_functional_error_clean = float(
        np.linalg.norm(clean_conditional_moment_sum / t_horizon)
    )

    conditional_functional_error_perturbed = float(
        np.linalg.norm(perturbed_conditional_moment_sum / t_horizon)
    )

    projection_correction_rate = (
        projection_correction / (eta * t_horizon)
    )

    deterministic_calibration_term = (
        2.0 * args.parameter_radius / (eta * t_horizon) + projection_correction_rate
    )

    drift_driver = (
        (1.0 + oracle_path_variation) / np.sqrt(t_horizon)
    )

    perturbation_driver = (
        budget / t_horizon
    )

    unified_regret_driver = (
        drift_driver + perturbation_driver
    )

    unified_calibration_driver = (
        1.0 / np.sqrt(t_horizon) + perturbation_driver
    )

    return {
        "T": t_horizon,
        "drift_intensity": (
            drift_intensity
        ),
        "rho": rho,
        "seed": seed,
        "eta": eta,
        "oracle_path_variation": (
            oracle_path_variation
        ),
        "perturbation_budget": budget,
        "budget_per_step": (
            perturbation_driver
        ),
        "flip_count": flip_count,
        "flip_rate": (
            flip_count / t_horizon
        ),
        "functional_error_clean": (
            functional_error_clean
        ),
        "functional_error_perturbed": (
            functional_error_perturbed
        ),
        "conditional_functional_error_clean": (
            conditional_functional_error_clean
        ),
        "conditional_functional_error_perturbed": (
            conditional_functional_error_perturbed
        ),
        "dynamic_regret_clean": (
            dynamic_regret_clean
        ),
        "dynamic_regret_clean_avg": (
            dynamic_regret_clean / t_horizon
        ),
        "dynamic_regret_perturbed": (
            dynamic_regret_perturbed
        ),
        "dynamic_regret_perturbed_avg": (
            dynamic_regret_perturbed / t_horizon
        ),
        "projection_correction_sum": (
            projection_correction
        ),
        "projection_correction_rate": (
            projection_correction_rate
        ),
        "deterministic_calibration_term": (
            deterministic_calibration_term
        ),
        "max_feature_norm": (
            max_feature_norm
        ),
        "minimum_q_clean": (
            min_q_clean
        ),
        "minimum_q_perturbed": (
            min_q_perturbed
        ),
        "drift_driver": (
            drift_driver
        ),
        "perturbation_driver": (
            perturbation_driver
        ),
        "unified_regret_driver": (
            unified_regret_driver
        ),
        "unified_calibration_driver": (
            unified_calibration_driver
        ),
    }


# Average repeated simulation paths for each experiment setting.
def aggregate_results(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in df.columns
        if column not in {*group_columns, "seed"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    grouped = (
        df.groupby(group_columns)[metric_columns].agg(["mean", "std", "count"])
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    summary = (
        grouped.reset_index()
    )

    for metric in metric_columns:
        count_column = (
            f"{metric}_count"
        )

        if count_column in summary.columns:
            summary[
                f"{metric}_se"
            ] = (
                summary[f"{metric}_std"] / np.sqrt(summary[count_column].clip(lower=1))
            )

    return summary


# Estimate a log-log slope only when there are enough positive finite values.
def safe_log_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    mask = (
        np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    )

    if mask.sum() < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)

    return float(slope)
