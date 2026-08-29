# Implement the four online calibration methods compared in the dissertation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


Array = np.ndarray


# Convert input values to a clean one-dimensional floating-point array.
def _as_1d(x: Array) -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)

    if not np.isfinite(arr).all():
        raise ValueError("Array contains NaN or infinite values.")

    return arr


# Convert context values to a clean two-dimensional feature matrix.
def _as_2d(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError("Expected a two-dimensional array.")

    if not np.isfinite(arr).all():
        raise ValueError("Array contains NaN or infinite values.")

    return arr


# Check the three interval arrays before starting an online run.
def _validate_interval_inputs(
    lower_raw: Array,
    upper_raw: Array,
    y_true: Array,
) -> tuple[Array, Array, Array]:
    lower = _as_1d(lower_raw)
    upper = _as_1d(upper_raw)
    y = _as_1d(y_true)

    if not (len(lower) == len(upper) == len(y)):
        raise ValueError("lower_raw, upper_raw and y_true must have equal length.")

    if np.any(lower > upper):
        raise ValueError("Found lower_raw > upper_raw.")

    return lower, upper, y


# Approximate an RBF kernel and centre the features using training data only.
class CenteredRandomFourierFeatures:
    """
    Centred Random Fourier Features for an RBF kernel.

    Raw map:
        phi(x) = sqrt(2 / D) cos(W^T x + b)

    Centred map:
        psi(x) = phi(x) - mean_train(phi(x))
    """

    # Store the feature-map settings; random values are created later in fit().
    def __init__(
        self,
        n_components: int = 128,
        length_scale: float = 2.0,
        random_state: int = 42,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        if length_scale <= 0:
            raise ValueError("length_scale must be positive.")

        self.n_components = int(n_components)
        self.length_scale = float(length_scale)
        self.random_state = int(random_state)

        self.random_weights_: Optional[Array] = None
        self.random_offsets_: Optional[Array] = None
        self.feature_mean_: Optional[Array] = None
        self.n_features_in_: Optional[int] = None

    # Apply the cosine map before subtracting the training-feature mean.
    def _raw_transform(self, x: Array) -> Array:
        x = _as_2d(x)

        if (
            self.random_weights_ is None or self.random_offsets_ is None
        ):
            raise RuntimeError("CenteredRandomFourierFeatures must be fitted first.")

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"got {x.shape[1]}."
            )

        projection = (
            x @ self.random_weights_ + self.random_offsets_
        )

        return (
            np.sqrt(2.0 / self.n_components) * np.cos(projection)
        )

    # Draw the random map once and remember its mean on the training rows.
    def fit(
        self,
        x: Array,
    ) -> "CenteredRandomFourierFeatures":
        x = _as_2d(x)

        self.n_features_in_ = x.shape[1]

        rng = np.random.default_rng(self.random_state)

        self.random_weights_ = rng.normal(
            loc=0.0,
            scale=1.0 / self.length_scale,
            size=(self.n_features_in_, self.n_components),
        )

        self.random_offsets_ = rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=self.n_components,
        )

        raw_train = self._raw_transform(x)
        self.feature_mean_ = raw_train.mean(axis=0)

        return self

    # Use the same random map and training mean for validation and test rows.
    def transform(self, x: Array) -> Array:
        if self.feature_mean_ is None:
            raise RuntimeError("CenteredRandomFourierFeatures must be fitted first.")

        return self._raw_transform(x) - self.feature_mean_

    # Fit the map and return the transformed training rows in one call.
    def fit_transform(self, x: Array) -> Array:
        return self.fit(x).transform(x)


# Remove the linear part from random features so this map focuses on nonlinearity.
class ResidualizedRandomFourierFeatures:
    """
    Nonlinear RFF features orthogonalised against the linear context.

    First construct centred RFF features phi_c(z). Then fit, on training data,

        phi_c(Z) approximately equals Z A.

    The residual feature map is

        psi_perp(z) = phi_c(z) - z A.

    Each residual feature is normalised using training residual standard
    deviation and then divided by sqrt(D), keeping its overall norm stable.
    """

    # Set up the centred base map and the ridge projection used during fitting.
    def __init__(
        self,
        n_components: int = 128,
        length_scale: float = 2.0,
        ridge: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")

        self.n_components = int(n_components)
        self.length_scale = float(length_scale)
        self.ridge = float(ridge)
        self.random_state = int(random_state)

        self.base_map = CenteredRandomFourierFeatures(
            n_components=n_components,
            length_scale=length_scale,
            random_state=random_state,
        )

        self.projection_coef_: Optional[Array] = None
        self.residual_scale_: Optional[Array] = None
        self.n_features_in_: Optional[int] = None

    # Learn how much of each random feature can already be explained linearly.
    def fit(
        self,
        x: Array,
    ) -> "ResidualizedRandomFourierFeatures":
        x = _as_2d(x)
        self.n_features_in_ = x.shape[1]

        centred_rff = self.base_map.fit_transform(x)

        gram = x.T @ x
        regularised_gram = (
            gram + self.ridge * np.eye(x.shape[1])
        )

        self.projection_coef_ = np.linalg.solve(regularised_gram, x.T @ centred_rff)

        residual = (
            centred_rff - x @ self.projection_coef_
        )

        residual_std = residual.std(axis=0, ddof=0)
        self.residual_scale_ = np.maximum(residual_std, 1e-6)

        return self

    # Return only the scaled nonlinear residual left after the linear projection.
    def transform(self, x: Array) -> Array:
        x = _as_2d(x)

        if (
            self.projection_coef_ is None or self.residual_scale_ is None
        ):
            raise RuntimeError(
                "ResidualizedRandomFourierFeatures must be fitted first."
            )

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"got {x.shape[1]}."
            )

        centred_rff = self.base_map.transform(x)

        residual = (
            centred_rff - x @ self.projection_coef_
        )

        normalised = residual / self.residual_scale_

        return normalised / np.sqrt(self.n_components)

    # Fit the residual map and transform the same training matrix.
    def fit_transform(self, x: Array) -> Array:
        return self.fit(x).transform(x)


# Keep every part of an online calibration path together for evaluation and saving.
@dataclass
class OnlineCalibrationResult:
    lower: Array
    upper: Array
    adjustment: Array
    miscoverage: Array
    feature_map: Array
    state_norm: Array
    global_component: Array
    linear_component: Array
    functional_component: Array

    # Convert the main one-dimensional outputs to a table for CSV files.
    def to_frame(
        self,
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lower": self.lower,
                "upper": self.upper,
                "adjustment": self.adjustment,
                "miscoverage": self.miscoverage.astype(int),
                "state_norm": self.state_norm,
                "global_component": self.global_component,
                "linear_component": self.linear_component,
                "functional_component": self.functional_component,
            },
            index=index,
        )


# Scalar ACI learns one global amount to add to both sides of every interval.
class ScalarACI:
    """Scalar online interval-widening baseline."""

    # Save the learning rate, target error rate, and allowed adjustment range.
    def __init__(
        self,
        alpha: float = 0.10,
        eta: float = 0.01,
        max_adjustment: float = 100.0,
        initial_adjustment: float = 0.0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")

        if eta <= 0:
            raise ValueError("eta must be positive.")

        if max_adjustment <= 0:
            raise ValueError("max_adjustment must be positive.")

        self.alpha = float(alpha)
        self.eta = float(eta)
        self.max_adjustment = float(max_adjustment)
        self.initial_adjustment = float(initial_adjustment)

        self.adjustment_ = self.initial_adjustment

    # Return the online state to the value used at the start of a fresh path.
    def reset(self) -> None:
        self.adjustment_ = self.initial_adjustment

    # Replay the observations in order and update one scalar after each outcome.
    def run(
        self,
        lower_raw: Array,
        upper_raw: Array,
        y_true: Array,
        *,
        reset: bool = True,
    ) -> OnlineCalibrationResult:
        lower_raw, upper_raw, y_true = _validate_interval_inputs(
            lower_raw,
            upper_raw,
            y_true,
        )

        if reset:
            self.reset()

        n = len(y_true)

        lower = np.empty(n)
        upper = np.empty(n)
        adjustment = np.empty(n)
        miscoverage = np.empty(n, dtype=int)
        state_norm = np.empty(n)
        global_component = np.empty(n)

        for t in range(n):
            # First form the interval from the old state, then observe whether it missed.
            q_t = float(np.clip(self.adjustment_, 0.0, self.max_adjustment))

            lower[t] = lower_raw[t] - q_t
            upper[t] = upper_raw[t] + q_t

            miss_t = int(y_true[t] < lower[t] or y_true[t] > upper[t])

            adjustment[t] = q_t
            miscoverage[t] = miss_t
            global_component[t] = q_t

            # Only future intervals can use the update based on miss_t.
            self.adjustment_ = float(
                np.clip(
                    self.adjustment_ + self.eta * (miss_t - self.alpha),
                    0.0,
                    self.max_adjustment,
                )
            )

            state_norm[t] = abs(self.adjustment_)

        zeros = np.zeros(n)

        return OnlineCalibrationResult(
            lower=lower,
            upper=upper,
            adjustment=adjustment,
            miscoverage=miscoverage,
            feature_map=np.ones((n, 1)),
            state_norm=state_norm,
            global_component=global_component,
            linear_component=zeros.copy(),
            functional_component=zeros.copy(),
        )


# Linear contextual ACI lets the adjustment change with the current context vector.
class LinearContextualACI:
    """
    Linear contextual ACI.

        q_t = clip(b_t + beta_t^T z_t, 0, B)
    """

    # Store separate learning rates for the global and context-dependent parts.
    def __init__(
        self,
        alpha: float = 0.10,
        eta_global: float = 0.01,
        eta_linear: float = 0.005,
        linear_radius: float = 10.0,
        max_adjustment: float = 100.0,
        initial_intercept: float = 0.0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")

        if eta_global <= 0 or eta_linear <= 0:
            raise ValueError("Learning rates must be positive.")

        if linear_radius <= 0:
            raise ValueError("linear_radius must be positive.")

        self.alpha = float(alpha)
        self.eta_global = float(eta_global)
        self.eta_linear = float(eta_linear)
        self.linear_radius = float(linear_radius)
        self.max_adjustment = float(max_adjustment)
        self.initial_intercept = float(initial_intercept)

        self.intercept_ = self.initial_intercept
        self.linear_weights_: Optional[Array] = None

    # Start a new online path with zero context weights.
    def reset(self, n_features: int) -> None:
        self.intercept_ = self.initial_intercept
        self.linear_weights_ = np.zeros(n_features)

    # Keep the linear weights inside the chosen radius to avoid unbounded updates.
    def _project_linear(self) -> None:
        norm = float(np.linalg.norm(self.linear_weights_))

        if norm > self.linear_radius:
            self.linear_weights_ *= self.linear_radius / norm

    # Make and update contextual intervals one time step at a time.
    def run(
        self,
        lower_raw: Array,
        upper_raw: Array,
        y_true: Array,
        context: Array,
        *,
        reset: bool = True,
    ) -> OnlineCalibrationResult:
        lower_raw, upper_raw, y_true = _validate_interval_inputs(
            lower_raw,
            upper_raw,
            y_true,
        )

        context = _as_2d(context)

        if len(context) != len(y_true):
            raise ValueError("context and y_true must align.")

        if reset or self.linear_weights_ is None:
            self.reset(context.shape[1])

        n = len(y_true)

        lower = np.empty(n)
        upper = np.empty(n)
        adjustment = np.empty(n)
        # Save the full path so the same predictions can be used by every metric.
        miscoverage = np.empty(n, dtype=int)
        state_norm = np.empty(n)
        global_component = np.empty(n)
        linear_component = np.empty(n)

        for t in range(n):
            # Prediction uses weights learned from earlier rows only.
            linear_t = float(self.linear_weights_ @ context[t])

            q_t = float(np.clip(self.intercept_ + linear_t, 0.0, self.max_adjustment))

            lower[t] = lower_raw[t] - q_t
            upper[t] = upper_raw[t] + q_t

            miss_t = int(y_true[t] < lower[t] or y_true[t] > upper[t])

            adjustment[t] = q_t
            miscoverage[t] = miss_t
            global_component[t] = self.intercept_
            linear_component[t] = linear_t

            error = float(miss_t - self.alpha)

            # Learn from this outcome only after its interval and error are stored.
            self.intercept_ = float(
                np.clip(
                    self.intercept_ + self.eta_global * error,
                    0.0,
                    self.max_adjustment,
                )
            )

            self.linear_weights_ += (
                self.eta_linear * error * context[t]
            )

            self._project_linear()

            state_norm[t] = float(
                np.sqrt(self.intercept_**2 + np.linalg.norm(self.linear_weights_)**2)
            )

        return OnlineCalibrationResult(
            lower=lower,
            upper=upper,
            adjustment=adjustment,
            miscoverage=miscoverage,
            feature_map=context,
            state_norm=state_norm,
            global_component=global_component,
            linear_component=linear_component,
            functional_component=np.zeros(n),
        )


# Functional ACI uses nonlinear random features instead of a linear context term.
class FunctionalACI:
    """
    Pure nonlinear Functional ACI using centred RFF.

        q_t = clip(b_t + w_t^T psi(z_t), 0, B)
    """

    # Build the random-feature map and store the two online learning rates.
    def __init__(
        self,
        alpha: float = 0.10,
        eta_global: float = 0.01,
        eta_functional: float = 0.005,
        n_components: int = 128,
        length_scale: float = 2.0,
        functional_radius: float = 10.0,
        max_adjustment: float = 100.0,
        initial_intercept: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.alpha = float(alpha)
        self.eta_global = float(eta_global)
        self.eta_functional = float(eta_functional)
        self.functional_radius = float(functional_radius)
        self.max_adjustment = float(max_adjustment)
        self.initial_intercept = float(initial_intercept)

        self.feature_map = CenteredRandomFourierFeatures(
            n_components=n_components,
            length_scale=length_scale,
            random_state=random_state,
        )

        self.intercept_ = self.initial_intercept
        self.functional_weights_: Optional[Array] = None

    # Expose the random-feature dimension used to size the online state.
    @property
    def n_components(self) -> int:
        return self.feature_map.n_components

    # Fit random features on training context only, before online evaluation begins.
    def fit_feature_map(
        self,
        train_context: Array,
    ) -> "FunctionalACI":
        self.feature_map.fit(train_context)
        self.reset()
        return self

    # Reset the intercept and all nonlinear weights for a fresh path.
    def reset(self) -> None:
        self.intercept_ = self.initial_intercept
        self.functional_weights_ = np.zeros(self.n_components)

    # Apply the already fitted training feature map to new context rows.
    def transform_context(self, context: Array) -> Array:
        return self.feature_map.transform(context)

    # Keep nonlinear weights inside their fixed radius after every update.
    def _project_functional(self) -> None:
        norm = float(np.linalg.norm(self.functional_weights_))

        if norm > self.functional_radius:
            self.functional_weights_ *= (
                self.functional_radius / norm
            )

    # Replay a sequence using the nonlinear adjustment learned so far.
    def run(
        self,
        lower_raw: Array,
        upper_raw: Array,
        y_true: Array,
        context: Array,
        *,
        reset: bool = True,
    ) -> OnlineCalibrationResult:
        lower_raw, upper_raw, y_true = _validate_interval_inputs(
            lower_raw,
            upper_raw,
            y_true,
        )

        context = _as_2d(context)

        if self.functional_weights_ is None:
            raise RuntimeError("Call fit_feature_map first.")

        if reset:
            self.reset()

        psi = self.transform_context(context)
        n = len(y_true)

        lower = np.empty(n)
        upper = np.empty(n)
        adjustment = np.empty(n)
        # Keep each part of the path for later coverage and component analysis.
        miscoverage = np.empty(n, dtype=int)
        state_norm = np.empty(n)
        global_component = np.empty(n)
        functional_component = np.empty(n)

        for t in range(n):
            # The current interval uses only the nonlinear weights from earlier rows.
            functional_t = float(self.functional_weights_ @ psi[t])

            q_t = float(
                np.clip(self.intercept_ + functional_t, 0.0, self.max_adjustment)
            )

            lower[t] = lower_raw[t] - q_t
            upper[t] = upper_raw[t] + q_t

            miss_t = int(y_true[t] < lower[t] or y_true[t] > upper[t])

            adjustment[t] = q_t
            miscoverage[t] = miss_t
            global_component[t] = self.intercept_
            functional_component[t] = functional_t

            error = float(miss_t - self.alpha)

            # Update the state after recording whether the current interval missed.
            self.intercept_ = float(
                np.clip(
                    self.intercept_ + self.eta_global * error,
                    0.0,
                    self.max_adjustment,
                )
            )

            self.functional_weights_ += (
                self.eta_functional * error * psi[t]
            )

            self._project_functional()

            state_norm[t] = float(
                np.sqrt(
                    self.intercept_**2 + np.linalg.norm(self.functional_weights_)**2
                )
            )

        return OnlineCalibrationResult(
            lower=lower,
            upper=upper,
            adjustment=adjustment,
            miscoverage=miscoverage,
            feature_map=psi,
            state_norm=state_norm,
            global_component=global_component,
            linear_component=np.zeros(n),
            functional_component=functional_component,
        )


# HF-ACI combines a readable linear term with a separate nonlinear residual term.
class HybridFunctionalACI:
    """
    Hybrid Functional ACI.

        q_t = clip(
            b_t
            + beta_t^T z_t
            + w_t^T psi_perp(z_t),
            0,
            B
        )

    The nonlinear feature map is residualised against the linear context,
    so the kernel component focuses on nonlinear residual structure.
    """

    # Store the three learning rates and create the residual random-feature map.
    def __init__(
        self,
        alpha: float = 0.10,
        eta_global: float = 0.01,
        eta_linear: float = 0.005,
        eta_functional: float = 0.005,
        n_components: int = 128,
        length_scale: float = 2.0,
        residual_ridge: float = 1e-3,
        linear_radius: float = 10.0,
        functional_radius: float = 5.0,
        max_adjustment: float = 100.0,
        initial_intercept: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.alpha = float(alpha)
        self.eta_global = float(eta_global)
        self.eta_linear = float(eta_linear)
        self.eta_functional = float(eta_functional)

        self.linear_radius = float(linear_radius)
        self.functional_radius = float(functional_radius)
        self.max_adjustment = float(max_adjustment)
        self.initial_intercept = float(initial_intercept)

        self.feature_map = ResidualizedRandomFourierFeatures(
            n_components=n_components,
            length_scale=length_scale,
            ridge=residual_ridge,
            random_state=random_state,
        )

        self.intercept_ = self.initial_intercept
        self.linear_weights_: Optional[Array] = None
        self.functional_weights_: Optional[Array] = None
        self.n_context_features_: Optional[int] = None

    # Expose the number of nonlinear weights needed by the online state.
    @property
    def n_components(self) -> int:
        return self.feature_map.n_components

    # Fit the nonlinear residual map on training context and initialise both states.
    def fit_feature_map(
        self,
        train_context: Array,
    ) -> "HybridFunctionalACI":
        train_context = _as_2d(train_context)

        self.n_context_features_ = train_context.shape[1]
        self.feature_map.fit(train_context)
        self.reset()

        return self

    # Start a fresh path with zero linear and nonlinear weights.
    def reset(self) -> None:
        if self.n_context_features_ is None:
            raise RuntimeError("Call fit_feature_map first.")

        self.intercept_ = self.initial_intercept
        self.linear_weights_ = np.zeros(self.n_context_features_)
        self.functional_weights_ = np.zeros(self.n_components)

    # Convert new context rows using the residual map fitted on training data.
    def transform_context(self, context: Array) -> Array:
        return self.feature_map.transform(context)

    # Keep the linear part inside its chosen weight radius.
    def _project_linear(self) -> None:
        norm = float(np.linalg.norm(self.linear_weights_))

        if norm > self.linear_radius:
            self.linear_weights_ *= self.linear_radius / norm

    # Keep the nonlinear part inside its own weight radius.
    def _project_functional(self) -> None:
        norm = float(np.linalg.norm(self.functional_weights_))

        if norm > self.functional_radius:
            self.functional_weights_ *= (
                self.functional_radius / norm
            )

    # Replay the hybrid adjustment in time order without looking ahead.
    def run(
        self,
        lower_raw: Array,
        upper_raw: Array,
        y_true: Array,
        context: Array,
        *,
        reset: bool = True,
    ) -> OnlineCalibrationResult:
        lower_raw, upper_raw, y_true = _validate_interval_inputs(
            lower_raw,
            upper_raw,
            y_true,
        )

        context = _as_2d(context)

        if self.linear_weights_ is None:
            raise RuntimeError("Call fit_feature_map first.")

        if len(context) != len(y_true):
            raise ValueError("context and y_true must align.")

        if reset:
            self.reset()

        psi = self.transform_context(context)
        n = len(y_true)

        lower = np.empty(n)
        upper = np.empty(n)
        adjustment = np.empty(n)
        # Store the separate components so their contribution can be inspected later.
        miscoverage = np.empty(n, dtype=int)
        state_norm = np.empty(n)

        global_component = np.empty(n)
        linear_component = np.empty(n)
        functional_component = np.empty(n)

        for t in range(n):
            # Both components use states learned only from earlier outcomes.
            linear_t = float(self.linear_weights_ @ context[t])

            functional_t = float(self.functional_weights_ @ psi[t])

            q_t = float(
                np.clip(
                    self.intercept_ + linear_t + functional_t,
                    0.0,
                    self.max_adjustment,
                )
            )

            lower[t] = lower_raw[t] - q_t
            upper[t] = upper_raw[t] + q_t

            miss_t = int(y_true[t] < lower[t] or y_true[t] > upper[t])

            adjustment[t] = q_t
            miscoverage[t] = miss_t

            global_component[t] = self.intercept_
            linear_component[t] = linear_t
            functional_component[t] = functional_t

            error = float(miss_t - self.alpha)

            # The current outcome updates all three states for the next prediction.
            self.intercept_ = float(
                np.clip(
                    self.intercept_ + self.eta_global * error,
                    0.0,
                    self.max_adjustment,
                )
            )

            self.linear_weights_ += (
                self.eta_linear * error * context[t]
            )

            self.functional_weights_ += (
                self.eta_functional * error * psi[t]
            )

            self._project_linear()
            self._project_functional()

            state_norm[t] = float(
                np.sqrt(
                    self.intercept_**2 + np.linalg.norm(self.linear_weights_)**2
                    + np.linalg.norm(self.functional_weights_)**2
                )
            )

        return OnlineCalibrationResult(
            lower=lower,
            upper=upper,
            adjustment=adjustment,
            miscoverage=miscoverage,
            feature_map=psi,
            state_norm=state_norm,
            global_component=global_component,
            linear_component=linear_component,
            functional_component=functional_component,
        )
