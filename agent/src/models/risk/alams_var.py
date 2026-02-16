"""
A-LAMS-VaR: Asymmetric Liquidity-Adjusted Markov-Switching Value-at-Risk

5-regime Markov-switching model with:
- Hamilton filter for regime probability estimation
- Asymmetric transition matrix (negative returns -> higher P of high-vol regimes)
- AMM liquidity adjustment (regime-dependent slippage)
- Two-stage MLE estimation via scipy.optimize

Mathematical framework:
  VaR_α(t) = Σ_k P(S_t=k | Ω_{t-1}) · [μ_k + σ_k · Φ^{-1}(α)] + λ(k, V_t) · Slippage(k)

References:
  - Hamilton (1989) - Regime switching models
  - Diebold, Lee, Weinbach (1994) - Asymmetric transitions
  - Cortex whitepaper - Liquidity adjustment for AMM slippage
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from scipy import optimize, stats

import structlog

from cortex.persistence import PersistentStore

logger = structlog.get_logger(__name__)

# Module-level persistent store for A-LAMS-VaR model states.
# Uses Redis when available, falls back to in-memory dict.
_alams_store = PersistentStore("alams_var")

# ============= CONFIGURATION =============


@dataclass
class ALAMSConfig:
    """Configuration for the A-LAMS-VaR model."""

    n_regimes: int = 5
    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])
    min_observations: int = 100
    max_iter: int = 200
    tol: float = 1e-6
    asymmetry_prior: float = 0.15
    regularization: float = 1e-4


@dataclass
class LiquidityConfig:
    """Configuration for AMM liquidity adjustment."""

    base_slippage_bps: float = 5.0
    regime_multipliers: list[float] = field(
        default_factory=lambda: [0.5, 0.8, 1.0, 1.5, 3.0]
    )
    impact_exponent: float = 1.5
    max_slippage_bps: float = 500.0


# ============= MODEL =============


class ALAMSVaRModel:
    """
    Asymmetric Liquidity-Adjusted Markov-Switching VaR.

    Estimates a K-regime model where each regime k has its own
    mean μ_k and variance σ²_k. Transition probabilities depend
    on the sign of past returns (asymmetry parameter δ).
    """

    def __init__(
        self,
        config: Optional[ALAMSConfig] = None,
        liquidity_config: Optional[LiquidityConfig] = None,
    ):
        self.config = config or ALAMSConfig()
        self.liquidity_config = liquidity_config or LiquidityConfig()

        K = self.config.n_regimes
        self.K = K

        # Regime parameters (estimated by fit)
        self.mu: np.ndarray = np.zeros(K)
        self.sigma: np.ndarray = np.ones(K)

        # Transition matrix P[i,j] = P(S_t=j | S_{t-1}=i, r > 0)
        self.P_base: np.ndarray = np.eye(K) * 0.9 + np.ones((K, K)) * 0.1 / K

        # Asymmetry parameter: shifts toward higher regimes after negative returns
        self.delta: float = self.config.asymmetry_prior

        # Filtered probabilities from last filter() call
        self.filtered_probs: Optional[np.ndarray] = None
        self.log_likelihood: float = -np.inf
        self.is_fitted: bool = False
        self.n_obs: int = 0

    # ============= ESTIMATION =============

    def fit(self, returns: np.ndarray) -> dict[str, float]:
        """
        Estimate model parameters via two-stage MLE.

        Stage 1: Estimate regime means and variances
        Stage 2: Estimate transition matrix and asymmetry parameter δ

        Args:
            returns: Array of log returns (T,)

        Returns:
            Dict with estimation diagnostics (log_likelihood, delta, n_obs, aic, bic)
        """
        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)

        if T < self.config.min_observations:
            raise ValueError(
                f"Need at least {self.config.min_observations} observations, got {T}"
            )

        self.n_obs = T
        K = self.K

        # Initialize regime params via quantile-based clustering
        self._initialize_params(returns)

        # Stage 1: Estimate μ_k, σ_k and P_base jointly
        logger.info("alams_var.fit.stage1", n_obs=T, n_regimes=K)
        result1 = self._estimate_stage1(returns)

        # Stage 2: Estimate asymmetry parameter δ
        logger.info("alams_var.fit.stage2", initial_delta=self.delta)
        result2 = self._estimate_stage2(returns)

        # Final filter pass with estimated parameters
        self.filtered_probs, self.log_likelihood = self._hamilton_filter(returns)

        self.is_fitted = True

        n_params = K * 2 + K * (K - 1) + 1  # mu, sigma, transitions, delta
        aic = -2 * self.log_likelihood + 2 * n_params
        bic = -2 * self.log_likelihood + n_params * np.log(T)

        diagnostics = {
            "log_likelihood": self.log_likelihood,
            "delta": self.delta,
            "n_obs": T,
            "n_regimes": K,
            "aic": aic,
            "bic": bic,
            "stage1_success": result1.success,
            "stage2_success": result2.success,
        }

        logger.info("alams_var.fit.complete", **diagnostics)
        return diagnostics

    def _initialize_params(self, returns: np.ndarray) -> None:
        """Initialize regime parameters via volatility quantiles."""
        K = self.K
        T = len(returns)

        # Sort absolute returns to define volatility regimes
        rolling_vol = np.array(
            [np.std(returns[max(0, i - 20) : i + 1]) for i in range(T)]
        )
        quantiles = np.linspace(0, 1, K + 1)[1:-1]
        thresholds = np.quantile(rolling_vol, quantiles)

        # Assign each observation to a regime based on local volatility
        assignments = np.digitize(rolling_vol, thresholds)

        for k in range(K):
            mask = assignments == k
            if mask.sum() > 1:
                self.mu[k] = np.mean(returns[mask])
                self.sigma[k] = max(np.std(returns[mask]), 1e-6)
            else:
                self.mu[k] = np.mean(returns)
                self.sigma[k] = np.std(returns) * (0.5 + k * 0.5)

        # Ensure sigmas are monotonically increasing
        self.sigma = np.sort(self.sigma)

        # Initialize transition matrix: high persistence on diagonal
        self.P_base = np.eye(K) * 0.85
        for i in range(K):
            off_diag = 0.15 / (K - 1)
            for j in range(K):
                if i != j:
                    self.P_base[i, j] = off_diag

    def _estimate_stage1(self, returns: np.ndarray) -> optimize.OptimizeResult:
        """Stage 1: Estimate regime means, variances, and base transition matrix."""
        K = self.K

        # Pack parameters: [mu_0..mu_K-1, log_sigma_0..log_sigma_K-1, P_logit_flat]
        # Transition matrix is parameterized via softmax rows
        x0 = self._pack_stage1()

        def neg_log_likelihood(x: np.ndarray) -> float:
            self._unpack_stage1(x)
            _, ll = self._hamilton_filter(returns)
            reg = self.config.regularization * np.sum(x**2)
            return -(ll - reg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                options={"maxiter": self.config.max_iter, "ftol": self.config.tol},
            )

        self._unpack_stage1(result.x)
        # Enforce sigma ordering after optimization
        order = np.argsort(self.sigma)
        self.sigma = self.sigma[order]
        self.mu = self.mu[order]
        self.P_base = self.P_base[order][:, order]

        return result

    def _estimate_stage2(self, returns: np.ndarray) -> optimize.OptimizeResult:
        """Stage 2: Estimate asymmetry parameter δ with fixed mu/sigma."""
        x0 = np.array([self.delta])

        def neg_log_likelihood(x: np.ndarray) -> float:
            self.delta = float(np.clip(x[0], 0.0, 0.5))
            _, ll = self._hamilton_filter(returns)
            return -ll

        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=[(0.001, 0.5)],
            options={"maxiter": 50, "ftol": self.config.tol},
        )

        self.delta = float(np.clip(result.x[0], 0.001, 0.5))
        return result

    def _pack_stage1(self) -> np.ndarray:
        """Pack model params into optimization vector."""
        K = self.K
        parts = [
            self.mu,
            np.log(np.clip(self.sigma, 1e-8, None)),
            self._transition_to_logit(self.P_base).ravel(),
        ]
        return np.concatenate(parts)

    def _unpack_stage1(self, x: np.ndarray) -> None:
        """Unpack optimization vector into model params."""
        K = self.K
        idx = 0
        self.mu = x[idx : idx + K].copy()
        idx += K
        self.sigma = np.exp(x[idx : idx + K])
        self.sigma = np.clip(self.sigma, 1e-8, 10.0)
        idx += K
        logits = x[idx:].reshape(K, K)
        self.P_base = self._logit_to_transition(logits)

    @staticmethod
    def _transition_to_logit(P: np.ndarray) -> np.ndarray:
        """Convert transition matrix to unconstrained logit space."""
        return np.log(np.clip(P, 1e-10, None))

    @staticmethod
    def _logit_to_transition(logits: np.ndarray) -> np.ndarray:
        """Convert logits back to valid transition matrix (rows sum to 1)."""
        # Softmax per row
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # ============= HAMILTON FILTER =============

    def _get_transition_matrix(self, r_prev: float) -> np.ndarray:
        """
        Get asymmetric transition matrix conditioned on previous return sign.

        When r_prev < 0, increase probability of transitioning to higher
        volatility regimes by δ. This captures the empirical finding that
        negative returns make high-volatility states more likely.
        """
        K = self.K
        P = self.P_base.copy()

        if r_prev < 0:
            # Shift probability mass toward higher regimes
            for i in range(K):
                shift = np.zeros(K)
                for j in range(K):
                    # Higher regimes get positive shift, lower get negative
                    direction = (j - i) / max(K - 1, 1)
                    shift[j] = self.delta * direction

                P[i] = P[i] + shift
                P[i] = np.clip(P[i], 1e-10, None)
                P[i] /= P[i].sum()

        return P

    def _regime_density(self, r: float) -> np.ndarray:
        """
        Compute P(r_t | S_t=k) for each regime k.
        Each regime is a Gaussian with mean μ_k and std σ_k.
        """
        densities = np.zeros(self.K)
        for k in range(self.K):
            densities[k] = stats.norm.pdf(r, loc=self.mu[k], scale=self.sigma[k])
        return densities

    def _hamilton_filter(
        self, returns: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Hamilton (1989) filter: forward recursion for regime probabilities.

        Returns:
            (filtered_probs, log_likelihood)
            filtered_probs: (T, K) array of P(S_t=k | Ω_t)
            log_likelihood: scalar log-likelihood of the data
        """
        T = len(returns)
        K = self.K

        # Storage
        filtered = np.zeros((T, K))
        log_lik = 0.0

        # Initial state: ergodic distribution of base transition matrix
        xi = self._ergodic_distribution(self.P_base)

        for t in range(T):
            # Prediction step: P(S_t=k | Ω_{t-1})
            if t == 0:
                predicted = xi
            else:
                P_t = self._get_transition_matrix(returns[t - 1])
                predicted = filtered[t - 1] @ P_t

            predicted = np.clip(predicted, 1e-10, None)
            predicted /= predicted.sum()

            # Update step: P(S_t=k | Ω_t) ∝ P(r_t|S_t=k) * P(S_t=k|Ω_{t-1})
            densities = self._regime_density(returns[t])
            joint = densities * predicted
            marginal = joint.sum()

            if marginal < 1e-300:
                filtered[t] = predicted
            else:
                filtered[t] = joint / marginal
                log_lik += np.log(marginal)

        return filtered, log_lik

    @staticmethod
    def _ergodic_distribution(P: np.ndarray) -> np.ndarray:
        """Compute stationary distribution of transition matrix."""
        K = P.shape[0]
        A = np.vstack([P.T - np.eye(K), np.ones(K)])
        b = np.zeros(K + 1)
        b[-1] = 1.0
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            result = np.clip(result, 1e-10, None)
            return result / result.sum()
        except np.linalg.LinAlgError:
            return np.ones(K) / K

    # ============= FILTERING (ONLINE) =============

    def filter(self, returns: np.ndarray) -> np.ndarray:
        """
        Run Hamilton filter on new data using fitted parameters.

        Args:
            returns: Array of log returns

        Returns:
            (T, K) array of filtered regime probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before filtering")

        self.filtered_probs, _ = self._hamilton_filter(np.asarray(returns))
        return self.filtered_probs

    def get_regime_probabilities(self) -> np.ndarray:
        """Return current regime probabilities (last time step)."""
        if self.filtered_probs is None:
            raise RuntimeError("No filtered probabilities available. Call fit() or filter() first.")
        return self.filtered_probs[-1].copy()

    def get_current_regime(self) -> int:
        """Return most likely current regime index (0 = lowest vol, K-1 = highest)."""
        probs = self.get_regime_probabilities()
        return int(np.argmax(probs))

    # ============= VAR CALCULATION =============

    def calculate_var(
        self,
        confidence: float = 0.95,
        returns: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate regime-weighted VaR (without liquidity adjustment).

        VaR_α = Σ_k P(S_t=k) · [μ_k + σ_k · Φ^{-1}(α)]

        Args:
            confidence: VaR confidence level (e.g. 0.95 for 95% VaR)
            returns: Optional new returns to filter first

        Returns:
            VaR as a positive number (loss magnitude at given confidence)
        """
        if returns is not None:
            self.filter(returns)

        probs = self.get_regime_probabilities()
        quantile = stats.norm.ppf(1 - confidence)

        var = 0.0
        for k in range(self.K):
            regime_var = self.mu[k] + self.sigma[k] * quantile
            var += probs[k] * regime_var

        # VaR is reported as positive loss
        return -var

    def calculate_liquidity_adjusted_var(
        self,
        confidence: float = 0.95,
        trade_size_usd: float = 0.0,
        pool_depth_usd: float = 1e9,
        returns: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Full A-LAMS-VaR: regime VaR + liquidity adjustment.

        VaR_α(t) = Σ_k P(S_t=k) · [μ_k + σ_k · Φ^{-1}(α)] + λ(k) · Slippage(k)

        Args:
            confidence: VaR confidence level
            trade_size_usd: Planned trade size in USD
            pool_depth_usd: AMM pool depth in USD
            returns: Optional new returns to filter first

        Returns:
            Dict with var_pure, slippage_component, var_total, regime, regime_probs
        """
        if returns is not None:
            self.filter(returns)

        probs = self.get_regime_probabilities()
        pure_var = self.calculate_var(confidence)

        # Liquidity adjustment
        slippage = self._estimate_slippage(trade_size_usd, pool_depth_usd, probs)

        return {
            "var_pure": pure_var,
            "slippage_component": slippage,
            "var_total": pure_var + slippage,
            "confidence": confidence,
            "current_regime": self.get_current_regime(),
            "regime_probs": probs.tolist(),
            "delta": self.delta,
            "regime_means": self.mu.tolist(),
            "regime_sigmas": self.sigma.tolist(),
        }

    def _estimate_slippage(
        self,
        trade_size_usd: float,
        pool_depth_usd: float,
        regime_probs: np.ndarray,
    ) -> float:
        """
        Estimate regime-weighted AMM slippage.

        Slippage scales with:
        - Trade size relative to pool depth (non-linear, exponent > 1)
        - Current regime (high-vol = worse liquidity = more slippage)
        """
        if trade_size_usd <= 0 or pool_depth_usd <= 0:
            return 0.0

        lc = self.liquidity_config
        size_ratio = trade_size_usd / pool_depth_usd

        # Non-linear price impact: slippage ~ (size/depth)^exponent
        base_impact = size_ratio**lc.impact_exponent

        # Regime-weighted slippage
        slippage_bps = 0.0
        for k in range(self.K):
            multiplier = (
                lc.regime_multipliers[k]
                if k < len(lc.regime_multipliers)
                else lc.regime_multipliers[-1]
            )
            regime_slippage = lc.base_slippage_bps * multiplier * (1 + base_impact * 100)
            slippage_bps += regime_probs[k] * regime_slippage

        slippage_bps = min(slippage_bps, lc.max_slippage_bps)

        # Convert bps to decimal
        return slippage_bps / 10000.0

    # ============= DIAGNOSTICS =============

    def summary(self) -> dict:
        """Return model summary for logging/reporting."""
        if not self.is_fitted:
            return {"is_fitted": False}

        probs = self.get_regime_probabilities()
        return {
            "is_fitted": True,
            "n_regimes": self.K,
            "n_obs": self.n_obs,
            "delta": round(self.delta, 4),
            "log_likelihood": round(self.log_likelihood, 2),
            "regime_means": [round(m, 6) for m in self.mu],
            "regime_sigmas": [round(s, 6) for s in self.sigma],
            "current_regime": self.get_current_regime(),
            "regime_probs": [round(p, 4) for p in probs],
            "var_95": round(self.calculate_var(0.95), 6),
            "var_99": round(self.calculate_var(0.99), 6),
        }

    # ============= PERSISTENCE =============

    def save_state(self, token: str = "default") -> None:
        """
        Persist all fitted model parameters to the Redis-backed store.

        Serialises regime means, variances, transition matrix, asymmetry
        parameter, filtered probabilities, and both config dataclasses.
        The store uses the same PersistentStore pattern as the rest of
        Cortex (lazy Redis write, in-memory fast path).

        Args:
            token: Identifier for this model instance (e.g. "SOL", "BTC").
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot persist an unfitted model.")

        state: dict[str, Any] = {
            "mu": self.mu,
            "sigma": self.sigma,
            "P_base": self.P_base,
            "delta": self.delta,
            "filtered_probs": self.filtered_probs,
            "log_likelihood": self.log_likelihood,
            "n_obs": self.n_obs,
            "K": self.K,
            "is_fitted": self.is_fitted,
            "config": dataclasses.asdict(self.config),
            "liquidity_config": dataclasses.asdict(self.liquidity_config),
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }
        _alams_store[token] = state
        logger.info("alams_var.save_state", token=token, n_obs=self.n_obs)

    @classmethod
    def load_state(cls, token: str = "default") -> "ALAMSVaRModel":
        """
        Reconstruct a fitted ALAMSVaRModel from the persistent store.

        Restores all regime parameters, transition matrix, asymmetry δ,
        filtered probabilities, and configuration so the model is immediately
        usable for VaR calculations without re-fitting.

        Args:
            token: Identifier for the model instance to load.

        Raises:
            KeyError: If no model is stored under the given token.
        """
        if token not in _alams_store:
            raise KeyError(f"No persisted A-LAMS model found for token '{token}'")

        state = _alams_store[token]

        config = ALAMSConfig(**state["config"])
        liq_config = LiquidityConfig(**state["liquidity_config"])
        model = cls(config=config, liquidity_config=liq_config)

        model.mu = np.asarray(state["mu"], dtype=np.float64)
        model.sigma = np.asarray(state["sigma"], dtype=np.float64)
        model.P_base = np.asarray(state["P_base"], dtype=np.float64)
        model.delta = float(state["delta"])
        model.filtered_probs = (
            np.asarray(state["filtered_probs"], dtype=np.float64)
            if state["filtered_probs"] is not None
            else None
        )
        model.log_likelihood = float(state["log_likelihood"])
        model.n_obs = int(state["n_obs"])
        model.K = int(state["K"])
        model.is_fitted = bool(state["is_fitted"])

        logger.info(
            "alams_var.load_state",
            token=token,
            n_obs=model.n_obs,
            delta=model.delta,
            fitted_at=state.get("fitted_at"),
        )
        return model

    @staticmethod
    async def restore_all() -> int:
        """Load all persisted models from Redis into the in-memory store."""
        return await _alams_store.restore()

    @staticmethod
    def list_models() -> list[str]:
        """Return list of persisted model tokens."""
        return list(_alams_store.keys())

    @staticmethod
    def delete_model(token: str) -> None:
        """Remove a persisted model from the store."""
        if token in _alams_store:
            del _alams_store[token]

    # ============= BACKTESTING =============

    def backtest(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        min_window: int = 100,
        refit_every: int = 50,
    ) -> dict[str, Any]:
        """
        Rolling out-of-sample VaR backtest with statistical validation.

        Generates a time series of one-step-ahead VaR forecasts using an
        expanding window, then evaluates model adequacy via Kupiec (1995)
        proportion-of-failures test and Christoffersen (1998) independence
        test from ``cortex.backtesting``.

        Algorithm:
            For t = min_window .. T-1:
                1. If (t - min_window) % refit_every == 0, re-fit on returns[0:t]
                2. Filter returns[0:t] to update regime probabilities
                3. Forecast VaR_{t+1} using current regime probs
                4. Record violation if returns[t+1] < -VaR_{t+1}

        Args:
            returns: Full array of log returns for backtesting.
            confidence: VaR confidence level (e.g. 0.95).
            min_window: Minimum observations before first forecast.
            refit_every: Re-estimate parameters every N steps.

        Returns:
            Dict with n_obs, n_violations, violation_rate, kupiec, christoffersen,
            var_forecasts, and per-step details.
        """
        from cortex.backtesting import christoffersen_test, kupiec_test

        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)

        if T < min_window + 10:
            raise ValueError(
                f"Need at least {min_window + 10} observations for backtest, got {T}"
            )

        var_forecasts: list[float] = []
        realized_returns: list[float] = []
        violations: list[int] = []
        regime_at_forecast: list[int] = []

        # Working model for backtesting (avoid mutating self)
        bt_model = ALAMSVaRModel(
            config=ALAMSConfig(
                n_regimes=self.config.n_regimes,
                confidence_levels=self.config.confidence_levels,
                min_observations=min(min_window, self.config.min_observations),
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                asymmetry_prior=self.config.asymmetry_prior,
                regularization=self.config.regularization,
            ),
            liquidity_config=self.liquidity_config,
        )

        last_fit_t = -refit_every  # force initial fit

        for t in range(min_window, T - 1):
            # Re-fit if needed
            if (t - min_window) % refit_every == 0 or not bt_model.is_fitted:
                try:
                    bt_model.fit(returns[:t])
                    last_fit_t = t
                except Exception:
                    if not bt_model.is_fitted:
                        continue
            else:
                # Just update filter with latest data
                bt_model.filter(returns[:t])

            # Forecast VaR for t+1
            var_t = bt_model.calculate_var(confidence)
            realized = returns[t + 1]

            # Violation: realized return is worse than -VaR (loss exceeds VaR)
            is_violation = 1 if realized < -var_t else 0

            var_forecasts.append(var_t)
            realized_returns.append(float(realized))
            violations.append(is_violation)
            regime_at_forecast.append(bt_model.get_current_regime())

        n_obs = len(violations)
        n_violations = sum(violations)
        violation_rate = n_violations / n_obs if n_obs > 0 else 0.0

        # Statistical tests
        confidence_pct = confidence * 100.0
        kup = kupiec_test(n_obs, n_violations, confidence_pct)
        chris = christoffersen_test(np.array(violations))

        # Conditional coverage (joint test)
        cc_statistic = kup["statistic"] + chris["statistic"]
        cc_pvalue = float(1.0 - stats.chi2.cdf(cc_statistic, df=2))

        result: dict[str, Any] = {
            "n_obs": n_obs,
            "n_violations": n_violations,
            "violation_rate": round(violation_rate, 6),
            "expected_rate": round(1.0 - confidence, 6),
            "confidence": confidence,
            "min_window": min_window,
            "refit_every": refit_every,
            "kupiec": kup,
            "christoffersen": chris,
            "conditional_coverage": {
                "statistic": round(cc_statistic, 4),
                "p_value": round(cc_pvalue, 4),
                "pass": cc_pvalue > 0.05,
            },
            "var_forecasts": var_forecasts,
            "realized_returns": realized_returns,
            "violations": violations,
            "regime_at_forecast": regime_at_forecast,
        }

        logger.info(
            "alams_var.backtest.complete",
            n_obs=n_obs,
            n_violations=n_violations,
            violation_rate=round(violation_rate, 4),
            kupiec_pass=kup["pass"],
            christoffersen_pass=chris["pass"],
            cc_pass=cc_pvalue > 0.05,
        )

        return result

    def backtest_multi_confidence(
        self,
        returns: np.ndarray,
        confidence_levels: Optional[list[float]] = None,
        min_window: int = 100,
        refit_every: int = 50,
    ) -> dict[str, Any]:
        """
        Run backtests at multiple confidence levels and return a summary.

        Args:
            returns: Full array of log returns.
            confidence_levels: List of confidence levels (default: [0.95, 0.99]).
            min_window: Minimum observations before first forecast.
            refit_every: Re-estimation frequency.

        Returns:
            Dict with per-level backtest results and overall assessment.
        """
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels

        results = {}
        all_pass = True

        for conf in confidence_levels:
            bt = self.backtest(returns, conf, min_window, refit_every)
            key = f"var_{int(conf * 100)}"
            results[key] = {
                "confidence": conf,
                "n_obs": bt["n_obs"],
                "n_violations": bt["n_violations"],
                "violation_rate": bt["violation_rate"],
                "expected_rate": bt["expected_rate"],
                "kupiec_pass": bt["kupiec"]["pass"],
                "kupiec_pvalue": bt["kupiec"]["p_value"],
                "christoffersen_pass": bt["christoffersen"]["pass"],
                "christoffersen_pvalue": bt["christoffersen"]["p_value"],
                "cc_pass": bt["conditional_coverage"]["pass"],
                "cc_pvalue": bt["conditional_coverage"]["p_value"],
            }
            if not bt["kupiec"]["pass"] or not bt["conditional_coverage"]["pass"]:
                all_pass = False

        return {
            "all_pass": all_pass,
            "confidence_levels": confidence_levels,
            "results": results,
        }
