"""
Multi-Asset Portfolio A-LAMS-VaR with Copula-Based Dependence.

Combines per-asset ALAMSVaRModel marginals with copula dependence modelling
from ``cortex.copula`` to compute portfolio-level VaR that respects:

  1. Regime-switching marginals — each asset has its own 5-regime A-LAMS model
     capturing asymmetric transition dynamics and liquidity-adjusted slippage.
  2. Copula joint dependence — captures non-linear tail co-movements that
     Gaussian correlation misses (Clayton lower-tail, Gumbel upper-tail,
     Student-t symmetric tail dependence).
  3. Regime-conditional copula mixture — crisis regimes use Student-t copula
     (stronger tail dependence) while calm regimes use Gaussian copula,
     blended proportionally to current regime probabilities.

Architecture:
  1. Fit individual ALAMSVaRModel per asset on its return series.
  2. Fit copula on joint pseudo-uniform margins (rank-transform → copula fit).
  3. Monte Carlo: sample joint uniforms from copula → invert per-asset
     marginal via regime-weighted σ_k · Φ⁻¹(u) → weighted portfolio return.
  4. Portfolio VaR = empirical quantile of simulated portfolio loss distribution.
  5. Per-asset liquidity adjustment via regime-weighted AMM slippage.

References:
  - Patton (2006) — Modelling asymmetric exchange rate dependence
  - McNeil, Frey & Embrechts (2005) — Quantitative Risk Management
  - Cortex whitepaper §4.3 — Copula-based portfolio risk
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

import structlog

from cortex.copula import (
    COPULA_FAMILIES,
    _ensure_positive_definite,
    _kendall_tau_matrix,
    _sample_copula,
    _tail_dependence,
    _to_uniform,
    _tau_to_pearson,
    compare_copulas,
    fit_copula,
)

from .alams_var import ALAMSConfig, ALAMSVaRModel, LiquidityConfig

logger = structlog.get_logger(__name__)


# ============= CONFIGURATION =============


@dataclasses.dataclass
class PortfolioVaRConfig:
    """Configuration for multi-asset portfolio VaR."""

    n_simulations: int = 10_000
    default_copula_family: str = "student_t"
    auto_select_copula: bool = True
    seed: int = 42
    min_obs_per_asset: int = 100
    regime_conditional: bool = True
    crisis_copula: str = "student_t"
    calm_copula: str = "gaussian"
    crisis_regime_threshold: float = 0.6


# ============= PORTFOLIO MODEL =============


class PortfolioALAMSVaR:
    """
    Multi-asset portfolio VaR using per-asset A-LAMS marginals + copula dependence.

    Fits individual ALAMSVaRModel per asset, then couples them via a copula
    (Gaussian, Student-t, Clayton, Gumbel, or Frank) for joint simulation.

    The copula captures tail dependence that Gaussian correlation misses,
    while per-asset A-LAMS marginals provide regime-switching, asymmetry,
    and liquidity adjustment.

    Example usage:
        portfolio = PortfolioALAMSVaR(["SOL", "BTC", "ETH"])
        portfolio.fit(returns_df)
        result = portfolio.calculate_portfolio_var(
            weights={"SOL": 0.5, "BTC": 0.3, "ETH": 0.2},
            confidence=0.95,
            trade_sizes_usd={"SOL": 10_000, "BTC": 5_000, "ETH": 3_000},
        )
    """

    def __init__(
        self,
        assets: list[str],
        config: Optional[PortfolioVaRConfig] = None,
        alams_config: Optional[ALAMSConfig] = None,
        liquidity_config: Optional[LiquidityConfig] = None,
    ):
        self.assets = list(assets)
        self.config = config or PortfolioVaRConfig()
        self.alams_config = alams_config or ALAMSConfig()
        self.liquidity_config = liquidity_config or LiquidityConfig()

        # Per-asset A-LAMS models
        self.models: dict[str, ALAMSVaRModel] = {}

        # Copula fit results
        self.copula_fit: Optional[dict] = None
        self.regime_copulas: Optional[list[dict]] = None

        # Joint returns used for fitting
        self._joint_returns: Optional[np.ndarray] = None

        self.is_fitted: bool = False

    # ============= FITTING =============

    def fit(
        self,
        returns: pd.DataFrame | dict[str, np.ndarray],
        alams_options: Optional[dict[str, dict]] = None,
    ) -> dict[str, Any]:
        """
        Fit per-asset A-LAMS models and joint copula on multivariate returns.

        Two-phase fitting:
          Phase 1: Fit individual ALAMSVaRModel per asset (parallel-safe).
          Phase 2: Fit copula on joint pseudo-uniform margins derived from
                   per-asset filtered probabilities and regime-weighted CDFs.

        Args:
            returns: DataFrame with columns per asset or dict {asset: ndarray}.
                     All series must have the same length.
            alams_options: Optional per-asset overrides for ALAMSConfig.
                           E.g. {"BTC": {"n_regimes": 3, "max_iter": 100}}.

        Returns:
            Dict with per-asset fit diagnostics, copula fit summary,
            and overall portfolio fitting status.

        Raises:
            ValueError: If an asset has fewer observations than min_obs_per_asset.
        """
        alams_options = alams_options or {}

        # Convert DataFrame to dict
        if isinstance(returns, pd.DataFrame):
            returns_dict = {col: returns[col].values for col in self.assets}
        else:
            returns_dict = returns

        # Validate all assets present and aligned
        n_obs = None
        for asset in self.assets:
            if asset not in returns_dict:
                raise ValueError(f"Missing return series for asset '{asset}'")
            arr = np.asarray(returns_dict[asset], dtype=np.float64)
            if n_obs is None:
                n_obs = len(arr)
            elif len(arr) != n_obs:
                raise ValueError(
                    f"Asset '{asset}' has {len(arr)} observations, "
                    f"expected {n_obs} (must be aligned)"
                )
            if len(arr) < self.config.min_obs_per_asset:
                raise ValueError(
                    f"Asset '{asset}' has {len(arr)} observations, "
                    f"need at least {self.config.min_obs_per_asset}"
                )

        # Phase 1: Fit per-asset A-LAMS models
        per_asset_diagnostics: dict[str, dict] = {}
        for asset in self.assets:
            asset_returns = np.asarray(returns_dict[asset], dtype=np.float64)

            # Merge base config with per-asset overrides
            cfg_overrides = alams_options.get(asset, {})
            cfg = dataclasses.replace(self.alams_config, **cfg_overrides)

            model = ALAMSVaRModel(config=cfg, liquidity_config=self.liquidity_config)
            diagnostics = model.fit(asset_returns)
            self.models[asset] = model
            per_asset_diagnostics[asset] = diagnostics

            logger.info(
                "portfolio_var.asset_fitted",
                asset=asset,
                n_obs=diagnostics["n_obs"],
                delta=round(diagnostics["delta"], 4),
                ll=round(diagnostics["log_likelihood"], 2),
            )

        # Build joint returns matrix (T x d) for copula fitting
        joint = np.column_stack(
            [returns_dict[asset] for asset in self.assets]
        )
        self._joint_returns = joint

        # Phase 2: Fit copula on joint returns
        copula_summary = self._fit_copula(joint)

        # Phase 2b: Optionally fit regime-conditional copulas
        regime_summary = None
        if self.config.regime_conditional:
            regime_summary = self._fit_regime_copulas(joint)

        self.is_fitted = True

        result = {
            "n_assets": len(self.assets),
            "assets": self.assets,
            "n_obs": n_obs,
            "per_asset": per_asset_diagnostics,
            "copula": copula_summary,
            "regime_copulas": regime_summary,
        }

        logger.info(
            "portfolio_var.fit_complete",
            n_assets=len(self.assets),
            copula_family=copula_summary["family"],
        )
        return result

    def _fit_copula(self, joint_returns: np.ndarray) -> dict:
        """Fit best copula (or specified family) on joint returns."""
        if self.config.auto_select_copula:
            ranked = compare_copulas(joint_returns)
            if ranked:
                self.copula_fit = ranked[0]
            else:
                self.copula_fit = fit_copula(
                    joint_returns, family=self.config.default_copula_family
                )
        else:
            self.copula_fit = fit_copula(
                joint_returns, family=self.config.default_copula_family
            )

        return {
            "family": self.copula_fit["family"],
            "log_likelihood": self.copula_fit["log_likelihood"],
            "aic": self.copula_fit["aic"],
            "bic": self.copula_fit["bic"],
            "tail_dependence": self.copula_fit["tail_dependence"],
            "n_params": self.copula_fit["n_params"],
        }

    def _fit_regime_copulas(self, joint_returns: np.ndarray) -> list[dict]:
        """
        Fit separate copulas per A-LAMS regime.

        For each regime k (across all assets, averaged):
          - Calm regimes (k < 60% of K) → Gaussian copula
          - Crisis regimes (k >= 60% of K) → Student-t copula

        This captures the empirical finding that tail dependence increases
        during market stress (Longin & Solnik, 2001).
        """
        # Average regime probabilities across assets to determine
        # which time periods belong to which aggregate regime
        K = self.alams_config.n_regimes
        n = joint_returns.shape[0]

        # Collect filtered probs from all asset models
        avg_filtered = np.zeros((n, K))
        for asset in self.assets:
            model = self.models[asset]
            if model.filtered_probs is not None and len(model.filtered_probs) == n:
                avg_filtered += model.filtered_probs
        avg_filtered /= len(self.assets)

        # Classify regimes: top 40% = crisis, bottom 60% = calm
        crisis_threshold = max(1, int(K * self.config.crisis_regime_threshold))

        self.regime_copulas = []
        for k in range(K):
            is_crisis = k >= crisis_threshold
            family = self.config.crisis_copula if is_crisis else self.config.calm_copula

            # Select observations where regime k has high probability
            w_k = avg_filtered[:, k]
            threshold = np.percentile(w_k, 50)
            mask = w_k >= threshold
            n_selected = int(mask.sum())

            # Need enough observations for copula fitting
            min_for_copula = max(30, len(self.assets) + 5)
            if n_selected < min_for_copula:
                regime_returns = joint_returns
            else:
                regime_returns = joint_returns[mask]

            try:
                copula = fit_copula(regime_returns, family=family)
            except Exception as exc:
                logger.warning(
                    "portfolio_var.regime_copula_fit_failed",
                    regime=k,
                    family=family,
                    error=str(exc),
                )
                # Fallback to Gaussian on full sample
                copula = fit_copula(joint_returns, family="gaussian")

            entry = {
                "regime": k,
                "is_crisis": is_crisis,
                "family": copula["family"],
                "n_obs_selected": n_selected,
                "copula": copula,
            }
            self.regime_copulas.append(entry)

        return [
            {
                "regime": rc["regime"],
                "is_crisis": rc["is_crisis"],
                "family": rc["family"],
                "n_obs_selected": rc["n_obs_selected"],
                "tail_dependence": rc["copula"]["tail_dependence"],
            }
            for rc in self.regime_copulas
        ]

    # ============= VaR CALCULATION =============

    def calculate_portfolio_var(
        self,
        weights: dict[str, float],
        confidence: float = 0.95,
        trade_sizes_usd: Optional[dict[str, float]] = None,
        pool_depths_usd: Optional[dict[str, float]] = None,
        n_simulations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Full portfolio A-LAMS-VaR with copula MC simulation.

        Algorithm:
          1. Sample joint uniform variates from fitted copula.
          2. For each asset i, convert u_i to return space using the
             per-asset regime-weighted marginal:
               r_i = Σ_k P(S_t=k) · σ_k · Φ⁻¹(u_i)
          3. Compute per-asset slippage (regime-weighted AMM impact).
          4. Portfolio return = Σ_i w_i · (r_i - slippage_i).
          5. VaR = empirical quantile of simulated portfolio losses.

        Args:
            weights: Asset weight dict, e.g. {"SOL": 0.5, "BTC": 0.3, "ETH": 0.2}.
            confidence: VaR confidence level (default 0.95).
            trade_sizes_usd: Per-asset planned trade sizes for slippage calc.
            pool_depths_usd: Per-asset AMM pool depths.
            n_simulations: Override default simulation count.
            seed: Override random seed.

        Returns:
            Dict with portfolio_var, per_asset_marginal_var, copula metadata,
            diversification_ratio, and simulation diagnostics.
        """
        if not self.is_fitted:
            raise RuntimeError("Portfolio model must be fitted first. Call fit().")

        n_sim = n_simulations or self.config.n_simulations
        rng_seed = seed or self.config.seed
        rng = np.random.RandomState(rng_seed)
        d = len(self.assets)
        alpha = 1.0 - confidence

        w = np.array([weights.get(a, 0.0) for a in self.assets])

        # Normalize weights if they don't sum to 1
        w_sum = w.sum()
        if abs(w_sum) > 1e-10:
            w_normalized = w / w_sum
        else:
            w_normalized = np.ones(d) / d

        # Decide which copula to sample from
        if self.config.regime_conditional and self.regime_copulas:
            u_sim = self._sample_regime_mixture(n_sim, d, rng)
        else:
            u_sim = self._sample_from_copula(self.copula_fit, n_sim, d, rng)

        # Per-asset marginal transformation: uniform → return space
        returns_sim = np.zeros((n_sim, d))
        per_asset_var = {}
        per_asset_slippage = {}

        for i, asset in enumerate(self.assets):
            model = self.models[asset]
            probs = model.get_regime_probabilities()

            # Regime-weighted standard deviation
            sigma_weighted = sum(
                probs[k] * model.sigma[k] for k in range(model.K)
            )
            # Regime-weighted mean
            mu_weighted = sum(
                probs[k] * model.mu[k] for k in range(model.K)
            )

            # Inverse CDF transform: u → z → return
            z_i = norm.ppf(np.clip(u_sim[:, i], 1e-10, 1 - 1e-10))
            returns_sim[:, i] = mu_weighted + sigma_weighted * z_i

            # Per-asset marginal VaR (without portfolio effect)
            per_asset_var[asset] = model.calculate_var(confidence)

            # Per-asset slippage
            trade_size = (trade_sizes_usd or {}).get(asset, 0.0)
            pool_depth = (pool_depths_usd or {}).get(asset, 1e9)
            slippage = model._estimate_slippage(trade_size, pool_depth, probs)
            per_asset_slippage[asset] = slippage

        # Slippage vector for portfolio adjustment
        slippage_vec = np.array([per_asset_slippage[a] for a in self.assets])

        # Portfolio returns (accounting for slippage as a cost)
        portfolio_returns = returns_sim @ w_normalized - np.dot(
            w_normalized, slippage_vec
        )

        # Portfolio VaR: alpha-quantile of portfolio loss distribution
        portfolio_var_pure = float(-np.percentile(portfolio_returns, alpha * 100))
        total_slippage = float(np.dot(w_normalized, slippage_vec))

        # Undiversified VaR: sum of weighted individual VaRs
        undiversified_var = sum(
            abs(weights.get(a, 0.0) / w_sum) * per_asset_var[a]
            for a in self.assets
        )

        # Diversification ratio
        div_ratio = (
            portfolio_var_pure / undiversified_var
            if undiversified_var > 1e-12
            else 1.0
        )

        # Expected shortfall (CVaR): average loss beyond VaR
        var_threshold = -portfolio_var_pure
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        expected_shortfall = float(-np.mean(tail_returns)) if len(tail_returns) > 0 else portfolio_var_pure

        # Copula tail info
        copula_info = {
            "family": self.copula_fit["family"] if self.copula_fit else "none",
            "tail_dependence": (
                self.copula_fit["tail_dependence"] if self.copula_fit else {}
            ),
        }

        result: dict[str, Any] = {
            "portfolio_var": round(portfolio_var_pure, 8),
            "portfolio_var_total": round(portfolio_var_pure + total_slippage, 8),
            "total_slippage": round(total_slippage, 8),
            "expected_shortfall": round(expected_shortfall, 8),
            "confidence": confidence,
            "undiversified_var": round(undiversified_var, 8),
            "diversification_ratio": round(div_ratio, 4),
            "per_asset_var": {a: round(v, 8) for a, v in per_asset_var.items()},
            "per_asset_slippage": {
                a: round(v, 8) for a, v in per_asset_slippage.items()
            },
            "per_asset_regime": {
                a: self.models[a].get_current_regime() for a in self.assets
            },
            "per_asset_regime_probs": {
                a: self.models[a].get_regime_probabilities().tolist()
                for a in self.assets
            },
            "copula": copula_info,
            "regime_conditional": (
                self.config.regime_conditional and self.regime_copulas is not None
            ),
            "n_simulations": n_sim,
            "weights": {a: round(w_normalized[i], 6) for i, a in enumerate(self.assets)},
        }

        logger.info(
            "portfolio_var.calculate_complete",
            portfolio_var=result["portfolio_var"],
            div_ratio=result["diversification_ratio"],
            copula=copula_info["family"],
            n_assets=d,
        )
        return result

    def regime_conditional_portfolio_var(
        self,
        weights: dict[str, float],
        confidence: float = 0.95,
        n_simulations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Per-regime portfolio VaR breakdown.

        Computes portfolio VaR conditioned on each aggregate regime,
        showing how portfolio risk changes across the regime spectrum
        (calm → crisis).

        Useful for stress testing: "What is our portfolio VaR if we
        enter crisis regime?"

        Args:
            weights: Asset weight dict.
            confidence: VaR confidence level.
            n_simulations: Simulation count per regime.
            seed: Random seed.

        Returns:
            Dict with per-regime portfolio VaR, aggregate probabilities,
            and regime-to-copula mapping.
        """
        if not self.is_fitted:
            raise RuntimeError("Portfolio model must be fitted first.")
        if not self.regime_copulas:
            raise RuntimeError(
                "Regime-conditional copulas not fitted. "
                "Set regime_conditional=True in config."
            )

        n_sim = n_simulations or self.config.n_simulations
        rng_seed = seed or self.config.seed
        K = self.alams_config.n_regimes
        d = len(self.assets)

        w = np.array([weights.get(a, 0.0) for a in self.assets])
        w_sum = w.sum()
        w_norm = w / w_sum if abs(w_sum) > 1e-10 else np.ones(d) / d

        alpha = 1.0 - confidence
        regime_results = []

        # Compute average regime probabilities
        avg_probs = np.zeros(K)
        for asset in self.assets:
            avg_probs += self.models[asset].get_regime_probabilities()
        avg_probs /= len(self.assets)

        for k in range(K):
            rng = np.random.RandomState(rng_seed + k)
            rc = self.regime_copulas[k]
            copula = rc["copula"]

            u_sim = self._sample_from_copula(copula, n_sim, d, rng)

            # Per-asset marginal: use regime k's sigma directly
            returns_sim = np.zeros((n_sim, d))
            for i, asset in enumerate(self.assets):
                model = self.models[asset]
                sigma_k = model.sigma[min(k, model.K - 1)]
                mu_k = model.mu[min(k, model.K - 1)]
                z_i = norm.ppf(np.clip(u_sim[:, i], 1e-10, 1 - 1e-10))
                returns_sim[:, i] = mu_k + sigma_k * z_i

            port_returns = returns_sim @ w_norm
            regime_var = float(-np.percentile(port_returns, alpha * 100))

            regime_results.append({
                "regime": k,
                "is_crisis": rc["is_crisis"],
                "copula_family": rc["family"],
                "probability": round(float(avg_probs[k]), 6),
                "portfolio_var": round(regime_var, 8),
                "tail_dependence": rc["copula"]["tail_dependence"],
            })

        # Probability-weighted VaR
        weighted_var = sum(
            rr["probability"] * rr["portfolio_var"] for rr in regime_results
        )

        return {
            "regime_breakdown": regime_results,
            "weighted_portfolio_var": round(weighted_var, 8),
            "confidence": confidence,
            "aggregate_regime_probs": avg_probs.tolist(),
            "dominant_regime": int(np.argmax(avg_probs)),
            "n_simulations_per_regime": n_sim,
        }

    # ============= SAMPLING HELPERS =============

    def _sample_from_copula(
        self,
        copula_fit: dict,
        n_samples: int,
        d: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Sample uniform variates from a fitted copula, deserializing params."""
        params = dict(copula_fit["params"])
        if "R" in params and isinstance(params["R"], list):
            params["R"] = np.array(params["R"])
        return _sample_copula(copula_fit["family"], params, n_samples, d, rng)

    def _sample_regime_mixture(
        self,
        n_samples: int,
        d: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Sample from regime-weighted copula mixture.

        Each regime k contributes samples proportional to its average
        current probability across assets. Crisis regimes produce
        heavier-tailed samples (Student-t), calm regimes lighter (Gaussian).
        """
        K = len(self.regime_copulas)
        avg_probs = np.zeros(K)
        for asset in self.assets:
            avg_probs += self.models[asset].get_regime_probabilities()
        avg_probs /= len(self.assets)

        all_samples = []
        for k in range(K):
            p_k = float(avg_probs[k])
            n_k = max(1, int(round(p_k * n_samples)))
            rc = self.regime_copulas[k]
            copula = rc["copula"]

            try:
                u_k = self._sample_from_copula(copula, n_k, d, rng)
            except Exception as exc:
                logger.warning(
                    "portfolio_var.regime_sample_failed",
                    regime=k,
                    error=str(exc),
                )
                R_fallback = np.eye(d)
                u_k = _sample_copula("gaussian", {"R": R_fallback}, n_k, d, rng)

            all_samples.append(u_k)

        combined = np.vstack(all_samples)
        rng.shuffle(combined)

        # Trim or pad to exact n_samples
        if len(combined) > n_samples:
            combined = combined[:n_samples]
        elif len(combined) < n_samples:
            extra = n_samples - len(combined)
            idx = rng.choice(len(combined), size=extra, replace=True)
            combined = np.vstack([combined, combined[idx]])

        return combined

    # ============= DIAGNOSTICS =============

    def summary(self) -> dict[str, Any]:
        """Return portfolio model summary."""
        if not self.is_fitted:
            return {"is_fitted": False, "assets": self.assets}

        per_asset_summary = {}
        for asset in self.assets:
            model = self.models[asset]
            per_asset_summary[asset] = {
                "n_obs": model.n_obs,
                "delta": round(model.delta, 4),
                "current_regime": model.get_current_regime(),
                "var_95": round(model.calculate_var(0.95), 6),
                "sigma": [round(s, 6) for s in model.sigma],
            }

        copula_summary = None
        if self.copula_fit:
            copula_summary = {
                "family": self.copula_fit["family"],
                "log_likelihood": round(self.copula_fit["log_likelihood"], 2),
                "aic": round(self.copula_fit["aic"], 2),
                "tail_dependence": self.copula_fit["tail_dependence"],
            }

        regime_info = None
        if self.regime_copulas:
            regime_info = [
                {
                    "regime": rc["regime"],
                    "is_crisis": rc["is_crisis"],
                    "family": rc["family"],
                    "tail_dep": rc["copula"]["tail_dependence"],
                }
                for rc in self.regime_copulas
            ]

        return {
            "is_fitted": True,
            "assets": self.assets,
            "n_assets": len(self.assets),
            "per_asset": per_asset_summary,
            "copula": copula_summary,
            "regime_copulas": regime_info,
            "config": dataclasses.asdict(self.config),
        }

    def get_correlation_matrix(self) -> dict[str, Any]:
        """
        Compute Kendall tau and Pearson correlation from the joint returns,
        plus copula-implied tail dependence.
        """
        if self._joint_returns is None:
            raise RuntimeError("No joint returns available. Call fit() first.")

        u = _to_uniform(self._joint_returns)
        tau_matrix = _kendall_tau_matrix(u)
        pearson_matrix = _tau_to_pearson(tau_matrix)

        return {
            "kendall_tau": {
                self.assets[i]: {
                    self.assets[j]: round(float(tau_matrix[i, j]), 4)
                    for j in range(len(self.assets))
                }
                for i in range(len(self.assets))
            },
            "pearson": {
                self.assets[i]: {
                    self.assets[j]: round(float(pearson_matrix[i, j]), 4)
                    for j in range(len(self.assets))
                }
                for i in range(len(self.assets))
            },
            "copula_tail_dependence": (
                self.copula_fit["tail_dependence"] if self.copula_fit else None
            ),
        }
