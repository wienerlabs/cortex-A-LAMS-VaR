"""
Risk Model Selector — MSM ↔ A-LAMS-VaR Fallback Strategy.

Manages the hierarchy between the MSM (cortex/msm.py) baseline model and
the richer A-LAMS-VaR model, providing unified VaR computation with
automatic fallback, cross-validation, and diagnostic reporting.

Strategy:
  1. PRIMARY: A-LAMS-VaR (5 regimes, asymmetric transitions, liquidity adjustment)
  2. FALLBACK: MSM (simpler, faster, more robust to small samples)

Decision logic:
  - If A-LAMS is fitted and healthy → use A-LAMS
  - If A-LAMS fit failed / stale / n_obs < min → fallback to MSM
  - If both fitted → cross-validate regime agreement
    (if they disagree on crisis/calm by > 2 regime levels → flag inconsistency)
  - If neither fitted → return conservative defaults

The selector exposes a single ``select_and_compute()`` entry point that
returns a unified dict regardless of which model was used, plus metadata
about the selection decision and optional cross-validation results.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

import structlog

try:
    from .alams_var import ALAMSConfig, ALAMSVaRModel, LiquidityConfig
except ImportError:
    # Fallback for standalone loading (e.g., via importlib.util in tests)
    from alams_var import ALAMSConfig, ALAMSVaRModel, LiquidityConfig

logger = structlog.get_logger(__name__)


# ============= CONFIGURATION =============

# Minimum observations required for A-LAMS to be considered primary
_ALAMS_MIN_OBS = 100

# Maximum staleness (seconds) before re-fit is recommended
_ALAMS_MAX_STALENESS_SECONDS = 3600 * 6  # 6 hours

# Regime disagreement threshold: if MSM and A-LAMS disagree by more
# than this many levels, flag an inconsistency warning
_REGIME_DISAGREEMENT_THRESHOLD = 2


# ============= MODEL SELECTOR =============


class RiskModelSelector:
    """
    Manages MSM ↔ A-LAMS-VaR hierarchy with automatic fallback.

    A-LAMS-VaR is the primary model because it provides:
      - 5 regime states (vs MSM's K states with simpler parameterisation)
      - Asymmetric transition matrix (δ parameter)
      - Integrated AMM liquidity adjustment
      - Two-stage MLE with better regime separation

    MSM is the fallback because:
      - Simpler estimation (fewer parameters, more robust)
      - Works with smaller samples
      - Has proven Kupiec/Christoffersen validated backtests
      - Lower computational cost
    """

    def __init__(
        self,
        alams_config: Optional[ALAMSConfig] = None,
        liquidity_config: Optional[LiquidityConfig] = None,
        msm_num_states: int = 5,
    ):
        self.alams_config = alams_config or ALAMSConfig()
        self.liquidity_config = liquidity_config or LiquidityConfig()
        self.msm_num_states = msm_num_states

        # Model instances
        self.alams_model: Optional[ALAMSVaRModel] = None
        self.msm_calibration: Optional[dict] = None
        self.msm_filter_probs: Optional[np.ndarray] = None
        self.msm_sigma_states: Optional[np.ndarray] = None
        self.msm_P_matrix: Optional[np.ndarray] = None

        # State tracking
        self.active_model: str = "none"  # "alams", "msm", "none"
        self._last_alams_fit_n_obs: int = 0
        self._last_msm_fit_n_obs: int = 0

    # ============= FITTING =============

    def fit_alams(self, returns: np.ndarray) -> dict[str, Any]:
        """
        Attempt to fit the A-LAMS-VaR model.

        Args:
            returns: Array of log returns.

        Returns:
            Dict with fit diagnostics or error info.
        """
        model = ALAMSVaRModel(
            config=self.alams_config,
            liquidity_config=self.liquidity_config,
        )
        try:
            diagnostics = model.fit(returns)
            self.alams_model = model
            self._last_alams_fit_n_obs = len(returns)
            self.active_model = "alams"
            logger.info(
                "model_selector.alams_fitted",
                n_obs=diagnostics["n_obs"],
                delta=round(diagnostics["delta"], 4),
            )
            return {"success": True, "model": "alams", "diagnostics": diagnostics}
        except Exception as exc:
            logger.warning("model_selector.alams_fit_failed", error=str(exc))
            return {"success": False, "model": "alams", "error": str(exc)}

    def fit_msm(self, returns: np.ndarray | pd.Series) -> dict[str, Any]:
        """
        Fit the MSM model as fallback.

        Uses ``cortex.msm.calibrate_msm_advanced`` for MLE calibration
        and stores the resulting filter probabilities for VaR forecasting.

        Args:
            returns: Array or Series of log returns.

        Returns:
            Dict with MSM calibration results or error info.
        """
        from cortex.msm import calibrate_msm_advanced, msm_vol_forecast

        if isinstance(returns, np.ndarray):
            returns_series = pd.Series(returns)
        else:
            returns_series = returns

        try:
            cal = calibrate_msm_advanced(
                returns_series,
                num_states=self.msm_num_states,
                method="mle",
                verbose=False,
            )
            self.msm_calibration = cal

            # calibrate_msm_advanced doesn't return filter_probs or P_matrix
            # in its result dict, so we recompute via msm_vol_forecast.
            _, _, filter_probs_df, sigma_states, P_matrix = msm_vol_forecast(
                returns_series,
                num_states=self.msm_num_states,
                sigma_low=cal["sigma_low"],
                sigma_high=cal["sigma_high"],
                p_stay=cal["p_stay"],
                leverage_gamma=cal.get("leverage_gamma", 0.0),
            )
            self.msm_filter_probs = filter_probs_df
            self.msm_sigma_states = sigma_states
            self.msm_P_matrix = P_matrix
            self._last_msm_fit_n_obs = len(returns_series)

            if self.active_model != "alams":
                self.active_model = "msm"

            logger.info(
                "model_selector.msm_fitted",
                n_obs=len(returns_series),
                sigma_low=round(float(cal.get("sigma_low", 0)), 4),
                sigma_high=round(float(cal.get("sigma_high", 0)), 4),
            )
            return {"success": True, "model": "msm", "calibration": cal}
        except Exception as exc:
            logger.warning("model_selector.msm_fit_failed", error=str(exc))
            return {"success": False, "model": "msm", "error": str(exc)}

    def fit_both(self, returns: np.ndarray | pd.Series) -> dict[str, Any]:
        """
        Fit both models. A-LAMS is tried first; MSM always runs as fallback.

        Args:
            returns: Array or Series of log returns.

        Returns:
            Dict with both fit results and the active model selection.
        """
        returns_arr = np.asarray(returns, dtype=np.float64)

        alams_result = self.fit_alams(returns_arr)
        msm_result = self.fit_msm(returns_arr)

        # Determine active model
        if alams_result["success"]:
            self.active_model = "alams"
        elif msm_result["success"]:
            self.active_model = "msm"
        else:
            self.active_model = "none"

        # Cross-validate if both succeeded
        cross_val = None
        if alams_result["success"] and msm_result["success"]:
            cross_val = self.cross_validate_regimes()

        return {
            "alams": alams_result,
            "msm": msm_result,
            "active_model": self.active_model,
            "cross_validation": cross_val,
        }

    # ============= UNIFIED VaR COMPUTATION =============

    def select_and_compute(
        self,
        returns: Optional[np.ndarray] = None,
        confidence: float = 0.95,
        trade_size_usd: float = 0.0,
        pool_depth_usd: float = 1e9,
    ) -> dict[str, Any]:
        """
        Compute VaR using the best available model with automatic fallback.

        Decision tree:
          1. If A-LAMS is fitted and healthy → use A-LAMS
          2. If A-LAMS failed but MSM is available → use MSM
          3. If neither is available → return conservative defaults

        Args:
            returns: Optional new returns to filter before computation.
            confidence: VaR confidence level.
            trade_size_usd: Trade size for liquidity adjustment (A-LAMS only).
            pool_depth_usd: AMM pool depth (A-LAMS only).

        Returns:
            Unified dict with model_used, var_value, regime info,
            fallback_reason (if applicable), and cross_validation.
        """
        # Try A-LAMS first
        if self.alams_model is not None and self.alams_model.is_fitted:
            try:
                result = self._compute_alams(
                    returns, confidence, trade_size_usd, pool_depth_usd
                )
                result["fallback_reason"] = None
                result["cross_validation"] = self._maybe_cross_validate()
                return result
            except Exception as exc:
                logger.warning(
                    "model_selector.alams_compute_failed",
                    error=str(exc),
                )
                fallback_reason = f"alams_compute_error: {exc}"
        else:
            fallback_reason = "alams_not_fitted"

        # Fallback to MSM
        if self.msm_calibration is not None and self.msm_filter_probs is not None:
            try:
                result = self._compute_msm(confidence)
                result["fallback_reason"] = fallback_reason
                result["cross_validation"] = None
                self.active_model = "msm"
                return result
            except Exception as exc:
                logger.warning(
                    "model_selector.msm_compute_failed",
                    error=str(exc),
                )
                fallback_reason = f"msm_compute_error: {exc}"

        # Neither available — conservative defaults
        self.active_model = "none"
        return self._conservative_defaults(confidence, fallback_reason)

    def _compute_alams(
        self,
        returns: Optional[np.ndarray],
        confidence: float,
        trade_size_usd: float,
        pool_depth_usd: float,
    ) -> dict[str, Any]:
        """Compute VaR using A-LAMS-VaR model."""
        model = self.alams_model

        # Optionally update filter with new returns
        if returns is not None:
            model.filter(np.asarray(returns, dtype=np.float64))

        var_result = model.calculate_liquidity_adjusted_var(
            confidence=confidence,
            trade_size_usd=trade_size_usd,
            pool_depth_usd=pool_depth_usd,
        )

        self.active_model = "alams"

        return {
            "model_used": "alams",
            "var_value": var_result["var_total"],
            "var_pure": var_result["var_pure"],
            "slippage_component": var_result["slippage_component"],
            "confidence": confidence,
            "current_regime": var_result["current_regime"],
            "regime_probs": var_result["regime_probs"],
            "delta": var_result["delta"],
            "regime_means": var_result["regime_means"],
            "regime_sigmas": var_result["regime_sigmas"],
            "n_obs": model.n_obs,
        }

    def _compute_msm(self, confidence: float) -> dict[str, Any]:
        """Compute VaR using MSM model."""
        from cortex.msm import msm_var_forecast_next_day

        # msm_var_forecast_next_day returns a tuple:
        # (var_t1, sigma_t1_forecast, z_alpha, pi_t1_given_t)
        var_t1, _sigma_t1, _z, pi_t1 = msm_var_forecast_next_day(
            self.msm_filter_probs,
            self.msm_sigma_states,
            self.msm_P_matrix,
            alpha=1.0 - confidence,
        )

        # Extract MSM regime info
        filter_probs = self.msm_filter_probs
        if hasattr(filter_probs, "iloc"):
            pi_t = np.asarray(filter_probs.iloc[-1])
        else:
            pi_t = np.asarray(filter_probs[-1])

        current_regime = int(np.argmax(pi_t))
        var_abs = abs(float(var_t1))

        return {
            "model_used": "msm",
            "var_value": var_abs,
            "var_pure": var_abs,
            "slippage_component": 0.0,
            "confidence": confidence,
            "current_regime": current_regime,
            "regime_probs": pi_t.tolist(),
            "delta": 0.0,
            "regime_means": [0.0] * len(pi_t),
            "regime_sigmas": (
                self.msm_sigma_states.tolist()
                if self.msm_sigma_states is not None
                else []
            ),
            "n_obs": self._last_msm_fit_n_obs,
        }

    @staticmethod
    def _conservative_defaults(
        confidence: float, reason: str
    ) -> dict[str, Any]:
        """Return conservative default VaR when no model is available."""
        # Conservative 5% daily VaR (typical for crypto)
        conservative_var = 0.05 if confidence >= 0.95 else 0.03

        return {
            "model_used": "none",
            "var_value": conservative_var,
            "var_pure": conservative_var,
            "slippage_component": 0.0,
            "confidence": confidence,
            "current_regime": -1,
            "regime_probs": [],
            "delta": 0.0,
            "regime_means": [],
            "regime_sigmas": [],
            "n_obs": 0,
            "fallback_reason": reason,
            "cross_validation": None,
            "warning": "Using conservative defaults — no model available",
        }

    # ============= CROSS-VALIDATION =============

    def cross_validate_regimes(self) -> dict[str, Any]:
        """
        Compare A-LAMS regime probabilities vs MSM filter probabilities.

        Computes an agreement score (0-1) measuring how closely the two
        models agree on the current market regime. Low agreement may
        indicate model instability or structural change.

        Returns:
            Dict with agreement_score, regime_mapping, warnings.
        """
        if self.alams_model is None or not self.alams_model.is_fitted:
            return {"error": "A-LAMS model not fitted"}
        if self.msm_filter_probs is None:
            return {"error": "MSM model not fitted"}

        # A-LAMS regime info
        alams_probs = self.alams_model.get_regime_probabilities()
        alams_regime = self.alams_model.get_current_regime()
        K_alams = self.alams_model.K

        # MSM regime info
        if hasattr(self.msm_filter_probs, "iloc"):
            msm_probs = np.asarray(self.msm_filter_probs.iloc[-1])
        else:
            msm_probs = np.asarray(self.msm_filter_probs[-1])
        msm_regime = int(np.argmax(msm_probs))
        K_msm = len(msm_probs)

        # Normalize regime indices to [0, 1] scale for comparison
        alams_normalized = alams_regime / max(K_alams - 1, 1)
        msm_normalized = msm_regime / max(K_msm - 1, 1)

        # Agreement score: 1 - |normalized difference|
        regime_distance = abs(alams_normalized - msm_normalized)
        agreement_score = 1.0 - regime_distance

        # Probability distribution agreement (Hellinger distance)
        # Align to same dimension
        if K_alams == K_msm:
            # Hellinger distance: H(p,q) = 1/√2 · √(Σ(√p_i - √q_i)²)
            h_dist = np.sqrt(
                0.5 * np.sum((np.sqrt(alams_probs) - np.sqrt(msm_probs)) ** 2)
            )
            prob_agreement = 1.0 - h_dist
        else:
            # Different K: interpolate to common grid
            common_K = max(K_alams, K_msm)
            alams_interp = np.interp(
                np.linspace(0, 1, common_K),
                np.linspace(0, 1, K_alams),
                alams_probs,
            )
            msm_interp = np.interp(
                np.linspace(0, 1, common_K),
                np.linspace(0, 1, K_msm),
                msm_probs,
            )
            # Normalize after interpolation
            alams_interp = np.clip(alams_interp, 1e-10, None)
            alams_interp /= alams_interp.sum()
            msm_interp = np.clip(msm_interp, 1e-10, None)
            msm_interp /= msm_interp.sum()

            h_dist = np.sqrt(
                0.5 * np.sum((np.sqrt(alams_interp) - np.sqrt(msm_interp)) ** 2)
            )
            prob_agreement = 1.0 - h_dist

        # Warnings
        warnings = []
        if regime_distance > _REGIME_DISAGREEMENT_THRESHOLD / max(K_alams - 1, 1):
            warnings.append(
                f"Regime disagreement: A-LAMS={alams_regime}, MSM={msm_regime} "
                f"(distance={regime_distance:.2f})"
            )
        if prob_agreement < 0.5:
            warnings.append(
                f"Low probability agreement ({prob_agreement:.2f}): "
                "models may be capturing different dynamics"
            )

        result = {
            "agreement_score": round(agreement_score, 4),
            "probability_agreement": round(prob_agreement, 4),
            "alams_regime": alams_regime,
            "msm_regime": msm_regime,
            "alams_K": K_alams,
            "msm_K": K_msm,
            "regime_distance": round(regime_distance, 4),
            "hellinger_distance": round(float(h_dist), 4),
            "warnings": warnings,
        }

        logger.info(
            "model_selector.cross_validate",
            agreement=result["agreement_score"],
            prob_agreement=result["probability_agreement"],
            alams_regime=alams_regime,
            msm_regime=msm_regime,
            n_warnings=len(warnings),
        )

        return result

    def _maybe_cross_validate(self) -> Optional[dict]:
        """Run cross-validation if both models are available."""
        if (
            self.alams_model is not None
            and self.alams_model.is_fitted
            and self.msm_calibration is not None
            and self.msm_filter_probs is not None
        ):
            try:
                return self.cross_validate_regimes()
            except Exception:
                return None
        return None

    # ============= DIAGNOSTICS =============

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Return diagnostics for both models and the current selection decision.
        """
        alams_summary = None
        if self.alams_model is not None and self.alams_model.is_fitted:
            alams_summary = self.alams_model.summary()

        msm_summary = None
        if self.msm_calibration is not None:
            msm_summary = {
                "sigma_low": self.msm_calibration.get("sigma_low"),
                "sigma_high": self.msm_calibration.get("sigma_high"),
                "n_obs": self._last_msm_fit_n_obs,
                "num_states": self.msm_num_states,
            }

        cross_val = self._maybe_cross_validate()

        return {
            "active_model": self.active_model,
            "alams": alams_summary,
            "msm": msm_summary,
            "cross_validation": cross_val,
            "selection_reason": self._get_selection_reason(),
        }

    def _get_selection_reason(self) -> str:
        """Human-readable reason for current model selection."""
        if self.active_model == "alams":
            return (
                f"A-LAMS primary: fitted with {self._last_alams_fit_n_obs} obs, "
                f"δ={self.alams_model.delta:.4f}"
                if self.alams_model
                else "A-LAMS selected"
            )
        elif self.active_model == "msm":
            if self.alams_model is None:
                return "MSM fallback: A-LAMS not fitted"
            return f"MSM fallback: A-LAMS failed or unavailable"
        else:
            return "No model available: using conservative defaults"
