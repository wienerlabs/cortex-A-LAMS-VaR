"""
Risk endpoints — A-LAMS-VaR model.

Provides calibration, VaR calculation, persistence, backtesting,
portfolio VaR, and unified model selection endpoints.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import structlog

from ...models.risk import ALAMSVaRModel, ALAMSConfig, LiquidityConfig, PortfolioALAMSVaR, PortfolioVaRConfig, RiskModelSelector

logger = structlog.get_logger()
router = APIRouter()

# Singleton model instance (lazy loaded on first fit or load)
_var_model: ALAMSVaRModel | None = None


def _get_or_create_model() -> ALAMSVaRModel:
    global _var_model
    if _var_model is None:
        _var_model = ALAMSVaRModel()
    return _var_model


# ========== Request / Response models ==========


class VaRFitRequest(BaseModel):
    """Request to fit (calibrate) the A-LAMS-VaR model."""

    returns: list[float] = Field(
        ..., min_length=100, description="Log returns array (min 100 observations)"
    )
    n_regimes: int = Field(default=5, ge=2, le=10)
    asymmetry_prior: float = Field(default=0.15, ge=0.0, le=0.5)
    max_iter: int = Field(default=200, ge=10, le=1000)
    token: str = Field(default="default", description="Persistence key for this model")


class VaRFitResponse(BaseModel):
    """Response after model calibration."""

    log_likelihood: float
    delta: float
    n_obs: int
    n_regimes: int
    aic: float
    bic: float
    stage1_success: bool
    stage2_success: bool
    persisted: bool = False


class VaRCalculateRequest(BaseModel):
    """Request to calculate VaR with optional liquidity adjustment."""

    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    trade_size_usd: float = Field(default=0.0, ge=0.0)
    pool_depth_usd: float = Field(default=1e9, gt=0.0)
    returns: Optional[list[float]] = Field(
        default=None, description="Optional: new returns to filter before calculation"
    )


class VaRCalculateResponse(BaseModel):
    """Response with VaR calculation results."""

    var_pure: float
    slippage_component: float
    var_total: float
    confidence: float
    current_regime: int
    regime_probs: list[float]
    delta: float
    regime_means: list[float]
    regime_sigmas: list[float]


class VaRLoadRequest(BaseModel):
    """Request to load a persisted model."""

    token: str = Field(default="default")


class VaRBacktestRequest(BaseModel):
    """Request to run VaR backtest."""

    returns: list[float] = Field(
        ..., min_length=120, description="Log returns array (min 120 for backtest)"
    )
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    min_window: int = Field(default=100, ge=30, le=1000)
    refit_every: int = Field(default=50, ge=10, le=500)


class VaRBacktestResponse(BaseModel):
    """Response with backtest results."""

    n_obs: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    confidence: float
    kupiec_pass: bool
    kupiec_pvalue: float
    kupiec_statistic: float
    christoffersen_pass: bool
    christoffersen_pvalue: float
    christoffersen_statistic: float
    cc_pass: bool
    cc_pvalue: float


# ========== Endpoints ==========


@router.post("/risk/var/fit", response_model=VaRFitResponse)
async def fit_var_model(request: VaRFitRequest) -> VaRFitResponse:
    """
    Calibrate the A-LAMS-VaR model on historical returns.

    Two-stage MLE:
    1. Regime means/variances and base transition matrix
    2. Asymmetry parameter δ

    Automatically persists the fitted model to the store.
    """
    global _var_model

    config = ALAMSConfig(
        n_regimes=request.n_regimes,
        asymmetry_prior=request.asymmetry_prior,
        max_iter=request.max_iter,
    )
    model = ALAMSVaRModel(config=config)

    try:
        returns = np.array(request.returns, dtype=np.float64)
        diagnostics = model.fit(returns)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("var_fit_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model fit failed: {e}")

    _var_model = model

    # Auto-persist
    persisted = False
    try:
        model.save_state(request.token)
        persisted = True
    except Exception as e:
        logger.warning("var_persist_failed", error=str(e))

    return VaRFitResponse(**diagnostics, persisted=persisted)


@router.post("/risk/var/calculate", response_model=VaRCalculateResponse)
async def calculate_var(request: VaRCalculateRequest) -> VaRCalculateResponse:
    """
    Calculate A-LAMS-VaR (regime-weighted VaR + liquidity adjustment).

    Requires model to be fitted first via /risk/var/fit or loaded via /risk/var/load.
    """
    model = _get_or_create_model()

    if not model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not fitted. Call /risk/var/fit or /risk/var/load first.",
        )

    try:
        returns_arr = (
            np.array(request.returns, dtype=np.float64)
            if request.returns is not None
            else None
        )

        result = model.calculate_liquidity_adjusted_var(
            confidence=request.confidence,
            trade_size_usd=request.trade_size_usd,
            pool_depth_usd=request.pool_depth_usd,
            returns=returns_arr,
        )
    except Exception as e:
        logger.error("var_calculate_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return VaRCalculateResponse(**result)


@router.get("/risk/var/summary")
async def var_summary() -> dict[str, Any]:
    """Get current A-LAMS-VaR model summary."""
    model = _get_or_create_model()
    return model.summary()


# ========== Persistence Endpoints ==========


@router.post("/risk/var/load")
async def load_var_model(request: VaRLoadRequest) -> dict[str, Any]:
    """Load a persisted A-LAMS-VaR model from the store."""
    global _var_model

    try:
        model = ALAMSVaRModel.load_state(request.token)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No persisted model found for token '{request.token}'",
        )
    except Exception as e:
        logger.error("var_load_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    _var_model = model
    return {
        "loaded": True,
        "token": request.token,
        "n_obs": model.n_obs,
        "n_regimes": model.K,
        "delta": model.delta,
    }


@router.get("/risk/var/models")
async def list_var_models() -> dict[str, Any]:
    """List all persisted A-LAMS-VaR model tokens."""
    tokens = ALAMSVaRModel.list_models()
    return {"tokens": tokens, "count": len(tokens)}


# ========== Backtest Endpoints ==========


@router.post("/risk/var/backtest", response_model=VaRBacktestResponse)
async def backtest_var_model(request: VaRBacktestRequest) -> VaRBacktestResponse:
    """
    Run rolling out-of-sample VaR backtest with Kupiec + Christoffersen validation.

    Fits the model on an expanding window and generates one-step-ahead VaR
    forecasts, then validates via standard statistical tests.
    """
    model = _get_or_create_model()

    try:
        returns = np.array(request.returns, dtype=np.float64)
        result = model.backtest(
            returns,
            confidence=request.confidence,
            min_window=request.min_window,
            refit_every=request.refit_every,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("var_backtest_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    return VaRBacktestResponse(
        n_obs=result["n_obs"],
        n_violations=result["n_violations"],
        violation_rate=result["violation_rate"],
        expected_rate=result["expected_rate"],
        confidence=result["confidence"],
        kupiec_pass=result["kupiec"]["pass"],
        kupiec_pvalue=result["kupiec"]["p_value"],
        kupiec_statistic=result["kupiec"]["statistic"],
        christoffersen_pass=result["christoffersen"]["pass"],
        christoffersen_pvalue=result["christoffersen"]["p_value"],
        christoffersen_statistic=result["christoffersen"]["statistic"],
        cc_pass=result["conditional_coverage"]["pass"],
        cc_pvalue=result["conditional_coverage"]["p_value"],
    )


# ========== Portfolio VaR Endpoints ==========

# Singleton portfolio model
_portfolio_model: PortfolioALAMSVaR | None = None


class PortfolioFitRequest(BaseModel):
    """Request to fit a multi-asset portfolio A-LAMS-VaR model."""

    assets: list[str] = Field(
        ..., min_length=2, description="Asset identifiers (e.g. ['SOL', 'BTC', 'ETH'])"
    )
    returns: dict[str, list[float]] = Field(
        ..., description="Per-asset log return series (all same length, min 100 each)"
    )
    copula_family: str = Field(
        default="student_t", description="Copula family or 'auto' for AIC selection"
    )
    regime_conditional: bool = Field(
        default=True, description="Fit regime-conditional copulas (Student-t for crisis, Gaussian for calm)"
    )
    n_regimes: int = Field(default=5, ge=2, le=10)


class PortfolioVaRRequest(BaseModel):
    """Request to calculate portfolio VaR."""

    weights: dict[str, float] = Field(
        ..., description="Per-asset weight dict (e.g. {'SOL': 0.5, 'BTC': 0.3, 'ETH': 0.2})"
    )
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    trade_sizes_usd: Optional[dict[str, float]] = Field(
        default=None, description="Per-asset planned trade sizes for slippage calc"
    )
    pool_depths_usd: Optional[dict[str, float]] = Field(
        default=None, description="Per-asset AMM pool depths"
    )
    n_simulations: int = Field(default=10_000, ge=1000, le=100_000)


@router.post("/risk/var/portfolio/fit")
async def fit_portfolio_var(request: PortfolioFitRequest) -> dict[str, Any]:
    """
    Fit multi-asset portfolio A-LAMS-VaR model.

    Phase 1: Fits per-asset A-LAMS models (5-regime Markov-switching with asymmetry).
    Phase 2: Fits copula on joint pseudo-uniform margins for tail dependence.
    Phase 2b: Optionally fits regime-conditional copulas (Student-t for crisis,
              Gaussian for calm) blended by current regime probabilities.
    """
    global _portfolio_model

    # Validate returns alignment
    lengths = {asset: len(rets) for asset, rets in request.returns.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"All return series must have the same length. Got: {lengths}",
        )

    for asset in request.assets:
        if asset not in request.returns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing return series for asset '{asset}'",
            )

    config = PortfolioVaRConfig(
        default_copula_family=request.copula_family if request.copula_family != "auto" else "student_t",
        auto_select_copula=request.copula_family == "auto",
        regime_conditional=request.regime_conditional,
    )
    alams_config = ALAMSConfig(n_regimes=request.n_regimes)

    portfolio = PortfolioALAMSVaR(
        assets=request.assets,
        config=config,
        alams_config=alams_config,
    )

    try:
        returns_dict = {
            asset: np.array(rets, dtype=np.float64)
            for asset, rets in request.returns.items()
        }
        result = portfolio.fit(returns_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("portfolio_var_fit_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio fit failed: {e}")

    _portfolio_model = portfolio
    return result


@router.post("/risk/var/portfolio/calculate")
async def calculate_portfolio_var_endpoint(request: PortfolioVaRRequest) -> dict[str, Any]:
    """
    Calculate portfolio A-LAMS-VaR with copula-based Monte Carlo simulation.

    Requires portfolio model to be fitted first via /risk/var/portfolio/fit.
    Combines per-asset regime-weighted marginals with copula tail dependence
    and per-asset AMM liquidity slippage.
    """
    if _portfolio_model is None or not _portfolio_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Portfolio model not fitted. Call /risk/var/portfolio/fit first.",
        )

    try:
        result = _portfolio_model.calculate_portfolio_var(
            weights=request.weights,
            confidence=request.confidence,
            trade_sizes_usd=request.trade_sizes_usd,
            pool_depths_usd=request.pool_depths_usd,
            n_simulations=request.n_simulations,
        )
    except Exception as e:
        logger.error("portfolio_var_calculate_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.get("/risk/var/portfolio/summary")
async def portfolio_var_summary() -> dict[str, Any]:
    """Get current portfolio model summary including per-asset regimes and copula info."""
    if _portfolio_model is None:
        return {"is_fitted": False, "assets": []}
    return _portfolio_model.summary()


@router.post("/risk/var/portfolio/regime-breakdown")
async def portfolio_regime_breakdown(request: PortfolioVaRRequest) -> dict[str, Any]:
    """
    Per-regime portfolio VaR breakdown for stress testing.

    Computes portfolio VaR conditioned on each regime (calm → crisis),
    showing how portfolio risk escalates across the regime spectrum.
    """
    if _portfolio_model is None or not _portfolio_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Portfolio model not fitted. Call /risk/var/portfolio/fit first.",
        )

    try:
        result = _portfolio_model.regime_conditional_portfolio_var(
            weights=request.weights,
            confidence=request.confidence,
            n_simulations=request.n_simulations,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("portfolio_regime_breakdown_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ========== Unified Model Selector Endpoints ==========

# Singleton model selector
_model_selector: RiskModelSelector | None = None


class UnifiedComputeRequest(BaseModel):
    """Request for unified VaR computation with automatic model selection."""

    returns: list[float] = Field(
        ..., min_length=100, description="Log returns for fitting/filtering"
    )
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    trade_size_usd: float = Field(default=0.0, ge=0.0)
    pool_depth_usd: float = Field(default=1e9, gt=0.0)
    n_regimes: int = Field(default=5, ge=2, le=10)
    force_model: Optional[str] = Field(
        default=None, description="Force 'alams' or 'msm', or None for auto-select"
    )


@router.post("/risk/var/compute")
async def unified_var_compute(request: UnifiedComputeRequest) -> dict[str, Any]:
    """
    Unified VaR computation with automatic MSM ↔ A-LAMS fallback.

    Tries A-LAMS-VaR first (primary, richer model). If it fails or is
    unavailable, falls back to MSM (simpler, more robust). Includes
    cross-validation when both models are available.

    Returns a unified result dict regardless of which model was used,
    plus metadata about the selection decision.
    """
    global _model_selector

    returns_arr = np.array(request.returns, dtype=np.float64)
    alams_config = ALAMSConfig(n_regimes=request.n_regimes)

    if _model_selector is None:
        _model_selector = RiskModelSelector(alams_config=alams_config)

    try:
        # Fit both models
        fit_result = _model_selector.fit_both(returns_arr)
    except Exception as e:
        logger.error("unified_compute_fit_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model fitting failed: {e}")

    # Override active model if forced
    if request.force_model == "alams":
        if not fit_result["alams"]["success"]:
            raise HTTPException(status_code=400, detail="A-LAMS fit failed; cannot force")
        _model_selector.active_model = "alams"
    elif request.force_model == "msm":
        if not fit_result["msm"]["success"]:
            raise HTTPException(status_code=400, detail="MSM fit failed; cannot force")
        _model_selector.active_model = "msm"

    try:
        result = _model_selector.select_and_compute(
            confidence=request.confidence,
            trade_size_usd=request.trade_size_usd,
            pool_depth_usd=request.pool_depth_usd,
        )
    except Exception as e:
        logger.error("unified_compute_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.get("/risk/var/compute/diagnostics")
async def unified_diagnostics() -> dict[str, Any]:
    """Get diagnostics for both A-LAMS and MSM models plus selection metadata."""
    if _model_selector is None:
        return {
            "active_model": "none",
            "alams": None,
            "msm": None,
            "cross_validation": None,
            "selection_reason": "No model selector initialized",
        }
    return _model_selector.get_diagnostics()
