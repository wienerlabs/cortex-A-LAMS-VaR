import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

# Add project root to path so we can import the MSM model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.models import (
    BacktestSummaryResponse,
    CalibrateRequest,
    CalibrateResponse,
    CalibrationMetrics,
    ErrorResponse,
    RegimeResponse,
    RegimeStreamMessage,
    TailProbResponse,
    VaRResponse,
    VolatilityForecastResponse,
    get_regime_name,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory model state (per-token)
_model_store: dict[str, dict] = {}


def _get_model(token: str) -> dict:
    if token not in _model_store:
        raise HTTPException(
            status_code=404,
            detail=f"No calibrated model for '{token}'. Call POST /calibrate first.",
        )
    return _model_store[token]


def _load_returns(req: CalibrateRequest) -> pd.Series:
    """Fetch data and convert to log-returns in %."""
    if req.data_source.value == "solana":
        from solana_data_adapter import get_token_ohlcv, ohlcv_to_returns

        df = get_token_ohlcv(req.token, req.start_date, req.end_date, req.interval)
        return ohlcv_to_returns(df)

    import yfinance as yf

    df = yf.download(req.token, start=req.start_date, end=req.end_date, progress=False)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No yfinance data for '{req.token}'")
    close = df["Close"].dropna()
    rets = 100 * np.diff(np.log(close.values))
    return pd.Series(rets, index=close.index[1:], name="r")


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest):
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")

    try:
        returns = _load_returns(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if len(returns) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 data points")

    cal = msm.calibrate_msm_advanced(
        returns,
        num_states=req.num_states,
        method=req.method.value,
        target_var_breach=req.target_var_breach,
        verbose=False,
    )

    sigma_f, sigma_filt, fprobs, sigma_states, P = msm.msm_vol_forecast(
        returns,
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
    )

    _model_store[req.token] = {
        "calibration": cal,
        "returns": returns,
        "sigma_forecast": sigma_f,
        "sigma_filtered": sigma_filt,
        "filter_probs": fprobs,
        "sigma_states": sigma_states,
        "P_matrix": P,
        "calibrated_at": datetime.now(timezone.utc),
    }

    return CalibrateResponse(
        token=req.token,
        method=cal["method"],
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
        sigma_states=cal["sigma_states"].tolist(),
        metrics=CalibrationMetrics(**cal["metrics"]),
        calibrated_at=_model_store[req.token]["calibrated_at"],
    )


@router.get("/regime/current", response_model=RegimeResponse)
def get_current_regime(token: str = Query(...)):
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
    m = _get_model(token)

    probs = np.asarray(m["filter_probs"].iloc[-1])
    state_idx = int(np.argmax(probs)) + 1
    num_states = m["calibration"]["num_states"]

    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05
    )

    return RegimeResponse(
        timestamp=datetime.now(timezone.utc),
        regime_state=state_idx,
        regime_name=get_regime_name(state_idx, num_states),
        regime_probabilities=probs.tolist(),
        volatility_filtered=float(m["sigma_filtered"].iloc[-1]),
        volatility_forecast=sigma_t1,
        var_95=var_t1,
        transition_matrix=m["P_matrix"].tolist(),
    )




@router.get("/var/{confidence}", response_model=VaRResponse)
def get_var(confidence: float, token: str = Query(...)):
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
    m = _get_model(token)

    alpha = 1.0 - confidence if confidence > 0.5 else confidence
    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=alpha
    )

    return VaRResponse(
        timestamp=datetime.now(timezone.utc),
        confidence=confidence,
        var_value=var_t1,
        sigma_forecast=sigma_t1,
        z_alpha=z_alpha,
        regime_probabilities=pi_t1.tolist(),
    )


@router.get("/volatility/forecast", response_model=VolatilityForecastResponse)
def get_volatility_forecast(token: str = Query(...)):
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
    m = _get_model(token)

    _, sigma_t1, _, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"]
    )

    return VolatilityForecastResponse(
        timestamp=datetime.now(timezone.utc),
        sigma_forecast=sigma_t1,
        sigma_filtered=float(m["sigma_filtered"].iloc[-1]),
        regime_probabilities=pi_t1.tolist(),
        sigma_states=m["sigma_states"].tolist(),
    )


@router.get("/backtest/summary", response_model=BacktestSummaryResponse)
def get_backtest_summary(token: str = Query(...), alpha: float = Query(0.05)):
    from importlib import import_module
    from scipy.stats import norm

    msm = import_module("MSM-VaR_MODEL")
    m = _get_model(token)

    returns = m["returns"]
    sigma_forecast = m["sigma_forecast"]
    z = norm.ppf(alpha)
    var_series = z * sigma_forecast
    breaches = (returns < var_series).astype(int)

    kupiec_lr, kupiec_p, x, n = msm.kupiec_test(breaches, alpha=alpha)
    chris_lr, chris_p, _ = msm.christoffersen_independence_test(breaches)

    return BacktestSummaryResponse(
        token=token,
        num_observations=int(n),
        var_alpha=alpha,
        breach_count=int(x),
        breach_rate=float(x / n) if n > 0 else 0.0,
        kupiec_lr=None if np.isnan(kupiec_lr) else float(kupiec_lr),
        kupiec_pvalue=None if np.isnan(kupiec_p) else float(kupiec_p),
        kupiec_pass=bool(kupiec_p > 0.05) if not np.isnan(kupiec_p) else False,
        christoffersen_lr=None if np.isnan(chris_lr) else float(chris_lr),
        christoffersen_pvalue=None if np.isnan(chris_p) else float(chris_p),
        christoffersen_pass=bool(chris_p > 0.05) if not np.isnan(chris_p) else False,
    )


@router.get("/tail-probs", response_model=TailProbResponse)
def get_tail_probs(
    token: str = Query(...),
    alpha: float = Query(0.05),
    use_student_t: bool = Query(False),
    nu: float = Query(5.0),
):
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
    m = _get_model(token)

    result = msm.msm_tail_probs(
        m["returns"],
        m["filter_probs"],
        m["sigma_states"],
        alpha=alpha,
        horizons=(1, 3, 5),
        use_student_t=use_student_t,
        nu=nu,
    )

    return TailProbResponse(
        l1_threshold=result["L1"],
        p1_day=result["p1"],
        horizon_probs={int(k): v for k, v in result["horizon_probs"].items()},
        distribution=result["distribution"],
    )


@router.websocket("/stream/regime")
async def stream_regime(ws: WebSocket, token: str = Query(...)):
    """Stream regime updates every 5 seconds for a calibrated token."""
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
    await ws.accept()

    try:
        while True:
            if token not in _model_store:
                await ws.send_json({"error": f"No model for '{token}'"})
                await asyncio.sleep(5)
                continue

            m = _model_store[token]
            probs = np.asarray(m["filter_probs"].iloc[-1])
            state_idx = int(np.argmax(probs)) + 1
            num_states = m["calibration"]["num_states"]

            var_t1, sigma_t1, _, _ = msm.msm_var_forecast_next_day(
                m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05
            )

            msg = RegimeStreamMessage(
                timestamp=datetime.now(timezone.utc),
                regime_state=state_idx,
                regime_name=get_regime_name(state_idx, num_states),
                regime_probabilities=probs.tolist(),
                volatility_forecast=sigma_t1,
                var_95=var_t1,
            )
            await ws.send_json(msg.model_dump(mode="json"))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for token=%s", token)
    except Exception as exc:
        logger.exception("WebSocket error for token=%s", token)
        await ws.close(code=1011, reason=str(exc)[:120])