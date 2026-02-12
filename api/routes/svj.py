"""SVJ (Stochastic Volatility with Jumps) endpoints."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from api.models import (
    SVJCalibrateRequest,
    SVJCalibrateResponse,
    SVJClustering,
    SVJDiagnosticsResponse,
    SVJEVTTail,
    SVJHawkesParams,
    SVJJumpRiskResponse,
    SVJJumpStats,
    SVJMomentComparison,
    SVJParameterQuality,
    SVJVaRResponse,
)
from api.stores import _get_model, _svj_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["svj"])


def _svj_calibrate_sync(req: SVJCalibrateRequest) -> dict:
    """CPU-bound SVJ calibration — runs in thread pool."""
    from cortex.svj import calibrate_svj

    m = _get_model(req.token)
    cal = calibrate_svj(
        m["returns"],
        use_hawkes=req.use_hawkes,
        jump_threshold_multiplier=req.jump_threshold_multiplier,
    )

    _svj_store[req.token] = {"calibration": cal, "returns": m["returns"]}

    return {
        "token": req.token,
        "kappa": cal["kappa"],
        "theta": cal["theta"],
        "sigma": cal["sigma"],
        "rho": cal["rho"],
        "lambda_": cal["lambda_"],
        "mu_j": cal["mu_j"],
        "sigma_j": cal["sigma_j"],
        "feller_ratio": cal["feller_ratio"],
        "feller_satisfied": cal["feller_satisfied"],
        "log_likelihood": cal.get("log_likelihood"),
        "aic": cal.get("aic"),
        "bic": cal.get("bic"),
        "n_obs": cal["n_obs"],
        "n_jumps_detected": cal["n_jumps_detected"],
        "jump_fraction": cal["jump_fraction"],
        "bns_statistic": cal["bns_statistic"],
        "bns_pvalue": cal["bns_pvalue"],
        "optimization_success": cal["optimization_success"],
        "use_hawkes": cal["use_hawkes"],
        "hawkes_params": cal.get("hawkes_params"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/svj/calibrate")
async def svj_calibrate(req: SVJCalibrateRequest, async_mode: bool = Query(False)):
    from api.tasks import create_task, run_in_background

    if async_mode:
        task_id = create_task("/svj/calibrate")
        asyncio.ensure_future(run_in_background(task_id, _svj_calibrate_sync, req))
        return JSONResponse(content={
            "task_id": task_id,
            "status": "pending",
            "endpoint": "/svj/calibrate",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    result = await asyncio.to_thread(_svj_calibrate_sync, req)
    hp = SVJHawkesParams(**result["hawkes_params"]) if result.get("hawkes_params") else None
    return SVJCalibrateResponse(
        token=result["token"],
        kappa=result["kappa"],
        theta=result["theta"],
        sigma=result["sigma"],
        rho=result["rho"],
        lambda_=result["lambda_"],
        mu_j=result["mu_j"],
        sigma_j=result["sigma_j"],
        feller_ratio=result["feller_ratio"],
        feller_satisfied=result["feller_satisfied"],
        log_likelihood=result.get("log_likelihood"),
        aic=result.get("aic"),
        bic=result.get("bic"),
        n_obs=result["n_obs"],
        n_jumps_detected=result["n_jumps_detected"],
        jump_fraction=result["jump_fraction"],
        bns_statistic=result["bns_statistic"],
        bns_pvalue=result["bns_pvalue"],
        optimization_success=result["optimization_success"],
        use_hawkes=result["use_hawkes"],
        hawkes_params=hp,
        timestamp=result["timestamp"],
    )


@router.get("/svj/var", response_model=SVJVaRResponse)
def svj_var_endpoint(
    token: str = Query(...),
    alpha: float = Query(0.05, ge=0.001, le=0.5),
    n_simulations: int = Query(50000, ge=1000, le=500000),
):
    from cortex.svj import svj_var

    if token not in _svj_store:
        raise HTTPException(status_code=404, detail=f"SVJ not calibrated for {token}. POST /svj/calibrate first.")

    s = _svj_store[token]
    result = svj_var(s["returns"], s["calibration"], alpha=alpha, n_simulations=n_simulations)

    return SVJVaRResponse(token=token, **result, timestamp=datetime.now(timezone.utc))


@router.get("/svj/jump-risk", response_model=SVJJumpRiskResponse)
def svj_jump_risk(token: str = Query(...)):
    from cortex.svj import decompose_risk

    if token not in _svj_store:
        raise HTTPException(status_code=404, detail=f"SVJ not calibrated for {token}. POST /svj/calibrate first.")

    s = _svj_store[token]
    result = decompose_risk(s["returns"], s["calibration"])

    return SVJJumpRiskResponse(token=token, **result, timestamp=datetime.now(timezone.utc))


@router.get("/svj/diagnostics", response_model=SVJDiagnosticsResponse)
def svj_diagnostics_endpoint(token: str = Query(...)):
    from cortex.svj import svj_diagnostics

    if token not in _svj_store:
        raise HTTPException(status_code=404, detail=f"SVJ not calibrated for {token}. POST /svj/calibrate first.")

    s = _svj_store[token]
    result = svj_diagnostics(s["returns"], s["calibration"])

    js = SVJJumpStats(**result["jump_stats"])
    pq = SVJParameterQuality(**result["parameter_quality"])
    mc = SVJMomentComparison(**result["moment_comparison"])
    et = SVJEVTTail(**result["evt_tail"]) if result.get("evt_tail") else None
    cl = SVJClustering(**result["clustering"]) if result.get("clustering") else None

    return SVJDiagnosticsResponse(
        token=token,
        jump_stats=js,
        parameter_quality=pq,
        moment_comparison=mc,
        evt_tail=et,
        clustering=cl,
        timestamp=datetime.now(timezone.utc),
    )

