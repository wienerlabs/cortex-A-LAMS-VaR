"""Guardian (Unified Risk Veto) endpoints — assess, Kelly stats, circuit breakers, debate."""

import logging

from fastapi import APIRouter, HTTPException

from api.models import (
    CircuitBreakerItem,
    CircuitBreakersResponse,
    DebateResponse,
    GuardianAssessRequest,
    GuardianAssessResponse,
    GuardianComponentScore,
    KellyStatsResponse,
    TradeOutcomeRequest,
)
from api.stores import (
    _current_regime_state,
    _evt_store,
    _hawkes_store,
    _model_store,
    _svj_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["guardian"])


@router.post("/guardian/assess", response_model=GuardianAssessResponse)
def guardian_assess(req: GuardianAssessRequest):
    from cortex.guardian import _cache as guardian_cache
    from cortex.guardian import assess_trade

    if req.urgency:
        cache_key = f"{req.token}:{req.direction}"
        guardian_cache.pop(cache_key, None)

    model_data = _model_store.get(req.token)
    evt_data = _evt_store.get(req.token)
    svj_data = _svj_store.get(req.token)
    hawkes_data = _hawkes_store.get(req.token)

    if not any([model_data, evt_data, svj_data, hawkes_data]):
        raise HTTPException(
            status_code=404,
            detail=f"No calibrated models for '{req.token}'. "
            "Calibrate at least one model first (MSM, EVT, SVJ, or Hawkes).",
        )

    news_data = None
    try:
        from cortex.news import fetch_news_intelligence
        regime_state = _current_regime_state()
        news_result = fetch_news_intelligence(
            regime_state=regime_state, max_items=30, timeout=10.0,
        )
        news_data = news_result.get("signal")
    except Exception:
        logger.debug("News intelligence unavailable for guardian assessment", exc_info=True)

    result = assess_trade(
        token=req.token,
        trade_size_usd=req.trade_size_usd,
        direction=req.direction,
        model_data=model_data,
        evt_data=evt_data,
        svj_data=svj_data,
        hawkes_data=hawkes_data,
        news_data=news_data,
        strategy=req.strategy,
        run_debate=req.run_debate,
    )

    return GuardianAssessResponse(
        approved=result["approved"],
        risk_score=result["risk_score"],
        veto_reasons=result["veto_reasons"],
        recommended_size=result["recommended_size"],
        regime_state=result["regime_state"],
        confidence=result["confidence"],
        expires_at=result["expires_at"],
        component_scores=[
            GuardianComponentScore(**s) for s in result["component_scores"]
        ],
        circuit_breaker=result.get("circuit_breaker"),
        portfolio_limits=result.get("portfolio_limits"),
        debate=result.get("debate"),
        from_cache=result["from_cache"],
    )


@router.get("/guardian/kelly-stats", response_model=KellyStatsResponse)
def get_kelly_stats():
    from cortex.guardian import get_kelly_stats as _get_stats
    return KellyStatsResponse(**_get_stats())


@router.post("/guardian/trade-outcome")
def record_trade_outcome(req: TradeOutcomeRequest):
    from cortex.guardian import record_trade_outcome as _record
    _record(pnl=req.pnl, size=req.size, token=req.token)
    return {"status": "recorded", "pnl": req.pnl, "size": req.size, "token": req.token}


@router.get("/guardian/circuit-breakers")
def get_circuit_breakers():
    import time
    from cortex.circuit_breaker import get_all_states
    states = get_all_states()
    return {"breakers": states, "timestamp": time.time()}


@router.post("/guardian/circuit-breakers/reset")
def reset_circuit_breakers(name: str | None = None):
    from cortex.circuit_breaker import reset_all, reset_breaker
    if name:
        ok = reset_breaker(name)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Breaker '{name}' not found")
        return {"status": "reset", "breaker": name}
    reset_all()
    return {"status": "all_reset"}


@router.post("/guardian/debate")
def run_debate_endpoint(req: GuardianAssessRequest):
    """Run adversarial debate for a trade proposal (standalone, without full assess)."""
    from cortex.debate import run_debate

    result = run_debate(
        risk_score=50.0,
        component_scores=[],
        veto_reasons=[],
        direction=req.direction,
        trade_size_usd=req.trade_size_usd,
        original_approved=True,
        strategy=req.strategy or "spot",
    )
    return result


@router.post("/guardian/trade-outcome/strategy")
def record_strategy_trade_outcome(
    strategy: str,
    success: bool,
    pnl: float = 0.0,
    loss_type: str = "",
    details: str = "",
):
    """Record a trade outcome for strategy-specific circuit breakers.

    This feeds the outcome-based circuit breakers:
    - LP: 3 consecutive IL → pause
    - Arb: 5 consecutive failed executions → pause
    - Perp: 2 consecutive stop-losses → pause
    """
    from cortex.circuit_breaker import record_trade_outcome
    result = record_trade_outcome(
        strategy=strategy, success=success, pnl=pnl,
        loss_type=loss_type, details=details,
    )
    return result


@router.get("/guardian/circuit-breakers/outcomes")
def get_outcome_circuit_breakers():
    """Get status of outcome-based circuit breakers only."""
    import time as _time
    from cortex.circuit_breaker import get_outcome_states
    states = get_outcome_states()
    return {"outcome_breakers": states, "timestamp": _time.time()}


# ── Debate Transcript Store Endpoints ────────────────────────────────────────

@router.get("/guardian/debates/recent")
def get_recent_debates(limit: int = 20):
    """Get most recent debate transcripts from HOT tier."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return {"transcripts": store.get_recent(limit=limit), "count": min(limit, len(store._hot))}


@router.get("/guardian/debates/stats")
def get_debate_stats(hours: float = 24.0):
    """Get aggregate debate decision statistics over a time window."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return store.get_decision_stats(hours=hours)


@router.get("/guardian/debates/storage/stats")
def get_debate_storage_stats():
    """Get storage statistics across all tiers (HOT/WARM/COLD)."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return store.get_storage_stats()


@router.post("/guardian/debates/storage/rotate")
def force_debate_rotation():
    """Manually trigger cold rotation of old debate transcripts."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return store.force_rotation()


@router.get("/guardian/debates/by-strategy/{strategy}")
def get_debates_by_strategy(strategy: str, limit: int = 50):
    """Query debate transcripts by strategy."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    transcripts = store.get_by_strategy(strategy, limit=limit)
    return {"strategy": strategy, "transcripts": transcripts, "count": len(transcripts)}


@router.get("/guardian/debates/by-token/{token}")
def get_debates_by_token(token: str, limit: int = 50):
    """Query debate transcripts by token."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    transcripts = store.get_by_token(token, limit=limit)
    return {"token": token, "transcripts": transcripts, "count": len(transcripts)}


@router.get("/guardian/debates/{transcript_id}")
def get_debate_transcript(transcript_id: str):
    """Get a specific debate transcript by ID."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    transcript = store.get_by_id(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"Transcript '{transcript_id}' not found")
    return transcript

