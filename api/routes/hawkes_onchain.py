"""Hawkes on-chain calibration and risk API routes (Wave 10.3)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models import (
    HawkesOnchainCalibrateRequest,
    HawkesOnchainCalibrateResponse,
    HawkesOnchainRiskResponse,
    OnchainEventsResponse,
)
from cortex.data.onchain_events import (
    collect_events,
    events_to_hawkes_times,
    get_event_type_counts,
)
from cortex.data.onchain_liquidity import fetch_swap_history
from cortex.hawkes import (
    fit_multivariate_hawkes,
    flash_crash_risk_onchain,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Hawkes On-Chain"])

# In-memory store for multivariate Hawkes fits
_hawkes_onchain_store: dict[str, dict] = {}


@router.post("/hawkes/calibrate-onchain", response_model=HawkesOnchainCalibrateResponse)
async def calibrate_hawkes_onchain(req: HawkesOnchainCalibrateRequest):
    """Calibrate multivariate Hawkes process from on-chain events."""
    swaps = fetch_swap_history(req.token_address, limit=5000)
    events = collect_events(swaps)

    # Filter to requested event types
    filtered = [e for e in events if e["event_type"] in req.event_types]
    times_by_type = events_to_hawkes_times(filtered, req.event_types)

    if not times_by_type:
        raise HTTPException(404, "No qualifying events found for the given token and event types.")

    fit = fit_multivariate_hawkes(times_by_type)

    # Store for later risk queries
    _hawkes_onchain_store[req.token_address] = {
        "fit": fit,
        "times_by_type": {k: v.tolist() for k, v in times_by_type.items()},
    }

    return HawkesOnchainCalibrateResponse(
        token_address=req.token_address,
        event_types=fit["event_types"],
        n_events_per_type=fit["n_events_per_type"],
        mu=fit["mu"],
        cross_excitation=fit["cross_excitation"],
        branching_matrix=fit["branching_matrix"],
        spectral_radius=fit["spectral_radius"],
        stationary=fit["stationary"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/hawkes/events/{token_address}", response_model=OnchainEventsResponse)
async def get_onchain_events(token_address: str, limit: int = 200):
    """List on-chain events for a token."""
    swaps = fetch_swap_history(token_address, limit=limit * 3)
    events = collect_events(swaps)
    counts = get_event_type_counts(events)

    return OnchainEventsResponse(
        token_address=token_address,
        events=events[:limit],
        n_events=len(events),
        event_type_counts=counts,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/hawkes/onchain-risk/{token_address}", response_model=HawkesOnchainRiskResponse)
async def get_hawkes_onchain_risk(token_address: str):
    """Get flash crash risk score from on-chain event intensity."""
    stored = _hawkes_onchain_store.get(token_address)
    if stored is None:
        raise HTTPException(404, f"No calibration for '{token_address}'. POST /hawkes/calibrate-onchain first.")

    import numpy as np
    times_by_type = {k: np.array(v) for k, v in stored["times_by_type"].items()}
    fit = stored["fit"]

    risk = flash_crash_risk_onchain(times_by_type, fit)

    return HawkesOnchainRiskResponse(
        token_address=token_address,
        flash_crash_score=risk["flash_crash_score"],
        current_intensities=risk["current_intensities"],
        baseline_intensities=risk["baseline_intensities"],
        dominant_event_type=risk["dominant_event_type"],
        risk_level=risk["risk_level"],
        timestamp=datetime.now(timezone.utc),
    )

