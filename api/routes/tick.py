"""Tick-level data and backtesting API routes (Wave 10.2)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models import (
    BacktestRequest,
    BacktestResponse,
    TickDataRequest,
    TickDataResponse,
)
from api.stores import _model_store
from cortex.backtesting import backtest_multi_horizon, simple_var_forecast
from cortex.data.onchain_liquidity import fetch_swap_history
from cortex.data.tick_data import (
    aggregate_imbalance_bars,
    aggregate_tick_bars,
    aggregate_time_bars,
    aggregate_volume_bars,
    bars_to_returns,
    reconstruct_tick_prices,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tick Data & Backtesting"])

_BAR_AGGREGATORS = {
    "time": lambda ticks, size, limit: aggregate_time_bars(ticks, bar_seconds=size, max_bars=limit),
    "volume": lambda ticks, size, limit: aggregate_volume_bars(ticks, bar_volume=float(size), max_bars=limit),
    "tick": lambda ticks, size, limit: aggregate_tick_bars(ticks, ticks_per_bar=size, max_bars=limit),
    "imbalance": lambda ticks, size, limit: aggregate_imbalance_bars(ticks, threshold=float(size), max_bars=limit),
}


@router.post("/ticks/aggregate", response_model=TickDataResponse)
async def aggregate_tick_data(req: TickDataRequest):
    """Fetch on-chain swap history and aggregate into bars."""
    swaps = fetch_swap_history(req.token_address, limit=req.limit * 5)
    ticks = reconstruct_tick_prices(swaps)

    aggregator = _BAR_AGGREGATORS.get(req.bar_type)
    if aggregator is None:
        raise HTTPException(400, f"Unknown bar_type: {req.bar_type}. Use: time|volume|tick|imbalance")

    bars = aggregator(ticks, req.bar_size, req.limit)

    return TickDataResponse(
        token_address=req.token_address,
        bar_type=req.bar_type,
        bar_size=req.bar_size,
        n_bars=len(bars),
        bars=bars,
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/backtest/tick-level", response_model=BacktestResponse)
async def run_tick_backtest(req: BacktestRequest):
    """Run multi-frequency VaR backtest using tick-level data."""
    if req.token not in _model_store:
        raise HTTPException(404, f"Token '{req.token}' not calibrated. POST /msm/calibrate first.")

    # Fetch swap history and build bars at each horizon
    swaps = []
    if req.token_address:
        swaps = fetch_swap_history(req.token_address, limit=10000)

    ticks = reconstruct_tick_prices(swaps)

    bars_by_horizon: dict[int, list[dict]] = {}
    for h_min in req.horizons:
        bars = aggregate_time_bars(ticks, bar_seconds=h_min * 60, max_bars=5000)
        bars_by_horizon[h_min] = bars

    horizon_results = backtest_multi_horizon(
        bars_by_horizon, simple_var_forecast, confidence=req.confidence
    )

    overall_pass = all(h.get("kupiec_pass", True) for h in horizon_results)

    return BacktestResponse(
        token=req.token,
        confidence=req.confidence,
        horizons=horizon_results,
        overall_pass=overall_pass,
        timestamp=datetime.now(timezone.utc),
    )

