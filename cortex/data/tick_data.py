"""Tick-level data reconstruction and bar aggregation from DEX swap transactions.

Reconstructs trade-by-trade price series from Raydium/Orca/Meteora swaps,
then aggregates into various bar types:
  - Time bars (fixed interval: 1m, 5m, 1h, etc.)
  - Volume bars (fixed volume per bar)
  - Tick bars (fixed number of trades per bar)
  - Imbalance bars (trade imbalance threshold)

References:
  - Lopez de Prado (2018) "Advances in Financial Machine Learning" ch. 2
"""
from __future__ import annotations

import numpy as np

from cortex.config import TICK_MAX_BARS


def reconstruct_tick_prices(swaps: list[dict]) -> list[dict]:
    """Convert parsed swap records into a tick-level price series.

    Each tick has: timestamp, price, volume, dex, direction (buy/sell).
    Swaps are sorted by slot (ascending).
    """
    if not swaps:
        return []

    sorted_swaps = sorted(swaps, key=lambda s: (s.get("slot", 0), s.get("block_time", 0)))
    ticks: list[dict] = []

    for s in sorted_swaps:
        price = s.get("price", 0.0)
        if price <= 0:
            continue

        volume = s.get("amount_in", 0.0)
        direction = "sell" if s.get("token_in", "") == "SOL" else "buy"

        ticks.append({
            "timestamp": float(s.get("block_time", 0)),
            "slot": s.get("slot", 0),
            "price": float(price),
            "volume": float(volume),
            "dex": s.get("dex", "unknown"),
            "direction": direction,
            "signature": s.get("signature", ""),
        })

    return ticks


def _make_bar(bar_start: float, o: float, h: float, l: float, c: float,
              vol: float, vwap_num: float, n_ticks: int, **extra) -> dict:
    vwap = vwap_num / vol if vol > 0 else c
    bar = {
        "timestamp": bar_start, "open": o, "high": h, "low": l, "close": c,
        "volume": vol, "n_ticks": n_ticks, "vwap": vwap,
    }
    bar.update(extra)
    return bar


def aggregate_time_bars(
    ticks: list[dict], bar_seconds: int = 300, max_bars: int = TICK_MAX_BARS
) -> list[dict]:
    """Aggregate ticks into fixed-time OHLCV bars."""
    if not ticks:
        return []

    bars: list[dict] = []
    bar_start = ticks[0]["timestamp"]
    bar_end = bar_start + bar_seconds
    o = h = l = c = ticks[0]["price"]
    vol = vwap_num = 0.0
    n = 0

    for t in ticks:
        ts, p, v = t["timestamp"], t["price"], t["volume"]

        while ts >= bar_end and n > 0:
            bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n))
            if len(bars) >= max_bars:
                return bars
            bar_start = bar_end
            bar_end = bar_start + bar_seconds
            o = h = l = c = p
            vol = vwap_num = 0.0
            n = 0

        if n == 0:
            o = h = l = p
        h, l, c = max(h, p), min(l, p), p
        vol += v
        vwap_num += p * v
        n += 1

    if n > 0:
        bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n))

    return bars[:max_bars]


def aggregate_volume_bars(
    ticks: list[dict], bar_volume: float = 100.0, max_bars: int = TICK_MAX_BARS
) -> list[dict]:
    """Aggregate ticks into fixed-volume bars (Lopez de Prado ch. 2)."""
    if not ticks:
        return []

    bars: list[dict] = []
    o = h = l = c = ticks[0]["price"]
    vol = vwap_num = 0.0
    n = 0
    bar_start = ticks[0]["timestamp"]

    for t in ticks:
        p, v = t["price"], t["volume"]
        if n == 0:
            bar_start = t["timestamp"]
            o = h = l = p
        h, l, c = max(h, p), min(l, p), p
        vol += v
        vwap_num += p * v
        n += 1

        if vol >= bar_volume:
            bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n))
            if len(bars) >= max_bars:
                return bars
            o = h = l = c = p
            vol = vwap_num = 0.0
            n = 0

    if n > 0:
        bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n))
    return bars[:max_bars]


def aggregate_tick_bars(
    ticks: list[dict], ticks_per_bar: int = 50, max_bars: int = TICK_MAX_BARS
) -> list[dict]:
    """Aggregate ticks into fixed-count tick bars."""
    if not ticks:
        return []

    bars: list[dict] = []
    o = h = l = c = ticks[0]["price"]
    vol = vwap_num = 0.0
    n = 0
    bar_start = ticks[0]["timestamp"]

    for t in ticks:
        p, v = t["price"], t["volume"]
        if n == 0:
            bar_start = t["timestamp"]
            o = h = l = p
        h, l, c = max(h, p), min(l, p), p
        vol += v
        vwap_num += p * v
        n += 1

        if n >= ticks_per_bar:
            bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n))
            if len(bars) >= max_bars:
                return bars
            o = h = l = c = p
            vol = vwap_num = 0.0
            n = 0

    if n > 0:
        bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n))
    return bars[:max_bars]


def aggregate_imbalance_bars(
    ticks: list[dict], threshold: float = 10.0, max_bars: int = TICK_MAX_BARS
) -> list[dict]:
    """Aggregate ticks into trade imbalance bars.

    A new bar forms when the cumulative buy-sell imbalance exceeds threshold.
    """
    if not ticks:
        return []

    bars: list[dict] = []
    o = h = l = c = ticks[0]["price"]
    vol = vwap_num = 0.0
    n = 0
    imbalance = 0.0
    bar_start = ticks[0]["timestamp"]

    for t in ticks:
        p, v = t["price"], t["volume"]
        sign = 1.0 if t.get("direction") == "buy" else -1.0
        if n == 0:
            bar_start = t["timestamp"]
            o = h = l = p
        h, l, c = max(h, p), min(l, p), p
        vol += v
        vwap_num += p * v
        n += 1
        imbalance += sign * v

        if abs(imbalance) >= threshold:
            bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n, imbalance=imbalance))
            if len(bars) >= max_bars:
                return bars
            o = h = l = c = p
            vol = vwap_num = 0.0
            n = 0
            imbalance = 0.0

    if n > 0:
        bars.append(_make_bar(bar_start, o, h, l, c, vol, vwap_num, n, imbalance=imbalance))
    return bars[:max_bars]


def bars_to_returns(bars: list[dict]) -> np.ndarray:
    """Convert OHLCV bars to log-returns (%) from close prices."""
    if len(bars) < 2:
        return np.array([])
    closes = np.array([b["close"] for b in bars])
    return 100.0 * np.diff(np.log(closes))

