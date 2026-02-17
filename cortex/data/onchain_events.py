"""On-chain event collection and classification for Hawkes process calibration.

Collects and classifies on-chain events from Solana DEX transactions:
  - Large swaps (>$50k notional)
  - Oracle price jumps (>2% in one slot)
  - Liquidation events (lending protocol interactions)

Events are timestamped at slot-level precision (~400ms) for Hawkes fitting.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from cortex.config import (
    ONCHAIN_LARGE_SWAP_USD,
    ONCHAIN_ORACLE_JUMP_PCT,
    ONCHAIN_FUNDING_SPIKE_BPS,
    ONCHAIN_EVENT_LOOKBACK_SLOTS,
)

logger = logging.getLogger(__name__)

TESTING = os.environ.get("TESTING", "0") == "1"

# Slot duration on Solana (~400ms)
SLOT_DURATION_S = 0.4


def classify_event(swap: dict, prev_price: float | None = None) -> dict | None:
    """Classify a parsed swap record into an event type.

    Returns event dict or None if the swap doesn't qualify as an event.
    """
    price = swap.get("price", 0.0)
    amount_in = swap.get("amount_in", 0.0)
    amount_out = swap.get("amount_out", 0.0)
    notional = max(amount_in * price, amount_out) if price > 0 else 0.0

    # Large swap detection
    if notional >= ONCHAIN_LARGE_SWAP_USD:
        return {
            "event_type": "large_swap",
            "slot": swap.get("slot", 0),
            "timestamp": float(swap.get("block_time", 0)),
            "magnitude": notional,
            "details": {
                "dex": swap.get("dex", "unknown"),
                "price": price,
                "notional_usd": notional,
                "direction": swap.get("direction", "unknown"),
            },
        }

    # Oracle price jump detection
    if prev_price is not None and prev_price > 0 and price > 0:
        pct_change = abs(price - prev_price) / prev_price * 100.0
        if pct_change >= ONCHAIN_ORACLE_JUMP_PCT:
            return {
                "event_type": "oracle_jump",
                "slot": swap.get("slot", 0),
                "timestamp": float(swap.get("block_time", 0)),
                "magnitude": pct_change,
                "details": {
                    "price_before": prev_price,
                    "price_after": price,
                    "pct_change": pct_change,
                    "direction": "up" if price > prev_price else "down",
                },
            }

    return None


def classify_liquidation(tx: dict) -> dict | None:
    """Classify a transaction as a liquidation event if it interacts with lending protocols."""
    from cortex.data.solana import LENDING_PROGRAMS

    account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
    logs = " ".join(tx.get("meta", {}).get("logMessages", []) or []).lower()

    for program_id, name in LENDING_PROGRAMS.items():
        if program_id in account_keys or name.lower() in logs:
            if "liquidat" in logs:
                return {
                    "event_type": "liquidation",
                    "slot": tx.get("slot", 0),
                    "timestamp": float(tx.get("blockTime", 0)),
                    "magnitude": 1.0,
                    "details": {"protocol": name, "signature": tx.get("transaction", {}).get("signatures", [""])[0]},
                }
    return None


def collect_events(
    swaps: list[dict],
    transactions: list[dict] | None = None,
) -> list[dict]:
    """Collect and classify events from swap records and raw transactions.

    Args:
        swaps: Parsed swap records (from parse_swap_from_tx).
        transactions: Optional raw transactions for liquidation detection.

    Returns:
        Sorted list of classified events.
    """
    events: list[dict] = []
    sorted_swaps = sorted(swaps, key=lambda s: s.get("slot", 0))

    prev_price = None
    for s in sorted_swaps:
        ev = classify_event(s, prev_price)
        if ev is not None:
            events.append(ev)
        price = s.get("price", 0.0)
        if price > 0:
            prev_price = price

    if transactions:
        for tx in transactions:
            liq = classify_liquidation(tx)
            if liq is not None:
                events.append(liq)

    events.sort(key=lambda e: (e["slot"], e["timestamp"]))
    return events




def events_to_hawkes_times(
    events: list[dict], event_types: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Convert classified events to Hawkes-compatible time arrays.

    Returns dict mapping event_type -> sorted array of event times (seconds).
    Times are normalized relative to the earliest event.
    """
    if not events:
        return {}

    filtered = events
    if event_types:
        filtered = [e for e in events if e["event_type"] in event_types]

    if not filtered:
        return {}

    t_min = min(e["timestamp"] for e in filtered)

    by_type: dict[str, list[float]] = {}
    for e in filtered:
        etype = e["event_type"]
        by_type.setdefault(etype, []).append(e["timestamp"] - t_min)

    return {k: np.array(sorted(v)) for k, v in by_type.items()}


def get_event_type_counts(events: list[dict]) -> dict[str, int]:
    """Count events by type."""
    counts: dict[str, int] = {}
    for e in events:
        etype = e["event_type"]
        counts[etype] = counts.get(etype, 0) + 1
    return counts