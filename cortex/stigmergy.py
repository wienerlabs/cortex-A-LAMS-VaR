"""DX-Research Task 4: Shared State Stigmergy.

DX01 finding: Termite-style indirect coordination outperforms direct messaging.
Agents deposit "pheromone" signals on a shared board and observe aggregate state
without explicit message passing. This produces emergent consensus while avoiding
groupthink (each agent still applies its own bias).

Architecture:
  - Pheromone Board: per-token signal accumulator with exponential decay
  - Signal Deposit: analysts write (direction, strength, source) tuples
  - Consensus Query: any agent can read aggregate consensus + conviction level
  - Decay: signals fade with configurable half-life (default 5 minutes)
  - Amplification: agreement from 3+ sources boosts conviction (swarm threshold)
"""
from __future__ import annotations

__all__ = [
    "PheromoneBoard",
    "PheromoneSignal",
    "get_board",
    "deposit_signal",
    "get_consensus",
    "get_board_snapshot",
]

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cortex.config import (
    STIGMERGY_DECAY_HALF_LIFE,
    STIGMERGY_ENABLED,
    STIGMERGY_SWARM_THRESHOLD,
)

logger = logging.getLogger(__name__)

_DECAY_LAMBDA = math.log(2) / max(STIGMERGY_DECAY_HALF_LIFE, 1.0)


@dataclass
class PheromoneSignal:
    """A single signal deposited on the board by an analyst."""
    source: str          # e.g. "momentum_analyst", "news_analyst"
    token: str           # e.g. "SOL", "BTC"
    direction: str       # "bullish", "bearish", "neutral"
    strength: float      # 0.0–1.0 conviction level
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()

    def decayed_strength(self, now: float | None = None) -> float:
        """Return strength after exponential decay."""
        now = now or time.time()
        elapsed = max(0.0, now - self.ts)
        return self.strength * math.exp(-_DECAY_LAMBDA * elapsed)

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {
            "source": self.source,
            "token": self.token,
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "decayed_strength": round(self.decayed_strength(now), 4),
            "metadata": self.metadata,
            "ts": self.ts,
            "age_seconds": round(now - self.ts, 1),
        }


@dataclass
class ConsensusResult:
    """Aggregated consensus state for a token."""
    token: str
    direction: str              # "bullish", "bearish", "neutral"
    conviction: float           # 0.0–1.0 weighted average of decayed strengths
    num_sources: int            # unique source count
    swarm_active: bool          # True if sources >= SWARM_THRESHOLD
    bullish_weight: float       # total decayed bullish strength
    bearish_weight: float       # total decayed bearish strength
    signals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "direction": self.direction,
            "conviction": round(self.conviction, 4),
            "num_sources": self.num_sources,
            "swarm_active": self.swarm_active,
            "bullish_weight": round(self.bullish_weight, 4),
            "bearish_weight": round(self.bearish_weight, 4),
            "signals": self.signals,
        }


class PheromoneBoard:
    """Shared pheromone board for indirect agent coordination.

    Signals are stored per-token and decay over time. Consensus is computed
    on read by aggregating all non-expired signals.
    """

    def __init__(self, max_signals_per_token: int = 50) -> None:
        self._signals: dict[str, list[PheromoneSignal]] = defaultdict(list)
        self._max_per_token = max_signals_per_token

    def deposit(self, signal: PheromoneSignal) -> None:
        """Deposit a pheromone signal on the board."""
        signals = self._signals[signal.token]

        # Replace existing signal from same source for same token
        self._signals[signal.token] = [
            s for s in signals if s.source != signal.source
        ]
        self._signals[signal.token].append(signal)

        # Trim oldest if over capacity
        if len(self._signals[signal.token]) > self._max_per_token:
            self._signals[signal.token] = sorted(
                self._signals[signal.token], key=lambda s: s.ts,
            )[-self._max_per_token:]

    def get_consensus(self, token: str, min_strength: float = 0.01) -> ConsensusResult:
        """Compute aggregate consensus for a token from all active signals."""
        now = time.time()
        signals = self._signals.get(token, [])

        bullish_weight = 0.0
        bearish_weight = 0.0
        sources: set[str] = set()
        active_signals: list[dict[str, Any]] = []

        for sig in signals:
            decayed = sig.decayed_strength(now)
            if decayed < min_strength:
                continue
            sources.add(sig.source)
            active_signals.append(sig.to_dict())

            if sig.direction == "bullish":
                bullish_weight += decayed
            elif sig.direction == "bearish":
                bearish_weight += decayed

        total_weight = bullish_weight + bearish_weight
        if total_weight > 0:
            conviction = abs(bullish_weight - bearish_weight) / total_weight
            direction = "bullish" if bullish_weight > bearish_weight else "bearish"
        else:
            conviction = 0.0
            direction = "neutral"

        num_sources = len(sources)
        swarm_active = num_sources >= STIGMERGY_SWARM_THRESHOLD

        # Swarm amplification: when enough sources agree, boost conviction
        if swarm_active and conviction > 0:
            amplification = 1.0 + 0.1 * (num_sources - STIGMERGY_SWARM_THRESHOLD)
            conviction = min(1.0, conviction * amplification)

        return ConsensusResult(
            token=token,
            direction=direction,
            conviction=round(conviction, 4),
            num_sources=num_sources,
            swarm_active=swarm_active,
            bullish_weight=bullish_weight,
            bearish_weight=bearish_weight,
            signals=active_signals,
        )

    def get_all_tokens(self) -> list[str]:
        """Return all tokens with active signals."""
        return list(self._signals.keys())

    def prune_expired(self, min_strength: float = 0.001) -> int:
        """Remove signals that have decayed below threshold. Returns count removed."""
        now = time.time()
        removed = 0
        for token in list(self._signals.keys()):
            before = len(self._signals[token])
            self._signals[token] = [
                s for s in self._signals[token]
                if s.decayed_strength(now) >= min_strength
            ]
            removed += before - len(self._signals[token])
            if not self._signals[token]:
                del self._signals[token]
        return removed

    def clear(self) -> None:
        """Clear all signals (for testing)."""
        self._signals.clear()

    def snapshot(self) -> dict[str, Any]:
        """Full board state for persistence / debugging."""
        tokens: dict[str, dict[str, Any]] = {}
        for token in self._signals:
            consensus = self.get_consensus(token)
            tokens[token] = consensus.to_dict()
        return {
            "total_tokens": len(tokens),
            "tokens": tokens,
            "stigmergy_enabled": STIGMERGY_ENABLED,
        }


# ── Module-level singleton ──

_board: PheromoneBoard | None = None


def get_board() -> PheromoneBoard:
    """Get or create the global pheromone board."""
    global _board
    if _board is None:
        _board = PheromoneBoard()
    return _board


def deposit_signal(
    source: str,
    token: str,
    direction: str,
    strength: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience: deposit a signal onto the global board."""
    if not STIGMERGY_ENABLED:
        return
    signal = PheromoneSignal(
        source=source, token=token, direction=direction,
        strength=max(0.0, min(1.0, strength)),
        metadata=metadata or {},
    )
    get_board().deposit(signal)
    logger.debug("Stigmergy deposit: %s → %s %s (%.2f)", source, direction, token, strength)


def get_consensus(token: str) -> ConsensusResult:
    """Convenience: get consensus for a token from the global board."""
    return get_board().get_consensus(token)


def get_board_snapshot() -> dict[str, Any]:
    """Get full board snapshot."""
    return get_board().snapshot()
