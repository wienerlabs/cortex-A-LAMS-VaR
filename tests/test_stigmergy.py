"""Tests for cortex/stigmergy.py — DX-Research Task 4: Shared State Stigmergy."""
import time
from unittest.mock import patch

import pytest

from cortex.stigmergy import (
    ConsensusResult,
    PheromoneBoard,
    PheromoneSignal,
    _board,
    deposit_signal,
    get_board,
    get_board_snapshot,
    get_consensus,
)
import cortex.stigmergy as stigmergy_module


@pytest.fixture(autouse=True)
def reset_board():
    """Reset the global board between tests."""
    old = stigmergy_module._board
    stigmergy_module._board = None
    yield
    stigmergy_module._board = old


class TestPheromoneSignal:
    def test_creation(self):
        sig = PheromoneSignal(source="arb_analyst", token="SOL", direction="bullish", strength=0.8)
        assert sig.source == "arb_analyst"
        assert sig.direction == "bullish"
        assert sig.ts > 0

    def test_decayed_strength_immediate(self):
        sig = PheromoneSignal(source="test", token="SOL", direction="bullish", strength=0.8, ts=time.time())
        # Immediately after creation, decay should be minimal
        assert sig.decayed_strength() > 0.79

    def test_decayed_strength_after_time(self):
        # Signal created 300 seconds ago (one half-life with default config)
        sig = PheromoneSignal(source="test", token="SOL", direction="bullish", strength=1.0, ts=time.time() - 300)
        decayed = sig.decayed_strength()
        # After one half-life, should be ~0.5
        assert 0.4 < decayed < 0.6

    def test_decayed_strength_very_old(self):
        sig = PheromoneSignal(source="test", token="SOL", direction="bullish", strength=1.0, ts=time.time() - 3600)
        decayed = sig.decayed_strength()
        # After 12 half-lives, should be near zero
        assert decayed < 0.01

    def test_to_dict(self):
        sig = PheromoneSignal(source="test", token="BTC", direction="bearish", strength=0.7, metadata={"reason": "vol"})
        d = sig.to_dict()
        assert d["source"] == "test"
        assert d["token"] == "BTC"
        assert d["direction"] == "bearish"
        assert "decayed_strength" in d
        assert "age_seconds" in d
        assert d["metadata"]["reason"] == "vol"


class TestPheromoneBoard:
    def test_deposit_and_consensus(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.8))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bullish", strength=0.7))

        c = board.get_consensus("SOL")
        assert c.direction == "bullish"
        assert c.num_sources == 2
        assert c.conviction > 0.5

    def test_mixed_signals(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.9))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bearish", strength=0.9))

        c = board.get_consensus("SOL")
        assert c.num_sources == 2
        # Equal and opposite signals → low conviction
        assert c.conviction < 0.2

    def test_bearish_consensus(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="ETH", direction="bearish", strength=0.9))
        board.deposit(PheromoneSignal(source="a2", token="ETH", direction="bearish", strength=0.8))
        board.deposit(PheromoneSignal(source="a3", token="ETH", direction="bullish", strength=0.2))

        c = board.get_consensus("ETH")
        assert c.direction == "bearish"
        assert c.conviction > 0.5

    def test_replace_same_source(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.8))
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bearish", strength=0.9))

        # Same source, should only have one signal
        c = board.get_consensus("SOL")
        assert c.num_sources == 1
        assert c.direction == "bearish"

    def test_empty_consensus(self):
        board = PheromoneBoard()
        c = board.get_consensus("UNKNOWN")
        assert c.direction == "neutral"
        assert c.conviction == 0.0
        assert c.num_sources == 0
        assert c.swarm_active is False

    def test_swarm_threshold(self):
        board = PheromoneBoard()
        # Default swarm threshold is 3
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.8))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bullish", strength=0.7))

        c = board.get_consensus("SOL")
        assert c.swarm_active is False

        board.deposit(PheromoneSignal(source="a3", token="SOL", direction="bullish", strength=0.6))
        c = board.get_consensus("SOL")
        assert c.swarm_active is True

    def test_swarm_amplification(self):
        board = PheromoneBoard()
        # Below swarm threshold
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.8))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bullish", strength=0.7))
        c_low = board.get_consensus("SOL")

        # Above swarm threshold (add 2 more)
        board.deposit(PheromoneSignal(source="a3", token="SOL", direction="bullish", strength=0.6))
        board.deposit(PheromoneSignal(source="a4", token="SOL", direction="bullish", strength=0.5))
        c_high = board.get_consensus("SOL")

        # Swarm should amplify conviction
        assert c_high.conviction >= c_low.conviction

    def test_prune_expired(self):
        board = PheromoneBoard()
        # Add very old signal
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.5, ts=time.time() - 7200))
        # Add fresh signal
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bullish", strength=0.5))

        removed = board.prune_expired()
        assert removed == 1
        assert len(board._signals["SOL"]) == 1

    def test_max_signals_per_token(self):
        board = PheromoneBoard(max_signals_per_token=3)
        for i in range(5):
            board.deposit(PheromoneSignal(source=f"a{i}", token="SOL", direction="bullish", strength=0.5))
        assert len(board._signals["SOL"]) == 3

    def test_get_all_tokens(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.5))
        board.deposit(PheromoneSignal(source="a1", token="BTC", direction="bearish", strength=0.5))
        tokens = board.get_all_tokens()
        assert set(tokens) == {"SOL", "BTC"}

    def test_clear(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.5))
        board.clear()
        assert board.get_all_tokens() == []

    def test_snapshot(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.8))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bullish", strength=0.7))
        snap = board.snapshot()
        assert snap["total_tokens"] == 1
        assert "SOL" in snap["tokens"]
        assert snap["tokens"]["SOL"]["direction"] == "bullish"

    def test_multi_token_isolation(self):
        board = PheromoneBoard()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bullish", strength=0.9))
        board.deposit(PheromoneSignal(source="a1", token="BTC", direction="bearish", strength=0.9))

        sol = board.get_consensus("SOL")
        btc = board.get_consensus("BTC")
        assert sol.direction == "bullish"
        assert btc.direction == "bearish"


class TestConsensusResult:
    def test_to_dict(self):
        c = ConsensusResult(
            token="SOL", direction="bullish", conviction=0.8,
            num_sources=3, swarm_active=True,
            bullish_weight=1.5, bearish_weight=0.3,
        )
        d = c.to_dict()
        assert d["token"] == "SOL"
        assert d["swarm_active"] is True
        assert d["conviction"] == 0.8


class TestModuleLevelAPI:
    def test_get_board_singleton(self):
        b1 = get_board()
        b2 = get_board()
        assert b1 is b2

    def test_deposit_signal_enabled(self):
        with patch("cortex.stigmergy.STIGMERGY_ENABLED", True):
            deposit_signal("analyst_1", "SOL", "bullish", 0.8)
            c = get_consensus("SOL")
            assert c.direction == "bullish"
            assert c.num_sources == 1

    def test_deposit_signal_disabled(self):
        with patch("cortex.stigmergy.STIGMERGY_ENABLED", False):
            deposit_signal("analyst_1", "SOL", "bullish", 0.8)
            c = get_consensus("SOL")
            assert c.num_sources == 0

    def test_deposit_clamps_strength(self):
        with patch("cortex.stigmergy.STIGMERGY_ENABLED", True):
            deposit_signal("analyst_1", "SOL", "bullish", 2.5)
            c = get_consensus("SOL")
            assert c.bullish_weight <= 1.0

    def test_get_board_snapshot(self):
        with patch("cortex.stigmergy.STIGMERGY_ENABLED", True):
            deposit_signal("a1", "SOL", "bullish", 0.8)
            snap = get_board_snapshot()
            assert snap["total_tokens"] == 1
            assert "stigmergy_enabled" in snap


class TestDebateIntegration:
    """Test that stigmergy evidence flows into the debate system."""

    def test_stigmergy_evidence_in_debate(self):
        """Debate should include stigmergy consensus as evidence when token provided."""
        from cortex.stigmergy import get_board

        # Deposit signals onto the board
        board = get_board()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bearish", strength=0.9))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bearish", strength=0.8))
        board.deposit(PheromoneSignal(source="a3", token="SOL", direction="bearish", strength=0.7))

        from cortex.debate import run_debate

        r = run_debate(
            risk_score=50.0,
            component_scores=[{"component": "vol", "score": 50}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            token="SOL",
            enrich=False,
        )
        assert r["stigmergy_active"] is True
        # Bearish stigmergy should add bearish evidence
        assert r["evidence_summary"]["bearish"] > 0

    def test_debate_without_token_no_stigmergy(self):
        """Without token, stigmergy evidence should not be collected."""
        from cortex.debate import run_debate

        r = run_debate(
            risk_score=50.0,
            component_scores=[{"component": "vol", "score": 50}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            enrich=False,
        )
        assert r["stigmergy_active"] is False

    def test_bullish_stigmergy_helps_approval(self):
        """Bullish swarm consensus should help trade get approved."""
        board = get_board()
        board.deposit(PheromoneSignal(source="a1", token="BTC", direction="bullish", strength=0.9))
        board.deposit(PheromoneSignal(source="a2", token="BTC", direction="bullish", strength=0.85))
        board.deposit(PheromoneSignal(source="a3", token="BTC", direction="bullish", strength=0.8))

        from cortex.debate import run_debate

        r = run_debate(
            risk_score=30.0,
            component_scores=[{"component": "vol", "score": 30}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            token="BTC",
            enrich=False,
        )
        assert r["final_decision"] == "approve"
        assert r["evidence_summary"]["bullish"] >= 2  # at least component + stigmergy
