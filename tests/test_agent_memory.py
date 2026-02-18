"""Tests for cortex/agent_memory.py — DX-Research Task 3: Layered Agent Memory."""
import time

import pytest

from cortex.agent_memory import (
    AgentMemory,
    LongTermSummary,
    ShortTermEntry,
    get_context_snapshot,
    get_memory,
    record_decision,
    _memories,
)


@pytest.fixture(autouse=True)
def clear_registry():
    _memories.clear()
    yield
    _memories.clear()


class TestShortTermEntry:
    def test_to_dict(self):
        e = ShortTermEntry(token="SOL", direction="long", score=35.0, approved=True, ts=1.0)
        d = e.to_dict()
        assert d["token"] == "SOL"
        assert d["approved"] is True
        assert d["pnl"] is None

    def test_with_pnl(self):
        e = ShortTermEntry(token="SOL", direction="long", score=35.0, approved=True, pnl=100.5, ts=1.0)
        assert e.to_dict()["pnl"] == 100.5


class TestLongTermSummary:
    def test_initial_state(self):
        s = LongTermSummary()
        assert s.win_rate == 0.5
        assert s.total_decisions == 0

    def test_update_tracks_decisions(self):
        s = LongTermSummary()
        s.update(30.0, won=True, regime=1)
        s.update(70.0, won=False, regime=2)
        assert s.total_decisions == 2
        assert s.total_wins == 1
        assert s.total_losses == 1
        assert s.win_rate == 0.5

    def test_ema_risk_score(self):
        s = LongTermSummary()
        for _ in range(20):
            s.update(80.0, won=None, regime=0)
        # EMA with decay=0.95 converges slowly: after 20 updates from 50→80, ~69
        assert s.avg_risk_score > 65.0

    def test_regime_distribution(self):
        s = LongTermSummary()
        for _ in range(10):
            s.update(50.0, won=None, regime=3)
        assert s.dominant_regime == 3
        assert 3 in s.regime_distribution

    def test_cumulative_pnl_not_affected_by_update(self):
        s = LongTermSummary()
        s.update(50.0, won=True)
        assert s.cumulative_pnl == 0.0

    def test_to_dict(self):
        s = LongTermSummary()
        s.update(40.0, won=True, regime=1)
        d = s.to_dict()
        assert "win_rate" in d
        assert "dominant_regime" in d


class TestAgentMemory:
    def test_record_decision(self):
        m = AgentMemory("test_agent")
        m.record_decision("SOL", "long", 35.0, True, regime=1)
        assert len(m.short_term) == 1
        assert m.long_term.total_decisions == 1

    def test_short_term_circular_buffer(self):
        m = AgentMemory("test_agent")
        for i in range(25):
            m.record_decision(f"TOKEN_{i}", "long", float(i), True)
        assert len(m.short_term) == 20  # maxlen

    def test_record_outcome_attaches_pnl(self):
        m = AgentMemory("test_agent")
        m.record_decision("SOL", "long", 35.0, True)
        m.record_outcome("SOL", 150.0)
        assert m.short_term[-1].pnl == 150.0
        assert m.long_term.total_wins == 1
        assert m.long_term.cumulative_pnl == 150.0

    def test_record_outcome_negative(self):
        m = AgentMemory("test_agent")
        m.record_decision("ETH", "short", 60.0, True)
        m.record_outcome("ETH", -50.0)
        assert m.long_term.total_losses == 1
        assert m.long_term.cumulative_pnl == -50.0

    def test_record_outcome_matches_most_recent(self):
        m = AgentMemory("test_agent")
        m.record_decision("SOL", "long", 30.0, True)
        m.record_decision("SOL", "long", 40.0, True)
        m.record_outcome("SOL", 100.0)
        # Should attach to the LAST SOL entry
        entries = list(m.short_term)
        assert entries[-1].pnl == 100.0
        assert entries[0].pnl is None

    def test_context_update(self):
        m = AgentMemory("test_agent")
        m.update_context("momentum_analyst", {"direction": "LONG", "score": 0.8})
        ctx = m.get_context()
        assert "momentum_analyst" in ctx
        assert ctx["momentum_analyst"]["signal"]["direction"] == "LONG"

    def test_context_expiry(self):
        m = AgentMemory("test_agent")
        m.contextual["old_agent"] = {"signal": {}, "ts": time.time() - 600}
        ctx = m.get_context(max_age_seconds=300)
        assert "old_agent" not in ctx

    def test_get_recent_decisions(self):
        m = AgentMemory("test_agent")
        for i in range(10):
            m.record_decision(f"T{i}", "long", float(i * 10), True)
        recent = m.get_recent_decisions(3)
        assert len(recent) == 3
        assert recent[-1]["token"] == "T9"

    def test_snapshot(self):
        m = AgentMemory("test_agent")
        m.record_decision("SOL", "long", 35.0, True, regime=2)
        m.update_context("other", {"x": 1})
        snap = m.snapshot()
        assert snap["agent_id"] == "test_agent"
        assert len(snap["short_term"]) == 1
        assert "win_rate" in snap["long_term"]
        assert "other" in snap["contextual"]


class TestModuleLevelAPI:
    def test_get_memory_creates_new(self):
        m = get_memory("analyst_1")
        assert isinstance(m, AgentMemory)
        assert m.agent_id == "analyst_1"

    def test_get_memory_returns_same_instance(self):
        m1 = get_memory("analyst_1")
        m2 = get_memory("analyst_1")
        assert m1 is m2

    def test_record_decision_convenience(self):
        record_decision("analyst_2", "SOL", "long", 30.0, True, regime=1)
        m = get_memory("analyst_2")
        assert len(m.short_term) == 1

    def test_get_context_snapshot(self):
        record_decision("analyst_3", "BTC", "short", 70.0, False)
        snap = get_context_snapshot("analyst_3")
        assert snap["agent_id"] == "analyst_3"
        assert len(snap["short_term"]) == 1
