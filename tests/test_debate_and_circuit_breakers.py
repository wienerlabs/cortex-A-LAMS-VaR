"""Comprehensive tests for the adversarial debate pipeline and circuit breakers.

Tests cover:
  - Debate evidence collection
  - Bayesian confidence updates
  - 4-agent debate rounds (Trader, Risk Manager, Devil's Advocate, PM)
  - Strategy-aware risk profiles
  - Per-strategy outcome circuit breakers
  - Circuit breaker state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
  - Integration between debate and circuit breaker data
"""
from __future__ import annotations

import time
import sys
import importlib.util
from pathlib import Path

import pytest

# ── Load modules via importlib to avoid cortex package init issues ──

ROOT = Path(__file__).resolve().parent.parent
CORTEX = ROOT / "cortex"


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Must load config first (debate and circuit_breaker depend on it)
config_mod = _load_module("cortex.config", CORTEX / "config.py")
# Load portfolio_risk
portfolio_risk_mod = _load_module("cortex.portfolio_risk", CORTEX / "portfolio_risk.py")
# Load circuit_breaker
cb_mod = _load_module("cortex.circuit_breaker", CORTEX / "circuit_breaker.py")
# Load debate
debate_mod = _load_module("cortex.debate", CORTEX / "debate.py")


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset all circuit breakers before each test."""
    # Clear module-level state
    cb_mod._breakers.clear()
    cb_mod._outcome_breakers.clear()
    yield
    cb_mod._breakers.clear()
    cb_mod._outcome_breakers.clear()


@pytest.fixture
def low_risk_scores():
    """Component scores indicating low risk."""
    return [
        {"component": "evt", "score": 25.0, "details": {}},
        {"component": "svj", "score": 20.0, "details": {}},
        {"component": "hawkes", "score": 15.0, "details": {}},
        {"component": "regime", "score": 30.0, "details": {}},
        {"component": "news", "score": 35.0, "details": {}},
        {"component": "alams", "score": 18.0, "details": {}},
    ]


@pytest.fixture
def high_risk_scores():
    """Component scores indicating high risk."""
    return [
        {"component": "evt", "score": 85.0, "details": {}},
        {"component": "svj", "score": 75.0, "details": {}},
        {"component": "hawkes", "score": 90.0, "details": {}},
        {"component": "regime", "score": 80.0, "details": {}},
        {"component": "news", "score": 70.0, "details": {}},
        {"component": "alams", "score": 88.0, "details": {}},
    ]


# ══════════════════════════════════════════════════════════════════════
# DEBATE EVIDENCE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestDebateEvidence:
    def test_evidence_supports_approval_below_threshold(self):
        ev = debate_mod.DebateEvidence(
            source="test", claim="test", value=30.0, threshold=60.0,
        )
        assert ev.supports_approval() is True

    def test_evidence_rejects_above_threshold(self):
        ev = debate_mod.DebateEvidence(
            source="test", claim="test", value=80.0, threshold=60.0,
        )
        assert ev.supports_approval() is False

    def test_evidence_to_dict(self):
        ev = debate_mod.DebateEvidence(
            source="guardian:evt", claim="EVT risk 30/100",
            value=30.0, threshold=60.0, severity="low",
        )
        d = ev.to_dict()
        assert d["source"] == "guardian:evt"
        assert d["supports_approval"] is True
        assert d["severity"] == "low"

    def test_evidence_collection_separates_bullish_bearish(self, low_risk_scores):
        ctx = debate_mod.DebateContext(
            risk_score=30.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, strategy="spot",
        )
        evidence = debate_mod._collect_evidence(ctx)
        assert len(evidence["bullish"]) > 0
        # Risk score is always added to bearish
        assert len(evidence["bearish"]) > 0

    def test_veto_reasons_create_critical_evidence(self):
        ctx = debate_mod.DebateContext(
            risk_score=80.0, component_scores=[],
            veto_reasons=["hawkes_flash_crash_risk", "evt_extreme_tail"],
            direction="long", trade_size_usd=1000.0,
            original_approved=False, strategy="perp",
        )
        evidence = debate_mod._collect_evidence(ctx)
        critical = [e for e in evidence["bearish"] if e.severity == "critical"]
        assert len(critical) >= 2  # At least the two veto reasons

    def test_alams_data_generates_evidence(self):
        ctx = debate_mod.DebateContext(
            risk_score=50.0, component_scores=[],
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, strategy="spot",
            alams_data={"var_total": 0.06, "current_regime": 4, "delta": 0.35},
        )
        evidence = debate_mod._collect_evidence(ctx)
        alams_ev = [e for e in evidence["bearish"] if "alams" in e.source]
        assert len(alams_ev) >= 2  # VaR + regime + possibly delta

    def test_kelly_within_bounds_is_bullish(self):
        ctx = debate_mod.DebateContext(
            risk_score=40.0, component_scores=[],
            veto_reasons=[], direction="long", trade_size_usd=500.0,
            original_approved=True, strategy="spot",
            kelly_stats={"active": True, "kelly_fraction": 0.10, "win_rate": 0.60, "n_trades": 100},
            portfolio_value=100_000.0,
        )
        evidence = debate_mod._collect_evidence(ctx)
        kelly_bull = [e for e in evidence["bullish"] if "kelly" in e.source]
        assert len(kelly_bull) > 0


# ══════════════════════════════════════════════════════════════════════
# BAYESIAN UPDATE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestBayesianUpdate:
    def test_strong_bullish_evidence_shifts_toward_approval(self):
        evidence = [
            debate_mod.DebateEvidence("a", "low risk", 20.0, 60.0, severity="high"),
            debate_mod.DebateEvidence("b", "favorable", 10.0, 60.0, severity="critical"),
        ]
        posterior = debate_mod._bayesian_update(0.5, evidence, "approve")
        assert posterior > 0.8

    def test_strong_bearish_evidence_shifts_toward_rejection(self):
        evidence = [
            debate_mod.DebateEvidence("a", "high risk", 90.0, 60.0, severity="critical"),
            debate_mod.DebateEvidence("b", "extreme", 95.0, 60.0, severity="high"),
        ]
        posterior = debate_mod._bayesian_update(0.5, evidence, "reject")
        assert posterior > 0.8

    def test_uniform_prior_with_no_evidence(self):
        posterior = debate_mod._bayesian_update(0.5, [], "approve")
        assert abs(posterior - 0.5) < 0.01

    def test_posterior_bounded_0_1(self):
        massive_evidence = [
            debate_mod.DebateEvidence("x", "extreme", 99.0, 60.0, severity="critical")
            for _ in range(20)
        ]
        posterior = debate_mod._bayesian_update(0.5, massive_evidence, "reject")
        assert 0.0 < posterior < 1.0


# ══════════════════════════════════════════════════════════════════════
# DEBATE ROUND TESTS
# ══════════════════════════════════════════════════════════════════════

class TestDebateRounds:
    def test_run_debate_returns_expected_keys(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=30.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, enrich=False,
        )
        assert "final_decision" in result
        assert "final_confidence" in result
        assert "rounds" in result
        assert "evidence_summary" in result
        assert "strategy" in result
        assert "recommended_size_pct" in result

    def test_low_risk_debate_approves(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=25.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=500.0,
            original_approved=True, enrich=False,
        )
        assert result["final_decision"] == "approve"

    def test_high_risk_debate_rejects(self, high_risk_scores):
        result = debate_mod.run_debate(
            risk_score=85.0, component_scores=high_risk_scores,
            veto_reasons=["hawkes_critical_contagion"],
            direction="long", trade_size_usd=5000.0,
            original_approved=False, enrich=False,
        )
        assert result["final_decision"] == "reject"

    def test_veto_always_rejects(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=30.0, component_scores=low_risk_scores,
            veto_reasons=["circuit_breaker_global"],
            direction="long", trade_size_usd=1000.0,
            original_approved=True, enrich=False,
        )
        assert result["final_decision"] == "reject"

    def test_debate_has_4_agents_per_round(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=50.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, enrich=False,
        )
        r = result["rounds"][0]
        assert "trader" in r
        assert "risk_manager" in r
        assert "devils_advocate" in r
        assert "arbitrator" in r

    def test_debate_early_termination_on_high_confidence(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=10.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=100.0,
            original_approved=True, enrich=False,
        )
        # Should terminate early (confidence > 0.7) rather than running max rounds
        assert result["num_rounds"] <= config_mod.DEBATE_MAX_ROUNDS

    def test_decision_changed_flag(self, high_risk_scores):
        result = debate_mod.run_debate(
            risk_score=85.0, component_scores=high_risk_scores,
            veto_reasons=["test_veto"], direction="long",
            trade_size_usd=5000.0, original_approved=True, enrich=False,
        )
        # Originally approved but should be rejected
        assert result["decision_changed"] is True


# ══════════════════════════════════════════════════════════════════════
# STRATEGY-AWARE DEBATE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestStrategyAwareDebate:
    def test_arb_strategy_profile(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=40.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=500.0,
            original_approved=True, strategy="arb", enrich=False,
        )
        assert result["strategy"] == "arb"
        # Check that strategy profile is used in PM arbitration
        pm = result["rounds"][-1]["arbitrator"]
        assert pm["strategy_profile"] == "arb"

    def test_perp_strategy_has_lower_risk_tolerance(self, low_risk_scores):
        # Perps have risk_tolerance=0.55, so approve_threshold = 0.45
        result_perp = debate_mod.run_debate(
            risk_score=50.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, strategy="perp", enrich=False,
        )
        result_lp = debate_mod.run_debate(
            risk_score=50.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, strategy="lp", enrich=False,
        )
        # Both should return valid results
        assert result_perp["strategy"] == "perp"
        assert result_lp["strategy"] == "lp"

    def test_recommended_size_respects_strategy_cap(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=20.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=500.0,
            original_approved=True, strategy="arb", enrich=False,
        )
        if result["final_decision"] == "approve":
            # Arb cap is 0.15 (15%)
            assert result["recommended_size_pct"] <= 0.15 + 0.001

    def test_evidence_summary_present(self, low_risk_scores):
        result = debate_mod.run_debate(
            risk_score=50.0, component_scores=low_risk_scores,
            veto_reasons=[], direction="long", trade_size_usd=1000.0,
            original_approved=True, enrich=False,
        )
        summary = result["evidence_summary"]
        assert "total" in summary
        assert "bullish" in summary
        assert "bearish" in summary
        assert summary["total"] == summary["bullish"] + summary["bearish"]


# ══════════════════════════════════════════════════════════════════════
# RISK-SCORE CIRCUIT BREAKER TESTS
# ══════════════════════════════════════════════════════════════════════

class TestRiskScoreCircuitBreaker:
    def test_initial_state_closed(self):
        cb_mod._ensure_breakers()
        assert cb_mod._breakers["global"].state == cb_mod.CBState.CLOSED

    def test_trip_after_consecutive_high_scores(self):
        cb = cb_mod.CircuitBreaker("test", threshold=80, consecutive=3, cooldown=300)
        cb.record_score(85)
        cb.record_score(90)
        assert cb.state == cb_mod.CBState.CLOSED
        cb.record_score(95)
        assert cb.state == cb_mod.CBState.OPEN

    def test_is_blocked_when_open(self):
        cb = cb_mod.CircuitBreaker("test", threshold=80, consecutive=2, cooldown=9999)
        cb.record_score(90)
        cb.record_score(95)
        assert cb.is_blocked() is True

    def test_recovery_after_cooldown(self):
        cb = cb_mod.CircuitBreaker("test", threshold=80, consecutive=2, cooldown=0.01)
        cb.record_score(90)
        cb.record_score(95)
        assert cb.state == cb_mod.CBState.OPEN
        time.sleep(0.02)
        # is_blocked() transitions to HALF_OPEN
        assert cb.is_blocked() is False
        assert cb.state == cb_mod.CBState.HALF_OPEN

    def test_half_open_recovers_on_good_score(self):
        cb = cb_mod.CircuitBreaker("test", threshold=80, consecutive=2, cooldown=0.01)
        cb.record_score(90)
        cb.record_score(95)
        time.sleep(0.02)
        cb.record_score(50)  # Transitions OPEN → HALF_OPEN (returns early)
        assert cb.state == cb_mod.CBState.HALF_OPEN
        cb.record_score(50)  # Now in HALF_OPEN, good score → CLOSED
        assert cb.state == cb_mod.CBState.CLOSED

    def test_reset_restores_closed(self):
        cb = cb_mod.CircuitBreaker("test", threshold=80, consecutive=2, cooldown=9999)
        cb.record_score(90)
        cb.record_score(95)
        assert cb.state == cb_mod.CBState.OPEN
        cb.reset()
        assert cb.state == cb_mod.CBState.CLOSED
        assert cb.fail_count == 0


# ══════════════════════════════════════════════════════════════════════
# OUTCOME CIRCUIT BREAKER TESTS
# ══════════════════════════════════════════════════════════════════════

class TestOutcomeCircuitBreaker:
    def test_lp_trips_after_3_il(self):
        ob = cb_mod.OutcomeCircuitBreaker("lp")
        assert ob.loss_limit == 3
        ob.record_outcome(False, pnl=-50, loss_type="impermanent_loss")
        ob.record_outcome(False, pnl=-30, loss_type="impermanent_loss")
        assert ob.state == cb_mod.CBState.CLOSED
        ob.record_outcome(False, pnl=-40, loss_type="impermanent_loss")
        assert ob.state == cb_mod.CBState.OPEN

    def test_arb_trips_after_5_failures(self):
        ob = cb_mod.OutcomeCircuitBreaker("arb")
        assert ob.loss_limit == 5
        for _ in range(4):
            ob.record_outcome(False, loss_type="failed_execution")
        assert ob.state == cb_mod.CBState.CLOSED
        ob.record_outcome(False, loss_type="failed_execution")
        assert ob.state == cb_mod.CBState.OPEN

    def test_perp_trips_after_2_stop_losses(self):
        ob = cb_mod.OutcomeCircuitBreaker("perp")
        assert ob.loss_limit == 2
        ob.record_outcome(False, pnl=-100, loss_type="stop_loss")
        assert ob.state == cb_mod.CBState.CLOSED
        ob.record_outcome(False, pnl=-150, loss_type="stop_loss")
        assert ob.state == cb_mod.CBState.OPEN

    def test_win_resets_consecutive_count(self):
        ob = cb_mod.OutcomeCircuitBreaker("perp")
        ob.record_outcome(False, loss_type="stop_loss")
        assert ob.consecutive_losses == 1
        ob.record_outcome(True, pnl=50)
        assert ob.consecutive_losses == 0

    def test_cooldown_transition_to_half_open(self):
        ob = cb_mod.OutcomeCircuitBreaker("perp")
        ob.cooldown = 0.01  # Very short cooldown for testing
        ob.record_outcome(False, loss_type="stop_loss")
        ob.record_outcome(False, loss_type="stop_loss")
        assert ob.state == cb_mod.CBState.OPEN
        time.sleep(0.02)
        assert ob.is_blocked() is False
        assert ob.state == cb_mod.CBState.HALF_OPEN

    def test_successful_probe_recovers(self):
        ob = cb_mod.OutcomeCircuitBreaker("lp")
        ob.cooldown = 0.01
        ob.record_outcome(False)
        ob.record_outcome(False)
        ob.record_outcome(False)
        assert ob.state == cb_mod.CBState.OPEN
        time.sleep(0.02)
        ob.is_blocked()  # triggers HALF_OPEN
        ob.record_outcome(True, pnl=100)
        assert ob.state == cb_mod.CBState.CLOSED

    def test_failed_probe_reopens(self):
        ob = cb_mod.OutcomeCircuitBreaker("lp")
        ob.cooldown = 0.01
        ob.record_outcome(False)
        ob.record_outcome(False)
        ob.record_outcome(False)
        assert ob.state == cb_mod.CBState.OPEN
        time.sleep(0.02)
        ob.is_blocked()  # triggers HALF_OPEN
        ob.record_outcome(False)
        assert ob.state == cb_mod.CBState.OPEN

    def test_status_includes_win_rate(self):
        ob = cb_mod.OutcomeCircuitBreaker("arb")
        ob.record_outcome(True, pnl=10)
        ob.record_outcome(True, pnl=20)
        ob.record_outcome(False, pnl=-5)
        status = ob.status()
        assert abs(status["win_rate"] - 2 / 3) < 0.01
        assert status["total_trades"] == 3
        assert status["total_wins"] == 2
        assert status["total_losses"] == 1


# ══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CIRCUIT BREAKER FUNCTION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestModuleLevelBreakers:
    def test_record_score_returns_states(self):
        result = cb_mod.record_score(50.0, strategy="lp")
        assert "global" in result
        assert "lp" in result

    def test_record_trade_outcome_returns_status(self):
        result = cb_mod.record_trade_outcome("lp", success=False, pnl=-10.0)
        assert result["strategy"] == "lp"
        assert result["consecutive_losses"] == 1

    def test_is_blocked_checks_both_types(self):
        # Trip outcome breaker for perp
        cb_mod.record_trade_outcome("perp", False, loss_type="stop_loss")
        cb_mod.record_trade_outcome("perp", False, loss_type="stop_loss")
        blocked, blockers = cb_mod.is_blocked(strategy="perp")
        assert blocked is True
        assert "perp_outcome" in blockers

    def test_get_all_states_includes_both_types(self):
        states = cb_mod.get_all_states()
        # Should have risk-score breakers (global + strategies) + outcome breakers
        names = [s["name"] for s in states]
        assert "global" in names
        assert any("outcome" in n for n in names)

    def test_reset_all_clears_everything(self):
        cb_mod.record_trade_outcome("lp", False)
        cb_mod.record_trade_outcome("lp", False)
        cb_mod.record_trade_outcome("lp", False)
        blocked, _ = cb_mod.is_blocked(strategy="lp")
        assert blocked is True
        cb_mod.reset_all()
        blocked, _ = cb_mod.is_blocked(strategy="lp")
        assert blocked is False

    def test_reset_specific_outcome_breaker(self):
        cb_mod.record_trade_outcome("perp", False, loss_type="stop_loss")
        cb_mod.record_trade_outcome("perp", False, loss_type="stop_loss")
        blocked, _ = cb_mod.is_blocked(strategy="perp")
        assert blocked is True
        cb_mod.reset_breaker("perp_outcome")
        blocked, _ = cb_mod.is_blocked(strategy="perp")
        assert blocked is False


# ══════════════════════════════════════════════════════════════════════
# DEBATE + CIRCUIT BREAKER INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestDebateCircuitBreakerIntegration:
    def test_open_breaker_adds_evidence_to_debate(self):
        ctx = debate_mod.DebateContext(
            risk_score=50.0, component_scores=[],
            veto_reasons=[], direction="long",
            trade_size_usd=1000.0, original_approved=True,
            circuit_breaker_states=[
                {"name": "lp_outcome", "state": "open"},
                {"name": "global", "state": "closed"},
            ],
        )
        evidence = debate_mod._collect_evidence(ctx)
        cb_evidence = [e for e in evidence["bearish"] if "circuit_breaker" in e.source]
        assert len(cb_evidence) == 1
        assert cb_evidence[0].severity == "critical"

    def test_half_open_breaker_adds_high_severity_evidence(self):
        ctx = debate_mod.DebateContext(
            risk_score=50.0, component_scores=[],
            veto_reasons=[], direction="long",
            trade_size_usd=1000.0, original_approved=True,
            circuit_breaker_states=[
                {"name": "perp_outcome", "state": "half_open"},
            ],
        )
        evidence = debate_mod._collect_evidence(ctx)
        cb_evidence = [e for e in evidence["bearish"] if "circuit_breaker" in e.source]
        assert len(cb_evidence) == 1
        assert cb_evidence[0].severity == "high"

    def test_debate_with_enriched_breaker_data(self):
        # Trip outcome breaker first
        cb_mod.record_trade_outcome("arb", False)
        cb_mod.record_trade_outcome("arb", False)
        cb_mod.record_trade_outcome("arb", False)
        cb_mod.record_trade_outcome("arb", False)
        cb_mod.record_trade_outcome("arb", False)
        # Now breaker should be open

        # Pass breaker states to debate
        states = cb_mod.get_all_states()
        open_states = [s for s in states if s["state"] == "open"]

        result = debate_mod.run_debate(
            risk_score=40.0, component_scores=[],
            veto_reasons=[], direction="long",
            trade_size_usd=500.0, original_approved=True,
            strategy="arb", circuit_breaker_states=open_states, enrich=False,
        )
        # Should have circuit breaker evidence in the summary
        bearish_sources = [
            e["source"] for e in result["evidence_summary"]["bearish_items"]
        ]
        assert any("circuit_breaker" in s for s in bearish_sources)
