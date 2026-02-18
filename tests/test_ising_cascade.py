"""Tests for cortex/ising_cascade.py — DX-Research Task 5: Ising Cascade Detection."""
from unittest.mock import patch

import pytest

from cortex.ising_cascade import (
    CascadeResult,
    _cascade_score,
    _compute_effective_temperature,
    _compute_magnetization,
    _compute_susceptibility,
    _risk_level,
    detect_cascade,
    get_cascade_score,
)
import cortex.stigmergy as stigmergy_module
from cortex.stigmergy import PheromoneSignal, get_board


@pytest.fixture(autouse=True)
def reset_stigmergy_board():
    """Reset the global stigmergy board between tests."""
    old = stigmergy_module._board
    stigmergy_module._board = None
    yield
    stigmergy_module._board = old


class TestComputeMagnetization:
    def test_no_sources_returns_zero(self):
        m, d = _compute_magnetization(0.5, "bullish", 0, False)
        assert m == 0.0
        assert d == "neutral"

    def test_single_source_bullish(self):
        m, d = _compute_magnetization(0.8, "bullish", 1, False)
        assert m > 0.0
        assert d == "bullish"

    def test_more_sources_stronger_magnetization(self):
        m1, _ = _compute_magnetization(0.8, "bullish", 1, False)
        m5, _ = _compute_magnetization(0.8, "bullish", 5, False)
        assert m5 > m1

    def test_swarm_amplification(self):
        m_no, _ = _compute_magnetization(0.6, "bullish", 3, False)
        m_yes, _ = _compute_magnetization(0.6, "bullish", 3, True)
        assert m_yes > m_no

    def test_magnetization_capped_at_one(self):
        m, _ = _compute_magnetization(1.0, "bullish", 10, True)
        assert m <= 1.0

    def test_bearish_direction(self):
        m, d = _compute_magnetization(0.9, "bearish", 3, True)
        assert d == "bearish"
        assert m > 0.5

    def test_neutral_direction(self):
        _, d = _compute_magnetization(0.0, "neutral", 2, False)
        assert d == "neutral"


class TestComputeEffectiveTemperature:
    def test_calm_regime_high_temperature(self):
        t = _compute_effective_temperature(regime=0, max_regimes=5)
        assert t > 0.8  # Calm regime → high temperature

    def test_crisis_regime_low_temperature(self):
        t = _compute_effective_temperature(regime=4, max_regimes=5)
        assert t < 0.7  # Crisis regime → low temperature

    def test_high_hawkes_lowers_temperature(self):
        t_low = _compute_effective_temperature(regime=2, hawkes_intensity_ratio=1.0)
        t_high = _compute_effective_temperature(regime=2, hawkes_intensity_ratio=3.0)
        assert t_high < t_low

    def test_temperature_bounded(self):
        t_min = _compute_effective_temperature(regime=4, hawkes_intensity_ratio=5.0)
        t_max = _compute_effective_temperature(regime=0, hawkes_intensity_ratio=0.5)
        assert t_min >= 0.1
        assert t_max <= 2.0

    def test_baseline_hawkes_no_effect(self):
        t = _compute_effective_temperature(regime=0, hawkes_intensity_ratio=1.0)
        assert 0.9 <= t <= 1.1  # Near 1.0 with calm regime and baseline Hawkes


class TestComputeSusceptibility:
    def test_low_magnetization_high_susceptibility(self):
        chi = _compute_susceptibility(0.1, 1.0)
        assert chi > 0.5  # System is responsive

    def test_high_magnetization_low_susceptibility(self):
        chi = _compute_susceptibility(0.99, 1.0)
        assert chi < 0.1  # System is saturated

    def test_low_temperature_amplifies(self):
        chi_high_t = _compute_susceptibility(0.5, 1.0)
        chi_low_t = _compute_susceptibility(0.5, 0.2)
        assert chi_low_t > chi_high_t

    def test_susceptibility_capped(self):
        chi = _compute_susceptibility(0.0, 0.001)
        assert chi <= 100.0


class TestCascadeScore:
    def test_zero_inputs(self):
        score = _cascade_score(0.0, 1.0, 1.0)
        assert score >= 0.0
        assert score < 20.0  # Low score with no magnetization

    def test_full_herding_crisis(self):
        score = _cascade_score(1.0, 0.1, 50.0)
        assert score > 80.0  # Critical cascade

    def test_magnetization_dominates(self):
        score_low = _cascade_score(0.2, 0.5, 1.0)
        score_high = _cascade_score(0.9, 0.5, 1.0)
        assert score_high > score_low

    def test_temperature_contribution(self):
        score_warm = _cascade_score(0.5, 0.9, 1.0)
        score_cold = _cascade_score(0.5, 0.2, 1.0)
        assert score_cold > score_warm

    def test_score_capped_at_100(self):
        score = _cascade_score(1.0, 0.0, 100.0)
        assert score <= 100.0


class TestRiskLevel:
    def test_low(self):
        assert _risk_level(20.0) == "low"

    def test_medium(self):
        assert _risk_level(50.0) == "medium"

    def test_high(self):
        assert _risk_level(65.0) == "high"

    def test_critical(self):
        assert _risk_level(85.0) == "critical"


class TestDetectCascade:
    def test_no_herding_low_score(self):
        r = detect_cascade(
            stigmergy_conviction=0.0,
            stigmergy_direction="neutral",
            num_sources=0,
        )
        assert r.cascade_score < 20.0
        assert r.cascade_risk == "low"
        assert r.herding_direction == "neutral"

    def test_moderate_herding(self):
        r = detect_cascade(
            stigmergy_conviction=0.6,
            stigmergy_direction="bearish",
            num_sources=3,
            swarm_active=True,
            regime=2,
        )
        assert r.cascade_score > 30.0
        assert r.magnetization > 0.3

    def test_strong_herding_crisis(self):
        r = detect_cascade(
            stigmergy_conviction=0.95,
            stigmergy_direction="bearish",
            num_sources=7,
            swarm_active=True,
            regime=4,
            hawkes_intensity_ratio=3.0,
        )
        assert r.cascade_score > 70.0
        assert r.cascade_risk in ("high", "critical")
        assert r.herding_direction == "bearish"

    def test_result_to_dict(self):
        r = detect_cascade(
            stigmergy_conviction=0.5,
            stigmergy_direction="bullish",
            num_sources=2,
        )
        d = r.to_dict()
        assert "magnetization" in d
        assert "cascade_score" in d
        assert "components" in d
        assert "stigmergy_conviction" in d["components"]

    def test_high_hawkes_amplifies(self):
        r_base = detect_cascade(
            stigmergy_conviction=0.7,
            stigmergy_direction="bearish",
            num_sources=4,
            swarm_active=True,
            regime=2,
            hawkes_intensity_ratio=1.0,
        )
        r_spike = detect_cascade(
            stigmergy_conviction=0.7,
            stigmergy_direction="bearish",
            num_sources=4,
            swarm_active=True,
            regime=2,
            hawkes_intensity_ratio=4.0,
        )
        assert r_spike.cascade_score > r_base.cascade_score

    def test_regime_effect(self):
        r_calm = detect_cascade(
            stigmergy_conviction=0.7,
            stigmergy_direction="bullish",
            num_sources=4,
            swarm_active=True,
            regime=0,
        )
        r_crisis = detect_cascade(
            stigmergy_conviction=0.7,
            stigmergy_direction="bullish",
            num_sources=4,
            swarm_active=True,
            regime=4,
        )
        assert r_crisis.cascade_score > r_calm.cascade_score


class TestGetCascadeScore:
    def test_disabled_returns_zero(self):
        with patch("cortex.ising_cascade.ISING_CASCADE_ENABLED", False):
            r = get_cascade_score("SOL")
            assert r.cascade_score == 0.0
            assert r.cascade_risk == "low"

    def test_with_stigmergy_data(self):
        board = get_board()
        board.deposit(PheromoneSignal(source="a1", token="SOL", direction="bearish", strength=0.9))
        board.deposit(PheromoneSignal(source="a2", token="SOL", direction="bearish", strength=0.8))
        board.deposit(PheromoneSignal(source="a3", token="SOL", direction="bearish", strength=0.7))

        with patch("cortex.ising_cascade.ISING_CASCADE_ENABLED", True):
            r = get_cascade_score("SOL", regime=3, hawkes_intensity_ratio=2.0)
            assert r.cascade_score > 30.0
            assert r.herding_direction == "bearish"

    def test_no_signals_low_score(self):
        with patch("cortex.ising_cascade.ISING_CASCADE_ENABLED", True):
            r = get_cascade_score("UNKNOWN_TOKEN")
            assert r.cascade_score < 15.0


class TestDebateIntegration:
    """Test that Ising cascade evidence flows into the debate system."""

    def test_cascade_evidence_in_debate(self):
        """High cascade risk should add bearish evidence to debate."""
        board = get_board()
        for i in range(5):
            board.deposit(PheromoneSignal(
                source=f"analyst_{i}", token="SOL",
                direction="bearish", strength=0.9,
            ))

        from cortex.debate import run_debate

        r = run_debate(
            risk_score=50.0,
            component_scores=[{"component": "vol", "score": 50}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            token="SOL",
            alams_data={"current_regime": 3},
            enrich=False,
        )
        # Should have Ising cascade evidence in bearish
        bearish_sources = [e["source"] for e in r["evidence_summary"]["bearish_items"]]
        assert any("ising" in s for s in bearish_sources) or r["evidence_summary"]["bearish"] >= 2

    def test_no_cascade_without_token(self):
        """Without token, no Ising cascade evidence."""
        from cortex.debate import run_debate

        r = run_debate(
            risk_score=30.0,
            component_scores=[{"component": "vol", "score": 30}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            enrich=False,
        )
        bearish_sources = [e["source"] for e in r["evidence_summary"].get("bearish_items", [])]
        assert not any("ising" in s for s in bearish_sources)
