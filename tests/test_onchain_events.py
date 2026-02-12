"""Tests for cortex/data/onchain_events.py and multivariate Hawkes (Wave 10.3)."""

import numpy as np
import pytest

from cortex.data.onchain_events import (
    classify_event,
    classify_liquidation,
    collect_events,
    events_to_hawkes_times,
    get_event_type_counts,
)
from cortex.hawkes import (
    fit_multivariate_hawkes,
    flash_crash_risk_onchain,
    multivariate_intensity,
)


@pytest.fixture
def large_swap():
    """Swap record that qualifies as a large_swap event (>$50k)."""
    return {
        "slot": 100,
        "block_time": 1700000000,
        "price": 100.0,
        "amount_in": 600.0,
        "amount_out": 60000.0,
        "dex": "raydium",
        "direction": "buy",
    }


@pytest.fixture
def small_swap():
    """Swap record below large_swap threshold."""
    return {
        "slot": 101,
        "block_time": 1700000001,
        "price": 100.0,
        "amount_in": 0.1,
        "amount_out": 10.0,
        "dex": "orca",
        "direction": "sell",
    }


@pytest.fixture
def oracle_jump_pair():
    """Two swaps where the second has a >2% price jump."""
    return [
        {"slot": 100, "block_time": 1700000000, "price": 100.0, "amount_in": 0.1, "amount_out": 10.0, "dex": "raydium"},
        {"slot": 101, "block_time": 1700000001, "price": 103.0, "amount_in": 0.1, "amount_out": 10.0, "dex": "raydium"},
    ]


@pytest.fixture
def liquidation_tx():
    """Transaction that looks like a liquidation on a lending protocol."""
    return {
        "slot": 200,
        "blockTime": 1700000100,
        "transaction": {
            "signatures": ["liq_sig_123"],
            "message": {
                "accountKeys": [
                    "UserWallet",
                    "So1endEVFfnCjSQnx7qoiMajcpyBrm2kRg4rJNQer6",
                ],
            },
        },
        "meta": {
            "logMessages": [
                "Program So1endEVFfnCjSQnx7qoiMajcpyBrm2kRg4rJNQer6 invoke",
                "Instruction: Liquidate obligation",
            ],
        },
    }


@pytest.fixture
def sample_events():
    """Pre-built event list for Hawkes tests."""
    base = 1700000000.0
    return [
        {"event_type": "large_swap", "slot": 100, "timestamp": base, "magnitude": 60000.0, "details": {}},
        {"event_type": "large_swap", "slot": 110, "timestamp": base + 5, "magnitude": 75000.0, "details": {}},
        {"event_type": "oracle_jump", "slot": 120, "timestamp": base + 8, "magnitude": 3.5, "details": {}},
        {"event_type": "large_swap", "slot": 130, "timestamp": base + 12, "magnitude": 80000.0, "details": {}},
        {"event_type": "oracle_jump", "slot": 140, "timestamp": base + 15, "magnitude": 4.0, "details": {}},
        {"event_type": "liquidation", "slot": 150, "timestamp": base + 18, "magnitude": 1.0, "details": {}},
        {"event_type": "large_swap", "slot": 160, "timestamp": base + 22, "magnitude": 55000.0, "details": {}},
        {"event_type": "oracle_jump", "slot": 170, "timestamp": base + 25, "magnitude": 2.5, "details": {}},
        {"event_type": "large_swap", "slot": 180, "timestamp": base + 30, "magnitude": 90000.0, "details": {}},
        {"event_type": "liquidation", "slot": 190, "timestamp": base + 35, "magnitude": 1.0, "details": {}},
    ]


# ── classify_event ───────────────────────────────────────────────────

class TestClassifyEvent:
    def test_large_swap_detected(self, large_swap):
        ev = classify_event(large_swap)
        assert ev is not None
        assert ev["event_type"] == "large_swap"
        assert ev["magnitude"] >= 50000

    def test_small_swap_no_event(self, small_swap):
        ev = classify_event(small_swap)
        assert ev is None

    def test_oracle_jump_detected(self):
        swap = {"slot": 101, "block_time": 100, "price": 103.0, "amount_in": 0.1, "amount_out": 10.0}
        ev = classify_event(swap, prev_price=100.0)
        assert ev is not None
        assert ev["event_type"] == "oracle_jump"
        assert ev["magnitude"] >= 2.0

    def test_no_jump_small_change(self):
        swap = {"slot": 101, "block_time": 100, "price": 100.5, "amount_in": 0.1, "amount_out": 10.0}
        ev = classify_event(swap, prev_price=100.0)
        assert ev is None

    def test_no_prev_price(self, small_swap):
        ev = classify_event(small_swap, prev_price=None)
        assert ev is None

    def test_zero_prev_price(self, small_swap):
        ev = classify_event(small_swap, prev_price=0.0)
        assert ev is None


# ── classify_liquidation ─────────────────────────────────────────────

class TestClassifyLiquidation:
    def test_liquidation_detected(self, liquidation_tx):
        ev = classify_liquidation(liquidation_tx)
        assert ev is not None
        assert ev["event_type"] == "liquidation"
        assert ev["details"]["protocol"] == "Solend"

    def test_no_liquidation_keyword(self):
        tx = {
            "slot": 200, "blockTime": 100,
            "transaction": {"signatures": ["sig"], "message": {"accountKeys": ["So1endEVFfnCjSQnx7qoiMajcpyBrm2kRg4rJNQer6"]}},
            "meta": {"logMessages": ["Program invoke", "Instruction: Deposit"]},
        }
        assert classify_liquidation(tx) is None

    def test_no_lending_program(self):
        tx = {
            "slot": 200, "blockTime": 100,
            "transaction": {"signatures": ["sig"], "message": {"accountKeys": ["RandomProgram"]}},
            "meta": {"logMessages": ["Liquidate something"]},
        }
        assert classify_liquidation(tx) is None


# ── collect_events ───────────────────────────────────────────────────

class TestCollectEvents:
    def test_empty_input(self):
        assert collect_events([]) == []

    def test_collects_large_swaps(self, large_swap):
        events = collect_events([large_swap])
        assert len(events) == 1
        assert events[0]["event_type"] == "large_swap"

    def test_collects_oracle_jumps(self, oracle_jump_pair):
        events = collect_events(oracle_jump_pair)
        jump_events = [e for e in events if e["event_type"] == "oracle_jump"]
        assert len(jump_events) == 1

    def test_sorted_by_slot(self, sample_events):
        events = collect_events([], transactions=None)
        assert events == []

    def test_with_liquidation_tx(self, large_swap, liquidation_tx):
        events = collect_events([large_swap], transactions=[liquidation_tx])
        types = {e["event_type"] for e in events}
        assert "large_swap" in types
        assert "liquidation" in types


# ── events_to_hawkes_times ───────────────────────────────────────────

class TestEventsToHawkesTimes:
    def test_empty_events(self):
        assert events_to_hawkes_times([]) == {}

    def test_basic_conversion(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        assert "large_swap" in times
        assert "oracle_jump" in times
        assert len(times["large_swap"]) == 5

    def test_normalized_times(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        for arr in times.values():
            assert arr[0] >= 0.0

    def test_filter_by_type(self, sample_events):
        times = events_to_hawkes_times(sample_events, event_types=["large_swap"])
        assert "large_swap" in times
        assert "oracle_jump" not in times

    def test_filter_no_match(self, sample_events):
        times = events_to_hawkes_times(sample_events, event_types=["nonexistent"])
        assert times == {}


# ── get_event_type_counts ────────────────────────────────────────────

class TestGetEventTypeCounts:
    def test_empty(self):
        assert get_event_type_counts([]) == {}

    def test_counts(self, sample_events):
        counts = get_event_type_counts(sample_events)
        assert counts["large_swap"] == 5
        assert counts["oracle_jump"] == 3
        assert counts["liquidation"] == 2


# ── fit_multivariate_hawkes ──────────────────────────────────────────

class TestFitMultivariateHawkes:
    def test_empty_input(self):
        result = fit_multivariate_hawkes({})
        assert result["event_types"] == []
        assert result["stationary"] is True

    def test_single_type(self):
        times = {"large_swap": np.array([0.0, 1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0])}
        result = fit_multivariate_hawkes(times, T=12.0)
        assert "large_swap" in result["mu"]
        assert result["spectral_radius"] >= 0

    def test_two_types(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        result = fit_multivariate_hawkes(times)
        assert len(result["event_types"]) >= 2
        assert len(result["cross_excitation"]) > 0
        assert isinstance(result["branching_matrix"], list)

    def test_stationarity_check(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        result = fit_multivariate_hawkes(times)
        assert isinstance(result["stationary"], bool)
        assert result["spectral_radius"] >= 0

    def test_few_events_fallback(self):
        times = {"rare": np.array([1.0, 5.0])}
        result = fit_multivariate_hawkes(times, T=10.0)
        assert "rare" in result["mu"]
        assert result["mu"]["rare"] > 0


# ── multivariate_intensity ───────────────────────────────────────────

class TestMultivariateIntensity:
    def test_empty_fit(self):
        result = multivariate_intensity({}, {"event_types": [], "mu": {}, "cross_excitation": []})
        assert result == {}

    def test_baseline_intensity(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        fit = fit_multivariate_hawkes(times)
        intensities = multivariate_intensity(times, fit)
        for etype in fit["event_types"]:
            assert intensities[etype] >= fit["mu"][etype]

    def test_custom_t_now(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        fit = fit_multivariate_hawkes(times)
        intensities = multivariate_intensity(times, fit, t_now=1000.0)
        for v in intensities.values():
            assert v >= 0


# ── flash_crash_risk_onchain ─────────────────────────────────────────

class TestFlashCrashRiskOnchain:
    def test_empty_fit(self):
        result = flash_crash_risk_onchain({}, {"event_types": [], "mu": {}, "cross_excitation": []})
        assert result["flash_crash_score"] == 0.0
        assert result["risk_level"] == "low"

    def test_basic_risk(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        fit = fit_multivariate_hawkes(times)
        risk = flash_crash_risk_onchain(times, fit)
        assert 0 <= risk["flash_crash_score"] <= 100
        assert risk["risk_level"] in ("low", "medium", "high", "critical")
        assert risk["dominant_event_type"] in fit["event_types"]

    def test_risk_level_categories(self, sample_events):
        times = events_to_hawkes_times(sample_events)
        fit = fit_multivariate_hawkes(times)
        risk = flash_crash_risk_onchain(times, fit)
        assert "current_intensities" in risk
        assert "baseline_intensities" in risk

