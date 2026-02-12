"""Tests for cortex/data/tick_data.py and cortex/backtesting.py (Wave 10.2)."""

import numpy as np
import pytest

from cortex.data.tick_data import (
    aggregate_imbalance_bars,
    aggregate_tick_bars,
    aggregate_time_bars,
    aggregate_volume_bars,
    bars_to_returns,
    reconstruct_tick_prices,
)
from cortex.backtesting import (
    backtest_multi_horizon,
    backtest_var,
    christoffersen_test,
    kupiec_test,
    simple_var_forecast,
)


@pytest.fixture
def sample_swaps():
    """Swap records mimicking parsed DEX transactions."""
    base_time = 1700000000.0
    return [
        {"slot": 100, "block_time": base_time, "price": 100.0, "amount_in": 1.0, "token_in": "USDC", "dex": "raydium", "signature": "sig1"},
        {"slot": 101, "block_time": base_time + 1, "price": 101.0, "amount_in": 2.0, "token_in": "SOL", "dex": "raydium", "signature": "sig2"},
        {"slot": 102, "block_time": base_time + 2, "price": 99.5, "amount_in": 0.5, "token_in": "USDC", "dex": "orca", "signature": "sig3"},
        {"slot": 103, "block_time": base_time + 3, "price": 102.0, "amount_in": 3.0, "token_in": "SOL", "dex": "raydium", "signature": "sig4"},
        {"slot": 104, "block_time": base_time + 4, "price": 98.0, "amount_in": 1.5, "token_in": "USDC", "dex": "orca", "signature": "sig5"},
        {"slot": 105, "block_time": base_time + 5, "price": 103.0, "amount_in": 0.8, "token_in": "SOL", "dex": "raydium", "signature": "sig6"},
        {"slot": 106, "block_time": base_time + 6, "price": 97.0, "amount_in": 2.5, "token_in": "USDC", "dex": "orca", "signature": "sig7"},
        {"slot": 107, "block_time": base_time + 7, "price": 104.0, "amount_in": 1.2, "token_in": "SOL", "dex": "raydium", "signature": "sig8"},
        {"slot": 108, "block_time": base_time + 8, "price": 96.0, "amount_in": 0.3, "token_in": "USDC", "dex": "orca", "signature": "sig9"},
        {"slot": 109, "block_time": base_time + 9, "price": 105.0, "amount_in": 4.0, "token_in": "SOL", "dex": "raydium", "signature": "sig10"},
    ]


# ── reconstruct_tick_prices ──────────────────────────────────────────

class TestReconstructTickPrices:
    def test_empty_input(self):
        assert reconstruct_tick_prices([]) == []

    def test_skips_zero_price(self):
        swaps = [{"slot": 1, "block_time": 100, "price": 0.0, "amount_in": 1.0}]
        assert reconstruct_tick_prices(swaps) == []

    def test_basic_reconstruction(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        assert len(ticks) == 10
        assert ticks[0]["price"] == 100.0
        assert ticks[-1]["price"] == 105.0

    def test_sorted_by_slot(self, sample_swaps):
        reversed_swaps = list(reversed(sample_swaps))
        ticks = reconstruct_tick_prices(reversed_swaps)
        slots = [t["slot"] for t in ticks]
        assert slots == sorted(slots)

    def test_direction_detection(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        assert ticks[0]["direction"] == "buy"   # USDC in = buy
        assert ticks[1]["direction"] == "sell"   # SOL in = sell

    def test_tick_fields(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        for t in ticks:
            assert "timestamp" in t
            assert "price" in t
            assert "volume" in t
            assert "dex" in t
            assert "direction" in t


# ── aggregate_time_bars ──────────────────────────────────────────────

class TestAggregateTimeBars:
    def test_empty_input(self):
        assert aggregate_time_bars([]) == []

    def test_single_bar(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_time_bars(ticks, bar_seconds=100)
        assert len(bars) == 1
        assert bars[0]["open"] == 100.0
        assert bars[0]["n_ticks"] == 10

    def test_multiple_bars(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_time_bars(ticks, bar_seconds=3)
        assert len(bars) >= 2
        for b in bars:
            assert b["high"] >= b["low"]
            assert b["volume"] >= 0

    def test_max_bars_limit(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_time_bars(ticks, bar_seconds=1, max_bars=2)
        assert len(bars) <= 2

    def test_vwap_calculation(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_time_bars(ticks, bar_seconds=100)
        bar = bars[0]
        assert bar["low"] <= bar["vwap"] <= bar["high"]


# ── aggregate_volume_bars ────────────────────────────────────────────

class TestAggregateVolumeBars:
    def test_empty_input(self):
        assert aggregate_volume_bars([]) == []

    def test_volume_threshold(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_volume_bars(ticks, bar_volume=3.0)
        assert len(bars) >= 2
        for b in bars[:-1]:
            assert b["volume"] >= 3.0

    def test_max_bars_limit(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_volume_bars(ticks, bar_volume=1.0, max_bars=3)
        assert len(bars) <= 3


# ── aggregate_tick_bars ──────────────────────────────────────────────

class TestAggregateTickBars:
    def test_empty_input(self):
        assert aggregate_tick_bars([]) == []

    def test_fixed_count(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_tick_bars(ticks, ticks_per_bar=3)
        for b in bars[:-1]:
            assert b["n_ticks"] == 3

    def test_remainder_bar(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_tick_bars(ticks, ticks_per_bar=3)
        assert bars[-1]["n_ticks"] <= 3


# ── aggregate_imbalance_bars ─────────────────────────────────────────

class TestAggregateImbalanceBars:
    def test_empty_input(self):
        assert aggregate_imbalance_bars([]) == []

    def test_imbalance_field(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars = aggregate_imbalance_bars(ticks, threshold=2.0)
        for b in bars[:-1]:
            assert "imbalance" in b
            assert abs(b["imbalance"]) >= 2.0


# ── kupiec_test ──────────────────────────────────────────────────────

class TestKupiecTest:
    def test_no_observations(self):
        result = kupiec_test(0, 0, 95.0)
        assert result["pass"] is True

    def test_expected_violations(self):
        result = kupiec_test(100, 5, 95.0)
        assert result["pass"] is True
        assert abs(result["violation_rate"] - 0.05) < 1e-10

    def test_too_many_violations(self):
        result = kupiec_test(100, 30, 95.0)
        assert result["pass"] is False

    def test_no_violations(self):
        result = kupiec_test(100, 0, 95.0)
        assert result["violation_rate"] == 0.0

    def test_all_violations(self):
        result = kupiec_test(100, 100, 95.0)
        assert result["pass"] is False


# ── christoffersen_test ──────────────────────────────────────────────

class TestChristoffersenTest:
    def test_short_series(self):
        result = christoffersen_test(np.array([0, 1, 0]))
        assert result["pass"] is True

    def test_independent_violations(self):
        np.random.seed(42)
        v = (np.random.rand(200) < 0.05).astype(int)
        result = christoffersen_test(v)
        assert "p_value" in result
        assert "statistic" in result

    def test_clustered_violations(self):
        v = np.zeros(200, dtype=int)
        v[50:60] = 1
        v[150:160] = 1
        result = christoffersen_test(v)
        assert "p_value" in result


# ── backtest_var ─────────────────────────────────────────────────────

class TestBacktestVar:
    def test_short_series(self):
        result = backtest_var(np.array([1.0, -1.0]), np.array([-2.0, -2.0]), 95.0)
        assert result["n_obs"] == 2
        assert result["n_violations"] == 0

    def test_with_violations(self):
        np.random.seed(42)
        rets = np.random.randn(100)
        var_vals = np.full(100, -1.0)
        result = backtest_var(rets, var_vals, 95.0)
        assert result["n_obs"] == 100
        assert result["n_violations"] > 0
        assert "kupiec" in result
        assert "christoffersen" in result


# ── simple_var_forecast ──────────────────────────────────────────────

class TestSimpleVarForecast:
    def test_basic(self):
        np.random.seed(42)
        rets = np.random.randn(1000)
        var = simple_var_forecast(rets, 95.0)
        assert var < 0  # VaR should be negative for normal returns

    def test_confidence_ordering(self):
        np.random.seed(42)
        rets = np.random.randn(1000)
        var_95 = simple_var_forecast(rets, 95.0)
        var_99 = simple_var_forecast(rets, 99.0)
        assert var_99 < var_95  # 99% VaR more extreme


# ── backtest_multi_horizon ───────────────────────────────────────────

class TestBacktestMultiHorizon:
    def test_basic(self, sample_swaps):
        ticks = reconstruct_tick_prices(sample_swaps)
        bars_5 = aggregate_time_bars(ticks, bar_seconds=2)
        bars_10 = aggregate_time_bars(ticks, bar_seconds=5)
        bars_by_horizon = {5: bars_5, 10: bars_10}
        results = backtest_multi_horizon(bars_by_horizon, simple_var_forecast, 95.0)
        assert len(results) == 2
        for r in results:
            assert "horizon_minutes" in r
            assert "kupiec_pass" in r

    def test_empty_horizon(self):
        results = backtest_multi_horizon({60: []}, simple_var_forecast, 95.0)
        assert len(results) == 1
        assert results[0]["n_observations"] == 0

class TestBarsToReturns:
    def test_empty_bars(self):
        assert len(bars_to_returns([])) == 0

    def test_single_bar(self):
        assert len(bars_to_returns([{"close": 100.0}])) == 0

    def test_log_returns(self):
        bars = [{"close": 100.0}, {"close": 110.0}, {"close": 105.0}]
        rets = bars_to_returns(bars)
        assert len(rets) == 2
        assert rets[0] > 0  # 100 -> 110 = positive
        assert rets[1] < 0  # 110 -> 105 = negative

