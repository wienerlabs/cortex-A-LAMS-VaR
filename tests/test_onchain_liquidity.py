"""Tests for cortex/data/onchain_liquidity.py — on-chain liquidity analysis."""

import numpy as np
import pytest

from cortex.data.onchain_liquidity import (
    _compute_token_deltas,
    _identify_dex,
    build_liquidity_depth_curve,
    compute_realized_spread,
    fetch_swap_history,
    get_volume_weighted_spread,
    parse_swap_from_tx,
)


@pytest.fixture
def sample_tx():
    """Minimal Solana transaction with Raydium swap."""
    return {
        "slot": 250_000_000,
        "blockTime": 1700000000,
        "transaction": {
            "signatures": ["5abc123"],
            "message": {
                "accountKeys": [
                    "UserWallet111",
                    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
                ],
            },
        },
        "meta": {
            "err": None,
            "logMessages": ["Program 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8 invoke"],
            "preTokenBalances": [
                {"mint": "SOLmint", "owner": "UserWallet111", "uiTokenAmount": {"uiAmount": 10.0, "decimals": 9, "amount": "10000000000"}},
                {"mint": "USDCmint", "owner": "UserWallet111", "uiTokenAmount": {"uiAmount": 1000.0, "decimals": 6, "amount": "1000000000"}},
            ],
            "postTokenBalances": [
                {"mint": "SOLmint", "owner": "UserWallet111", "uiTokenAmount": {"uiAmount": 9.0, "decimals": 9, "amount": "9000000000"}},
                {"mint": "USDCmint", "owner": "UserWallet111", "uiTokenAmount": {"uiAmount": 1150.0, "decimals": 6, "amount": "1150000000"}},
            ],
            "preBalances": [5000000000, 0],
            "postBalances": [4999000000, 0],
        },
    }


@pytest.fixture
def sample_swaps():
    """List of parsed swap records for spread calculation."""
    return [
        {"slot": 100, "price": 150.0, "amount_in": 1.0, "dex": "raydium_amm_v4", "token_in": "A", "token_out": "B", "amount_out": 150.0, "block_time": 1000, "signature": "s1", "token_deltas": {}},
        {"slot": 101, "price": 150.5, "amount_in": 2.0, "dex": "raydium_amm_v4", "token_in": "A", "token_out": "B", "amount_out": 301.0, "block_time": 1001, "signature": "s2", "token_deltas": {}},
        {"slot": 102, "price": 149.8, "amount_in": 1.5, "dex": "orca_whirlpool", "token_in": "A", "token_out": "B", "amount_out": 224.7, "block_time": 1002, "signature": "s3", "token_deltas": {}},
        {"slot": 103, "price": 150.2, "amount_in": 3.0, "dex": "raydium_amm_v4", "token_in": "A", "token_out": "B", "amount_out": 450.6, "block_time": 1003, "signature": "s4", "token_deltas": {}},
        {"slot": 104, "price": 149.9, "amount_in": 0.5, "dex": "orca_whirlpool", "token_in": "A", "token_out": "B", "amount_out": 74.95, "block_time": 1004, "signature": "s5", "token_deltas": {}},
    ]


class TestIdentifyDex:
    def test_raydium_from_account_keys(self):
        tx = {"transaction": {"message": {"accountKeys": ["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"]}}}
        assert _identify_dex("", tx) == "raydium_amm_v4"

    def test_orca_from_account_keys(self):
        tx = {"transaction": {"message": {"accountKeys": ["whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"]}}}
        assert _identify_dex("", tx) == "orca_whirlpool"

    def test_raydium_from_logs(self):
        tx = {"transaction": {"message": {"accountKeys": []}}}
        assert _identify_dex("program raydium invoke", tx) == "raydium_amm_v4"

    def test_whirlpool_from_logs(self):
        tx = {"transaction": {"message": {"accountKeys": []}}}
        assert _identify_dex("whirlpool swap executed", tx) == "orca_whirlpool"

    def test_meteora_from_logs(self):
        tx = {"transaction": {"message": {"accountKeys": []}}}
        assert _identify_dex("meteora dlmm swap", tx) == "meteora_dlmm"

    def test_unknown_returns_none(self):
        tx = {"transaction": {"message": {"accountKeys": ["SomeRandomProgram"]}}}
        assert _identify_dex("no dex here", tx) is None


class TestComputeTokenDeltas:
    def test_basic_delta(self):
        pre = [{"mint": "A", "owner": "U", "uiTokenAmount": {"uiAmount": 10.0, "decimals": 6, "amount": "10000000"}}]
        post = [{"mint": "A", "owner": "U", "uiTokenAmount": {"uiAmount": 8.0, "decimals": 6, "amount": "8000000"}}]
        deltas = _compute_token_deltas(pre, post)
        assert "A" in deltas
        assert abs(deltas["A"] - (-2.0)) < 0.01

    def test_two_token_swap(self):
        pre = [
            {"mint": "SOL", "owner": "U", "uiTokenAmount": {"uiAmount": 10.0, "decimals": 9, "amount": "10000000000"}},
            {"mint": "USDC", "owner": "U", "uiTokenAmount": {"uiAmount": 0.0, "decimals": 6, "amount": "0"}},
        ]
        post = [
            {"mint": "SOL", "owner": "U", "uiTokenAmount": {"uiAmount": 9.0, "decimals": 9, "amount": "9000000000"}},
            {"mint": "USDC", "owner": "U", "uiTokenAmount": {"uiAmount": 150.0, "decimals": 6, "amount": "150000000"}},
        ]
        deltas = _compute_token_deltas(pre, post)
        assert len(deltas) == 2
        assert deltas["SOL"] < 0
        assert deltas["USDC"] > 0

    def test_empty_balances(self):
        assert _compute_token_deltas([], []) == {}


class TestParseSwapFromTx:
    def test_valid_swap(self, sample_tx):
        result = parse_swap_from_tx(sample_tx)
        assert result is not None
        assert result["dex"] == "raydium_amm_v4"
        assert result["slot"] == 250_000_000
        assert result["price"] > 0

    def test_failed_tx_returns_none(self, sample_tx):
        sample_tx["meta"]["err"] = {"InstructionError": [0, "Custom"]}
        assert parse_swap_from_tx(sample_tx) is None

    def test_non_dex_tx_returns_none(self):
        tx = {
            "slot": 100,
            "transaction": {"signatures": ["abc"], "message": {"accountKeys": ["SystemProgram"]}},
            "meta": {"err": None, "logMessages": ["transfer"], "preTokenBalances": [], "postTokenBalances": [], "preBalances": [100], "postBalances": [99]},
        }


class TestComputeRealizedSpread:
    def test_empty_swaps(self):
        result = compute_realized_spread([])
        assert result["realized_spread_pct"] == 0.0
        assert result["n_swaps"] == 0

    def test_consecutive_pair_spread(self, sample_swaps):
        result = compute_realized_spread(sample_swaps)
        assert result["realized_spread_pct"] > 0
        assert result["n_swaps"] == 5
        assert "by_dex" in result

    def test_with_reference_prices(self, sample_swaps):
        ref = {100: 150.0, 101: 150.0, 102: 150.0, 103: 150.0, 104: 150.0}
        result = compute_realized_spread(sample_swaps, reference_prices=ref)
        assert result["realized_spread_pct"] > 0
        assert result["n_swaps"] == 5

    def test_by_dex_breakdown(self, sample_swaps):
        result = compute_realized_spread(sample_swaps)
        by_dex = result["by_dex"]
        assert len(by_dex) > 0
        for dex_info in by_dex.values():
            assert "mean_spread_pct" in dex_info
            assert "n_swaps" in dex_info

    def test_single_swap_no_crash(self, sample_swaps):
        result = compute_realized_spread(sample_swaps[:1])
        assert result["n_swaps"] == 1


class TestVolumeWeightedSpread:
    def test_empty_swaps(self):
        result = get_volume_weighted_spread([])
        assert result["vwas_pct"] == 0.0
        assert result["n_swaps"] == 0

    def test_basic_vwas(self, sample_swaps):
        result = get_volume_weighted_spread(sample_swaps)
        assert result["vwas_pct"] >= 0
        assert result["total_volume"] > 0
        assert result["n_swaps"] == 5

    def test_single_swap(self, sample_swaps):
        result = get_volume_weighted_spread(sample_swaps[:1])
        assert result["vwas_pct"] == 0.0


class TestBuildLiquidityDepthCurve:
    def test_testing_mode_returns_empty(self):
        result = build_liquidity_depth_curve("SomePool123")
        assert result["pool"] == "SomePool123"
        assert result["ticks"] == []

    def test_returns_expected_keys(self):
        result = build_liquidity_depth_curve("SomePool123")
        assert "pool" in result
        assert "bid_depth" in result
        assert "ask_depth" in result


class TestFetchSwapHistory:
    def test_testing_mode_returns_empty(self):
        result = fetch_swap_history("SomeToken123")
        assert result == []


class TestLVaROnchainIntegration:
    def test_onchain_lvar_fallback_to_roll(self):
        from cortex.liquidity import liquidity_adjusted_var_with_onchain

        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(200) * 0.5)
        result = liquidity_adjusted_var_with_onchain(
            var_value=-3.0,
            prices=prices,
            position_value=100_000,
        )
        assert result["lvar"] < 0
        assert result["spread_source"] in ("roll", "default")
        assert result["base_var"] == -3.0

    def test_onchain_lvar_default_fallback(self):
        from cortex.liquidity import liquidity_adjusted_var_with_onchain

        result = liquidity_adjusted_var_with_onchain(var_value=-2.5)
        assert result["spread_source"] == "default"
        assert result["lvar"] < 0

