#!/usr/bin/env python3
"""
Deep Strategy Analysis - Before 365-Day Collection Decision.

PARTS:
1. Volatility Analysis (2-day OHLCV)
2. Alternative Token Pair Analysis (RAY, ORCA, BONK, JUP)
3. Flash Opportunity Detection
4. Fee Optimization Scenarios
5. Strategy Comparison & Final Report
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# API Keys
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")

# Token addresses
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
}

# Fee structures
DEX_FEES = {"Raydium": 0.0025, "Orca": 0.003, "Phoenix": 0.001, "Meteora": 0.002, "Jupiter": 0.0}
CEX_FEES = {
    "Binance": {"trading": 0.001, "maker": 0.0, "withdrawal_sol": 0.01},
    "Coinbase": {"trading": 0.006, "maker": 0.004, "withdrawal_sol": 0.0},
    "Kraken": {"trading": 0.0026, "maker": 0.0016, "withdrawal_sol": 0.0025},
    "OKX": {"trading": 0.001, "maker": 0.0008, "withdrawal_sol": 0.008},
}
SOLANA_TX_FEE = 0.000105  # ~$0.013 at $125


def print_section(title: str):
    print("\n" + "━" * 70)
    print(f"  {title}")
    print("━" * 70)


def print_subsection(title: str):
    print(f"\n▸ {title}")


async def make_request(client: httpx.AsyncClient, method: str, url: str,
                       headers: dict = None, timeout: int = 30) -> tuple:
    try:
        resp = await client.get(url, headers=headers, timeout=timeout) if method == "GET" else None
        if resp and resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"HTTP {resp.status_code if resp else 'N/A'}"}
    except Exception as e:
        return False, {"error": str(e)}


# =============================================================================
# PART 1: VOLATILITY ANALYSIS
# =============================================================================

async def analyze_volatility(client: httpx.AsyncClient) -> dict:
    """Analyze 2-day volatility patterns."""
    print_section("PART 1: VOLATILITY ANALYSIS (2-Day SOL/USDC)")

    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
    now = int(datetime.now(timezone.utc).timestamp())
    time_from = now - (48 * 3600)

    url = (f"https://public-api.birdeye.so/defi/ohlcv?"
           f"address={TOKENS['SOL']}&type=1H&time_from={time_from}&time_to={now}")

    success, data = await make_request(client, "GET", url, headers=headers)

    if not success or not data.get("success"):
        print(f"   ❌ Failed to fetch OHLCV: {data}")
        return {}

    items = data.get("data", {}).get("items", [])
    if not items:
        print("   ❌ No OHLCV data")
        return {}

    df = pd.DataFrame(items)

    # Normalize column names (Birdeye uses 'c' for close, 'o' for open, etc.)
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=col_map)

    # Handle different timestamp formats
    if "unixTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["unixTime"], unit="s", utc=True)
    elif "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        # Try first column as timestamp
        df["datetime"] = pd.to_datetime(df.iloc[:, 0], unit="s", utc=True)

    df["hour"] = df["datetime"].dt.hour
    df["returns"] = df["close"].pct_change() * 100
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"]) * 100
    df["volatility"] = df["returns"].abs()

    print(f"\n   📊 Data: {len(df)} hourly candles")
    print(f"   📅 Period: {df['datetime'].min()} to {df['datetime'].max()}")

    # 1. Highest volatility hours
    print_subsection("Highest Volatility Hours (UTC)")
    hourly_vol = df.groupby("hour")["volatility"].mean().sort_values(ascending=False)
    for hour, vol in hourly_vol.head(5).items():
        print(f"      {hour:02d}:00 UTC - Avg volatility: {vol:.3f}%")

    # 2. Biggest price movements
    print_subsection("Biggest Price Movements")

    # Biggest pump
    max_pump_idx = df["returns"].idxmax()
    if pd.notna(max_pump_idx):
        pump = df.loc[max_pump_idx]
        print(f"   🚀 Biggest Pump: +{pump['returns']:.2f}% @ {pump['datetime']}")
        print(f"      Price: ${pump['open']:.2f} → ${pump['close']:.2f}")

    # Biggest dump
    max_dump_idx = df["returns"].idxmin()
    if pd.notna(max_dump_idx):
        dump = df.loc[max_dump_idx]
        print(f"   📉 Biggest Dump: {dump['returns']:.2f}% @ {dump['datetime']}")
        print(f"      Price: ${dump['open']:.2f} → ${dump['close']:.2f}")

    # 3. Range analysis
    print_subsection("Price Range Analysis")
    avg_range = df["range_pct"].mean()
    max_range = df["range_pct"].max()
    print(f"   Avg Hourly Range: {avg_range:.3f}%")
    print(f"   Max Hourly Range: {max_range:.3f}%")

    # Is range enough for arb?
    min_arb_spread = 0.60  # DEX-DEX minimum
    profitable_hours = len(df[df["range_pct"] > min_arb_spread])
    print(f"   Hours with range > {min_arb_spread}%: {profitable_hours}/{len(df)} ({profitable_hours/len(df)*100:.1f}%)")

    return {
        "df": df,
        "hourly_volatility": hourly_vol.to_dict(),
        "avg_range_pct": avg_range,
        "max_range_pct": max_range,
        "profitable_hours_ratio": profitable_hours / len(df) if len(df) > 0 else 0
    }


# =============================================================================
# PART 2: ALTERNATIVE TOKEN PAIR ANALYSIS
# =============================================================================

async def analyze_altcoin_pairs(client: httpx.AsyncClient) -> dict:
    """Analyze spreads for alternative token pairs."""
    print_section("PART 2: ALTERNATIVE TOKEN PAIR ANALYSIS")

    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
    jup_headers = {"Content-Type": "application/json"}
    if JUPITER_API_KEY:
        jup_headers["x-api-key"] = JUPITER_API_KEY

    results = {}
    pairs = ["RAY", "ORCA", "BONK", "JUP"]

    for token in pairs:
        print_subsection(f"{token}/USDC Analysis")
        await asyncio.sleep(1.5)  # Rate limit

        token_addr = TOKENS.get(token)
        if not token_addr:
            continue

        # Get Birdeye price
        url = f"https://public-api.birdeye.so/defi/price?address={token_addr}"
        success, data = await make_request(client, "GET", url, headers=headers)
        birdeye_price = data.get("data", {}).get("value", 0) if success and data.get("success") else 0

        # Get Jupiter quote - Use small amount to minimize slippage
        # Token decimals: RAY=6, ORCA=6, BONK=5, JUP=6
        decimals = {"RAY": 6, "ORCA": 6, "BONK": 5, "JUP": 6}
        token_decimals = decimals.get(token, 6)
        amount = 10 ** token_decimals  # 1 token

        jup_url = f"https://api.jup.ag/ultra/v1/order?inputMint={token_addr}&outputMint={TOKENS['USDC']}&amount={amount}"
        success, jdata = await make_request(client, "GET", jup_url, headers=jup_headers)

        jup_price = 0
        if success and jdata.get("outAmount"):
            out_usdc = int(jdata.get("outAmount", 0)) / 1_000_000  # USDC has 6 decimals
            in_tokens = amount / (10 ** token_decimals)  # 1 token
            jup_price = out_usdc / in_tokens if in_tokens > 0 else 0

        # Get CEX price (Binance)
        cex_price = 0
        try:
            cex_url = f"https://api.binance.com/api/v3/ticker/price?symbol={token}USDT"
            success, cdata = await make_request(client, "GET", cex_url)
            if success:
                cex_price = float(cdata.get("price", 0))
        except:
            pass

        # Calculate spreads
        dex_dex_spread = abs(birdeye_price - jup_price) / min(birdeye_price, jup_price) * 100 if birdeye_price and jup_price else 0
        cex_dex_spread = abs(cex_price - jup_price) / min(cex_price, jup_price) * 100 if cex_price and jup_price else 0

        # Estimate liquidity from Jupiter
        liquidity = "Low" if token in ["BONK"] else "Medium" if token in ["JUP", "ORCA"] else "High"

        results[token] = {
            "birdeye_price": birdeye_price,
            "jupiter_price": jup_price,
            "binance_price": cex_price,
            "dex_dex_spread": dex_dex_spread,
            "cex_dex_spread": cex_dex_spread,
            "liquidity": liquidity,
            "profitable_after_fees": dex_dex_spread > 0.6 or cex_dex_spread > 0.4
        }

        print(f"   Birdeye: ${birdeye_price:.6f}")
        print(f"   Jupiter: ${jup_price:.6f}")
        print(f"   Binance: ${cex_price:.6f}" if cex_price else "   Binance: N/A")
        print(f"   DEX-DEX Spread: {dex_dex_spread:.4f}%")
        print(f"   CEX-DEX Spread: {cex_dex_spread:.4f}%")
        print(f"   Liquidity: {liquidity}")
        print(f"   Profitable: {'✅ YES' if results[token]['profitable_after_fees'] else '❌ NO'}")

    # Find best pairs
    print_subsection("Best Pairs Ranking")
    sorted_pairs = sorted(results.items(), key=lambda x: max(x[1]["dex_dex_spread"], x[1]["cex_dex_spread"]), reverse=True)
    for i, (token, data) in enumerate(sorted_pairs[:3], 1):
        best_spread = max(data["dex_dex_spread"], data["cex_dex_spread"])
        print(f"   #{i} {token}/USDC - Best spread: {best_spread:.4f}%")

    return results


# =============================================================================
# PART 3: FLASH OPPORTUNITY DETECTION
# =============================================================================

async def detect_flash_opportunities(client: httpx.AsyncClient, vol_data: dict) -> dict:
    """Detect micro-spreads and whale effects in historical data."""
    print_section("PART 3: FLASH OPPORTUNITY DETECTION")

    df = vol_data.get("df", pd.DataFrame())
    if df.empty:
        print("   ❌ No data for flash analysis")
        return {}

    # 1. Micro-spread windows (high volatility = potential opportunity)
    print_subsection("High Volatility Windows (Potential Flash Opportunities)")

    df["vol_zscore"] = (df["volatility"] - df["volatility"].mean()) / df["volatility"].std()
    high_vol_periods = df[df["vol_zscore"] > 1.5]  # 1.5 std above mean

    print(f"   High volatility periods found: {len(high_vol_periods)}")
    for _, row in high_vol_periods.head(5).iterrows():
        print(f"      {row['datetime']} - Volatility: {row['volatility']:.3f}% (z={row['vol_zscore']:.2f})")

    # 2. Volume spike analysis (whale indicator)
    print_subsection("Volume Spike Analysis (Whale Indicator)")

    df["volume_ma"] = df["volume"].rolling(window=6).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    whale_spikes = df[df["volume_ratio"] > 2.0]  # 2x normal volume

    print(f"   Whale spikes detected (>2x avg volume): {len(whale_spikes)}")

    # Check if whale spikes correlate with price movement
    whale_with_movement = whale_spikes[whale_spikes["volatility"] > df["volatility"].mean()]
    print(f"   Spikes with significant price movement: {len(whale_with_movement)}")

    for _, row in whale_with_movement.head(3).iterrows():
        direction = "🚀 UP" if row["returns"] > 0 else "📉 DOWN"
        print(f"      {row['datetime']} - Volume {row['volume_ratio']:.1f}x, Price {direction} {abs(row['returns']):.2f}%")

    # 3. Consecutive movement patterns
    print_subsection("Momentum Patterns (Consecutive Moves)")

    df["direction"] = np.sign(df["returns"])
    df["streak"] = (df["direction"] != df["direction"].shift()).cumsum()
    streaks = df.groupby("streak").agg({
        "returns": ["count", "sum"],
        "datetime": "first"
    })
    streaks.columns = ["length", "total_return", "start_time"]
    long_streaks = streaks[streaks["length"] >= 3].sort_values("length", ascending=False)

    print(f"   Momentum streaks (≥3 consecutive): {len(long_streaks)}")
    for _, row in long_streaks.head(3).iterrows():
        direction = "🚀 Bullish" if row["total_return"] > 0 else "📉 Bearish"
        print(f"      {row['start_time']} - {int(row['length'])} hours, {direction} {abs(row['total_return']):.2f}%")

    return {
        "high_vol_periods": len(high_vol_periods),
        "whale_spikes": len(whale_spikes),
        "whale_with_movement": len(whale_with_movement),
        "momentum_streaks": len(long_streaks),
        "df_enriched": df
    }


# =============================================================================
# PART 4: FEE OPTIMIZATION SCENARIOS
# =============================================================================

def analyze_fee_scenarios(sol_price: float = 125.0) -> dict:
    """Analyze different fee scenarios for break-even."""
    print_section("PART 4: FEE OPTIMIZATION SCENARIOS")

    results = {}

    # Scenario 1: Standard fees
    print_subsection("Scenario 1: Standard Taker Fees")
    std_dex_dex = (DEX_FEES["Raydium"] + DEX_FEES["Orca"] + SOLANA_TX_FEE) * 100
    std_cex_dex = (CEX_FEES["Binance"]["trading"] + DEX_FEES["Raydium"] +
                   CEX_FEES["Binance"]["withdrawal_sol"] * sol_price / 100 + SOLANA_TX_FEE) * 100
    print(f"   DEX-DEX (Raydium → Orca): {std_dex_dex:.3f}%")
    print(f"   CEX-DEX (Binance → Raydium): {std_cex_dex:.3f}%")
    results["standard"] = {"dex_dex": std_dex_dex, "cex_dex": std_cex_dex}

    # Scenario 2: Maker orders on CEX
    print_subsection("Scenario 2: Maker Orders on CEX (0% or reduced fee)")
    maker_cex_dex = (CEX_FEES["Binance"]["maker"] + DEX_FEES["Raydium"] +
                    CEX_FEES["Binance"]["withdrawal_sol"] * sol_price / 100 + SOLANA_TX_FEE) * 100
    print(f"   CEX-DEX (Binance Maker → Raydium): {maker_cex_dex:.3f}%")
    print(f"   Savings vs taker: {(std_cex_dex - maker_cex_dex):.3f}%")
    results["maker"] = {"cex_dex": maker_cex_dex}

    # Scenario 3: No priority fee
    print_subsection("Scenario 3: Zero Priority Fee (slower execution)")
    no_priority = SOLANA_TX_FEE - 0.0001  # Remove priority fee
    nopri_dex_dex = (DEX_FEES["Raydium"] + DEX_FEES["Orca"] + no_priority) * 100
    print(f"   DEX-DEX: {nopri_dex_dex:.3f}%")
    print(f"   ⚠️ Warning: May miss opportunities due to slower execution")
    results["no_priority"] = {"dex_dex": nopri_dex_dex}

    # Scenario 4: Phoenix (lowest fee DEX)
    print_subsection("Scenario 4: Phoenix Order Book (0.1% fee)")
    phoenix_dex_dex = (DEX_FEES["Phoenix"] * 2 + SOLANA_TX_FEE) * 100
    print(f"   DEX-DEX (Phoenix → Phoenix): {phoenix_dex_dex:.3f}%")
    print(f"   Savings vs Raydium: {(std_dex_dex - phoenix_dex_dex):.3f}%")
    results["phoenix"] = {"dex_dex": phoenix_dex_dex}

    # Trade size analysis
    print_subsection("Trade Size Impact (Slippage Estimate)")
    for size in [1, 10, 100, 1000]:
        slippage = 0.001 * (size ** 0.5)  # Rough estimate
        total_cost = std_dex_dex + slippage * 100
        print(f"   {size:4d} SOL - Est. slippage: {slippage*100:.3f}%, Total: {total_cost:.3f}%")

    # Optimal break-even
    print_subsection("Break-Even Summary")
    print(f"   Minimum viable spread (standard): {std_dex_dex:.3f}%")
    print(f"   Minimum viable spread (optimized): {phoenix_dex_dex:.3f}%")
    print(f"   Optimal trade size: 1-10 SOL (lowest slippage)")

    return results



# =============================================================================
# PART 5: STRATEGY COMPARISON & FINAL REPORT
# =============================================================================

def generate_final_report(vol_data: dict, altcoin_data: dict, flash_data: dict, fee_data: dict):
    """Generate comprehensive strategy comparison report."""
    print_section("PART 5: STRATEGY COMPARISON & FINAL REPORT")

    # Strategy 1: Cross-DEX Arbitrage (SOL/USDC)
    print_subsection("Strategy 1: Cross-DEX Arbitrage (SOL/USDC)")
    sol_profitable_ratio = vol_data.get("profitable_hours_ratio", 0) * 100
    print(f"   Profitable hours (spread > 0.6%): {sol_profitable_ratio:.1f}%")
    print(f"   Avg hourly range: {vol_data.get('avg_range_pct', 0):.3f}%")
    print(f"   Min required spread: 0.56% (optimized)")

    sol_score = 2 if sol_profitable_ratio > 20 else 1 if sol_profitable_ratio > 5 else 0
    print(f"   Score: {'⭐' * sol_score}{'☆' * (3 - sol_score)} ({sol_score}/3)")

    # Strategy 2: Altcoin Arbitrage
    print_subsection("Strategy 2: Altcoin Arbitrage (Best Pair)")
    if altcoin_data:
        best_altcoin = max(altcoin_data.items(),
                          key=lambda x: max(x[1]["dex_dex_spread"], x[1]["cex_dex_spread"]))
        token, data = best_altcoin
        best_spread = max(data["dex_dex_spread"], data["cex_dex_spread"])
        print(f"   Best pair: {token}/USDC")
        print(f"   Current spread: {best_spread:.4f}%")
        print(f"   Liquidity: {data['liquidity']}")
        print(f"   Currently profitable: {'✅ YES' if data['profitable_after_fees'] else '❌ NO'}")

        alt_score = 3 if best_spread > 1.0 else 2 if best_spread > 0.5 else 1 if best_spread > 0.2 else 0
    else:
        alt_score = 0
    print(f"   Score: {'⭐' * alt_score}{'☆' * (3 - alt_score)} ({alt_score}/3)")

    # Strategy 3: Volatility-Based Trading
    print_subsection("Strategy 3: Volatility-Based Trading")
    high_vol_periods = flash_data.get("high_vol_periods", 0)
    whale_with_movement = flash_data.get("whale_with_movement", 0)
    print(f"   High volatility windows: {high_vol_periods}")
    print(f"   Whale spikes with movement: {whale_with_movement}")
    print(f"   Momentum streaks: {flash_data.get('momentum_streaks', 0)}")

    vol_score = 3 if high_vol_periods > 5 else 2 if high_vol_periods > 2 else 1 if high_vol_periods > 0 else 0
    print(f"   Score: {'⭐' * vol_score}{'☆' * (3 - vol_score)} ({vol_score}/3)")

    # Final Recommendation
    print_section("🎯 FINAL RECOMMENDATION")

    strategies = [
        ("Cross-DEX Arbitrage (SOL)", sol_score),
        ("Altcoin Arbitrage", alt_score),
        ("Volatility Trading", vol_score)
    ]
    strategies.sort(key=lambda x: x[1], reverse=True)

    print(f"\n   📊 Strategy Ranking:")
    for i, (name, score) in enumerate(strategies, 1):
        print(f"      #{i} {name}: {'⭐' * score}{'☆' * (3 - score)}")

    total_score = sol_score + alt_score + vol_score

    if total_score >= 6:
        recommendation = "✅ PROCEED with 365-day data collection"
        reason = "Strong opportunity signals detected"
    elif total_score >= 3:
        recommendation = "⚠️ CONDITIONAL - Consider smaller scope first"
        reason = "Moderate signals - test with 30-day data first"
    else:
        recommendation = "❌ PIVOT - Consider alternative approaches"
        reason = "Weak signals - markets too efficient for simple arb"

    print(f"\n   💡 Recommendation: {recommendation}")
    print(f"      Reason: {reason}")

    # Alternative pivot suggestions
    if total_score < 6:
        print(f"\n   📌 Alternative Approaches to Consider:")
        print(f"      1. MEV/Sandwich detection (observing, not executing)")
        print(f"      2. Liquidation monitoring (Solend, Marginfi)")
        print(f"      3. New token launch sniping (higher risk)")
        print(f"      4. Cross-chain arbitrage (Solana ↔ other chains)")
        print(f"      5. Market making (provide liquidity, earn fees)")

    return {
        "sol_score": sol_score,
        "alt_score": alt_score,
        "vol_score": vol_score,
        "total_score": total_score,
        "recommendation": recommendation,
        "best_strategy": strategies[0][0]
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def run_deep_analysis():
    """Run complete deep analysis."""
    print("\n" + "🔬" * 35)
    print("  DEEP STRATEGY ANALYSIS")
    print("🔬" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Objective: Decide whether to proceed with 365-day collection")

    async with httpx.AsyncClient(timeout=60) as client:
        # Part 1: Volatility
        vol_data = await analyze_volatility(client)
        await asyncio.sleep(2)  # Rate limit

        # Part 2: Altcoin pairs
        altcoin_data = await analyze_altcoin_pairs(client)
        await asyncio.sleep(2)

        # Part 3: Flash opportunities
        flash_data = await detect_flash_opportunities(client, vol_data)

        # Part 4: Fee scenarios (no API calls needed)
        sol_price = vol_data.get("df", pd.DataFrame()).get("close", pd.Series([125])).iloc[-1] if not vol_data.get("df", pd.DataFrame()).empty else 125
        fee_data = analyze_fee_scenarios(sol_price)

        # Part 5: Final report
        report = generate_final_report(vol_data, altcoin_data, flash_data, fee_data)

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70)

        return report


if __name__ == "__main__":
    report = asyncio.run(run_deep_analysis())
    sys.exit(0)