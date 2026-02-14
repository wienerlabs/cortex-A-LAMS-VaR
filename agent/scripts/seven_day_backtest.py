#!/usr/bin/env python3
"""
7-Day Historical Backtest for Arbitrage Strategy.

Collects and analyzes:
1. 168 hours of OHLCV data from Birdeye
2. CEX prices from Binance (historical via Klines)
3. Fee calculations per hour
4. Profitable opportunity detection
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# API Keys
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")

# Token addresses
SOL_MINT = "So11111111111111111111111111111111111111112"

# Fee structure (optimized Phoenix scenario)
FEES = {
    "dex_dex_standard": 0.0056,    # Raydium + Orca + TX
    "dex_dex_phoenix": 0.0021,     # Phoenix + TX
    "cex_dex": 0.0162,             # Binance + Raydium + withdrawal + TX
    "cex_dex_maker": 0.0152,       # Binance maker + Raydium + TX
}


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def make_request(client: httpx.AsyncClient, url: str, headers: dict = None) -> tuple:
    try:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return True, resp.json()
        elif resp.status_code == 429:
            return False, {"error": "rate_limit"}
        return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


async def collect_birdeye_ohlcv(client: httpx.AsyncClient, hours: int = 168) -> pd.DataFrame:
    """Collect hourly OHLCV data from Birdeye."""
    print(f"\n▸ Collecting Birdeye OHLCV ({hours} hours)...")

    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
    now = int(datetime.now(timezone.utc).timestamp())
    time_from = now - (hours * 3600)

    url = (f"https://public-api.birdeye.so/defi/ohlcv?"
           f"address={SOL_MINT}&type=1H&time_from={time_from}&time_to={now}")

    for attempt in range(3):
        success, data = await make_request(client, url, headers)
        if success and data.get("success"):
            break
        if data.get("error") == "rate_limit":
            print(f"   ⏳ Rate limited, waiting {(attempt+1)*5}s...")
            await asyncio.sleep((attempt+1) * 5)

    if not success or not data.get("success"):
        print(f"   ❌ Failed to fetch OHLCV")
        return pd.DataFrame()

    items = data.get("data", {}).get("items", [])
    df = pd.DataFrame(items)

    # Normalize columns
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=col_map)

    if "unixTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["unixTime"], unit="s", utc=True)

    print(f"   ✅ Collected {len(df)} candles")
    return df


async def collect_binance_klines(client: httpx.AsyncClient, hours: int = 168) -> pd.DataFrame:
    """Collect hourly klines from Binance."""
    print(f"\n▸ Collecting Binance Klines ({hours} hours)...")

    url = f"https://api.binance.com/api/v3/klines?symbol=SOLUSDC&interval=1h&limit={min(hours, 1000)}"
    success, data = await make_request(client, url)

    if not success or not isinstance(data, list):
        print(f"   ❌ Failed to fetch Binance klines")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["binance_close"] = df["close"].astype(float)
    df["binance_high"] = df["high"].astype(float)
    df["binance_low"] = df["low"].astype(float)

    print(f"   ✅ Collected {len(df)} klines")
    return df[["datetime", "binance_close", "binance_high", "binance_low"]]


def calculate_spreads(dex_df: pd.DataFrame, cex_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate hourly spreads between DEX and CEX."""
    print(f"\n▸ Calculating Cross-Platform Spreads...")

    # Merge on datetime
    merged = pd.merge(dex_df, cex_df, on="datetime", how="inner")

    if merged.empty:
        print("   ❌ No matching data")
        return pd.DataFrame()

    # Calculate spreads
    merged["dex_price"] = merged["close"]
    merged["cex_price"] = merged["binance_close"]

    # CEX-DEX spread (absolute difference)
    merged["cex_dex_spread"] = abs(merged["cex_price"] - merged["dex_price"])
    merged["cex_dex_spread_pct"] = (merged["cex_dex_spread"] / merged["dex_price"]) * 100

    # Intra-hour range as proxy for DEX-DEX opportunity
    merged["dex_range"] = merged["high"] - merged["low"]
    merged["dex_range_pct"] = (merged["dex_range"] / merged["close"]) * 100

    # CEX intra-hour range
    merged["cex_range"] = merged["binance_high"] - merged["binance_low"]
    merged["cex_range_pct"] = (merged["cex_range"] / merged["binance_close"]) * 100

    print(f"   ✅ Calculated spreads for {len(merged)} hours")
    return merged


def find_profitable_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """Find hours with profitable arbitrage opportunities."""
    print(f"\n▸ Detecting Profitable Opportunities...")

    opportunities = []

    for _, row in df.iterrows():
        # CEX-DEX opportunity (if CEX price != DEX price)
        cex_dex_spread = row["cex_dex_spread_pct"] / 100
        cex_dex_fee = FEES["cex_dex_maker"]
        cex_dex_profit = cex_dex_spread - cex_dex_fee

        # DEX-DEX opportunity (use intra-hour range as proxy)
        dex_range = row["dex_range_pct"] / 100
        # Assume we can capture 50% of the range
        dex_dex_potential = dex_range * 0.5
        dex_dex_fee = FEES["dex_dex_phoenix"]
        dex_dex_profit = dex_dex_potential - dex_dex_fee

        opportunities.append({
            "datetime": row["datetime"],
            "dex_price": row["dex_price"],
            "cex_price": row["cex_price"],
            "cex_dex_spread_pct": row["cex_dex_spread_pct"],
            "cex_dex_profit_pct": cex_dex_profit * 100,
            "cex_dex_profitable": cex_dex_profit > 0,
            "dex_range_pct": row["dex_range_pct"],
            "dex_dex_potential_pct": dex_dex_potential * 100,
            "dex_dex_profit_pct": dex_dex_profit * 100,
            "dex_dex_profitable": dex_dex_profit > 0,
            "any_profitable": cex_dex_profit > 0 or dex_dex_profit > 0,
            "best_profit_pct": max(cex_dex_profit, dex_dex_profit) * 100,
            "best_strategy": "CEX-DEX" if cex_dex_profit > dex_dex_profit else "DEX-DEX"
        })

    opp_df = pd.DataFrame(opportunities)

    profitable_hours = len(opp_df[opp_df["any_profitable"]])
    print(f"   ✅ Profitable hours: {profitable_hours}/{len(opp_df)} ({profitable_hours/len(opp_df)*100:.1f}%)")

    return opp_df


def generate_report(opp_df: pd.DataFrame):
    """Generate final analysis report."""
    print_section("7-DAY BACKTEST RESULTS")

    total_hours = len(opp_df)
    profitable_hours = len(opp_df[opp_df["any_profitable"]])

    # 1. Summary Stats
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"   Total hours analyzed: {total_hours}")
    print(f"   Profitable hours: {profitable_hours} ({profitable_hours/total_hours*100:.1f}%)")
    print(f"   Date range: {opp_df['datetime'].min()} to {opp_df['datetime'].max()}")

    # 2. Spread Analysis
    print(f"\n📈 SPREAD ANALYSIS")
    print(f"   CEX-DEX Spread:")
    print(f"      Average: {opp_df['cex_dex_spread_pct'].mean():.4f}%")
    print(f"      Max: {opp_df['cex_dex_spread_pct'].max():.4f}%")
    print(f"      Hours > 0.6%: {len(opp_df[opp_df['cex_dex_spread_pct'] > 0.6])}")

    print(f"   DEX Range (Intra-hour):")
    print(f"      Average: {opp_df['dex_range_pct'].mean():.4f}%")
    print(f"      Max: {opp_df['dex_range_pct'].max():.4f}%")
    print(f"      Hours > 0.6%: {len(opp_df[opp_df['dex_range_pct'] > 0.6])}")

    # 3. Profit Analysis
    print(f"\n💰 PROFIT ANALYSIS (After Fees)")
    profitable = opp_df[opp_df["any_profitable"]]
    if len(profitable) > 0:
        print(f"   Average net profit: {profitable['best_profit_pct'].mean():.4f}%")
        print(f"   Max net profit: {profitable['best_profit_pct'].max():.4f}%")
        print(f"   Total profit potential (sum): {profitable['best_profit_pct'].sum():.4f}%")

        # By strategy
        cex_dex_wins = len(profitable[profitable["best_strategy"] == "CEX-DEX"])
        dex_dex_wins = len(profitable[profitable["best_strategy"] == "DEX-DEX"])
        print(f"   Best strategy breakdown:")
        print(f"      CEX-DEX wins: {cex_dex_wins}")
        print(f"      DEX-DEX wins: {dex_dex_wins}")
    else:
        print(f"   ❌ No profitable opportunities found")

    # 4. Best 5 Opportunities
    print(f"\n🏆 TOP 5 OPPORTUNITIES")
    top5 = opp_df.nlargest(5, "best_profit_pct")
    for i, row in top5.iterrows():
        status = "✅" if row["any_profitable"] else "❌"
        print(f"   {status} {row['datetime']} | {row['best_strategy']:7} | "
              f"Spread: {row['cex_dex_spread_pct']:.3f}% | Net: {row['best_profit_pct']:.3f}%")

    # 5. Hourly Distribution
    print(f"\n⏰ PROFITABLE HOURS BY TIME (UTC)")
    if len(profitable) > 0:
        profitable["hour"] = profitable["datetime"].dt.hour
        hourly = profitable.groupby("hour").size().sort_values(ascending=False)
        for hour, count in hourly.head(5).items():
            print(f"      {hour:02d}:00 UTC - {count} opportunities")

    # 6. Weekly Pattern
    print(f"\n📅 PROFITABLE HOURS BY DAY")
    if len(profitable) > 0:
        profitable["day"] = profitable["datetime"].dt.day_name()
        daily = profitable.groupby("day").size()
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            if day in daily.index:
                print(f"      {day}: {daily[day]} opportunities")

    # 7. Final Recommendation
    print_section("🎯 FINAL RECOMMENDATION")

    if profitable_hours / total_hours > 0.3:
        print(f"\n   ✅ PROCEED with 365-day data collection")
        print(f"   Reason: {profitable_hours/total_hours*100:.1f}% of hours show opportunity")
    elif profitable_hours / total_hours > 0.1:
        print(f"\n   ⚠️ CONDITIONAL - Consider different approach")
        print(f"   Reason: Only {profitable_hours/total_hours*100:.1f}% profitable hours")
        print(f"   Suggestion: Focus on high-volatility hours only")
    else:
        print(f"\n   ❌ PIVOT recommended")
        print(f"   Reason: Only {profitable_hours/total_hours*100:.1f}% profitable hours")
        print(f"   Markets too efficient for simple arbitrage")

    return {
        "total_hours": total_hours,
        "profitable_hours": profitable_hours,
        "profitable_ratio": profitable_hours / total_hours,
        "avg_profit": profitable["best_profit_pct"].mean() if len(profitable) > 0 else 0,
        "max_profit": profitable["best_profit_pct"].max() if len(profitable) > 0 else 0
    }


async def run_backtest():
    """Run the 7-day backtest."""
    print("\n" + "📊" * 35)
    print("  7-DAY HISTORICAL BACKTEST")
    print("📊" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")

    async with httpx.AsyncClient(timeout=60) as client:
        # 1. Collect data
        print_section("DATA COLLECTION")

        dex_df = await collect_birdeye_ohlcv(client, hours=168)
        await asyncio.sleep(2)  # Rate limit

        cex_df = await collect_binance_klines(client, hours=168)

        if dex_df.empty or cex_df.empty:
            print("\n❌ Failed to collect data")
            return None

        # 2. Calculate spreads
        print_section("SPREAD CALCULATION")
        spread_df = calculate_spreads(dex_df, cex_df)

        if spread_df.empty:
            print("\n❌ Failed to calculate spreads")
            return None

        # 3. Find opportunities
        print_section("OPPORTUNITY DETECTION")
        opp_df = find_profitable_opportunities(spread_df)

        # 4. Generate report
        result = generate_report(opp_df)

        return result


if __name__ == "__main__":
    result = asyncio.run(run_backtest())
    sys.exit(0 if result else 1)

