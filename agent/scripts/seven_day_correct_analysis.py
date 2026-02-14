#!/usr/bin/env python3
"""
7-Day CEX-DEX Arbitrage Analysis - CORRECT METHODOLOGY

Only compares Birdeye vs Binance at SAME TIMESTAMP.
No intra-hour volatility proxy - real cross-platform spreads only.
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
SOL_MINT = "So11111111111111111111111111111111111111112"

# CORRECT FEE STRUCTURE (as percentage of trade)
# For $100 trade with SOL at ~$125:
# - Binance taker: 0.1%
# - Binance withdrawal: 0.001 SOL = $0.125 = 0.125% of $100
# - Raydium swap: 0.25%
# - Solana tx: 0.00025 SOL = $0.03 = 0.03% of $100
# TOTAL = 0.1 + 0.125 + 0.25 + 0.03 = 0.505%
# For larger trades ($1000), withdrawal becomes 0.0125%, tx 0.003% → ~0.365%
TOTAL_FEE_PCT = 0.36  # Conservative for ~$1000 trade


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def make_request(client: httpx.AsyncClient, url: str, headers: dict = None) -> tuple:
    try:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


async def collect_data(client: httpx.AsyncClient) -> pd.DataFrame:
    """Collect Birdeye and Binance data, merge on timestamp."""

    # 1. Birdeye OHLCV (7 days = 168 hours)
    print("\n▸ Collecting Birdeye OHLCV (168 hours)...")
    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
    now = int(datetime.now(timezone.utc).timestamp())
    time_from = now - (168 * 3600)

    url = (f"https://public-api.birdeye.so/defi/ohlcv?"
           f"address={SOL_MINT}&type=1H&time_from={time_from}&time_to={now}")

    success, data = await make_request(client, url, headers)
    if not success or not data.get("success"):
        print("   ❌ Birdeye failed")
        return pd.DataFrame()

    items = data.get("data", {}).get("items", [])
    birdeye_df = pd.DataFrame(items)
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    birdeye_df = birdeye_df.rename(columns=col_map)
    birdeye_df["datetime"] = pd.to_datetime(birdeye_df["unixTime"], unit="s", utc=True)
    birdeye_df["birdeye_close"] = birdeye_df["close"].astype(float)
    print(f"   ✅ Birdeye: {len(birdeye_df)} candles")

    await asyncio.sleep(1)

    # 2. Binance Klines
    print("\n▸ Collecting Binance Klines (168 hours)...")
    url = "https://api.binance.com/api/v3/klines?symbol=SOLUSDC&interval=1h&limit=168"
    success, data = await make_request(client, url)

    if not success or not isinstance(data, list):
        print("   ❌ Binance failed")
        return pd.DataFrame()

    binance_df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    binance_df["datetime"] = pd.to_datetime(binance_df["open_time"], unit="ms", utc=True)
    binance_df["binance_close"] = binance_df["close"].astype(float)
    print(f"   ✅ Binance: {len(binance_df)} klines")

    # 3. Merge on datetime (same timestamp)
    print("\n▸ Merging data on SAME TIMESTAMP...")
    merged = pd.merge(
        birdeye_df[["datetime", "birdeye_close"]],
        binance_df[["datetime", "binance_close"]],
        on="datetime",
        how="inner"
    )
    print(f"   ✅ Matched: {len(merged)} hours")

    return merged


def analyze_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate real CEX-DEX spreads."""
    print_section("CEX-DEX SPREAD CALCULATION")

    # Calculate spread (absolute difference as %)
    df["spread_usd"] = abs(df["binance_close"] - df["birdeye_close"])
    df["spread_pct"] = (df["spread_usd"] / df["birdeye_close"]) * 100

    # Direction: which platform is higher?
    df["direction"] = df.apply(
        lambda r: "BUY_DEX_SELL_CEX" if r["binance_close"] > r["birdeye_close"] else "BUY_CEX_SELL_DEX",
        axis=1
    )

    # Net profit after fees
    df["net_profit_pct"] = df["spread_pct"] - TOTAL_FEE_PCT
    df["is_profitable"] = df["net_profit_pct"] > 0

    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day_name()

    return df


def generate_report(df: pd.DataFrame):
    """Generate comprehensive analysis report."""
    print_section("7-DAY CEX-DEX ANALYSIS RESULTS")

    total = len(df)
    profitable = df[df["is_profitable"]]
    profitable_count = len(profitable)
    profitable_pct = (profitable_count / total) * 100

    # 1. Summary Statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"   Total hours analyzed: {total}")
    print(f"   Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"   Fee threshold: {TOTAL_FEE_PCT}%")

    # 2. Spread Statistics
    print(f"\n📈 SPREAD STATISTICS (Same Timestamp)")
    print(f"   Average spread: {df['spread_pct'].mean():.4f}%")
    print(f"   Max spread: {df['spread_pct'].max():.4f}%")
    print(f"   Min spread: {df['spread_pct'].min():.4f}%")
    print(f"   Std deviation: {df['spread_pct'].std():.4f}%")
    print(f"   Hours with spread > 0.36%: {len(df[df['spread_pct'] > 0.36])}")
    print(f"   Hours with spread > 0.50%: {len(df[df['spread_pct'] > 0.50])}")
    print(f"   Hours with spread > 1.00%: {len(df[df['spread_pct'] > 1.00])}")

    # 3. Profitability
    print(f"\n💰 PROFITABILITY ANALYSIS")
    print(f"   Profitable hours: {profitable_count}/{total} ({profitable_pct:.1f}%)")

    if profitable_count > 0:
        print(f"   Average net profit (profitable): {profitable['net_profit_pct'].mean():.4f}%")
        print(f"   Max net profit: {profitable['net_profit_pct'].max():.4f}%")
        print(f"   Total profit potential: {profitable['net_profit_pct'].sum():.4f}%")
    else:
        print(f"   ❌ No profitable hours found")

    # 4. Top 10 Opportunities
    print(f"\n🏆 TOP 10 OPPORTUNITIES")
    top10 = df.nlargest(10, "spread_pct")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        status = "✅" if row["is_profitable"] else "❌"
        print(f"   {i:2}. {status} {row['datetime']}")
        print(f"       Birdeye: ${row['birdeye_close']:.2f} | Binance: ${row['binance_close']:.2f}")
        print(f"       Spread: {row['spread_pct']:.4f}% | Net: {row['net_profit_pct']:.4f}%")
        print(f"       Direction: {row['direction']}")

    # 5. Hourly Distribution
    print(f"\n⏰ SPREAD BY HOUR (UTC)")
    hourly = df.groupby("hour")["spread_pct"].agg(["mean", "max", "count"])
    hourly = hourly.sort_values("mean", ascending=False)
    for hour in hourly.head(5).index:
        stats = hourly.loc[hour]
        print(f"      {hour:02d}:00 UTC - Avg: {stats['mean']:.4f}%, Max: {stats['max']:.4f}%")

    # 6. Daily Distribution
    print(f"\n📅 SPREAD BY DAY")
    daily = df.groupby("day")["spread_pct"].agg(["mean", "max"])
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        if day in daily.index:
            stats = daily.loc[day]
            print(f"      {day}: Avg: {stats['mean']:.4f}%, Max: {stats['max']:.4f}%")

    # 7. Direction Analysis
    print(f"\n🔄 DIRECTION ANALYSIS")
    dir_counts = df["direction"].value_counts()
    for direction, count in dir_counts.items():
        pct = (count / total) * 100
        print(f"      {direction}: {count} ({pct:.1f}%)")

    # 8. FINAL DECISION
    print_section("🎯 FINAL DECISION")

    if profitable_pct < 5:
        decision = "PIVOT"
        reason = f"Only {profitable_pct:.1f}% of hours profitable - market too efficient"
        next_step = "Consider MEV detection, liquidation monitoring, or cross-chain arbitrage"
    elif profitable_pct < 20:
        decision = "30-DAY TEST"
        reason = f"{profitable_pct:.1f}% hours profitable - worth deeper investigation"
        next_step = "Collect 30 days of data with minute-level granularity"
    else:
        decision = "PROCEED TO 365-DAY"
        reason = f"{profitable_pct:.1f}% hours profitable - strong opportunity signal"
        next_step = "Start full 365-day historical data collection"

    print(f"\n   Decision: {'✅' if decision != 'PIVOT' else '❌'} {decision}")
    print(f"   Reason: {reason}")
    print(f"   Next Step: {next_step}")

    # Return summary for programmatic use
    return {
        "total_hours": total,
        "profitable_hours": profitable_count,
        "profitable_pct": profitable_pct,
        "avg_spread": df["spread_pct"].mean(),
        "max_spread": df["spread_pct"].max(),
        "avg_net_profit": profitable["net_profit_pct"].mean() if profitable_count > 0 else 0,
        "decision": decision
    }


async def run_analysis():
    """Main analysis entry point."""
    print("\n" + "🔬" * 35)
    print("  7-DAY CEX-DEX ARBITRAGE ANALYSIS")
    print("  CORRECT METHODOLOGY: Same Timestamp Comparison")
    print("🔬" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Fee Structure: Binance(0.1%) + Withdrawal(0.125%) + Raydium(0.25%) + TX(0.03%) = {TOTAL_FEE_PCT}%")

    async with httpx.AsyncClient(timeout=60) as client:
        # Collect data
        print_section("DATA COLLECTION")
        df = await collect_data(client)

        if df.empty:
            print("\n❌ Failed to collect data")
            return None

        # Analyze spreads
        df = analyze_spreads(df)

        # Generate report
        result = generate_report(df)

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70)

        return result


if __name__ == "__main__":
    result = asyncio.run(run_analysis())
    sys.exit(0 if result else 1)
