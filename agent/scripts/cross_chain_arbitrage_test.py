#!/usr/bin/env python3
"""
Cross-Chain Arbitrage Test: Solana ↔ Ethereum

Compares SOL prices across chains to find arbitrage opportunities.
Uses Binance for both chains' reference prices (SOL/USDC).
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

# COST STRUCTURE
# Wormhole bridge: $2-5 flat fee (use $3.5 average)
# Bridge time: 5-15 min (price risk)
# DEX fees: 0.25% each side = 0.5% total
# Slippage estimate: 0.1% each side = 0.2% total
# Total percentage cost: 0.7%
# Total flat cost: $3.5

BRIDGE_FLAT_FEE_USD = 3.5
PERCENTAGE_FEES = 0.70  # 0.7% total (DEX + slippage both sides)


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
    """Collect price data from multiple sources."""

    # 1. Birdeye (Solana DEX prices) - 7 days
    print("\n▸ Collecting Birdeye OHLCV (Solana DEX)...")
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
    solana_df = pd.DataFrame(items)
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    solana_df = solana_df.rename(columns=col_map)
    solana_df["datetime"] = pd.to_datetime(solana_df["unixTime"], unit="s", utc=True)
    solana_df["solana_price"] = solana_df["close"].astype(float)
    print(f"   ✅ Solana DEX: {len(solana_df)} candles")

    await asyncio.sleep(1)

    # 2. Binance (CEX - represents "Ethereum side" as it's the main off-chain price)
    print("\n▸ Collecting Binance Klines (CEX/ETH proxy)...")
    url = "https://api.binance.com/api/v3/klines?symbol=SOLUSDC&interval=1h&limit=168"
    success, data = await make_request(client, url)

    if not success or not isinstance(data, list):
        print("   ❌ Binance failed")
        return pd.DataFrame()

    eth_df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    eth_df["datetime"] = pd.to_datetime(eth_df["open_time"], unit="ms", utc=True)
    eth_df["eth_price"] = eth_df["close"].astype(float)
    print(f"   ✅ ETH/CEX proxy: {len(eth_df)} klines")

    # 3. Merge on datetime
    print("\n▸ Merging data on SAME TIMESTAMP...")
    merged = pd.merge(
        solana_df[["datetime", "solana_price"]],
        eth_df[["datetime", "eth_price"]],
        on="datetime",
        how="inner"
    )
    print(f"   ✅ Matched: {len(merged)} hours")

    return merged


def analyze_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-chain arbitrage opportunities."""
    print_section("CROSS-CHAIN SPREAD CALCULATION")

    # Calculate spread
    df["spread_usd"] = abs(df["eth_price"] - df["solana_price"])
    df["spread_pct"] = (df["spread_usd"] / df["solana_price"]) * 100

    # Direction
    df["direction"] = df.apply(
        lambda r: "ETH→SOL" if r["eth_price"] > r["solana_price"] else "SOL→ETH",
        axis=1
    )

    # Calculate costs for a $1000 trade
    trade_size = 1000
    df["pct_cost"] = PERCENTAGE_FEES
    df["flat_cost_pct"] = (BRIDGE_FLAT_FEE_USD / trade_size) * 100  # As percentage
    df["total_cost_pct"] = df["pct_cost"] + df["flat_cost_pct"]

    # Net profit
    df["net_profit_pct"] = df["spread_pct"] - df["total_cost_pct"]
    df["is_profitable"] = df["net_profit_pct"] > 0

    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day_name()

    print(f"   Trade size: ${trade_size}")
    print(f"   Percentage fees: {PERCENTAGE_FEES}%")
    print(f"   Bridge flat fee: ${BRIDGE_FLAT_FEE_USD} ({df['flat_cost_pct'].iloc[0]:.2f}%)")
    print(f"   Total cost threshold: {df['total_cost_pct'].iloc[0]:.2f}%")

    return df


def generate_report(df: pd.DataFrame):
    """Generate comprehensive cross-chain analysis report."""
    print_section("7-DAY CROSS-CHAIN ANALYSIS RESULTS")

    total = len(df)
    profitable = df[df["is_profitable"]]
    profitable_count = len(profitable)

    # 1. Summary
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"   Total hours analyzed: {total}")
    print(f"   Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"   Cost threshold: {df['total_cost_pct'].iloc[0]:.2f}%")

    # 2. Spread Statistics
    print(f"\n📈 SPREAD STATISTICS")
    print(f"   Average spread: {df['spread_pct'].mean():.4f}%")
    print(f"   Max spread: {df['spread_pct'].max():.4f}%")
    print(f"   Min spread: {df['spread_pct'].min():.4f}%")
    print(f"   Std deviation: {df['spread_pct'].std():.4f}%")

    # Threshold analysis
    thresholds = [0.5, 0.7, 1.0, 1.5, 2.0]
    print(f"\n   Spread distribution:")
    for t in thresholds:
        count = len(df[df['spread_pct'] > t])
        print(f"      > {t}%: {count} hours ({count/total*100:.1f}%)")

    # 3. Profitability
    print(f"\n💰 PROFITABILITY ANALYSIS")
    print(f"   Profitable hours: {profitable_count}/{total} ({profitable_count/total*100:.1f}%)")

    if profitable_count > 0:
        print(f"   Average net profit: {profitable['net_profit_pct'].mean():.4f}%")
        print(f"   Max net profit: {profitable['net_profit_pct'].max():.4f}%")
        print(f"   Total profit potential: {profitable['net_profit_pct'].sum():.4f}%")
    else:
        print(f"   ❌ No profitable opportunities found")

    # 4. Direction Analysis
    print(f"\n🔄 DIRECTION ANALYSIS")
    dir_counts = df["direction"].value_counts()
    for direction, count in dir_counts.items():
        pct = (count / total) * 100
        avg_spread = df[df["direction"] == direction]["spread_pct"].mean()
        print(f"      {direction}: {count} hours ({pct:.1f}%) | Avg spread: {avg_spread:.4f}%")

    # 5. Top 5 Opportunities
    print(f"\n🏆 TOP 5 OPPORTUNITIES")
    top5 = df.nlargest(5, "spread_pct")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        status = "✅" if row["is_profitable"] else "❌"
        print(f"   {i}. {status} {row['datetime']}")
        print(f"      Solana: ${row['solana_price']:.2f} | ETH/CEX: ${row['eth_price']:.2f}")
        print(f"      Spread: {row['spread_pct']:.4f}% | Net: {row['net_profit_pct']:.4f}%")
        print(f"      Direction: {row['direction']}")

    # 6. Time patterns
    print(f"\n⏰ SPREAD BY HOUR (UTC)")
    hourly = df.groupby("hour")["spread_pct"].agg(["mean", "max"])
    hourly = hourly.sort_values("mean", ascending=False)
    for hour in hourly.head(5).index:
        stats = hourly.loc[hour]
        print(f"      {hour:02d}:00 UTC - Avg: {stats['mean']:.4f}%, Max: {stats['max']:.4f}%")

    # 7. FINAL VERDICT
    print_section("🎯 FINAL VERDICT")

    if profitable_count < 5:
        verdict = "PROJE DUR"
        emoji = "❌"
        reason = f"Sadece {profitable_count} karlı saat - yetersiz"
        next_step = "Cross-chain arbitrage viable değil. Başka strateji bul."
    elif profitable_count < 15:
        verdict = "PROMISING"
        emoji = "⚠️"
        reason = f"{profitable_count} karlı saat - potansiyel var"
        next_step = "30 günlük detaylı analiz yap"
    else:
        verdict = "365-GÜN COLLECTION"
        emoji = "✅"
        reason = f"{profitable_count} karlı saat - güçlü sinyal"
        next_step = "Tam 365 günlük veri topla ve strateji geliştir"

    print(f"\n   Karlı Saatler: {profitable_count}/168")
    print(f"\n   Karar: {emoji} {verdict}")
    print(f"   Neden: {reason}")
    print(f"   Sonraki Adım: {next_step}")

    # Risk factors
    print(f"\n⚠️ RISK FACTORS")
    print(f"   • Bridge time: 5-15 dakika (fiyat değişebilir)")
    print(f"   • Bridge failure riski")
    print(f"   • Slippage büyük trade'lerde artabilir")
    print(f"   • MEV bots on both chains")

    return {
        "profitable_count": profitable_count,
        "verdict": verdict,
        "avg_spread": df["spread_pct"].mean(),
        "max_spread": df["spread_pct"].max()
    }


async def run_test():
    """Run the cross-chain arbitrage test."""
    print("\n" + "🌉" * 35)
    print("  CROSS-CHAIN ARBITRAGE TEST: SOLANA ↔ ETHEREUM")
    print("🌉" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Bridge: Wormhole | Fee: ${BRIDGE_FLAT_FEE_USD} + {PERCENTAGE_FEES}%")

    async with httpx.AsyncClient(timeout=60) as client:
        # Collect data
        print_section("DATA COLLECTION")
        df = await collect_data(client)

        if df.empty:
            print("\n❌ Failed to collect data")
            return None

        # Analyze opportunities
        df = analyze_opportunities(df)

        # Generate report
        result = generate_report(df)

        print("\n" + "=" * 70)
        print("  TEST COMPLETE")
        print("=" * 70)

        return result


if __name__ == "__main__":
    result = asyncio.run(run_test())
    sys.exit(0 if result else 1)
