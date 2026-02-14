#!/usr/bin/env python3
"""
Collect 5-Minute Interval Swap Data from Uniswap V3.

Fetches swap-level data and aggregates into 5-minute OHLCV buckets.
Target: 365 days × 288 intervals = ~105,000 rows
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

from src.config import settings

UNISWAP_URL = f"https://gateway.thegraph.com/api/{settings.thegraph_api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# Pools to compare for arbitrage
POOL_005 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # ETH/USDC 0.05%
POOL_03 = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"   # ETH/USDC 0.3%


def sqrt_price_to_price(sqrt_price_x96: str) -> float:
    """
    Convert sqrtPriceX96 to ETH price in USDC.

    For ETH/USDC pool (0x88e6a...):
    - token0 = USDC (6 decimals)
    - token1 = WETH (18 decimals)
    - sqrtPriceX96 = sqrt(token1/token0) * 2^96

    To get USDC per ETH:
    1. Calculate raw_price = (sqrtPriceX96 / 2^96)^2
    2. Invert and adjust decimals: (1/raw_price) * 10^12
    """
    try:
        sqrt_price = int(sqrt_price_x96)
        raw_price = (sqrt_price / (2**96)) ** 2
        # Invert and adjust for decimals (18 - 6 = 12)
        eth_price_usdc = (1 / raw_price) * (10 ** 12)
        return eth_price_usdc
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


async def fetch_swaps(session: aiohttp.ClientSession, pool: str, 
                      timestamp_gt: int, timestamp_lt: int, skip: int = 0) -> list:
    """Fetch swaps for a time range."""
    query = """
    query Swaps($pool: String!, $timestamp_gt: Int!, $timestamp_lt: Int!, $skip: Int!) {
      swaps(
        first: 1000
        skip: $skip
        orderBy: timestamp
        orderDirection: asc
        where: {
          pool: $pool
          timestamp_gte: $timestamp_gt
          timestamp_lt: $timestamp_lt
        }
      ) {
        timestamp
        sqrtPriceX96
        amount0
        amount1
        amountUSD
      }
    }
    """
    
    variables = {
        "pool": pool.lower(),
        "timestamp_gt": timestamp_gt,
        "timestamp_lt": timestamp_lt,
        "skip": skip
    }
    
    try:
        async with session.post(UNISWAP_URL, json={"query": query, "variables": variables}) as resp:
            if resp.status == 200:
                data = await resp.json()
                if "errors" in data:
                    return []
                return data.get("data", {}).get("swaps", [])
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    return []


async def fetch_day_swaps(session: aiohttp.ClientSession, pool: str, date: datetime) -> list:
    """Fetch all swaps for a single day."""
    start_ts = int(date.timestamp())
    end_ts = int((date + timedelta(days=1)).timestamp())
    
    all_swaps = []
    skip = 0
    
    while True:
        swaps = await fetch_swaps(session, pool, start_ts, end_ts, skip)
        if not swaps:
            break
        all_swaps.extend(swaps)
        skip += 1000
        if len(swaps) < 1000:
            break
        await asyncio.sleep(0.1)  # Rate limit
    
    return all_swaps


def aggregate_to_5min(swaps: list, pool_name: str) -> pd.DataFrame:
    """Aggregate swaps into 5-minute OHLCV buckets."""
    if not swaps:
        return pd.DataFrame()
    
    df = pd.DataFrame(swaps)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["price"] = df["sqrtPriceX96"].apply(lambda x: sqrt_price_to_price(x))
    df["amountUSD"] = pd.to_numeric(df["amountUSD"], errors="coerce").fillna(0)
    
    # Round to 5-minute intervals
    df["interval"] = df["datetime"].dt.floor("5min")
    
    # Aggregate OHLCV
    ohlcv = df.groupby("interval").agg({
        "price": ["first", "max", "min", "last"],
        "amountUSD": "sum",
        "timestamp": "count"
    }).reset_index()
    
    ohlcv.columns = ["datetime", "open", "high", "low", "close", "volume", "trade_count"]
    ohlcv["pool"] = pool_name
    
    return ohlcv


async def collect_pool_data(pool_address: str, pool_name: str, days: int = 30,
                           checkpoint_dir: Path = None) -> pd.DataFrame:
    """Collect data for a single pool with checkpoint support."""
    print(f"\n📊 Collecting {pool_name} ({days} days)...")

    # Setup checkpoint
    checkpoint_file = None
    collected_dates = set()
    if checkpoint_dir:
        checkpoint_file = checkpoint_dir / f"checkpoint_{pool_name}.csv"
        if checkpoint_file.exists():
            existing = pd.read_csv(checkpoint_file)
            collected_dates = set(pd.to_datetime(existing["datetime"]).dt.date.astype(str))
            print(f"   📁 Resuming from checkpoint: {len(collected_dates)} days already collected")

    all_data = []
    end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async with aiohttp.ClientSession() as session:
        for day_offset in range(days):
            date = end_date - timedelta(days=day_offset + 1)
            date_str = date.strftime("%Y-%m-%d")

            # Skip if already collected
            if date_str in collected_dates:
                continue

            swaps = await fetch_day_swaps(session, pool_address, date)

            if swaps:
                ohlcv = aggregate_to_5min(swaps, pool_name)
                if not ohlcv.empty:
                    all_data.append(ohlcv)

                    # Save checkpoint every 10 days
                    if checkpoint_file and len(all_data) % 10 == 0:
                        temp_df = pd.concat(all_data, ignore_index=True)
                        if checkpoint_file.exists():
                            existing = pd.read_csv(checkpoint_file)
                            temp_df = pd.concat([existing, temp_df], ignore_index=True)
                            temp_df = temp_df.drop_duplicates(subset=["datetime"])
                        temp_df.to_csv(checkpoint_file, index=False)
                        print(f"   💾 Checkpoint saved: {len(temp_df):,} rows")

            if (day_offset + 1) % 10 == 0:
                collected = len(all_data) + len(collected_dates)
                print(f"   Day {day_offset + 1}/{days} - {collected} days total")

            await asyncio.sleep(0.25)  # Rate limit (slightly slower for 365 days)

    # Final merge with checkpoint
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        if checkpoint_file and checkpoint_file.exists():
            existing = pd.read_csv(checkpoint_file)
            existing["datetime"] = pd.to_datetime(existing["datetime"])
            combined = pd.concat([existing, combined], ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime"])
        combined = combined.sort_values("datetime").reset_index(drop=True)

        # Save final checkpoint
        if checkpoint_file:
            combined.to_csv(checkpoint_file, index=False)

        print(f"   ✅ {len(combined):,} intervals collected")
        return combined

    # Return existing checkpoint if no new data
    if checkpoint_file and checkpoint_file.exists():
        existing = pd.read_csv(checkpoint_file)
        existing["datetime"] = pd.to_datetime(existing["datetime"])
        return existing

    return pd.DataFrame()


def calculate_spreads_and_labels(df_005: pd.DataFrame, df_03: pd.DataFrame) -> pd.DataFrame:
    """Calculate spreads between pools and create labels with realistic costs."""
    print("\n📊 Calculating spreads with gas + slippage...")

    # Merge on datetime
    df_005 = df_005.rename(columns={
        "open": "open_005", "high": "high_005", "low": "low_005",
        "close": "close_005", "volume": "volume_005", "trade_count": "trades_005"
    })
    df_03 = df_03.rename(columns={
        "open": "open_03", "high": "high_03", "low": "low_03",
        "close": "close_03", "volume": "volume_03", "trade_count": "trades_03"
    })

    merged = pd.merge(
        df_005[["datetime", "open_005", "high_005", "low_005", "close_005", "volume_005", "trades_005"]],
        df_03[["datetime", "open_03", "high_03", "low_03", "close_03", "volume_03", "trades_03"]],
        on="datetime",
        how="inner"
    )

    # Calculate spread (percentage difference)
    merged["spread"] = (merged["close_005"] - merged["close_03"]) / merged["close_03"] * 100
    merged["spread_abs"] = merged["spread"].abs()

    # Features
    merged["volume_total"] = merged["volume_005"] + merged["volume_03"]
    merged["trades_total"] = merged["trades_005"] + merged["trades_03"]
    merged["price"] = merged["close_005"]
    merged["price_change"] = merged["price"].pct_change() * 100

    # Rolling features
    merged["spread_ma12"] = merged["spread_abs"].rolling(12).mean()  # 1 hour
    merged["spread_std12"] = merged["spread_abs"].rolling(12).std()
    merged["volume_ma12"] = merged["volume_total"].rolling(12).mean()
    merged["volatility_12"] = merged["price_change"].rolling(12).std()

    # ==========================================
    # GAS + SLIPPAGE COST CALCULATION
    # ==========================================

    # Gas estimation (optimized)
    # Optimized arbitrage contract: ~120,000 gas (flash loan + 2 swaps)
    # Average gas price: 20 gwei (can execute during low periods)
    GAS_UNITS = 120000
    GAS_GWEI = 20  # Realistic for off-peak execution

    # Gas cost in ETH, then convert to USD
    gas_cost_eth = (GAS_UNITS * GAS_GWEI) / 1e9  # ETH
    merged["gas_cost_usd"] = gas_cost_eth * merged["price"]  # ~$13-15 at current prices

    # Slippage estimation
    # Formula: slippage = trade_size / liquidity * impact_factor
    # Flash loan arbitrage typically uses larger sizes ($50k-100k)
    TRADE_SIZE_USD = 50000  # $50k trade size (flash loan)
    BASE_SLIPPAGE_PCT = 0.05  # 0.05% base slippage (optimized routing)

    # Higher volume = lower slippage (inverse relationship)
    # Slippage increases when volume_total is low
    merged["slippage_pct"] = BASE_SLIPPAGE_PCT * (1 + 10000 / (merged["volume_total"] + 1000))
    merged["slippage_pct"] = merged["slippage_pct"].clip(upper=1.0)  # Cap at 1%

    # ==========================================
    # REALISTIC PROFIT CALCULATION
    # ==========================================

    # Fixed costs (percentage)
    DEX_FEE_005 = 0.05   # 0.05% fee tier
    DEX_FEE_03 = 0.30    # 0.3% fee tier
    FLASH_LOAN_FEE = 0.09  # Aave flash loan fee
    TOTAL_DEX_FEE = DEX_FEE_005 + DEX_FEE_03 + FLASH_LOAN_FEE  # 0.44%

    # Gas cost as percentage of trade
    merged["gas_cost_pct"] = (merged["gas_cost_usd"] / TRADE_SIZE_USD) * 100

    # Total cost percentage
    merged["total_cost_pct"] = TOTAL_DEX_FEE + merged["gas_cost_pct"] + merged["slippage_pct"]

    # Net profit after all costs
    merged["net_profit_pct"] = merged["spread_abs"] - merged["total_cost_pct"]
    merged["expected_profit"] = merged["net_profit_pct"].clip(lower=0)

    # Binary label: profitable after ALL costs
    merged["profitable"] = (merged["net_profit_pct"] > 0).astype(int)

    # Also create regression target (basis points)
    merged["profit_bps"] = merged["net_profit_pct"] * 100

    # Time features
    merged["hour"] = merged["datetime"].dt.hour
    merged["day_of_week"] = merged["datetime"].dt.dayofweek
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)

    # Drop warmup period
    merged = merged.dropna().reset_index(drop=True)

    # Stats
    profitable_pct = merged["profitable"].mean() * 100
    avg_gas = merged["gas_cost_usd"].mean()
    avg_slippage = merged["slippage_pct"].mean()
    avg_total_cost = merged["total_cost_pct"].mean()

    print(f"   ✅ {len(merged):,} rows")
    print(f"   💰 Profitable: {profitable_pct:.1f}%")
    print(f"   ⛽ Avg gas cost: ${avg_gas:.2f} ({merged['gas_cost_pct'].mean():.3f}%)")
    print(f"   📉 Avg slippage: {avg_slippage:.3f}%")
    print(f"   💸 Avg total cost: {avg_total_cost:.3f}%")

    return merged


async def main(days: int = 30):
    """Main collection function."""
    print("\n" + "="*60)
    print("🚀 5-MINUTE INTERVAL DATA COLLECTION")
    print(f"   Target: {days} days × 288 intervals = ~{days * 288:,} rows per pool")
    print("="*60)

    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path("data/checkpoints")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Collect both pools with checkpoint support
    df_005 = await collect_pool_data(POOL_005, "ETH_USDC_005", days, checkpoint_path)
    df_03 = await collect_pool_data(POOL_03, "ETH_USDC_03", days, checkpoint_path)

    if df_005.empty or df_03.empty:
        print("❌ Failed to collect data")
        return

    # Save raw data
    df_005.to_csv(output_path / f"swaps_5min_005_{days}d.csv", index=False)
    df_03.to_csv(output_path / f"swaps_5min_03_{days}d.csv", index=False)

    # Calculate spreads and labels
    training_df = calculate_spreads_and_labels(df_005, df_03)

    # Save training data
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    filepath = processed_path / f"training_data_5min_{days}d.csv"
    training_df.to_csv(filepath, index=False)

    print("\n" + "="*60)
    print("✅ COLLECTION COMPLETE")
    print("="*60)
    print(f"\n📁 Output: {filepath}")
    print(f"   • Rows: {len(training_df):,}")
    print(f"   • Features: {len(training_df.columns)}")
    print(f"   • Date range: {training_df['datetime'].min()} to {training_df['datetime'].max()}")
    print(f"   • Profitable: {training_df['profitable'].sum()} ({training_df['profitable'].mean()*100:.1f}%)")

    # Sample
    print("\n📊 Sample:")
    print(training_df[["datetime", "price", "spread_abs", "volume_total", "profitable"]].tail(10))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Days to collect")
    args = parser.parse_args()

    asyncio.run(main(days=args.days))

