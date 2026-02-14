#!/usr/bin/env python3
"""
Cross-DEX Arbitrage Data Collection: Uniswap V3 vs Uniswap V2

Collects swap data from both DEX versions and calculates real arbitrage opportunities.
V2 and V3 have different price mechanisms, creating arbitrage opportunities.
Target: 365 days of 5-minute data for ETH/USDC pair.
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

from src.config import settings

# API URLs
UNISWAP_V3_URL = settings.get_uniswap_url()
UNISWAP_V2_URL = settings.get_uniswap_v2_url()

# Pool addresses
# V3: 0.05% fee pool (most liquid)
UNISWAP_V3_ETH_USDC = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
# V2: USDC/WETH pair (token0=USDC, token1=WETH)
UNISWAP_V2_ETH_USDC = "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"


def sqrt_price_to_price(sqrt_price_x96: str) -> float:
    """Convert Uniswap V3 sqrtPriceX96 to ETH price in USDC."""
    try:
        sqrt_price = int(sqrt_price_x96)
        raw_price = (sqrt_price / (2**96)) ** 2
        eth_price_usdc = (1 / raw_price) * (10 ** 12)
        return eth_price_usdc
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def v2_swap_to_price(swap: dict) -> float:
    """
    Get ETH price from V2 swap data.
    Uses token0Price from pair data (ETH price in USDC).
    """
    try:
        # token0Price = ETH price in USDC (already calculated by subgraph)
        return float(swap['pair']['token0Price'])
    except (KeyError, TypeError, ValueError):
        return 0.0


async def fetch_v3_swaps(session: aiohttp.ClientSession,
                         timestamp_gt: int, timestamp_lt: int, skip: int = 0) -> list:
    """Fetch Uniswap V3 swaps."""
    query = """
    query Swaps($pool: String!, $timestamp_gt: Int!, $timestamp_lt: Int!, $skip: Int!) {
      swaps(first: 1000, skip: $skip, orderBy: timestamp, orderDirection: asc,
            where: {pool: $pool, timestamp_gte: $timestamp_gt, timestamp_lt: $timestamp_lt}) {
        timestamp
        sqrtPriceX96
        amount0
        amount1
        amountUSD
      }
    }
    """
    variables = {
        "pool": UNISWAP_V3_ETH_USDC.lower(),
        "timestamp_gt": timestamp_gt,
        "timestamp_lt": timestamp_lt,
        "skip": skip
    }
    try:
        async with session.post(UNISWAP_V3_URL, json={"query": query, "variables": variables}) as resp:
            if resp.status == 200:
                data = await resp.json()
                if "errors" not in data:
                    return data.get("data", {}).get("swaps", [])
    except Exception as e:
        print(f"   ⚠️ V3 error: {e}")
    return []


async def fetch_v2_swaps(session: aiohttp.ClientSession,
                         timestamp_gt: int, timestamp_lt: int, skip: int = 0) -> list:
    """Fetch Uniswap V2 swaps with token0Price for ETH price."""
    query = """
    query Swaps($pair: String!, $timestamp_gt: Int!, $timestamp_lt: Int!, $skip: Int!) {
      swaps(first: 1000, skip: $skip, orderBy: timestamp, orderDirection: asc,
            where: {pair: $pair, timestamp_gte: $timestamp_gt, timestamp_lt: $timestamp_lt}) {
        timestamp
        amount0In
        amount0Out
        amount1In
        amount1Out
        amountUSD
        pair {
          token0Price
        }
      }
    }
    """
    variables = {
        "pair": UNISWAP_V2_ETH_USDC.lower(),
        "timestamp_gt": timestamp_gt,
        "timestamp_lt": timestamp_lt,
        "skip": skip
    }
    try:
        async with session.post(UNISWAP_V2_URL, json={"query": query, "variables": variables}) as resp:
            if resp.status == 200:
                data = await resp.json()
                if "errors" not in data:
                    return data.get("data", {}).get("swaps", [])
    except Exception as e:
        print(f"   ⚠️ V2 error: {e}")
    return []


async def fetch_day_data(session: aiohttp.ClientSession, date: datetime) -> tuple:
    """Fetch one day of data from both Uniswap V3 and V2."""
    start_ts = int(date.timestamp())
    end_ts = int((date + timedelta(days=1)).timestamp())

    # Uniswap V3 swaps
    v3_swaps = []
    skip = 0
    while True:
        swaps = await fetch_v3_swaps(session, start_ts, end_ts, skip)
        if not swaps:
            break
        v3_swaps.extend(swaps)
        skip += 1000
        if len(swaps) < 1000:
            break
        await asyncio.sleep(0.1)

    # Uniswap V2 swaps
    v2_swaps = []
    skip = 0
    while True:
        swaps = await fetch_v2_swaps(session, start_ts, end_ts, skip)
        if not swaps:
            break
        v2_swaps.extend(swaps)
        skip += 1000
        if len(swaps) < 1000:
            break
        await asyncio.sleep(0.1)

    return v3_swaps, v2_swaps


def aggregate_to_5min(v3_swaps: list, v2_swaps: list, date: datetime) -> pd.DataFrame:
    """
    Aggregate swaps to 5-minute intervals and calculate V3 vs V2 spreads.
    """
    rows = []

    # Create 5-minute buckets
    for interval_start in range(0, 86400, 300):  # 5 min = 300 sec
        bucket_start = int(date.timestamp()) + interval_start
        bucket_end = bucket_start + 300

        # Filter swaps in this bucket
        v3_bucket = [s for s in v3_swaps
                     if bucket_start <= int(s['timestamp']) < bucket_end]
        v2_bucket = [s for s in v2_swaps
                     if bucket_start <= int(s['timestamp']) < bucket_end]

        # Skip if no swaps in either version
        if not v3_bucket and not v2_bucket:
            continue

        # Calculate VWAP for V3
        v3_price = 0
        v3_volume = 0
        for s in v3_bucket:
            price = sqrt_price_to_price(s['sqrtPriceX96'])
            volume = abs(float(s.get('amountUSD', 0)))
            if price > 0 and volume > 0:
                v3_price += price * volume
                v3_volume += volume
        v3_vwap = v3_price / v3_volume if v3_volume > 0 else 0

        # Calculate VWAP for V2 (using token0Price)
        v2_price = 0
        v2_volume = 0
        for s in v2_bucket:
            try:
                price = v2_swap_to_price(s)
                volume = float(s.get('amountUSD', 0))
                if price > 0 and volume > 0:
                    v2_price += price * volume
                    v2_volume += volume
            except (KeyError, TypeError):
                continue
        v2_vwap = v2_price / v2_volume if v2_volume > 0 else 0

        # Skip if we don't have both prices
        if v3_vwap <= 0 or v2_vwap <= 0:
            continue

        # Calculate spread
        spread_abs = abs(v3_vwap - v2_vwap) / min(v3_vwap, v2_vwap) * 100

        # Direction: where to buy, where to sell
        if v3_vwap < v2_vwap:
            buy_dex = 'uniswap_v3'
            sell_dex = 'uniswap_v2'
        else:
            buy_dex = 'uniswap_v2'
            sell_dex = 'uniswap_v3'

        row = {
            'datetime': datetime.fromtimestamp(bucket_start, tz=timezone.utc),
            'v3_price': v3_vwap,
            'v2_price': v2_vwap,
            'spread_abs': spread_abs,
            'buy_dex': buy_dex,
            'sell_dex': sell_dex,
            'v3_volume': v3_volume,
            'v2_volume': v2_volume,
            'total_volume': v3_volume + v2_volume,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_costs_and_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate realistic costs and net profit for V3 vs V2 arbitrage.

    Costs:
    - Uniswap V3 fee: 0.05% (for ETH/USDC 0.05% pool)
    - Uniswap V2 fee: 0.30%
    - Flash loan fee (Aave): 0.09%
    - Gas: ~200k gas (same protocol, simpler routing)
    - Slippage: based on volume
    """
    if df.empty:
        return df

    df = df.copy()

    # DEX Fees (percentage)
    V3_FEE = 0.05   # 0.05%
    V2_FEE = 0.30   # 0.30%
    FLASH_LOAN_FEE = 0.09  # Aave flash loan

    # Gas cost (same protocol = lower gas)
    GAS_UNITS = 200000
    GAS_GWEI = 20
    ETH_PRICE = df['v3_price'].mean()  # Use average ETH price
    gas_cost_eth = (GAS_UNITS * GAS_GWEI * 1e-9)
    GAS_COST_USD = gas_cost_eth * ETH_PRICE

    # Trade size for percentage calculations
    TRADE_SIZE_USD = 50000

    # Total DEX fees (both sides)
    df['dex_fees_pct'] = V3_FEE + V2_FEE
    df['flash_loan_pct'] = FLASH_LOAN_FEE

    # Gas as percentage of trade
    df['gas_cost_usd'] = GAS_COST_USD
    df['gas_cost_pct'] = (GAS_COST_USD / TRADE_SIZE_USD) * 100

    # Slippage based on volume (lower volume = higher slippage)
    df['min_volume'] = df[['v3_volume', 'v2_volume']].min(axis=1)
    df['slippage_pct'] = np.clip(
        0.05 + (TRADE_SIZE_USD / (df['min_volume'] + 1000)) * 0.5,
        0.05, 1.0  # Min 0.05%, max 1%
    )

    # Total cost
    df['total_cost_pct'] = (
        df['dex_fees_pct'] +
        df['flash_loan_pct'] +
        df['gas_cost_pct'] +
        df['slippage_pct']
    )

    # Net profit
    df['net_profit_pct'] = df['spread_abs'] - df['total_cost_pct']

    # Profitable label
    df['profitable'] = (df['net_profit_pct'] > 0).astype(int)

    return df


async def collect_data(days: int = 365) -> pd.DataFrame:
    """Main collection function."""
    print("\n" + "="*60)
    print("🚀 CROSS-DEX ARBITRAGE DATA COLLECTION")
    print(f"   Uniswap V3 vs Uniswap V2 | ETH/USDC")
    print(f"   Target: {days} days of 5-minute data")
    print("="*60)

    checkpoint_dir = agent_root / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "cross_dex_checkpoint.csv"

    # Load checkpoint if exists
    all_data = []
    last_date = None
    if checkpoint_file.exists():
        checkpoint_df = pd.read_csv(checkpoint_file, parse_dates=['datetime'])
        all_data.append(checkpoint_df)
        last_date = checkpoint_df['datetime'].max().date()
        print(f"\n📁 Resuming from checkpoint: {len(checkpoint_df):,} rows")

    # Calculate date range
    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    if last_date:
        start_date = max(start_date, datetime.combine(last_date, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(days=1))

    if start_date >= end_date:
        print("\n✅ Data already up to date!")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    total_days = (end_date - start_date).days
    print(f"\n📊 Collecting {total_days} days: {start_date.date()} to {end_date.date()}")

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        current_date = start_date
        daily_dfs = []

        while current_date < end_date:
            print(f"\r   📅 {current_date.date()} ", end="", flush=True)

            try:
                v3_swaps, v2_swaps = await fetch_day_data(session, current_date)

                if v3_swaps or v2_swaps:
                    day_df = aggregate_to_5min(v3_swaps, v2_swaps, current_date)
                    if not day_df.empty:
                        daily_dfs.append(day_df)
                        print(f"✓ {len(day_df)} intervals, V3:{len(v3_swaps)} V2:{len(v2_swaps)}")
                    else:
                        print("(no overlapping data)")
                else:
                    print("(no swaps)")

            except Exception as e:
                print(f"⚠️ Error: {e}")

            current_date += timedelta(days=1)
            await asyncio.sleep(0.2)  # Rate limiting

            # Save checkpoint every 30 days
            if len(daily_dfs) >= 30:
                chunk = pd.concat(daily_dfs, ignore_index=True)
                all_data.append(chunk)
                combined = pd.concat(all_data, ignore_index=True)
                combined.to_csv(checkpoint_file, index=False)
                print(f"\n   💾 Checkpoint saved: {len(combined):,} rows")
                daily_dfs = []

        # Save remaining data
        if daily_dfs:
            chunk = pd.concat(daily_dfs, ignore_index=True)
            all_data.append(chunk)

    # Combine all data
    if not all_data:
        print("\n⚠️ No data collected!")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical and time features for ML."""
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Rolling statistics
    for window in [12, 24, 48]:  # 1h, 2h, 4h windows
        df[f'spread_ma_{window}'] = df['spread_abs'].rolling(window, min_periods=1).mean()
        df[f'spread_std_{window}'] = df['spread_abs'].rolling(window, min_periods=1).std().fillna(0)
        df[f'volume_ma_{window}'] = df['total_volume'].rolling(window, min_periods=1).mean()

    # Spread momentum
    df['spread_change'] = df['spread_abs'].diff().fillna(0)
    df['spread_pct_change'] = df['spread_abs'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # Volume ratio
    df['volume_ratio'] = df['v3_volume'] / (df['v2_volume'] + 1)

    # Price momentum (use V3 as reference)
    df['price_ma_12'] = df['v3_price'].rolling(12, min_periods=1).mean()
    df['price_std_12'] = df['v3_price'].rolling(12, min_periods=1).std().fillna(0)
    df['price_volatility'] = df['price_std_12'] / df['price_ma_12'] * 100

    return df


async def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Collect cross-DEX arbitrage data")
    parser.add_argument("--days", type=int, default=365, help="Number of days to collect")
    args = parser.parse_args()

    # Collect data
    df = await collect_data(args.days)

    if df.empty:
        print("\n❌ No data collected!")
        return

    # Calculate costs and profit
    print("\n📊 Calculating costs and profit...")
    df = calculate_costs_and_profit(df)

    # Add features
    print("🔧 Adding features...")
    df = add_features(df)

    # Save
    output_dir = agent_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"cross_dex_data_{args.days}d.csv"
    df.to_csv(output_file, index=False)

    # Final checkpoint
    checkpoint_file = agent_root / "data" / "checkpoints" / "cross_dex_checkpoint.csv"
    df.to_csv(checkpoint_file, index=False)

    # Summary
    print("\n" + "="*60)
    print("✅ COLLECTION COMPLETE")
    print("="*60)
    print(f"\n📁 Output: {output_file}")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Features: {len(df.columns)}")
    if 'datetime' in df.columns and not df.empty:
        print(f"   • Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    if 'profitable' in df.columns:
        profitable_count = df['profitable'].sum()
        profitable_pct = df['profitable'].mean() * 100
        print(f"   • Profitable: {profitable_count:,} ({profitable_pct:.1f}%)")
    if 'spread_abs' in df.columns:
        print(f"   • Avg spread: {df['spread_abs'].mean():.3f}%")
    if 'total_cost_pct' in df.columns:
        print(f"   • Avg cost: {df['total_cost_pct'].mean():.3f}%")

    # Show sample
    print("\n📊 Sample:")
    cols = ['datetime', 'v3_price', 'v2_price', 'spread_abs', 'net_profit_pct', 'profitable']
    print(df[cols].tail(10).to_string())


if __name__ == "__main__":
    asyncio.run(main())

