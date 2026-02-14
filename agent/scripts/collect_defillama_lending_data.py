#!/usr/bin/env python3
"""
DeFiLlama Lending Data Collection

Collects historical and real-time lending data from Kamino via DeFiLlama Yields API.
This is the PRIMARY data source for the lending strategy.

Data collected:
- Supply APY (apyBase)
- Reward APY (apyReward)
- Total APY (apy)
- TVL in USD
- Historical APY/TVL data (daily snapshots)

Data source:
- DeFiLlama Yields API (FREE - no API key required)
- Endpoint: https://yields.llama.fi/pools
- Historical: https://yields.llama.fi/chart/{pool_id}
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import time
from typing import Dict, List, Any, Optional

# Add agent root to path
agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

# Output directory
DATA_DIR = agent_root / "data" / "lending"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# DeFiLlama Yields API (FREE)
DEFILLAMA_API = "https://yields.llama.fi"

# Collection parameters
COLLECTION_DAYS = 90  # 90 days of historical data
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds


class DeFiLlamaLendingCollector:
    """Collects lending data from DeFiLlama Yields API."""
    
    def __init__(self):
        self.api_base = DEFILLAMA_API
        self.data: List[Dict[str, Any]] = []
        
    def collect_current_pools(self) -> List[Dict[str, Any]]:
        """Collect current APY/TVL data for all Kamino lending pools."""
        print("\n📊 Collecting current Kamino lending data from DeFiLlama...")
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = requests.get(
                    f"{self.api_base}/pools",
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # Filter for Kamino lending pools on Solana
                kamino_pools = [
                    pool for pool in data.get("data", [])
                    if pool.get("project") == "kamino-lend" and pool.get("chain") == "Solana"
                ]
                
                print(f"✅ Found {len(kamino_pools)} Kamino lending pools")
                
                # Transform to our format
                pools_data = []
                for pool in kamino_pools:
                    pools_data.append({
                        "protocol": "kamino",
                        "asset": pool.get("symbol", "Unknown"),
                        "supply_apy": pool.get("apyBase", 0),  # Supply APY (base)
                        "reward_apy": pool.get("apyReward", 0) if pool.get("apyReward") else 0,
                        "total_apy": pool.get("apy", 0),  # Total APY (base + reward)
                        "tvl_usd": pool.get("tvlUsd", 0),
                        "pool_id": pool.get("pool"),  # For historical data
                        "underlying_token": pool.get("underlyingTokens", [None])[0],
                        "stablecoin": pool.get("stablecoin", False),
                        "timestamp": datetime.now().isoformat()
                    })
                
                return pools_data
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print("❌ Failed to collect current pools data")
                    return []
        
        return []
    
    def collect_historical_pool_data(self, pool_id: str, pool_symbol: str, days: int = 90) -> pd.DataFrame:
        """Collect historical APY/TVL data for a specific pool."""
        print(f"  📈 Collecting {days} days of historical data for {pool_symbol}...")
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = requests.get(
                    f"{self.api_base}/chart/{pool_id}",
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "success":
                    print(f"  ❌ Failed to get historical data for {pool_symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(data.get("data", []))
                
                if df.empty:
                    print(f"  ⚠️  No historical data for {pool_symbol}")
                    return df
                
                # Parse timestamp (make timezone-aware)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

                # Filter to last N days (make cutoff timezone-aware)
                cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=days)
                df = df[df['timestamp'] >= cutoff_date]
                
                # Add pool metadata
                df['pool_id'] = pool_id
                df['asset'] = pool_symbol
                df['protocol'] = 'kamino'
                
                print(f"  ✅ Collected {len(df)} data points for {pool_symbol}")
                return df
                
            except Exception as e:
                print(f"  ❌ Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed for {pool_symbol}: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ Failed to collect historical data for {pool_symbol}")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def collect_all_historical_data(self, days: int = 90) -> pd.DataFrame:
        """Collect historical data for ALL Kamino lending pools."""
        print(f"\n🔄 Collecting {days} days of historical data for all Kamino pools...")

        # First get current pools to get pool IDs
        current_pools = self.collect_current_pools()

        if not current_pools:
            print("❌ No pools found")
            return pd.DataFrame()

        # Collect historical data for each pool
        all_historical_data = []
        for pool in current_pools:
            pool_id = pool.get("pool_id")
            pool_symbol = pool.get("asset")

            if not pool_id:
                print(f"  ⚠️  No pool ID for {pool_symbol}, skipping...")
                continue

            df = self.collect_historical_pool_data(pool_id, pool_symbol, days)
            if not df.empty:
                all_historical_data.append(df)

            # Rate limiting - be nice to the API
            time.sleep(0.5)

        if not all_historical_data:
            print("❌ No historical data collected")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_historical_data, ignore_index=True)

        # Rename columns to match our training format
        combined_df = combined_df.rename(columns={
            'apy': 'total_apy',
            'apyBase': 'supply_apy',
            'apyReward': 'reward_apy',
            'tvlUsd': 'tvl_usd'
        })

        # Fill missing values
        combined_df['supply_apy'] = combined_df['supply_apy'].fillna(combined_df['total_apy'])
        combined_df['reward_apy'] = combined_df['reward_apy'].fillna(0)

        print(f"\n✅ Collected {len(combined_df)} total data points across {len(all_historical_data)} pools")
        print(f"   Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

        return combined_df

    def save_data(self, df: pd.DataFrame, filename: str = None):
        """Save collected data to CSV."""
        if df.empty:
            print("⚠️  No data to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kamino_lending_defillama_{timestamp}.csv"

        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"\n💾 Saved {len(df)} rows to {filepath}")
        print(f"   File size: {filepath.stat().st_size / 1024:.2f} KB")


def main():
    """Main collection function."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect Kamino lending data from DeFiLlama")
    parser.add_argument("--days", type=int, default=90, help="Number of days of historical data to collect")
    parser.add_argument("--current-only", action="store_true", help="Only collect current pool data (no historical)")
    parser.add_argument("--output", type=str, help="Output filename (default: auto-generated)")

    args = parser.parse_args()

    collector = DeFiLlamaLendingCollector()

    if args.current_only:
        # Collect only current data
        pools = collector.collect_current_pools()
        df = pd.DataFrame(pools)
        collector.save_data(df, args.output)
    else:
        # Collect historical data (includes current data in the latest snapshot)
        df = collector.collect_all_historical_data(days=args.days)
        collector.save_data(df, args.output)

    print("\n✅ Data collection complete!")


if __name__ == "__main__":
    main()


