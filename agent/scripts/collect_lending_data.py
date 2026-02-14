#!/usr/bin/env python3
"""
Solana Lending Protocol Data Collection

Collects historical lending data from MarginFi, Kamino, and Solend for ML training.
Target: 90 days of hourly snapshots for each protocol.

Data collected:
- Supply APY
- Borrow APY
- Utilization rate
- Protocol TVL
- Available liquidity
- Asset-specific metrics (SOL, USDC, USDT, etc.)

Data sources:
- MarginFi: On-chain data via RPC + marginfi-client-v2
- Kamino: Kamino API + on-chain data
- Solend: Solend API + on-chain data
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Optional

# Add agent root to path
agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

from src.config import settings

# Output directory
DATA_DIR = agent_root / "data" / "lending"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Collection parameters
COLLECTION_DAYS = 90  # 90 days of historical data
SNAPSHOT_INTERVAL_HOURS = 1  # Hourly snapshots
TOTAL_SNAPSHOTS = COLLECTION_DAYS * 24

# Supported assets
ASSETS = {
    'SOL': 'So11111111111111111111111111111111111111112',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
    'JitoSOL': 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
    'mSOL': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
}

# Protocol configurations
PROTOCOLS = {
    'marginfi': {
        'name': 'MarginFi',
        'api_base': None,  # Uses on-chain data
        'enabled': True
    },
    'kamino': {
        'name': 'Kamino',
        'api_base': 'https://api.kamino.finance',
        'enabled': True
    },
    'solend': {
        'name': 'Solend',
        'api_base': 'https://api.solend.fi',
        'enabled': True
    }
}


class LendingDataCollector:
    """Collects lending data from Solana protocols."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.data: List[Dict[str, Any]] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_kamino_data(self) -> List[Dict[str, Any]]:
        """
        Collect REAL data from Kamino Finance API.

        Uses official Kamino API: https://api.kamino.finance
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        print("\n📊 Collecting Kamino REAL on-chain data...")
        data_points = []

        try:
            # Kamino lending markets endpoint (REAL API)
            url = f"{PROTOCOLS['kamino']['api_base']}/kamino-market"
            params = {'env': 'mainnet-beta'}

            async with self.session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    markets_data = await resp.json()

                    # Process each market
                    for market in markets_data:
                        market_name = market.get('name', 'Unknown')
                        reserves = market.get('reserves', [])
                        market_tvl = float(market.get('totalSupplyUsd', 0))

                        for reserve in reserves:
                            asset_symbol = reserve.get('symbol', 'UNKNOWN')

                            # Only collect data for supported assets
                            if asset_symbol not in ASSETS:
                                continue

                            # Extract REAL on-chain data
                            supply_apy = float(reserve.get('supplyApy', 0))
                            borrow_apy = float(reserve.get('borrowApy', 0))
                            total_supply = float(reserve.get('totalSupply', 0))
                            total_borrow = float(reserve.get('totalBorrow', 0))
                            utilization = total_borrow / total_supply if total_supply > 0 else 0

                            data_point = {
                                'timestamp': datetime.now(timezone.utc),
                                'protocol': 'kamino',
                                'asset': asset_symbol,
                                'supply_apy': supply_apy,
                                'borrow_apy': borrow_apy,
                                'utilization_rate': utilization,
                                'total_supply': total_supply,
                                'total_borrow': total_borrow,
                                'available_liquidity': total_supply - total_borrow,
                                'protocol_tvl_usd': market_tvl,
                                'market_name': market_name,
                            }

                            data_points.append(data_point)
                            print(f"   ✓ {asset_symbol}: APY={supply_apy:.2%}, "
                                  f"Util={utilization:.1%}, TVL=${market_tvl:,.0f}")
                else:
                    print(f"   ✗ Kamino API error: {resp.status}")
                    # Try to read error message
                    try:
                        error_text = await resp.text()
                        print(f"      Error: {error_text[:200]}")
                    except:
                        pass

        except Exception as e:
            print(f"   ✗ Kamino collection error: {e}")

        return data_points

    async def collect_solend_data(self) -> List[Dict[str, Any]]:
        """
        Collect data from Solend API.

        Solend provides market data via their API.
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        print("\n📊 Collecting Solend data...")
        data_points = []

        try:
            # Solend markets endpoint
            url = f"{PROTOCOLS['solend']['api_base']}/v1/markets/configs"

            async with self.session.get(url, timeout=30) as resp:
                if resp.status == 200:
                    markets_data = await resp.json()

                    # Process each market
                    for market in markets_data:
                        reserves = market.get('reserves', [])
                        market_name = market.get('name', 'Unknown')

                        for reserve in reserves:
                            asset_symbol = reserve.get('symbol', 'UNKNOWN')

                            # Only collect data for supported assets
                            if asset_symbol not in ASSETS:
                                continue

                            # Calculate utilization
                            total_supply = float(reserve.get('totalSupply', 0))
                            total_borrow = float(reserve.get('totalBorrow', 0))
                            utilization = total_borrow / total_supply if total_supply > 0 else 0

                            data_point = {
                                'timestamp': datetime.now(timezone.utc),
                                'protocol': 'solend',
                                'asset': asset_symbol,
                                'supply_apy': float(reserve.get('supplyInterest', 0)),
                                'borrow_apy': float(reserve.get('borrowInterest', 0)),
                                'utilization_rate': utilization,
                                'total_supply': total_supply,
                                'total_borrow': total_borrow,
                                'available_liquidity': total_supply - total_borrow,
                                'protocol_tvl_usd': float(market.get('totalSupplyUsd', 0)),
                                'market_name': market_name,
                            }

                            data_points.append(data_point)
                            print(f"   ✓ {asset_symbol}: APY={data_point['supply_apy']:.2%}, "
                                  f"Util={data_point['utilization_rate']:.1%}")
                else:
                    print(f"   ✗ Solend API error: {resp.status}")

        except Exception as e:
            print(f"   ✗ Solend collection error: {e}")

        return data_points

    async def collect_marginfi_data(self) -> List[Dict[str, Any]]:
        """
        Collect data from MarginFi.

        Note: MarginFi doesn't have a public REST API for historical data.
        We'll use on-chain data or a third-party aggregator.
        For now, we'll use placeholder logic and recommend using Dialect API.
        """
        print("\n📊 Collecting MarginFi data...")
        data_points = []

        try:
            # Option 1: Use Dialect Markets API (recommended)
            # https://docs.dialect.to/documentation/markets-api
            # This provides unified lending data across protocols

            # Option 2: Query on-chain data directly (more complex)
            # Would require using marginfi-client-v2 SDK

            # For now, we'll use a placeholder
            print("   ⚠️ MarginFi collection requires on-chain queries or Dialect API")
            print("   💡 Recommendation: Use Dialect Markets API for unified data")

            # Placeholder data structure
            for asset_symbol in ['SOL', 'USDC', 'USDT']:
                data_point = {
                    'timestamp': datetime.now(timezone.utc),
                    'protocol': 'marginfi',
                    'asset': asset_symbol,
                    'supply_apy': 0.0,  # Would come from on-chain data
                    'borrow_apy': 0.0,
                    'utilization_rate': 0.0,
                    'total_supply': 0.0,
                    'total_borrow': 0.0,
                    'available_liquidity': 0.0,
                    'protocol_tvl_usd': 0.0,
                }
                data_points.append(data_point)

        except Exception as e:
            print(f"   ✗ MarginFi collection error: {e}")

        return data_points

    async def collect_snapshot(self) -> List[Dict[str, Any]]:
        """Collect a single snapshot from all protocols."""
        all_data = []

        # Collect from each protocol
        if PROTOCOLS['kamino']['enabled']:
            kamino_data = await self.collect_kamino_data()
            all_data.extend(kamino_data)

        if PROTOCOLS['solend']['enabled']:
            solend_data = await self.collect_solend_data()
            all_data.extend(solend_data)

        if PROTOCOLS['marginfi']['enabled']:
            marginfi_data = await self.collect_marginfi_data()
            all_data.extend(marginfi_data)

        return all_data

    def save_data(self, data: List[Dict[str, Any]], filename: str = "lending_data.csv"):
        """Save collected data to CSV."""
        if not data:
            print("\n⚠️ No data to save")
            return

        df = pd.DataFrame(data)

        # Sort by timestamp and protocol
        df = df.sort_values(['timestamp', 'protocol', 'asset'])

        # Save to CSV
        output_path = DATA_DIR / filename
        df.to_csv(output_path, index=False)

        print(f"\n💾 Saved {len(df)} records to {output_path}")

        # Print summary statistics
        print("\n📊 Data Summary:")
        print(f"   Total records: {len(df)}")
        print(f"   Protocols: {df['protocol'].nunique()}")
        print(f"   Assets: {df['asset'].nunique()}")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Protocol breakdown
        print("\n   Protocol breakdown:")
        for protocol in df['protocol'].unique():
            count = len(df[df['protocol'] == protocol])
            print(f"   - {protocol}: {count} records")

        # Asset breakdown
        print("\n   Asset breakdown:")
        for asset in df['asset'].unique():
            count = len(df[df['asset'] == asset])
            avg_apy = df[df['asset'] == asset]['supply_apy'].mean()
            print(f"   - {asset}: {count} records, avg APY: {avg_apy:.2%}")


async def collect_single_snapshot():
    """Collect a single snapshot (for testing or daemon mode)."""
    print("=" * 60)
    print("SOLANA LENDING DATA COLLECTION - SINGLE SNAPSHOT")
    print("=" * 60)

    async with LendingDataCollector() as collector:
        data = await collector.collect_snapshot()

        if data:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"lending_snapshot_{timestamp}.csv"
            collector.save_data(data, filename)
        else:
            print("\n❌ No data collected")


async def collect_historical_simulation(days: int = 90):
    """
    Simulate historical data collection.

    Note: This creates synthetic historical data for testing.
    For real historical data, you would need:
    1. Access to historical API endpoints (if available)
    2. On-chain historical queries
    3. Third-party data providers (e.g., Dialect, DeFiLlama)
    """
    print("=" * 60)
    print(f"SOLANA LENDING DATA COLLECTION - {days} DAYS SIMULATION")
    print("=" * 60)
    print("\n⚠️ Note: This creates simulated historical data")
    print("   For real historical data, use Dialect Markets API or on-chain queries\n")

    all_data = []

    async with LendingDataCollector() as collector:
        # Collect current snapshot
        current_data = await collector.collect_snapshot()

        if not current_data:
            print("\n❌ Failed to collect current data")
            return

        # Generate historical data by adding timestamps
        print(f"\n📅 Generating {days} days of hourly data...")

        for day in range(days):
            for hour in range(24):
                # Calculate timestamp
                hours_ago = (days - day - 1) * 24 + (24 - hour - 1)
                timestamp = datetime.now(timezone.utc) - timedelta(hours=hours_ago)

                # Create data points with timestamp
                for data_point in current_data:
                    historical_point = data_point.copy()
                    historical_point['timestamp'] = timestamp

                    # Add some random variation to make it more realistic
                    # (In real scenario, this would be actual historical data)
                    variation = np.random.normal(1.0, 0.05)  # ±5% variation
                    historical_point['supply_apy'] *= variation
                    historical_point['borrow_apy'] *= variation
                    historical_point['utilization_rate'] *= np.random.normal(1.0, 0.1)
                    historical_point['utilization_rate'] = np.clip(
                        historical_point['utilization_rate'], 0, 1
                    )

                    all_data.append(historical_point)

            if (day + 1) % 10 == 0:
                print(f"   Generated {day + 1}/{days} days...")

        print(f"\n✅ Generated {len(all_data)} data points")

        # Save all data
        collector.save_data(all_data, "lending_historical_simulated.csv")


async def collect_using_dialect_api():
    """
    Collect data using Dialect Markets API.

    Dialect provides unified lending data across MarginFi, Kamino, and other protocols.
    https://docs.dialect.to/documentation/markets-api
    """
    print("=" * 60)
    print("SOLANA LENDING DATA COLLECTION - DIALECT API")
    print("=" * 60)

    print("\n💡 Dialect Markets API provides:")
    print("   - Real-time lending rates across MarginFi, Kamino, Solend")
    print("   - Historical APY tracking")
    print("   - Unified data format")
    print("\n📚 Documentation: https://docs.dialect.to/documentation/markets-api")
    print("\n⚠️ Requires Dialect API key (set DIALECT_API_KEY env variable)")

    # Check for API key
    dialect_api_key = settings.get('dialect_api_key')

    if not dialect_api_key:
        print("\n❌ DIALECT_API_KEY not set in environment")
        print("   Get a key at: https://dialect.to")
        return

    async with aiohttp.ClientSession() as session:
        try:
            # Dialect Markets API endpoint
            url = "https://api.dialect.to/v1/markets/lending"
            headers = {"Authorization": f"Bearer {dialect_api_key}"}

            async with session.get(url, headers=headers, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\n✅ Received data from Dialect API")
                    print(f"   Markets: {len(data.get('markets', []))}")

                    # Process and save data
                    # (Implementation depends on Dialect API response format)
                else:
                    print(f"\n❌ Dialect API error: {resp.status}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Solana lending protocol data for ML training"
    )
    parser.add_argument(
        "--mode",
        choices=["snapshot", "historical", "dialect"],
        default="snapshot",
        help="Collection mode"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of historical data to collect (for historical mode)"
    )

    args = parser.parse_args()

    if args.mode == "snapshot":
        await collect_single_snapshot()
    elif args.mode == "historical":
        await collect_historical_simulation(args.days)
    elif args.mode == "dialect":
        await collect_using_dialect_api()


if __name__ == "__main__":
    asyncio.run(main())



