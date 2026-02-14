#!/usr/bin/env python3
"""
Generate Synthetic Lending Data for Training

Creates realistic synthetic lending data based on typical Solana protocol parameters.
This is for testing and development purposes only.

Usage:
    python scripts/generate_synthetic_lending_data.py --days 90 --output data/lending/synthetic_lending_data.csv
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Realistic parameter ranges for Solana lending protocols
PROTOCOL_PARAMS = {
    'marginfi': {
        'SOL': {'base_apy': 0.035, 'apy_std': 0.015, 'base_util': 0.65, 'tvl': 120_000_000},
        'USDC': {'base_apy': 0.045, 'apy_std': 0.020, 'base_util': 0.75, 'tvl': 150_000_000},
        'USDT': {'base_apy': 0.040, 'apy_std': 0.018, 'base_util': 0.70, 'tvl': 80_000_000},
        'JitoSOL': {'base_apy': 0.055, 'apy_std': 0.025, 'base_util': 0.60, 'tvl': 60_000_000},
    },
    'kamino': {
        'SOL': {'base_apy': 0.040, 'apy_std': 0.018, 'base_util': 0.70, 'tvl': 180_000_000},
        'USDC': {'base_apy': 0.050, 'apy_std': 0.022, 'base_util': 0.78, 'tvl': 200_000_000},
        'USDT': {'base_apy': 0.042, 'apy_std': 0.019, 'base_util': 0.72, 'tvl': 90_000_000},
        'mSOL': {'base_apy': 0.048, 'apy_std': 0.020, 'base_util': 0.65, 'tvl': 70_000_000},
    },
    'solend': {
        'SOL': {'base_apy': 0.032, 'apy_std': 0.012, 'base_util': 0.68, 'tvl': 100_000_000},
        'USDC': {'base_apy': 0.038, 'apy_std': 0.015, 'base_util': 0.73, 'tvl': 130_000_000},
        'USDT': {'base_apy': 0.035, 'apy_std': 0.014, 'base_util': 0.71, 'tvl': 70_000_000},
    }
}


def generate_lending_data(days: int = 90, interval_hours: int = 1) -> pd.DataFrame:
    """
    Generate synthetic lending data.
    
    Args:
        days: Number of days of historical data
        interval_hours: Snapshot interval in hours
        
    Returns:
        DataFrame with lending data
    """
    print(f"Generating {days} days of synthetic lending data...")
    
    # Calculate timestamps
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_hours}H')
    
    data = []
    
    for timestamp in timestamps:
        for protocol, assets in PROTOCOL_PARAMS.items():
            for asset, params in assets.items():
                # Generate realistic values with some randomness
                supply_apy = max(0, np.random.normal(params['base_apy'], params['apy_std']))
                
                # Borrow APY is typically higher than supply APY
                borrow_apy = supply_apy * np.random.uniform(1.3, 1.8)
                
                # Utilization with some variation
                utilization = np.clip(
                    np.random.normal(params['base_util'], 0.10),
                    0.1, 0.95
                )
                
                # TVL with some variation
                tvl = params['tvl'] * np.random.uniform(0.8, 1.2)
                
                # Calculate supply and borrow based on utilization
                total_supply = tvl / 100  # Simplified
                total_borrow = total_supply * utilization
                available_liquidity = total_supply - total_borrow
                
                data.append({
                    'timestamp': timestamp,
                    'protocol': protocol,
                    'asset': asset,
                    'supply_apy': supply_apy,
                    'borrow_apy': borrow_apy,
                    'utilization_rate': utilization,
                    'total_supply': total_supply,
                    'total_borrow': total_borrow,
                    'available_liquidity': available_liquidity,
                    'protocol_tvl_usd': tvl,
                })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Generated {len(df)} records")
    print(f"   Protocols: {df['protocol'].nunique()}")
    print(f"   Assets: {df['asset'].nunique()}")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic lending data")
    parser.add_argument("--days", type=int, default=90, help="Days of data to generate")
    parser.add_argument("--interval", type=int, default=1, help="Interval in hours")
    parser.add_argument(
        "--output",
        type=str,
        default="data/lending/synthetic_lending_data.csv",
        help="Output CSV file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC LENDING DATA GENERATION")
    print("=" * 60)
    print()
    
    # Generate data
    df = generate_lending_data(days=args.days, interval_hours=args.interval)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved to: {output_path}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Average Supply APY: {df['supply_apy'].mean():.2%}")
    print(f"   Average Utilization: {df['utilization_rate'].mean():.1%}")
    print(f"   Average TVL: ${df['protocol_tvl_usd'].mean():,.0f}")
    
    print("\n✅ Synthetic data generation complete!")


if __name__ == "__main__":
    main()

