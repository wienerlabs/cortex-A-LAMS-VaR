#!/usr/bin/env python3
"""
Enrich DeFiLlama lending data with estimated utilization and borrow APY.

Since DeFiLlama only provides supply APY and TVL, we estimate:
- Utilization rate: Based on typical lending market dynamics
- Borrow APY: Calculated from supply APY using standard lending formulas

Formula:
- Supply APY = Borrow APY * Utilization Rate * (1 - Reserve Factor)
- Typical reserve factor for Kamino: 10-15%
- Borrow APY ≈ Supply APY / (Utilization * 0.85)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add agent root to path
agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

DATA_DIR = agent_root / "data" / "lending"

# Typical reserve factors for different asset types
RESERVE_FACTORS = {
    'stablecoin': 0.10,  # 10% for stablecoins
    'sol': 0.15,  # 15% for SOL and SOL derivatives
    'other': 0.20,  # 20% for other assets
}

# Typical utilization ranges
UTILIZATION_RANGES = {
    'stablecoin': (0.60, 0.85),  # Stablecoins: 60-85%
    'sol': (0.40, 0.70),  # SOL: 40-70%
    'other': (0.30, 0.60),  # Other: 30-60%
}

# Stablecoin symbols
STABLECOINS = ['USDC', 'USDT', 'USDS', 'PYUSD', 'USDG', 'EURC', 'CASH', 'FDUSD', 'UXD']

# SOL and SOL derivatives
SOL_ASSETS = ['SOL', 'JITOSOL', 'MSOL', 'JUPSOL', 'DSOL', 'VSOL', 'BSOL', 'PSOL', 'JSOL',
              'BBSOL', 'HSOL', 'DFDVSOL', 'CGNTSOL', 'BONKSOL', 'XBTC', 'FWDSOL',
              'LAINESOL', 'STRONGSOL', 'LANTERNSOL', 'NXSOL', 'PICOSOL', 'CDCSOL',
              'HUBSOL', 'STKESOL', 'BNSOL', 'ADRASOL']

# Asset tiers (1 = highest quality, 3 = lowest)
ASSET_TIERS = {
    # Tier 1: Major stablecoins and SOL
    'USDC': 1, 'USDT': 1, 'SOL': 1,

    # Tier 2: Liquid staked SOL and major stablecoins
    'JITOSOL': 2, 'MSOL': 2, 'PYUSD': 2, 'USDS': 2,

    # Tier 3: Everything else
}


def classify_asset(symbol: str) -> str:
    """Classify asset type."""
    if symbol in STABLECOINS:
        return 'stablecoin'
    elif symbol in SOL_ASSETS:
        return 'sol'
    else:
        return 'other'


def estimate_utilization(asset_type: str, supply_apy: float) -> float:
    """
    Estimate utilization rate based on asset type and supply APY.
    Higher supply APY typically indicates higher utilization.
    """
    min_util, max_util = UTILIZATION_RANGES[asset_type]
    
    # Normalize supply APY to 0-1 range (assuming max supply APY of 20%)
    normalized_apy = min(supply_apy / 20.0, 1.0)
    
    # Linear interpolation between min and max utilization
    estimated_util = min_util + (max_util - min_util) * normalized_apy
    
    # Add some randomness to make it more realistic (±5%)
    noise = np.random.uniform(-0.05, 0.05)
    estimated_util = np.clip(estimated_util + noise, 0.0, 0.95)
    
    return estimated_util


def calculate_borrow_apy(supply_apy: float, utilization: float, reserve_factor: float) -> float:
    """
    Calculate borrow APY from supply APY using lending market formula.
    
    Supply APY = Borrow APY * Utilization * (1 - Reserve Factor)
    Borrow APY = Supply APY / (Utilization * (1 - Reserve Factor))
    """
    if utilization < 0.01:  # Avoid division by zero
        return 0.0
    
    borrow_apy = supply_apy / (utilization * (1 - reserve_factor))
    
    # Sanity check: borrow APY should be higher than supply APY
    borrow_apy = max(borrow_apy, supply_apy * 1.1)
    
    return borrow_apy


def enrich_data(input_file: str, output_file: str = None):
    """Enrich DeFiLlama data with estimated utilization and borrow APY."""
    print(f"\n📊 Enriching lending data from {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} rows")
    
    # Classify assets
    df['asset_type'] = df['asset'].apply(classify_asset)
    
    # Estimate utilization
    df['utilization_rate'] = df.apply(
        lambda row: estimate_utilization(row['asset_type'], row['supply_apy']),
        axis=1
    )
    
    # Calculate borrow APY
    df['reserve_factor'] = df['asset_type'].map(RESERVE_FACTORS)
    df['borrow_apy'] = df.apply(
        lambda row: calculate_borrow_apy(
            row['supply_apy'],
            row['utilization_rate'],
            row['reserve_factor']
        ),
        axis=1
    )
    
    # Calculate total borrows and available liquidity
    df['total_borrows'] = df['tvl_usd'] * df['utilization_rate']
    df['available_liquidity'] = df['tvl_usd'] - df['total_borrows']

    # Add protocol_tvl_usd (same as tvl_usd for now, since all data is from Kamino)
    df['protocol_tvl_usd'] = df['tvl_usd']

    # Add asset tier
    df['asset_tier'] = df['asset'].map(ASSET_TIERS).fillna(3).astype(int)

    # Add total_supply (TVL in USD)
    df['total_supply'] = df['tvl_usd']

    # Add total_borrow (calculated from utilization)
    df['total_borrow'] = df['total_borrows']

    # Drop temporary columns
    df = df.drop(columns=['asset_type', 'reserve_factor'])
    
    # Save enriched data
    if output_file is None:
        output_file = input_file.replace('.csv', '_enriched.csv')
    
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved enriched data to {output_file}")
    print(f"   Total rows: {len(df)}")
    print(f"   New columns: utilization_rate, borrow_apy, total_borrows, available_liquidity")
    
    # Show statistics
    print(f"\n📊 Enriched Data Statistics:")
    print(df[['supply_apy', 'borrow_apy', 'utilization_rate', 'tvl_usd']].describe())
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich DeFiLlama lending data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", help="Output CSV file (default: input_enriched.csv)")
    
    args = parser.parse_args()
    
    enrich_data(args.input, args.output)
    print("\n✅ Data enrichment complete!")


if __name__ == "__main__":
    main()

