#!/usr/bin/env python3
"""
Prepare Training Data for Arbitrage Model.

Takes raw Uniswap data and creates:
1. Spread calculations between pools
2. Feature engineering
3. Label calculation (profitable trade or not)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_data() -> pd.DataFrame:
    """Load raw Uniswap data."""
    data_path = Path("data/raw/uniswap_all_pools.csv")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price spreads between ETH/USDC pools."""
    print("📊 Calculating spreads...")
    
    # Pivot to get pools side by side
    eth_usdc_005 = df[df["pool"] == "ETH_USDC_005"][["date", "close", "volumeUSD", "tvlUSD", "feesUSD"]].copy()
    eth_usdc_03 = df[df["pool"] == "ETH_USDC_03"][["date", "close", "volumeUSD", "tvlUSD", "feesUSD"]].copy()
    
    eth_usdc_005.columns = ["date", "close_005", "volume_005", "tvl_005", "fees_005"]
    eth_usdc_03.columns = ["date", "close_03", "volume_03", "tvl_03", "fees_03"]
    
    # Merge on date
    merged = pd.merge(eth_usdc_005, eth_usdc_03, on="date", how="inner")
    
    # Calculate spread (price difference as percentage)
    merged["spread"] = (merged["close_005"] - merged["close_03"]) / merged["close_03"] * 100
    merged["spread_abs"] = merged["spread"].abs()
    
    # Add ETH/USDT for more spread opportunities
    eth_usdt = df[df["pool"] == "ETH_USDT_03"][["date", "close", "volumeUSD", "tvlUSD"]].copy()
    eth_usdt.columns = ["date", "close_usdt", "volume_usdt", "tvl_usdt"]
    
    merged = pd.merge(merged, eth_usdt, on="date", how="left")
    
    # Spread between USDC and USDT pools
    merged["spread_usdc_usdt"] = (merged["close_005"] - merged["close_usdt"]) / merged["close_usdt"] * 100
    merged["spread_usdc_usdt"] = merged["spread_usdc_usdt"].fillna(0)
    
    print(f"   ✅ {len(merged)} days with spread data")
    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features."""
    print("🔧 Adding features...")
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Price features
    df["price"] = df["close_005"]  # Use 0.05% pool as reference
    df["price_pct_change"] = df["price"].pct_change() * 100
    df["price_ma7"] = df["price"].rolling(7).mean()
    df["price_ma30"] = df["price"].rolling(30).mean()
    df["price_std7"] = df["price"].rolling(7).std()
    
    # Volume features
    df["volume_total"] = df["volume_005"] + df["volume_03"] + df["volume_usdt"].fillna(0)
    df["volume_ma7"] = df["volume_total"].rolling(7).mean()
    df["volume_ratio"] = df["volume_total"] / df["volume_ma7"]
    
    # Liquidity features
    df["tvl_total"] = df["tvl_005"] + df["tvl_03"] + df["tvl_usdt"].fillna(0)
    df["tvl_ma7"] = df["tvl_total"].rolling(7).mean()
    
    # Spread features
    df["spread_ma7"] = df["spread_abs"].rolling(7).mean()
    df["spread_std7"] = df["spread_abs"].rolling(7).std()
    df["spread_zscore"] = (df["spread_abs"] - df["spread_ma7"]) / df["spread_std7"]
    
    # Volatility
    df["volatility_7d"] = df["price_pct_change"].rolling(7).std()
    df["volatility_30d"] = df["price_pct_change"].rolling(30).std()
    
    # Day of week (market patterns)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Fee ratio
    df["fee_ratio"] = (df["fees_005"] + df["fees_03"]) / df["volume_total"]
    
    print(f"   ✅ {len(df.columns)} features")
    return df


def calculate_labels(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Calculate arbitrage profitability labels.
    
    Label = 1 if spread > gas + fees + slippage
    """
    print("🏷️ Calculating labels...")
    
    # Assume gas cost ~0.03% of trade at current ETH prices
    # Fees: 0.05% + 0.3% = 0.35% for a round trip
    # Slippage: ~0.02% assumed
    TOTAL_COST_PCT = 0.03 + 0.35 + 0.02  # ~0.4%
    
    # Profitable if spread covers costs + threshold
    df["profitable"] = (df["spread_abs"] > (TOTAL_COST_PCT + threshold)).astype(int)
    
    # Profit amount (in percentage)
    df["expected_profit_pct"] = df["spread_abs"] - TOTAL_COST_PCT
    df["expected_profit_pct"] = df["expected_profit_pct"].clip(lower=0)
    
    profitable_days = df["profitable"].sum()
    print(f"   ✅ {profitable_days} profitable days ({profitable_days/len(df)*100:.1f}%)")
    
    return df


def main():
    print("\n" + "="*60)
    print("🚀 PREPARE TRAINING DATA")
    print("="*60 + "\n")
    
    # Load data
    df = load_data()
    print(f"📁 Loaded {len(df)} rows")
    
    # Process
    df = calculate_spreads(df)
    df = add_features(df)
    df = calculate_labels(df)
    
    # Remove warmup period
    df = df.dropna().reset_index(drop=True)
    
    # Save
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / "training_data.csv"
    df.to_csv(filepath, index=False)
    
    print("\n" + "="*60)
    print("✅ TRAINING DATA READY")
    print("="*60)
    print(f"\n📁 Output: {filepath}")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Features: {len(df.columns)}")
    print(f"   • Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   • Profitable days: {df['profitable'].sum()} ({df['profitable'].mean()*100:.1f}%)")
    
    # Show sample
    print("\n📊 Sample Data:")
    print(df[["date", "price", "spread_abs", "volume_total", "profitable"]].tail(10).to_string())
    
    return df


if __name__ == "__main__":
    main()

