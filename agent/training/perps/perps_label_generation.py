#!/usr/bin/env python3
"""
Perpetuals Label Generation

Creates binary labels for funding rate arbitrage:
- Label = 1 (TRADE): |funding_rate| > threshold (profitable to collect funding)
- Label = 0 (NO TRADE): |funding_rate| <= threshold (not worth trading)

When funding_rate > +0.15%: SHORT is profitable (shorts receive funding from longs)
When funding_rate < -0.15%: LONG is profitable (longs receive funding from shorts)

Usage:
    python perps_label_generation.py --input ./features/perps_features.csv --threshold 0.15
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_labels(
    features_path: str,
    output_path: str = None,
    threshold_pct: float = 0.15,
) -> pd.DataFrame:
    """
    Generate binary labels for funding rate arbitrage.
    
    Args:
        features_path: Path to features CSV
        output_path: Path to save labeled features (default: same as input)
        threshold_pct: Funding rate threshold in percentage (default: 0.15%)
    
    Returns:
        DataFrame with labels added
    """
    print("=" * 60)
    print("  PERPS LABEL GENERATION")
    print("=" * 60)
    
    # Load features
    print(f"\n📊 Loading features from {features_path}")
    df = pd.read_csv(features_path)
    print(f"  Samples: {len(df)}")
    
    # Check for target column
    if "target_funding_rate" not in df.columns:
        raise ValueError("Missing 'target_funding_rate' column in features")
    
    target = df["target_funding_rate"]
    threshold = threshold_pct / 100  # Convert to decimal (0.15% -> 0.0015)
    
    print(f"\n🎯 Label Generation:")
    print(f"  Threshold: ±{threshold_pct}% (±{threshold:.6f})")
    print(f"  Target range: [{target.min():.6f}, {target.max():.6f}]")
    
    # Generate labels
    # Label = 1 if |funding_rate| > threshold (profitable trade opportunity)
    # Label = 0 otherwise (no trade)
    df["label"] = (np.abs(target) > threshold).astype(int)
    
    # Also add direction for informational purposes
    # direction = 1 (SHORT): funding > threshold (shorts collect)
    # direction = -1 (LONG): funding < -threshold (longs collect)
    # direction = 0: no trade
    df["trade_direction"] = np.where(
        target > threshold, 1,  # SHORT
        np.where(target < -threshold, -1, 0)  # LONG or NO TRADE
    )
    
    # Calculate label distribution
    label_counts = df["label"].value_counts().sort_index()
    total = len(df)
    
    print(f"\n📈 Label Distribution:")
    print(f"  Label 0 (NO TRADE): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total:.1%})")
    print(f"  Label 1 (TRADE):    {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total:.1%})")
    
    # Direction breakdown
    dir_counts = df["trade_direction"].value_counts().sort_index()
    print(f"\n📊 Trade Direction Breakdown:")
    print(f"  LONG  (-1): {dir_counts.get(-1, 0):,}")
    print(f"  NONE  (0):  {dir_counts.get(0, 0):,}")
    print(f"  SHORT (1):  {dir_counts.get(1, 0):,}")
    
    # Class balance
    if label_counts.get(1, 0) > 0:
        imbalance_ratio = label_counts.get(0, 0) / label_counts.get(1, 0)
        print(f"\n⚖️ Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Save
    if output_path is None:
        output_path = features_path  # Overwrite
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved labeled features to {output_path}")
    
    # Summary statistics for trade opportunities
    trade_df = df[df["label"] == 1]
    if len(trade_df) > 0:
        print(f"\n📊 Trade Opportunity Statistics:")
        print(f"  Mean |funding| when trading: {np.abs(trade_df['target_funding_rate']).mean():.6f}")
        print(f"  Max |funding| when trading:  {np.abs(trade_df['target_funding_rate']).max():.6f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate labels for perps model")
    parser.add_argument("--input", type=str, default="./features/perps_features.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: overwrite input)")
    parser.add_argument("--threshold", type=float, default=0.15, help="Funding rate threshold in %%")
    args = parser.parse_args()
    
    generate_labels(
        features_path=args.input,
        output_path=args.output,
        threshold_pct=args.threshold,
    )


if __name__ == "__main__":
    main()

