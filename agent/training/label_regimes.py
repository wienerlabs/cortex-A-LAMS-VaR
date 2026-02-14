#!/usr/bin/env python3
"""
Historical Data Regime Labeling

Labels all historical training data with market regime classification
(BULL, BEAR, SIDEWAYS) for regime-specific model validation.

Expected regime distribution based on historical data:
- 2022: Mostly BEAR (crypto winter)
- 2023 H1: SIDEWAYS (consolidation)
- 2023 H2: BULL (rally)
- 2024: BULL (continued rally)
- 2025: BULL/SIDEWAYS (mixed)

Usage:
    python label_regimes.py --input ./data/features.csv --output ./data/features_labeled.csv
    python label_regimes.py --input ./data --output ./data/labeled --batch
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.regimeDetector import (
    RegimeDetector,
    RegimeConfig,
    MarketRegime,
    label_data_with_regimes,
)


def detect_price_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the price column in a DataFrame."""
    # Common price column names in priority order
    candidates = [
        "close", "price", "close_price", "oracle_twap",
        "mark_price", "spot_price", "last_price", "avg_price",
    ]
    
    for col in candidates:
        if col in df.columns:
            return col
    
    # Look for columns with "price" in name
    price_cols = [c for c in df.columns if "price" in c.lower()]
    if price_cols:
        return price_cols[0]
    
    return None


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the timestamp column in a DataFrame."""
    candidates = ["timestamp", "datetime", "time", "date", "ts"]
    
    for col in candidates:
        if col in df.columns:
            return col
    
    return None


def label_single_file(
    input_path: str,
    output_path: str,
    config: Optional[RegimeConfig] = None,
    price_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
) -> Dict:
    """
    Label a single CSV file with regime data.
    
    Returns:
        Dictionary with labeling statistics
    """
    print(f"\n📊 Processing: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    original_rows = len(df)
    
    # Auto-detect columns if not specified
    if price_column is None:
        price_column = detect_price_column(df)
        if price_column is None:
            raise ValueError(f"Could not detect price column in {input_path}")
    
    if timestamp_column is None:
        timestamp_column = detect_timestamp_column(df)
    
    print(f"  📈 Using price column: {price_column}")
    if timestamp_column:
        print(f"  🕐 Using timestamp column: {timestamp_column}")
    
    # Label with regimes
    df_labeled = label_data_with_regimes(
        df,
        price_column=price_column,
        timestamp_column=timestamp_column,
        config=config,
    )
    
    # Calculate statistics
    regime_counts = df_labeled["regime"].value_counts()
    regime_pcts = (regime_counts / len(df_labeled) * 100).round(1)
    
    stats = {
        "input_file": input_path,
        "output_file": output_path,
        "total_rows": original_rows,
        "regimes": {
            regime: {
                "count": int(regime_counts.get(regime, 0)),
                "percentage": float(regime_pcts.get(regime, 0)),
            }
            for regime in ["BULL", "BEAR", "SIDEWAYS", "UNKNOWN"]
        },
        "labeled_at": datetime.now().isoformat(),
    }
    
    # Print statistics
    print(f"\n  📊 Regime Distribution:")
    for regime in ["BULL", "BEAR", "SIDEWAYS", "UNKNOWN"]:
        count = regime_counts.get(regime, 0)
        pct = regime_pcts.get(regime, 0)
        bar = "█" * int(pct / 5)
        print(f"    {regime:10s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Save labeled data
    df_labeled.to_csv(output_path, index=False)
    print(f"\n  ✅ Saved to: {output_path}")
    
    return stats


def label_batch_files(
    input_dir: str,
    output_dir: str,
    config: Optional[RegimeConfig] = None,
) -> List[Dict]:
    """
    Label all CSV files in a directory.
    
    Returns:
        List of statistics dictionaries for each file
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob("**/*.csv"))
    print(f"\n📁 Found {len(csv_files)} CSV files in {input_dir}")
    
    all_stats = []
    
    for csv_file in csv_files:
        # Create corresponding output path
        rel_path = csv_file.relative_to(input_path)
        out_file = output_path / rel_path.with_stem(rel_path.stem + "_labeled")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            stats = label_single_file(
                str(csv_file),
                str(out_file),
                config=config,
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  ❌ Error processing {csv_file}: {e}")
            all_stats.append({
                "input_file": str(csv_file),
                "error": str(e),
            })

    return all_stats


def print_summary(all_stats: List[Dict]) -> None:
    """Print summary of batch labeling."""
    print("\n" + "=" * 60)
    print("  LABELING SUMMARY")
    print("=" * 60)

    successful = [s for s in all_stats if "error" not in s]
    failed = [s for s in all_stats if "error" in s]

    print(f"\n  ✅ Successfully labeled: {len(successful)} files")
    print(f"  ❌ Failed: {len(failed)} files")

    if successful:
        # Aggregate regime statistics
        total_rows = sum(s.get("total_rows", 0) for s in successful)
        total_bull = sum(s.get("regimes", {}).get("BULL", {}).get("count", 0) for s in successful)
        total_bear = sum(s.get("regimes", {}).get("BEAR", {}).get("count", 0) for s in successful)
        total_sideways = sum(s.get("regimes", {}).get("SIDEWAYS", {}).get("count", 0) for s in successful)

        print(f"\n  📊 Aggregate Statistics:")
        print(f"    Total rows: {total_rows:,}")
        print(f"    BULL:     {total_bull:,} ({total_bull/total_rows*100:.1f}%)")
        print(f"    BEAR:     {total_bear:,} ({total_bear/total_rows*100:.1f}%)")
        print(f"    SIDEWAYS: {total_sideways:,} ({total_sideways/total_rows*100:.1f}%)")

    if failed:
        print(f"\n  ❌ Failed files:")
        for s in failed:
            print(f"    - {s['input_file']}: {s['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Label historical data with market regimes"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file or directory"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all CSV files in input directory"
    )
    parser.add_argument(
        "--price-column", "-p",
        default=None,
        help="Name of the price column (auto-detected if not specified)"
    )
    parser.add_argument(
        "--timestamp-column", "-t",
        default=None,
        help="Name of the timestamp column (auto-detected if not specified)"
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=30,
        help="Lookback window for regime detection (default: 30)"
    )
    parser.add_argument(
        "--bull-threshold",
        type=float,
        default=0.15,
        help="Return threshold for BULL classification (default: 0.15)"
    )
    parser.add_argument(
        "--bear-threshold",
        type=float,
        default=-0.15,
        help="Return threshold for BEAR classification (default: -0.15)"
    )

    args = parser.parse_args()

    # Create config
    config = RegimeConfig(
        return_window=args.window,
        volatility_window=args.window,
        bull_return_threshold=args.bull_threshold,
        bear_return_threshold=args.bear_threshold,
    )

    print("=" * 60)
    print("  MARKET REGIME LABELING")
    print("=" * 60)
    print(f"  Return window: {config.return_window} periods")
    print(f"  BULL threshold: >{config.bull_return_threshold*100:.0f}%")
    print(f"  BEAR threshold: <{config.bear_return_threshold*100:.0f}%")

    if args.batch:
        all_stats = label_batch_files(args.input, args.output, config)
        print_summary(all_stats)
    else:
        stats = label_single_file(
            args.input,
            args.output,
            config=config,
            price_column=args.price_column,
            timestamp_column=args.timestamp_column,
        )
        print("\n✅ Labeling complete!")


if __name__ == "__main__":
    main()

