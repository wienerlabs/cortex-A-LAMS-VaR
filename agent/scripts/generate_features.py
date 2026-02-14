#!/usr/bin/env python3
"""
Feature Engineering for LP Rebalancer ML Model

Generates features from historical pool data:
1. APY features (current, MA, trends, volatility)
2. Volume features (vol/TVL ratio, trends)
3. TVL features (stability, trends)
4. IL features (estimated impermanent loss)
5. Token volatility
6. Time features (hour, day of week)
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

DATA_DIR = Path(__file__).parent.parent / "data" / "lp_rebalancer" / "historical"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "lp_rebalancer" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_progress(step: int, total: int, desc: str):
    """Print progress bar."""
    pct = step / total * 100
    bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"   [{bar}] {step}/{total} - {desc}")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Load historical data files."""
    print("\n📂 Loading historical data...")
    
    # Token prices
    with open(DATA_DIR / "token_prices_30d.json") as f:
        tokens_raw = json.load(f)
    
    # Pool OHLCV
    with open(DATA_DIR / "pool_ohlcv_30d.json") as f:
        pools_raw = json.load(f)
    
    # Protocol TVL
    with open(DATA_DIR / "defillama_protocol_tvl.json") as f:
        protocol_tvl = json.load(f)
    
    # Convert token prices to DataFrame
    token_dfs = []
    for t in tokens_raw:
        if t["data"]:
            df = pd.DataFrame(t["data"])
            df["token"] = t["token_symbol"]
            df["timestamp"] = pd.to_datetime(df["unixTime"], unit="s", utc=True)
            token_dfs.append(df)
    
    token_df = pd.concat(token_dfs, ignore_index=True) if token_dfs else pd.DataFrame()
    print(f"   ✅ Token prices: {len(token_df)} rows")
    
    # Convert pool OHLCV to DataFrame
    pool_dfs = []
    for p in pools_raw:
        if p["candles"] > 0 and p["data"]:
            df = pd.DataFrame(p["data"])
            df["pool_address"] = p["pool_address"]
            df["pool_name"] = p["pool_name"]
            df["dex"] = p["dex"]
            df["timestamp"] = pd.to_datetime(df["unixTime"], unit="s", utc=True)
            pool_dfs.append(df)
    
    pool_df = pd.concat(pool_dfs, ignore_index=True) if pool_dfs else pd.DataFrame()
    print(f"   ✅ Pool OHLCV: {len(pool_df)} rows ({len(pool_dfs)} pools)")
    
    return token_df, pool_df, protocol_tvl


def calculate_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage returns."""
    return series.pct_change(periods) * 100


def calculate_volatility(series: pd.Series, window: int = 24) -> pd.Series:
    """Calculate rolling volatility (std of returns)."""
    returns = calculate_returns(series)
    return returns.rolling(window=window, min_periods=1).std()


def calculate_ma(series: pd.Series, window: int) -> pd.Series:
    """Calculate moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def calculate_trend(series: pd.Series, window: int = 168) -> pd.Series:
    """Calculate trend as slope of linear regression over window."""
    def slope(x):
        if len(x) < 2:
            return 0
        return np.polyfit(range(len(x)), x, 1)[0]
    return series.rolling(window=window, min_periods=2).apply(slope, raw=True)


def estimate_il(price_ratio_change: float) -> float:
    """
    Estimate impermanent loss from price ratio change.
    IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
    """
    if price_ratio_change <= -1:
        return -1.0
    ratio = 1 + price_ratio_change
    if ratio <= 0:
        return -1.0
    il = 2 * np.sqrt(ratio) / (1 + ratio) - 1
    return il * 100  # Return as percentage


def generate_token_features(token_df: pd.DataFrame) -> pd.DataFrame:
    """Generate token-level features."""
    print("\n📊 Generating token features...")
    
    features = []
    for token in token_df["token"].unique():
        df = token_df[token_df["token"] == token].copy()
        df = df.sort_values("timestamp")
        
        # Price features
        df[f"{token}_price"] = df["c"]  # Close price
        df[f"{token}_return_1h"] = calculate_returns(df["c"], 1)
        df[f"{token}_return_24h"] = calculate_returns(df["c"], 24)
        df[f"{token}_volatility_24h"] = calculate_volatility(df["c"], 24)
        df[f"{token}_volatility_168h"] = calculate_volatility(df["c"], 168)
        df[f"{token}_ma_6h"] = calculate_ma(df["c"], 6)
        df[f"{token}_ma_24h"] = calculate_ma(df["c"], 24)
        df[f"{token}_trend_7d"] = calculate_trend(df["c"], 168)
        
        features.append(df[["timestamp", f"{token}_price", f"{token}_return_1h", 
                           f"{token}_return_24h", f"{token}_volatility_24h",
                           f"{token}_volatility_168h", f"{token}_ma_6h",
                           f"{token}_ma_24h", f"{token}_trend_7d"]])
    
    # Merge all token features
    if features:
        merged = features[0]
        for f in features[1:]:
            merged = pd.merge(merged, f, on="timestamp", how="outer")
        print(f"   ✅ Generated {len(merged.columns)-1} token features")
        return merged
    
    return pd.DataFrame()


def generate_pool_features(pool_df: pd.DataFrame, token_features: pd.DataFrame) -> pd.DataFrame:
    """Generate pool-level features."""
    print("\n📊 Generating pool features...")
    
    all_features = []
    pools = pool_df["pool_address"].unique()
    
    for i, pool_addr in enumerate(pools):
        df = pool_df[pool_df["pool_address"] == pool_addr].copy()
        df = df.sort_values("timestamp")
        pool_name = df["pool_name"].iloc[0]
        dex = df["dex"].iloc[0]
        
        print_progress(i+1, len(pools), pool_name)
        
        # Basic columns
        features = pd.DataFrame()
        features["timestamp"] = df["timestamp"]
        features["pool_address"] = pool_addr
        features["pool_name"] = pool_name
        features["dex"] = dex

        # ─────────────────────────────────────────────────────────────
        # 1. VOLUME FEATURES
        # ─────────────────────────────────────────────────────────────
        features["volume_1h"] = df["v"].values
        features["volume_ma_6h"] = calculate_ma(df["v"], 6).values
        features["volume_ma_24h"] = calculate_ma(df["v"], 24).values
        features["volume_ma_168h"] = calculate_ma(df["v"], 168).values
        features["volume_trend_7d"] = calculate_trend(df["v"], 168).values
        features["volume_volatility_24h"] = calculate_volatility(df["v"], 24).values

        # ─────────────────────────────────────────────────────────────
        # 2. PRICE/APY PROXY FEATURES (using close price as proxy)
        # ─────────────────────────────────────────────────────────────
        features["price_close"] = df["c"].values
        features["price_high"] = df["h"].values
        features["price_low"] = df["l"].values
        features["price_range"] = (df["h"] - df["l"]).values
        features["price_range_pct"] = ((df["h"] - df["l"]) / df["c"] * 100).values

        features["price_ma_6h"] = calculate_ma(df["c"], 6).values
        features["price_ma_24h"] = calculate_ma(df["c"], 24).values
        features["price_ma_168h"] = calculate_ma(df["c"], 168).values
        features["price_trend_7d"] = calculate_trend(df["c"], 168).values
        features["price_volatility_24h"] = calculate_volatility(df["c"], 24).values
        features["price_volatility_168h"] = calculate_volatility(df["c"], 168).values

        # Price momentum
        features["price_return_1h"] = calculate_returns(df["c"], 1).values
        features["price_return_6h"] = calculate_returns(df["c"], 6).values
        features["price_return_24h"] = calculate_returns(df["c"], 24).values
        features["price_return_168h"] = calculate_returns(df["c"], 168).values

        # ─────────────────────────────────────────────────────────────
        # 3. TVL/LIQUIDITY PROXY FEATURES
        # ─────────────────────────────────────────────────────────────
        # Use volume as TVL proxy (vol/price gives liquidity estimate)
        tvl_proxy = df["v"] / (df["c"] + 1e-10)
        features["tvl_proxy"] = tvl_proxy.values
        features["tvl_ma_24h"] = calculate_ma(tvl_proxy, 24).values
        features["tvl_stability_7d"] = (1 / (calculate_volatility(tvl_proxy, 168) + 1)).values
        features["tvl_trend_7d"] = calculate_trend(tvl_proxy, 168).values

        # Volume/TVL ratio (higher = more active)
        features["vol_tvl_ratio"] = (df["v"] / (tvl_proxy + 1e-10)).values
        features["vol_tvl_ma_24h"] = calculate_ma(features["vol_tvl_ratio"], 24).values

        # ─────────────────────────────────────────────────────────────
        # 4. IL (IMPERMANENT LOSS) FEATURES
        # ─────────────────────────────────────────────────────────────
        # IL estimate based on price ratio changes
        price_ratio_change_24h = calculate_returns(df["c"], 24) / 100
        features["il_estimate_24h"] = price_ratio_change_24h.apply(estimate_il).values

        price_ratio_change_168h = calculate_returns(df["c"], 168) / 100
        features["il_estimate_7d"] = price_ratio_change_168h.apply(estimate_il).values

        features["il_change_24h"] = features["il_estimate_24h"].diff()

        # ─────────────────────────────────────────────────────────────
        # 5. TIME FEATURES
        # ─────────────────────────────────────────────────────────────
        features["hour_of_day"] = df["timestamp"].dt.hour.values
        features["day_of_week"] = df["timestamp"].dt.dayofweek.values
        features["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int).values

        # Cyclical encoding for hour
        features["hour_sin"] = np.sin(2 * np.pi * features["hour_of_day"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour_of_day"] / 24)

        # Cyclical encoding for day
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        all_features.append(features)

    # Combine all pool features
    combined = pd.concat(all_features, ignore_index=True)

    # Merge with token features
    if not token_features.empty:
        combined = pd.merge(combined, token_features, on="timestamp", how="left")

    print(f"\n   ✅ Generated {len(combined.columns)} total features")
    return combined


def generate_labels(df: pd.DataFrame, forward_hours: int = 24) -> pd.DataFrame:
    """Generate labels based on future price changes."""
    print("\n🏷️ Generating labels...")

    df = df.sort_values(["pool_address", "timestamp"])

    # Future price (for label)
    df["future_price"] = df.groupby("pool_address")["price_close"].shift(-forward_hours)

    # Label: 1 if price goes up (STAY), 0 if price goes down (EXIT)
    df["label"] = (df["future_price"] > df["price_close"]).astype(int)

    # Price change magnitude (for regression)
    df["future_return"] = (df["future_price"] - df["price_close"]) / df["price_close"] * 100

    # Remove rows without labels
    valid_labels = df["label"].notna().sum()
    print(f"   ✅ Generated {valid_labels} valid labels")

    return df


def save_features(df: pd.DataFrame):
    """Save processed features."""
    print("\n💾 Saving features...")

    # Save as CSV (readable)
    csv_path = OUTPUT_DIR / "pool_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Saved to {csv_path}")

    # Save feature summary
    summary = {
        "total_rows": len(df),
        "total_features": len(df.columns),
        "pools": df["pool_address"].nunique(),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "features": list(df.columns),
        "label_distribution": df["label"].value_counts().to_dict() if "label" in df.columns else {},
    }

    with open(OUTPUT_DIR / "feature_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main():
    """Main feature engineering pipeline."""
    print("\n" + "🔧" * 30)
    print("  FEATURE ENGINEERING PIPELINE")
    print("🔧" * 30)

    # 1. Load data
    token_df, pool_df, protocol_tvl = load_data()

    if pool_df.empty:
        print("❌ No pool data to process!")
        return

    # 2. Generate token features
    token_features = generate_token_features(token_df)

    # 3. Generate pool features
    pool_features = generate_pool_features(pool_df, token_features)

    # 4. Generate labels
    final_df = generate_labels(pool_features)

    # 5. Save
    summary = save_features(final_df)

    # Final summary
    print("\n" + "=" * 60)
    print("  📊 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"   Total rows: {summary['total_rows']:,}")
    print(f"   Total features: {summary['total_features']}")
    print(f"   Pools: {summary['pools']}")
    print(f"   Date range: {summary['date_range']['start'][:10]} → {summary['date_range']['end'][:10]}")

    if summary.get("label_distribution"):
        print(f"\n   Label distribution:")
        for label, count in summary["label_distribution"].items():
            pct = count / summary["total_rows"] * 100
            label_name = "STAY" if label == 1 else "EXIT"
            print(f"      {label_name} ({label}): {count:,} ({pct:.1f}%)")

    print("\n✅ Feature engineering complete!")


if __name__ == "__main__":
    main()

