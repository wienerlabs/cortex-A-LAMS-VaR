#!/usr/bin/env python3
"""
Perpetuals Feature Engineering Pipeline

Creates ~50 features for funding rate arbitrage prediction:
- Funding rate features (current, lagged, rolling stats)
- Price features (returns, volatility, momentum)
- Open interest features (changes, ratios)
- Volume features (VWAP, volume ratios)
- Cross-market features (correlations, spreads)
- Time features (hour, day of week, etc.)

Usage:
    python perps_feature_engineering.py --input ./data --output ./features
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    input_dir: str = "./data"
    output_dir: str = "./features"
    lookback_windows: List[int] = None
    target_horizon: int = 1  # Hours ahead to predict
    min_samples: int = 100

    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [1, 4, 8, 24, 48, 168]  # 1h, 4h, 8h, 1d, 2d, 1w


class PerpsFeatureEngineer:
    """Feature engineering for perpetuals funding rate prediction."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load funding rates and contracts data."""
        funding_path = os.path.join(self.config.input_dir, "funding_rates_latest.csv")
        contracts_path = os.path.join(self.config.input_dir, "contracts_latest.csv")

        funding_df = pd.read_csv(funding_path, parse_dates=["datetime"])
        contracts_df = pd.read_csv(contracts_path)

        print(f"Loaded {len(funding_df)} funding rate records")
        print(f"Loaded {len(contracts_df)} contract records")

        return funding_df, contracts_df

    def create_funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create funding rate features for a single market."""
        features = pd.DataFrame(index=df.index)

        # Current funding rate
        features["funding_rate"] = df["funding_rate_pct"]
        features["funding_rate_raw"] = df["funding_rate_raw"]

        # Lagged funding rates
        for lag in [1, 2, 4, 8, 12, 24]:
            features[f"funding_lag_{lag}h"] = df["funding_rate_pct"].shift(lag)

        # Rolling statistics
        for window in self.config.lookback_windows:
            fr = df["funding_rate_pct"]
            features[f"funding_mean_{window}h"] = fr.rolling(window).mean()
            features[f"funding_std_{window}h"] = fr.rolling(window).std()
            features[f"funding_min_{window}h"] = fr.rolling(window).min()
            features[f"funding_max_{window}h"] = fr.rolling(window).max()
            features[f"funding_skew_{window}h"] = fr.rolling(window).skew()

        # Funding rate momentum
        features["funding_momentum_4h"] = df["funding_rate_pct"] - df["funding_rate_pct"].shift(4)
        features["funding_momentum_24h"] = df["funding_rate_pct"] - df["funding_rate_pct"].shift(24)

        # Cumulative funding (long vs short)
        if "cumulative_funding_rate_long" in df.columns:
            features["cum_funding_long"] = df["cumulative_funding_rate_long"]
            features["cum_funding_short"] = df["cumulative_funding_rate_short"]
            features["cum_funding_diff"] = (
                df["cumulative_funding_rate_long"] - df["cumulative_funding_rate_short"]
            )

        # Funding rate sign changes
        features["funding_sign"] = np.sign(df["funding_rate_pct"])
        features["funding_sign_change"] = (features["funding_sign"] != features["funding_sign"].shift(1)).astype(int)
        features["funding_sign_changes_24h"] = features["funding_sign_change"].rolling(24).sum()

        # Extreme funding detection
        funding_std = df["funding_rate_pct"].rolling(168).std()  # 1 week
        funding_mean = df["funding_rate_pct"].rolling(168).mean()
        features["funding_zscore"] = (df["funding_rate_pct"] - funding_mean) / funding_std

        return features

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        features = pd.DataFrame(index=df.index)

        if "oracle_twap" not in df.columns:
            return features

        price = df["oracle_twap"]

        # Price returns
        for period in [1, 4, 8, 24, 48]:
            features[f"return_{period}h"] = price.pct_change(period)

        # Volatility
        returns = price.pct_change()
        for window in [24, 48, 168]:
            features[f"volatility_{window}h"] = returns.rolling(window).std() * np.sqrt(window)

        # Price momentum indicators
        for window in [24, 48]:
            features[f"price_momentum_{window}h"] = price / price.shift(window) - 1

        # Mark vs Oracle spread (basis)
        if "mark_twap" in df.columns and df["mark_twap"].sum() > 0:
            features["basis"] = (df["mark_twap"] - df["oracle_twap"]) / df["oracle_twap"]
        else:
            features["basis"] = 0

        return features

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=df.index)

        dt = df["datetime"]
        features["hour"] = dt.dt.hour
        features["day_of_week"] = dt.dt.dayofweek
        features["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        # Cyclical encoding
        features["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        features["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

        return features

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create prediction target: future funding rate."""
        horizon = self.config.target_horizon
        target = df["funding_rate_pct"].shift(-horizon)
        return target.rename("target_funding_rate")

    def engineer_features_for_market(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Create all features for a single market."""
        print(f"  Engineering features for {market}...")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Create feature groups
        funding_features = self.create_funding_features(df)
        price_features = self.create_price_features(df)
        time_features = self.create_time_features(df)
        target = self.create_target(df)

        # Combine all features
        features = pd.concat([
            df[["market", "timestamp", "datetime"]],
            funding_features,
            price_features,
            time_features,
            target
        ], axis=1)

        # Drop rows with NaN in target (future data not available)
        features = features.dropna(subset=["target_funding_rate"])

        print(f"    Created {len(features.columns) - 4} features, {len(features)} samples")
        return features

    def engineer_all_features(self) -> pd.DataFrame:
        """Engineer features for all markets."""
        print("\n🔧 Feature Engineering Pipeline")
        print("=" * 60)

        funding_df, contracts_df = self.load_data()

        all_features = []
        markets = funding_df["market"].unique()

        for market in markets:
            market_df = funding_df[funding_df["market"] == market].copy()
            if len(market_df) < self.config.min_samples:
                print(f"  Skipping {market}: only {len(market_df)} samples")
                continue

            features = self.engineer_features_for_market(market_df, market)
            all_features.append(features)

        if not all_features:
            print("❌ No features created!")
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=True)
        print(f"\n✅ Total: {len(combined)} samples, {len(combined.columns)} columns")

        return combined

    def save_features(self, features: pd.DataFrame) -> str:
        """Save engineered features to CSV."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        filepath = os.path.join(self.config.output_dir, "perps_features.csv")
        features.to_csv(filepath, index=False)
        print(f"\n💾 Saved features to {filepath}")

        # Also save feature list
        feature_cols = [c for c in features.columns if c not in ["market", "timestamp", "datetime", "target_funding_rate"]]
        feature_list_path = os.path.join(self.config.output_dir, "feature_list.txt")
        with open(feature_list_path, "w") as f:
            f.write("\n".join(feature_cols))
        print(f"   Saved feature list to {feature_list_path}")

        return filepath

    def run(self) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        features = self.engineer_all_features()
        if not features.empty:
            self.save_features(features)
            self._print_summary(features)
        return features

    def _print_summary(self, features: pd.DataFrame) -> None:
        """Print feature summary statistics."""
        print("\n" + "=" * 60)
        print("  FEATURE SUMMARY")
        print("=" * 60)

        feature_cols = [c for c in features.columns if c not in ["market", "timestamp", "datetime", "target_funding_rate"]]
        print(f"\nTotal features: {len(feature_cols)}")
        print(f"Total samples: {len(features)}")
        print(f"Markets: {features['market'].nunique()}")

        # Target statistics
        target = features["target_funding_rate"]
        print(f"\nTarget (funding rate) statistics:")
        print(f"  Mean: {target.mean():.6f}%")
        print(f"  Std:  {target.std():.6f}%")
        print(f"  Min:  {target.min():.6f}%")
        print(f"  Max:  {target.max():.6f}%")

        # Missing values
        missing = features[feature_cols].isnull().sum()
        if missing.sum() > 0:
            print(f"\nFeatures with missing values: {(missing > 0).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Engineer features for perps ML model")
    parser.add_argument("--input", type=str, default="./data", help="Input data directory")
    parser.add_argument("--output", type=str, default="./features", help="Output features directory")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    args = parser.parse_args()

    config = FeatureConfig(
        input_dir=args.input,
        output_dir=args.output,
        target_horizon=args.horizon,
    )

    engineer = PerpsFeatureEngineer(config)
    engineer.run()


if __name__ == "__main__":
    main()

