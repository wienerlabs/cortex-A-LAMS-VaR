"""
Feature Engineering for LP Rebalancer Model

Transforms raw pool data into ML-ready features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PoolSnapshot:
    """Single hourly snapshot of pool state."""
    timestamp: pd.Timestamp
    pool_address: str
    pool_name: str
    dex: str
    
    # Core metrics
    apy: float
    tvl: float
    volume_24h: float
    fee_tier: float
    
    # Token prices
    token_a_price: float
    token_b_price: float
    token_a_symbol: str
    token_b_symbol: str
    
    # Market context
    sol_price: float
    btc_price: float


def calculate_trend(series: pd.Series, window: int) -> float:
    """Calculate linear regression slope as trend indicator."""
    if len(series) < window:
        return 0.0
    
    recent = series.tail(window).values
    x = np.arange(len(recent))
    
    # Linear regression slope
    slope = np.polyfit(x, recent, 1)[0]
    
    # Normalize by mean to get percentage change per hour
    mean_val = np.mean(recent)
    if mean_val == 0:
        return 0.0
    
    return (slope / mean_val) * 100


def calculate_volatility(series: pd.Series, window: int) -> float:
    """Calculate rolling volatility (std dev of returns)."""
    if len(series) < window:
        return 0.0
    
    returns = series.pct_change().dropna()
    return returns.tail(window).std() * 100


def calculate_il(price_a_start: float, price_a_end: float,
                  price_b_start: float, price_b_end: float) -> float:
    """
    Calculate Impermanent Loss percentage.
    
    IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
    where price_ratio = (Pa_end/Pa_start) / (Pb_end/Pb_start)
    """
    if price_a_start == 0 or price_b_start == 0:
        return 0.0
    
    ratio_a = price_a_end / price_a_start
    ratio_b = price_b_end / price_b_start
    
    if ratio_b == 0:
        return 0.0
    
    price_ratio = ratio_a / ratio_b
    
    il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
    return abs(il) * 100  # Return as percentage


class FeatureEngineer:
    """Transforms raw pool data into ML features."""
    
    def __init__(self, config=None):
        from .config import FeatureConfig, MA_WINDOWS, TREND_WINDOW_HOURS
        self.config = config or FeatureConfig()
        self.ma_windows = MA_WINDOWS
        self.trend_window = TREND_WINDOW_HOURS
    
    def compute_features(self, df: pd.DataFrame, pool_address: str) -> pd.DataFrame:
        """Compute all features for a single pool's historical data."""
        
        pool_df = df[df["pool_address"] == pool_address].copy()
        pool_df = pool_df.sort_values("timestamp")
        
        features = pd.DataFrame()
        features["timestamp"] = pool_df["timestamp"]
        features["pool_address"] = pool_address
        
        # APY features
        if self.config.apy_current:
            features["apy_current"] = pool_df["apy"]
        
        for window in [6, 24, 168]:
            col_name = f"apy_ma_{window}h"
            if getattr(self.config, col_name, False):
                features[col_name] = pool_df["apy"].rolling(window, min_periods=1).mean()
        
        if self.config.apy_trend_7d:
            features["apy_trend_7d"] = pool_df["apy"].rolling(168).apply(
                lambda x: calculate_trend(x, len(x)), raw=False
            )
        
        if self.config.apy_volatility_24h:
            features["apy_volatility_24h"] = pool_df["apy"].rolling(24).std()
        
        # Volume features
        if self.config.volume_24h:
            features["volume_24h"] = pool_df["volume_24h"]
        
        if self.config.volume_tvl_ratio:
            features["volume_tvl_ratio"] = pool_df["volume_24h"] / pool_df["tvl"].replace(0, 1)
        
        if self.config.volume_trend_7d:
            features["volume_trend_7d"] = pool_df["volume_24h"].rolling(168).apply(
                lambda x: calculate_trend(x, len(x)), raw=False
            )
        
        # TVL features
        if self.config.tvl_current:
            features["tvl_current"] = pool_df["tvl"]
        
        if self.config.tvl_stability_7d:
            rolling_mean = pool_df["tvl"].rolling(168).mean()
            rolling_std = pool_df["tvl"].rolling(168).std()
            features["tvl_stability_7d"] = 1 - (rolling_std / rolling_mean.replace(0, 1))
        
        if self.config.tvl_trend_7d:
            features["tvl_trend_7d"] = pool_df["tvl"].rolling(168).apply(
                lambda x: calculate_trend(x, len(x)), raw=False
            )
        
        # Token volatility
        if self.config.token_a_volatility_24h:
            features["token_a_volatility_24h"] = calculate_volatility(
                pool_df["token_a_price"], 24
            )
        
        if self.config.token_b_volatility_24h:
            features["token_b_volatility_24h"] = calculate_volatility(
                pool_df["token_b_price"], 24
            )
        
        # Time features
        if self.config.hour_of_day:
            features["hour_of_day"] = pool_df["timestamp"].dt.hour
        
        if self.config.day_of_week:
            features["day_of_week"] = pool_df["timestamp"].dt.dayofweek
        
        return features.fillna(0)

