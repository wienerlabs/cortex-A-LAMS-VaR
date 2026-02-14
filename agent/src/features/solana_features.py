"""
Solana-specific feature engineering for DeFi ML models.

Key differences from Ethereum:
- Slot-based timing (400ms vs 12s blocks)
- Priority fees instead of gas price
- Different DEX mechanics (Raydium AMM vs Orca Whirlpool)
- Lower costs enable more frequent trading
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import structlog

from ..config import FEATURE_PARAMS, SOLANA_CHAIN_PARAMS

logger = structlog.get_logger()


class SolanaFeatureEngineer:
    """
    Feature engineering for Solana DeFi data.
    
    Creates features optimized for:
    - Cross-DEX arbitrage (Raydium vs Orca)
    - Low-latency trading (400ms slots)
    - Cost-aware decisions (priority fees)
    """
    
    def __init__(self, config: dict | None = None):
        self.config = config or FEATURE_PARAMS
        self.chain_params = SOLANA_CHAIN_PARAMS
        
        # Feature parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        self.logger = logger.bind(component="solana_features")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering to raw OHLCV data.
        
        Args:
            df: DataFrame with columns: datetime, open, high, low, close, volume
        
        Returns:
            DataFrame with all engineered features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure datetime index
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        
        # Price features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Solana-specific features
        df = self._add_solana_features(df)
        
        # Time features
        df = self._add_time_features(df)
        
        # Drop NaN rows from rolling calculations
        df = df.dropna()
        
        return df.reset_index()
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_15'] = df['close'].pct_change(15)
        
        # Log returns (better for ML)
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        
        # Price momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_15'] = df['close'] - df['close'].shift(15)
        
        # Price range (volatility proxy)
        df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_15'] = df['close'].rolling(window=15).mean()
        df['sma_60'] = df['close'].rolling(window=60).mean()
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        
        # MA crossovers
        df['sma_cross'] = (df['sma_5'] > df['sma_15']).astype(int)
        df['ema_cross'] = (df['ema_5'] > df['ema_15']).astype(int)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_15'] = df['volume'].rolling(window=15).mean()
        df['volume_sma_60'] = df['volume'].rolling(window=60).mean()
        
        # Volume ratio (current vs average)
        df['volume_ratio_5'] = df['volume'] / (df['volume_sma_5'] + 1e-8)
        df['volume_ratio_15'] = df['volume'] / (df['volume_sma_15'] + 1e-8)
        
        # Volume trend
        df['volume_trend'] = df['volume'].pct_change(5)
        
        # Price-volume correlation (rolling)
        df['pv_corr'] = df['close'].rolling(window=15).corr(df['volume'])

        return df

    def _add_solana_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Solana-specific features.

        These features capture Solana's unique characteristics:
        - Fast block times (400ms slots)
        - Low transaction costs
        - Priority fee dynamics
        """
        # If we have spread data (from cross-DEX collection)
        if 'spread_abs' in df.columns:
            # Spread momentum
            df['spread_momentum'] = df['spread_abs'].diff(3)
            df['spread_sma_5'] = df['spread_abs'].rolling(window=5).mean()
            df['spread_std_5'] = df['spread_abs'].rolling(window=5).std()

            # Spread z-score (how unusual is current spread)
            df['spread_zscore'] = (
                (df['spread_abs'] - df['spread_sma_5']) /
                (df['spread_std_5'] + 1e-8)
            )

        # If we have priority fee data
        if 'priority_fee_lamports' in df.columns:
            # Fee relative to threshold
            max_fee = self.chain_params.get('priority_fee_lamports', 50000)
            df['fee_ratio'] = df['priority_fee_lamports'] / max_fee
            df['fee_acceptable'] = (df['priority_fee_lamports'] <= max_fee).astype(int)

        # If we have net profit data
        if 'net_profit_pct' in df.columns:
            # Profit momentum
            df['profit_momentum'] = df['net_profit_pct'].diff(3)
            df['profit_sma_5'] = df['net_profit_pct'].rolling(window=5).mean()

            # Profit threshold features
            min_profit = self.chain_params.get('min_profit_threshold', 0.001)
            df['above_min_profit'] = (df['net_profit_pct'] > min_profit).astype(int)

        # Volatility regime (Solana-specific thresholds)
        if 'range_pct' in df.columns:
            df['high_volatility'] = (df['range_pct'] > 0.5).astype(int)
            df['low_volatility'] = (df['range_pct'] < 0.1).astype(int)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Important for Solana because:
        - Different activity patterns by hour
        - Weekend vs weekday differences
        - Asian/US/EU session patterns
        """
        # Get datetime from index
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        else:
            return df  # Can't add time features without datetime

        # Hour of day (cyclical encoding)
        df['hour'] = dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (cyclical encoding)
        df['day_of_week'] = dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Trading sessions (UTC-based)
        df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_eu_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        # Weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names for model training."""
        return [
            # Price features
            'return_1', 'return_5', 'return_15',
            'log_return_1', 'log_return_5',
            'momentum_5', 'momentum_15',
            'range_pct', 'price_position',
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position',
            'sma_cross', 'ema_cross',
            # Volume features
            'volume_ratio_5', 'volume_ratio_15',
            'volume_trend', 'pv_corr',
            # Solana-specific
            'spread_momentum', 'spread_zscore',
            'above_min_profit', 'high_volatility',
            # Time features
            'hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos',
            'is_asia_session', 'is_eu_session', 'is_us_session',
            'is_weekend'
        ]

