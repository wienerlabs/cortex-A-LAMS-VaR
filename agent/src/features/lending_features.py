"""
Lending strategy feature engineering for Solana.

Specialized features for lending protocol selection:
- APY dynamics and sustainability
- Protocol health metrics (TVL, utilization, age)
- Health factor and risk features
- Asset quality features
- Cross-protocol comparison features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import structlog

logger = structlog.get_logger()


class LendingFeatureEngineer:
    """
    Feature engineering for lending strategy on Solana.
    
    Creates features specifically for:
    - Protocol selection (MarginFi, Kamino, Solend)
    - APY-based switching decisions
    - Risk management and health factor monitoring
    - Optimal lending opportunities
    """
    
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        
        # APY parameters
        self.apy_windows = self.config.get('apy_rolling_windows', [7, 14, 30])
        self.apy_volatility_window = self.config.get('apy_volatility_window', 30)
        
        # Utilization parameters
        self.util_windows = self.config.get('utilization_rolling_windows', [1, 7, 14])
        self.util_spike_threshold = self.config.get('utilization_spike_threshold', 0.10)
        
        # TVL parameters
        self.tvl_windows = self.config.get('tvl_rolling_windows', [7, 14, 30])
        self.tvl_change_threshold = self.config.get('tvl_change_threshold', 0.15)
        
        # Protocol thresholds
        self.min_protocol_tvl = self.config.get('min_protocol_tvl_usd', 50_000_000)
        self.max_utilization = self.config.get('max_utilization_rate', 0.90)
        self.min_apy = self.config.get('min_acceptable_apy', 0.02)
        self.max_apy = self.config.get('max_acceptable_apy', 0.50)
        
        self.logger = logger.bind(component="lending_features")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply lending-specific feature engineering.
        
        Expects DataFrame with columns:
        - protocol: Protocol name (marginfi, kamino, solend)
        - supply_apy: Supply APY (decimal, e.g., 0.08 for 8%)
        - borrow_apy: Borrow APY (if leveraged)
        - utilization_rate: Protocol utilization (0-1)
        - protocol_tvl_usd: Protocol TVL in USD
        - asset: Asset being lent (SOL, USDC, etc.)
        - timestamp: Timestamp of data point
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Sort by timestamp for time-series features
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # APY features
        df = self._add_apy_features(df)
        
        # Protocol health features
        df = self._add_protocol_health_features(df)
        
        # Utilization features
        df = self._add_utilization_features(df)
        
        # TVL features
        df = self._add_tvl_features(df)
        
        # Risk features
        df = self._add_risk_features(df)
        
        # Asset quality features
        df = self._add_asset_features(df)
        
        # Cross-protocol comparison features
        if 'protocol' in df.columns:
            df = self._add_cross_protocol_features(df)
        
        # Time-based features
        if 'timestamp' in df.columns:
            df = self._add_time_features(df)
        
        return df
    
    def _add_apy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add APY-related features."""
        # Net APY (supply - borrow if leveraged)
        if 'supply_apy' in df.columns:
            df['net_apy'] = df['supply_apy']
            
            if 'borrow_apy' in df.columns and 'leverage' in df.columns:
                # Net APY with leverage: supply_apy - (borrow_apy * (leverage - 1))
                df['net_apy'] = df['supply_apy'] - (
                    df['borrow_apy'] * (df['leverage'] - 1)
                )
            
            # APY rolling statistics
            for window in self.apy_windows:
                df[f'apy_sma_{window}d'] = df['supply_apy'].rolling(window=window*24).mean()
                df[f'apy_std_{window}d'] = df['supply_apy'].rolling(window=window*24).std()
            
            # APY momentum
            df['apy_change_1d'] = df['supply_apy'].diff(24)  # 24 hours
            df['apy_change_7d'] = df['supply_apy'].diff(7*24)
            df['apy_momentum'] = df['supply_apy'] - df.get('apy_sma_7d', df['supply_apy'])
            
            # APY volatility
            df['apy_volatility_30d'] = df['supply_apy'].rolling(
                window=self.apy_volatility_window*24
            ).std()
            
            # APY z-score (normalized)
            apy_mean = df['supply_apy'].rolling(window=30*24).mean()
            apy_std = df['supply_apy'].rolling(window=30*24).std()
            df['apy_zscore'] = (df['supply_apy'] - apy_mean) / (apy_std + 1e-8)
            
            # APY sustainability score (compare to historical average)
            df['apy_vs_avg_30d'] = df['supply_apy'] / (apy_mean + 1e-8)
            
            # APY spike detection
            df['apy_spike'] = (
                (df['apy_change_1d'] / (df['supply_apy'] + 1e-8)) > 0.50
            ).astype(int)
            
            # APY range check
            df['apy_in_range'] = (
                (df['supply_apy'] >= self.min_apy) &
                (df['supply_apy'] <= self.max_apy)
            ).astype(int)

        return df

    def _add_protocol_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add protocol health features."""
        if 'protocol_tvl_usd' in df.columns:
            # TVL adequacy
            df['tvl_adequate'] = (
                df['protocol_tvl_usd'] >= self.min_protocol_tvl
            ).astype(int)

            # TVL tier (for scoring)
            df['tvl_tier'] = pd.cut(
                df['protocol_tvl_usd'],
                bins=[0, 50_000_000, 100_000_000, float('inf')],
                labels=[0, 1, 2]  # 0: <$50M, 1: $50-100M, 2: >$100M
            ).astype(int)

        # Protocol age (if available)
        if 'protocol_age_days' in df.columns:
            df['protocol_mature'] = (df['protocol_age_days'] >= 180).astype(int)  # 6 months
            df['protocol_age_tier'] = pd.cut(
                df['protocol_age_days'],
                bins=[0, 180, 365, float('inf')],
                labels=[0, 1, 2]  # 0: <6mo, 1: 6-12mo, 2: >12mo
            ).astype(int)

        # Audit status (if available)
        if 'audited' in df.columns:
            df['audit_score'] = df['audited'].astype(int)

        return df

    def _add_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add utilization-related features."""
        if 'utilization_rate' not in df.columns:
            return df

        # Utilization rolling statistics
        for window in self.util_windows:
            df[f'util_sma_{window}d'] = df['utilization_rate'].rolling(
                window=window*24
            ).mean()
            df[f'util_std_{window}d'] = df['utilization_rate'].rolling(
                window=window*24
            ).std()

        # Utilization momentum
        df['util_change_1d'] = df['utilization_rate'].diff(24)
        df['util_change_7d'] = df['utilization_rate'].diff(7*24)

        # Utilization spike detection
        df['util_spike'] = (
            df['util_change_1d'] > self.util_spike_threshold
        ).astype(int)

        # Utilization safety zones
        df['util_safe'] = (df['utilization_rate'] < 0.70).astype(int)
        df['util_warning'] = (
            (df['utilization_rate'] >= 0.70) &
            (df['utilization_rate'] < 0.85)
        ).astype(int)
        df['util_danger'] = (
            (df['utilization_rate'] >= 0.85) &
            (df['utilization_rate'] < 0.90)
        ).astype(int)
        df['util_critical'] = (df['utilization_rate'] >= 0.90).astype(int)

        # Available liquidity (1 - utilization)
        df['available_liquidity'] = 1 - df['utilization_rate']

        return df

    def _add_tvl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add TVL-related features."""
        if 'protocol_tvl_usd' not in df.columns:
            return df

        # TVL rolling statistics
        for window in self.tvl_windows:
            df[f'tvl_sma_{window}d'] = df['protocol_tvl_usd'].rolling(
                window=window*24
            ).mean()
            df[f'tvl_std_{window}d'] = df['protocol_tvl_usd'].rolling(
                window=window*24
            ).std()

        # TVL changes
        df['tvl_change_1d'] = df['protocol_tvl_usd'].pct_change(24)
        df['tvl_change_7d'] = df['protocol_tvl_usd'].pct_change(7*24)
        df['tvl_change_30d'] = df['protocol_tvl_usd'].pct_change(30*24)

        # TVL drop detection
        df['tvl_drop_warning'] = (
            df['tvl_change_1d'] < -self.tvl_change_threshold
        ).astype(int)
        df['tvl_drop_critical'] = (
            df['tvl_change_1d'] < -0.30  # 30% drop
        ).astype(int)

        # TVL growth
        df['tvl_growing'] = (df['tvl_change_7d'] > 0).astype(int)

        return df

    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk and health factor features."""
        # Health factor features (if available)
        if 'health_factor' in df.columns:
            df['hf_safe'] = (df['health_factor'] >= 2.5).astype(int)
            df['hf_adequate'] = (
                (df['health_factor'] >= 2.0) &
                (df['health_factor'] < 2.5)
            ).astype(int)
            df['hf_warning'] = (
                (df['health_factor'] >= 1.8) &
                (df['health_factor'] < 2.0)
            ).astype(int)
            df['hf_danger'] = (
                (df['health_factor'] >= 1.5) &
                (df['health_factor'] < 1.8)
            ).astype(int)
            df['hf_critical'] = (df['health_factor'] < 1.5).astype(int)

            # Health factor volatility
            df['hf_volatility'] = df['health_factor'].rolling(window=24).std()

        # Leverage features (if available)
        if 'leverage' in df.columns:
            df['is_leveraged'] = (df['leverage'] > 1.0).astype(int)
            df['leverage_safe'] = (df['leverage'] <= 1.5).astype(int)
            df['leverage_risk'] = df['leverage'] - 1.0  # Risk above 1x

        # Liquidation distance (if available)
        if 'liquidation_price' in df.columns and 'current_price' in df.columns:
            df['liquidation_distance'] = (
                (df['current_price'] - df['liquidation_price']) /
                df['current_price']
            )

        return df

    def _add_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add asset quality features."""
        if 'asset' not in df.columns:
            return df

        # Asset tier mapping
        tier_1_assets = ['SOL', 'USDC', 'USDT']
        tier_2_assets = ['JitoSOL', 'mSOL', 'stSOL', 'BONK', 'JUP']

        df['asset_tier'] = df['asset'].apply(
            lambda x: 1 if x in tier_1_assets else
                     2 if x in tier_2_assets else 3
        )

        # Asset type
        df['is_stablecoin'] = df['asset'].isin(['USDC', 'USDT']).astype(int)
        df['is_sol'] = (df['asset'] == 'SOL').astype(int)
        df['is_lst'] = df['asset'].isin(['JitoSOL', 'mSOL', 'stSOL']).astype(int)

        # Asset quality score (1.0 for tier 1, 0.7 for tier 2, 0.4 for tier 3)
        df['asset_quality_score'] = df['asset_tier'].map({1: 1.0, 2: 0.7, 3: 0.4})

        return df

    def _add_cross_protocol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-protocol comparison features."""
        if 'protocol' not in df.columns or 'timestamp' not in df.columns:
            return df

        # Group by timestamp to compare protocols at same time
        for timestamp, group in df.groupby('timestamp'):
            if len(group) < 2:
                continue

            # APY comparison
            if 'supply_apy' in df.columns:
                max_apy = group['supply_apy'].max()
                df.loc[group.index, 'apy_vs_best'] = group['supply_apy'] / (max_apy + 1e-8)
                df.loc[group.index, 'is_best_apy'] = (
                    group['supply_apy'] == max_apy
                ).astype(int)

            # TVL comparison
            if 'protocol_tvl_usd' in df.columns:
                max_tvl = group['protocol_tvl_usd'].max()
                df.loc[group.index, 'tvl_vs_largest'] = (
                    group['protocol_tvl_usd'] / (max_tvl + 1e-8)
                )

            # Utilization comparison
            if 'utilization_rate' in df.columns:
                min_util = group['utilization_rate'].min()
                df.loc[group.index, 'util_vs_lowest'] = (
                    group['utilization_rate'] / (min_util + 1e-8)
                )

        # Protocol preference score (based on preference order)
        protocol_preference = {'marginfi': 3, 'kamino': 2, 'solend': 1}
        df['protocol_preference_score'] = df['protocol'].map(
            lambda x: protocol_preference.get(x.lower(), 0)
        )

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            return df

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        # Time-based patterns
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17)
        ).astype(int)

        # Position duration (if available)
        if 'position_entry_time' in df.columns:
            df['position_duration_hours'] = (
                (df['timestamp'] - pd.to_datetime(df['position_entry_time'])).dt.total_seconds() / 3600
            )
            df['position_duration_days'] = df['position_duration_hours'] / 24

        return df

    def create_confidence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create confidence score for lending decisions.

        Based on strategy document:
        - Protocol safety: 30%
        - APY sustainability: 25%
        - Utilization health: 20%
        - Liquidity depth: 15%
        - Asset quality: 10%
        """
        if df.empty:
            return df

        df = df.copy()

        # Protocol safety score (30%)
        protocol_score = 0.0
        if 'protocol' in df.columns:
            protocol_scores = {'marginfi': 1.0, 'kamino': 0.9, 'solend': 0.8}
            protocol_score = df['protocol'].map(
                lambda x: protocol_scores.get(x.lower(), 0.5)
            )

        # APY sustainability score (25%)
        apy_score = 0.0
        if 'apy_vs_avg_30d' in df.columns:
            # Near average = 1.0, 2x average = 0.5
            apy_score = np.clip(2.0 - df['apy_vs_avg_30d'], 0, 1)

        # Utilization health score (20%)
        util_score = 0.0
        if 'utilization_rate' in df.columns:
            util_score = pd.cut(
                df['utilization_rate'],
                bins=[0, 0.70, 0.80, 0.90, 1.0],
                labels=[1.0, 0.8, 0.5, 0.2]
            ).astype(float)

        # Liquidity depth score (15%)
        liquidity_score = 0.0
        if 'protocol_tvl_usd' in df.columns:
            liquidity_score = pd.cut(
                df['protocol_tvl_usd'],
                bins=[0, 50_000_000, 100_000_000, float('inf')],
                labels=[0.4, 0.7, 1.0]
            ).astype(float)

        # Asset quality score (10%)
        asset_score = 0.0
        if 'asset_quality_score' in df.columns:
            asset_score = df['asset_quality_score']

        # Weighted confidence score
        df['confidence_score'] = (
            protocol_score * 0.30 +
            apy_score * 0.25 +
            util_score * 0.20 +
            liquidity_score * 0.15 +
            asset_score * 0.10
        )

        # Confidence level
        df['confidence_level'] = pd.cut(
            df['confidence_score'],
            bins=[0, 0.4, 0.6, 0.75, 0.85, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )

        return df

