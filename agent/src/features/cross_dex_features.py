"""
Cross-DEX arbitrage feature engineering for Solana.

Specialized features for Raydium vs Orca arbitrage:
- Spread dynamics between DEXs
- Liquidity imbalance features
- Cost-adjusted profit features
- Execution timing features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import structlog

from ..config import SOLANA_CHAIN_PARAMS, LABELING_PARAMS

logger = structlog.get_logger()


class CrossDexFeatureEngineer:
    """
    Feature engineering for cross-DEX arbitrage on Solana.
    
    Creates features specifically for:
    - Raydium vs Orca price differences
    - Arbitrage opportunity detection
    - Optimal execution timing
    """
    
    def __init__(self, config: dict | None = None):
        self.chain_params = SOLANA_CHAIN_PARAMS
        self.label_params = LABELING_PARAMS
        
        # Cost parameters
        self.raydium_fee = self.chain_params.get('raydium_fee_pct', 0.0025)
        self.orca_fee = self.chain_params.get('orca_fee_pct', 0.003)
        self.tx_fee_sol = self.chain_params.get('base_tx_fee_lamports', 5000) / 1e9
        self.priority_fee_sol = self.chain_params.get('priority_fee_lamports', 50000) / 1e9
        
        self.logger = logger.bind(component="cross_dex_features")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-DEX specific feature engineering.
        
        Expects DataFrame with columns:
        - raydium_price, orca_price (or spread_abs)
        - sol_price
        - volume
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Spread features
        df = self._add_spread_features(df)
        
        # Cost features
        df = self._add_cost_features(df)
        
        # Arbitrage opportunity features
        df = self._add_arbitrage_features(df)
        
        # Execution features
        df = self._add_execution_features(df)
        
        return df
    
    def _add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spread-related features."""
        # Calculate spread if not present
        if 'spread_abs' not in df.columns:
            if 'raydium_price' in df.columns and 'orca_price' in df.columns:
                df['spread_abs'] = abs(
                    (df['raydium_price'] - df['orca_price']) / 
                    df['sol_price'] * 100
                )
            else:
                # Default to 0 if no price data
                df['spread_abs'] = 0
        
        # Spread statistics
        df['spread_sma_5'] = df['spread_abs'].rolling(window=5).mean()
        df['spread_sma_15'] = df['spread_abs'].rolling(window=15).mean()
        df['spread_std_5'] = df['spread_abs'].rolling(window=5).std()
        df['spread_std_15'] = df['spread_abs'].rolling(window=15).std()
        
        # Spread momentum
        df['spread_change_1'] = df['spread_abs'].diff(1)
        df['spread_change_5'] = df['spread_abs'].diff(5)
        df['spread_momentum'] = df['spread_abs'] - df['spread_sma_5']
        
        # Spread z-score (normalized)
        df['spread_zscore'] = (
            (df['spread_abs'] - df['spread_sma_15']) / 
            (df['spread_std_15'] + 1e-8)
        )
        
        # Spread percentile (rolling)
        df['spread_percentile'] = df['spread_abs'].rolling(window=60).apply(
            lambda x: (x.iloc[-1] > x).mean() * 100, raw=False
        )
        
        # Spread regime
        df['spread_high'] = (df['spread_abs'] > df['spread_sma_15'] * 1.5).astype(int)
        df['spread_low'] = (df['spread_abs'] < df['spread_sma_15'] * 0.5).astype(int)
        
        return df
    
    def _add_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cost-related features."""
        # Total DEX fees (both sides)
        df['total_dex_fee_pct'] = (self.raydium_fee + self.orca_fee) * 100
        
        # Transaction cost in percentage (based on trade size)
        # Assume $10k trade size for percentage calculation
        trade_size_usd = 10000
        sol_price = df['sol_price'].mean() if 'sol_price' in df.columns else 200
        
        tx_cost_sol = (self.tx_fee_sol + self.priority_fee_sol) * 2  # Two txs
        tx_cost_usd = tx_cost_sol * sol_price
        df['tx_cost_pct'] = (tx_cost_usd / trade_size_usd) * 100
        
        # Total cost
        df['total_cost_pct'] = df['total_dex_fee_pct'] + df['tx_cost_pct']
        
        # If we have slippage data
        if 'slippage_pct' in df.columns:
            df['total_cost_pct'] += df['slippage_pct']
        
        # Cost-adjusted spread
        df['net_spread'] = df['spread_abs'] - df['total_cost_pct']
        
        return df
    
    def _add_arbitrage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add arbitrage opportunity features."""
        # Minimum profit threshold
        min_profit = self.label_params.get('min_profit_pct', 0.001) * 100
        
        # Is profitable after costs?
        df['is_profitable'] = (df['net_spread'] > min_profit).astype(int)
        
        # Profit margin (how much above threshold)
        df['profit_margin'] = df['net_spread'] - min_profit
        
        # Consecutive profitable periods
        df['profitable_streak'] = (
            df['is_profitable']
            .groupby((df['is_profitable'] != df['is_profitable'].shift()).cumsum())
            .cumsum()
        )
        
        # Profit potential score (0-1)
        max_spread = df['spread_abs'].quantile(0.99)
        df['profit_score'] = np.clip(df['net_spread'] / max_spread, 0, 1)

        return df

    def _add_execution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add execution timing features.

        Solana's fast block times (400ms) mean timing is critical.
        """
        # Spread velocity (how fast is spread changing)
        df['spread_velocity'] = df['spread_abs'].diff(1) / 1  # Per interval
        df['spread_acceleration'] = df['spread_velocity'].diff(1)

        # Spread direction
        df['spread_increasing'] = (df['spread_velocity'] > 0).astype(int)
        df['spread_decreasing'] = (df['spread_velocity'] < 0).astype(int)

        # Time since last profitable opportunity
        df['time_since_profitable'] = (
            (~df['is_profitable'].astype(bool))
            .groupby((df['is_profitable'] != df['is_profitable'].shift()).cumsum())
            .cumsum()
        )

        # Urgency score (higher = act now)
        # Based on: high spread + decreasing trend = urgent
        df['urgency_score'] = np.where(
            (df['spread_high'] == 1) & (df['spread_decreasing'] == 1),
            1.0,
            np.where(
                df['spread_high'] == 1,
                0.7,
                np.where(
                    df['is_profitable'] == 1,
                    0.5,
                    0.0
                )
            )
        )

        return df

    def create_labels(
        self,
        df: pd.DataFrame,
        lookahead: int = 1,
        min_profit_pct: float = 0.01
    ) -> pd.DataFrame:
        """
        Create labels for supervised learning.

        Args:
            df: DataFrame with features
            lookahead: Number of periods to look ahead
            min_profit_pct: Minimum profit percentage for positive label

        Returns:
            DataFrame with 'label' column added
        """
        df = df.copy()

        # Binary label: Will the next N periods be profitable?
        if 'net_spread' in df.columns:
            # Look at future net spread
            future_profit = df['net_spread'].shift(-lookahead)
            df['label'] = (future_profit > min_profit_pct).astype(int)
        elif 'profitable' in df.columns:
            # Use existing profitable column
            df['label'] = df['profitable'].shift(-lookahead).fillna(0).astype(int)
        else:
            # Default to spread-based label
            df['label'] = (df['spread_abs'] > min_profit_pct * 100).astype(int)

        # Drop rows where we can't calculate future label
        df = df.iloc[:-lookahead] if lookahead > 0 else df

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of cross-DEX specific feature names."""
        return [
            # Spread features
            'spread_abs', 'spread_sma_5', 'spread_sma_15',
            'spread_std_5', 'spread_std_15',
            'spread_change_1', 'spread_change_5', 'spread_momentum',
            'spread_zscore', 'spread_percentile',
            'spread_high', 'spread_low',
            # Cost features
            'total_dex_fee_pct', 'tx_cost_pct', 'total_cost_pct',
            'net_spread',
            # Arbitrage features
            'is_profitable', 'profit_margin', 'profitable_streak',
            'profit_score',
            # Execution features
            'spread_velocity', 'spread_acceleration',
            'spread_increasing', 'spread_decreasing',
            'time_since_profitable', 'urgency_score'
        ]

    def get_all_features(
        self,
        df: pd.DataFrame,
        include_base_features: bool = True
    ) -> pd.DataFrame:
        """
        Get all features including base Solana features.

        Args:
            df: Raw DataFrame
            include_base_features: Whether to include SolanaFeatureEngineer features

        Returns:
            DataFrame with all features
        """
        from .solana_features import SolanaFeatureEngineer

        if include_base_features:
            base_engineer = SolanaFeatureEngineer()
            df = base_engineer.engineer_features(df)

        df = self.engineer_features(df)

        return df
