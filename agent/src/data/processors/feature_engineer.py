"""
Feature Engineering for DeFi ML models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from typing import Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .technical_indicators import TechnicalIndicators

try:
    from ...config import FEATURE_PARAMS, LABELING_PARAMS
except ImportError:
    FEATURE_PARAMS = {}
    LABELING_PARAMS = {"target_horizon": 24, "profit_threshold": 0.003}

logger = structlog.get_logger()


class FeatureEngineer:
    """
    Feature engineering pipeline for DeFi data.
    
    Creates features for:
    - Arbitrage detection (price spreads, gas, liquidity)
    - Lending optimization (rate differentials, utilization)
    - LP provision (TVL, APY, volume trends)
    """
    
    def __init__(
        self,
        feature_params: dict[str, Any] | None = None,
        labeling_params: dict[str, Any] | None = None
    ):
        self.feature_params = feature_params or FEATURE_PARAMS
        self.labeling_params = labeling_params or LABELING_PARAMS
        self.technical = TechnicalIndicators(self.feature_params)
        self.scaler = StandardScaler()
        self.logger = logger.bind(component="feature_engineer")
    
    def process_arbitrage_features(
        self,
        dex_data: pd.DataFrame,
        gas_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features for arbitrage strategy.
        
        Features:
        - Price spread between DEXes
        - Spread volatility
        - Gas price (current and trend)
        - Liquidity on both DEXes
        - Historical spread patterns
        """
        df = dex_data.copy()
        
        # Price spread features
        if "price_uniswap" in df.columns and "price_sushiswap" in df.columns:
            df["spread"] = (df["price_uniswap"] - df["price_sushiswap"]) / df["price_sushiswap"]
            df["spread_abs"] = np.abs(df["spread"])
            df["spread_direction"] = np.sign(df["spread"])
            
            # Spread rolling statistics
            df["spread_sma_5"] = df["spread_abs"].rolling(5).mean()
            df["spread_std_5"] = df["spread_abs"].rolling(5).std()
            df["spread_max_5"] = df["spread_abs"].rolling(5).max()
        
        # Gas features
        if not gas_data.empty:
            df = df.merge(
                gas_data[["timestamp", "gas_price_gwei"]],
                on="timestamp",
                how="left"
            )
            df["gas_price_gwei"] = df["gas_price_gwei"].ffill()
            df["gas_sma_5"] = df["gas_price_gwei"].rolling(5).mean()
            df["gas_trend"] = df["gas_price_gwei"].pct_change(5)
        
        # Liquidity features
        if "liquidity_usd" in df.columns:
            df["liquidity_log"] = np.log1p(df["liquidity_usd"])
            df["liquidity_change"] = df["liquidity_usd"].pct_change()
        
        # Technical indicators on price
        if "close" in df.columns:
            df = self.technical.calculate_all(df)
        
        return df
    
    def process_lending_features(
        self,
        marginfi_data: pd.DataFrame,
        kamino_data: pd.DataFrame,
        solend_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create features for lending optimization strategy (Solana).

        Features:
        - Supply APY differential
        - Borrow APY differential
        - Utilization rate trends
        - Rate volatility
        - Protocol-specific features
        """
        # Merge MarginFi and Kamino data
        df = marginfi_data.merge(
            kamino_data,
            on=["timestamp", "asset"],
            suffixes=("_marginfi", "_kamino"),
            how="outer"
        )

        # Rate differentials
        df["supply_apy_diff"] = df["supply_apy_marginfi"] - df["supply_apy_kamino"]
        df["borrow_apy_diff"] = df["borrow_apy_marginfi"] - df["borrow_apy_kamino"]

        # Best protocol flags
        df["best_supply"] = np.where(
            df["supply_apy_marginfi"] > df["supply_apy_kamino"],
            0,  # MarginFi
            1   # Kamino
        )
        df["best_borrow"] = np.where(
            df["borrow_apy_aave"] < df["borrow_apy_compound"],
            1,  # Aave (lower is better for borrowing)
            0   # Compound
        )
        
        # Utilization trends
        for col in ["utilization_aave", "utilization_compound"]:
            if col in df.columns:
                df[f"{col}_sma_5"] = df[col].rolling(5).mean()
                df[f"{col}_trend"] = df[col].pct_change(5)
        
        # Rate volatility
        for col in ["supply_apy_aave", "supply_apy_compound"]:
            if col in df.columns:
                df[f"{col}_std_5"] = df[col].rolling(5).std()
        
        return df
    
    def process_lp_features(self, curve_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for LP provision strategy.
        
        Features:
        - APY trends
        - TVL changes
        - Volume patterns
        - Impermanent loss indicators
        """
        df = curve_data.copy()
        
        # APY features
        if "base_apy" in df.columns:
            df["total_apy"] = df["base_apy"] + df.get("reward_apy", 0)
            df["apy_sma_5"] = df["total_apy"].rolling(5).mean()
            df["apy_trend"] = df["total_apy"].pct_change(5)
        
        # TVL features
        if "tvl_usd" in df.columns:
            df["tvl_log"] = np.log1p(df["tvl_usd"])
            df["tvl_change"] = df["tvl_usd"].pct_change()
            df["tvl_sma_5"] = df["tvl_usd"].rolling(5).mean()
        
        # Volume features
        if "volume_24h" in df.columns:
            df["volume_log"] = np.log1p(df["volume_24h"])
            df["volume_tvl_ratio"] = df["volume_24h"] / df["tvl_usd"].replace(0, 1)

        return df

    def create_arbitrage_labels(
        self,
        df: pd.DataFrame,
        min_profit_pct: float | None = None,
        gas_cost_estimate: bool = True,
        slippage: float | None = None
    ) -> pd.DataFrame:
        """
        Create labels for arbitrage prediction.

        Label = 1 if: spread - fee - gas_cost - slippage > min_profit

        Note: This uses Solana-native gas costs (very low compared to Ethereum).
        Solana tx fees: ~0.000055 SOL (~$0.01 at $200 SOL)
        """
        df = df.copy()
        min_profit = min_profit_pct or self.labeling_params.get("min_profit_pct", 0.003)
        slippage = slippage or self.labeling_params.get("slippage_estimate", 0.001)

        # Solana DEX fees: Raydium 0.25%, Orca 0.30%
        # Use 0.3% as conservative estimate for main pools
        fee = 0.003

        # Calculate net profit
        df["gross_profit"] = df["spread_abs"] - fee

        # Solana gas costs are negligible (~$0.01 per tx)
        # For a $10k trade, this is 0.0001% - effectively zero
        # Set to 0 since Solana tx fees don't materially impact profitability
        df["gas_cost_pct"] = 0

        df["net_profit"] = df["gross_profit"] - df["gas_cost_pct"] - slippage

        # Binary label
        df["profitable"] = (df["net_profit"] > min_profit).astype(int)

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        df = df.copy()

        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = self.scaler.transform(df[feature_columns])

        return df

    def get_feature_columns(self, strategy: str) -> list[str]:
        """Get feature column names for a strategy."""
        base_features = [
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_width", "bb_pct_b"
        ]

        if strategy == "arbitrage":
            return base_features + [
                "spread_abs", "spread_direction", "spread_sma_5", "spread_std_5",
                "gas_price_gwei", "gas_sma_5", "gas_trend",
                "liquidity_log", "liquidity_change",
                "return_1", "return_5"
            ]
        elif strategy == "lending":
            return [
                "supply_apy_diff", "borrow_apy_diff",
                "utilization_aave", "utilization_compound",
                "utilization_aave_trend", "utilization_compound_trend",
                "supply_apy_aave_std_5", "supply_apy_compound_std_5"
            ]
        elif strategy == "lp":
            return [
                "total_apy", "apy_sma_5", "apy_trend",
                "tvl_log", "tvl_change",
                "volume_log", "volume_tvl_ratio"
            ]
        else:
            return base_features

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        strategy: str,
        train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Prepare final dataset for model training/inference.

        Returns:
            Tuple of (features_df, labels_series or None)
        """
        df = df.copy()

        # Get feature columns
        feature_cols = self.get_feature_columns(strategy)
        available_cols = [c for c in feature_cols if c in df.columns]

        # Drop rows with NaN in feature columns
        df = df.dropna(subset=available_cols)

        # Extract features
        X = df[available_cols]

        # Extract labels if training
        y = None
        if train and "profitable" in df.columns:
            y = df["profitable"]
        elif train and "target" in df.columns:
            y = df["target"]

        self.logger.info(
            "Prepared dataset",
            strategy=strategy,
            samples=len(X),
            features=len(available_cols)
        )

        return X, y
