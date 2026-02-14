"""
LP Rebalancer Model Configuration
"""
from dataclasses import dataclass
from typing import List

# Target pools for data collection
TARGET_POOLS = [
    # Raydium
    {"dex": "raydium", "name": "WSOL/pippin", "address": ""},
    {"dex": "raydium", "name": "WSOL/USDC", "address": ""},
    {"dex": "raydium", "name": "WSOL/Fartcoin", "address": ""},
    {"dex": "raydium", "name": "MEW/WSOL", "address": ""},
    # Orca
    {"dex": "orca", "name": "SOL-USDC", "address": ""},
    {"dex": "orca", "name": "SOL-cbBTC", "address": ""},
    {"dex": "orca", "name": "JTO-JitoSOL", "address": ""},
    {"dex": "orca", "name": "JLP-USDC", "address": ""},
    # Meteora
    {"dex": "meteora", "name": "SOL-USDC", "address": ""},
    {"dex": "meteora", "name": "TRUMP-USDC", "address": ""},
]

# Collection parameters
COLLECTION_INTERVAL_HOURS = 1
COLLECTION_DURATION_DAYS = 30
TOTAL_SAMPLES_PER_POOL = 24 * 30  # 720 hourly samples

# Feature windows
TREND_WINDOW_HOURS = 168  # 7 days
VOLATILITY_WINDOW_HOURS = 24
MA_WINDOWS = [6, 12, 24, 48, 168]  # Moving average windows

# Label generation
FORWARD_WINDOW_HOURS = 24  # Look 24h ahead for labeling
APY_THRESHOLD_PCT = 0  # If future APY > current, label = 1


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    
    # APY features
    apy_current: bool = True
    apy_ma_6h: bool = True
    apy_ma_24h: bool = True
    apy_ma_168h: bool = True
    apy_trend_7d: bool = True  # Slope of 7d trend
    apy_volatility_24h: bool = True
    
    # Volume features
    volume_24h: bool = True
    volume_tvl_ratio: bool = True
    volume_trend_7d: bool = True
    volume_ma_24h: bool = True
    
    # TVL features
    tvl_current: bool = True
    tvl_stability_7d: bool = True  # Std dev / mean
    tvl_trend_7d: bool = True
    
    # IL features
    il_estimated: bool = True  # Based on price divergence
    il_change_24h: bool = True
    
    # Token volatility
    token_a_volatility_24h: bool = True
    token_b_volatility_24h: bool = True
    pair_correlation: bool = True
    
    # Fee features
    fee_earnings_24h: bool = True
    fee_apy_contribution: bool = True
    
    # Market context
    sol_price_change_24h: bool = True
    btc_price_change_24h: bool = True
    market_volatility: bool = True
    
    # Time features
    hour_of_day: bool = True
    day_of_week: bool = True
    
    @property
    def feature_count(self) -> int:
        return sum(1 for v in self.__dict__.values() if v is True)


@dataclass 
class ModelConfig:
    """XGBoost model configuration."""
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    
    # Training
    test_size: float = 0.2
    validation_size: float = 0.1
    early_stopping_rounds: int = 10
    
    # Thresholds
    exit_threshold: float = 0.4  # If P(stay) < 0.4, recommend exit
    stay_threshold: float = 0.6  # If P(stay) > 0.6, recommend stay
    
    # Rebalancing constraints
    min_hold_hours: int = 24  # Minimum hold time before exit
    max_positions: int = 5  # Maximum concurrent positions
    rebalance_cooldown_hours: int = 6  # Minimum time between rebalances


# Database paths
DATA_DIR = "data/lp_rebalancer"
RAW_DATA_PATH = f"{DATA_DIR}/raw_pool_data.parquet"
FEATURES_PATH = f"{DATA_DIR}/features.parquet"
MODEL_PATH = f"{DATA_DIR}/xgboost_model.json"
BACKTEST_PATH = f"{DATA_DIR}/backtest_results.parquet"

