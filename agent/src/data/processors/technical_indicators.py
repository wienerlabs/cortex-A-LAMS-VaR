"""
Technical Indicators using TA-Lib.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from typing import Any

# TA-Lib is optional - fallback to numpy implementations if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False

try:
    from ...config import FEATURE_PARAMS
except ImportError:
    FEATURE_PARAMS = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "ema_periods": [7, 14, 30],
    }

logger = structlog.get_logger()


class TechnicalIndicators:
    """
    Calculate technical indicators for price data using TA-Lib.
    
    Indicators calculated:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - EMA (Exponential Moving Average)
    - Volume-based indicators
    """
    
    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or FEATURE_PARAMS
        self.logger = logger.bind(component="technical_indicators")
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the dataframe.
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            
        Returns:
            DataFrame with added indicator columns.
        """
        df = df.copy()
        
        # Ensure required columns exist
        required = ["close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate indicators
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_ema(df)
        df = self._add_price_lags(df)
        
        # Volume-based if volume exists
        if "volume" in df.columns:
            df = self._add_volume_features(df)
        
        self.logger.info("Calculated technical indicators", columns=list(df.columns))
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator."""
        period = self.params.get("rsi_period", 14)
        if HAS_TALIB:
            df["rsi"] = talib.RSI(df["close"].values, timeperiod=period)
        else:
            # Numpy fallback for RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        fast = self.params.get("macd_fast", 12)
        slow = self.params.get("macd_slow", 26)
        signal_period = self.params.get("macd_signal", 9)

        if HAS_TALIB:
            macd, macd_signal, macd_hist = talib.MACD(
                df["close"].values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal_period
            )
        else:
            # Numpy fallback for MACD
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
            macd_hist = macd - macd_signal

        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        period = self.params.get("bb_period", 20)
        std_dev = self.params.get("bb_std", 2)

        if HAS_TALIB:
            upper, middle, lower = talib.BBANDS(
                df["close"].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
        else:
            # Numpy fallback for Bollinger Bands
            middle = df["close"].rolling(window=period).mean()
            std = df["close"].rolling(window=period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)

        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower

        # Bollinger Band width and %B
        df["bb_width"] = (upper - lower) / middle
        df["bb_pct_b"] = (df["close"] - lower) / (upper - lower)
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        periods = [9, 21, 50, 200]

        for period in periods:
            if len(df) >= period:
                if HAS_TALIB:
                    df[f"ema_{period}"] = talib.EMA(df["close"].values, timeperiod=period)
                else:
                    df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        return df

    def _add_price_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price lag features."""
        lags = self.params.get("price_lag", [1, 3, 5, 10])

        for lag in lags:
            df[f"price_lag_{lag}"] = df["close"].shift(lag)
            df[f"return_{lag}"] = df["close"].pct_change(lag)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        windows = self.params.get("volume_rolling_window", [5, 15, 60])

        for window in windows:
            df[f"volume_sma_{window}"] = df["volume"].rolling(window=window).mean()
            df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_sma_{window}"]

        # Volume Rate of Change
        df["volume_roc"] = df["volume"].pct_change()

        # On-Balance Volume (OBV) - numpy fallback
        if HAS_TALIB:
            df["obv"] = talib.OBV(df["close"].values, df["volume"].values)
        else:
            obv = [0]
            for i in range(1, len(df)):
                if df["close"].iloc[i] > df["close"].iloc[i-1]:
                    obv.append(obv[-1] + df["volume"].iloc[i])
                elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                    obv.append(obv[-1] - df["volume"].iloc[i])
                else:
                    obv.append(obv[-1])
            df["obv"] = obv

        return df
