"""
Market Regime Detection Module

Classifies market conditions as BULL, BEAR, or SIDEWAYS based on
price action, trend strength, and volatility.

Used for:
- Regime-specific model validation
- Historical data labeling
- Real-time regime detection
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeConfig:
    """Configuration for regime detection thresholds."""
    # Return thresholds (15% for strong directional move)
    bull_return_threshold: float = 0.15  # >15% gain = BULL
    bear_return_threshold: float = -0.15  # <-15% loss = BEAR
    
    # Trend strength thresholds (0-1 scale, using price vs moving average)
    min_trend_strength: float = 0.6  # Need 60% trend strength for directional regime
    
    # Window sizes (in periods - typically hours or days)
    return_window: int = 30  # Look at last 30 periods for returns
    volatility_window: int = 30  # Look at last 30 periods for volatility
    trend_window: int = 14  # 14-period trend calculation
    
    # Volatility thresholds (annualized)
    high_volatility_threshold: float = 1.0  # Above 100% annualized vol
    low_volatility_threshold: float = 0.3   # Below 30% annualized vol


@dataclass
class RegimeResult:
    """Result of regime detection for a single point in time."""
    regime: MarketRegime
    confidence: float  # 0-1 confidence in the classification
    returns: float     # Period return used for classification
    trend_strength: float  # 0-1 trend strength score
    volatility: float  # Annualized volatility
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "returns": round(self.returns, 4),
            "trend_strength": round(self.trend_strength, 4),
            "volatility": round(self.volatility, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class RegimeDetector:
    """
    Detects market regime based on price action and trend analysis.
    
    Classifies markets as:
    - BULL: Price up >15% in window with strong uptrend (trend_strength > 0.6)
    - BEAR: Price down >15% in window with strong downtrend (trend_strength > 0.6)
    - SIDEWAYS: Price within ±15% range or weak trend
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._regime_history: List[RegimeResult] = []
    
    def detect(
        self, 
        price_data: pd.Series,
        timestamp: Optional[datetime] = None
    ) -> RegimeResult:
        """
        Classify current market regime based on recent price action.
        
        Args:
            price_data: Series of prices (must have at least config.return_window periods)
            timestamp: Optional timestamp for the result
            
        Returns:
            RegimeResult with classification and metrics
        """
        if len(price_data) < self.config.return_window:
            return RegimeResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                returns=0.0,
                trend_strength=0.0,
                volatility=0.0,
                timestamp=timestamp,
            )
        
        # Calculate period returns
        returns = self._calculate_returns(price_data)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(price_data)
        
        # Calculate volatility (annualized)
        volatility = self._calculate_volatility(price_data)
        
        # Classify regime
        regime, confidence = self._classify(returns, trend_strength, volatility)
        
        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            returns=returns,
            trend_strength=trend_strength,
            volatility=volatility,
            timestamp=timestamp,
        )
        
        self._regime_history.append(result)
        return result
    
    def _calculate_returns(self, price_data: pd.Series) -> float:
        """Calculate returns over the configured window."""
        window = min(self.config.return_window, len(price_data) - 1)
        if window <= 0:
            return 0.0
        
        current_price = price_data.iloc[-1]
        past_price = price_data.iloc[-(window + 1)]
        
        if past_price == 0 or pd.isna(past_price):
            return 0.0
        
        return (current_price - past_price) / past_price
    
    def _calculate_trend_strength(self, price_data: pd.Series) -> float:
        """
        Calculate trend strength using price vs moving average.
        
        Returns value between 0 (no trend) and 1 (strong trend).
        Positive values indicate uptrend, negative indicate downtrend,
        but we return absolute strength.
        """
        window = min(self.config.trend_window, len(price_data))
        if window < 3:
            return 0.0
        
        # Calculate moving average
        ma = price_data.rolling(window=window).mean()
        
        if ma.iloc[-1] == 0 or pd.isna(ma.iloc[-1]):
            return 0.0
        
        # Count how many periods price is consistently above/below MA
        recent_prices = price_data.iloc[-window:]
        recent_ma = ma.iloc[-window:]

        # Calculate consistency: how many periods are in same direction
        above_ma = (recent_prices > recent_ma).sum()
        consistency = max(above_ma, window - above_ma) / window

        # Calculate magnitude of deviation from MA
        current_deviation = abs(price_data.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]
        magnitude = min(1.0, current_deviation / 0.10)  # Normalize to 10% deviation

        # Combine consistency and magnitude
        trend_strength = (consistency * 0.6 + magnitude * 0.4)

        return min(1.0, trend_strength)

    def _calculate_volatility(self, price_data: pd.Series) -> float:
        """Calculate annualized volatility from returns."""
        window = min(self.config.volatility_window, len(price_data) - 1)
        if window < 2:
            return 0.0

        returns = price_data.pct_change().dropna()
        if len(returns) < 2:
            return 0.0

        recent_returns = returns.iloc[-window:]

        # Daily volatility to annualized (assuming 365 periods per year for crypto)
        daily_vol = recent_returns.std()
        annualized_vol = daily_vol * np.sqrt(365)

        return float(annualized_vol) if not pd.isna(annualized_vol) else 0.0

    def _classify(
        self,
        returns: float,
        trend_strength: float,
        volatility: float
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on returns, trend strength, and volatility.

        Returns:
            Tuple of (regime, confidence)
        """
        cfg = self.config

        # Strong uptrend: high positive returns with strong trend
        if returns > cfg.bull_return_threshold and trend_strength > cfg.min_trend_strength:
            confidence = min(1.0, (returns / cfg.bull_return_threshold) * trend_strength)
            return MarketRegime.BULL, confidence

        # Strong downtrend: high negative returns with strong trend
        if returns < cfg.bear_return_threshold and trend_strength > cfg.min_trend_strength:
            confidence = min(1.0, (abs(returns) / abs(cfg.bear_return_threshold)) * trend_strength)
            return MarketRegime.BEAR, confidence

        # Sideways: weak returns or weak trend
        # Higher confidence if returns are near zero and trend is weak
        return_neutrality = 1.0 - min(1.0, abs(returns) / cfg.bull_return_threshold)
        trend_weakness = 1.0 - trend_strength
        confidence = (return_neutrality * 0.7 + trend_weakness * 0.3)

        return MarketRegime.SIDEWAYS, confidence

    def get_regime_history(self) -> List[RegimeResult]:
        """Get history of regime detections."""
        return self._regime_history.copy()

    def clear_history(self) -> None:
        """Clear regime detection history."""
        self._regime_history = []


def detect_regime(
    price_data: pd.Series,
    window: int = 30,
    timestamp: Optional[datetime] = None
) -> str:
    """
    Convenience function to detect current market regime.

    Args:
        price_data: Series of prices
        window: Lookback window for analysis
        timestamp: Optional timestamp

    Returns:
        Regime string: "BULL", "BEAR", "SIDEWAYS", or "UNKNOWN"
    """
    config = RegimeConfig(return_window=window, volatility_window=window)
    detector = RegimeDetector(config)
    result = detector.detect(price_data, timestamp)
    return result.regime.value


def label_data_with_regimes(
    df: pd.DataFrame,
    price_column: str = "close",
    timestamp_column: Optional[str] = "timestamp",
    config: Optional[RegimeConfig] = None
) -> pd.DataFrame:
    """
    Add regime labels to a DataFrame with historical data.

    Args:
        df: DataFrame with price data
        price_column: Name of the price column
        timestamp_column: Name of the timestamp column (optional)
        config: Optional regime detection configuration

    Returns:
        DataFrame with added 'regime', 'regime_confidence',
        'regime_returns', 'regime_trend_strength' columns
    """
    cfg = config or RegimeConfig()
    detector = RegimeDetector(cfg)

    # Prepare output columns
    regimes = []
    confidences = []
    returns_list = []
    trend_strengths = []
    volatilities = []

    prices = df[price_column]

    for i in range(len(df)):
        # Use all data up to current point
        price_slice = prices.iloc[:i + 1]

        # Get timestamp if available
        ts = None
        if timestamp_column and timestamp_column in df.columns:
            ts_val = df[timestamp_column].iloc[i]
            if isinstance(ts_val, (int, float)):
                ts = datetime.fromtimestamp(ts_val)
            elif isinstance(ts_val, str):
                ts = datetime.fromisoformat(ts_val.replace('Z', '+00:00'))
            elif isinstance(ts_val, datetime):
                ts = ts_val

        result = detector.detect(price_slice, ts)

        regimes.append(result.regime.value)
        confidences.append(result.confidence)
        returns_list.append(result.returns)
        trend_strengths.append(result.trend_strength)
        volatilities.append(result.volatility)

    # Add columns to DataFrame
    df_out = df.copy()
    df_out["regime"] = regimes
    df_out["regime_confidence"] = confidences
    df_out["regime_returns"] = returns_list
    df_out["regime_trend_strength"] = trend_strengths
    df_out["regime_volatility"] = volatilities

    return df_out

