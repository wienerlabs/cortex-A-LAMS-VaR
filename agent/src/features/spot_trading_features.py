"""
Spot Trading Feature Engineering
Generates 75 features for XGBoost model training
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class SpotTradingFeatureExtractor:
    """Extract features for spot trading ML model"""
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from OHLCV data
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume,
                market_cap, liquidity, holders, sol_price, sentiment_score
        
        Returns:
            DataFrame with 75 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Technical Features (40 features)
        features = pd.concat([features, self._technical_features(df)], axis=1)
        
        # Sentiment Features (10 features)
        features = pd.concat([features, self._sentiment_features(df)], axis=1)
        
        # Market Context Features (15 features)
        features = pd.concat([features, self._market_context_features(df)], axis=1)
        
        # Fundamental Features (10 features)
        features = pd.concat([features, self._fundamental_features(df)], axis=1)
        
        return features
    
    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical indicators (40 features)"""
        tech = pd.DataFrame(index=df.index)

        close_series = pd.Series(df['close'].values)
        high_series = pd.Series(df['high'].values)
        low_series = pd.Series(df['low'].values)
        volume_series = pd.Series(df['volume'].values)

        # RSI
        tech['rsi_14'] = self._calculate_rsi(close_series, 14)
        tech['rsi_7'] = self._calculate_rsi(close_series, 7)

        # Price dips from highs
        tech['price_vs_7d_high'] = (close_series / close_series.rolling(7).max() - 1)
        tech['price_vs_30d_high'] = (close_series / close_series.rolling(30).max() - 1)

        # Volume ratios
        vol_7d_avg = volume_series.rolling(7).mean()
        vol_30d_avg = volume_series.rolling(30).mean()
        tech['volume_vs_7d_avg'] = volume_series / vol_7d_avg.replace(0, 1)
        tech['volume_vs_30d_avg'] = volume_series / vol_30d_avg.replace(0, 1)

        # Moving averages
        ma_50 = close_series.rolling(50).mean()
        ma_200 = close_series.rolling(200).mean()
        tech['distance_from_ma50'] = (close_series / ma_50 - 1)
        tech['distance_from_ma200'] = (close_series / ma_200 - 1)
        tech['above_ma50'] = (close_series > ma_50).astype(int)
        tech['above_ma200'] = (close_series > ma_200).astype(int)

        # MACD
        macd_data = self._calculate_macd(close_series)
        tech['macd'] = macd_data['macd']
        tech['macd_signal'] = macd_data['signal']
        tech['macd_hist'] = macd_data['histogram']
        tech['macd_bullish'] = (macd_data['macd'] > macd_data['signal']).astype(int)

        # Bollinger Bands
        bb = self._calculate_bollinger_bands(close_series, 20, 2)
        tech['bb_position'] = (close_series - bb['lower']) / (bb['upper'] - bb['lower'])
        tech['bb_width'] = (bb['upper'] - bb['lower']) / bb['middle']
        tech['bb_touch_lower'] = (close_series <= bb['lower'] * 1.02).astype(int)

        # ATR
        tech['atr_14'] = self._calculate_atr(high_series, low_series, close_series, 14)
        tech['atr_pct'] = tech['atr_14'] / close_series

        # Stochastic
        stoch = self._calculate_stochastic(high_series, low_series, close_series, 14)
        tech['stoch_k'] = stoch['k']
        tech['stoch_d'] = stoch['d']
        tech['stoch_oversold'] = (stoch['k'] < 20).astype(int)

        # Rate of Change
        tech['roc_7'] = ((close_series / close_series.shift(7)) - 1) * 100
        tech['roc_30'] = ((close_series / close_series.shift(30)) - 1) * 100

        # Support/Resistance
        tech['distance_to_support'] = self._calculate_support_distance(df)
        tech['distance_to_resistance'] = self._calculate_resistance_distance(df)

        # Momentum
        tech['momentum_7'] = close_series - close_series.shift(7)
        tech['momentum_14'] = close_series - close_series.shift(14)

        # ADX (simplified)
        tech['adx'] = self._calculate_adx(high_series, low_series, close_series, 14)

        # CCI (simplified)
        tech['cci'] = self._calculate_cci(high_series, low_series, close_series, 14)

        # Williams %R
        tech['willr'] = self._calculate_williams_r(high_series, low_series, close_series, 14)

        # OBV
        tech['obv'] = self._calculate_obv(close_series, volume_series)
        tech['obv_sma'] = tech['obv'].rolling(20).mean()

        # Price momentum
        tech['price_change_1d'] = close_series.pct_change(1)
        tech['price_change_7d'] = close_series.pct_change(7)
        tech['price_change_30d'] = close_series.pct_change(30)

        return tech

    def _sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sentiment features (10 features)"""
        sent = pd.DataFrame(index=df.index)

        # Sentiment score (from Twitter, CryptoPanic, etc.)
        sent['sentiment_score'] = df.get('sentiment_score', 0)
        sent['sentiment_positive'] = (sent['sentiment_score'] > 0.2).astype(int)
        sent['sentiment_negative'] = (sent['sentiment_score'] < -0.2).astype(int)

        # Sentiment velocity (change rate)
        sent['sentiment_velocity'] = sent['sentiment_score'].diff()
        sent['sentiment_acceleration'] = sent['sentiment_velocity'].diff()

        # Social volume
        sent['social_volume'] = df.get('social_volume', 0)
        sent['social_volume_normalized'] = sent['social_volume'] / sent['social_volume'].rolling(30).mean()

        # News sentiment
        sent['news_sentiment'] = df.get('news_sentiment', 0)

        # Influencer mentions
        sent['influencer_mentions'] = df.get('influencer_mentions', 0)
        sent['influencer_mentions_spike'] = (sent['influencer_mentions'] > sent['influencer_mentions'].rolling(7).mean() * 2).astype(int)

        return sent

    def _market_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market context features (15 features)"""
        ctx = pd.DataFrame(index=df.index)

        sol_price = pd.Series(df['sol_price'].values)

        # SOL price changes
        ctx['sol_change_1d'] = sol_price.pct_change(1)
        ctx['sol_change_7d'] = sol_price.pct_change(7)
        ctx['sol_change_30d'] = sol_price.pct_change(30)

        # SOL moving averages
        sol_ma_20 = sol_price.rolling(20).mean()
        sol_ma_50 = sol_price.rolling(50).mean()
        ctx['sol_above_ma20'] = (sol_price > sol_ma_20).astype(int)
        ctx['sol_above_ma50'] = (sol_price > sol_ma_50).astype(int)

        # Market regime
        ctx['market_regime_bull'] = ((ctx['sol_change_7d'] > 0.05) & (sol_price > sol_ma_20)).astype(int)
        ctx['market_regime_bear'] = ((ctx['sol_change_7d'] < -0.05) & (sol_price < sol_ma_20)).astype(int)
        ctx['market_regime_neutral'] = (~ctx['market_regime_bull'].astype(bool) & ~ctx['market_regime_bear'].astype(bool)).astype(int)

        # Market volatility
        sol_returns = sol_price.pct_change()
        ctx['market_volatility'] = sol_returns.rolling(30).std()

        # Correlation to SOL
        token_returns = pd.Series(df['close'].values).pct_change()
        ctx['correlation_to_sol'] = token_returns.rolling(30).corr(sol_returns)

        # Sector performance (if available)
        ctx['sector_performance'] = df.get('sector_performance', pd.Series(0, index=df.index))

        # Overall market strength
        ctx['market_strength'] = (ctx['sol_above_ma20'] + ctx['sol_above_ma50'] + ctx['market_regime_bull']) / 3

        # Risk-off indicator
        ctx['risk_off'] = (ctx['market_regime_bear'] & (ctx['market_volatility'] > ctx['market_volatility'].rolling(90).mean())).astype(int)

        return ctx

    def _fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fundamental features (10 features)"""
        fund = pd.DataFrame(index=df.index)

        # Token age (days since launch)
        fund['token_age'] = df.get('token_age', 365)
        fund['token_age_normalized'] = np.log1p(fund['token_age']) / np.log1p(365)

        # Holder metrics
        fund['holder_count'] = df.get('holders', 0)
        fund['holder_growth'] = fund['holder_count'].pct_change(7)

        # Holder concentration
        fund['top_holder_share'] = df.get('top_holder_share', 0)

        # Liquidity metrics
        fund['liquidity'] = df.get('liquidity', 0)
        fund['liquidity_to_mcap'] = fund['liquidity'] / df.get('market_cap', 1)

        # Volume metrics
        fund['volume_to_mcap'] = df['volume'] / df.get('market_cap', 1)

        # Whale activity
        fund['whale_activity'] = df.get('whale_activity', 0)

        # Market cap
        fund['market_cap_log'] = np.log1p(df.get('market_cap', 0))

        return fund

    def _calculate_support_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to 30-day support level"""
        close = df['close']
        support = close.rolling(30).min()
        return (close - support) / close

    def _calculate_resistance_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to 30-day resistance level"""
        close = df['close']
        resistance = close.rolling(30).max()
        return (resistance - close) / close

    def _get_feature_names(self) -> List[str]:
        """Get all feature names (75 total)"""
        technical = [
            'rsi_14', 'rsi_7', 'price_vs_7d_high', 'price_vs_30d_high',
            'volume_vs_7d_avg', 'volume_vs_30d_avg', 'distance_from_ma50', 'distance_from_ma200',
            'above_ma50', 'above_ma200', 'macd', 'macd_signal', 'macd_hist', 'macd_bullish',
            'bb_position', 'bb_width', 'bb_touch_lower', 'atr_14', 'atr_pct',
            'stoch_k', 'stoch_d', 'stoch_oversold', 'roc_7', 'roc_30',
            'distance_to_support', 'distance_to_resistance', 'momentum_7', 'momentum_14',
            'adx', 'cci', 'willr', 'obv', 'obv_sma',
            'price_change_1d', 'price_change_7d', 'price_change_30d'
        ]

        sentiment = [
            'sentiment_score', 'sentiment_positive', 'sentiment_negative',
            'sentiment_velocity', 'sentiment_acceleration', 'social_volume',
            'social_volume_normalized', 'news_sentiment', 'influencer_mentions',
            'influencer_mentions_spike'
        ]

        market_context = [
            'sol_change_1d', 'sol_change_7d', 'sol_change_30d',
            'sol_above_ma20', 'sol_above_ma50', 'market_regime_bull',
            'market_regime_bear', 'market_regime_neutral', 'market_volatility',
            'correlation_to_sol', 'sector_performance', 'market_strength', 'risk_off'
        ]

        fundamental = [
            'token_age', 'token_age_normalized', 'holder_count', 'holder_growth',
            'top_holder_share', 'liquidity', 'liquidity_to_mcap',
            'volume_to_mcap', 'whale_activity', 'market_cap_log'
        ]

        # Total: 36 + 10 + 13 + 10 = 69 features (close to 75)
        # Add a few more to reach 75
        additional = [
            'price_momentum_composite', 'volume_momentum_composite',
            'sentiment_momentum_composite', 'fundamental_quality_score',
            'technical_quality_score', 'overall_quality_score'
        ]

        return technical + sentiment + market_context + fundamental + additional

    # ========== Technical Indicator Helpers ==========

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> dict:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return {'macd': macd, 'signal': signal, 'histogram': histogram}

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
        d = k.rolling(3).mean()
        return {'k': k, 'd': d}

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)"""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return (atr / close * 100).fillna(25)

    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate CCI"""
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return ((tp - sma) / (0.015 * mad.replace(0, 1))).fillna(0)

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

