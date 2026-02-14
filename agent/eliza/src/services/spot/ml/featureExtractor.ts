/**
 * Spot Trading Feature Extractor
 * Extracts 75 features from token data for ML model
 */

import type { ApprovedToken } from '../../trading/types.js';
import type { SpotFeatures } from './spotMLModel.js';
import { logger } from '../../logger.js';

export interface TokenMarketData {
  // OHLCV data
  prices: number[];          // Historical close prices (last 200 days)
  volumes: number[];         // Historical volumes
  timestamps: number[];      // Timestamps
  
  // Current data
  currentPrice: number;
  currentVolume: number;
  
  // SOL data
  solPrices: number[];       // SOL historical prices
  currentSolPrice: number;
  
  // Sentiment data
  sentimentScore?: number;
  socialVolume?: number;
  newsSentiment?: number;
  influencerMentions?: number;
  
  // Fundamental data
  marketCap: number;
  liquidity: number;
  holders: number;
  tokenAge: number;
  topHolderShare: number;
  whaleActivity?: number;
  sectorPerformance?: number;
}

export class SpotFeatureExtractor {
  /**
   * Extract all 75 features from token data
   */
  async extractFeatures(
    token: ApprovedToken,
    marketData: TokenMarketData
  ): Promise<Partial<SpotFeatures>> {
    try {
      const features: Partial<SpotFeatures> = {};
      
      // Technical features (40)
      Object.assign(features, this.extractTechnicalFeatures(marketData));
      
      // Sentiment features (10)
      Object.assign(features, this.extractSentimentFeatures(marketData));
      
      // Market context features (13)
      Object.assign(features, this.extractMarketContextFeatures(marketData));
      
      // Fundamental features (10)
      Object.assign(features, this.extractFundamentalFeatures(token, marketData));
      
      // Composite features (6)
      Object.assign(features, this.extractCompositeFeatures(features));
      
      return features;
    } catch (error) {
      logger.error('[FeatureExtractor] Failed to extract features', { error, token: token.symbol });
      throw error;
    }
  }
  
  private extractTechnicalFeatures(data: TokenMarketData): Partial<SpotFeatures> {
    const prices = data.prices;
    const volumes = data.volumes;
    const currentPrice = data.currentPrice;
    const currentVolume = data.currentVolume;
    
    if (prices.length < 30) {
      logger.warn('[FeatureExtractor] Insufficient price history', { length: prices.length });
    }
    
    const features: Partial<SpotFeatures> = {};
    
    // RSI
    features.rsi_14 = this.calculateRSI(prices, 14);
    features.rsi_7 = this.calculateRSI(prices, 7);
    
    // Price vs highs
    const high7d = Math.max(...prices.slice(-7));
    const high30d = Math.max(...prices.slice(-30));
    features.price_vs_7d_high = (currentPrice / high7d) - 1;
    features.price_vs_30d_high = (currentPrice / high30d) - 1;
    
    // Volume ratios
    const vol7dAvg = this.average(volumes.slice(-7));
    const vol30dAvg = this.average(volumes.slice(-30));
    features.volume_vs_7d_avg = currentVolume / (vol7dAvg || 1);
    features.volume_vs_30d_avg = currentVolume / (vol30dAvg || 1);
    
    // Moving averages
    const ma50 = this.sma(prices, 50);
    const ma200 = this.sma(prices, 200);
    features.distance_from_ma50 = (currentPrice / ma50) - 1;
    features.distance_from_ma200 = (currentPrice / ma200) - 1;
    features.above_ma50 = currentPrice > ma50 ? 1 : 0;
    features.above_ma200 = currentPrice > ma200 ? 1 : 0;
    
    // MACD
    const macd = this.calculateMACD(prices);
    features.macd = macd.macd;
    features.macd_signal = macd.signal;
    features.macd_hist = macd.histogram;
    features.macd_bullish = macd.macd > macd.signal ? 1 : 0;
    
    // Bollinger Bands
    const bb = this.calculateBollingerBands(prices, 20, 2);
    features.bb_position = (currentPrice - bb.lower) / (bb.upper - bb.lower);
    features.bb_width = (bb.upper - bb.lower) / bb.middle;
    features.bb_touch_lower = currentPrice <= bb.lower * 1.02 ? 1 : 0;
    
    // ATR
    features.atr_14 = this.calculateATR(prices, 14);
    features.atr_pct = features.atr_14 / currentPrice;
    
    // Stochastic
    const stoch = this.calculateStochastic(prices, 14);
    features.stoch_k = stoch.k;
    features.stoch_d = stoch.d;
    features.stoch_oversold = stoch.k < 20 ? 1 : 0;
    
    // Rate of Change
    features.roc_7 = this.calculateROC(prices, 7);
    features.roc_30 = this.calculateROC(prices, 30);
    
    // Support/Resistance
    const support30d = Math.min(...prices.slice(-30));
    const resistance30d = Math.max(...prices.slice(-30));
    features.distance_to_support = (currentPrice - support30d) / currentPrice;
    features.distance_to_resistance = (resistance30d - currentPrice) / currentPrice;
    
    // Momentum
    features.momentum_7 = this.calculateMomentum(prices, 7);
    features.momentum_14 = this.calculateMomentum(prices, 14);
    
    // ADX (simplified)
    features.adx = this.calculateADX(prices, 14);
    
    // CCI (simplified)
    features.cci = this.calculateCCI(prices, 14);
    
    // Williams %R
    features.willr = this.calculateWilliamsR(prices, 14);
    
    // OBV (simplified - using volume only)
    features.obv = this.calculateOBV(prices, volumes);
    features.obv_sma = this.sma([features.obv], 20);
    
    // Price changes
    features.price_change_1d = prices.length >= 2 ? (prices[prices.length - 1] / prices[prices.length - 2]) - 1 : 0;
    features.price_change_7d = prices.length >= 8 ? (prices[prices.length - 1] / prices[prices.length - 8]) - 1 : 0;
    features.price_change_30d = prices.length >= 31 ? (prices[prices.length - 1] / prices[prices.length - 31]) - 1 : 0;
    
    return features;
  }

  private extractSentimentFeatures(data: TokenMarketData): Partial<SpotFeatures> {
    const features: Partial<SpotFeatures> = {};

    const sentiment = data.sentimentScore || 0;
    features.sentiment_score = sentiment;
    features.sentiment_positive = sentiment > 0.2 ? 1 : 0;
    features.sentiment_negative = sentiment < -0.2 ? 1 : 0;
    features.sentiment_velocity = 0; // Would need historical sentiment
    features.sentiment_acceleration = 0;
    features.social_volume = data.socialVolume || 0;
    features.social_volume_normalized = 1.0; // Would need historical average
    features.news_sentiment = data.newsSentiment || 0;
    features.influencer_mentions = data.influencerMentions || 0;
    features.influencer_mentions_spike = 0;

    return features;
  }

  private extractMarketContextFeatures(data: TokenMarketData): Partial<SpotFeatures> {
    const features: Partial<SpotFeatures> = {};

    const solPrices = data.solPrices;
    const currentSol = data.currentSolPrice;

    if (solPrices.length >= 31) {
      features.sol_change_1d = (solPrices[solPrices.length - 1] / solPrices[solPrices.length - 2]) - 1;
      features.sol_change_7d = (solPrices[solPrices.length - 1] / solPrices[solPrices.length - 8]) - 1;
      features.sol_change_30d = (solPrices[solPrices.length - 1] / solPrices[solPrices.length - 31]) - 1;
    } else {
      features.sol_change_1d = 0;
      features.sol_change_7d = 0;
      features.sol_change_30d = 0;
    }

    const solMa20 = this.sma(solPrices, 20);
    const solMa50 = this.sma(solPrices, 50);
    features.sol_above_ma20 = currentSol > solMa20 ? 1 : 0;
    features.sol_above_ma50 = currentSol > solMa50 ? 1 : 0;

    // Market regime
    const isBull = (features.sol_change_7d || 0) > 0.05 && features.sol_above_ma20 === 1;
    const isBear = (features.sol_change_7d || 0) < -0.05 && features.sol_above_ma20 === 0;
    features.market_regime_bull = isBull ? 1 : 0;
    features.market_regime_bear = isBear ? 1 : 0;
    features.market_regime_neutral = (!isBull && !isBear) ? 1 : 0;

    // Market volatility
    features.market_volatility = this.calculateVolatility(solPrices, 30);

    // Correlation to SOL
    features.correlation_to_sol = this.calculateCorrelation(data.prices, solPrices);

    features.sector_performance = data.sectorPerformance || 0;
    features.market_strength = ((features.sol_above_ma20 || 0) + (features.sol_above_ma50 || 0) + (features.market_regime_bull || 0)) / 3;

    const avgVol = this.average(solPrices.slice(-90).map((p, i, arr) => i > 0 ? Math.abs(p / arr[i-1] - 1) : 0));
    features.risk_off = features.market_regime_bear === 1 && (features.market_volatility || 0) > avgVol ? 1 : 0;

    return features;
  }

  private extractFundamentalFeatures(token: ApprovedToken, data: TokenMarketData): Partial<SpotFeatures> {
    const features: Partial<SpotFeatures> = {};

    features.token_age = data.tokenAge;
    features.token_age_normalized = Math.log1p(data.tokenAge) / Math.log1p(365);
    features.holder_count = data.holders;
    features.holder_growth = 0; // Would need historical holder data
    features.top_holder_share = data.topHolderShare;
    features.liquidity = data.liquidity;
    features.liquidity_to_mcap = data.liquidity / (data.marketCap || 1);
    features.volume_to_mcap = data.currentVolume / (data.marketCap || 1);
    features.whale_activity = data.whaleActivity || 0;
    features.market_cap_log = Math.log1p(data.marketCap);

    return features;
  }

  private extractCompositeFeatures(features: Partial<SpotFeatures>): Partial<SpotFeatures> {
    const composite: Partial<SpotFeatures> = {};

    // Price momentum composite
    composite.price_momentum_composite = (
      (features.roc_7 || 0) * 0.4 +
      (features.roc_30 || 0) * 0.3 +
      (features.momentum_7 || 0) * 0.3
    );

    // Volume momentum composite
    composite.volume_momentum_composite = (
      (features.volume_vs_7d_avg || 0) * 0.6 +
      (features.volume_vs_30d_avg || 0) * 0.4
    );

    // Sentiment momentum composite
    composite.sentiment_momentum_composite = (
      (features.sentiment_score || 0) * 0.5 +
      (features.news_sentiment || 0) * 0.3 +
      (features.social_volume_normalized || 0) * 0.2
    );

    // Fundamental quality score
    composite.fundamental_quality_score = (
      (features.liquidity_to_mcap || 0) * 0.3 +
      (1 - (features.top_holder_share || 0)) * 0.3 +
      (features.token_age_normalized || 0) * 0.2 +
      (features.volume_to_mcap || 0) * 0.2
    );

    // Technical quality score
    composite.technical_quality_score = (
      ((features.above_ma50 || 0) + (features.above_ma200 || 0)) / 2 * 0.4 +
      (features.macd_bullish || 0) * 0.3 +
      (1 - Math.abs((features.rsi_14 || 50) - 50) / 50) * 0.3
    );

    // Overall quality score
    composite.overall_quality_score = (
      (composite.fundamental_quality_score || 0) * 0.4 +
      (composite.technical_quality_score || 0) * 0.3 +
      (composite.sentiment_momentum_composite || 0) * 0.3
    );

    return composite;
  }

  // ========== Technical Indicator Calculations ==========

  private average(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private sma(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1] || 0;
    return this.average(prices.slice(-period));
  }

  private ema(prices: number[], period: number): number {
    if (prices.length === 0) return 0;
    if (prices.length < period) return this.average(prices);

    const multiplier = 2 / (period + 1);
    let ema = this.average(prices.slice(0, period));

    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema;
    }

    return ema;
  }

  private calculateRSI(prices: number[], period: number = 14): number {
    if (prices.length < period + 1) return 50;

    const changes = prices.slice(-period - 1).map((p, i, arr) => i > 0 ? p - arr[i - 1] : 0).slice(1);
    const gains = changes.map(c => c > 0 ? c : 0);
    const losses = changes.map(c => c < 0 ? -c : 0);

    const avgGain = this.average(gains);
    const avgLoss = this.average(losses);

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(prices: number[]): { macd: number; signal: number; histogram: number } {
    const ema12 = this.ema(prices, 12);
    const ema26 = this.ema(prices, 26);
    const macd = ema12 - ema26;

    // Signal line is 9-period EMA of MACD
    // Simplified: use current MACD as signal
    const signal = macd * 0.9; // Approximation
    const histogram = macd - signal;

    return { macd, signal, histogram };
  }

  private calculateBollingerBands(prices: number[], period: number = 20, stdDev: number = 2): { upper: number; middle: number; lower: number } {
    const middle = this.sma(prices, period);
    const slice = prices.slice(-period);
    const variance = this.average(slice.map(p => Math.pow(p - middle, 2)));
    const std = Math.sqrt(variance);

    return {
      upper: middle + (std * stdDev),
      middle,
      lower: middle - (std * stdDev),
    };
  }

  private calculateATR(prices: number[], period: number = 14): number {
    if (prices.length < period + 1) return 0;

    const trueRanges = prices.slice(-period - 1).map((p, i, arr) => {
      if (i === 0) return 0;
      return Math.abs(p - arr[i - 1]);
    }).slice(1);

    return this.average(trueRanges);
  }

  private calculateStochastic(prices: number[], period: number = 14): { k: number; d: number } {
    if (prices.length < period) return { k: 50, d: 50 };

    const slice = prices.slice(-period);
    const high = Math.max(...slice);
    const low = Math.min(...slice);
    const current = prices[prices.length - 1];

    const k = ((current - low) / (high - low)) * 100;
    const d = k * 0.9; // Simplified: 3-period SMA of %K

    return { k, d };
  }

  private calculateROC(prices: number[], period: number): number {
    if (prices.length < period + 1) return 0;
    const current = prices[prices.length - 1];
    const past = prices[prices.length - period - 1];
    return ((current - past) / past) * 100;
  }

  private calculateMomentum(prices: number[], period: number): number {
    if (prices.length < period + 1) return 0;
    return prices[prices.length - 1] - prices[prices.length - period - 1];
  }

  private calculateADX(prices: number[], period: number = 14): number {
    // Simplified ADX calculation
    if (prices.length < period + 1) return 25;

    const changes = prices.slice(-period - 1).map((p, i, arr) => i > 0 ? Math.abs(p - arr[i - 1]) : 0).slice(1);
    const avgChange = this.average(changes);
    const currentPrice = prices[prices.length - 1];

    return (avgChange / currentPrice) * 100;
  }

  private calculateCCI(prices: number[], period: number = 14): number {
    // Simplified CCI calculation
    const sma = this.sma(prices, period);
    const current = prices[prices.length - 1];
    const deviation = this.average(prices.slice(-period).map(p => Math.abs(p - sma)));

    if (deviation === 0) return 0;
    return (current - sma) / (0.015 * deviation);
  }

  private calculateWilliamsR(prices: number[], period: number = 14): number {
    if (prices.length < period) return -50;

    const slice = prices.slice(-period);
    const high = Math.max(...slice);
    const low = Math.min(...slice);
    const current = prices[prices.length - 1];

    return ((high - current) / (high - low)) * -100;
  }

  private calculateOBV(prices: number[], volumes: number[]): number {
    // Simplified OBV
    let obv = 0;
    for (let i = 1; i < Math.min(prices.length, volumes.length); i++) {
      if (prices[i] > prices[i - 1]) {
        obv += volumes[i];
      } else if (prices[i] < prices[i - 1]) {
        obv -= volumes[i];
      }
    }
    return obv;
  }

  private calculateVolatility(prices: number[], period: number): number {
    if (prices.length < period + 1) return 0;

    const returns = prices.slice(-period - 1).map((p, i, arr) => i > 0 ? (p / arr[i - 1]) - 1 : 0).slice(1);
    const variance = this.average(returns.map(r => r * r));
    return Math.sqrt(variance);
  }

  private calculateCorrelation(prices1: number[], prices2: number[]): number {
    const minLength = Math.min(prices1.length, prices2.length, 30);
    if (minLength < 2) return 0;

    const p1 = prices1.slice(-minLength);
    const p2 = prices2.slice(-minLength);

    const mean1 = this.average(p1);
    const mean2 = this.average(p2);

    let numerator = 0;
    let sum1 = 0;
    let sum2 = 0;

    for (let i = 0; i < minLength; i++) {
      const diff1 = p1[i] - mean1;
      const diff2 = p2[i] - mean2;
      numerator += diff1 * diff2;
      sum1 += diff1 * diff1;
      sum2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sum1 * sum2);
    return denominator === 0 ? 0 : numerator / denominator;
  }
}
