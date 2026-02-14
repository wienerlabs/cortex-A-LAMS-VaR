/**
 * Feature Extractor for Perps Funding Rate Prediction
 * 
 * Ports the Python feature engineering to TypeScript for real-time inference.
 * Computes 65 features from funding rate history and market data.
 * 
 * Features:
 * - Funding rate features (current, lagged, rolling stats)
 * - Price features (returns, volatility, momentum)
 * - Time features (hour, day of week, cyclical encoding)
 */
import { FEATURE_NAMES, NUM_FEATURES } from './modelLoader.js';
import { logger } from '../../logger.js';

/** Single data point for feature extraction */
export interface FundingDataPoint {
  timestamp: Date;
  fundingRate: number;           // Funding rate in decimal (0.001 = 0.1%)
  fundingRateRaw?: number;       // Raw funding rate
  oraclePrice?: number;          // Oracle TWAP
  markPrice?: number;            // Mark TWAP
  cumFundingLong?: number;       // Cumulative funding for longs
  cumFundingShort?: number;      // Cumulative funding for shorts
}

/** Buffer of historical data for feature computation */
export interface FeatureBuffer {
  data: FundingDataPoint[];
  maxSize: number;               // Max history to keep (168 hours = 1 week)
}

// Lookback windows in hours
const LOOKBACK_WINDOWS = [1, 4, 8, 24, 48, 168];
const LAG_HOURS = [1, 2, 4, 8, 12, 24];
const RETURN_PERIODS = [1, 4, 8, 24, 48];
const VOLATILITY_WINDOWS = [24, 48, 168];
const MOMENTUM_WINDOWS = [24, 48];
const MIN_HISTORY = 168; // 1 week for reliable features

/**
 * Feature Extractor for real-time perps trading
 */
export class PerpsFeatureExtractor {
  private buffer: FeatureBuffer;

  constructor(maxSize: number = 200) {
    this.buffer = {
      data: [],
      maxSize,
    };
  }

  /** Add a new data point to the buffer */
  addDataPoint(point: FundingDataPoint): void {
    this.buffer.data.push(point);
    
    // Trim to max size
    if (this.buffer.data.length > this.buffer.maxSize) {
      this.buffer.data = this.buffer.data.slice(-this.buffer.maxSize);
    }
  }

  /** Load historical data into buffer */
  loadHistory(data: FundingDataPoint[]): void {
    this.buffer.data = data.slice(-this.buffer.maxSize);
    logger.debug('Loaded history into feature buffer', { size: this.buffer.data.length });
  }

  /** Check if we have enough history for reliable features */
  hasEnoughHistory(): boolean {
    return this.buffer.data.length >= MIN_HISTORY;
  }

  /** Get current buffer size */
  getBufferSize(): number {
    return this.buffer.data.length;
  }

  /** Clear the buffer */
  clear(): void {
    this.buffer.data = [];
  }

  /**
   * Extract features from current buffer state
   * Returns array of 65 features in the order expected by the model
   */
  extractFeatures(): number[] {
    const n = this.buffer.data.length;
    if (n < 2) {
      throw new Error('Not enough data for feature extraction');
    }

    const fundingRates = this.buffer.data.map(d => d.fundingRate);
    const prices = this.buffer.data.map(d => d.oraclePrice ?? 0);
    const current = this.buffer.data[n - 1];
    const currentFunding = current.fundingRate;

    const features: Record<string, number> = {};

    // Current funding rate
    features['funding_rate'] = currentFunding;
    features['funding_rate_raw'] = current.fundingRateRaw ?? currentFunding;

    // Lagged funding rates
    for (const lag of LAG_HOURS) {
      const idx = n - 1 - lag;
      features[`funding_lag_${lag}h`] = idx >= 0 ? fundingRates[idx] : 0;
    }

    // Rolling statistics for each window
    for (const window of LOOKBACK_WINDOWS) {
      const slice = fundingRates.slice(-window);
      features[`funding_mean_${window}h`] = this.mean(slice);
      features[`funding_std_${window}h`] = this.std(slice);
      features[`funding_min_${window}h`] = Math.min(...slice);
      features[`funding_max_${window}h`] = Math.max(...slice);
      features[`funding_skew_${window}h`] = this.skewness(slice);
    }

    // Funding momentum
    features['funding_momentum_4h'] = currentFunding - (n > 4 ? fundingRates[n - 5] : 0);
    features['funding_momentum_24h'] = currentFunding - (n > 24 ? fundingRates[n - 25] : 0);

    // Cumulative funding
    features['cum_funding_long'] = current.cumFundingLong ?? 0;
    features['cum_funding_short'] = current.cumFundingShort ?? 0;
    features['cum_funding_diff'] = (current.cumFundingLong ?? 0) - (current.cumFundingShort ?? 0);

    // Funding sign features
    features['funding_sign'] = Math.sign(currentFunding);
    const prevSign = n > 1 ? Math.sign(fundingRates[n - 2]) : features['funding_sign'];
    features['funding_sign_change'] = features['funding_sign'] !== prevSign ? 1 : 0;
    features['funding_sign_changes_24h'] = this.countSignChanges(fundingRates.slice(-24));

    // Z-score (using 1 week window)
    const weekSlice = fundingRates.slice(-168);
    const weekMean = this.mean(weekSlice);
    const weekStd = this.std(weekSlice);
    features['funding_zscore'] = weekStd > 0 ? (currentFunding - weekMean) / weekStd : 0;

    // Price features (returns)
    for (const period of RETURN_PERIODS) {
      const currentPrice = prices[n - 1];
      const pastPrice = n > period ? prices[n - 1 - period] : currentPrice;
      features[`return_${period}h`] = pastPrice > 0 ? (currentPrice - pastPrice) / pastPrice : 0;
    }

    // Volatility
    const returns = this.computeReturns(prices);
    for (const window of VOLATILITY_WINDOWS) {
      const slice = returns.slice(-window);
      features[`volatility_${window}h`] = this.std(slice) * Math.sqrt(window);
    }

    // Price momentum
    for (const window of MOMENTUM_WINDOWS) {
      const currentPrice = prices[n - 1];
      const pastPrice = n > window ? prices[n - 1 - window] : currentPrice;
      features[`price_momentum_${window}h`] = pastPrice > 0 ? currentPrice / pastPrice - 1 : 0;
    }

    // Basis (mark - oracle spread)
    const oracle = current.oraclePrice ?? 0;
    const mark = current.markPrice ?? oracle;
    features['basis'] = oracle > 0 ? (mark - oracle) / oracle : 0;

    // Time features
    const dt = current.timestamp;
    const hour = dt.getUTCHours();
    const dow = dt.getUTCDay();
    features['hour'] = hour;
    features['day_of_week'] = dow;
    features['is_weekend'] = dow === 0 || dow === 6 ? 1 : 0;
    features['hour_sin'] = Math.sin(2 * Math.PI * hour / 24);
    features['hour_cos'] = Math.cos(2 * Math.PI * hour / 24);
    features['dow_sin'] = Math.sin(2 * Math.PI * dow / 7);
    features['dow_cos'] = Math.cos(2 * Math.PI * dow / 7);

    // Convert to ordered array
    return FEATURE_NAMES.map(name => features[name] ?? 0);
  }

  /** Get current funding rate */
  getCurrentFundingRate(): number {
    const n = this.buffer.data.length;
    return n > 0 ? this.buffer.data[n - 1].fundingRate : 0;
  }

  /** Get latest data point */
  getLatestDataPoint(): FundingDataPoint | null {
    const n = this.buffer.data.length;
    return n > 0 ? this.buffer.data[n - 1] : null;
  }

  // ============= HELPER METHODS =============

  private mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private std(arr: number[]): number {
    if (arr.length < 2) return 0;
    const avg = this.mean(arr);
    const squaredDiffs = arr.map(x => (x - avg) ** 2);
    return Math.sqrt(this.mean(squaredDiffs));
  }

  private skewness(arr: number[]): number {
    if (arr.length < 3) return 0;
    const n = arr.length;
    const avg = this.mean(arr);
    const stdDev = this.std(arr);
    if (stdDev === 0) return 0;

    const m3 = arr.reduce((acc, x) => acc + ((x - avg) / stdDev) ** 3, 0) / n;
    return m3;
  }

  private computeReturns(prices: number[]): number[] {
    if (prices.length < 2) return [];
    const returns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      const prev = prices[i - 1];
      returns.push(prev > 0 ? (prices[i] - prev) / prev : 0);
    }
    return returns;
  }

  private countSignChanges(arr: number[]): number {
    if (arr.length < 2) return 0;
    let changes = 0;
    for (let i = 1; i < arr.length; i++) {
      if (Math.sign(arr[i]) !== Math.sign(arr[i - 1])) {
        changes++;
      }
    }
    return changes;
  }
}

// Export factory function
export function createFeatureExtractor(maxSize?: number): PerpsFeatureExtractor {
  return new PerpsFeatureExtractor(maxSize);
}
