/**
 * Feature Engineering for LP Rebalancer
 * 
 * Transforms raw OHLCV data into 61 ML features.
 * TypeScript port of Python feature engineering logic.
 */
import type { OHLCVData } from '../providers/birdeye.js';
import type { PoolFeatures } from '../inference/model.js';

interface TokenPriceHistory {
  SOL: number[];
  USDC: number[];
  USDT: number[];
}

// Calculate moving average
function movingAverage(data: number[], window: number): number {
  if (data.length === 0) return 0;
  const slice = data.slice(-window);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}

// Calculate standard deviation
function standardDeviation(data: number[]): number {
  if (data.length < 2) return 0;
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const squaredDiffs = data.map((x) => Math.pow(x - mean, 2));
  return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / data.length);
}

// Calculate returns
function calculateReturn(current: number, previous: number): number {
  if (previous === 0) return 0;
  return (current - previous) / previous;
}

// Calculate trend (linear regression slope normalized)
function calculateTrend(data: number[]): number {
  if (data.length < 2) return 0;
  
  const n = data.length;
  const xMean = (n - 1) / 2;
  const yMean = data.reduce((a, b) => a + b, 0) / n;
  
  let numerator = 0;
  let denominator = 0;
  
  for (let i = 0; i < n; i++) {
    numerator += (i - xMean) * (data[i] - yMean);
    denominator += Math.pow(i - xMean, 2);
  }
  
  if (denominator === 0 || yMean === 0) return 0;
  
  const slope = numerator / denominator;
  return (slope / yMean) * 100;
}

// Calculate volatility (std dev of returns)
function calculateVolatility(prices: number[]): number {
  if (prices.length < 2) return 0;
  
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push(calculateReturn(prices[i], prices[i - 1]));
  }
  
  return standardDeviation(returns) * 100;
}

// Calculate impermanent loss
function calculateIL(priceAStart: number, priceAEnd: number, priceBStart: number, priceBEnd: number): number {
  if (priceAStart === 0 || priceBStart === 0 || priceBEnd === 0) return 0;
  
  const ratioA = priceAEnd / priceAStart;
  const ratioB = priceBEnd / priceBStart;
  const priceRatio = ratioA / ratioB;
  
  const il = 2 * Math.sqrt(priceRatio) / (1 + priceRatio) - 1;
  return Math.abs(il) * 100;
}

export function engineerFeatures(
  ohlcvData: OHLCVData[],
  tokenPrices: TokenPriceHistory,
  currentTime: Date
): PoolFeatures {
  // Sort by timestamp
  const sorted = [...ohlcvData].sort((a, b) => a.timestamp - b.timestamp);
  
  // Extract price/volume series
  const closes = sorted.map((d) => d.close);
  const highs = sorted.map((d) => d.high);
  const lows = sorted.map((d) => d.low);
  const volumes = sorted.map((d) => d.volume);
  
  const latest = sorted[sorted.length - 1];
  const current = latest ?? { close: 0, high: 0, low: 0, volume: 0 };

  // Time features
  const hour = currentTime.getUTCHours();
  const day = currentTime.getUTCDay();
  const isWeekend = day === 0 || day === 6 ? 1 : 0;

  // Helper to get price N hours ago
  const getPrice = (arr: number[], hoursAgo: number) => arr[Math.max(0, arr.length - 1 - hoursAgo)] ?? 0;

  // Build features object
  const features: PoolFeatures = {
    // Volume features
    volume_1h: current.volume,
    volume_ma_6h: movingAverage(volumes, 6),
    volume_ma_24h: movingAverage(volumes, 24),
    volume_ma_168h: movingAverage(volumes, 168),
    volume_trend_7d: calculateTrend(volumes.slice(-168)),
    volume_volatility_24h: calculateVolatility(volumes.slice(-24)),

    // Price features
    price_close: current.close,
    price_high: current.high,
    price_low: current.low,
    price_range: current.high - current.low,
    price_range_pct: current.low > 0 ? ((current.high - current.low) / current.low) * 100 : 0,
    price_ma_6h: movingAverage(closes, 6),
    price_ma_24h: movingAverage(closes, 24),
    price_ma_168h: movingAverage(closes, 168),
    price_trend_7d: calculateTrend(closes.slice(-168)),
    price_volatility_24h: calculateVolatility(closes.slice(-24)),
    price_volatility_168h: calculateVolatility(closes.slice(-168)),
    price_return_1h: calculateReturn(current.close, getPrice(closes, 1)),
    price_return_6h: calculateReturn(current.close, getPrice(closes, 6)),
    price_return_24h: calculateReturn(current.close, getPrice(closes, 24)),
    price_return_168h: calculateReturn(current.close, getPrice(closes, 168)),

    // TVL proxy (using volume as proxy since we don't have direct TVL)
    tvl_proxy: movingAverage(volumes, 24) * 100,
    tvl_ma_24h: movingAverage(volumes, 24) * 100,
    tvl_stability_7d: volumes.length >= 168 ? 1 - (standardDeviation(volumes.slice(-168)) / (movingAverage(volumes, 168) || 1)) : 0.5,
    tvl_trend_7d: calculateTrend(volumes.slice(-168)),
    vol_tvl_ratio: 0.01, // Placeholder
    vol_tvl_ma_24h: 0.01,

    // IL estimates
    il_estimate_24h: calculateIL(getPrice(closes, 24), current.close, 1, 1),
    il_estimate_7d: calculateIL(getPrice(closes, 168), current.close, 1, 1),
    il_change_24h: 0,

    // Time features
    hour_of_day: hour,
    day_of_week: day,
    is_weekend: isWeekend,
    hour_sin: Math.sin((2 * Math.PI * hour) / 24),
    hour_cos: Math.cos((2 * Math.PI * hour) / 24),
    day_sin: Math.sin((2 * Math.PI * day) / 7),
    day_cos: Math.cos((2 * Math.PI * day) / 7),

    // Token features - will be filled below
    SOL_price: 0, SOL_return_1h: 0, SOL_return_24h: 0, SOL_volatility_24h: 0, SOL_volatility_168h: 0, SOL_ma_6h: 0, SOL_ma_24h: 0, SOL_trend_7d: 0,
    USDC_price: 0, USDC_return_1h: 0, USDC_return_24h: 0, USDC_volatility_24h: 0, USDC_volatility_168h: 0, USDC_ma_6h: 0, USDC_ma_24h: 0, USDC_trend_7d: 0,
    USDT_price: 0, USDT_return_1h: 0, USDT_return_24h: 0, USDT_volatility_24h: 0, USDT_volatility_168h: 0, USDT_ma_6h: 0, USDT_ma_24h: 0, USDT_trend_7d: 0,
  };

  // Fill token features
  for (const token of ['SOL', 'USDC', 'USDT'] as const) {
    const prices = tokenPrices[token];
    if (prices.length > 0) {
      const latest = prices[prices.length - 1];
      features[`${token}_price`] = latest;
      features[`${token}_return_1h`] = calculateReturn(latest, getPrice(prices, 1));
      features[`${token}_return_24h`] = calculateReturn(latest, getPrice(prices, 24));
      features[`${token}_volatility_24h`] = calculateVolatility(prices.slice(-24));
      features[`${token}_volatility_168h`] = calculateVolatility(prices.slice(-168));
      features[`${token}_ma_6h`] = movingAverage(prices, 6);
      features[`${token}_ma_24h`] = movingAverage(prices, 24);
      features[`${token}_trend_7d`] = calculateTrend(prices.slice(-168));
    }
  }

  return features;
}

