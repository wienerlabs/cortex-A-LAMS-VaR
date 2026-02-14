/**
 * Feature Engineering Tests
 */
import { describe, it, expect } from 'vitest';
import { engineerFeatures } from '../features/engineer.js';
import type { OHLCVData } from '../providers/birdeye.js';

interface TokenPrices {
  SOL: number[];
  USDC: number[];
  USDT: number[];
}

describe('Feature Engineering', () => {
  const mockOHLCV: OHLCVData[] = Array.from({ length: 168 }, (_, i) => ({
    timestamp: Date.now() - (168 - i) * 3600000,
    open: 100 + Math.sin(i / 10) * 5,
    high: 105 + Math.sin(i / 10) * 5,
    low: 95 + Math.sin(i / 10) * 5,
    close: 100 + Math.sin(i / 10) * 5 + (Math.random() - 0.5) * 2,
    volume: 1000000 + Math.random() * 500000,
  }));

  const mockTokenPrices: TokenPrices = {
    SOL: Array.from({ length: 168 }, () => 200 + Math.random() * 10),
    USDC: Array.from({ length: 168 }, () => 1.0),
    USDT: Array.from({ length: 168 }, () => 1.0),
  };

  it('should generate 61 features', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    const featureKeys = Object.keys(features);
    expect(featureKeys.length).toBe(61);
  });

  it('should calculate volatility features', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    expect(features.price_volatility_24h).toBeTypeOf('number');
    expect(features.price_volatility_168h).toBeTypeOf('number');
    expect(features.price_volatility_24h).toBeGreaterThanOrEqual(0);
  });

  it('should calculate volume features', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    expect(features.volume_1h).toBeTypeOf('number');
    expect(features.volume_ma_24h).toBeTypeOf('number');
    expect(features.volume_ma_168h).toBeTypeOf('number');
  });

  it('should calculate price features', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    expect(features.price_return_1h).toBeTypeOf('number');
    expect(features.price_return_24h).toBeTypeOf('number');
    expect(features.price_close).toBeTypeOf('number');
  });

  it('should calculate time features', () => {
    const testDate = new Date('2024-03-15T14:30:00Z');
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, testDate);

    expect(features.hour_of_day).toBe(14);
    expect(features.day_of_week).toBe(5); // Friday
  });

  it('should handle weekend detection', () => {
    const saturday = new Date('2024-03-16T12:00:00Z');
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, saturday);

    expect(features.is_weekend).toBe(1);
  });

  it('should calculate SOL token features', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    expect(features.SOL_price).toBeTypeOf('number');
    expect(features.SOL_return_24h).toBeTypeOf('number');
    expect(features.SOL_volatility_24h).toBeTypeOf('number');
  });

  it('should calculate trend features', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    expect(features.price_trend_7d).toBeTypeOf('number');
    expect(features.volume_trend_7d).toBeTypeOf('number');
  });

  it('should handle insufficient data gracefully', () => {
    const shortOHLCV = mockOHLCV.slice(0, 10);
    const features = engineerFeatures(shortOHLCV, mockTokenPrices, new Date());

    // Should still return all 61 features
    expect(Object.keys(features).length).toBe(61);
    // Values should be finite
    Object.values(features).forEach(value => {
      expect(Number.isFinite(value)).toBe(true);
    });
  });

  it('should calculate IL estimates', () => {
    const features = engineerFeatures(mockOHLCV, mockTokenPrices, new Date());

    expect(features.il_estimate_24h).toBeTypeOf('number');
    expect(features.il_estimate_7d).toBeTypeOf('number');
    expect(features.il_estimate_24h).toBeGreaterThanOrEqual(0);
  });
});

