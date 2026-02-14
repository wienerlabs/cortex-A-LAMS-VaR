/**
 * Unit Tests for Adrena Production Client Helper Functions
 *
 * Tests parameter conversion logic ONLY - no devnet/integration tests.
 */
import { describe, it, expect } from 'vitest';
import {
  mapMarketToToken,
  mapTokenToMarket,
  calculatePositionSize,
  calculateRequiredCollateral,
  calculateStopLossPrice,
  calculateTakeProfitPrice,
  getSupportedMarkets,
  isMarketSupported,
  ADRENA_MARKET_MAP,
  PRICE_SYMBOL_MAP,
} from '../services/perps/adrenaHelpers.js';

// ============= TOKEN MAPPING TESTS =============

describe('Token Mapping', () => {
  describe('mapMarketToToken', () => {
    it('should map SOL-PERP to JITOSOL', () => {
      expect(mapMarketToToken('SOL-PERP')).toBe('JITOSOL');
    });

    it('should map JITOSOL-PERP to JITOSOL', () => {
      expect(mapMarketToToken('JITOSOL-PERP')).toBe('JITOSOL');
    });

    it('should map BTC-PERP to WBTC', () => {
      expect(mapMarketToToken('BTC-PERP')).toBe('WBTC');
    });

    it('should map BONK-PERP to BONK', () => {
      expect(mapMarketToToken('BONK-PERP')).toBe('BONK');
    });

    it('should return null for unsupported markets', () => {
      expect(mapMarketToToken('ETH-PERP')).toBeNull();
      expect(mapMarketToToken('DOGE-PERP')).toBeNull();
      expect(mapMarketToToken('')).toBeNull();
      expect(mapMarketToToken('invalid')).toBeNull();
    });
  });

  describe('mapTokenToMarket', () => {
    it('should map JITOSOL to SOL-PERP', () => {
      expect(mapTokenToMarket('JITOSOL')).toBe('SOL-PERP');
    });

    it('should map WBTC to BTC-PERP', () => {
      expect(mapTokenToMarket('WBTC')).toBe('BTC-PERP');
    });

    it('should map BONK to BONK-PERP', () => {
      expect(mapTokenToMarket('BONK')).toBe('BONK-PERP');
    });
  });

  describe('ADRENA_MARKET_MAP', () => {
    it('should have correct mappings', () => {
      expect(ADRENA_MARKET_MAP['SOL-PERP']).toBe('JITOSOL');
      expect(ADRENA_MARKET_MAP['BTC-PERP']).toBe('WBTC');
      expect(ADRENA_MARKET_MAP['BONK-PERP']).toBe('BONK');
    });
  });

  describe('PRICE_SYMBOL_MAP', () => {
    it('should have correct price symbol mappings', () => {
      expect(PRICE_SYMBOL_MAP['JITOSOL']).toBe('SOLUSD');
      expect(PRICE_SYMBOL_MAP['WBTC']).toBe('BTCUSD');
      expect(PRICE_SYMBOL_MAP['BONK']).toBe('BONKUSD');
    });
  });

  describe('getSupportedMarkets', () => {
    it('should return all supported markets', () => {
      const markets = getSupportedMarkets();
      expect(markets).toContain('SOL-PERP');
      expect(markets).toContain('BTC-PERP');
      expect(markets).toContain('BONK-PERP');
      expect(markets).toContain('JITOSOL-PERP');
      expect(markets.length).toBe(4);
    });
  });

  describe('isMarketSupported', () => {
    it('should return true for supported markets', () => {
      expect(isMarketSupported('SOL-PERP')).toBe(true);
      expect(isMarketSupported('BTC-PERP')).toBe(true);
      expect(isMarketSupported('BONK-PERP')).toBe(true);
    });

    it('should return false for unsupported markets', () => {
      expect(isMarketSupported('ETH-PERP')).toBe(false);
      expect(isMarketSupported('DOGE-PERP')).toBe(false);
    });
  });
});

// ============= COLLATERAL CALCULATION TESTS =============

describe('Collateral Calculation', () => {
  describe('calculatePositionSize', () => {
    it('should calculate position size correctly', () => {
      expect(calculatePositionSize(100, 1)).toBe(100);
      expect(calculatePositionSize(100, 2)).toBe(200);
      expect(calculatePositionSize(100, 5)).toBe(500);
      expect(calculatePositionSize(100, 10)).toBe(1000);
      expect(calculatePositionSize(100, 100)).toBe(10000);
    });

    it('should handle decimal collateral amounts', () => {
      expect(calculatePositionSize(50.5, 2)).toBe(101);
      expect(calculatePositionSize(0.1, 10)).toBeCloseTo(1);
    });

    it('should throw for negative collateral', () => {
      expect(() => calculatePositionSize(-100, 2)).toThrow('Collateral amount cannot be negative');
    });

    it('should throw for leverage below 1', () => {
      expect(() => calculatePositionSize(100, 0.5)).toThrow('Leverage must be at least 1');
      expect(() => calculatePositionSize(100, 0)).toThrow('Leverage must be at least 1');
    });

    it('should throw for leverage above 100', () => {
      expect(() => calculatePositionSize(100, 101)).toThrow('Leverage cannot exceed 100');
    });
  });

  describe('calculateRequiredCollateral', () => {
    it('should calculate required collateral correctly', () => {
      expect(calculateRequiredCollateral(1000, 10)).toBe(100);
      expect(calculateRequiredCollateral(500, 5)).toBe(100);
      expect(calculateRequiredCollateral(200, 2)).toBe(100);
    });

    it('should be inverse of calculatePositionSize', () => {
      const collateral = 100;
      const leverage = 5;
      const size = calculatePositionSize(collateral, leverage);
      expect(calculateRequiredCollateral(size, leverage)).toBe(collateral);
    });
  });
});

// ============= STOP LOSS PRICE TESTS =============

describe('Stop Loss Price Calculation', () => {
  describe('calculateStopLossPrice - Long positions', () => {
    it('should calculate SL below entry for long positions', () => {
      // Long: SL is BELOW entry (price drops = loss)
      expect(calculateStopLossPrice(100, 'long', 0.05)).toBe(95);  // 5% below
      expect(calculateStopLossPrice(100, 'long', 0.10)).toBe(90);  // 10% below
      expect(calculateStopLossPrice(100, 'long', 0.20)).toBe(80);  // 20% below
    });

    it('should handle different entry prices for longs', () => {
      expect(calculateStopLossPrice(200, 'long', 0.05)).toBe(190);
      expect(calculateStopLossPrice(50, 'long', 0.10)).toBe(45);
      expect(calculateStopLossPrice(42000, 'long', 0.05)).toBe(39900); // BTC-like price
    });

    it('should handle small percentages for longs', () => {
      expect(calculateStopLossPrice(100, 'long', 0.01)).toBe(99);  // 1% below
      expect(calculateStopLossPrice(100, 'long', 0.001)).toBe(99.9);  // 0.1% below
    });
  });

  describe('calculateStopLossPrice - Short positions', () => {
    it('should calculate SL above entry for short positions', () => {
      // Short: SL is ABOVE entry (price rises = loss)
      expect(calculateStopLossPrice(100, 'short', 0.05)).toBeCloseTo(105, 10);  // 5% above
      expect(calculateStopLossPrice(100, 'short', 0.10)).toBeCloseTo(110, 10);  // 10% above
      expect(calculateStopLossPrice(100, 'short', 0.20)).toBeCloseTo(120, 10);  // 20% above
    });

    it('should handle different entry prices for shorts', () => {
      expect(calculateStopLossPrice(200, 'short', 0.05)).toBeCloseTo(210, 10);
      expect(calculateStopLossPrice(50, 'short', 0.10)).toBeCloseTo(55, 10);
      expect(calculateStopLossPrice(42000, 'short', 0.05)).toBeCloseTo(44100, 10); // BTC-like price
    });
  });

  describe('calculateStopLossPrice - Edge cases', () => {
    it('should return entry price for 0% stop loss', () => {
      expect(calculateStopLossPrice(100, 'long', 0)).toBe(100);
      expect(calculateStopLossPrice(100, 'short', 0)).toBe(100);
    });

    it('should throw for negative entry price', () => {
      expect(() => calculateStopLossPrice(-100, 'long', 0.05)).toThrow('Entry price must be positive');
      expect(() => calculateStopLossPrice(0, 'short', 0.05)).toThrow('Entry price must be positive');
    });

    it('should throw for negative stop loss percent', () => {
      expect(() => calculateStopLossPrice(100, 'long', -0.05)).toThrow('Stop loss percent cannot be negative');
    });

    it('should throw for stop loss >= 100%', () => {
      expect(() => calculateStopLossPrice(100, 'long', 1)).toThrow('Stop loss percent must be less than 100%');
      expect(() => calculateStopLossPrice(100, 'long', 1.5)).toThrow('Stop loss percent must be less than 100%');
    });
  });
});

// ============= TAKE PROFIT PRICE TESTS =============

describe('Take Profit Price Calculation', () => {
  describe('calculateTakeProfitPrice - Long positions', () => {
    it('should calculate TP above entry for long positions', () => {
      // Long: TP is ABOVE entry (price rises = profit)
      expect(calculateTakeProfitPrice(100, 'long', 0.10)).toBeCloseTo(110, 10);  // 10% above
      expect(calculateTakeProfitPrice(100, 'long', 0.20)).toBeCloseTo(120, 10);  // 20% above
      expect(calculateTakeProfitPrice(100, 'long', 0.50)).toBeCloseTo(150, 10);  // 50% above
    });

    it('should handle different entry prices for longs', () => {
      expect(calculateTakeProfitPrice(200, 'long', 0.10)).toBeCloseTo(220, 10);
      expect(calculateTakeProfitPrice(50, 'long', 0.20)).toBeCloseTo(60, 10);
      expect(calculateTakeProfitPrice(42000, 'long', 0.10)).toBeCloseTo(46200, 10); // BTC-like price
    });

    it('should handle large take profit percentages for longs', () => {
      expect(calculateTakeProfitPrice(100, 'long', 1)).toBeCloseTo(200, 10);  // 100% = 2x
      expect(calculateTakeProfitPrice(100, 'long', 2)).toBeCloseTo(300, 10);  // 200% = 3x
    });
  });

  describe('calculateTakeProfitPrice - Short positions', () => {
    it('should calculate TP below entry for short positions', () => {
      // Short: TP is BELOW entry (price drops = profit)
      expect(calculateTakeProfitPrice(100, 'short', 0.10)).toBe(90);  // 10% below
      expect(calculateTakeProfitPrice(100, 'short', 0.20)).toBe(80);  // 20% below
      expect(calculateTakeProfitPrice(100, 'short', 0.50)).toBe(50);  // 50% below
    });

    it('should handle different entry prices for shorts', () => {
      expect(calculateTakeProfitPrice(200, 'short', 0.10)).toBe(180);
      expect(calculateTakeProfitPrice(50, 'short', 0.20)).toBe(40);
      expect(calculateTakeProfitPrice(42000, 'short', 0.10)).toBe(37800); // BTC-like price
    });
  });

  describe('calculateTakeProfitPrice - Edge cases', () => {
    it('should return entry price for 0% take profit', () => {
      expect(calculateTakeProfitPrice(100, 'long', 0)).toBe(100);
      expect(calculateTakeProfitPrice(100, 'short', 0)).toBe(100);
    });

    it('should throw for negative entry price', () => {
      expect(() => calculateTakeProfitPrice(-100, 'long', 0.10)).toThrow('Entry price must be positive');
      expect(() => calculateTakeProfitPrice(0, 'short', 0.10)).toThrow('Entry price must be positive');
    });

    it('should throw for negative take profit percent', () => {
      expect(() => calculateTakeProfitPrice(100, 'long', -0.10)).toThrow('Take profit percent cannot be negative');
    });
  });
});

// ============= INTEGRATED SCENARIO TESTS =============

describe('Integrated Scenarios', () => {
  it('should calculate correct SL/TP for a typical long trade', () => {
    const entryPrice = 150;  // SOL at $150
    const slPercent = 0.05;  // 5% stop loss
    const tpPercent = 0.10;  // 10% take profit

    const sl = calculateStopLossPrice(entryPrice, 'long', slPercent);
    const tp = calculateTakeProfitPrice(entryPrice, 'long', tpPercent);

    expect(sl).toBe(142.5);  // SL at $142.50 (5% below $150)
    expect(tp).toBe(165);    // TP at $165 (10% above $150)
    expect(sl).toBeLessThan(entryPrice);
    expect(tp).toBeGreaterThan(entryPrice);
  });

  it('should calculate correct SL/TP for a typical short trade', () => {
    const entryPrice = 42000;  // BTC at $42,000
    const slPercent = 0.03;    // 3% stop loss
    const tpPercent = 0.08;    // 8% take profit

    const sl = calculateStopLossPrice(entryPrice, 'short', slPercent);
    const tp = calculateTakeProfitPrice(entryPrice, 'short', tpPercent);

    expect(sl).toBe(43260);  // SL at $43,260 (3% above $42,000)
    expect(tp).toBe(38640);  // TP at $38,640 (8% below $42,000)
    expect(sl).toBeGreaterThan(entryPrice);
    expect(tp).toBeLessThan(entryPrice);
  });

  it('should calculate position size with collateral and leverage', () => {
    const collateral = 100;  // $100 USDC
    const leverage = 5;

    const positionSize = calculatePositionSize(collateral, leverage);
    expect(positionSize).toBe(500);  // $500 notional position

    // Verify inverse calculation
    const requiredCollateral = calculateRequiredCollateral(positionSize, leverage);
    expect(requiredCollateral).toBe(collateral);
  });
});
