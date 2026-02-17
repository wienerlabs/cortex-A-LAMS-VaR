/**
 * Jito MEV Protection Service Tests
 */
import { describe, it, expect } from 'vitest';
import { 
  calculateDynamicTip, 
  checkJitoHealth, 
  getJitoTipStats,
  JITO_DEFAULT_CONFIG 
} from '../jitoService.js';

describe('JitoService', () => {
  describe('calculateDynamicTip', () => {
    it('should return base tip for low volatility and small trades', () => {
      const tip = calculateDynamicTip(0.02, 500);
      expect(tip).toBe(JITO_DEFAULT_CONFIG.tipLamports);
    });

    it('should increase tip for high volatility (>5%)', () => {
      const lowVolTip = calculateDynamicTip(0.03, 1000);
      const highVolTip = calculateDynamicTip(0.06, 1000);
      expect(highVolTip).toBeGreaterThan(lowVolTip);
    });

    it('should increase tip for very high volatility (>10%)', () => {
      const medVolTip = calculateDynamicTip(0.06, 1000);
      const veryHighVolTip = calculateDynamicTip(0.12, 1000);
      expect(veryHighVolTip).toBeGreaterThan(medVolTip);
    });

    it('should increase tip for large trades (>$5000)', () => {
      const smallTradeTip = calculateDynamicTip(0.03, 1000);
      const largeTradeTip = calculateDynamicTip(0.03, 6000);
      expect(largeTradeTip).toBeGreaterThan(smallTradeTip);
    });

    it('should cap tip at maxTipLamports', () => {
      // Very high volatility + very large trade
      const tip = calculateDynamicTip(0.15, 20000);
      expect(tip).toBeLessThanOrEqual(JITO_DEFAULT_CONFIG.maxTipLamports);
    });

    it('should use custom config values', () => {
      const customConfig = {
        ...JITO_DEFAULT_CONFIG,
        tipLamports: 5000,
        maxTipLamports: 20000,
      };
      const tip = calculateDynamicTip(0.02, 500, customConfig);
      expect(tip).toBe(5000);
    });
  });

  describe('checkJitoHealth', () => {
    it('should check mainnet health endpoint', async () => {
      const isHealthy = await checkJitoHealth('mainnet');
      // This may fail in test environment without network
      expect(typeof isHealthy).toBe('boolean');
    });

    it('should return false for invalid network', async () => {
      // @ts-expect-error - testing invalid input
      const isHealthy = await checkJitoHealth('invalid');
      expect(isHealthy).toBe(false);
    });
  });

  describe('getJitoTipStats', () => {
    it('should return tip statistics', async () => {
      const stats = await getJitoTipStats();
      expect(stats).not.toBeNull();
      if (stats) {
        expect(stats.minTip).toBeDefined();
        expect(stats.medianTip).toBeDefined();
        expect(stats.maxTip).toBeDefined();
        expect(stats.minTip).toBeLessThanOrEqual(stats.medianTip);
        expect(stats.medianTip).toBeLessThanOrEqual(stats.maxTip);
      }
    });
  });

  describe('JITO_DEFAULT_CONFIG', () => {
    it('should have reasonable default values', () => {
      expect(JITO_DEFAULT_CONFIG.tipLamports).toBe(10000); // 0.00001 SOL
      expect(JITO_DEFAULT_CONFIG.maxTipLamports).toBe(100000); // 0.0001 SOL
      expect(JITO_DEFAULT_CONFIG.useBundle).toBe(true);
      expect(JITO_DEFAULT_CONFIG.fallbackToRpc).toBe(true);
      expect(JITO_DEFAULT_CONFIG.network).toBe('mainnet');
    });
  });
});

