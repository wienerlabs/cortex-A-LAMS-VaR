/**
 * MomentumAnalyst Tests
 * 
 * Tests for funding rate arbitrage analyst
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MomentumAnalyst, DEFAULT_MOMENTUM_CONFIG } from '../MomentumAnalyst.js';
import type { FundingArbitrageOpportunity } from '../../../services/marketScanner/types.js';
import type { MomentumAnalysisInput } from '../MomentumAnalyst.js';

// Mock sentiment integration
vi.mock('../../../services/sentiment/sentimentIntegration.js', () => ({
  getSentimentIntegration: () => ({
    getAdjustedScore: async (_token: string, mlConfidence: number, _strategy: string) => ({
      mlConfidence,
      rawSentiment: 0.5,
      normalizedSentiment: 0.75,
      finalScore: mlConfidence, // Return ML confidence as final score for testing
      sentimentWeight: 0.05,
      signal: 'neutral' as const,
      sentimentAvailable: true,
      reasoning: 'Test sentiment',
    }),
  }),
}));

// Helper to create mock funding arbitrage opportunity
function createMockFundingArb(overrides: Partial<FundingArbitrageOpportunity> = {}): FundingArbitrageOpportunity {
  return {
    market: 'SOL-PERP',
    longVenue: 'drift',
    shortVenue: 'jupiter',
    longRate: 0.0001,
    shortRate: -0.0001,
    netSpread: 0.0002,
    annualizedSpread: 0.073,
    estimatedProfitBps: 20,
    confidence: 'high',
    ...overrides,
  };
}

// Helper to create analysis input
function createInput(opportunities: FundingArbitrageOpportunity[], volatility = 0.05): MomentumAnalysisInput {
  return {
    opportunities,
    volatility24h: volatility,
    portfolioValueUsd: 10000,
  };
}

describe('MomentumAnalyst', () => {
  let analyst: MomentumAnalyst;

  beforeEach(() => {
    analyst = new MomentumAnalyst();
  });

  // ============= BASIC FUNCTIONALITY =============
  describe('Basic Functionality', () => {
    it('should instantiate with default config', () => {
      expect(analyst).toBeDefined();
      expect(analyst.getName()).toBe('MomentumAnalyst');
    });

    it('should instantiate with custom config', () => {
      const customAnalyst = new MomentumAnalyst({
        minSpreadBps: 10,
        minConfidenceThreshold: 0.85,
      });
      expect(customAnalyst).toBeDefined();
      expect(customAnalyst.getMomentumConfig().minSpreadBps).toBe(10);
      expect(customAnalyst.getMomentumConfig().minConfidenceThreshold).toBe(0.85);
    });

    it('should return agent name', () => {
      expect(analyst.getName()).toBe('MomentumAnalyst');
    });

    it('should analyze funding arbitrage opportunities', async () => {
      const input = createInput([createMockFundingArb()]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(1);
      expect(results[0].type).toBe('funding_arb');
      expect(results[0].name).toContain('SOL-PERP');
      expect(results[0].name).toContain('drift↔jupiter');
    });

    it('should process multiple opportunities', async () => {
      const input = createInput([
        createMockFundingArb({ market: 'SOL-PERP', estimatedProfitBps: 20 }),
        createMockFundingArb({ market: 'ETH-PERP', estimatedProfitBps: 15 }),
        createMockFundingArb({ market: 'BTC-PERP', estimatedProfitBps: 10 }),
      ]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(3);
      expect(results[0].raw.market).toBe('SOL-PERP');
      expect(results[1].raw.market).toBe('ETH-PERP');
      expect(results[2].raw.market).toBe('BTC-PERP');
    });
  });

  // ============= SPREAD CALCULATION =============
  describe('Spread Calculation', () => {
    it('should calculate annualized return correctly', async () => {
      const input = createInput([createMockFundingArb({ annualizedSpread: 0.073 })]);
      const results = await analyst.analyze(input);

      expect(results[0].expectedReturn).toBeCloseTo(7.3, 1); // 7.3% annualized
    });

    it('should reject spreads below minimum threshold (5bps)', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 3 })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
      expect(results[0].rejectReason).toContain('3.0bps');
    });

    it('should approve spreads above minimum threshold', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 20 })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(true);
      expect(results[0].rejectReason).toBeUndefined();
    });

    it('should use custom spread threshold', async () => {
      const customAnalyst = new MomentumAnalyst({ minSpreadBps: 15 });
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 10 })]);
      const results = await customAnalyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
      expect(results[0].rejectReason).toContain('min 15bps');
    });
  });

  // ============= LONG/SHORT DETECTION =============
  describe('Long/Short Detection', () => {
    it('should identify long and short venues', async () => {
      const input = createInput([createMockFundingArb({
        longVenue: 'drift',
        shortVenue: 'jupiter',
      })]);
      const results = await analyst.analyze(input);

      expect(results[0].name).toContain('drift↔jupiter');
      expect(results[0].raw.longVenue).toBe('drift');
      expect(results[0].raw.shortVenue).toBe('jupiter');
    });

    it('should handle different venue combinations', async () => {
      const input = createInput([
        createMockFundingArb({ longVenue: 'drift', shortVenue: 'flash' }),
        createMockFundingArb({ longVenue: 'jupiter', shortVenue: 'adrena' }),
      ]);
      const results = await analyst.analyze(input);

      expect(results[0].name).toContain('drift↔flash');
      expect(results[1].name).toContain('jupiter↔adrena');
    });

    it('should calculate net spread from long and short rates', async () => {
      const input = createInput([createMockFundingArb({
        longRate: 0.0002,
        shortRate: -0.0001,
        netSpread: 0.0003,
      })]);
      const results = await analyst.analyze(input);

      expect(results[0].raw.netSpread).toBe(0.0003);
    });
  });

  // ============= CONFIDENCE SCORING =============
  describe('Confidence Scoring', () => {
    it('should assign high confidence (0.9) for spreads >= 20bps', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 20 })]);
      const results = await analyst.analyze(input);

      expect(results[0].confidence).toBeGreaterThanOrEqual(0.9);
    });

    it('should assign medium confidence (0.8) for spreads >= 10bps', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 15 })]);
      const results = await analyst.analyze(input);

      expect(results[0].confidence).toBeGreaterThanOrEqual(0.8);
    });

    it('should assign base confidence (0.6) for spreads < 10bps', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 8 })]);
      const results = await analyst.analyze(input);

      expect(results[0].confidence).toBeCloseTo(0.6, 1);
    });

    it('should filter below minimum confidence threshold (80%)', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 8 })]);
      const results = await analyst.analyze(input);

      // Base confidence is 0.6, below 0.8 threshold
      expect(results[0].approved).toBe(false);
    });

    it('should use custom confidence threshold', async () => {
      const customAnalyst = new MomentumAnalyst({ minConfidenceThreshold: 0.7 });
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 15 })]);
      const results = await customAnalyst.analyze(input);

      // Confidence is 0.8, above 0.7 threshold
      expect(results[0].approved).toBe(true);
    });
  });

  // ============= RISK SCORING =============
  describe('Risk Scoring', () => {
    it('should assign low risk score (3) for high spreads >= 20bps', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 20 })]);
      const results = await analyst.analyze(input);

      expect(results[0].riskScore).toBe(3);
    });

    it('should assign medium risk score (4) for spreads < 20bps', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 15 })]);
      const results = await analyst.analyze(input);

      expect(results[0].riskScore).toBe(4);
    });

    it('should calculate risk-adjusted return', async () => {
      const input = createInput([createMockFundingArb({
        annualizedSpread: 0.073, // 7.3% annual
        estimatedProfitBps: 20,  // Risk score = 3
      })]);
      const results = await analyst.analyze(input);

      // Daily return = 7.3 / 365 = 0.02%
      // Risk-adjusted = 0.02 * (1 - 3/20) = 0.02 * 0.85 = 0.017%
      expect(results[0].riskAdjustedReturn).toBeGreaterThan(0);
      expect(results[0].riskAdjustedReturn).toBeLessThan(results[0].expectedReturn / 365);
    });
  });

  // ============= DELTA-NEUTRAL STRATEGY =============
  describe('Delta-Neutral Strategy', () => {
    it('should use 10% position size for delta-neutral', async () => {
      const config = analyst.getMomentumConfig();
      expect(config.deltaNeutralPositionPct).toBe(10);
    });

    it('should have lower risk than directional strategies', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 20 })]);
      const results = await analyst.analyze(input);

      // Delta-neutral should have risk score <= 4
      expect(results[0].riskScore).toBeLessThanOrEqual(4);
    });
  });

  // ============= EMPTY RESULTS =============
  describe('Empty Results', () => {
    it('should return empty array for no opportunities', async () => {
      const input = createInput([]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(0);
    });

    it('should return results even if all rejected', async () => {
      const input = createInput([
        createMockFundingArb({ estimatedProfitBps: 2 }), // Below threshold
        createMockFundingArb({ estimatedProfitBps: 3 }), // Below threshold
      ]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(2);
      expect(results[0].approved).toBe(false);
      expect(results[1].approved).toBe(false);
    });
  });

  // ============= CONFIGURATION UPDATES =============
  describe('Configuration Updates', () => {
    it('should update config via updateMomentumConfig', () => {
      analyst.updateMomentumConfig({ minSpreadBps: 15 });
      expect(analyst.getMomentumConfig().minSpreadBps).toBe(15);
    });

    it('should update config via updateConfig', () => {
      analyst.updateConfig({ minConfidence: 0.85 });
      expect(analyst.getConfig().minConfidence).toBe(0.85);
    });

    it('should use volatility from input over config', async () => {
      const input = createInput([createMockFundingArb()], 0.15);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(1);
      // High volatility might affect risk check, but should still process
    });
  });

  // ============= EDGE CASES =============
  describe('Edge Cases', () => {
    it('should handle zero spread', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: 0 })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
    });

    it('should handle negative spread', async () => {
      const input = createInput([createMockFundingArb({ estimatedProfitBps: -5 })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
    });

    it('should extract token from market name', async () => {
      const input = createInput([createMockFundingArb({ market: 'ETH-PERP' })]);
      const results = await analyst.analyze(input);

      expect(results[0].name).toContain('ETH-PERP');
    });

    it('should handle market without hyphen', async () => {
      const input = createInput([createMockFundingArb({ market: 'SOLPERP' })]);
      const results = await analyst.analyze(input);

      expect(results[0].name).toContain('SOLPERP');
    });
  });
});


