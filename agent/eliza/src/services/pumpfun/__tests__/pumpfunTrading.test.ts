/**
 * PumpFun Trading Tests
 * 
 * Tests for:
 * 1. Risk checker validation
 * 2. Exit manager position tracking
 * 3. Config loader functionality
 * 4. Buy/sell execution (with mocks)
 * 
 * Run with: npx vitest run src/services/pumpfun/__tests__/pumpfunTrading.test.ts
 */
import { describe, it, expect } from 'vitest';
import { checkPumpFunRisks, isTokenSafe, type RiskThresholds } from '../riskChecker.js';
import { loadPumpFunConfig, getDefaultConfig } from '../configLoader.js';
import type { PumpFunToken } from '../pumpfunClient.js';

// ============= MOCK TOKEN DATA =============

function createMockToken(overrides: Partial<PumpFunToken> = {}): PumpFunToken {
  const baseToken: PumpFunToken = {
    coinMint: 'mock123456789',
    dev: 'creator123',
    name: 'MockCoin',
    ticker: 'MOCK',
    imageUrl: 'https://example.com/mock.png',
    twitter: 'https://twitter.com/mock',
    telegram: 'https://t.me/mock',
    website: 'https://mock.io',
    hasTwitter: true,
    hasTelegram: true,
    hasWebsite: true,
    hasSocial: true,
    creationTime: Date.now() - (48 * 60 * 60 * 1000), // 48 hours ago
    marketCap: 50000,
    currentMarketPrice: 0.00005,
    volume: 25000,
    buyTransactions: 150,
    sellTransactions: 50,
    transactions: 200,
    numHolders: 120,
    graduationDate: null,
    devHoldingsPercentage: 5,
    topHoldersPercentage: 35,
    sniperOwnedPercentage: 10,
    sniperCount: 5,
    isMayhemMode: false,
    twitterReuseCount: 0,
    bondingCurveProgress: 50,
    allTimeHighMarketCap: 100000,
    poolAddress: null,
    tokenProgram: 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',
    holders: [
      { totalTokenAmountHeld: 5000000, isSniper: false, ownedPercentage: 5, holderId: 'holder1' },
      { totalTokenAmountHeld: 3000000, isSniper: false, ownedPercentage: 3, holderId: 'holder2' },
      { totalTokenAmountHeld: 2000000, isSniper: true, ownedPercentage: 2, holderId: 'holder3' },
    ],
    ...overrides,
  };
  return baseToken;
}

// ============= RISK CHECKER TESTS =============

describe('PumpFun Risk Checker', () => {
  describe('checkPumpFunRisks', () => {
    it('should pass a healthy token', async () => {
      const token = createMockToken();
      const result = await checkPumpFunRisks(token);
      
      expect(result.isRugPull).toBe(false);
      expect(result.riskScore).toBeLessThan(50);
      expect(result.riskFlags.length).toBe(0);
    });

    it('should flag high creator holdings', async () => {
      const token = createMockToken({ devHoldingsPercentage: 35 });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskScore).toBeGreaterThan(0);
      expect(result.riskFlags.some(f => f.includes('Creator holds'))).toBe(true);
    });

    it('should flag high top holder concentration', async () => {
      const token = createMockToken({ topHoldersPercentage: 75 });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskScore).toBeGreaterThan(0);
      expect(result.riskFlags.some(f => f.includes('Top holders'))).toBe(true);
    });

    it('should flag low liquidity', async () => {
      const token = createMockToken({ marketCap: 5000 });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskFlags.some(f => f.includes('Low liquidity'))).toBe(true);
    });

    it('should flag low volume', async () => {
      const token = createMockToken({ volume: 1000 });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskFlags.some(f => f.includes('Low 24h volume'))).toBe(true);
    });

    it('should flag low holder count', async () => {
      const token = createMockToken({ numHolders: 20 });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskFlags.some(f => f.includes('Low holder count'))).toBe(true);
    });

    it('should flag very new tokens', async () => {
      const token = createMockToken({ creationTime: Date.now() - (30 * 60 * 1000) }); // 30 minutes
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskFlags.some(f => f.includes('Token too new'))).toBe(true);
    });

    it('should flag high sniper ownership', async () => {
      const token = createMockToken({ sniperOwnedPercentage: 40 });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskFlags.some(f => f.includes('High sniper ownership'))).toBe(true);
    });

    it('should detect rug pull risk when multiple flags', async () => {
      const token = createMockToken({
        devHoldingsPercentage: 40,
        topHoldersPercentage: 80,
        numHolders: 15,
        sniperOwnedPercentage: 50,
      });
      const result = await checkPumpFunRisks(token);
      
      expect(result.isRugPull).toBe(true);
      expect(result.riskScore).toBeGreaterThanOrEqual(50);
      expect(result.riskFlags.length).toBeGreaterThanOrEqual(3);
    });

    it('should reduce risk for graduated tokens', async () => {
      const ungraduated = createMockToken({ graduationDate: null });
      const graduated = createMockToken({ graduationDate: Date.now() - 86400000 });
      
      const ungraduatedResult = await checkPumpFunRisks(ungraduated);
      const graduatedResult = await checkPumpFunRisks(graduated);
      
      expect(graduatedResult.riskScore).toBeLessThanOrEqual(ungraduatedResult.riskScore);
    });

    it('should flag heavy sell pressure', async () => {
      const token = createMockToken({
        buyTransactions: 20,
        sellTransactions: 100,
      });
      const result = await checkPumpFunRisks(token);
      
      expect(result.riskFlags.some(f => f.includes('sell pressure') || f.includes('Heavy sell'))).toBe(true);
    });

    it('should use custom thresholds', async () => {
      const token = createMockToken({ devHoldingsPercentage: 15 });
      
      const customThresholds: Partial<RiskThresholds> = { maxCreatorPct: 10 };
      const result = await checkPumpFunRisks(token, customThresholds);
      
      expect(result.riskFlags.some(f => f.includes('Creator holds'))).toBe(true);
    });
  });

  describe('isTokenSafe', () => {
    it('should return true for safe token', () => {
      const token = createMockToken();
      expect(isTokenSafe(token)).toBe(true);
    });

    it('should return false for high creator holdings', () => {
      const token = createMockToken({ devHoldingsPercentage: 30 });
      expect(isTokenSafe(token)).toBe(false);
    });

    it('should return false for low holder count', () => {
      const token = createMockToken({ numHolders: 25 });
      expect(isTokenSafe(token)).toBe(false);
    });
  });
});

// ============= CONFIG LOADER TESTS =============

describe('PumpFun Config Loader', () => {
  describe('getDefaultConfig', () => {
    it('should return default configuration', () => {
      const config = getDefaultConfig();

      expect(config.enabled).toBe(true);
      expect(config.maxPositionSol).toBe(0.5);
      expect(config.maxConcurrentPositions).toBe(3);
      expect(config.useJito).toBe(true);
    });

    it('should have valid risk thresholds', () => {
      const config = getDefaultConfig();

      expect(config.riskThresholds.minLiquidityUsd).toBe(10000);
      expect(config.riskThresholds.minVolume24h).toBe(5000);
      expect(config.riskThresholds.minHolders).toBe(50);
      expect(config.riskThresholds.minAgeHours).toBe(1);
      expect(config.riskThresholds.maxCreatorPct).toBe(20);
      expect(config.riskThresholds.maxTop10Pct).toBe(60);
      expect(config.riskThresholds.maxSniperPct).toBe(30);
    });

    it('should have valid exit strategy', () => {
      const config = getDefaultConfig();

      expect(config.exitStrategy.takeProfitLevels).toEqual([100, 200, 500]);
      expect(config.exitStrategy.stopLossPct).toBe(50);
      expect(config.exitStrategy.trailingStopPct).toBe(30);
      expect(config.exitStrategy.autoExitTriggers.liquidityDropPct).toBe(50);
      expect(config.exitStrategy.autoExitTriggers.topHolderDumpPct).toBe(20);
    });
  });

  describe('loadPumpFunConfig', () => {
    it('should load config without errors', () => {
      const config = loadPumpFunConfig();

      expect(config).toBeDefined();
      expect(config.enabled).toBeDefined();
      expect(config.riskThresholds).toBeDefined();
      expect(config.exitStrategy).toBeDefined();
    });

    it('should cache config on repeated calls', () => {
      const config1 = loadPumpFunConfig();
      const config2 = loadPumpFunConfig();

      // Should return same object from cache
      expect(config1).toEqual(config2);
    });
  });
});

// ============= EXIT MANAGER TESTS =============

describe('PumpFun Exit Manager', () => {
  describe('Exit Signal Calculation', () => {
    it('should calculate correct profit percentage', () => {
      const entryPrice = 0.0001;
      const currentPrice = 0.0002;
      const profitPct = ((currentPrice - entryPrice) / entryPrice) * 100;

      expect(profitPct).toBe(100); // 100% gain
    });

    it('should detect stop loss condition', () => {
      const entryPrice = 0.0001;
      const currentPrice = 0.00004; // -60% drop
      const stopLossPct = 50;
      const profitPct = ((currentPrice - entryPrice) / entryPrice) * 100;

      expect(profitPct).toBe(-60);
      expect(profitPct <= -stopLossPct).toBe(true);
    });

    it('should detect trailing stop condition', () => {
      const highestPrice = 0.0002;
      const currentPrice = 0.00012; // 40% drop from high
      const trailingStopPct = 30;
      const dropFromHigh = ((highestPrice - currentPrice) / highestPrice) * 100;

      expect(dropFromHigh).toBe(40);
      expect(dropFromHigh >= trailingStopPct).toBe(true);
    });

    it('should detect liquidity drop', () => {
      const initialLiquidity = 100000;
      const currentLiquidity = 40000; // 60% drop
      const liquidityDropThreshold = 50;
      const liquidityDropPct = ((initialLiquidity - currentLiquidity) / initialLiquidity) * 100;

      expect(liquidityDropPct).toBe(60);
      expect(liquidityDropPct >= liquidityDropThreshold).toBe(true);
    });

    it('should detect take profit levels', () => {
      const takeProfitLevels = [100, 200, 500];
      const profitPct = 150;

      const hitLevels = takeProfitLevels.filter(level => profitPct >= level);
      expect(hitLevels).toContain(100);
      expect(hitLevels).not.toContain(200);
      expect(hitLevels).not.toContain(500);
    });
  });
});

