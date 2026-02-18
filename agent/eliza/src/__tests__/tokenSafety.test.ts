import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { KitExecuteResult } from '../services/solanaAgentKit/types.js';

vi.mock('../services/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

const mockFetch = vi.fn();
global.fetch = mockFetch;

// Build a mock KitService that satisfies the interface used by tokenSafety
function makeMockKitService(overrides: {
  initialized?: boolean;
  hasRugCheck?: boolean;
  executeResult?: KitExecuteResult;
  executeThrows?: boolean;
} = {}) {
  const {
    initialized = true,
    hasRugCheck = true,
    executeResult = { success: true, data: { score: 20, risks: [] }, action: 'rugCheck', timestamp: Date.now() },
    executeThrows = false,
  } = overrides;

  return {
    isInitialized: () => initialized,
    hasAction: (name: string) => name === 'rugCheck' && hasRugCheck,
    execute: executeThrows
      ? vi.fn(async () => { throw new Error('Kit exploded'); })
      : vi.fn(async () => executeResult),
    // Unused methods to satisfy type
    initialize: vi.fn(),
    getActionNames: vi.fn(() => []),
    getRawAgent: vi.fn(() => null),
  };
}

const TEST_TOKEN = 'So11111111111111111111111111111111111111112';

// Dynamic import to reset module state between tests
async function importTokenSafety() {
  return import('../services/solanaAgentKit/tokenSafety.js');
}

describe('checkTokenSafety', () => {
  let checkTokenSafety: Awaited<ReturnType<typeof importTokenSafety>>['checkTokenSafety'];

  beforeEach(async () => {
    vi.clearAllMocks();
    mockFetch.mockReset();
    vi.resetModules();
    const mod = await importTokenSafety();
    checkTokenSafety = mod.checkTokenSafety;
  });

  describe('Kit rugCheck path', () => {
    it('should return LOW risk for safe token (score < 50)', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: true,
          data: { score: 15, risks: [] },
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);

      expect(result.safe).toBe(true);
      expect(result.riskScore).toBe(15);
      expect(result.riskLevel).toBe('LOW');
      expect(result.source).toBe('kit_rugcheck');
      expect(result.flags).toHaveLength(0);
      expect(result.warnings).toHaveLength(0);
      expect(kit.execute).toHaveBeenCalledWith('rugCheck', { mint: TEST_TOKEN });
    });

    it('should return HIGH risk for dangerous token (score >= 50)', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: true,
          data: { score: 65, risks: ['Mutable metadata', 'Low liquidity'] },
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);

      expect(result.safe).toBe(false);
      expect(result.riskScore).toBe(65);
      expect(result.riskLevel).toBe('HIGH');
      expect(result.source).toBe('kit_rugcheck');
      // score >= 50, so risks go to flags
      expect(result.flags).toContain('Mutable metadata');
      expect(result.flags).toContain('Low liquidity');
    });

    it('should return CRITICAL risk for very dangerous token (score >= 70)', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: true,
          data: { score: 85, risks: ['Honeypot detected'] },
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);

      expect(result.safe).toBe(false);
      expect(result.riskScore).toBe(85);
      expect(result.riskLevel).toBe('CRITICAL');
      expect(result.flags).toContain('Honeypot detected');
    });

    it('should clamp score to 0-100 range', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: true,
          data: { score: 150, risks: [] },
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);
      expect(result.riskScore).toBe(100);
    });

    it('should default score to 50 when missing', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: true,
          data: { risks: ['Unknown risk'] },
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);
      expect(result.riskScore).toBe(50);
      // score == 50, so risks go to flags
      expect(result.flags).toContain('Unknown risk');
    });

    it('should put risks in warnings when score < 50', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: true,
          data: { score: 30, risks: ['Minor issue'] },
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);
      expect(result.warnings).toContain('Minor issue');
      expect(result.flags).toHaveLength(0);
    });
  });

  describe('fallback to DexScreener', () => {
    it('should fall back when Kit is not initialized', async () => {
      const kit = makeMockKitService({ initialized: false });

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          pairs: [{
            liquidity: { usd: 500_000 },
            volume: { h24: 200_000 },
            fdv: 5_000_000,
            pairCreatedAt: Date.now() - 7 * 24 * 60 * 60 * 1000,
          }],
        }),
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);

      expect(result.source).toBe('dexscreener_fallback');
      expect(result.safe).toBe(true);
      expect(kit.execute).not.toHaveBeenCalled();
    });

    it('should fall back when Kit lacks rugCheck action', async () => {
      const kit = makeMockKitService({ hasRugCheck: false });

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          pairs: [{
            liquidity: { usd: 100_000 },
            volume: { h24: 50_000 },
            fdv: 1_000_000,
          }],
        }),
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);
      expect(result.source).toBe('dexscreener_fallback');
    });

    it('should fall back when Kit execute returns failure', async () => {
      const kit = makeMockKitService({
        executeResult: {
          success: false,
          error: 'API rate limited',
          action: 'rugCheck',
          timestamp: Date.now(),
        },
      });

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          pairs: [{
            liquidity: { usd: 100_000 },
            volume: { h24: 50_000 },
            fdv: 1_000_000,
          }],
        }),
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);
      expect(result.source).toBe('dexscreener_fallback');
    });

    it('should fall back when Kit execute throws', async () => {
      const kit = makeMockKitService({ executeThrows: true });

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          pairs: [{
            liquidity: { usd: 100_000 },
            volume: { h24: 50_000 },
            fdv: 1_000_000,
          }],
        }),
      });

      const result = await checkTokenSafety(TEST_TOKEN, kit as never);
      expect(result.source).toBe('dexscreener_fallback');
    });

    it('should fall back when kitService is null', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          pairs: [{
            liquidity: { usd: 100_000 },
            volume: { h24: 50_000 },
            fdv: 1_000_000,
          }],
        }),
      });

      const result = await checkTokenSafety(TEST_TOKEN, null);
      expect(result.source).toBe('dexscreener_fallback');
    });
  });

  describe('DexScreener evaluation', () => {
    function mockDexScreenerResponse(pairs: unknown[]) {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ pairs }),
      });
    }

    it('should flag token with no trading pairs', async () => {
      mockDexScreenerResponse([]);
      const result = await checkTokenSafety(TEST_TOKEN, null);

      expect(result.safe).toBe(false);
      expect(result.riskScore).toBe(80);
      expect(result.riskLevel).toBe('HIGH');
      expect(result.flags).toContain('No trading pairs found on DexScreener');
    });

    it('should flag low liquidity token', async () => {
      mockDexScreenerResponse([{
        liquidity: { usd: 5_000 },
        volume: { h24: 50_000 },
        fdv: 100_000,
      }]);

      const result = await checkTokenSafety(TEST_TOKEN, null);

      expect(result.flags.some(f => f.includes('Low liquidity'))).toBe(true);
      expect(result.riskScore).toBeGreaterThanOrEqual(30);
    });

    it('should flag low volume token', async () => {
      mockDexScreenerResponse([{
        liquidity: { usd: 500_000 },
        volume: { h24: 2_000 },
        fdv: 1_000_000,
      }]);

      const result = await checkTokenSafety(TEST_TOKEN, null);
      expect(result.flags.some(f => f.includes('Low 24h volume'))).toBe(true);
    });

    it('should flag extreme FDV/liquidity ratio', async () => {
      mockDexScreenerResponse([{
        liquidity: { usd: 50_000 },
        volume: { h24: 50_000 },
        fdv: 10_000_000, // 200x ratio
      }]);

      const result = await checkTokenSafety(TEST_TOKEN, null);
      expect(result.flags.some(f => f.includes('FDV/Liquidity ratio'))).toBe(true);
    });

    it('should flag very new token (< 24h)', async () => {
      mockDexScreenerResponse([{
        liquidity: { usd: 500_000 },
        volume: { h24: 200_000 },
        fdv: 5_000_000,
        pairCreatedAt: Date.now() - 6 * 60 * 60 * 1000, // 6 hours ago
      }]);

      const result = await checkTokenSafety(TEST_TOKEN, null);
      expect(result.flags.some(f => f.includes('Very new token'))).toBe(true);
    });

    it('should warn about moderately new token (24-72h)', async () => {
      mockDexScreenerResponse([{
        liquidity: { usd: 500_000 },
        volume: { h24: 200_000 },
        fdv: 5_000_000,
        pairCreatedAt: Date.now() - 48 * 60 * 60 * 1000, // 48 hours ago
      }]);

      const result = await checkTokenSafety(TEST_TOKEN, null);
      expect(result.warnings.some(w => w.includes('New token'))).toBe(true);
    });

    it('should rate safe token with good metrics as LOW risk', async () => {
      mockDexScreenerResponse([{
        liquidity: { usd: 1_000_000 },
        volume: { h24: 500_000 },
        fdv: 10_000_000,
        pairCreatedAt: Date.now() - 30 * 24 * 60 * 60 * 1000, // 30 days ago
      }]);

      const result = await checkTokenSafety(TEST_TOKEN, null);

      expect(result.safe).toBe(true);
      expect(result.riskLevel).toBe('LOW');
      expect(result.flags).toHaveLength(0);
    });

    it('should pick highest-liquidity pair from multiple', async () => {
      mockDexScreenerResponse([
        { liquidity: { usd: 5_000 }, volume: { h24: 1_000 }, fdv: 100_000 },
        { liquidity: { usd: 500_000 }, volume: { h24: 200_000 }, fdv: 5_000_000 },
        { liquidity: { usd: 50_000 }, volume: { h24: 10_000 }, fdv: 500_000 },
      ]);

      const result = await checkTokenSafety(TEST_TOKEN, null);
      // Should evaluate based on the 500K liquidity pair
      expect(result.data.liquidityUsd).toBe(500_000);
    });
  });

  describe('DexScreener error handling', () => {
    it('should return unavailable when API returns non-200', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 429 });
      const result = await checkTokenSafety(TEST_TOKEN, null);

      expect(result.source).toBe('unavailable');
      expect(result.safe).toBe(true); // doesn't block trades
      expect(result.warnings).toContain('Token safety data unavailable — proceed with caution');
    });

    it('should return unavailable when fetch throws (network error)', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));
      const result = await checkTokenSafety(TEST_TOKEN, null);

      expect(result.source).toBe('unavailable');
      expect(result.safe).toBe(true);
    });

    it('should return unavailable when response has no pairs key', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      const result = await checkTokenSafety(TEST_TOKEN, null);
      expect(result.safe).toBe(false); // no pairs = treated as risky
      expect(result.riskScore).toBe(80);
    });
  });
});
