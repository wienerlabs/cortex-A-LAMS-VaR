import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { SolanaAgentKitConfig } from '../services/solanaAgentKit/types.js';

const {
  fetchPriceHandler,
  tradeHandler,
  rugCheckHandler,
  throwingHandler,
} = vi.hoisted(() => ({
  fetchPriceHandler: vi.fn(async () => ({ price: 142.5 })),
  tradeHandler: vi.fn(async () => ({ txId: 'abc123' })),
  rugCheckHandler: vi.fn(async () => ({ score: 20, risks: [] })),
  throwingHandler: vi.fn(async () => { throw new Error('RPC timeout'); }),
}));

vi.mock('../services/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

vi.mock('@solana/web3.js', () => ({
  Keypair: {
    fromSecretKey: vi.fn(() => ({
      publicKey: { toBase58: () => 'ABC123testPublicKey' },
      secretKey: new Uint8Array(64),
    })),
  },
}));

vi.mock('bs58', () => ({
  default: { decode: vi.fn(() => new Uint8Array(64)) },
}));

vi.mock('solana-agent-kit', () => ({
  SolanaAgentKit: vi.fn(function () {
    return {
      actions: [
        { name: 'fetchPrice', handler: fetchPriceHandler },
        { name: 'trade', handler: tradeHandler },
        { name: 'rugCheck', handler: rugCheckHandler },
        { name: 'throwingAction', handler: throwingHandler },
      ],
      use: vi.fn(function (this: unknown) { return this; }),
    };
  }),
  KeypairWallet: vi.fn(function () { return {}; }),
}));

vi.mock('@solana-agent-kit/plugin-token', () => ({ default: { name: 'token-plugin' } }));
vi.mock('@solana-agent-kit/plugin-defi', () => ({ default: { name: 'defi-plugin' } }));
vi.mock('@solana-agent-kit/plugin-misc', () => ({ default: { name: 'misc-plugin' } }));

const TEST_CONFIG: SolanaAgentKitConfig = {
  rpcUrl: 'https://api.mainnet-beta.solana.com',
  privateKey: 'FakeBase58PrivateKeyForTestingPurposesOnly1234567890ABCDEF',
  openaiApiKey: 'sk-test',
  heliusApiKey: 'helius-test',
  coinGeckoApiKey: 'cg-test',
};

import { SolanaAgentKitService } from '../services/solanaAgentKit/index.js';

describe('SolanaAgentKitService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('should start uninitialized', () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      expect(service.isInitialized()).toBe(false);
      expect(service.getActionNames()).toEqual([]);
      expect(service.getRawAgent()).toBeNull();
    });
  });

  describe('initialization', () => {
    it('should initialize successfully with valid config', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      const result = await service.initialize();

      expect(result).toBe(true);
      expect(service.isInitialized()).toBe(true);
      expect(service.getRawAgent()).not.toBeNull();
    });

    it('should return true on repeated initialization (idempotent)', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();
      const secondCall = await service.initialize();
      expect(secondCall).toBe(true);
    });

    it('should return false when private key is missing', async () => {
      const service = new SolanaAgentKitService({ rpcUrl: 'https://rpc.test' });
      const result = await service.initialize();

      expect(result).toBe(false);
      expect(service.isInitialized()).toBe(false);
    });

    it('should cache action handlers from plugins', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      const names = service.getActionNames();
      expect(names).toContain('fetchPrice');
      expect(names).toContain('trade');
      expect(names).toContain('rugCheck');
      expect(names).toContain('throwingAction');
      expect(names).toHaveLength(4);
    });
  });

  describe('hasAction', () => {
    it('should return true for registered actions', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      expect(service.hasAction('fetchPrice')).toBe(true);
      expect(service.hasAction('trade')).toBe(true);
      expect(service.hasAction('rugCheck')).toBe(true);
    });

    it('should return false for unregistered actions', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      expect(service.hasAction('nonExistentAction')).toBe(false);
      expect(service.hasAction('stakeSOL')).toBe(false);
    });

    it('should return false when not initialized', () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      expect(service.hasAction('fetchPrice')).toBe(false);
    });
  });

  describe('execute', () => {
    it('should execute a registered action and return success', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      const result = await service.execute('fetchPrice', { mint: 'SOL' });

      expect(result.success).toBe(true);
      expect(result.data).toEqual({ price: 142.5 });
      expect(result.action).toBe('fetchPrice');
      expect(result.timestamp).toBeGreaterThan(0);
      expect(fetchPriceHandler).toHaveBeenCalled();
    });

    it('should return error when not initialized', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      const result = await service.execute('fetchPrice');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Kit not initialized');
      expect(result.action).toBe('fetchPrice');
    });

    it('should return error for unknown action', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      const result = await service.execute('nonExistent');

      expect(result.success).toBe(false);
      expect(result.error).toContain("'nonExistent' not found");
      expect(result.action).toBe('nonExistent');
    });

    it('should catch handler errors and return failure result', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      const result = await service.execute('throwingAction', {});

      expect(result.success).toBe(false);
      expect(result.error).toBe('RPC timeout');
      expect(result.action).toBe('throwingAction');
    });

    it('should pass params to the handler', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      await service.execute('fetchPrice', { mint: 'SOL', outputMint: 'USDC' });

      expect(fetchPriceHandler).toHaveBeenCalledWith(
        expect.anything(),
        { mint: 'SOL', outputMint: 'USDC' },
      );
    });

    it('should default params to empty object', async () => {
      const service = new SolanaAgentKitService(TEST_CONFIG);
      await service.initialize();

      await service.execute('fetchPrice');

      expect(fetchPriceHandler).toHaveBeenCalledWith(expect.anything(), {});
    });
  });

  describe('getSolanaAgentKitService (singleton factory)', () => {
    it('should return the same instance on repeated calls', async () => {
      vi.resetModules();
      const mod = await import('../services/solanaAgentKit/index.js');

      const a = mod.getSolanaAgentKitService(TEST_CONFIG);
      const b = mod.getSolanaAgentKitService();
      expect(a).toBe(b);
    });

    it('should use explicit config over env vars', async () => {
      vi.resetModules();
      const mod = await import('../services/solanaAgentKit/index.js');

      const service = mod.getSolanaAgentKitService({
        rpcUrl: 'https://custom-rpc.test',
        privateKey: 'custom-key',
      });
      expect(service).toBeInstanceOf(mod.SolanaAgentKitService);
    });

    it('should fall back to env vars when no config provided', async () => {
      vi.resetModules();
      const mod = await import('../services/solanaAgentKit/index.js');

      process.env.SOLANA_RPC_URL = 'https://env-rpc.test';
      process.env.SOLANA_PRIVATE_KEY = 'env-private-key';
      process.env.OPENAI_API_KEY = 'env-openai';

      const service = mod.getSolanaAgentKitService();
      expect(service).toBeInstanceOf(mod.SolanaAgentKitService);

      delete process.env.SOLANA_RPC_URL;
      delete process.env.SOLANA_PRIVATE_KEY;
      delete process.env.OPENAI_API_KEY;
    });
  });

  // Isolated: this test uses vi.doMock which poisons the module cache
  describe('initialization failure', () => {
    it('should return false when dynamic import fails', async () => {
      vi.doMock('solana-agent-kit', () => { throw new Error('Module not found'); });
      vi.resetModules();

      const mod = await import('../services/solanaAgentKit/index.js');
      const service = new mod.SolanaAgentKitService(TEST_CONFIG);
      const result = await service.initialize();

      expect(result).toBe(false);
      expect(service.isInitialized()).toBe(false);
    });
  });
});
