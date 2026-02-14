/**
 * LendingExecutor Tests
 *
 * Tests for the multi-protocol lending executor.
 * Note: These tests focus on the executor logic, not individual clients.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Keypair, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';

// Mock SDKs to avoid avsc and other dependencies
vi.mock('@mrgnlabs/marginfi-client-v2', () => ({
  MarginfiClient: { fetch: vi.fn() },
  getConfig: vi.fn(() => ({
    environment: 'production',
    cluster: 'mainnet-beta',
    programId: new PublicKey('MFv2hWf31Z9kbCa1snEPYctwafyhdvnV7FZnsebVacA'),
    groupPk: new PublicKey('4qp6Fx6tnZkY5Wropq9wUYgtFxXKwE6viZxFHg3rdAG8'),
  })),
}));

vi.mock('@mrgnlabs/mrgn-common', () => ({
  Wallet: class MockWallet {},
}));

vi.mock('@kamino-finance/klend-sdk', () => ({
  KaminoMarket: { load: vi.fn() },
  KaminoAction: { buildDepositTxns: vi.fn(), actionToIxs: vi.fn(() => []) },
  VanillaObligation: class MockVanillaObligation {},
  PROGRAM_ID: 'KLend2g3cP87ber41YYiuKt9NpkqEb4vGnHKxKBgvg',
}));

vi.mock('@solendprotocol/solend-sdk', () => ({
  SolendActionCore: { buildDepositTxns: vi.fn() },
  getReservesOfPool: vi.fn(() => []),
  fetchPoolByAddress: vi.fn(() => ({
    info: { lendingMarket: new PublicKey('11111111111111111111111111111111') },
  })),
}));

import { LendingExecutor, resetLendingExecutor, getLendingExecutor } from '../index.js';
import type { LendingConfig } from '../types.js';

// Generate a valid test keypair
const testKeypair = Keypair.generate();
const testPrivateKey = bs58.encode(testKeypair.secretKey);

// Config with all protocols disabled to avoid network calls
const mockConfig: LendingConfig = {
  rpcUrl: 'https://api.mainnet-beta.solana.com',
  privateKey: testPrivateKey,
  enableMarginFi: false,
  enableKamino: false,
  enableSolend: false,
};

describe('LendingExecutor', () => {
  let executor: LendingExecutor;

  beforeEach(() => {
    resetLendingExecutor();
    executor = new LendingExecutor(mockConfig);
  });

  describe('initialization', () => {
    it('should create executor instance', () => {
      expect(executor).toBeDefined();
      expect(executor.isInitialized()).toBe(false);
    });

    it('should initialize successfully without protocols', async () => {
      await executor.initialize();
      expect(executor.isInitialized()).toBe(true);
    });

    it('should not reinitialize if already initialized', async () => {
      await executor.initialize();
      await executor.initialize(); // Second call should be no-op
      expect(executor.isInitialized()).toBe(true);
    });
  });

  describe('protocol errors when disabled', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should throw for marginfi when not initialized', async () => {
      await expect(executor.deposit('marginfi', { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('MarginFi not initialized');
    });

    it('should throw for kamino when not initialized', async () => {
      await expect(executor.deposit('kamino', { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('Kamino not initialized');
    });

    it('should throw for solend when not initialized', async () => {
      await expect(executor.deposit('solend', { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('Solend not initialized');
    });

    it('should throw for unknown protocol', async () => {
      await expect(executor.deposit('unknown' as any, { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('Unknown protocol');
    });

    it('should throw for kamino withdraw when not initialized', async () => {
      await expect(executor.withdraw('kamino', { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('Kamino not initialized');
    });

    it('should throw for solend borrow when not initialized', async () => {
      await expect(executor.borrow('solend', { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('Solend not initialized');
    });

    it('should throw for kamino repay when not initialized', async () => {
      await expect(executor.repay('kamino', { asset: 'USDC', amount: 100 }))
        .rejects.toThrow('Kamino not initialized');
    });
  });

  describe('getAllPositions', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should return empty positions array when no protocols', async () => {
      const positions = await executor.getAllPositions();
      expect(Array.isArray(positions)).toBe(true);
      expect(positions.length).toBe(0);
    });
  });

  describe('getAllAPYs', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should return APYs object with protocol keys', async () => {
      const apys = await executor.getAllAPYs();
      expect(apys).toHaveProperty('marginfi');
      expect(apys).toHaveProperty('kamino');
      expect(apys).toHaveProperty('solend');
      expect(apys.marginfi).toEqual([]);
    });
  });

  describe('getHealthFactor', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should return 0 for marginfi when not initialized', () => {
      const hf = executor.getHealthFactor('marginfi');
      expect(hf).toBe(0);
    });

    it('should return 0 for kamino when not initialized', () => {
      const hf = executor.getHealthFactor('kamino');
      expect(hf).toBe(0);
    });

    it('should return 0 for solend when not initialized', () => {
      const hf = executor.getHealthFactor('solend');
      expect(hf).toBe(0);
    });
  });

  describe('getSupportedProtocols', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should return empty array when no protocols enabled', () => {
      const protocols = executor.getSupportedProtocols();
      expect(Array.isArray(protocols)).toBe(true);
      expect(protocols.length).toBe(0);
    });
  });

  describe('getBestLendingRate', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should return null when no protocols have APYs', async () => {
      const best = await executor.getBestLendingRate('USDC');
      expect(best).toBeNull();
    });
  });

  describe('getBestBorrowRate', () => {
    beforeEach(async () => {
      await executor.initialize();
    });

    it('should return null when no protocols have APYs', async () => {
      const best = await executor.getBestBorrowRate('USDC');
      expect(best).toBeNull();
    });
  });
});

describe('getLendingExecutor singleton', () => {
  beforeEach(() => {
    resetLendingExecutor();
  });

  it('should create singleton with config', () => {
    const executor = getLendingExecutor(mockConfig);
    expect(executor).toBeDefined();
  });

  it('should return same instance on subsequent calls', () => {
    const executor1 = getLendingExecutor(mockConfig);
    const executor2 = getLendingExecutor();
    expect(executor1).toBe(executor2);
  });

  it('should throw if called without config before initialization', () => {
    expect(() => getLendingExecutor()).toThrow('LendingExecutor not initialized');
  });
});
