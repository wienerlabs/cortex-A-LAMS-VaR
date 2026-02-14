/**
 * Kamino Client Tests
 *
 * Unit tests for the Kamino lending client.
 * Integration tests require valid RPC and wallet.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { PublicKey, Keypair } from '@solana/web3.js';
import bs58 from 'bs58';
import { KaminoLendingClient } from '../kaminoClient.js';

// Generate a valid test keypair
const testKeypair = Keypair.generate();
const testPrivateKey = bs58.encode(testKeypair.secretKey);

describe('KaminoLendingClient', () => {
  let client: KaminoLendingClient;

  beforeEach(() => {
    client = new KaminoLendingClient({
      rpcUrl: 'https://api.mainnet-beta.solana.com',
      privateKey: testPrivateKey,
    });
  });

  describe('constructor', () => {
    it('should create client instance', () => {
      expect(client).toBeDefined();
      expect(client.isInitialized()).toBe(false);
    });

    it('should have correct public key', () => {
      expect(client.publicKey).toBeInstanceOf(PublicKey);
    });
  });

  describe('before initialization', () => {
    it('should return error when depositing without init', async () => {
      const result = await client.deposit({ asset: 'USDC', amount: 100 });
      expect(result.success).toBe(false);
      expect(result.error).toContain('not initialized');
    });

    it('should return error when withdrawing without init', async () => {
      const result = await client.withdraw({ asset: 'USDC', amount: 100 });
      expect(result.success).toBe(false);
      expect(result.error).toContain('not initialized');
    });

    it('should return error when borrowing without init', async () => {
      const result = await client.borrow({ asset: 'USDC', amount: 100 });
      expect(result.success).toBe(false);
      expect(result.error).toContain('not initialized');
    });

    it('should return error when repaying without init', async () => {
      const result = await client.repay({ asset: 'USDC', amount: 100 });
      expect(result.success).toBe(false);
      expect(result.error).toContain('not initialized');
    });

    it('should return 0 for health factor when not initialized', () => {
      expect(client.getHealthFactor()).toBe(0);
    });
  });

  describe('getPositions', () => {
    it('should return empty array when not initialized', async () => {
      const positions = await client.getPositions();
      expect(positions).toEqual([]);
    });
  });

  describe('getAPYs', () => {
    it('should return empty array when not initialized', async () => {
      const apys = await client.getAPYs();
      expect(apys).toEqual([]);
    });
  });
});

describe('KaminoLendingClient - Integration', () => {
  // These tests require valid RPC and wallet - skip in CI
  const RPC_URL = process.env.RPC_URL;
  const PRIVATE_KEY = process.env.TEST_PRIVATE_KEY;

  const shouldRunIntegration = RPC_URL && PRIVATE_KEY;

  it.skipIf(!shouldRunIntegration)('should initialize with real connection', async () => {
    const client = new KaminoLendingClient({
      rpcUrl: RPC_URL!,
      privateKey: PRIVATE_KEY!,
    });

    await client.initialize();
    expect(client.isInitialized()).toBe(true);
  });

  it.skipIf(!shouldRunIntegration)('should fetch APYs', async () => {
    const client = new KaminoLendingClient({
      rpcUrl: RPC_URL!,
      privateKey: PRIVATE_KEY!,
    });

    await client.initialize();
    const apys = await client.getAPYs();

    expect(apys.length).toBeGreaterThan(0);
    expect(apys[0]).toHaveProperty('asset');
    expect(apys[0]).toHaveProperty('supplyAPY');
    expect(apys[0]).toHaveProperty('borrowAPY');
  });
});

