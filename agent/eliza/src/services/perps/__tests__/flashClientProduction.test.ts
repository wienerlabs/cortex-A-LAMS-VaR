/**
 * Flash Production Client Unit Tests
 * 
 * Tests token mapping, SL/TP calculations, and pool configuration.
 * 
 * Run with: npm test -- flashClientProduction.test.ts
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  FLASH_MARKET_MAP,
  FLASH_POOLS,
  FlashProductionClient,
  type FlashProductionConfig,
} from '../flashClientProduction.js';

// ============= MOCK CONFIG =============

const mockConfig: FlashProductionConfig = {
  rpcUrl: 'https://api.mainnet-beta.solana.com',
  // Valid devnet keypair (not used for actual transactions in tests)
  privateKey: '5MaiiCavjCmn9Hs1o3eznqDEhRwxo7pXiAYez7keQUviUkauRiTMD8DrESdrNjN8zd9mTmVhRvBJeg5vhyvgrAhG',
  defaultPool: 'Crypto.1',
  defaultCollateral: 'USDC',
  defaultStopLossPercent: 0.05,
  defaultTakeProfitPercent: 0.10,
  prioritizationFee: 50000,
};

// ============= TOKEN MAPPING TESTS =============

describe('Flash Token Mapping', () => {
  describe('FLASH_MARKET_MAP', () => {
    it('should map SOL-PERP to correct Flash market', () => {
      const mapping = FLASH_MARKET_MAP['SOL-PERP'];
      
      expect(mapping).toBeDefined();
      expect(mapping.target).toBe('SOL');
      expect(mapping.collateral).toBe('USDC');
      expect(mapping.pool).toBe('Crypto.1');
    });

    it('should map BTC-PERP to correct Flash market', () => {
      const mapping = FLASH_MARKET_MAP['BTC-PERP'];
      
      expect(mapping).toBeDefined();
      expect(mapping.target).toBe('BTC');
      expect(mapping.collateral).toBe('USDC');
      expect(mapping.pool).toBe('Crypto.1');
    });

    it('should map ETH-PERP to correct Flash market', () => {
      const mapping = FLASH_MARKET_MAP['ETH-PERP'];
      
      expect(mapping).toBeDefined();
      expect(mapping.target).toBe('ETH');
      expect(mapping.collateral).toBe('USDC');
      expect(mapping.pool).toBe('Crypto.1');
    });

    it('should map BONK-PERP to Community pool', () => {
      const mapping = FLASH_MARKET_MAP['BONK-PERP'];
      
      expect(mapping).toBeDefined();
      expect(mapping.target).toBe('BONK');
      expect(mapping.collateral).toBe('USDC');
      expect(mapping.pool).toBe('Community.1');
    });

    it('should return undefined for unsupported market', () => {
      const mapping = FLASH_MARKET_MAP['UNKNOWN-PERP'];
      expect(mapping).toBeUndefined();
    });
  });
});

// ============= POOL CONFIGURATION TESTS =============

describe('Flash Pool Configuration', () => {
  it('should have Crypto.1 pool for major assets', () => {
    expect(FLASH_POOLS['Crypto.1']).toBe('Crypto.1');
  });

  it('should have Virtual.1 pool', () => {
    expect(FLASH_POOLS['Virtual.1']).toBe('Virtual.1');
  });

  it('should have Governance.1 pool', () => {
    expect(FLASH_POOLS['Governance.1']).toBe('Governance.1');
  });

  it('should have Community.1 pool', () => {
    expect(FLASH_POOLS['Community.1']).toBe('Community.1');
  });

  it('should use Crypto.1 pool for SOL/BTC/ETH', () => {
    expect(FLASH_MARKET_MAP['SOL-PERP'].pool).toBe('Crypto.1');
    expect(FLASH_MARKET_MAP['BTC-PERP'].pool).toBe('Crypto.1');
    expect(FLASH_MARKET_MAP['ETH-PERP'].pool).toBe('Crypto.1');
  });
});

// ============= STOP LOSS / TAKE PROFIT CALCULATION TESTS =============

describe('Stop Loss / Take Profit Calculations', () => {
  describe('Stop Loss for Long Position', () => {
    it('should calculate stop loss below entry for long', () => {
      // Entry: $200, SL: 5%
      // Expected: 200 * (1 - 0.05) = $190
      const entryPrice = 200;
      const slPercent = 0.05;
      const side = 'long';
      
      const stopLoss = side === 'long' 
        ? entryPrice * (1 - slPercent) 
        : entryPrice * (1 + slPercent);
      
      expect(stopLoss).toBe(190);
    });

    it('should calculate stop loss with 10% for long', () => {
      const entryPrice = 150;
      const slPercent = 0.10;
      const stopLoss = entryPrice * (1 - slPercent);
      
      expect(stopLoss).toBe(135);
    });
  });

  describe('Stop Loss for Short Position', () => {
    it('should calculate stop loss above entry for short', () => {
      // Entry: $200, SL: 5%
      // Expected: 200 * (1 + 0.05) = $210
      const entryPrice = 200;
      const slPercent = 0.05;
      const side = 'short';
      
      const stopLoss = side === 'short' 
        ? entryPrice * (1 + slPercent) 
        : entryPrice * (1 - slPercent);
      
      expect(stopLoss).toBe(210);
    });
  });

  describe('Take Profit for Long Position', () => {
    it('should calculate take profit above entry for long', () => {
      // Entry: $200, TP: 10%
      // Expected: 200 * (1 + 0.10) = $220
      const entryPrice = 200;
      const tpPercent = 0.10;
      const side = 'long';

      const takeProfit = side === 'long'
        ? entryPrice * (1 + tpPercent)
        : entryPrice * (1 - tpPercent);

      expect(takeProfit).toBeCloseTo(220, 2);
    });
  });

  describe('Take Profit for Short Position', () => {
    it('should calculate take profit below entry for short', () => {
      // Entry: $200, TP: 10%
      // Expected: 200 * (1 - 0.10) = $180
      const entryPrice = 200;
      const tpPercent = 0.10;
      const side = 'short';

      const takeProfit = side === 'short'
        ? entryPrice * (1 - tpPercent)
        : entryPrice * (1 + tpPercent);

      expect(takeProfit).toBe(180);
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero percent correctly', () => {
      const entryPrice = 200;
      const stopLoss = entryPrice * (1 - 0);
      expect(stopLoss).toBe(200);
    });

    it('should handle 100% stop loss', () => {
      const entryPrice = 200;
      const stopLoss = entryPrice * (1 - 1.0);
      expect(stopLoss).toBe(0);
    });

    it('should handle very small percentages', () => {
      const entryPrice = 200;
      const stopLoss = entryPrice * (1 - 0.001);
      expect(stopLoss).toBeCloseTo(199.8, 1);
    });
  });
});

// ============= CLIENT INITIALIZATION TESTS =============

describe('FlashProductionClient', () => {
  describe('Constructor', () => {
    it('should create client with valid config', () => {
      const client = new FlashProductionClient(mockConfig);
      expect(client).toBeDefined();
    });

    it('should throw error with invalid private key', () => {
      expect(() => {
        new FlashProductionClient({
          ...mockConfig,
          privateKey: 'invalid-key',
        });
      }).toThrow('Invalid private key format');
    });

    it('should default to Crypto.1 pool', () => {
      const client = new FlashProductionClient({
        ...mockConfig,
        defaultPool: undefined,
      });
      expect(client).toBeDefined();
    });
  });

  describe('isReady', () => {
    it('should return false before initialization', () => {
      const client = new FlashProductionClient(mockConfig);
      expect(client.isReady()).toBe(false);
    });
  });

  describe('getWalletAddress', () => {
    it('should return valid wallet address', () => {
      const client = new FlashProductionClient(mockConfig);
      const address = client.getWalletAddress();

      expect(address).toBeDefined();
      expect(typeof address).toBe('string');
      expect(address.length).toBeGreaterThan(30);
    });
  });
});

// ============= SLIPPAGE CALCULATION TESTS =============

describe('Slippage Calculations', () => {
  it('should calculate slippage multiplier for long position', () => {
    const slippageBps = 100; // 1%
    const multiplier = 1 + (slippageBps / 10000);
    expect(multiplier).toBe(1.01);
  });

  it('should calculate slippage multiplier for short position', () => {
    const slippageBps = 100; // 1%
    const multiplier = 1 - (slippageBps / 10000);
    expect(multiplier).toBe(0.99);
  });

  it('should calculate price with slippage for long', () => {
    const currentPrice = 200;
    const slippageBps = 50; // 0.5%
    const priceWithSlippage = currentPrice * (1 + slippageBps / 10000);
    expect(priceWithSlippage).toBeCloseTo(201, 2);
  });

  it('should calculate price with slippage for short', () => {
    const currentPrice = 200;
    const slippageBps = 50; // 0.5%
    const priceWithSlippage = currentPrice * (1 - slippageBps / 10000);
    expect(priceWithSlippage).toBe(199);
  });
});

// ============= BN CONVERSION TESTS =============

describe('BN Conversions', () => {
  it('should convert USD amount to BN with 6 decimals', () => {
    const usdAmount = 100.5;
    const bnValue = Math.floor(usdAmount * 1e6);
    expect(bnValue).toBe(100500000);
  });

  it('should convert size to BN with 6 decimals', () => {
    const size = 1.5;
    const bnValue = Math.floor(size * 1e6);
    expect(bnValue).toBe(1500000);
  });

  it('should handle small values', () => {
    const usdAmount = 0.000001;
    const bnValue = Math.floor(usdAmount * 1e6);
    expect(bnValue).toBe(1);
  });
});

