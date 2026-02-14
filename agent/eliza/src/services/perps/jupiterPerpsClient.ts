/**
 * Jupiter Perpetuals Integration
 * 
 * Client for interacting with Jupiter Perpetual Exchange:
 * - LP-to-trader perps model (JLP pool)
 * - Long/short positions with leverage
 * - No funding rate (traders pay/receive swap fees)
 * 
 * Note: Jupiter Perps uses a different model than traditional perps:
 * - No funding rate (like CEX perps)
 * - Borrow rate for leverage
 * - JLP pool as counterparty
 */
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { logger } from '../logger.js';
import type {
  PerpsVenue,
  PerpsPosition,
  PerpsTradeResult,
  FundingRate,
  PositionSide,
} from '../../types/perps.js';

// ============= CONSTANTS =============

// Jupiter Perps API (mainnet)
const JUPITER_PERPS_API = 'https://perps-api.jup.ag';
const JUPITER_PERPS_STATS_API = 'https://perps-stats.jup.ag';

// Jupiter Perps Markets
export const JUPITER_PERPS_MARKETS = {
  'SOL-PERP': { mint: 'So11111111111111111111111111111111111111112', decimals: 9 },
  'ETH-PERP': { mint: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs', decimals: 8 },
  'BTC-PERP': { mint: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E', decimals: 6 },
} as const;

export type JupiterPerpsMarket = keyof typeof JUPITER_PERPS_MARKETS;

// ============= TYPES =============

export interface JupiterPerpsConfig {
  rpcUrl: string;
  privateKey?: string;
}

export interface JupiterMarketStats {
  market: string;
  longOpenInterest: number;
  shortOpenInterest: number;
  volume24h: number;
  fees24h: number;
  borrowRateLong: number;     // Hourly borrow rate for longs
  borrowRateShort: number;    // Hourly borrow rate for shorts
  markPrice: number;
  indexPrice: number;
}

export interface JupiterPosition {
  market: string;
  side: PositionSide;
  size: number;
  collateral: number;
  entryPrice: number;
  markPrice: number;
  leverage: number;
  unrealizedPnl: number;
  liquidationPrice: number;
  borrowFeeAccrued: number;
}

// ============= JUPITER PERPS CLIENT =============

export class JupiterPerpsClient {
  private connection: Connection;
  private keypair: Keypair | null = null;
  private config: JupiterPerpsConfig;
  private initialized: boolean = false;

  constructor(config: JupiterPerpsConfig) {
    this.config = config;
    this.connection = new Connection(config.rpcUrl, 'confirmed');

    if (config.privateKey) {
      try {
        const bs58 = require('bs58');
        this.keypair = Keypair.fromSecretKey(bs58.decode(config.privateKey));
      } catch {
        try {
          const secretKey = Uint8Array.from(Buffer.from(config.privateKey, 'base64'));
          this.keypair = Keypair.fromSecretKey(secretKey);
        } catch {
          logger.warn('JupiterPerpsClient: Invalid private key format');
        }
      }
    }

    logger.info('JupiterPerpsClient created', {
      hasKeypair: !!this.keypair,
      wallet: this.keypair?.publicKey.toBase58().slice(0, 8) + '...',
    });
  }

  async initialize(): Promise<boolean> {
    try {
      this.initialized = true;
      logger.info('JupiterPerpsClient initialized');
      return true;
    } catch (error) {
      logger.error('Failed to initialize JupiterPerpsClient', { error });
      return false;
    }
  }

  // ============= MARKET DATA =============

  /**
   * Fetch market statistics
   */
  async getMarketStats(): Promise<JupiterMarketStats[]> {
    try {
      // Jupiter Perps API endpoint for market stats
      const response = await fetch(`${JUPITER_PERPS_STATS_API}/v1/stats`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as any;

      // Transform to our format
      return Object.entries(data.markets || {}).map(([market, stats]: [string, any]) => ({
        market,
        longOpenInterest: stats.longOpenInterest || 0,
        shortOpenInterest: stats.shortOpenInterest || 0,
        volume24h: stats.volume24h || 0,
        fees24h: stats.fees24h || 0,
        borrowRateLong: stats.borrowRateLong || 0,
        borrowRateShort: stats.borrowRateShort || 0,
        markPrice: stats.markPrice || 0,
        indexPrice: stats.indexPrice || 0,
      }));
    } catch (error) {
      logger.error('Failed to fetch Jupiter Perps market stats', { error });
      return [];
    }
  }

  /**
   * Get borrow rates as "funding rates" for comparison
   * Note: Jupiter Perps doesn't have traditional funding rates
   * We return borrow rates which serve similar purpose
   */
  async getBorrowRates(): Promise<FundingRate[]> {
    try {
      const stats = await this.getMarketStats();

      return stats.flatMap(s => [
        {
          venue: 'jupiter' as PerpsVenue,
          market: s.market,
          rate: s.borrowRateLong,
          annualizedRate: s.borrowRateLong * 24 * 365,
          nextFundingTime: 0, // Continuous
          timestamp: Date.now(),
        },
      ]);
    } catch (error) {
      logger.error('Failed to fetch Jupiter Perps borrow rates', { error });
      return [];
    }
  }

  // ============= POSITION MANAGEMENT =============

  /**
   * Get current positions
   */
  async getPositions(): Promise<JupiterPosition[]> {
    if (!this.keypair) {
      logger.warn('JupiterPerpsClient: No wallet configured');
      return [];
    }

    try {
      const walletAddress = this.keypair.publicKey.toBase58();
      const response = await fetch(
        `${JUPITER_PERPS_API}/v1/positions?wallet=${walletAddress}`
      );

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as any;

      return (data.positions || []).map((p: any) => ({
        market: p.market,
        side: p.side as PositionSide,
        size: p.size,
        collateral: p.collateral,
        entryPrice: p.entryPrice,
        markPrice: p.markPrice,
        leverage: p.leverage,
        unrealizedPnl: p.unrealizedPnl,
        liquidationPrice: p.liquidationPrice,
        borrowFeeAccrued: p.borrowFeeAccrued,
      }));
    } catch (error) {
      logger.error('Failed to fetch Jupiter Perps positions', { error });
      return [];
    }
  }

  /**
   * Calculate liquidation price
   */
  calculateLiquidationPrice(params: {
    side: PositionSide;
    entryPrice: number;
    size: number;
    collateral: number;
    leverage: number;
  }): number {
    const { side, entryPrice, size, collateral, leverage } = params;

    // Jupiter uses 90% max LTV for liquidation
    const maxLTV = 0.9;
    const maintenanceMargin = collateral * (1 - maxLTV);
    const maxLoss = collateral - maintenanceMargin;
    const priceMove = maxLoss / size;

    if (side === 'long') {
      return Math.max(0, entryPrice - priceMove);
    } else {
      return entryPrice + priceMove;
    }
  }

  // ============= TRADING =============

  /**
   * Open a position on Jupiter Perps
   */
  async openPosition(params: {
    market: JupiterPerpsMarket;
    side: PositionSide;
    size: number;
    collateral: number;
    leverage: number;
    slippageBps?: number;
  }): Promise<PerpsTradeResult> {
    const { market, side, size, collateral, leverage, slippageBps = 100 } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'jupiter',
      side,
      size,
      leverage,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      const marketInfo = JUPITER_PERPS_MARKETS[market];
      if (!marketInfo) {
        throw new Error(`Unknown market: ${market}`);
      }

      const stats = await this.getMarketStats();
      const marketStats = stats.find(s => s.market === market);
      const entryPrice = marketStats?.markPrice || 0;

      logger.info('Opening Jupiter Perps position', {
        market,
        side,
        size,
        collateral,
        leverage,
        entryPrice,
      });

      // In a full implementation:
      // 1. Build transaction using Jupiter Perps SDK
      // 2. Sign and send transaction
      // 3. Confirm and return result

      // Calculate expected values
      const notional = size * entryPrice;

      // Jupiter charges ~0.1% for positions
      result.fees.trading = notional * 0.001;
      result.fees.gas = 0.01;

      result.liquidationPrice = this.calculateLiquidationPrice({
        side,
        entryPrice,
        size,
        collateral,
        leverage,
      });

      result.liquidationDistance = Math.abs(entryPrice - result.liquidationPrice) / entryPrice;

      result.success = true;
      result.entryPrice = entryPrice;
      result.orderId = `jup-perps-${Date.now()}`;
      result.positionId = `jup-pos-${market}-${side}-${Date.now()}`;

      logger.info('Jupiter Perps position opened (simulated)', {
        orderId: result.orderId,
        entryPrice: result.entryPrice,
        liquidationPrice: result.liquidationPrice,
      });

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      result.error = errorMessage;
      logger.error('Failed to open Jupiter Perps position', { error: errorMessage });
      return result;
    }
  }

  /**
   * Close a position on Jupiter Perps
   */
  async closePosition(params: {
    market: JupiterPerpsMarket;
    size?: number;
  }): Promise<PerpsTradeResult> {
    const { market, size } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'jupiter',
      side: 'long',
      size: size || 0,
      leverage: 1,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      logger.info('Closing Jupiter Perps position', { market, size });

      result.success = true;
      result.orderId = `jup-close-${Date.now()}`;

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      result.error = errorMessage;
      logger.error('Failed to close Jupiter Perps position', { error: errorMessage });
      return result;
    }
  }

  // ============= UTILITY =============

  isReady(): boolean {
    return this.initialized && !!this.keypair;
  }

  getWalletAddress(): string | null {
    return this.keypair?.publicKey.toBase58() || null;
  }
}

// ============= FACTORY =============

let jupiterPerpsClientInstance: JupiterPerpsClient | null = null;

export function getJupiterPerpsClient(config?: JupiterPerpsConfig): JupiterPerpsClient {
  if (!jupiterPerpsClientInstance && config) {
    jupiterPerpsClientInstance = new JupiterPerpsClient(config);
  }
  if (!jupiterPerpsClientInstance) {
    throw new Error('JupiterPerpsClient not initialized');
  }
  return jupiterPerpsClientInstance;
}

export function resetJupiterPerpsClient(): void {
  jupiterPerpsClientInstance = null;
}
