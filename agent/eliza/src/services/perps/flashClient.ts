/**
 * Flash Trade Integration
 * 
 * Client for interacting with Flash Trade perpetual futures:
 * - Similar LP-to-trader model
 * - Long/short positions with leverage
 * - Focus on low fees and fast execution
 * 
 * GitHub: https://github.com/flash-trade/flash-trade-sdk
 */
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { logger } from '../logger.js';
import type {
  PerpsVenue,
  PerpsTradeResult,
  FundingRate,
  PositionSide,
} from '../../types/perps.js';

// ============= CONSTANTS =============

// Flash Trade API endpoints
const FLASH_API = 'https://api.flash.trade';
const FLASH_STATS_API = 'https://stats.flash.trade';

// Flash Program ID (mainnet) - placeholder, Flash uses API-based trading
// Using a valid base58 placeholder since Flash doesn't have a public program ID
export const FLASH_PROGRAM_ID = new PublicKey('11111111111111111111111111111111');

// Flash Markets
export const FLASH_MARKETS = {
  'SOL-PERP': { index: 0, decimals: 9 },
  'ETH-PERP': { index: 1, decimals: 8 },
  'BTC-PERP': { index: 2, decimals: 6 },
  'BONK-PERP': { index: 3, decimals: 5 },
} as const;

export type FlashMarket = keyof typeof FLASH_MARKETS;

// ============= TYPES =============

export interface FlashConfig {
  rpcUrl: string;
  privateKey?: string;
}

export interface FlashMarketStats {
  market: string;
  markPrice: number;
  indexPrice: number;
  fundingRate: number;
  fundingRateApr: number;
  longOpenInterest: number;
  shortOpenInterest: number;
  volume24h: number;
  maxLeverage: number;
}

export interface FlashPosition {
  market: string;
  side: PositionSide;
  size: number;
  collateral: number;
  entryPrice: number;
  markPrice: number;
  leverage: number;
  unrealizedPnl: number;
  liquidationPrice: number;
  fundingPaid: number;
}

// ============= FLASH CLIENT =============

export class FlashClient {
  private connection: Connection;
  private keypair: Keypair | null = null;
  private config: FlashConfig;
  private initialized: boolean = false;

  constructor(config: FlashConfig) {
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
          logger.warn('FlashClient: Invalid private key format');
        }
      }
    }

    logger.info('FlashClient created', {
      hasKeypair: !!this.keypair,
      wallet: this.keypair?.publicKey.toBase58().slice(0, 8) + '...',
    });
  }

  async initialize(): Promise<boolean> {
    try {
      this.initialized = true;
      logger.info('FlashClient initialized');
      return true;
    } catch (error) {
      logger.error('Failed to initialize FlashClient', { error });
      return false;
    }
  }

  // ============= MARKET DATA =============

  /**
   * Fetch market statistics from Flash API
   */
  async getMarketStats(): Promise<FlashMarketStats[]> {
    try {
      const response = await fetch(`${FLASH_STATS_API}/v1/markets`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as any;

      return (data.markets || []).map((m: any) => ({
        market: m.symbol,
        markPrice: m.markPrice,
        indexPrice: m.indexPrice,
        fundingRate: m.fundingRate || 0,
        fundingRateApr: (m.fundingRate || 0) * 24 * 365,
        longOpenInterest: m.longOpenInterest || 0,
        shortOpenInterest: m.shortOpenInterest || 0,
        volume24h: m.volume24h || 0,
        maxLeverage: m.maxLeverage || 50,
      }));
    } catch (error) {
      logger.error('Failed to fetch Flash market stats', { error });
      return [];
    }
  }

  /**
   * Get funding rates from Flash
   */
  async getFundingRates(): Promise<FundingRate[]> {
    try {
      const stats = await this.getMarketStats();

      return stats.map(s => ({
        venue: 'flash' as PerpsVenue,
        market: s.market,
        rate: s.fundingRate,
        annualizedRate: s.fundingRateApr,
        nextFundingTime: Date.now() + 3600000, // Hourly
        timestamp: Date.now(),
      }));
    } catch (error) {
      logger.error('Failed to fetch Flash funding rates', { error });
      return [];
    }
  }

  // ============= POSITION MANAGEMENT =============

  async getPositions(): Promise<FlashPosition[]> {
    if (!this.keypair) {
      logger.warn('FlashClient: No wallet configured');
      return [];
    }

    try {
      const walletAddress = this.keypair.publicKey.toBase58();
      const response = await fetch(`${FLASH_API}/v1/positions?wallet=${walletAddress}`);

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
        fundingPaid: p.fundingPaid,
      }));
    } catch (error) {
      logger.error('Failed to fetch Flash positions', { error });
      return [];
    }
  }

  calculateLiquidationPrice(params: {
    side: PositionSide;
    entryPrice: number;
    size: number;
    collateral: number;
  }): number {
    const { side, entryPrice, size, collateral } = params;
    const maintenanceMarginRatio = 0.05;
    const notional = entryPrice * size;
    const maintenanceMargin = notional * maintenanceMarginRatio;
    const maxLoss = collateral - maintenanceMargin;
    const priceMove = maxLoss / size;

    return side === 'long'
      ? Math.max(0, entryPrice - priceMove)
      : entryPrice + priceMove;
  }

  // ============= TRADING =============

  async openPosition(params: {
    market: FlashMarket;
    side: PositionSide;
    size: number;
    collateral: number;
    leverage: number;
  }): Promise<PerpsTradeResult> {
    const { market, side, size, collateral, leverage } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'flash',
      side,
      size,
      leverage,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      const stats = await this.getMarketStats();
      const marketStats = stats.find(s => s.market === market);
      const entryPrice = marketStats?.markPrice || 0;

      logger.info('Opening Flash position', {
        market, side, size, collateral, leverage, entryPrice,
      });

      const notional = size * entryPrice;
      result.fees.trading = notional * 0.0006; // 0.06% fee
      result.fees.gas = 0.005;

      result.liquidationPrice = this.calculateLiquidationPrice({
        side, entryPrice, size, collateral,
      });

      result.liquidationDistance = Math.abs(entryPrice - result.liquidationPrice) / entryPrice;
      result.success = true;
      result.entryPrice = entryPrice;
      result.orderId = `flash-${Date.now()}`;
      result.positionId = `flash-pos-${market}-${side}-${Date.now()}`;

      logger.info('Flash position opened (simulated)', { orderId: result.orderId });

      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
      logger.error('Failed to open Flash position', { error: result.error });
      return result;
    }
  }

  async closePosition(params: {
    market: FlashMarket;
    size?: number;
  }): Promise<PerpsTradeResult> {
    const { market, size } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'flash',
      side: 'long',
      size: size || 0,
      leverage: 1,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      logger.info('Closing Flash position', { market, size });
      result.success = true;
      result.orderId = `flash-close-${Date.now()}`;

      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
      logger.error('Failed to close Flash position', { error: result.error });
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

let flashClientInstance: FlashClient | null = null;

export function getFlashClient(config?: FlashConfig): FlashClient {
  if (!flashClientInstance && config) {
    flashClientInstance = new FlashClient(config);
  }
  if (!flashClientInstance) {
    throw new Error('FlashClient not initialized');
  }
  return flashClientInstance;
}

export function resetFlashClient(): void {
  flashClientInstance = null;
}
