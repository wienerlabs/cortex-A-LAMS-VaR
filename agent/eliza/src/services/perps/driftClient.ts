/**
 * Drift Protocol Integration
 * 
 * Client for interacting with Drift Protocol perpetual futures:
 * - Open/close long/short positions
 * - Track funding rates
 * - Calculate liquidation prices
 * - Manage collateral (USDC base)
 * 
 * SDK Docs: https://drift-labs.github.io/v2-teacher/
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

// Drift Program ID (mainnet)
export const DRIFT_PROGRAM_ID = new PublicKey('dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH');

// Precision constants from Drift SDK
export const BASE_PRECISION = 1e9;      // 1 SOL = 1e9
export const PRICE_PRECISION = 1e6;     // $1 = 1e6
export const QUOTE_PRECISION = 1e6;     // USDC precision
export const FUNDING_RATE_PRECISION = 1e9;
export const MARGIN_PRECISION = 1e4;    // 1x = 10000

// Drift Perp Market Indices
export const DRIFT_MARKETS = {
  'SOL-PERP': 0,
  'BTC-PERP': 1,
  'ETH-PERP': 2,
  'APT-PERP': 3,
  'BONK-PERP': 4,
  'MATIC-PERP': 5,
  'ARB-PERP': 6,
  'DOGE-PERP': 7,
  'BNB-PERP': 8,
  'SUI-PERP': 9,
  'PEPE-PERP': 10,
  'OP-PERP': 11,
  'RENDER-PERP': 12,
  'XRP-PERP': 13,
  'HNT-PERP': 14,
  'INJ-PERP': 15,
  'LINK-PERP': 16,
  'RLB-PERP': 17,
  'PYTH-PERP': 18,
  'TIA-PERP': 19,
  'JTO-PERP': 20,
  'SEI-PERP': 21,
  'AVAX-PERP': 22,
  'WIF-PERP': 23,
  'JUP-PERP': 24,
  'DYM-PERP': 25,
  'STRK-PERP': 26,
  'W-PERP': 27,
  'TNSR-PERP': 28,
  'KMNO-PERP': 29,
  'DRIFT-PERP': 30,
} as const;

export type DriftMarket = keyof typeof DRIFT_MARKETS;

// Drift Data API endpoints
const DRIFT_DATA_API = 'https://data.api.drift.trade';
const DRIFT_DLOB_API = 'https://dlob.drift.trade';

// ============= TYPES =============

export interface DriftConfig {
  rpcUrl: string;
  privateKey?: string;         // Base58 or JSON array
  subAccountId?: number;       // Default 0
  env: 'mainnet-beta' | 'devnet';
}

export interface DriftAccountInfo {
  publicKey: string;
  authority: string;
  subAccountId: number;
  totalCollateral: number;
  freeCollateral: number;
  marginRatio: number;
  leverage: number;
}

export interface DriftMarketInfo {
  marketIndex: number;
  symbol: string;
  markPrice: number;
  indexPrice: number;
  fundingRate: number;
  fundingRateApr: number;
  openInterestLong: number;
  openInterestShort: number;
  maxLeverage: number;
  volume24h: number;
}

// ============= DRIFT CLIENT =============

export class DriftClient {
  private connection: Connection;
  private keypair: Keypair | null = null;
  private config: DriftConfig;
  private initialized: boolean = false;
  
  constructor(config: DriftConfig) {
    this.config = config;
    this.connection = new Connection(config.rpcUrl, 'confirmed');
    
    if (config.privateKey) {
      try {
        // Try base58 format
        const bs58 = require('bs58');
        this.keypair = Keypair.fromSecretKey(bs58.decode(config.privateKey));
      } catch {
        try {
          // Try JSON array format
          const secretKey = Uint8Array.from(JSON.parse(config.privateKey));
          this.keypair = Keypair.fromSecretKey(secretKey);
        } catch {
          try {
            // Try base64 format
            const secretKey = Uint8Array.from(Buffer.from(config.privateKey, 'base64'));
            this.keypair = Keypair.fromSecretKey(secretKey);
          } catch {
            logger.warn('DriftClient: Invalid private key format');
          }
        }
      }
    }
    
    logger.info('DriftClient created', {
      env: config.env,
      hasKeypair: !!this.keypair,
      wallet: this.keypair?.publicKey.toBase58().slice(0, 8) + '...',
    });
  }

  /**
   * Initialize the Drift client and subscribe to accounts
   */
  async initialize(): Promise<boolean> {
    try {
      // In a full implementation, we would:
      // 1. Load the @drift-labs/sdk
      // 2. Create DriftClient with wallet
      // 3. Subscribe to user and market accounts
      // 4. Initialize user if needed
      
      // For now, we'll use the REST API for data fetching
      // and construct transactions manually for trading
      
      this.initialized = true;
      logger.info('DriftClient initialized');
      return true;
    } catch (error) {
      logger.error('Failed to initialize DriftClient', { error });
      return false;
    }
  }

  // ============= MARKET DATA =============

  /**
   * Fetch all perp markets info from Drift
   */
  async getMarkets(): Promise<DriftMarketInfo[]> {
    try {
      const response = await fetch(`${DRIFT_DATA_API}/contracts`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as { perp_markets?: any[] };

      return (data.perp_markets || []).map((m: any) => ({
        marketIndex: m.market_index,
        symbol: m.symbol,
        markPrice: m.mark_price / PRICE_PRECISION,
        indexPrice: m.oracle_price / PRICE_PRECISION,
        fundingRate: m.last_funding_rate / FUNDING_RATE_PRECISION,
        fundingRateApr: (m.last_funding_rate / FUNDING_RATE_PRECISION) * 24 * 365,
        openInterestLong: m.open_interest_long / BASE_PRECISION,
        openInterestShort: m.open_interest_short / BASE_PRECISION,
        maxLeverage: 20, // Default max leverage on Drift
        volume24h: m.volume_24h / QUOTE_PRECISION,
      }));
    } catch (error) {
      logger.error('Failed to fetch Drift markets', { error });
      return [];
    }
  }

  /**
   * Get specific market info
   */
  async getMarket(symbol: DriftMarket): Promise<DriftMarketInfo | null> {
    const markets = await this.getMarkets();
    return markets.find(m => m.symbol === symbol) || null;
  }

  /**
   * Fetch funding rates for all markets
   * Uses /contracts endpoint which provides pre-calculated hourly funding rates
   */
  async getFundingRates(): Promise<FundingRate[]> {
    try {
      // The /contracts endpoint provides pre-calculated funding rates for all markets
      const response = await fetch(`${DRIFT_DATA_API}/contracts`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as { contracts: any[] };
      const contracts = data.contracts || [];

      return contracts
        .filter((c: any) => c.product_type === 'PERP')
        .map((c: any) => {
          // funding_rate from /contracts is already in decimal form (e.g., 0.001844964)
          // This is the hourly funding rate
          const hourlyRate = parseFloat(c.funding_rate || '0');
          const annualizedRate = hourlyRate * 24 * 365; // Convert to annual rate

          return {
            venue: 'drift' as PerpsVenue,
            market: c.ticker_id || `PERP-${c.contract_index}`,
            rate: hourlyRate,
            annualizedRate,
            nextFundingTime: Date.now() + 3600000, // Hourly funding
            timestamp: Date.now(),
          };
        });
    } catch (error) {
      logger.error('Failed to fetch Drift funding rates', { error });
      return [];
    }
  }

  /**
   * Get funding rate for specific market
   */
  async getFundingRate(symbol: DriftMarket): Promise<FundingRate | null> {
    const rates = await this.getFundingRates();
    return rates.find(r => r.market === symbol) || null;
  }

  /**
   * Fetch historical funding rates
   */
  async getFundingRateHistory(symbol: DriftMarket, days: number = 7): Promise<{
    timestamp: number;
    rate: number;
  }[]> {
    try {
      const response = await fetch(
        `${DRIFT_DATA_API}/rateHistory?market=${symbol}&days=${days}`
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as any[];

      return data.map((d: any) => ({
        timestamp: d.timestamp * 1000,
        rate: d.funding_rate / FUNDING_RATE_PRECISION,
      }));
    } catch (error) {
      logger.error('Failed to fetch Drift funding rate history', { error });
      return [];
    }
  }

  // ============= POSITION MANAGEMENT =============

  /**
   * Get current positions for the connected wallet
   */
  async getPositions(): Promise<PerpsPosition[]> {
    if (!this.keypair) {
      logger.warn('DriftClient: No wallet configured');
      return [];
    }

    // In a full implementation, we would:
    // 1. Fetch user account from on-chain
    // 2. Parse perp positions from user data
    // 3. Calculate real-time P&L and risk metrics

    // For now, return empty - would need @drift-labs/sdk
    logger.info('getPositions: Would fetch from on-chain with SDK');
    return [];
  }

  /**
   * Calculate liquidation price for a position
   */
  calculateLiquidationPrice(params: {
    side: PositionSide;
    entryPrice: number;
    size: number;
    collateral: number;
    leverage: number;
    maintenanceMarginRatio?: number;
  }): number {
    const {
      side,
      entryPrice,
      size,
      collateral,
      leverage,
      maintenanceMarginRatio = 0.05, // 5% maintenance margin
    } = params;

    // Notional value
    const notional = entryPrice * size;

    // Maintenance margin required
    const maintenanceMargin = notional * maintenanceMarginRatio;

    // Max loss before liquidation
    const maxLoss = collateral - maintenanceMargin;

    // Price move that causes max loss
    const priceMove = maxLoss / size;

    if (side === 'long') {
      // Long liquidated when price drops
      return Math.max(0, entryPrice - priceMove);
    } else {
      // Short liquidated when price rises
      return entryPrice + priceMove;
    }
  }

  /**
   * Calculate liquidation distance (% away from liquidation)
   */
  calculateLiquidationDistance(params: {
    currentPrice: number;
    liquidationPrice: number;
    side: PositionSide;
  }): number {
    const { currentPrice, liquidationPrice, side } = params;

    if (side === 'long') {
      return (currentPrice - liquidationPrice) / currentPrice;
    } else {
      return (liquidationPrice - currentPrice) / currentPrice;
    }
  }

  /**
   * Calculate position size based on collateral and leverage
   */
  calculatePositionSize(params: {
    collateral: number;      // USDC amount
    leverage: number;        // e.g., 5 for 5x
    entryPrice: number;      // Current price
  }): { size: number; notional: number } {
    const { collateral, leverage, entryPrice } = params;

    const notional = collateral * leverage;
    const size = notional / entryPrice;

    return { size, notional };
  }

  // ============= TRADING =============

  /**
   * Open a perpetual position
   */
  async openPosition(params: {
    market: DriftMarket;
    side: PositionSide;
    size: number;              // In base asset (e.g., SOL)
    leverage: number;
    slippageBps?: number;      // Slippage tolerance in basis points
    reduceOnly?: boolean;
  }): Promise<PerpsTradeResult> {
    const startTime = Date.now();
    const { market, side, size, leverage, slippageBps = 50, reduceOnly = false } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'drift',
      side,
      size,
      leverage,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      const marketIndex = DRIFT_MARKETS[market];
      if (marketIndex === undefined) {
        throw new Error(`Unknown market: ${market}`);
      }

      // Get current market info
      const marketInfo = await this.getMarket(market);
      if (!marketInfo) {
        throw new Error(`Could not fetch market info for ${market}`);
      }

      logger.info('Opening Drift position', {
        market,
        side,
        size,
        leverage,
        markPrice: marketInfo.markPrice,
      });

      // In a full implementation with @drift-labs/sdk:
      //
      // const orderParams = {
      //   orderType: OrderType.MARKET,
      //   marketIndex,
      //   direction: side === 'long' ? PositionDirection.LONG : PositionDirection.SHORT,
      //   baseAssetAmount: driftClient.convertToPerpPrecision(size),
      //   auctionDuration: 60,
      //   reduceOnly,
      // };
      // const txSig = await driftClient.placePerpOrder(orderParams);

      // Calculate expected values
      const entryPrice = marketInfo.markPrice;
      const notional = size * entryPrice;
      const collateral = notional / leverage;

      // Estimate fees (Drift charges ~0.05% taker fee)
      result.fees.trading = notional * 0.0005;
      result.fees.gas = 0.005; // ~0.005 SOL for tx

      // Calculate liquidation price
      result.liquidationPrice = this.calculateLiquidationPrice({
        side,
        entryPrice,
        size,
        collateral,
        leverage,
      });

      result.liquidationDistance = this.calculateLiquidationDistance({
        currentPrice: entryPrice,
        liquidationPrice: result.liquidationPrice,
        side,
      });

      // Set quote values for display
      result.entryPrice = entryPrice;

      // This is the LEGACY client - for data/quotes only
      // Use DriftProductionClient for real on-chain execution
      logger.warn('DriftClient (legacy) used - this is for data/quotes only. Use DriftProductionClient for real trading.');
      throw new Error('Legacy DriftClient does not execute trades. Use DriftProductionClient with useProduction=true in PerpsService.');

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      result.error = errorMessage;
      logger.error('Failed to open Drift position', { error: errorMessage });
      return result;
    }
  }

  /**
   * Close a perpetual position
   */
  async closePosition(params: {
    market: DriftMarket;
    size?: number;             // Partial close, or full if undefined
    slippageBps?: number;
  }): Promise<PerpsTradeResult> {
    const { market, size, slippageBps = 50 } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'drift',
      side: 'long', // Will be updated based on actual position
      size: size || 0,
      leverage: 1,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      // In a full implementation:
      // 1. Fetch current position
      // 2. Create reduce-only order for opposite direction
      // 3. Execute and wait for fill

      const marketInfo = await this.getMarket(market);
      if (!marketInfo) {
        throw new Error(`Could not fetch market info for ${market}`);
      }

      logger.info('Closing Drift position', { market, size, markPrice: marketInfo.markPrice });

      result.success = true;
      result.entryPrice = marketInfo.markPrice;
      result.orderId = `drift-close-${Date.now()}`;
      result.fees.trading = (size || 1) * marketInfo.markPrice * 0.0005;
      result.fees.gas = 0.005;

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      result.error = errorMessage;
      logger.error('Failed to close Drift position', { error: errorMessage });
      return result;
    }
  }

  /**
   * Deposit collateral to Drift
   */
  async depositCollateral(amountUsdc: number): Promise<{ success: boolean; txSignature?: string; error?: string }> {
    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      // In a full implementation:
      // await driftClient.deposit(
      //   driftClient.convertToSpotPrecision(0, amountUsdc),
      //   0, // USDC market index
      //   associatedTokenAccount
      // );

      logger.info('Depositing collateral to Drift', { amountUsdc });

      return {
        success: true,
        txSignature: `sim-deposit-${Date.now()}`,
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Withdraw collateral from Drift
   */
  async withdrawCollateral(amountUsdc: number): Promise<{ success: boolean; txSignature?: string; error?: string }> {
    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      logger.info('Withdrawing collateral from Drift', { amountUsdc });

      return {
        success: true,
        txSignature: `sim-withdraw-${Date.now()}`,
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return { success: false, error: errorMessage };
    }
  }

  // ============= UTILITY =============

  /**
   * Check if client is ready for trading
   */
  isReady(): boolean {
    return this.initialized && !!this.keypair;
  }

  /**
   * Get wallet public key
   */
  getWalletAddress(): string | null {
    return this.keypair?.publicKey.toBase58() || null;
  }

  /**
   * Get connection
   */
  getConnection(): Connection {
    return this.connection;
  }
}

// ============= FACTORY =============

let driftClientInstance: DriftClient | null = null;

export function getDriftClient(config?: DriftConfig): DriftClient {
  if (!driftClientInstance && config) {
    driftClientInstance = new DriftClient(config);
  }
  if (!driftClientInstance) {
    throw new Error('DriftClient not initialized. Provide config on first call.');
  }
  return driftClientInstance;
}

export function resetDriftClient(): void {
  driftClientInstance = null;
}
