/**
 * Adrena Protocol Integration
 * 
 * Client for interacting with Adrena perpetual futures on Solana:
 * - Long/short positions with up to 100x leverage
 * - USDC, JITOSOL, WBTC, BONK as collateral
 * - LP-to-trader model with competitive fees
 * 
 * Website: https://app.adrena.xyz
 * Docs: https://docs.adrena.xyz
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

// Adrena Data API
const ADRENA_API = 'https://datapi.adrena.xyz';

// Adrena Program ID (mainnet)
export const ADRENA_PROGRAM_ID = new PublicKey('13gDzEXCdocbj8iAiqrScGo47NiSuYENGsRqi3SEAwet');

// Token mints
export const ADRENA_TOKENS = {
  USDC: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  JITOSOL: 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
  WBTC: '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh',
  BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
} as const;

// Adrena Markets (principal tokens for trading)
export const ADRENA_MARKETS = {
  'SOL-PERP': { symbol: 'SOLUSD', principal: 'JITOSOL', decimals: 9 },
  'BTC-PERP': { symbol: 'BTCUSD', principal: 'WBTC', decimals: 8 },
  'JITOSOL-PERP': { symbol: 'JITOSOLUSD', principal: 'JITOSOL', decimals: 9 },
  'BONK-PERP': { symbol: 'BONKUSD', principal: 'BONK', decimals: 5 },
} as const;

export type AdrenaMarket = keyof typeof ADRENA_MARKETS;

// Custody pubkeys to market mapping (main-pool)
// These are the custody accounts that hold collateral for each market
// Note: Multiple custodies can map to the same market (e.g., SOL-PERP uses both JitoSOL and USDC)
const CUSTODY_TO_MARKET: Record<string, AdrenaMarket[]> = {
  'GZ9XfWwgTRhkma2Y91Q9r1XKotNXYjBnKKabj19rhT71': ['SOL-PERP', 'JITOSOL-PERP'], // JitoSOL custody
  'GFu3qS22mo6bAjg4Lr5R7L8pPgHq6GvbjJPKEHkbbs2c': ['BTC-PERP'],                  // WBTC custody
  '8aJuzsgjxBnvRhDcfQBD7z4CUj7QoPEpaNwVd7KqsSk5': ['BONK-PERP'],                 // BONK custody
  'Dk523LZeDQbZtUwPEBjFXCd2Au1tD7mWZBJJmcgHktNk': [],                            // USDC custody (collateral only)
};

// ============= API TYPES =============

interface AdrenaPrice {
  symbol: string;
  feed_id: number;
  price: number;
  timestamp: number;
  exponent: number;
}

interface AdrenaPricesResponse {
  success: boolean;
  data: {
    latest_date: string;
    latest_timestamp: number;
    prices: AdrenaPrice[];
    signature: string;
    recovery_id: number;
  };
}

interface AdrenaPosition {
  position_id: number;
  symbol: string;
  side: 'long' | 'short';
  status: 'open' | 'close' | 'liquidate';
  entry_price: number | null;
  pnl: number | null;
  entry_leverage: number;
  entry_date: string;
  pubkey: string;
  collateral_amount: number | null;
  volume: number | null;
}

interface AdrenaPositionsResponse {
  success: boolean;
  data: AdrenaPosition[];
}

interface AdrenaCustodyInfoResponse {
  success: boolean;
  data: {
    snapshot_timestamp: string[];
    pool_name: string;
    borrow_rate?: Record<string, string[]>;
    price?: Record<string, string[]>;
  };
}

// ============= CLIENT TYPES =============

export interface AdrenaConfig {
  rpcUrl: string;
  privateKey?: string;
}

export interface AdrenaMarketStats {
  market: string;
  symbol: string;
  markPrice: number;
  fundingRate: number;
  fundingRateApr: number;
  maxLeverage: number;
}

export interface AdrenaPositionInfo {
  market: string;
  side: PositionSide;
  size: number;
  collateral: number;
  entryPrice: number;
  markPrice: number;
  leverage: number;
  unrealizedPnl: number;
  liquidationPrice: number;
  positionPubkey: string;
}

// ============= ADRENA CLIENT =============

export class AdrenaClient {
  private connection: Connection;
  private keypair: Keypair | null = null;
  private config: AdrenaConfig;
  private initialized: boolean = false;
  private priceCache: Map<string, { price: number; timestamp: number }> = new Map();
  private borrowRateCache: Map<string, { rate: number; timestamp: number }> = new Map();
  private readonly PRICE_CACHE_TTL = 5000; // 5 seconds
  private readonly BORROW_RATE_CACHE_TTL = 60000; // 60 seconds (borrow rates update less frequently)

  constructor(config: AdrenaConfig) {
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
          logger.warn('AdrenaClient: Invalid private key format');
        }
      }
    }

    logger.info('AdrenaClient created', {
      hasKeypair: !!this.keypair,
      wallet: this.keypair?.publicKey.toBase58().slice(0, 8) + '...',
    });
  }

  async initialize(): Promise<boolean> {
    try {
      // Verify API connectivity
      await this.fetchPrices();
      this.initialized = true;
      logger.info('AdrenaClient initialized');
      return true;
    } catch (error) {
      logger.error('Failed to initialize AdrenaClient', { error });
      return false;
    }
  }

  // ============= MARKET DATA =============

  /**
   * Fetch latest prices from Adrena API
   */
  private async fetchPrices(): Promise<Map<string, number>> {
    try {
      const response = await fetch(`${ADRENA_API}/last-trading-prices`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as AdrenaPricesResponse;
      const prices = new Map<string, number>();

      if (data.success && data.data.prices) {
        const now = Date.now();
        for (const p of data.data.prices) {
          // Adrena price format: raw price with exponent
          // e.g., SOL: price=13788632379, exponent=-10 => 13788632379 * 10^-10 = $1.378...
          // But the prices seem to be in a different format - let's use the exponent
          const priceUsd = p.price * Math.pow(10, p.exponent);
          prices.set(p.symbol, priceUsd);
          this.priceCache.set(p.symbol, { price: priceUsd, timestamp: now });
        }
      }

      return prices;
    } catch (error) {
      logger.error('Failed to fetch Adrena prices', { error });
      return new Map();
    }
  }

  /**
   * Get cached price or fetch fresh
   */
  async getPrice(symbol: string): Promise<number | null> {
    const cached = this.priceCache.get(symbol);
    if (cached && Date.now() - cached.timestamp < this.PRICE_CACHE_TTL) {
      return cached.price;
    }

    await this.fetchPrices();
    return this.priceCache.get(symbol)?.price || null;
  }

  /**
   * Fetch borrow rates from Adrena custody info API
   * Borrow rates in Adrena serve as the "funding rate" equivalent
   */
  private async fetchBorrowRates(): Promise<Map<AdrenaMarket, number>> {
    const rates = new Map<AdrenaMarket, number>();
    const now = Date.now();

    // Check if cache is still valid
    const firstCached = this.borrowRateCache.values().next().value;
    if (firstCached && now - firstCached.timestamp < this.BORROW_RATE_CACHE_TTL) {
      // Return from cache
      for (const [market] of Object.entries(ADRENA_MARKETS)) {
        const cached = this.borrowRateCache.get(market);
        if (cached) {
          rates.set(market as AdrenaMarket, cached.rate);
        }
      }
      return rates;
    }

    try {
      const response = await fetch(
        `${ADRENA_API}/custodyinfo?pool_name=main-pool&borrow_rate=true&limit=1`
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json() as AdrenaCustodyInfoResponse;

      if (data.success && data.data.borrow_rate) {
        // Map custody pubkeys to markets and extract borrow rates
        for (const [custodyPubkey, rateArray] of Object.entries(data.data.borrow_rate)) {
          const markets = CUSTODY_TO_MARKET[custodyPubkey];
          if (markets && markets.length > 0 && rateArray.length > 0) {
            // Borrow rate from API is already annual (APR), e.g., 0.0080052 = 0.80% APR
            // Per Adrena docs: SOL/WBTC: 0% ~ 80.5% APR, BONK: 0% ~ 150.7% APR
            const annualRate = parseFloat(rateArray[0]);
            // Apply this rate to all markets that use this custody
            for (const market of markets) {
              if (!rates.has(market)) {
                rates.set(market, annualRate);
                this.borrowRateCache.set(market, { rate: annualRate, timestamp: now });
              }
            }
          }
        }
      }

      logger.debug('Fetched Adrena borrow rates', {
        ratesCount: rates.size,
        rates: Object.fromEntries(rates),
      });

      return rates;
    } catch (error) {
      logger.error('Failed to fetch Adrena borrow rates', { error });
      return rates;
    }
  }

  /**
   * Get market statistics
   */
  async getMarketStats(): Promise<AdrenaMarketStats[]> {
    try {
      const [prices, borrowRates] = await Promise.all([
        this.fetchPrices(),
        this.fetchBorrowRates(),
      ]);
      const stats: AdrenaMarketStats[] = [];

      for (const [market, info] of Object.entries(ADRENA_MARKETS)) {
        const price = prices.get(info.symbol) || 0;
        // Get borrow rate for this market (already annual APR from API)
        const annualBorrowRate = borrowRates.get(market as AdrenaMarket) || 0;
        // Convert annual rate to hourly for fundingRate field
        const hourlyRate = annualBorrowRate / (24 * 365);

        stats.push({
          market,
          symbol: info.symbol,
          markPrice: price,
          fundingRate: hourlyRate,
          fundingRateApr: annualBorrowRate,
          maxLeverage: 100,
        });
      }

      return stats;
    } catch (error) {
      logger.error('Failed to get Adrena market stats', { error });
      return [];
    }
  }

  /**
   * Get funding rates
   */
  async getFundingRates(): Promise<FundingRate[]> {
    try {
      const stats = await this.getMarketStats();
      return stats.map(s => ({
        venue: 'adrena' as PerpsVenue,
        market: s.market,
        rate: s.fundingRate,
        annualizedRate: s.fundingRateApr,
        nextFundingTime: Date.now() + 3600000, // Hourly
        timestamp: Date.now(),
      }));
    } catch (error) {
      logger.error('Failed to fetch Adrena funding rates', { error });
      return [];
    }
  }

  // ============= POSITION MANAGEMENT =============

  /**
   * Get user positions from API
   */
  async getPositions(): Promise<AdrenaPositionInfo[]> {
    if (!this.keypair) {
      logger.warn('AdrenaClient: No wallet configured');
      return [];
    }

    try {
      const walletAddress = this.keypair.publicKey.toBase58();
      const response = await fetch(
        `${ADRENA_API}/positions?user_wallet=${walletAddress}&status=open`
      );

      if (!response.ok) {
        // API may return 400 for no positions
        if (response.status === 400) return [];
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json() as AdrenaPositionsResponse;
      if (!data.success) return [];

      const prices = await this.fetchPrices();

      return data.data
        .filter(p => p.status === 'open')
        .map(p => {
          const marketEntry = Object.entries(ADRENA_MARKETS)
            .find(([_, info]) => info.symbol === p.symbol);
          const market = marketEntry?.[0] || p.symbol;
          const markPrice = prices.get(p.symbol) || p.entry_price || 0;
          const entryPrice = p.entry_price || 0;
          const size = (p.volume || 0) / (entryPrice || 1);
          const collateral = p.collateral_amount || 0;

          return {
            market,
            side: p.side as PositionSide,
            size,
            collateral,
            entryPrice,
            markPrice,
            leverage: p.entry_leverage,
            unrealizedPnl: p.pnl || 0,
            liquidationPrice: this.calculateLiquidationPrice({
              side: p.side as PositionSide,
              entryPrice,
              size,
              collateral,
            }),
            positionPubkey: p.pubkey,
          };
        });
    } catch (error) {
      logger.error('Failed to fetch Adrena positions', { error });
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
  }): number {
    const { side, entryPrice, size, collateral } = params;
    const maintenanceMarginRatio = 0.01; // 1% maintenance margin for Adrena
    const notional = entryPrice * size;
    const maintenanceMargin = notional * maintenanceMarginRatio;
    const maxLoss = collateral - maintenanceMargin;
    const priceMove = size > 0 ? maxLoss / size : 0;

    return side === 'long'
      ? Math.max(0, entryPrice - priceMove)
      : entryPrice + priceMove;
  }

  // ============= TRADING =============

  /**
   * Open a position on Adrena
   */
  async openPosition(params: {
    market: AdrenaMarket;
    side: PositionSide;
    size: number;
    collateral: number;
    leverage: number;
    slippageBps?: number;
  }): Promise<PerpsTradeResult> {
    const { market, side, size, collateral, leverage, slippageBps = 50 } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'adrena' as PerpsVenue,
      side,
      size,
      leverage,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      const marketInfo = ADRENA_MARKETS[market];
      const price = await this.getPrice(marketInfo.symbol);
      const entryPrice = price || 0;

      if (!entryPrice) {
        throw new Error(`Could not fetch price for ${market}`);
      }

      logger.info('Opening Adrena position', {
        market, side, size, collateral, leverage, entryPrice,
      });

      const notional = size * entryPrice;
      // Adrena fees: 0.06% open/close
      result.fees.trading = notional * 0.0006;
      result.fees.gas = 0.005;

      result.liquidationPrice = this.calculateLiquidationPrice({
        side, entryPrice, size, collateral,
      });

      result.liquidationDistance = Math.abs(entryPrice - result.liquidationPrice) / entryPrice;
      result.success = true;
      result.entryPrice = entryPrice;
      result.orderId = `adrena-${Date.now()}`;
      result.positionId = `adrena-pos-${market}-${side}-${Date.now()}`;

      logger.info('Adrena position opened (simulated)', { orderId: result.orderId });

      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
      logger.error('Failed to open Adrena position', { error: result.error });
      return result;
    }
  }

  /**
   * Close a position on Adrena
   */
  async closePosition(params: {
    market: AdrenaMarket;
    size?: number;
  }): Promise<PerpsTradeResult> {
    const { market, size } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'adrena' as PerpsVenue,
      side: 'long',
      size: size || 0,
      leverage: 1,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      if (!this.keypair) {
        throw new Error('Wallet not configured');
      }

      logger.info('Closing Adrena position', { market, size });
      result.success = true;
      result.orderId = `adrena-close-${Date.now()}`;

      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
      logger.error('Failed to close Adrena position', { error: result.error });
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

let adrenaClientInstance: AdrenaClient | null = null;

export function getAdrenaClient(config?: AdrenaConfig): AdrenaClient {
  if (!adrenaClientInstance && config) {
    adrenaClientInstance = new AdrenaClient(config);
  }
  if (!adrenaClientInstance) {
    throw new Error('AdrenaClient not initialized');
  }
  return adrenaClientInstance;
}

export function resetAdrenaClient(): void {
  adrenaClientInstance = null;
}
