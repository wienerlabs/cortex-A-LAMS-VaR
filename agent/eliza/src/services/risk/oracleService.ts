import { Connection, PublicKey } from '@solana/web3.js';
import { logger } from '../logger.js';
import { getSolanaConnection } from '../solana/connection.js';
import type { OracleConfig, OracleStatus } from './types.js';

// ============= ORACLE CONSTANTS =============

// Pyth Price Feed IDs (Solana mainnet on-chain accounts — kept for reference)
export const PYTH_PRICE_FEEDS = {
  SOL_USD: 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
  BTC_USD: 'GVXRSBjFk6e6J3NbVPXohDJetcTjaeeuykUpbQF8UoMU',
  ETH_USD: 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
  BONK_USD: '8ihFLu5FimgTQ1Unh4dVyEHUGodJ5gJQCrQf4KUVB9bN',
  JUP_USD: 'g6eRCbboSwK4tSWngn773RCMexr1APQr4uA9bGZBYfo',
} as const;

const PYTH_HERMES_URL = 'https://hermes.pyth.network';

// Birdeye API for real-time prices
const BIRDEYE_API = 'https://public-api.birdeye.so';

// Jupiter Price API
const JUPITER_PRICE_API = 'https://api.jup.ag/price/v2';

// Token mints for Jupiter
const TOKEN_MINTS: Record<string, string> = {
  SOL: 'So11111111111111111111111111111111111111112',
  USDC: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  BTC: '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh',
  ETH: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',
  BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
  JUP: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
};

export const DEFAULT_ORACLE_CONFIG: OracleConfig = {
  maxStalenessSeconds: 30,        // Reject trades if >30s stale
  emergencyExitSeconds: 60,       // Emergency exit if >60s stale
  requiredConfirmations: 1,
};

// ============= ORACLE SERVICE CLASS =============

export class OracleService {
  private connection: Connection;
  private config: OracleConfig;
  private priceCache: Map<string, OracleStatus> = new Map();
  private birdeyeApiKey?: string;
  private pythFeedIdCache: Map<string, string> = new Map();

  constructor(
    rpcUrl?: string,
    config: Partial<OracleConfig> = {},
    birdeyeApiKey?: string
  ) {
    this.connection = rpcUrl
      ? new Connection(rpcUrl, 'confirmed')
      : getSolanaConnection();
    this.config = { ...DEFAULT_ORACLE_CONFIG, ...config };
    this.birdeyeApiKey = birdeyeApiKey || process.env.BIRDEYE_API_KEY;
  }

  /**
   * Resolve token symbol to Pyth Hermes feed ID.
   * Same approach as solana-agent-kit's fetchPythPriceFeedID but cached.
   */
  private async resolvePythFeedId(symbol: string): Promise<string> {
    const upper = symbol.toUpperCase();
    const cached = this.pythFeedIdCache.get(upper);
    if (cached) return cached;

    const res = await fetch(
      `${PYTH_HERMES_URL}/v2/price_feeds?query=${upper}&asset_type=crypto`
    );
    if (!res.ok) throw new Error(`Pyth feed lookup HTTP ${res.status}`);

    const feeds = (await res.json()) as Array<{
      id: string;
      attributes: { base: string };
    }>;
    if (feeds.length === 0) throw new Error(`No Pyth feed for ${upper}`);

    // Exact match when multiple feeds returned (e.g. "SOL" vs "WSOL")
    const exact = feeds.find(
      (f) => f.attributes.base.toUpperCase() === upper
    );
    const feedId = exact ? exact.id : feeds[0].id;
    this.pythFeedIdCache.set(upper, feedId);
    return feedId;
  }

  /**
   * Get price from Pyth via Hermes API.
   * Returns price, confidence interval, and publish timestamp.
   */
  async getPythPrice(symbol: string): Promise<OracleStatus> {
    try {
      const feedId = await this.resolvePythFeedId(symbol);

      const res = await fetch(
        `${PYTH_HERMES_URL}/v2/updates/price/latest?ids[]=${feedId}`
      );
      if (!res.ok) throw new Error(`Pyth price HTTP ${res.status}`);

      const data = (await res.json()) as {
        parsed: Array<{
          price: { price: string; expo: number; conf: string; publish_time: number };
          ema_price: { price: string; expo: number; conf: string };
        }>;
      };

      if (!data.parsed?.length) throw new Error(`No Pyth data for ${symbol}`);

      const entry = data.parsed[0];
      const expo = entry.price.expo;
      const factor = Math.pow(10, expo);

      const price = Number(entry.price.price) * factor;
      const conf = Number(entry.price.conf) * factor;

      const timestamp = new Date(entry.price.publish_time * 1000);
      const stalenessSeconds = (Date.now() - timestamp.getTime()) / 1000;

      // confidence as ratio of price — lower is better (tighter spread)
      const confidenceRatio = price > 0 ? conf / price : 1;

      logger.info('[Oracle] Pyth price', {
        symbol,
        price: price.toFixed(4),
        confidence: conf.toFixed(6),
        confidenceRatio: (confidenceRatio * 100).toFixed(4) + '%',
        stalenessSeconds: stalenessSeconds.toFixed(1),
      });

      const status: OracleStatus = {
        source: 'pyth',
        price,
        timestamp,
        stalenessSeconds,
        isStale: stalenessSeconds > this.config.maxStalenessSeconds,
        isEmergency: stalenessSeconds > this.config.emergencyExitSeconds,
        confidence: confidenceRatio,
      };

      this.priceCache.set(`${symbol}-pyth`, status);
      return status;
    } catch (error) {
      logger.error('Pyth price fetch failed', { symbol, error });
      throw error;
    }
  }

  /**
   * Get price from Jupiter Price API with timestamp
   * Jupiter aggregates across DEXes - real on-chain prices
   */
  async getJupiterPrice(symbol: string): Promise<OracleStatus> {
    const mint = TOKEN_MINTS[symbol.toUpperCase()];
    if (!mint) {
      throw new Error(`Unknown token: ${symbol}`);
    }

    try {
      const response = await fetch(`${JUPITER_PRICE_API}?ids=${mint}`);
      const data = await response.json() as {
        data?: Record<string, { price?: number; extraInfo?: { lastSwappedPrice?: { lastJupiterSellAt?: number } } }>
      };
      
      const priceData = data.data?.[mint];
      if (!priceData?.price) {
        throw new Error(`No price data for ${symbol}`);
      }

      // Jupiter includes last swap timestamp
      const lastSwapTime = priceData.extraInfo?.lastSwappedPrice?.lastJupiterSellAt;
      const timestamp = lastSwapTime ? new Date(lastSwapTime * 1000) : new Date();
      const stalenessSeconds = (Date.now() - timestamp.getTime()) / 1000;

      const status: OracleStatus = {
        source: 'jupiter',
        price: priceData.price,
        timestamp,
        stalenessSeconds,
        isStale: stalenessSeconds > this.config.maxStalenessSeconds,
        isEmergency: stalenessSeconds > this.config.emergencyExitSeconds,
      };

      this.priceCache.set(`${symbol}-jupiter`, status);
      return status;
    } catch (error) {
      logger.error('Jupiter price fetch failed', { symbol, error });
      throw error;
    }
  }

  /**
   * Get price from Birdeye with on-chain verification
   */
  async getBirdeyePrice(symbol: string): Promise<OracleStatus> {
    const mint = TOKEN_MINTS[symbol.toUpperCase()];
    if (!mint) {
      throw new Error(`Unknown token: ${symbol}`);
    }

    if (!this.birdeyeApiKey) {
      throw new Error('Birdeye API key not configured');
    }

    try {
      const response = await fetch(`${BIRDEYE_API}/defi/price?address=${mint}`, {
        headers: {
          'X-API-KEY': this.birdeyeApiKey,
          'x-chain': 'solana',
        },
      });
      const data = await response.json() as {
        success?: boolean;
        data?: { value?: number; updateUnixTime?: number };
      };
      
      if (!data.success || !data.data?.value) {
        throw new Error(`Birdeye returned no data for ${symbol}`);
      }

      const updateTime = data.data.updateUnixTime || Date.now() / 1000;
      const timestamp = new Date(updateTime * 1000);
      const stalenessSeconds = (Date.now() - timestamp.getTime()) / 1000;

      const status: OracleStatus = {
        source: 'birdeye',
        price: data.data.value,
        timestamp,
        stalenessSeconds,
        isStale: stalenessSeconds > this.config.maxStalenessSeconds,
        isEmergency: stalenessSeconds > this.config.emergencyExitSeconds,
      };

      this.priceCache.set(`${symbol}-birdeye`, status);
      return status;
    } catch (error) {
      logger.error('Birdeye price fetch failed', { symbol, error });
      throw error;
    }
  }

  /**
   * Get aggregated price from multiple sources
   * Returns median price and worst staleness
   */
  async getAggregatedPrice(symbol: string): Promise<OracleStatus> {
    const statuses: OracleStatus[] = [];

    // Try Pyth first (on-chain oracle with confidence intervals)
    try {
      statuses.push(await this.getPythPrice(symbol));
    } catch { /* continue to fallbacks */ }

    // Try Jupiter (most reliable for Solana DEX prices)
    try {
      statuses.push(await this.getJupiterPrice(symbol));
    } catch { /* continue */ }

    // Try Birdeye if available
    if (this.birdeyeApiKey) {
      try {
        statuses.push(await this.getBirdeyePrice(symbol));
      } catch { /* continue */ }
    }

    if (statuses.length === 0) {
      throw new Error(`No oracle data available for ${symbol}`);
    }

    // Use median price for safety
    const prices = statuses.map(s => s.price).sort((a, b) => a - b);
    const medianPrice = prices[Math.floor(prices.length / 2)];

    // Use worst staleness (most conservative)
    const worstStaleness = Math.max(...statuses.map(s => s.stalenessSeconds));
    const oldestTimestamp = new Date(Math.min(...statuses.map(s => s.timestamp.getTime())));

    // Carry Pyth confidence through if available
    const pythStatus = statuses.find(s => s.source === 'pyth');

    return {
      source: 'aggregated',
      price: medianPrice,
      timestamp: oldestTimestamp,
      stalenessSeconds: worstStaleness,
      isStale: worstStaleness > this.config.maxStalenessSeconds,
      isEmergency: worstStaleness > this.config.emergencyExitSeconds,
      confidence: pythStatus?.confidence ?? 1 / statuses.length,
    };
  }
}

