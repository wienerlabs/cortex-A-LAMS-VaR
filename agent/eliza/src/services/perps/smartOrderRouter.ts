/**
 * Smart Order Router (SOR) for Multi-Venue Perpetual Futures Trading
 * 
 * Routes orders to the best venue based on:
 * - Price comparison across venues
 * - Fee normalization (trading fees, funding, gas)
 * - Slippage estimation based on liquidity
 * - Venue reliability scoring
 * 
 * Supported venues: Drift, Adrena
 */
import { logger } from '../logger.js';
import type { PerpsVenue, PositionSide, FundingRate } from '../../types/perps.js';
import { DriftClient } from './driftClient.js';
import { AdrenaClient } from './adrenaClient.js';

// ============= TYPES =============

/** Quote from a single venue */
export interface VenueQuote {
  venue: PerpsVenue;
  market: string;
  side: PositionSide;
  
  // Price data
  price: number;
  timestamp: number;
  
  // Fee breakdown (in basis points)
  fees: {
    tradingFeeBps: number;      // Opening/closing fee
    fundingRateBps: number;     // Current hourly funding * 24 (daily cost)
    borrowRateBps: number;      // Borrow rate for shorts (Adrena)
    gasFeeBps: number;          // Estimated gas as % of trade
  };
  
  // Slippage estimation
  estimatedSlippageBps: number;
  liquidityDepth: number;       // Available liquidity in USD
  
  // Reliability metrics
  latencyMs: number;
  isAvailable: boolean;
  error?: string;
}

/** Scored route for comparison */
export interface RouteScore {
  venue: PerpsVenue;
  quote: VenueQuote;
  
  // Cost breakdown
  totalCostBps: number;         // price impact + fees + slippage
  priceImpactBps: number;       // vs index/oracle price
  feeCostBps: number;           // normalized total fees
  slippageCostBps: number;      // estimated slippage
  
  // Reliability score (0-1, higher is better)
  reliabilityScore: number;
  
  // Final weighted score (lower is better)
  finalScore: number;
  
  // Comparison
  savingsVsWorstBps: number;
}

/** Selected route result */
export interface SelectedRoute {
  selectedVenue: PerpsVenue;
  expectedPrice: number;
  estimatedCostBps: number;
  savingsVsWorstBps: number;
  
  // All evaluated routes
  allRoutes: RouteScore[];
  
  // Routing decision
  reason: string;
  fallbackVenue?: PerpsVenue;
  
  // Execution info
  canExecute: boolean;
  executionVenue: PerpsVenue;  // May differ if selected venue is read-only
}

/** SOR configuration */
export interface SORConfig {
  // Timeouts
  quoteTimeoutMs: number;       // Max time to wait for quotes (default: 2000)
  cacheTtlMs: number;           // Price cache TTL (default: 5000)
  
  // Venue-specific fee configs (in basis points)
  driftFees: {
    takerFeeBps: number;        // 10 bps = 0.1%
    makerFeeBps: number;        // 5 bps = 0.05%
    gasFeeBps: number;          // Estimated gas as bps
  };
  adreneFees: {
    openFeeBps: number;         // 10 bps = 0.1%
    closeFeeBps: number;        // 16 bps = 0.16%
    borrowRateBps: number;      // Variable, typically 0-5 bps/hour
    gasFeeBps: number;
  };
  
  // Slippage impact factors
  driftImpactFactor: number;    // 0.001 = 0.1%
  adrenaImpactFactor: number;   // 0 (zero slippage claim)
  
  // Route selection
  priceDifferenceThresholdBps: number;  // If diff < this, prefer Drift (50 = 0.5%)
  preferredVenue: PerpsVenue;           // Default preference when close
  
  // Reliability tracking
  reliabilityWindowSize: number;        // Track last N requests (default: 100)
  errorRatePenaltyThreshold: number;    // Penalize if error rate > this (0.05 = 5%)
}

export const DEFAULT_SOR_CONFIG: SORConfig = {
  quoteTimeoutMs: 2000,
  cacheTtlMs: 5000,
  
  driftFees: {
    takerFeeBps: 10,
    makerFeeBps: 5,
    gasFeeBps: 1,
  },
  adreneFees: {
    openFeeBps: 10,
    closeFeeBps: 16,
    borrowRateBps: 2,
    gasFeeBps: 1,
  },
  
  driftImpactFactor: 0.001,
  adrenaImpactFactor: 0,
  
  priceDifferenceThresholdBps: 50,
  preferredVenue: 'drift',
  
  reliabilityWindowSize: 100,
  errorRatePenaltyThreshold: 0.05,
};

// ============= RELIABILITY TRACKER =============

interface RequestRecord {
  timestamp: number;
  success: boolean;
  latencyMs: number;
}

class VenueReliabilityTracker {
  private records: Map<PerpsVenue, RequestRecord[]> = new Map();
  private windowSize: number;
  
  constructor(windowSize: number = 100) {
    this.windowSize = windowSize;
  }
  
  record(venue: PerpsVenue, success: boolean, latencyMs: number): void {
    const records = this.records.get(venue) || [];
    records.push({ timestamp: Date.now(), success, latencyMs });

    // Keep only last N records
    if (records.length > this.windowSize) {
      records.shift();
    }

    this.records.set(venue, records);
  }

  getStats(venue: PerpsVenue): {
    successRate: number;
    avgLatencyMs: number;
    requestCount: number;
  } {
    const records = this.records.get(venue) || [];
    if (records.length === 0) {
      return { successRate: 1, avgLatencyMs: 0, requestCount: 0 };
    }

    const successCount = records.filter(r => r.success).length;
    const totalLatency = records.reduce((sum, r) => sum + r.latencyMs, 0);

    return {
      successRate: successCount / records.length,
      avgLatencyMs: totalLatency / records.length,
      requestCount: records.length,
    };
  }

  getReliabilityScore(venue: PerpsVenue, errorThreshold: number): number {
    const stats = this.getStats(venue);

    // Start with base score of 1.0
    let score = 1.0;

    // Penalize for error rate above threshold
    if (stats.successRate < (1 - errorThreshold)) {
      score *= stats.successRate;
    }

    // Slight penalty for high latency (>1000ms)
    if (stats.avgLatencyMs > 1000) {
      score *= 0.95;
    }

    return score;
  }
}

// ============= SMART ORDER ROUTER =============

/** Singleton instance */
let sorInstance: SmartOrderRouter | null = null;

export class SmartOrderRouter {
  private config: SORConfig;
  private driftClient: DriftClient | null = null;
  private adrenaClient: AdrenaClient | null = null;
  private reliabilityTracker: VenueReliabilityTracker;

  // Price cache
  private priceCache: Map<string, { price: number; timestamp: number }> = new Map();

  constructor(config: Partial<SORConfig> = {}) {
    this.config = { ...DEFAULT_SOR_CONFIG, ...config };
    this.reliabilityTracker = new VenueReliabilityTracker(this.config.reliabilityWindowSize);

    logger.info('SmartOrderRouter initialized', {
      quoteTimeoutMs: this.config.quoteTimeoutMs,
      cacheTtlMs: this.config.cacheTtlMs,
      preferredVenue: this.config.preferredVenue,
    });
  }

  /**
   * Set venue clients for quote fetching
   */
  setClients(drift: DriftClient | null, adrena: AdrenaClient | null): void {
    this.driftClient = drift;
    this.adrenaClient = adrena;
    logger.debug('SOR clients configured', {
      driftAvailable: !!drift,
      adrenaAvailable: !!adrena,
    });
  }

  /**
   * Fetch quotes from all venues in parallel with timeout
   */
  async fetchQuotes(
    market: string,
    side: PositionSide,
    sizeUsd: number
  ): Promise<VenueQuote[]> {
    const quotes: VenueQuote[] = [];
    const startTime = Date.now();

    // Create quote fetch promises with timeout
    const fetchPromises: Promise<VenueQuote | null>[] = [];

    if (this.driftClient) {
      fetchPromises.push(
        this.fetchDriftQuote(market, side, sizeUsd)
          .catch(e => this.handleQuoteError('drift', market, side, e))
      );
    }

    if (this.adrenaClient) {
      fetchPromises.push(
        this.fetchAdrenaQuote(market, side, sizeUsd)
          .catch(e => this.handleQuoteError('adrena', market, side, e))
      );
    }

    // Wait for all with timeout
    const timeoutPromise = new Promise<null>(resolve =>
      setTimeout(() => resolve(null), this.config.quoteTimeoutMs)
    );

    const results = await Promise.all(
      fetchPromises.map(p => Promise.race([p, timeoutPromise]))
    );

    for (const result of results) {
      if (result) {
        quotes.push(result);
      }
    }

    logger.debug('Fetched quotes', {
      market,
      side,
      sizeUsd,
      quoteCount: quotes.length,
      durationMs: Date.now() - startTime,
    });

    return quotes;
  }

  /**
   * Fetch quote from Drift
   */
  private async fetchDriftQuote(
    market: string,
    side: PositionSide,
    sizeUsd: number
  ): Promise<VenueQuote> {
    const startTime = Date.now();

    try {
      // Get market info from Drift
      const markets = await this.driftClient!.getMarkets();
      const marketInfo = markets.find(m => m.symbol === market);

      if (!marketInfo) {
        throw new Error(`Market ${market} not found on Drift`);
      }

      const price = marketInfo.markPrice;
      const latencyMs = Date.now() - startTime;

      // Calculate fees
      const tradingFeeBps = this.config.driftFees.takerFeeBps;
      const fundingRateBps = Math.abs(marketInfo.fundingRate) * 24 * 10000; // Convert to daily bps

      // Estimate slippage based on liquidity
      const totalOI = marketInfo.openInterestLong + marketInfo.openInterestShort;
      const liquidityDepth = totalOI * price;
      const slippageBps = this.calculateSlippage(
        sizeUsd,
        liquidityDepth,
        this.config.driftImpactFactor
      );

      // Record success
      this.reliabilityTracker.record('drift', true, latencyMs);

      return {
        venue: 'drift',
        market,
        side,
        price,
        timestamp: Date.now(),
        fees: {
          tradingFeeBps,
          fundingRateBps,
          borrowRateBps: 0,
          gasFeeBps: this.config.driftFees.gasFeeBps,
        },
        estimatedSlippageBps: slippageBps,
        liquidityDepth,
        latencyMs,
        isAvailable: true,
      };
    } catch (error) {
      const latencyMs = Date.now() - startTime;
      this.reliabilityTracker.record('drift', false, latencyMs);
      throw error;
    }
  }

  /**
   * Fetch quote from Adrena
   */
  private async fetchAdrenaQuote(
    market: string,
    side: PositionSide,
    sizeUsd: number
  ): Promise<VenueQuote> {
    const startTime = Date.now();

    try {
      // Get market stats from Adrena
      const stats = await this.adrenaClient!.getMarketStats();
      const marketStats = stats.find(s => s.market === market);

      if (!marketStats) {
        throw new Error(`Market ${market} not found on Adrena`);
      }

      const price = marketStats.markPrice;
      const latencyMs = Date.now() - startTime;

      // Adrena fees: opening fee + borrow rate (for shorts)
      const tradingFeeBps = this.config.adreneFees.openFeeBps;
      const fundingRateBps = Math.abs(marketStats.fundingRate) * 24 * 10000;
      const borrowRateBps = side === 'short' ? this.config.adreneFees.borrowRateBps : 0;

      // Adrena claims zero slippage
      const slippageBps = this.calculateSlippage(
        sizeUsd,
        Infinity, // Adrena uses LP-to-trader model
        this.config.adrenaImpactFactor
      );

      // Record success
      this.reliabilityTracker.record('adrena', true, latencyMs);

      return {
        venue: 'adrena',
        market,
        side,
        price,
        timestamp: Date.now(),
        fees: {
          tradingFeeBps,
          fundingRateBps,
          borrowRateBps,
          gasFeeBps: this.config.adreneFees.gasFeeBps,
        },
        estimatedSlippageBps: slippageBps,
        liquidityDepth: Infinity, // LP-to-trader model
        latencyMs,
        isAvailable: true,
      };
    } catch (error) {
      const latencyMs = Date.now() - startTime;
      this.reliabilityTracker.record('adrena', false, latencyMs);
      throw error;
    }
  }

  /**
   * Handle quote fetch error and return unavailable quote
   */
  private handleQuoteError(
    venue: PerpsVenue,
    market: string,
    side: PositionSide,
    error: unknown
  ): VenueQuote {
    const errorMsg = error instanceof Error ? error.message : String(error);
    logger.warn(`Quote fetch failed for ${venue}`, { market, error: errorMsg });

    return {
      venue,
      market,
      side,
      price: 0,
      timestamp: Date.now(),
      fees: {
        tradingFeeBps: 0,
        fundingRateBps: 0,
        borrowRateBps: 0,
        gasFeeBps: 0,
      },
      estimatedSlippageBps: 0,
      liquidityDepth: 0,
      latencyMs: this.config.quoteTimeoutMs,
      isAvailable: false,
      error: errorMsg,
    };
  }

  /**
   * Calculate slippage based on order size and liquidity
   * Formula: slippage = (orderSize / availableLiquidity) * impactFactor
   */
  private calculateSlippage(
    orderSizeUsd: number,
    liquidityDepth: number,
    impactFactor: number
  ): number {
    if (liquidityDepth === 0 || liquidityDepth === Infinity) {
      return 0;
    }

    // Slippage in bps
    const slippage = (orderSizeUsd / liquidityDepth) * impactFactor * 10000;
    return Math.min(slippage, 500); // Cap at 5%
  }

  /**
   * Score routes for comparison
   */
  scoreRoutes(quotes: VenueQuote[], indexPrice?: number): RouteScore[] {
    const availableQuotes = quotes.filter(q => q.isAvailable);

    if (availableQuotes.length === 0) {
      return [];
    }

    // Use first available price as reference if no index price
    const refPrice = indexPrice || availableQuotes[0].price;

    const scores: RouteScore[] = availableQuotes.map(quote => {
      // Price impact vs reference
      const priceImpactBps = Math.abs((quote.price - refPrice) / refPrice) * 10000;

      // Total fees
      const feeCostBps =
        quote.fees.tradingFeeBps +
        quote.fees.fundingRateBps +
        quote.fees.borrowRateBps +
        quote.fees.gasFeeBps;

      // Slippage
      const slippageCostBps = quote.estimatedSlippageBps;

      // Total cost
      const totalCostBps = priceImpactBps + feeCostBps + slippageCostBps;

      // Reliability score
      const reliabilityScore = this.reliabilityTracker.getReliabilityScore(
        quote.venue,
        this.config.errorRatePenaltyThreshold
      );

      // Final score: lower is better
      // Divide by reliability to penalize unreliable venues
      const finalScore = totalCostBps / reliabilityScore;

      return {
        venue: quote.venue,
        quote,
        totalCostBps,
        priceImpactBps,
        feeCostBps,
        slippageCostBps,
        reliabilityScore,
        finalScore,
        savingsVsWorstBps: 0, // Calculated after
      };
    });

    // Sort by final score (lower is better)
    scores.sort((a, b) => a.finalScore - b.finalScore);

    // Calculate savings vs worst
    if (scores.length > 1) {
      const worstScore = scores[scores.length - 1].totalCostBps;
      for (const score of scores) {
        score.savingsVsWorstBps = worstScore - score.totalCostBps;
      }
    }

    return scores;
  }

  /**
   * Select the best route for execution
   */
  async selectBestRoute(
    market: string,
    side: PositionSide,
    sizeUsd: number
  ): Promise<SelectedRoute> {
    logger.info('Selecting best route', { market, side, sizeUsd });

    // Fetch quotes from all venues
    const quotes = await this.fetchQuotes(market, side, sizeUsd);

    // Handle no available quotes
    if (quotes.filter(q => q.isAvailable).length === 0) {
      logger.error('No venues available for quote', { market });
      return {
        selectedVenue: 'drift',
        expectedPrice: 0,
        estimatedCostBps: 0,
        savingsVsWorstBps: 0,
        allRoutes: [],
        reason: 'No venues available - all quote fetches failed',
        canExecute: false,
        executionVenue: 'drift',
      };
    }

    // Score all routes
    const scoredRoutes = this.scoreRoutes(quotes);

    if (scoredRoutes.length === 0) {
      return {
        selectedVenue: 'drift',
        expectedPrice: 0,
        estimatedCostBps: 0,
        savingsVsWorstBps: 0,
        allRoutes: [],
        reason: 'No valid routes after scoring',
        canExecute: false,
        executionVenue: 'drift',
      };
    }

    // Get best and second best
    const best = scoredRoutes[0];
    const secondBest = scoredRoutes.length > 1 ? scoredRoutes[1] : null;

    // Check if price difference is small enough to prefer Drift
    let selectedVenue = best.venue;
    let reason = `Best score: ${best.finalScore.toFixed(2)} bps`;

    if (secondBest && best.venue !== this.config.preferredVenue) {
      const priceDiffBps = Math.abs(best.totalCostBps - secondBest.totalCostBps);

      if (priceDiffBps < this.config.priceDifferenceThresholdBps &&
          secondBest.venue === this.config.preferredVenue) {
        // Switch to preferred venue if difference is small
        selectedVenue = this.config.preferredVenue;
        reason = `Price diff ${priceDiffBps.toFixed(1)}bps < ${this.config.priceDifferenceThresholdBps}bps threshold, prefer ${this.config.preferredVenue}`;
      }
    }

    // Determine execution venue
    // Adrena is read-only for now, fallback to Drift
    let executionVenue = selectedVenue;
    let canExecute = true;
    let fallbackVenue: PerpsVenue | undefined;

    if (selectedVenue === 'adrena') {
      // Adrena doesn't have execution yet, use Drift as fallback
      executionVenue = 'drift';
      fallbackVenue = 'drift';
      reason += ' (Adrena read-only, executing on Drift)';

      // Check if Drift quote is available
      const driftQuote = quotes.find(q => q.venue === 'drift' && q.isAvailable);
      if (!driftQuote) {
        canExecute = false;
        reason += ' - Drift unavailable for fallback!';
      }
    }

    const selectedRoute = scoredRoutes.find(r => r.venue === selectedVenue) || best;

    const result: SelectedRoute = {
      selectedVenue,
      expectedPrice: selectedRoute.quote.price,
      estimatedCostBps: selectedRoute.totalCostBps,
      savingsVsWorstBps: selectedRoute.savingsVsWorstBps,
      allRoutes: scoredRoutes,
      reason,
      fallbackVenue,
      canExecute,
      executionVenue,
    };

    // Log routing decision
    logger.info('Route selected', {
      selectedVenue,
      executionVenue,
      expectedPrice: selectedRoute.quote.price,
      totalCostBps: selectedRoute.totalCostBps,
      savingsVsWorstBps: selectedRoute.savingsVsWorstBps,
      reason,
      allVenues: scoredRoutes.map(r => ({
        venue: r.venue,
        price: r.quote.price,
        costBps: r.totalCostBps,
        score: r.finalScore,
      })),
    });

    return result;
  }

  /**
   * Get venue reliability stats for monitoring
   */
  getVenueStats(): Record<PerpsVenue, {
    successRate: number;
    avgLatencyMs: number;
    requestCount: number;
    reliabilityScore: number;
  }> {
    const venues: PerpsVenue[] = ['drift', 'adrena', 'jupiter', 'flash'];
    const stats: Record<string, any> = {};

    for (const venue of venues) {
      const baseStats = this.reliabilityTracker.getStats(venue);
      stats[venue] = {
        ...baseStats,
        reliabilityScore: this.reliabilityTracker.getReliabilityScore(
          venue,
          this.config.errorRatePenaltyThreshold
        ),
      };
    }

    return stats as Record<PerpsVenue, any>;
  }

  /**
   * Clear price cache
   */
  clearCache(): void {
    this.priceCache.clear();
    logger.debug('SOR price cache cleared');
  }
}

// ============= SINGLETON =============

export function getSmartOrderRouter(config?: Partial<SORConfig>): SmartOrderRouter {
  if (!sorInstance) {
    sorInstance = new SmartOrderRouter(config);
  }
  return sorInstance;
}

export function resetSmartOrderRouter(): void {
  sorInstance = null;
}