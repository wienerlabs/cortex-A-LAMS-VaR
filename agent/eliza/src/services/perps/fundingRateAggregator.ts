/**
 * Funding Rate Aggregator
 * 
 * Aggregates funding rates from all perpetual venues:
 * - Drift Protocol
 * - Jupiter Perps (borrow rates)
 * - Flash Trade
 * 
 * Provides:
 * - Real-time funding rate comparison
 * - Funding arbitrage opportunities
 * - Historical funding analysis
 */
import { logger } from '../logger.js';
import type { FundingRate, PerpsVenue } from '../../types/perps.js';
import { DriftClient } from './driftClient.js';
import { DriftProductionClient } from './driftClientProduction.js';
import { JupiterPerpsClient } from './jupiterPerpsClient.js';
import { FlashClient } from './flashClient.js';
import { FlashProductionClient } from './flashClientProduction.js';
import { AdrenaClient } from './adrenaClient.js';
import { AdrenaProductionClient } from './adrenaClientProduction.js';

// ============= TYPES =============

// Interface for any client that can provide funding rates
interface FundingRateProvider {
  getFundingRates(): Promise<FundingRate[]>;
}

// Union type for Drift clients (legacy or production)
type DriftClientType = DriftClient | DriftProductionClient;

// Union type for Flash clients (legacy or production)
type FlashClientType = FlashClient | FlashProductionClient;

// Union type for Adrena clients (legacy or production)
type AdrenaClientType = AdrenaClient | AdrenaProductionClient;

export interface FundingRateComparison {
  market: string;
  rates: {
    venue: PerpsVenue;
    rate: number;
    annualizedRate: number;
    nextFundingTime: number;
  }[];
  bestLongVenue: PerpsVenue | null;    // Lowest rate for longs
  bestShortVenue: PerpsVenue | null;   // Highest rate for shorts
  spreadOpportunity: number;            // Potential arb spread
}

export interface FundingArbitrageOpportunity {
  market: string;
  longVenue: PerpsVenue;
  shortVenue: PerpsVenue;
  longRate: number;
  shortRate: number;
  netSpread: number;
  annualizedSpread: number;
  estimatedProfitBps: number;
}

export interface AggregatedFundingData {
  timestamp: number;
  rates: FundingRate[];
  comparisons: FundingRateComparison[];
  arbitrageOpportunities: FundingArbitrageOpportunity[];
}

// ============= FUNDING RATE AGGREGATOR =============

export class FundingRateAggregator {
  private driftClient: DriftClientType | null = null;
  private jupiterClient: JupiterPerpsClient | null = null;
  private flashClient: FlashClientType | null = null;
  private adrenaClient: AdrenaClientType | null = null;

  private cachedRates: FundingRate[] = [];
  private lastFetchTime: number = 0;
  private cacheDurationMs: number = 60000; // 1 minute cache

  constructor() {
    logger.info('FundingRateAggregator initialized');
  }

  /**
   * Set venue clients - accepts both legacy and production clients
   */
  setClients(params: {
    drift?: DriftClientType;
    jupiter?: JupiterPerpsClient;
    flash?: FlashClientType;
    adrena?: AdrenaClientType;
  }): void {
    if (params.drift) this.driftClient = params.drift;
    if (params.jupiter) this.jupiterClient = params.jupiter;
    if (params.flash) this.flashClient = params.flash;
    if (params.adrena) this.adrenaClient = params.adrena;
  }

  /**
   * Fetch funding rates from all venues
   */
  async fetchAllFundingRates(): Promise<FundingRate[]> {
    const now = Date.now();
    
    // Return cached if fresh
    if (now - this.lastFetchTime < this.cacheDurationMs && this.cachedRates.length > 0) {
      return this.cachedRates;
    }

    const allRates: FundingRate[] = [];
    const fetchPromises: Promise<FundingRate[]>[] = [];

    // Fetch from Drift
    if (this.driftClient) {
      fetchPromises.push(
        this.driftClient.getFundingRates().catch(err => {
          logger.warn('Failed to fetch Drift funding rates', { error: err });
          return [];
        })
      );
    }

    // Fetch from Jupiter (borrow rates)
    if (this.jupiterClient) {
      fetchPromises.push(
        this.jupiterClient.getBorrowRates().catch(err => {
          logger.warn('Failed to fetch Jupiter borrow rates', { error: err });
          return [];
        })
      );
    }

    // Fetch from Flash
    if (this.flashClient) {
      fetchPromises.push(
        this.flashClient.getFundingRates().catch(err => {
          logger.warn('Failed to fetch Flash funding rates', { error: err });
          return [];
        })
      );
    }

    // Fetch from Adrena
    if (this.adrenaClient) {
      fetchPromises.push(
        this.adrenaClient.getFundingRates().catch(err => {
          logger.warn('Failed to fetch Adrena funding rates', { error: err });
          return [];
        })
      );
    }

    const results = await Promise.all(fetchPromises);
    for (const rates of results) {
      allRates.push(...rates);
    }

    this.cachedRates = allRates;
    this.lastFetchTime = now;

    logger.info('Fetched funding rates from all venues', {
      totalRates: allRates.length,
      venues: [...new Set(allRates.map(r => r.venue))],
    });

    return allRates;
  }

  /**
   * Compare funding rates across venues for a specific market
   */
  async compareFundingRates(market: string): Promise<FundingRateComparison> {
    const allRates = await this.fetchAllFundingRates();
    
    // Normalize market name for comparison
    const normalizedMarket = market.toUpperCase().replace('-', '');
    
    const marketRates = allRates.filter(r => {
      const normalized = r.market.toUpperCase().replace('-', '').replace('PERP', '');
      return normalized.includes(normalizedMarket.replace('PERP', ''));
    });

    const rates = marketRates.map(r => ({
      venue: r.venue,
      rate: r.rate,
      annualizedRate: r.annualizedRate,
      nextFundingTime: r.nextFundingTime,
    }));

    // Find best venues
    let bestLongVenue: PerpsVenue | null = null;
    let bestShortVenue: PerpsVenue | null = null;
    let lowestRate = Infinity;
    let highestRate = -Infinity;

    for (const r of rates) {
      if (r.rate < lowestRate) {
        lowestRate = r.rate;
        bestLongVenue = r.venue;
      }
      if (r.rate > highestRate) {
        highestRate = r.rate;
        bestShortVenue = r.venue;
      }
    }

    // Calculate spread opportunity
    const spreadOpportunity = rates.length >= 2 ? highestRate - lowestRate : 0;

    return {
      market,
      rates,
      bestLongVenue,
      bestShortVenue,
      spreadOpportunity,
    };
  }

  /**
   * Find funding rate arbitrage opportunities
   */
  async findArbitrageOpportunities(minSpreadBps: number = 10): Promise<FundingArbitrageOpportunity[]> {
    const allRates = await this.fetchAllFundingRates();
    const opportunities: FundingArbitrageOpportunity[] = [];

    // Group rates by normalized market
    const marketGroups = new Map<string, FundingRate[]>();

    for (const rate of allRates) {
      const normalized = rate.market.toUpperCase()
        .replace('-PERP', '')
        .replace('PERP', '')
        .replace('-', '');

      if (!marketGroups.has(normalized)) {
        marketGroups.set(normalized, []);
      }
      marketGroups.get(normalized)!.push(rate);
    }

    // Find opportunities in each market
    for (const [market, rates] of marketGroups) {
      if (rates.length < 2) continue;

      // Sort by rate
      rates.sort((a, b) => a.rate - b.rate);

      const lowest = rates[0];
      const highest = rates[rates.length - 1];

      const netSpread = highest.rate - lowest.rate;
      const spreadBps = netSpread * 10000;

      if (spreadBps >= minSpreadBps) {
        opportunities.push({
          market: `${market}-PERP`,
          longVenue: lowest.venue,
          shortVenue: highest.venue,
          longRate: lowest.rate,
          shortRate: highest.rate,
          netSpread,
          annualizedSpread: netSpread * 24 * 365,
          estimatedProfitBps: spreadBps,
        });
      }
    }

    // Sort by profit potential
    opportunities.sort((a, b) => b.estimatedProfitBps - a.estimatedProfitBps);

    logger.info('Found funding arbitrage opportunities', {
      count: opportunities.length,
      topOpportunity: opportunities[0]?.market,
      topSpreadBps: opportunities[0]?.estimatedProfitBps,
    });

    return opportunities;
  }

  /**
   * Get aggregated funding data for all markets
   */
  async getAggregatedData(): Promise<AggregatedFundingData> {
    const rates = await this.fetchAllFundingRates();

    // Get unique markets
    const markets = [...new Set(rates.map(r => r.market))];

    // Build comparisons for each market
    const comparisons: FundingRateComparison[] = [];
    for (const market of markets) {
      const comparison = await this.compareFundingRates(market);
      comparisons.push(comparison);
    }

    // Find arbitrage opportunities
    const arbitrageOpportunities = await this.findArbitrageOpportunities();

    return {
      timestamp: Date.now(),
      rates,
      comparisons,
      arbitrageOpportunities,
    };
  }

  /**
   * Get funding rate for specific venue and market
   */
  async getFundingRate(venue: PerpsVenue, market: string): Promise<FundingRate | null> {
    const allRates = await this.fetchAllFundingRates();

    const normalizedMarket = market.toUpperCase();

    return allRates.find(r =>
      r.venue === venue &&
      r.market.toUpperCase().includes(normalizedMarket.replace('-PERP', ''))
    ) || null;
  }

  /**
   * Get average funding rate across all venues for a market
   */
  async getAverageFundingRate(market: string): Promise<{
    averageRate: number;
    averageAnnualized: number;
    venueCount: number;
  }> {
    const comparison = await this.compareFundingRates(market);

    if (comparison.rates.length === 0) {
      return { averageRate: 0, averageAnnualized: 0, venueCount: 0 };
    }

    const totalRate = comparison.rates.reduce((sum, r) => sum + r.rate, 0);
    const totalAnnualized = comparison.rates.reduce((sum, r) => sum + r.annualizedRate, 0);

    return {
      averageRate: totalRate / comparison.rates.length,
      averageAnnualized: totalAnnualized / comparison.rates.length,
      venueCount: comparison.rates.length,
    };
  }

  /**
   * Clear cached rates
   */
  clearCache(): void {
    this.cachedRates = [];
    this.lastFetchTime = 0;
  }
}

// ============= SINGLETON =============

let aggregatorInstance: FundingRateAggregator | null = null;

export function getFundingRateAggregator(): FundingRateAggregator {
  if (!aggregatorInstance) {
    aggregatorInstance = new FundingRateAggregator();
  }
  return aggregatorInstance;
}

export function resetFundingRateAggregator(): void {
  aggregatorInstance = null;
}
