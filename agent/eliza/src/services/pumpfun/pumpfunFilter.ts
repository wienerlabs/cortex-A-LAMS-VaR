/**
 * Pump.fun Token Filter
 * Filters memecoin tokens based on safety criteria
 */

import { logger } from '../logger.js';
import type { PumpFunToken } from './pumpfunClient.js';

export interface FilteredPumpFunToken {
  mint: string;
  symbol: string;
  name: string;
  tvl: number;
  holderCount: number;
  ageHours: number;
  marketCap: number;
  creator: string;
  complete: boolean;
  twitter?: string;
  telegram?: string;
  website?: string;
}

export interface PumpFunFilterCriteria {
  minTvl: number;          // Minimum TVL in USD (default: $10,000)
  minHolders: number;      // Minimum holder count (default: 50)
  minAgeHours: number;     // Minimum token age in hours (default: 24)
  maxTopHolderPct: number; // Maximum top holder percentage (default: 50%)
}

export const DEFAULT_PUMPFUN_FILTER: PumpFunFilterCriteria = {
  minTvl: 10_000,        // $10K minimum TVL
  minHolders: 50,        // 50+ holders
  minAgeHours: 24,       // 24+ hours old
  maxTopHolderPct: 50,   // Top holder < 50%
};

/**
 * Get current SOL price from environment or default
 */
function getSolPrice(): number {
  // In production, this should come from a price oracle
  // For now, use a reasonable default or env variable
  const envPrice = process.env.SOL_PRICE;
  return envPrice ? parseFloat(envPrice) : 150; // Default $150
}

/**
 * Filter Pump.fun tokens based on safety criteria
 */
export async function filterPumpFunTokens(
  tokens: PumpFunToken[],
  criteria: PumpFunFilterCriteria = DEFAULT_PUMPFUN_FILTER
): Promise<FilteredPumpFunToken[]> {
  const filtered: FilteredPumpFunToken[] = [];
  const solPrice = getSolPrice();
  
  logger.info('[PumpFunFilter] Starting filter', {
    totalTokens: tokens.length,
    criteria,
    solPrice,
  });

  let filteredByTvl = 0;
  let filteredByAge = 0;
  let filteredByHolders = 0;
  let filteredByConcentration = 0;

  for (const token of tokens) {
    try {
      // Use market cap as TVL (already in USD)
      const tvl = token.marketCap;

      if (tvl < criteria.minTvl) {
        filteredByTvl++;
        continue;
      }

      // Check token age
      const ageMs = Date.now() - token.creationTime;
      const ageHours = ageMs / (1000 * 60 * 60);

      if (ageHours < criteria.minAgeHours) {
        filteredByAge++;
        continue;
      }

      // Check holder count (already provided by API)
      const holderCount = token.numHolders;
      if (holderCount < criteria.minHolders) {
        filteredByHolders++;
        continue;
      }

      // Check top holder concentration (already provided by API)
      if (token.topHoldersPercentage > criteria.maxTopHolderPct) {
        filteredByConcentration++;
        continue;
      }

      // Token passed all filters
      filtered.push({
        mint: token.coinMint,
        symbol: token.ticker,
        name: token.name,
        tvl,
        holderCount,
        ageHours,
        marketCap: token.marketCap,
        creator: token.dev,
        complete: token.graduationDate !== null,
        twitter: token.twitter,
        telegram: token.telegram,
        website: token.website,
      });

      logger.info('[PumpFunFilter] Token passed', {
        symbol: token.ticker,
        tvl: `$${(tvl / 1000).toFixed(1)}K`,
        holders: holderCount,
        age: `${ageHours.toFixed(1)}h`,
      });

    } catch (error: any) {
      logger.error('[PumpFunFilter] Error filtering token', {
        mint: token.coinMint,
        symbol: token.ticker,
        error: error.message,
      });
    }
  }

  logger.info('[PumpFunFilter] Filtering complete', {
    total: tokens.length,
    passed: filtered.length,
    rejected: tokens.length - filtered.length,
    reasons: {
      tvl: filteredByTvl,
      age: filteredByAge,
      holders: filteredByHolders,
      concentration: filteredByConcentration,
    },
  });

  return filtered;
}

