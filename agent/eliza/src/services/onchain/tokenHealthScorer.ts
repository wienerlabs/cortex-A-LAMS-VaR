/**
 * Token Health Scorer
 * 
 * Calculates health score (0-100) based on on-chain metrics.
 * 
 * Scoring Breakdown:
 * 1. Holder Concentration (40 points)
 *    - Top 10 holders < 50% = good (40 points)
 *    - Top 10 holders 50-70% = medium (20 points)
 *    - Top 10 holders > 70% = bad (0 points)
 * 
 * 2. Liquidity Depth (30 points)
 *    - TVL > $5M = excellent (30 points)
 *    - TVL $1M-$5M = good (20 points)
 *    - TVL < $1M = poor (0 points)
 * 
 * 3. Token Age (20 points)
 *    - > 1 year = mature (20 points)
 *    - 3-12 months = established (15 points)
 *    - < 3 months = new (5 points)
 * 
 * 4. Whale Activity (10 points)
 *    - No large dumps in 7d = safe (10 points)
 *    - 1-2 large transfers = caution (5 points)
 *    - 3+ large transfers = risky (0 points)
 */

import { logger } from '../logger.js';
import type { TokenOnChainData } from './heliusClient.js';

// ============= TYPES =============

export interface TokenHealthMetrics {
  holderConcentration: number; // 0-40 points
  liquidityDepth: number; // 0-30 points
  tokenAge: number; // 0-20 points
  whaleActivity: number; // 0-10 points
}

export interface TokenHealthScore {
  token: string;
  totalScore: number; // 0-100
  rating: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
  metrics: TokenHealthMetrics;
  details: {
    holderCount: number; // -1 = too many to count, 0+ = actual count
    topHoldersPercentage: number;
    tvlUsd: number;
    ageInDays: number;
    whaleTransferCount: number;
    isHighlyDistributed: boolean;
  };
}

// ============= SCORER =============

export class TokenHealthScorer {
  /**
   * Calculate complete health score for a token
   */
  calculateHealthScore(
    onChainData: TokenOnChainData,
    tvlUsd: number = 0
  ): TokenHealthScore {
    const metrics: TokenHealthMetrics = {
      holderConcentration: this.scoreHolderConcentration(
        onChainData.holderCount,
        onChainData.topHoldersPercentage,
        onChainData.isHighlyDistributed
      ),
      liquidityDepth: this.scoreLiquidityDepth(tvlUsd),
      tokenAge: this.scoreTokenAge(onChainData.metadata.age),
      whaleActivity: this.scoreWhaleActivity(onChainData.whaleActivityCount),
    };

    const totalScore =
      metrics.holderConcentration +
      metrics.liquidityDepth +
      metrics.tokenAge +
      metrics.whaleActivity;

    const rating = this.getRating(totalScore);

    logger.info('[TokenHealthScorer] Score calculated', {
      token: onChainData.mint,
      totalScore,
      rating,
      metrics,
    });

    return {
      token: onChainData.mint,
      totalScore,
      rating,
      metrics,
      details: {
        holderCount: onChainData.holderCount,
        topHoldersPercentage: onChainData.topHoldersPercentage,
        tvlUsd,
        ageInDays: onChainData.metadata.age,
        whaleTransferCount: onChainData.whaleActivityCount,
        isHighlyDistributed: onChainData.isHighlyDistributed,
      },
    };
  }

  /**
   * Score holder concentration (40 points max)
   *
   * Dynamic detection:
   * - holderCount = -1 means too many holders to count (highly distributed) = EXCELLENT
   * - Otherwise, score based on top 10 holder concentration
   */
  private scoreHolderConcentration(
    holderCount: number,
    topHoldersPercentage: number,
    isHighlyDistributed: boolean
  ): number {
    // If holder count exceeds API limit, it's a highly distributed token
    if (holderCount === -1 || isHighlyDistributed) {
      logger.info('[TokenHealthScorer] Highly distributed token detected', {
        holderCount,
        reason: 'Too many holders for API (likely >1M)',
        score: 40,
        interpretation: 'Excellent distribution - full points',
      });
      return 40; // Full points - excellent distribution
    }

    // Normal scoring based on top 10 concentration
    if (topHoldersPercentage < 50) {
      return 40; // Good distribution
    } else if (topHoldersPercentage < 70) {
      return 20; // Medium concentration
    } else {
      return 0; // High concentration (risky)
    }
  }

  /**
   * Score liquidity depth (30 points max)
   */
  private scoreLiquidityDepth(tvlUsd: number): number {
    if (tvlUsd > 5_000_000) {
      return 30; // Excellent liquidity
    } else if (tvlUsd > 1_000_000) {
      return 20; // Good liquidity
    } else {
      return 0; // Poor liquidity
    }
  }

  /**
   * Score token age (20 points max)
   */
  private scoreTokenAge(ageInDays: number): number {
    if (ageInDays > 365) {
      return 20; // Mature token (> 1 year)
    } else if (ageInDays > 90) {
      return 15; // Established token (3-12 months)
    } else {
      return 5; // New token (< 3 months)
    }
  }

  /**
   * Score whale activity (10 points max)
   */
  private scoreWhaleActivity(whaleTransferCount: number): number {
    if (whaleTransferCount === 0) {
      return 10; // No whale activity (safe)
    } else if (whaleTransferCount <= 2) {
      return 5; // Some whale activity (caution)
    } else {
      return 0; // High whale activity (risky)
    }
  }

  /**
   * Get rating from total score
   */
  private getRating(totalScore: number): 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR' {
    if (totalScore >= 80) {
      return 'EXCELLENT';
    } else if (totalScore >= 60) {
      return 'GOOD';
    } else if (totalScore >= 40) {
      return 'FAIR';
    } else {
      return 'POOR';
    }
  }
}

// ============= SINGLETON =============

let tokenHealthScorer: TokenHealthScorer | null = null;

export function getTokenHealthScorer(): TokenHealthScorer {
  if (!tokenHealthScorer) {
    tokenHealthScorer = new TokenHealthScorer();
  }
  return tokenHealthScorer;
}

