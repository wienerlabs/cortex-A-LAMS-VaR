/**
 * Token Whitelist Builder
 * Filters tokens based on strict criteria for spot trading
 */

import type { ApprovedToken, TokenCriteria } from './types.js';
import { logger } from '../logger.js';

export const DEFAULT_TOKEN_CRITERIA: TokenCriteria = {
  minMarketCap: 20_000_000,        // $20M
  minTokenAge: 90,                  // 90 days (3 months)
  minLiquidity: 200_000,            // $200K
  min24hVolume: 100_000,            // $100K
  minHolders: 2500,
  maxTopHolderShare: 0.20,          // 20%
  excludeCategories: ['MEMECOIN', 'WRAPPED', 'REBASING', 'TRANSFER_TAX'],
  requiredDEXs: ['JUPITER', 'RAYDIUM_OR_ORCA'],
  requireContractVerification: true,
};

export class TokenWhitelistBuilder {
  private criteria: TokenCriteria;
  private whitelist: Map<string, ApprovedToken> = new Map();

  constructor(criteria: TokenCriteria = DEFAULT_TOKEN_CRITERIA) {
    this.criteria = criteria;
  }

  /**
   * Build whitelist from market scanner data
   */
  async buildWhitelist(tokens: any[]): Promise<ApprovedToken[]> {
    const approved: ApprovedToken[] = [];

    for (const token of tokens) {
      try {
        const result = await this.evaluateToken(token);
        if (result) {
          approved.push(result);
          this.whitelist.set(token.address, result);
        }
      } catch (error) {
        logger.warn('[TokenWhitelist] Error evaluating token', {
          symbol: token.symbol,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    logger.info('[TokenWhitelist] Whitelist built', {
      totalEvaluated: tokens.length,
      approved: approved.length,
      rejected: tokens.length - approved.length,
    });

    return approved;
  }

  /**
   * Evaluate single token against criteria
   */
  private async evaluateToken(token: any): Promise<ApprovedToken | null> {
    // Market cap check
    if (token.marketCap < this.criteria.minMarketCap) {
      return null;
    }

    // Liquidity check
    if (token.liquidity < this.criteria.minLiquidity) {
      return null;
    }

    // Volume check
    if (token.volume24h < this.criteria.min24hVolume) {
      return null;
    }

    // Holders check (if available)
    if (token.holders && token.holders < this.criteria.minHolders) {
      return null;
    }

    // Top holder concentration check (if available)
    if (token.topHolderShare && token.topHolderShare > this.criteria.maxTopHolderShare) {
      return null;
    }

    // Category exclusions
    if (token.category && this.criteria.excludeCategories.includes(token.category.toUpperCase())) {
      return null;
    }

    // DEX availability check
    const hasDEX = this.checkDEXAvailability(token.dexes || []);
    if (!hasDEX) {
      return null;
    }

    // Contract verification (if required)
    if (this.criteria.requireContractVerification && !token.verified) {
      return null;
    }

    // Calculate token age (if available)
    const age = token.createdAt ? Math.floor((Date.now() - token.createdAt) / (1000 * 60 * 60 * 24)) : 365;
    if (age < this.criteria.minTokenAge) {
      return null;
    }

    // Determine tier based on quality metrics
    const tier = this.calculateTier(token);

    const approvedToken: ApprovedToken = {
      symbol: token.symbol,
      address: token.address,
      marketCap: token.marketCap,
      liquidity: token.liquidity,
      volume24h: token.volume24h,
      holders: token.holders || 0,
      age,
      tier,
      dexes: token.dexes || [],
      verified: token.verified || false,
      approvedAt: Date.now(),
    };

    return approvedToken;
  }

  /**
   * Check if token is available on required DEXs
   */
  private checkDEXAvailability(dexes: string[]): boolean {
    const dexesUpper = dexes.map(d => d.toUpperCase());
    
    // Check for Jupiter
    if (this.criteria.requiredDEXs.includes('JUPITER') && !dexesUpper.includes('JUPITER')) {
      return false;
    }

    // Check for Raydium or Orca
    if (this.criteria.requiredDEXs.includes('RAYDIUM_OR_ORCA')) {
      const hasRaydiumOrOrca = dexesUpper.includes('RAYDIUM') || dexesUpper.includes('ORCA');
      if (!hasRaydiumOrOrca) {
        return false;
      }
    }

    return true;
  }

  /**
   * Calculate token tier (1 = highest quality, 3 = lowest)
   */
  private calculateTier(token: any): 1 | 2 | 3 {
    let score = 0;

    // Market cap scoring
    if (token.marketCap > 100_000_000) score += 3;
    else if (token.marketCap > 50_000_000) score += 2;
    else score += 1;

    // Liquidity scoring
    if (token.liquidity > 1_000_000) score += 2;
    else if (token.liquidity > 500_000) score += 1;

    // Volume scoring
    if (token.volume24h > 500_000) score += 2;
    else if (token.volume24h > 250_000) score += 1;

    // Tier assignment
    if (score >= 6) return 1;
    if (score >= 4) return 2;
    return 3;
  }

  /**
   * Check if token is whitelisted
   */
  isWhitelisted(address: string): boolean {
    return this.whitelist.has(address);
  }

  /**
   * Get whitelisted token
   */
  getToken(address: string): ApprovedToken | undefined {
    return this.whitelist.get(address);
  }

  /**
   * Get all whitelisted tokens
   */
  getAllTokens(): ApprovedToken[] {
    return Array.from(this.whitelist.values());
  }
}

