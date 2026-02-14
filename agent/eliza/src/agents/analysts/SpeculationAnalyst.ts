/**
 * SpeculationAnalyst - Multi-Source Sentiment Analyst
 * 
 * Evaluates trading opportunities based on sentiment from multiple sources:
 * - Twitter (40% weight)
 * - CryptoPanic (40% weight)
 * - Telegram (20% weight)
 * 
 * Capabilities:
 * - Multi-source sentiment aggregation
 * - Confidence-based filtering
 * - Signal generation (BULLISH/BEARISH/NEUTRAL)
 * - Graceful fallback when sources unavailable
 */

import { BaseAnalyst, DEFAULT_ANALYST_CONFIG, type AnalystConfig } from './BaseAnalyst.js';
import { getMultiSourceAggregator, type MultiSourceSentiment, type SentimentSignal } from '../../services/sentiment/multiSourceAggregator.js';
import { type NewsItem } from '../../services/sentiment/cryptopanicCollector.js';
import { logger } from '../../services/logger.js';

// ============= TYPES =============

/**
 * Input for SpeculationAnalyst
 */
export interface SpeculationAnalysisInput {
  tokens: string[];
  portfolioValueUsd?: number;
}

/**
 * Output from SpeculationAnalyst
 */
export interface SpeculationOpportunityResult {
  type: 'speculation';
  name: string;
  token: string;
  signal: SentimentSignal;
  sentimentScore: number; // -1 to 1
  expectedReturn: number; // Estimated based on sentiment strength
  riskScore: number; // 1-10
  confidence: number; // 0-1
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  raw: MultiSourceSentiment;
  sources: {
    twitter: { available: boolean; score: number };
    cryptopanic: { available: boolean; score: number };
    telegram: { available: boolean; score: number };
  };
  news: NewsItem[]; // Top 3 news items from CryptoPanic
}

/**
 * Configuration for SpeculationAnalyst
 */
export interface SpeculationAnalystConfig extends AnalystConfig {
  minSentimentScore: number;      // Minimum absolute sentiment score (0-1)
  minConfidenceThreshold: number; // Minimum confidence to approve
  maxRiskScore: number;           // Maximum risk score to approve
  estimatedReturnMultiplier: number; // Multiplier for sentiment -> expected return
}

/**
 * Default configuration
 */
export const DEFAULT_SPECULATION_CONFIG: SpeculationAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minSentimentScore: 0.3,         // Require at least 0.3 sentiment strength
  minConfidenceThreshold: 0.6,    // 60% minimum confidence
  maxRiskScore: 7,                // Max risk 7/10
  estimatedReturnMultiplier: 10,  // sentiment * 10 = expected return %
};

// ============= ANALYST CLASS =============

/**
 * SpeculationAnalyst - Multi-Source Sentiment Evaluator
 */
export class SpeculationAnalyst extends BaseAnalyst<SpeculationAnalysisInput, SpeculationOpportunityResult> {
  private speculationConfig: SpeculationAnalystConfig;
  private aggregator = getMultiSourceAggregator();

  constructor(config: Partial<SpeculationAnalystConfig> = {}) {
    const mergedConfig = { ...DEFAULT_SPECULATION_CONFIG, ...config };
    super(mergedConfig);
    this.speculationConfig = mergedConfig;
    logger.info('[SpeculationAnalyst] Initialized', { config: this.speculationConfig });
  }

  getName(): string {
    return 'SpeculationAnalyst';
  }

  /**
   * Analyze tokens for sentiment-based opportunities
   */
  async analyze(input: SpeculationAnalysisInput): Promise<SpeculationOpportunityResult[]> {
    const results: SpeculationOpportunityResult[] = [];

    logger.info('[SpeculationAnalyst] Analyzing tokens', { count: input.tokens.length });

    // Fetch sentiment for all tokens in parallel
    const sentimentPromises = input.tokens.map(token =>
      this.aggregator.aggregateSentiment(token).catch(error => {
        logger.warn('[SpeculationAnalyst] Failed to fetch sentiment', { token, error: error.message });
        return null;
      })
    );

    const sentiments = await Promise.all(sentimentPromises);

    // Evaluate each sentiment result
    for (const sentiment of sentiments) {
      if (!sentiment) continue;

      const result = this.evaluateSentiment(sentiment);
      results.push(result);
    }

    // Sort by risk-adjusted return (best first)
    results.sort((a, b) => b.riskAdjustedReturn - a.riskAdjustedReturn);

    logger.info('[SpeculationAnalyst] Analysis complete', {
      total: results.length,
      approved: results.filter(r => r.approved).length,
      bullish: results.filter(r => r.signal === 'BULLISH').length,
      bearish: results.filter(r => r.signal === 'BEARISH').length,
    });

    return results;
  }

  /**
   * Evaluate a single sentiment result
   */
  private evaluateSentiment(sentiment: MultiSourceSentiment): SpeculationOpportunityResult {
    const { token, overallScore, confidence, signal, sources } = sentiment;

    // Calculate expected return based on sentiment strength
    // Stronger sentiment = higher expected return
    const sentimentStrength = Math.abs(overallScore);
    const expectedReturn = sentimentStrength * this.speculationConfig.estimatedReturnMultiplier;

    // Calculate risk score (1-10)
    // Lower confidence = higher risk
    // Neutral signal = higher risk
    let riskScore = 5; // Base risk

    // Adjust for confidence
    riskScore += (1 - confidence) * 3; // Low confidence adds up to 3 risk points

    // Adjust for signal strength
    if (signal === 'NEUTRAL') {
      riskScore += 2; // Neutral signals are riskier
    }

    // Adjust for source availability
    const availableSources = [
      sources.twitter.available,
      sources.cryptopanic.available,
      sources.telegram.available,
    ].filter(Boolean).length;

    if (availableSources < 2) {
      riskScore += 2; // Less than 2 sources = higher risk
    }

    // Cap risk score at 10
    riskScore = Math.min(10, riskScore);

    // Calculate risk-adjusted return
    const riskAdjustedReturn = expectedReturn * (1 - riskScore / 20);

    // Determine approval
    const warnings: string[] = [];
    let rejectReason: string | undefined;

    // Check minimum sentiment strength
    if (sentimentStrength < this.speculationConfig.minSentimentScore) {
      rejectReason = `Sentiment too weak: ${sentimentStrength.toFixed(3)} (min ${this.speculationConfig.minSentimentScore})`;
    }

    // Check minimum confidence
    if (confidence < this.speculationConfig.minConfidenceThreshold) {
      rejectReason = rejectReason || `Confidence too low: ${confidence.toFixed(3)} (min ${this.speculationConfig.minConfidenceThreshold})`;
    }

    // Check risk score
    if (riskScore > this.speculationConfig.maxRiskScore) {
      rejectReason = rejectReason || `Risk too high: ${riskScore.toFixed(1)}/10 (max ${this.speculationConfig.maxRiskScore})`;
    }

    // Check signal
    if (signal === 'NEUTRAL') {
      warnings.push('Neutral sentiment signal - no clear direction');
    }

    // Warn about missing sources
    if (!sources.twitter.available) {
      warnings.push('Twitter data unavailable');
    }
    if (!sources.cryptopanic.available) {
      warnings.push('CryptoPanic data unavailable');
    }
    if (!sources.telegram.available) {
      warnings.push('Telegram data unavailable');
    }

    const approved = !rejectReason;

    // Extract top 3 news items from CryptoPanic
    const news: NewsItem[] = sources.cryptopanic.topNews?.slice(0, 3) || [];

    const result: SpeculationOpportunityResult = {
      type: 'speculation',
      name: `${signal} ${token} (Sentiment)`,
      token,
      signal,
      sentimentScore: overallScore,
      expectedReturn,
      riskScore,
      confidence,
      riskAdjustedReturn,
      approved,
      rejectReason,
      warnings,
      raw: sentiment,
      sources: {
        twitter: {
          available: sources.twitter.available,
          score: sources.twitter.score,
        },
        cryptopanic: {
          available: sources.cryptopanic.available,
          score: sources.cryptopanic.score,
        },
        telegram: {
          available: sources.telegram.available,
          score: sources.telegram.score,
        },
      },
      news, // Top 3 news items from CryptoPanic
    };

    logger.debug('[SpeculationAnalyst] Evaluated sentiment', {
      token,
      signal,
      sentimentScore: overallScore.toFixed(3),
      confidence: confidence.toFixed(3),
      expectedReturn: expectedReturn.toFixed(2),
      riskScore: riskScore.toFixed(1),
      approved,
      rejectReason,
    });

    return result;
  }
}

