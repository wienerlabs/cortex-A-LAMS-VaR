/**
 * Sentiment Integration for Trading Strategies
 * 
 * Provides weighted sentiment integration for trading decisions.
 * Sentiment does NOT veto trades - it only adjusts confidence scores.
 * 
 * Weights per strategy type:
 * - Perps: 25% sentiment, 75% ML (high sentiment impact)
 * - LP: 15% sentiment, 85% ML (moderate sentiment impact)
 * - Arbitrage: 5% sentiment, 95% ML (minimal sentiment impact)
 */
import { logger } from '../logger.js';
import { getSentimentAnalyst, type SentimentAnalysis, RateLimitError } from './sentimentAnalyst.js';

// ============= CONFIGURATION =============
export interface SentimentIntegrationConfig {
  /** Enable sentiment integration */
  enabled: boolean;
  /** Timeout for sentiment fetch in milliseconds */
  timeoutMs: number;
  /** Weight configurations per strategy type */
  weights: {
    lp: number;        // 0.15 = 15% sentiment weight
    arbitrage: number; // 0.05 = 5% sentiment weight
    perps: number;     // 0.25 = 25% sentiment weight
  };
  /** Minimum tweets required for confidence */
  minTweetsForConfidence: number;
}

export const DEFAULT_SENTIMENT_CONFIG: SentimentIntegrationConfig = {
  enabled: true,
  timeoutMs: 5000, // 5 seconds
  weights: {
    lp: 0.15,        // 15% per spec (10-15% range)
    arbitrage: 0.05, // 5% per spec
    perps: 0.25,     // 25% per spec (25-30% range)
  },
  minTweetsForConfidence: 10,
};

// ============= TYPES =============
export type StrategyType = 'lp' | 'arbitrage' | 'perps';

export interface SentimentAdjustedScore {
  /** Original ML confidence (0-1) */
  mlConfidence: number;
  /** Raw sentiment score (-1 to +1) */
  rawSentiment: number;
  /** Normalized sentiment (0-1) */
  normalizedSentiment: number;
  /** Final combined score (0-1) */
  finalScore: number;
  /** Strategy weight used */
  sentimentWeight: number;
  /** Sentiment signal */
  signal: 'bullish' | 'bearish' | 'neutral';
  /** Full sentiment analysis (if available) */
  sentimentAnalysis?: SentimentAnalysis;
  /** Whether sentiment was successfully fetched */
  sentimentAvailable: boolean;
  /** Human-readable reasoning */
  reasoning: string;
}

// ============= HELPER FUNCTIONS =============

/**
 * Normalize sentiment score from -1/+1 range to 0-1 range
 */
export function normalizeSentiment(sentimentScore: number): number {
  return (sentimentScore + 1) / 2;
}

/**
 * Combine ML confidence with sentiment score using weighted average
 */
export function combineScores(
  mlConfidence: number,
  normalizedSentiment: number,
  sentimentWeight: number
): number {
  const mlWeight = 1 - sentimentWeight;
  return (mlConfidence * mlWeight) + (normalizedSentiment * sentimentWeight);
}

/**
 * Get sentiment weight for a strategy type
 */
export function getSentimentWeight(
  strategyType: StrategyType,
  config: SentimentIntegrationConfig = DEFAULT_SENTIMENT_CONFIG
): number {
  return config.weights[strategyType] ?? 0.1;
}

// ============= MAIN INTEGRATION CLASS =============
export class SentimentIntegration {
  private config: SentimentIntegrationConfig;
  private cache: Map<string, { analysis: SentimentAnalysis; timestamp: number }> = new Map();
  private cacheTtlMs = 60000; // 1 minute cache

  constructor(config?: Partial<SentimentIntegrationConfig>) {
    this.config = { ...DEFAULT_SENTIMENT_CONFIG, ...config };
    logger.info('SentimentIntegration initialized', { config: this.config });
  }

  /**
   * Get adjusted confidence score with sentiment integration
   * 
   * @param symbol - Token symbol (e.g., 'SOL')
   * @param mlConfidence - Original ML model confidence (0-1)
   * @param strategyType - Type of strategy ('lp', 'arbitrage', 'perps')
   * @returns Adjusted score with sentiment integration
   */
  async getAdjustedScore(
    symbol: string,
    mlConfidence: number,
    strategyType: StrategyType
  ): Promise<SentimentAdjustedScore> {
    const sentimentWeight = getSentimentWeight(strategyType, this.config);
    const mlWeight = 1 - sentimentWeight;

    // If sentiment disabled, return ML-only score
    if (!this.config.enabled) {
      return this.createFallbackResult(mlConfidence, strategyType, 'Sentiment integration disabled');
    }

    try {
      // Fetch sentiment with timeout
      const analysis = await this.fetchSentimentWithTimeout(symbol);

      if (!analysis) {
        return this.createFallbackResult(mlConfidence, strategyType, 'Sentiment fetch failed/timed out');
      }

      // Normalize sentiment from -1/+1 to 0-1
      const normalizedSentiment = normalizeSentiment(analysis.score);

      // Combine scores
      const finalScore = combineScores(mlConfidence, normalizedSentiment, sentimentWeight);

      // Generate reasoning
      const reasoning = this.generateReasoning(
        symbol, mlConfidence, analysis.score, normalizedSentiment,
        finalScore, sentimentWeight, strategyType, analysis
      );

      logger.info('Sentiment-adjusted score calculated', {
        symbol,
        strategyType,
        mlConfidence: mlConfidence.toFixed(3),
        rawSentiment: analysis.score.toFixed(3),
        normalizedSentiment: normalizedSentiment.toFixed(3),
        finalScore: finalScore.toFixed(3),
        sentimentWeight,
      });

      return {
        mlConfidence,
        rawSentiment: analysis.score,
        normalizedSentiment,
        finalScore,
        sentimentWeight,
        signal: analysis.signal,
        sentimentAnalysis: analysis,
        sentimentAvailable: true,
        reasoning,
      };
    } catch (error) {
      // Handle rate limit errors gracefully
      if (error instanceof RateLimitError) {
        logger.warn('Sentiment rate limited, using ML-only', { symbol, resetTime: error.resetTime });
        return this.createFallbackResult(mlConfidence, strategyType, `Rate limited until ${error.resetTime.toISOString()}`);
      }

      logger.warn('Sentiment fetch error, using ML-only', {
        symbol,
        error: error instanceof Error ? error.message : String(error),
      });
      return this.createFallbackResult(mlConfidence, strategyType, 'Sentiment service error');
    }
  }

  /**
   * Fetch sentiment with timeout and caching
   */
  private async fetchSentimentWithTimeout(symbol: string): Promise<SentimentAnalysis | null> {
    // Check cache first
    const cached = this.cache.get(symbol);
    if (cached && Date.now() - cached.timestamp < this.cacheTtlMs) {
      logger.debug('Using cached sentiment', { symbol });
      return cached.analysis;
    }

    const analyst = getSentimentAnalyst();
    if (!analyst) {
      logger.warn('Sentiment analyst not available');
      return null;
    }

    // Create timeout promise
    const timeoutPromise = new Promise<null>((resolve) => {
      setTimeout(() => resolve(null), this.config.timeoutMs);
    });

    // Race sentiment fetch against timeout
    const analysis = await Promise.race([
      analyst.analyze(symbol),
      timeoutPromise,
    ]);

    if (analysis) {
      this.cache.set(symbol, { analysis, timestamp: Date.now() });
    }

    return analysis;
  }

  /**
   * Create fallback result when sentiment is unavailable
   */
  private createFallbackResult(
    mlConfidence: number,
    strategyType: StrategyType,
    reason: string
  ): SentimentAdjustedScore {
    return {
      mlConfidence,
      rawSentiment: 0,
      normalizedSentiment: 0.5, // Neutral
      finalScore: mlConfidence, // Use ML-only
      sentimentWeight: getSentimentWeight(strategyType, this.config),
      signal: 'neutral',
      sentimentAvailable: false,
      reasoning: `Using ML-only confidence (${(mlConfidence * 100).toFixed(1)}%). ${reason}`,
    };
  }

  /**
   * Generate human-readable reasoning for the score adjustment
   */
  private generateReasoning(
    symbol: string,
    mlConfidence: number,
    rawSentiment: number,
    normalizedSentiment: number,
    finalScore: number,
    sentimentWeight: number,
    strategyType: StrategyType,
    analysis: SentimentAnalysis
  ): string {
    const mlPct = (mlConfidence * 100).toFixed(1);
    const sentPct = (normalizedSentiment * 100).toFixed(1);
    const finalPct = (finalScore * 100).toFixed(1);
    const weightPct = (sentimentWeight * 100).toFixed(0);
    const mlWeightPct = ((1 - sentimentWeight) * 100).toFixed(0);

    const direction = finalScore > mlConfidence ? 'boosted' : finalScore < mlConfidence ? 'reduced' : 'unchanged';
    const delta = Math.abs(finalScore - mlConfidence) * 100;

    return `${symbol} ${strategyType.toUpperCase()}: ML confidence ${mlPct}% ${direction} by ${delta.toFixed(1)}% to ${finalPct}% ` +
      `(${analysis.signal} sentiment ${sentPct}%, weight: ${weightPct}% sentiment / ${mlWeightPct}% ML, ` +
      `${analysis.rawData.tweetCount} tweets analyzed)`;
  }

  /**
   * Clear the sentiment cache
   */
  clearCache(): void {
    this.cache.clear();
    logger.debug('Sentiment cache cleared');
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SentimentIntegrationConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('SentimentIntegration config updated', { config: this.config });
  }

  /**
   * Get current configuration
   */
  getConfig(): SentimentIntegrationConfig {
    return { ...this.config };
  }
}

// ============= SINGLETON INSTANCE =============
let sentimentIntegration: SentimentIntegration | null = null;

export function getSentimentIntegration(): SentimentIntegration {
  if (!sentimentIntegration) {
    sentimentIntegration = new SentimentIntegration();
  }
  return sentimentIntegration;
}

export function initializeSentimentIntegration(config?: Partial<SentimentIntegrationConfig>): SentimentIntegration {
  sentimentIntegration = new SentimentIntegration(config);
  return sentimentIntegration;
}

