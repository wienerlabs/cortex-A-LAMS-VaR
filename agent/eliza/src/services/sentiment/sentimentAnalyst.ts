/**
 * Sentiment Analyst
 * 
 * High-level sentiment analysis combining Twitter data collection and NLP scoring.
 * 
 * Calculates:
 * - Overall sentiment score (weighted average)
 * - Sentiment velocity (change rate over time)
 * - Volume anomaly detection (compare to baseline)
 * - Credibility weighting (follower count, verification)
 */
import { logger } from '../logger.js';
import { TwitterCollector, getTwitterCollector, RateLimitError, type Tweet } from './twitterCollector.js';
import { SentimentScorer, getSentimentScorer, type AggregatedSentiment } from './sentimentScorer.js';

// Re-export RateLimitError for convenience
export { RateLimitError } from './twitterCollector.js';

// ============= TYPES =============
export type SentimentVelocity = 'accelerating' | 'stable' | 'declining';
export type SentimentSignal = 'bullish' | 'bearish' | 'neutral';

export interface SentimentAnalysis {
  symbol: string;
  score: number;                    // -1 to +1
  velocity: SentimentVelocity;      // Change rate over time
  divergence: boolean;              // Price vs sentiment mismatch
  anomaly: boolean;                 // Unusual volume spike
  credibilityScore: number;         // 0-1, weighted by source quality
  confidence: number;               // 0-1
  signal: SentimentSignal;
  reasoning: string;                // Human-readable explanation
  rawData: {
    tweetCount: number;
    totalEngagement: number;
    verifiedRatio: number;
    scoreDistribution: AggregatedSentiment['scoreDistribution'];
    topTweets: {
      positive: string[];
      negative: string[];
    };
  };
  timestamp: Date;
}

export interface HistoricalDataPoint {
  symbol: string;
  score: number;
  volume: number;
  timestamp: Date;
}

export interface SentimentAnalystConfig {
  baselineVolume?: number;          // Expected tweets per query
  anomalyThreshold?: number;        // Multiplier for anomaly detection
  velocityWindow?: number;          // Minutes to consider for velocity
  minTweetsForConfidence?: number;  // Minimum tweets for high confidence
}

// ============= CONSTANTS =============
const DEFAULT_CONFIG: Required<SentimentAnalystConfig> = {
  baselineVolume: 50,
  anomalyThreshold: 2.0,
  velocityWindow: 60,
  minTweetsForConfidence: 20,
};

// ============= SENTIMENT ANALYST CLASS =============
export class SentimentAnalyst {
  private twitterCollector: TwitterCollector;
  private sentimentScorer: SentimentScorer;
  private config: Required<SentimentAnalystConfig>;
  private historicalData: Map<string, HistoricalDataPoint[]> = new Map();

  constructor(config?: SentimentAnalystConfig) {
    this.twitterCollector = getTwitterCollector();
    this.sentimentScorer = getSentimentScorer();
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    logger.info('SentimentAnalyst initialized', { config: this.config });
  }

  /**
   * Calculate sentiment velocity from historical data
   */
  private calculateVelocity(symbol: string, currentScore: number): SentimentVelocity {
    const history = this.historicalData.get(symbol) || [];
    
    if (history.length < 2) return 'stable';

    // Get data points within velocity window
    const now = Date.now();
    const windowMs = this.config.velocityWindow * 60 * 1000;
    const recentHistory = history.filter(h => now - h.timestamp.getTime() < windowMs);

    if (recentHistory.length < 2) return 'stable';

    // Calculate trend
    const oldScore = recentHistory[0].score;
    const scoreDiff = currentScore - oldScore;

    if (scoreDiff > 0.1) return 'accelerating';
    if (scoreDiff < -0.1) return 'declining';
    return 'stable';
  }

  /**
   * Detect volume anomaly
   */
  private detectAnomaly(tweetCount: number): boolean {
    return tweetCount > this.config.baselineVolume * this.config.anomalyThreshold;
  }

  /**
   * Calculate credibility score based on author quality
   */
  private calculateCredibilityScore(tweets: Tweet[]): number {
    if (tweets.length === 0) return 0;

    let totalCredibility = 0;
    let verifiedCount = 0;
    let highFollowerCount = 0;

    for (const tweet of tweets) {
      if (!tweet.author) continue;

      if (tweet.author.verified) {
        verifiedCount++;
        totalCredibility += 1;
      } else {
        totalCredibility += 0.3;
      }

      if (tweet.author.followers_count > 10000) {
        highFollowerCount++;
        totalCredibility += 0.5;
      }
    }

    // Normalize to 0-1
    const avgCredibility = totalCredibility / (tweets.length * 1.5);
    return Math.min(1, avgCredibility);
  }

  /**
   * Calculate confidence based on data quality
   */
  private calculateConfidence(tweets: Tweet[], credibilityScore: number): number {
    const volumeConfidence = Math.min(1, tweets.length / this.config.minTweetsForConfidence);
    const combinedConfidence = (volumeConfidence + credibilityScore) / 2;
    return Math.round(combinedConfidence * 100) / 100;
  }

  /**
   * Determine signal from score
   */
  private determineSignal(score: number): SentimentSignal {
    if (score > 0.15) return 'bullish';
    if (score < -0.15) return 'bearish';
    return 'neutral';
  }

  /**
   * Generate human-readable reasoning
   */
  private generateReasoning(analysis: Omit<SentimentAnalysis, 'reasoning'>): string {
    const parts: string[] = [];

    // Overall sentiment
    const sentimentDesc = analysis.score > 0.3 ? 'strongly positive' :
      analysis.score > 0.1 ? 'moderately positive' :
      analysis.score < -0.3 ? 'strongly negative' :
      analysis.score < -0.1 ? 'moderately negative' : 'neutral';
    
    parts.push(`${analysis.symbol} sentiment is ${sentimentDesc} (${(analysis.score * 100).toFixed(1)}%)`);

    // Volume
    parts.push(`Based on ${analysis.rawData.tweetCount} tweets`);

    // Velocity
    if (analysis.velocity === 'accelerating') {
      parts.push('with improving sentiment momentum');
    } else if (analysis.velocity === 'declining') {
      parts.push('with declining sentiment momentum');
    }

    // Anomaly
    if (analysis.anomaly) {
      parts.push('ALERT: Unusual tweet volume detected');
    }

    // Credibility
    if (analysis.credibilityScore > 0.7) {
      parts.push('High credibility sources');
    } else if (analysis.credibilityScore < 0.3) {
      parts.push('Low credibility sources - use caution');
    }

    // Confidence
    if (analysis.confidence < 0.5) {
      parts.push('Low confidence due to limited data');
    }

    return parts.join('. ') + '.';
  }

  /**
   * Analyze sentiment for a token symbol
   */
  async analyze(symbol: string, options?: { maxTweets?: number }): Promise<SentimentAnalysis> {
    const maxTweets = options?.maxTweets || 100;

    logger.info('Starting sentiment analysis', { symbol, maxTweets });

    try {
      // Fetch tweets
      const searchResult = await this.twitterCollector.searchTweets(symbol, { maxResults: maxTweets });
      const tweets = searchResult.tweets;

      // Analyze sentiment
      const aggregated = this.sentimentScorer.analyzeTweets(tweets, symbol);

      // Calculate metrics
      const velocity = this.calculateVelocity(symbol, aggregated.weightedAverageScore);
      const anomaly = this.detectAnomaly(tweets.length);
      const credibilityScore = this.calculateCredibilityScore(tweets);
      const confidence = this.calculateConfidence(tweets, credibilityScore);
      const signal = this.determineSignal(aggregated.weightedAverageScore);

      // Calculate total engagement
      const stats = this.twitterCollector.getAggregateStats(tweets);
      const totalEngagement = stats.totalLikes + stats.totalRetweets + stats.totalReplies;

      // Build partial analysis (without reasoning)
      const partialAnalysis = {
        symbol,
        score: Math.round(aggregated.weightedAverageScore * 1000) / 1000,
        velocity,
        divergence: false, // Would need price data to calculate
        anomaly,
        credibilityScore: Math.round(credibilityScore * 100) / 100,
        confidence,
        signal,
        rawData: {
          tweetCount: tweets.length,
          totalEngagement,
          verifiedRatio: tweets.length > 0 ? stats.verifiedCount / tweets.length : 0,
          scoreDistribution: aggregated.scoreDistribution,
          topTweets: {
            positive: aggregated.topPositive.map(s => s.tweet.text.slice(0, 100)),
            negative: aggregated.topNegative.map(s => s.tweet.text.slice(0, 100)),
          },
        },
        timestamp: new Date(),
      };

      // Generate reasoning
      const reasoning = this.generateReasoning(partialAnalysis);

      const analysis: SentimentAnalysis = {
        ...partialAnalysis,
        reasoning,
      };

      // Store in history for velocity calculation
      const history = this.historicalData.get(symbol) || [];
      history.push({
        symbol,
        score: analysis.score,
        volume: tweets.length,
        timestamp: new Date(),
      });

      // Keep only last hour of history
      const oneHourAgo = Date.now() - 60 * 60 * 1000;
      this.historicalData.set(
        symbol,
        history.filter(h => h.timestamp.getTime() > oneHourAgo)
      );

      logger.info('Sentiment analysis completed', {
        symbol,
        score: analysis.score,
        signal: analysis.signal,
        confidence: analysis.confidence,
        tweetCount: tweets.length,
      });

      return analysis;
    } catch (error) {
      logger.error('Sentiment analysis failed', {
        symbol,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Check price-sentiment divergence (requires external price data)
   */
  checkDivergence(sentimentScore: number, priceChange24h: number): boolean {
    // Divergence: sentiment and price moving in opposite directions
    const sentimentPositive = sentimentScore > 0.1;
    const sentimentNegative = sentimentScore < -0.1;
    const pricePositive = priceChange24h > 5; // >5% up
    const priceNegative = priceChange24h < -5; // >5% down

    return (sentimentPositive && priceNegative) || (sentimentNegative && pricePositive);
  }

  /**
   * Get rate limit status from Twitter collector
   */
  getRateLimitStatus() {
    return this.twitterCollector.getRateLimitStatus();
  }

  /**
   * Clear caches
   */
  clearCaches(): void {
    this.twitterCollector.clearCache();
    this.historicalData.clear();
    logger.debug('SentimentAnalyst caches cleared');
  }
}

// Singleton instance
let sentimentAnalystInstance: SentimentAnalyst | null = null;

export function getSentimentAnalyst(config?: SentimentAnalystConfig): SentimentAnalyst {
  if (!sentimentAnalystInstance) {
    sentimentAnalystInstance = new SentimentAnalyst(config);
  }
  return sentimentAnalystInstance;
}

export function resetSentimentAnalyst(): void {
  sentimentAnalystInstance = null;
}

