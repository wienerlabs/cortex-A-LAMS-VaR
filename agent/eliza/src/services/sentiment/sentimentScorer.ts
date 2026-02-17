/**
 * Sentiment Scorer
 * 
 * NLP-based sentiment analysis for tweets.
 * Features:
 * - Analyze tweet text using sentiment library
 * - Return normalized score: -1 (very negative) to +1 (very positive)
 * - Aggregate scores for symbol
 * - Weight by engagement (high engagement = higher weight)
 */
import Sentiment from 'sentiment';
import { logger } from '../logger.js';
import type { Tweet } from './twitterCollector.js';

// ============= TYPES =============
export interface TweetSentiment {
  tweet: Tweet;
  rawScore: number;           // Raw sentiment score
  normalizedScore: number;    // -1 to +1
  comparative: number;        // Score per word
  positiveWords: string[];
  negativeWords: string[];
  engagementWeight: number;   // Weight based on engagement
  credibilityWeight: number;  // Weight based on author credibility
  weightedScore: number;      // Final weighted score
}

export interface AggregatedSentiment {
  symbol: string;
  tweetCount: number;
  averageScore: number;           // Simple average
  weightedAverageScore: number;   // Engagement-weighted average
  medianScore: number;
  scoreDistribution: {
    veryNegative: number;  // < -0.5
    negative: number;      // -0.5 to -0.1
    neutral: number;       // -0.1 to 0.1
    positive: number;      // 0.1 to 0.5
    veryPositive: number;  // > 0.5
  };
  topPositive: TweetSentiment[];
  topNegative: TweetSentiment[];
  timestamp: Date;
}

// ============= CONSTANTS =============
// Crypto-specific sentiment modifiers
const CRYPTO_POSITIVE_WORDS: Record<string, number> = {
  moon: 3,
  mooning: 4,
  bullish: 3,
  pump: 2,
  pumping: 3,
  ath: 3,
  breakout: 2,
  hodl: 2,
  diamond: 2,
  rocket: 2,
  lambo: 2,
  gem: 2,
  alpha: 2,
  degen: 1,
  wagmi: 2,
  gm: 1,
  lfg: 2,
  based: 2,
  chad: 1,
  accumulate: 2,
  accumulating: 2,
  undervalued: 2,
};

const CRYPTO_NEGATIVE_WORDS: Record<string, number> = {
  dump: -3,
  dumping: -4,
  bearish: -3,
  rug: -4,
  rugged: -5,
  scam: -5,
  crash: -3,
  crashing: -4,
  rekt: -3,
  ngmi: -2,
  sell: -1,
  selling: -2,
  fud: -2,
  overvalued: -2,
  ponzi: -4,
  shitcoin: -3,
  dead: -3,
  dying: -3,
};

// ============= SENTIMENT SCORER CLASS =============
export class SentimentScorer {
  private sentiment: Sentiment;

  constructor() {
    this.sentiment = new Sentiment();
    
    // Register crypto-specific words
    this.sentiment.registerLanguage('en', {
      labels: { ...CRYPTO_POSITIVE_WORDS, ...CRYPTO_NEGATIVE_WORDS },
    });
    
    logger.info('SentimentScorer initialized with crypto-specific vocabulary');
  }

  /**
   * Calculate engagement weight for a tweet
   * Higher engagement = higher weight
   */
  private calculateEngagementWeight(tweet: Tweet): number {
    const { like_count, retweet_count, reply_count, quote_count } = tweet.public_metrics;
    
    // Log scale to prevent viral tweets from dominating
    const totalEngagement = like_count + (retweet_count * 2) + (reply_count * 1.5) + (quote_count * 1.5);
    
    // Normalize: 0 engagement = 0.1 weight, 1000+ engagement = 1.0 weight
    const weight = Math.min(1, 0.1 + (Math.log10(totalEngagement + 1) / 3));
    
    return weight;
  }

  /**
   * Calculate credibility weight based on author metrics
   */
  private calculateCredibilityWeight(tweet: Tweet): number {
    if (!tweet.author) return 0.5;

    const { followers_count, verified } = tweet.author;
    
    let weight = 0.3; // Base weight
    
    // Verified accounts get bonus
    if (verified) weight += 0.3;
    
    // Follower count bonus (log scale)
    if (followers_count > 0) {
      weight += Math.min(0.4, Math.log10(followers_count) / 15);
    }
    
    return Math.min(1, weight);
  }

  /**
   * Normalize raw sentiment score to -1 to +1 range
   */
  private normalizeScore(rawScore: number, wordCount: number): number {
    if (wordCount === 0) return 0;
    
    // Comparative score is already normalized per word
    // Scale it to roughly -1 to +1 range
    const comparative = rawScore / wordCount;
    
    // Clamp to -1 to +1
    return Math.max(-1, Math.min(1, comparative / 3));
  }

  /**
   * Analyze sentiment of a single tweet
   */
  analyzeTweet(tweet: Tweet): TweetSentiment {
    const result = this.sentiment.analyze(tweet.text);
    
    const wordCount = tweet.text.split(/\s+/).length;
    const normalizedScore = this.normalizeScore(result.score, wordCount);
    const engagementWeight = this.calculateEngagementWeight(tweet);
    const credibilityWeight = this.calculateCredibilityWeight(tweet);
    
    // Combined weight
    const combinedWeight = (engagementWeight + credibilityWeight) / 2;
    const weightedScore = normalizedScore * combinedWeight;

    return {
      tweet,
      rawScore: result.score,
      normalizedScore,
      comparative: result.comparative,
      positiveWords: result.positive,
      negativeWords: result.negative,
      engagementWeight,
      credibilityWeight,
      weightedScore,
    };
  }

  /**
   * Analyze sentiment of multiple tweets and aggregate
   */
  analyzeTweets(tweets: Tweet[], symbol: string): AggregatedSentiment {
    const sentiments = tweets.map(tweet => this.analyzeTweet(tweet));

    if (sentiments.length === 0) {
      return {
        symbol,
        tweetCount: 0,
        averageScore: 0,
        weightedAverageScore: 0,
        medianScore: 0,
        scoreDistribution: {
          veryNegative: 0,
          negative: 0,
          neutral: 0,
          positive: 0,
          veryPositive: 0,
        },
        topPositive: [],
        topNegative: [],
        timestamp: new Date(),
      };
    }

    // Calculate averages
    const scores = sentiments.map(s => s.normalizedScore);
    const weightedScores = sentiments.map(s => s.weightedScore);
    const weights = sentiments.map(s => (s.engagementWeight + s.credibilityWeight) / 2);

    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    const averageScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
    const weightedAverageScore = totalWeight > 0
      ? weightedScores.reduce((sum, s) => sum + s, 0) / totalWeight
      : 0;

    // Calculate median
    const sortedScores = [...scores].sort((a, b) => a - b);
    const medianScore = sortedScores.length % 2 === 0
      ? (sortedScores[sortedScores.length / 2 - 1] + sortedScores[sortedScores.length / 2]) / 2
      : sortedScores[Math.floor(sortedScores.length / 2)];

    // Calculate distribution
    const distribution = {
      veryNegative: 0,
      negative: 0,
      neutral: 0,
      positive: 0,
      veryPositive: 0,
    };

    for (const score of scores) {
      if (score < -0.5) distribution.veryNegative++;
      else if (score < -0.1) distribution.negative++;
      else if (score <= 0.1) distribution.neutral++;
      else if (score <= 0.5) distribution.positive++;
      else distribution.veryPositive++;
    }

    // Get top positive and negative tweets
    const sorted = [...sentiments].sort((a, b) => b.normalizedScore - a.normalizedScore);
    const topPositive = sorted.slice(0, 3).filter(s => s.normalizedScore > 0);
    const topNegative = sorted.slice(-3).reverse().filter(s => s.normalizedScore < 0);

    logger.info('Sentiment analysis completed', {
      symbol,
      tweetCount: tweets.length,
      averageScore: averageScore.toFixed(3),
      weightedAverageScore: weightedAverageScore.toFixed(3),
    });

    return {
      symbol,
      tweetCount: sentiments.length,
      averageScore,
      weightedAverageScore,
      medianScore,
      scoreDistribution: distribution,
      topPositive,
      topNegative,
      timestamp: new Date(),
    };
  }

  /**
   * Get sentiment label from score
   */
  static getSentimentLabel(score: number): 'very_negative' | 'negative' | 'neutral' | 'positive' | 'very_positive' {
    if (score < -0.5) return 'very_negative';
    if (score < -0.1) return 'negative';
    if (score <= 0.1) return 'neutral';
    if (score <= 0.5) return 'positive';
    return 'very_positive';
  }
}

// Singleton instance
let sentimentScorerInstance: SentimentScorer | null = null;

export function getSentimentScorer(): SentimentScorer {
  if (!sentimentScorerInstance) {
    sentimentScorerInstance = new SentimentScorer();
  }
  return sentimentScorerInstance;
}

export function resetSentimentScorer(): void {
  sentimentScorerInstance = null;
}
