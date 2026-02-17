/**
 * Multi-Source Sentiment Aggregator
 *
 * Combines sentiment data from multiple sources:
 * - Twitter (35% weight)
 * - CryptoPanic (30% weight)
 * - Telegram (15% weight)
 * - Discord (20% weight)
 *
 * Features:
 * - Weighted sentiment scoring
 * - Confidence calculation based on data availability
 * - Signal generation (BULLISH/BEARISH/NEUTRAL)
 * - Graceful fallback when sources are unavailable
 */
import { logger } from '../logger.js';
import { getSentimentAnalyst, type SentimentAnalysis } from './sentimentAnalyst.js';
import { getCryptoPanicCollector, type CryptoPanicSentiment, type NewsItem, QuotaExceededError } from './cryptopanicCollector.js';
import { getTelegramCollector, type TelegramSentiment } from './telegramCollector.js';
import { getDiscordCollector, type DiscordSentiment } from './discordCollector.js';

// ============= TYPES =============
export type SentimentSignal = 'BULLISH' | 'BEARISH' | 'NEUTRAL';

export interface SourceSentiment {
  score: number; // -1 to 1
  volume: number;
  velocity?: number;
  available: boolean;
  error?: string;
}

export interface TwitterSourceData extends SourceSentiment {
  tweetCount: number;
  credibilityScore: number;
}

export interface CryptoPanicSourceData extends SourceSentiment {
  bullishVotes: number;
  bearishVotes: number;
  newsCount: number;
  topNews?: NewsItem[];
}

export interface TelegramSourceData extends SourceSentiment {
  messageCount: number;
  channelCount: number;
}

export interface DiscordSourceData extends SourceSentiment {
  messageCount: number;
  channelCount: number;
  guildCount: number;
}

export interface MultiSourceSentiment {
  token: string;
  overallScore: number; // -1 to 1 (weighted average)
  sources: {
    twitter: TwitterSourceData;
    cryptopanic: CryptoPanicSourceData;
    telegram: TelegramSourceData;
    discord: DiscordSourceData;
  };
  confidence: number; // 0 to 1 (based on data availability and agreement)
  signal: SentimentSignal;
  timestamp: number;
}

// ============= CONSTANTS =============
const WEIGHTS = {
  twitter: 0.35,      // 35% weight
  cryptopanic: 0.30,  // 30% weight
  telegram: 0.15,     // 15% weight
  discord: 0.20,      // 20% weight
};

const SIGNAL_THRESHOLDS = {
  bullish: 0.3,   // Score > 0.3 = BULLISH
  bearish: -0.3,  // Score < -0.3 = BEARISH
  // Between -0.3 and 0.3 = NEUTRAL
};

const MIN_CONFIDENCE_THRESHOLD = 0.4; // Minimum confidence to generate signal

// ============= MULTI-SOURCE AGGREGATOR CLASS =============
export class MultiSourceAggregator {
  private sentimentAnalyst = getSentimentAnalyst();
  private cryptoPanicCollector = getCryptoPanicCollector();
  private telegramCollector = getTelegramCollector();
  private discordCollector = getDiscordCollector();

  constructor() {
    logger.info('MultiSourceAggregator initialized', {
      weights: WEIGHTS,
      signalThresholds: SIGNAL_THRESHOLDS,
    });
  }

  /**
   * Fetch Twitter sentiment
   */
  private async fetchTwitterSentiment(token: string): Promise<TwitterSourceData> {
    try {
      const analysis: SentimentAnalysis = await this.sentimentAnalyst.analyze(token, { maxTweets: 100 });

      // Convert velocity string to number: accelerating=1, stable=0, declining=-1
      const velocityMap: Record<string, number> = {
        'accelerating': 1,
        'stable': 0,
        'declining': -1,
      };
      const velocityValue = velocityMap[analysis.velocity] ?? 0;

      return {
        score: analysis.score,
        volume: analysis.rawData.tweetCount,
        velocity: velocityValue,
        available: true,
        tweetCount: analysis.rawData.tweetCount,
        credibilityScore: analysis.credibilityScore,
      };
    } catch (error) {
      logger.warn('Twitter sentiment fetch failed', { token, error: (error as Error).message });
      return {
        score: 0,
        volume: 0,
        available: false,
        error: (error as Error).message,
        tweetCount: 0,
        credibilityScore: 0,
      };
    }
  }

  /**
   * Fetch CryptoPanic sentiment
   */
  private async fetchCryptoPanicSentiment(token: string): Promise<CryptoPanicSourceData> {
    try {
      const sentiment: CryptoPanicSentiment = await this.cryptoPanicCollector.fetchPosts(token);

      return {
        score: sentiment.sentimentScore,
        volume: sentiment.newsCount,
        available: true,
        bullishVotes: sentiment.bullishVotes,
        bearishVotes: sentiment.bearishVotes,
        newsCount: sentiment.newsCount,
        topNews: sentiment.topNews, // Include top news items
      };
    } catch (error) {
      if (error instanceof QuotaExceededError) {
        logger.warn('CryptoPanic quota exceeded - falling back to other sources', {
          token,
          resetDate: error.resetDate.toISOString(),
        });
      } else {
        logger.warn('CryptoPanic sentiment fetch failed', { token, error: (error as Error).message });
      }

      return {
        score: 0,
        volume: 0,
        available: false,
        error: (error as Error).message,
        bullishVotes: 0,
        bearishVotes: 0,
        newsCount: 0,
        topNews: [], // Empty array when unavailable
      };
    }
  }

  /**
   * Fetch Telegram sentiment
   */
  private async fetchTelegramSentiment(token: string): Promise<TelegramSourceData> {
    try {
      const sentiment: TelegramSentiment = await this.telegramCollector.fetchSentiment(token);

      return {
        score: sentiment.averageSentiment,
        volume: sentiment.volume24h,
        available: true,
        messageCount: sentiment.totalMessages,
        channelCount: sentiment.channels.length,
      };
    } catch (error) {
      logger.warn('Telegram sentiment fetch failed', { token, error: (error as Error).message });
      return {
        score: 0,
        volume: 0,
        available: false,
        error: (error as Error).message,
        messageCount: 0,
        channelCount: 0,
      };
    }
  }

  /**
   * Fetch Discord sentiment
   */
  private async fetchDiscordSentiment(token: string): Promise<DiscordSourceData> {
    try {
      if (!this.discordCollector.isConfigured()) {
        return {
          score: 0,
          volume: 0,
          available: false,
          error: 'Discord not configured',
          messageCount: 0,
          channelCount: 0,
          guildCount: 0,
        };
      }

      const sentiment: DiscordSentiment = await this.discordCollector.fetchSentiment(token);

      const uniqueGuilds = new Set(sentiment.channels.map(c => c.guildId));

      return {
        score: sentiment.averageSentiment,
        volume: sentiment.volume24h,
        available: true,
        messageCount: sentiment.totalMessages,
        channelCount: sentiment.channels.length,
        guildCount: uniqueGuilds.size,
      };
    } catch (error) {
      logger.warn('Discord sentiment fetch failed', { token, error: (error as Error).message });
      return {
        score: 0,
        volume: 0,
        available: false,
        error: (error as Error).message,
        messageCount: 0,
        channelCount: 0,
        guildCount: 0,
      };
    }
  }

  /**
   * Calculate confidence based on data availability and agreement
   */
  private calculateConfidence(
    twitter: TwitterSourceData,
    cryptopanic: CryptoPanicSourceData,
    telegram: TelegramSourceData,
    discord: DiscordSourceData
  ): number {
    let availabilityScore = 0;
    let availableCount = 0;

    if (twitter.available) { availabilityScore += WEIGHTS.twitter; availableCount++; }
    if (cryptopanic.available) { availabilityScore += WEIGHTS.cryptopanic; availableCount++; }
    if (telegram.available) { availabilityScore += WEIGHTS.telegram; availableCount++; }
    if (discord.available) { availabilityScore += WEIGHTS.discord; availableCount++; }

    if (availableCount === 0) return 0;

    // Agreement score
    const scores: number[] = [];
    if (twitter.available) scores.push(twitter.score);
    if (cryptopanic.available) scores.push(cryptopanic.score);
    if (telegram.available) scores.push(telegram.score);
    if (discord.available) scores.push(discord.score);

    const mean = scores.reduce((sum, s) => sum + s, 0) / scores.length;
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    const agreementScore = Math.max(0, 1 - stdDev);

    // Volume boost
    let volumeBoost = 0;
    if (twitter.available && twitter.tweetCount > 50) volumeBoost += 0.075;
    if (cryptopanic.available && cryptopanic.newsCount > 10) volumeBoost += 0.075;
    if (telegram.available && telegram.messageCount > 100) volumeBoost += 0.075;
    if (discord.available && discord.messageCount > 50) volumeBoost += 0.075;

    const confidence = Math.min(1, (
      availabilityScore * 0.5 +
      agreementScore * 0.4 +
      volumeBoost
    ));

    return confidence;
  }

  /**
   * Determine signal from overall score
   */
  private determineSignal(score: number, confidence: number): SentimentSignal {
    // If confidence too low, return NEUTRAL
    if (confidence < MIN_CONFIDENCE_THRESHOLD) {
      return 'NEUTRAL';
    }

    if (score > SIGNAL_THRESHOLDS.bullish) {
      return 'BULLISH';
    } else if (score < SIGNAL_THRESHOLDS.bearish) {
      return 'BEARISH';
    } else {
      return 'NEUTRAL';
    }
  }

  /**
   * Aggregate sentiment from all sources
   */
  async aggregateSentiment(token: string): Promise<MultiSourceSentiment> {
    logger.info('Aggregating multi-source sentiment', { token });

    // Fetch from all sources in parallel
    const [twitter, cryptopanic, telegram, discord] = await Promise.all([
      this.fetchTwitterSentiment(token),
      this.fetchCryptoPanicSentiment(token),
      this.fetchTelegramSentiment(token),
      this.fetchDiscordSentiment(token),
    ]);

    // Calculate weighted average score
    let weightedScore = 0;
    let totalWeight = 0;

    if (twitter.available) {
      weightedScore += twitter.score * WEIGHTS.twitter;
      totalWeight += WEIGHTS.twitter;
    }
    if (cryptopanic.available) {
      weightedScore += cryptopanic.score * WEIGHTS.cryptopanic;
      totalWeight += WEIGHTS.cryptopanic;
    }
    if (telegram.available) {
      weightedScore += telegram.score * WEIGHTS.telegram;
      totalWeight += WEIGHTS.telegram;
    }
    if (discord.available) {
      weightedScore += discord.score * WEIGHTS.discord;
      totalWeight += WEIGHTS.discord;
    }

    const overallScore = totalWeight > 0 ? weightedScore / totalWeight : 0;
    const confidence = this.calculateConfidence(twitter, cryptopanic, telegram, discord);
    const signal = this.determineSignal(overallScore, confidence);

    const result: MultiSourceSentiment = {
      token,
      overallScore,
      sources: {
        twitter,
        cryptopanic,
        telegram,
        discord,
      },
      confidence,
      signal,
      timestamp: Date.now(),
    };

    logger.info('Multi-source sentiment aggregated', {
      token,
      overallScore: overallScore.toFixed(3),
      confidence: confidence.toFixed(3),
      signal,
      sourcesAvailable: {
        twitter: twitter.available,
        cryptopanic: cryptopanic.available,
        telegram: telegram.available,
        discord: discord.available,
      },
    });

    return result;
  }
}

// ============= SINGLETON =============
let instance: MultiSourceAggregator | null = null;

export function getMultiSourceAggregator(): MultiSourceAggregator {
  if (!instance) {
    instance = new MultiSourceAggregator();
  }
  return instance;
}
