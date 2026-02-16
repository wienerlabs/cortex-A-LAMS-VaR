/**
 * Multi-Source Sentiment Analysis Module
 *
 * Provides sentiment analysis from multiple sources:
 * - Twitter
 * - CryptoPanic
 * - Telegram
 */

// Twitter data collection
export {
  TwitterCollector,
  getTwitterCollector,
  resetTwitterCollector,
  RateLimitError,
  type TwitterCredentials,
  type TwitterApiPlan,
  type TweetMetrics,
  type TweetAuthor,
  type Tweet,
  type TwitterSearchResult,
  type CachedResult,
} from './twitterCollector.js';

// NLP sentiment scoring
export {
  SentimentScorer,
  getSentimentScorer,
  resetSentimentScorer,
  type TweetSentiment,
  type AggregatedSentiment,
} from './sentimentScorer.js';

// High-level sentiment analysis
export {
  SentimentAnalyst,
  getSentimentAnalyst,
  resetSentimentAnalyst,
  type SentimentVelocity,
  type SentimentSignal,
  type SentimentAnalysis,
  type HistoricalDataPoint,
  type SentimentAnalystConfig,
} from './sentimentAnalyst.js';

// Trading strategy integration
export {
  SentimentIntegration,
  getSentimentIntegration,
  initializeSentimentIntegration,
  normalizeSentiment,
  combineScores,
  getSentimentWeight,
  DEFAULT_SENTIMENT_CONFIG,
  type SentimentIntegrationConfig,
  type StrategyType,
  type SentimentAdjustedScore,
} from './sentimentIntegration.js';

// CryptoPanic data collection
export {
  CryptoPanicCollector,
  getCryptoPanicCollector,
  QuotaExceededError,
  CryptoPanicRateLimitError,
  type CryptoPanicPost,
  type CryptoPanicResponse,
  type CryptoPanicSentiment,
  type CachedCryptoPanicResult,
} from './cryptopanicCollector.js';

// Telegram data collection
export {
  TelegramCollector,
  getTelegramCollector,
  type TelegramMessage,
  type TelegramUpdate,
  type TelegramChannelData,
  type TelegramSentiment,
  type CachedTelegramResult,
} from './telegramCollector.js';

// Discord data collection
export {
  DiscordCollector,
  getDiscordCollector,
  resetDiscordCollector,
  type DiscordMessage,
  type DiscordChannelData,
  type DiscordSentiment,
  type CachedDiscordResult,
  type DiscordChannelConfig,
} from './discordCollector.js';

// Multi-source aggregation
export {
  MultiSourceAggregator,
  getMultiSourceAggregator,
  type SentimentSignal as MultiSourceSentimentSignal,
  type SourceSentiment,
  type TwitterSourceData,
  type CryptoPanicSourceData,
  type TelegramSourceData,
  type MultiSourceSentiment,
  type DiscordSourceData,
} from './multiSourceAggregator.js';
