/**
 * Twitter Sentiment Data Collector
 * 
 * Collects tweets mentioning token symbols using Twitter API v2.
 * Features:
 * - Bearer Token authentication
 * - Recent tweet search (/2/tweets/search/recent)
 * - Engagement metrics (likes, retweets, replies)
 * - Rate limit handling (450 requests/15min)
 * - Result caching
 */
import { logger } from '../logger.js';

// ============= TYPES =============
export interface TwitterCredentials {
  bearerToken: string;
  apiKey?: string;
  apiSecret?: string;
  clientId?: string;
}

export interface TweetMetrics {
  retweet_count: number;
  reply_count: number;
  like_count: number;
  quote_count: number;
  impression_count?: number;
}

export interface TweetAuthor {
  id: string;
  name: string;
  username: string;
  verified?: boolean;
  followers_count: number;
  following_count: number;
  tweet_count: number;
  description?: string;
}

export interface Tweet {
  id: string;
  text: string;
  author_id: string;
  author?: TweetAuthor;
  created_at: string;
  public_metrics: TweetMetrics;
  lang?: string;
}

export interface TwitterSearchResult {
  tweets: Tweet[];
  meta: {
    newest_id?: string;
    oldest_id?: string;
    result_count: number;
    next_token?: string;
  };
  rateLimitRemaining: number;
  rateLimitReset: Date;
}

export interface CachedResult {
  data: TwitterSearchResult;
  timestamp: number;
  symbol: string;
}

// ============= CONSTANTS =============
const TWITTER_API_BASE = 'https://api.twitter.com';
const CACHE_TTL_MS = 60 * 1000; // 1 minute cache

/**
 * X API v2 Rate Limits for /2/tweets/search/recent
 * See: https://docs.x.com/x-api/fundamentals/rate-limits
 *
 * Pro Plan:
 *   - App-only auth (Bearer Token): 450 requests / 15 min
 *   - Per user (OAuth): 300 requests / 15 min
 *
 * Basic Plan:
 *   - App-only auth (Bearer Token): 60 requests / 15 min
 *   - Per user (OAuth): 60 requests / 15 min
 *
 * Free Plan:
 *   - Very limited access, search not available
 */
export type TwitterApiPlan = 'free' | 'basic' | 'pro';

const RATE_LIMITS: Record<TwitterApiPlan, { appOnly: number; perUser: number }> = {
  free: { appOnly: 1, perUser: 1 },      // Minimal - search may not be available
  basic: { appOnly: 60, perUser: 60 },   // 60 requests per 15 min
  pro: { appOnly: 450, perUser: 300 },   // 450 app-only, 300 per-user per 15 min
};

const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes

// ============= ERROR CLASSES =============
export class RateLimitError extends Error {
  public readonly resetTime: Date;
  public readonly plan: TwitterApiPlan;

  constructor(message: string, resetTime: Date, plan: TwitterApiPlan) {
    super(message);
    this.name = 'RateLimitError';
    this.resetTime = resetTime;
    this.plan = plan;
  }

  getWaitMs(): number {
    return Math.max(0, this.resetTime.getTime() - Date.now());
  }
}

// ============= TWITTER COLLECTOR CLASS =============
export class TwitterCollector {
  private credentials: TwitterCredentials;
  private cache: Map<string, CachedResult> = new Map();
  private requestTimestamps: number[] = [];
  private rateLimitRemaining: number;
  private rateLimitReset: Date = new Date();
  private plan: TwitterApiPlan;
  private maxRequests: number;

  constructor(credentials?: TwitterCredentials, plan: TwitterApiPlan = 'basic') {
    // Load from environment if not provided
    this.credentials = credentials || {
      bearerToken: process.env.TWITTER_BEARER_TOKEN || '',
      apiKey: process.env.TWITTER_API_KEY,
      apiSecret: process.env.TWITTER_API_SECRET,
      clientId: process.env.TWITTER_CLIENT_ID,
    };

    // Set plan and rate limits
    this.plan = plan;
    this.maxRequests = RATE_LIMITS[plan].appOnly; // Using app-only (Bearer Token) limits
    this.rateLimitRemaining = this.maxRequests;

    if (!this.credentials.bearerToken) {
      logger.warn('Twitter Bearer Token not configured - Twitter collector will not work');
    } else {
      logger.info('TwitterCollector initialized', {
        plan: this.plan,
        maxRequestsPer15Min: this.maxRequests,
        hasApiKey: !!this.credentials.apiKey,
        hasClientId: !!this.credentials.clientId,
      });
    }
  }

  /**
   * Check if we can make a request (rate limiting)
   */
  private canMakeRequest(): boolean {
    const now = Date.now();

    // If rate limit reset time has passed, reset counter
    if (now > this.rateLimitReset.getTime()) {
      this.rateLimitRemaining = this.maxRequests;
      this.rateLimitReset = new Date(now + RATE_LIMIT_WINDOW_MS);
    }

    // Clean old timestamps outside the window
    this.requestTimestamps = this.requestTimestamps.filter(
      ts => now - ts < RATE_LIMIT_WINDOW_MS
    );

    return this.requestTimestamps.length < this.maxRequests && this.rateLimitRemaining > 0;
  }

  /**
   * Check rate limit and throw immediately if exceeded (no waiting)
   * This allows the caller to handle the rate limit gracefully
   */
  private checkRateLimit(): void {
    if (this.canMakeRequest()) return;

    const waitTime = Math.max(0, this.rateLimitReset.getTime() - Date.now());
    const waitMinutes = Math.ceil(waitTime / 60000);

    // Throw immediately instead of waiting - let the caller handle it
    throw new RateLimitError(
      `X API rate limit exceeded. Reset in ${waitMinutes} minutes at ${this.rateLimitReset.toISOString()}`,
      this.rateLimitReset,
      this.plan
    );
  }

  /**
   * Make an authenticated request to Twitter API
   */
  private async makeRequest<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    // Check rate limit first - throws immediately if exceeded (no waiting)
    this.checkRateLimit();

    const url = new URL(`${TWITTER_API_BASE}${endpoint}`);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, value);
      });
    }

    logger.debug('Twitter API request', { endpoint, params });

    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Bearer ${this.credentials.bearerToken}`,
        'Content-Type': 'application/json',
      },
    });

    // Track request for rate limiting
    this.requestTimestamps.push(Date.now());

    // Update rate limit info from headers (always, even on error)
    const remaining = response.headers.get('x-rate-limit-remaining');
    const reset = response.headers.get('x-rate-limit-reset');

    if (remaining !== null) this.rateLimitRemaining = parseInt(remaining, 10);
    if (reset !== null) this.rateLimitReset = new Date(parseInt(reset, 10) * 1000);

    // Handle rate limit error specifically
    if (response.status === 429) {
      // Set remaining to 0 since we're rate limited
      this.rateLimitRemaining = 0;

      // Parse reset time from header or default to 15 minutes
      const resetTime = reset
        ? new Date(parseInt(reset, 10) * 1000)
        : new Date(Date.now() + RATE_LIMIT_WINDOW_MS);
      this.rateLimitReset = resetTime;

      const waitMinutes = Math.ceil((resetTime.getTime() - Date.now()) / 60000);

      throw new RateLimitError(
        `X API rate limit exceeded. Reset in ${waitMinutes} minutes at ${resetTime.toISOString()}`,
        resetTime,
        this.plan
      );
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Twitter API error ${response.status}: ${errorText}`);
    }

    return response.json() as Promise<T>;
  }

  /**
   * Search recent tweets mentioning a symbol
   */
  async searchTweets(
    symbol: string,
    options: {
      maxResults?: number;
      sinceId?: string;
      untilId?: string;
    } = {}
  ): Promise<TwitterSearchResult> {
    const { maxResults = 100, sinceId, untilId } = options;

    // Check cache first
    const cacheKey = `${symbol}:${maxResults}:${sinceId || ''}:${untilId || ''}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      logger.debug('Using cached Twitter data', { symbol, age: Date.now() - cached.timestamp });
      return cached.data;
    }

    if (!this.credentials.bearerToken) {
      throw new Error('Twitter Bearer Token not configured');
    }

    // Build query - search for symbol mentions
    // Note: $ cashtag operator requires Pro plan, so we use simple text search for Basic plan
    const query = this.plan === 'pro'
      ? `$${symbol} OR #${symbol} -is:retweet lang:en`
      : `${symbol} OR #${symbol} -is:retweet lang:en`;

    const params: Record<string, string> = {
      query,
      max_results: Math.min(maxResults, 100).toString(),
      'tweet.fields': 'author_id,created_at,public_metrics,lang',
      'user.fields': 'name,username,verified,public_metrics,description',
      expansions: 'author_id',
    };

    if (sinceId) params.since_id = sinceId;
    if (untilId) params.until_id = untilId;

    interface TwitterApiResponse {
      data?: Array<{
        id: string;
        text: string;
        author_id: string;
        created_at: string;
        public_metrics: TweetMetrics;
        lang?: string;
      }>;
      includes?: {
        users?: Array<{
          id: string;
          name: string;
          username: string;
          verified?: boolean;
          public_metrics: {
            followers_count: number;
            following_count: number;
            tweet_count: number;
          };
          description?: string;
        }>;
      };
      meta: {
        newest_id?: string;
        oldest_id?: string;
        result_count: number;
        next_token?: string;
      };
    }

    const response = await this.makeRequest<TwitterApiResponse>(
      '/2/tweets/search/recent',
      params
    );

    // Map authors by ID for easy lookup
    const authorsById = new Map<string, TweetAuthor>();
    if (response.includes?.users) {
      for (const user of response.includes.users) {
        authorsById.set(user.id, {
          id: user.id,
          name: user.name,
          username: user.username,
          verified: user.verified,
          followers_count: user.public_metrics.followers_count,
          following_count: user.public_metrics.following_count,
          tweet_count: user.public_metrics.tweet_count,
          description: user.description,
        });
      }
    }

    // Build tweets with author info
    const tweets: Tweet[] = (response.data || []).map(tweet => ({
      ...tweet,
      author: authorsById.get(tweet.author_id),
    }));

    const result: TwitterSearchResult = {
      tweets,
      meta: response.meta,
      rateLimitRemaining: this.rateLimitRemaining,
      rateLimitReset: this.rateLimitReset,
    };

    // Cache the result
    this.cache.set(cacheKey, {
      data: result,
      timestamp: Date.now(),
      symbol,
    });

    logger.info('Twitter search completed', {
      symbol,
      tweetCount: tweets.length,
      rateLimitRemaining: this.rateLimitRemaining,
    });

    return result;
  }

  /**
   * Get aggregate statistics for tweets
   */
  getAggregateStats(tweets: Tweet[]): {
    totalTweets: number;
    totalLikes: number;
    totalRetweets: number;
    totalReplies: number;
    avgEngagement: number;
    verifiedCount: number;
    totalFollowers: number;
  } {
    let totalLikes = 0;
    let totalRetweets = 0;
    let totalReplies = 0;
    let verifiedCount = 0;
    let totalFollowers = 0;

    for (const tweet of tweets) {
      totalLikes += tweet.public_metrics.like_count;
      totalRetweets += tweet.public_metrics.retweet_count;
      totalReplies += tweet.public_metrics.reply_count;

      if (tweet.author?.verified) verifiedCount++;
      if (tweet.author?.followers_count) totalFollowers += tweet.author.followers_count;
    }

    const avgEngagement = tweets.length > 0
      ? (totalLikes + totalRetweets + totalReplies) / tweets.length
      : 0;

    return {
      totalTweets: tweets.length,
      totalLikes,
      totalRetweets,
      totalReplies,
      avgEngagement,
      verifiedCount,
      totalFollowers,
    };
  }

  /**
   * Get rate limit status
   */
  getRateLimitStatus(): {
    remaining: number;
    reset: Date;
    canMakeRequest: boolean;
    plan: TwitterApiPlan;
    maxRequestsPer15Min: number;
  } {
    return {
      remaining: this.rateLimitRemaining,
      reset: this.rateLimitReset,
      canMakeRequest: this.canMakeRequest(),
      plan: this.plan,
      maxRequestsPer15Min: this.maxRequests,
    };
  }

  /**
   * Get the API plan being used
   */
  getPlan(): TwitterApiPlan {
    return this.plan;
  }

  /**
   * Clear the cache
   */
  clearCache(): void {
    this.cache.clear();
    logger.debug('Twitter cache cleared');
  }
}

// Singleton instance
let twitterCollectorInstance: TwitterCollector | null = null;

export function getTwitterCollector(plan?: TwitterApiPlan): TwitterCollector {
  if (!twitterCollectorInstance) {
    // Check environment for plan, default to 'basic'
    const envPlan = (process.env.TWITTER_API_PLAN as TwitterApiPlan) || plan || 'basic';
    twitterCollectorInstance = new TwitterCollector(undefined, envPlan);
  }
  return twitterCollectorInstance;
}

export function resetTwitterCollector(): void {
  twitterCollectorInstance = null;
}
