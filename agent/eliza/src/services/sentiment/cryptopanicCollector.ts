/**
 * CryptoPanic Sentiment Data Collector
 *
 * Collects news and community sentiment from CryptoPanic API.
 * Features:
 * - News articles with titles and descriptions
 * - Rate limiting (2 req/sec)
 * - Monthly quota tracking (100 req/month)
 * - Aggressive caching (30 min TTL)
 * - Graceful fallback on quota exceeded
 *
 * API Plan: Developer v2
 * - Endpoint: /api/developer/v2/posts/
 * - Rate Limit: 2 req/sec
 * - Monthly Quota: 100 requests
 * - History: 20 items per request
 * - News Delay: 24 hours (not real-time)
 *
 * Available Fields (Developer Plan):
 * ✅ id, slug, title, description, published_at, created_at, kind
 *
 * NOT Available (requires Business/Enterprise):
 * ❌ url, image, author, source, sentiment, votes, panic_score, metadata
 */
import { logger } from '../logger.js';

// ============= TYPES =============
export interface CryptoPanicPost {
  id: number;
  slug: string;
  title: string;
  description: string;
  published_at: string;
  created_at: string;
  kind: 'news' | 'media';
  // Optional fields that may not be in v2 API
  url?: string;
  domain?: string;
  votes?: {
    negative: number;
    positive: number;
    important: number;
    liked: number;
    disliked: number;
    lol: number;
    toxic: number;
    saved: number;
    comments: number;
  };
  source?: {
    title: string;
    region: string;
    domain: string;
    path: string | null;
  };
  currencies?: Array<{
    code: string;
    title: string;
    slug: string;
    url: string;
  }>;
}

export interface CryptoPanicResponse {
  count: number;
  next: string | null;
  previous: string | null;
  results: CryptoPanicPost[];
}

export interface NewsItem {
  title: string;
  description: string;
  url: string;
  votes: {
    positive: number;
    negative: number;
  };
  totalVotes: number;
  publishedAt: string;
}

export interface CryptoPanicSentiment {
  token: string;
  posts: CryptoPanicPost[];
  bullishVotes: number;
  bearishVotes: number;
  newsCount: number;
  sentimentScore: number; // -1 to 1
  topNews: NewsItem[]; // Top 5 news items sorted by vote count
  timestamp: number;
}

export interface CachedCryptoPanicResult {
  data: CryptoPanicSentiment;
  timestamp: number;
  token: string;
}

// ============= CONSTANTS =============
const CRYPTOPANIC_API_BASE = 'https://cryptopanic.com/api/developer/v2';
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes cache (aggressive due to quota)
const RATE_LIMIT_PER_SEC = 2; // 2 requests per second
const RATE_LIMIT_WINDOW_MS = 1000; // 1 second
const MONTHLY_QUOTA = 100; // 100 requests per month
const QUOTA_RESET_DAY = 1; // Reset on 1st of each month

// ============= ERROR CLASSES =============
export class QuotaExceededError extends Error {
  public readonly resetDate: Date;
  public readonly requestsUsed: number;
  public readonly monthlyQuota: number;

  constructor(message: string, resetDate: Date, requestsUsed: number, monthlyQuota: number) {
    super(message);
    this.name = 'QuotaExceededError';
    this.resetDate = resetDate;
    this.requestsUsed = requestsUsed;
    this.monthlyQuota = monthlyQuota;
  }
}

export class CryptoPanicRateLimitError extends Error {
  public readonly retryAfterMs: number;

  constructor(message: string, retryAfterMs: number) {
    super(message);
    this.name = 'CryptoPanicRateLimitError';
    this.retryAfterMs = retryAfterMs;
  }
}

// ============= CRYPTOPANIC COLLECTOR CLASS =============
export class CryptoPanicCollector {
  private apiKey: string;
  private cache: Map<string, CachedCryptoPanicResult> = new Map();
  private requestTimestamps: number[] = [];
  private monthlyRequestCount: number = 0;
  private currentMonth: number;
  private quotaResetDate: Date;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || process.env.CRYPTOPANIC_API_KEY || '';

    // Initialize quota tracking
    const now = new Date();
    this.currentMonth = now.getMonth();
    this.quotaResetDate = this.getNextResetDate(now);

    // Load request count from persistent storage (simplified - in production use DB)
    this.monthlyRequestCount = 0;

    if (!this.apiKey) {
      logger.warn('CryptoPanic API key not configured - CryptoPanic collector will not work');
    } else {
      logger.info('CryptoPanicCollector initialized', {
        monthlyQuota: MONTHLY_QUOTA,
        quotaResetDate: this.quotaResetDate.toISOString(),
        cacheMinutes: CACHE_TTL_MS / 60000,
      });
    }
  }

  /**
   * Get next quota reset date (1st of next month)
   */
  private getNextResetDate(now: Date): Date {
    const nextMonth = new Date(now.getFullYear(), now.getMonth() + 1, QUOTA_RESET_DAY);
    return nextMonth;
  }

  /**
   * Check if quota has reset (new month)
   */
  private checkQuotaReset(): void {
    const now = new Date();
    if (now.getMonth() !== this.currentMonth) {
      this.currentMonth = now.getMonth();
      this.monthlyRequestCount = 0;
      this.quotaResetDate = this.getNextResetDate(now);
      logger.info('CryptoPanic quota reset', {
        newMonth: this.currentMonth,
        resetDate: this.quotaResetDate.toISOString(),
      });
    }
  }

  /**
   * Check if we can make a request (rate limiting + quota)
   */
  private canMakeRequest(): { allowed: boolean; reason?: string } {
    // Check quota reset
    this.checkQuotaReset();

    // Check monthly quota
    if (this.monthlyRequestCount >= MONTHLY_QUOTA) {
      return {
        allowed: false,
        reason: `Monthly quota exceeded (${this.monthlyRequestCount}/${MONTHLY_QUOTA})`,
      };
    }

    // Check rate limit (2 req/sec)
    const now = Date.now();
    this.requestTimestamps = this.requestTimestamps.filter(
      ts => now - ts < RATE_LIMIT_WINDOW_MS
    );

    if (this.requestTimestamps.length >= RATE_LIMIT_PER_SEC) {
      return {
        allowed: false,
        reason: 'Rate limit exceeded (2 req/sec)',
      };
    }

    return { allowed: true };
  }

  /**
   * Wait for rate limit window
   */
  private async waitForRateLimit(): Promise<void> {
    const now = Date.now();
    this.requestTimestamps = this.requestTimestamps.filter(
      ts => now - ts < RATE_LIMIT_WINDOW_MS
    );

    if (this.requestTimestamps.length >= RATE_LIMIT_PER_SEC) {
      const oldestTimestamp = Math.min(...this.requestTimestamps);
      const waitTime = RATE_LIMIT_WINDOW_MS - (now - oldestTimestamp) + 100; // +100ms buffer
      logger.debug('CryptoPanic rate limit wait', { waitMs: waitTime });
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }

  /**
   * Make an authenticated request to CryptoPanic API
   */
  private async makeRequest<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    // Check quota and rate limits
    const check = this.canMakeRequest();
    if (!check.allowed) {
      if (check.reason?.includes('quota')) {
        throw new QuotaExceededError(
          check.reason,
          this.quotaResetDate,
          this.monthlyRequestCount,
          MONTHLY_QUOTA
        );
      }
      // Rate limit - wait and retry
      await this.waitForRateLimit();
    }

    const url = new URL(`${CRYPTOPANIC_API_BASE}${endpoint}`);
    url.searchParams.append('auth_token', this.apiKey);

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, value);
      });
    }

    logger.debug('CryptoPanic API request', { endpoint, params });

    const response = await fetch(url.toString(), {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Track request
    this.requestTimestamps.push(Date.now());
    this.monthlyRequestCount++;

    if (response.status === 429) {
      throw new CryptoPanicRateLimitError(
        'CryptoPanic rate limit exceeded',
        RATE_LIMIT_WINDOW_MS
      );
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`CryptoPanic API error ${response.status}: ${errorText}`);
    }

    const data = await response.json() as T;
    logger.debug('CryptoPanic API response', {
      endpoint,
      status: response.status,
      hasData: !!data,
      dataKeys: data ? Object.keys(data) : []
    });
    return data;
  }

  /**
   * Fetch news posts for a token
   */
  async fetchPosts(token: string, options?: { maxResults?: number }): Promise<CryptoPanicSentiment> {
    // Check cache first
    const cached = this.cache.get(token);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      logger.debug('CryptoPanic cache hit', { token, age: Date.now() - cached.timestamp });
      return cached.data;
    }

    // Check if API key is configured
    if (!this.apiKey) {
      throw new Error('CryptoPanic API key not configured');
    }

    try {
      // Fetch posts filtered by currency
      const response = await this.makeRequest<CryptoPanicResponse>('/posts/', {
        currencies: token.toUpperCase(),
        kind: 'news', // Only news articles
        public: 'true',
      });

      // Validate response structure
      if (!response || !response.results || !Array.isArray(response.results)) {
        logger.warn('Invalid CryptoPanic response structure', {
          token,
          hasResponse: !!response,
          hasResults: !!(response && response.results),
          isArray: !!(response && response.results && Array.isArray(response.results))
        });
        throw new Error('Invalid CryptoPanic API response structure');
      }

      // Get posts
      const posts = response.results.slice(0, options?.maxResults || 20);

      // NOTE: Developer v2 API does not include votes
      // We'll use news for display only, sentiment score will be 0
      const bullishVotes = 0;
      const bearishVotes = 0;
      const sentimentScore = 0;

      // Extract top news items (by recency since no votes available)
      // Generate URL from slug: https://cryptopanic.com/news/{id}/{slug}
      const topNews: NewsItem[] = posts
        .slice(0, 5) // Top 5 most recent news items
        .map(post => ({
          title: post.title,
          description: post.description || '',
          url: post.url || `https://cryptopanic.com/news/${post.id}/${post.slug}`,
          votes: {
            positive: 0,
            negative: 0,
          },
          totalVotes: 0,
          publishedAt: post.published_at,
        }));

      const sentiment: CryptoPanicSentiment = {
        token,
        posts,
        bullishVotes,
        bearishVotes,
        newsCount: posts.length,
        sentimentScore,
        topNews,
        timestamp: Date.now(),
      };

      // Cache result
      this.cache.set(token, {
        data: sentiment,
        timestamp: Date.now(),
        token,
      });

      logger.info('CryptoPanic sentiment fetched', {
        token,
        newsCount: posts.length,
        bullishVotes,
        bearishVotes,
        sentimentScore: sentimentScore.toFixed(3),
        quotaUsed: `${this.monthlyRequestCount}/${MONTHLY_QUOTA}`,
      });

      return sentiment;
    } catch (error) {
      if (error instanceof QuotaExceededError) {
        logger.warn('CryptoPanic quota exceeded', {
          token,
          resetDate: error.resetDate.toISOString(),
          requestsUsed: error.requestsUsed,
        });
        throw error;
      }

      logger.error('CryptoPanic fetch error', { token, error });
      throw error;
    }
  }

  /**
   * Get current quota status
   */
  getQuotaStatus(): {
    used: number;
    total: number;
    remaining: number;
    resetDate: Date;
    percentUsed: number;
  } {
    this.checkQuotaReset();
    return {
      used: this.monthlyRequestCount,
      total: MONTHLY_QUOTA,
      remaining: MONTHLY_QUOTA - this.monthlyRequestCount,
      resetDate: this.quotaResetDate,
      percentUsed: (this.monthlyRequestCount / MONTHLY_QUOTA) * 100,
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    logger.info('CryptoPanic cache cleared');
  }
}

// ============= SINGLETON =============
let instance: CryptoPanicCollector | null = null;

export function getCryptoPanicCollector(apiKey?: string): CryptoPanicCollector {
  if (!instance) {
    instance = new CryptoPanicCollector(apiKey);
  }
  return instance;
}
