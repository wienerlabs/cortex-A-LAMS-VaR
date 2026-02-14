/**
 * Telegram Sentiment Data Collector
 * 
 * Collects messages from Telegram channels for sentiment analysis.
 * Features:
 * - Bot API integration
 * - Channel monitoring (requires manual bot join)
 * - Message volume tracking
 * - Keyword-based sentiment scoring
 * - Rate limit handling (30 msg/sec per bot)
 * - Result caching
 * 
 * Note: Bot must be added to channels manually before monitoring
 */
import { logger } from '../logger.js';

// ============= TYPES =============
export interface TelegramMessage {
  message_id: number;
  date: number;
  text?: string;
  from?: {
    id: number;
    is_bot: boolean;
    first_name: string;
    username?: string;
  };
  chat: {
    id: number;
    title?: string;
    username?: string;
    type: string;
  };
  forward_from_chat?: {
    id: number;
    title: string;
    username?: string;
  };
}

export interface TelegramUpdate {
  update_id: number;
  message?: TelegramMessage;
  channel_post?: TelegramMessage;
}

export interface TelegramChannelData {
  channelId: string;
  channelName: string;
  messages: TelegramMessage[];
  messageCount: number;
  sentimentScore: number; // -1 to 1
  volume24h: number;
  timestamp: number;
}

export interface TelegramSentiment {
  token: string;
  channels: TelegramChannelData[];
  totalMessages: number;
  averageSentiment: number; // -1 to 1
  volume24h: number;
  timestamp: number;
}

export interface CachedTelegramResult {
  data: TelegramSentiment;
  timestamp: number;
  token: string;
}

// ============= CONSTANTS =============
const TELEGRAM_API_BASE = 'https://api.telegram.org';
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes cache
const RATE_LIMIT_PER_SEC = 30; // 30 messages per second per bot
const RATE_LIMIT_WINDOW_MS = 1000; // 1 second

// Sentiment keywords
const BULLISH_KEYWORDS = [
  'moon', 'bullish', 'buy', 'pump', 'rocket', '🚀', '📈', 'ath', 'breakout',
  'long', 'calls', 'green', 'profit', 'gains', 'winning', 'up', 'surge'
];

const BEARISH_KEYWORDS = [
  'dump', 'bearish', 'sell', 'crash', 'drop', 'down', '📉', 'short', 'puts',
  'red', 'loss', 'losing', 'fall', 'decline', 'plunge', 'tank'
];

// ============= TELEGRAM COLLECTOR CLASS =============
export class TelegramCollector {
  private botToken: string;
  private channels: string[]; // Channel usernames or IDs
  private cache: Map<string, CachedTelegramResult> = new Map();
  private requestTimestamps: number[] = [];
  private lastUpdateId: number = 0;

  constructor(botToken?: string, channels?: string[]) {
    this.botToken = botToken || process.env.TELEGRAM_BOT_TOKEN || '';
    this.channels = channels || this.parseChannelsFromEnv();

    if (!this.botToken) {
      logger.warn('Telegram Bot Token not configured - Telegram collector will not work');
    } else {
      logger.info('TelegramCollector initialized', {
        channelCount: this.channels.length,
        channels: this.channels,
        rateLimit: `${RATE_LIMIT_PER_SEC} req/sec`,
      });
    }
  }

  /**
   * Parse channels from environment variable
   */
  private parseChannelsFromEnv(): string[] {
    const channelsEnv = process.env.TELEGRAM_CHANNELS;
    if (!channelsEnv) return [];
    
    try {
      return JSON.parse(channelsEnv);
    } catch {
      // Fallback: comma-separated
      return channelsEnv.split(',').map(c => c.trim()).filter(Boolean);
    }
  }

  /**
   * Check if we can make a request (rate limiting)
   */
  private canMakeRequest(): boolean {
    const now = Date.now();
    this.requestTimestamps = this.requestTimestamps.filter(
      ts => now - ts < RATE_LIMIT_WINDOW_MS
    );
    return this.requestTimestamps.length < RATE_LIMIT_PER_SEC;
  }

  /**
   * Wait for rate limit window
   */
  private async waitForRateLimit(): Promise<void> {
    if (this.canMakeRequest()) return;

    const now = Date.now();
    const oldestTimestamp = Math.min(...this.requestTimestamps);
    const waitTime = RATE_LIMIT_WINDOW_MS - (now - oldestTimestamp) + 50; // +50ms buffer
    logger.debug('Telegram rate limit wait', { waitMs: waitTime });
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }

  /**
   * Make an authenticated request to Telegram Bot API
   */
  private async makeRequest<T>(method: string, params?: Record<string, any>): Promise<T> {
    await this.waitForRateLimit();

    const url = `${TELEGRAM_API_BASE}/bot${this.botToken}/${method}`;

    logger.debug('Telegram API request', { method, params });

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: params ? JSON.stringify(params) : undefined,
    });

    // Track request
    this.requestTimestamps.push(Date.now());

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Telegram API error ${response.status}: ${errorText}`);
    }

    const result = await response.json() as { ok: boolean; result: T; description?: string };

    if (!result.ok) {
      throw new Error(`Telegram API error: ${result.description || 'Unknown error'}`);
    }

    return result.result;
  }

  /**
   * Calculate sentiment score from message text
   */
  private calculateMessageSentiment(text: string): number {
    if (!text) return 0;

    const lowerText = text.toLowerCase();
    let bullishCount = 0;
    let bearishCount = 0;

    for (const keyword of BULLISH_KEYWORDS) {
      if (lowerText.includes(keyword.toLowerCase())) {
        bullishCount++;
      }
    }

    for (const keyword of BEARISH_KEYWORDS) {
      if (lowerText.includes(keyword.toLowerCase())) {
        bearishCount++;
      }
    }

    const total = bullishCount + bearishCount;
    if (total === 0) return 0;

    return (bullishCount - bearishCount) / total;
  }

  /**
   * Fetch recent messages from a channel
   */
  async fetchChannelMessages(channelId: string, limit: number = 100): Promise<TelegramChannelData> {
    if (!this.botToken) {
      throw new Error('Telegram Bot Token not configured');
    }

    try {
      // Get channel info
      const chat = await this.makeRequest<any>('getChat', {
        chat_id: channelId,
      });

      // Get recent updates (messages)
      const updates = await this.makeRequest<TelegramUpdate[]>('getUpdates', {
        offset: this.lastUpdateId + 1,
        limit,
        allowed_updates: ['message', 'channel_post'],
      });

      // Update last update ID
      if (updates.length > 0) {
        this.lastUpdateId = Math.max(...updates.map(u => u.update_id));
      }

      // Filter messages from this channel
      const messages: TelegramMessage[] = [];
      for (const update of updates) {
        const msg = update.channel_post || update.message;
        if (msg && (msg.chat.id.toString() === channelId || msg.chat.username === channelId.replace('@', ''))) {
          messages.push(msg);
        }
      }

      // Calculate sentiment
      let totalSentiment = 0;
      let messageCount = 0;

      for (const msg of messages) {
        if (msg.text) {
          totalSentiment += this.calculateMessageSentiment(msg.text);
          messageCount++;
        }
      }

      const sentimentScore = messageCount > 0 ? totalSentiment / messageCount : 0;

      // Calculate 24h volume (messages in last 24h)
      const now = Date.now() / 1000;
      const volume24h = messages.filter(m => now - m.date < 86400).length;

      return {
        channelId,
        channelName: chat.title || chat.username || channelId,
        messages,
        messageCount: messages.length,
        sentimentScore,
        volume24h,
        timestamp: Date.now(),
      };
    } catch (error) {
      logger.error('Telegram channel fetch error', { channelId, error });
      throw error;
    }
  }

  /**
   * Fetch sentiment for a token across all configured channels
   */
  async fetchSentiment(token: string): Promise<TelegramSentiment> {
    // Check cache first
    const cached = this.cache.get(token);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      logger.debug('Telegram cache hit', { token, age: Date.now() - cached.timestamp });
      return cached.data;
    }

    if (!this.botToken) {
      throw new Error('Telegram Bot Token not configured');
    }

    if (this.channels.length === 0) {
      logger.warn('No Telegram channels configured');
      return {
        token,
        channels: [],
        totalMessages: 0,
        averageSentiment: 0,
        volume24h: 0,
        timestamp: Date.now(),
      };
    }

    try {
      // Fetch data from all channels
      const channelDataPromises = this.channels.map(channelId =>
        this.fetchChannelMessages(channelId).catch(error => {
          logger.warn('Failed to fetch from channel', { channelId, error: error.message });
          return null;
        })
      );

      const channelDataResults = await Promise.all(channelDataPromises);
      const channelData = channelDataResults.filter((d): d is TelegramChannelData => d !== null);

      // Aggregate sentiment
      let totalMessages = 0;
      let totalSentiment = 0;
      let totalVolume24h = 0;

      for (const data of channelData) {
        totalMessages += data.messageCount;
        totalSentiment += data.sentimentScore * data.messageCount; // Weighted by message count
        totalVolume24h += data.volume24h;
      }

      const averageSentiment = totalMessages > 0 ? totalSentiment / totalMessages : 0;

      const sentiment: TelegramSentiment = {
        token,
        channels: channelData,
        totalMessages,
        averageSentiment,
        volume24h: totalVolume24h,
        timestamp: Date.now(),
      };

      // Cache result
      this.cache.set(token, {
        data: sentiment,
        timestamp: Date.now(),
        token,
      });

      logger.info('Telegram sentiment fetched', {
        token,
        channelCount: channelData.length,
        totalMessages,
        averageSentiment: averageSentiment.toFixed(3),
        volume24h: totalVolume24h,
      });

      return sentiment;
    } catch (error) {
      logger.error('Telegram sentiment fetch error', { token, error });
      throw error;
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    logger.info('Telegram cache cleared');
  }

  /**
   * Add a channel to monitor
   */
  addChannel(channelId: string): void {
    if (!this.channels.includes(channelId)) {
      this.channels.push(channelId);
      logger.info('Telegram channel added', { channelId });
    }
  }

  /**
   * Remove a channel from monitoring
   */
  removeChannel(channelId: string): void {
    this.channels = this.channels.filter(c => c !== channelId);
    logger.info('Telegram channel removed', { channelId });
  }

  /**
   * Get configured channels
   */
  getChannels(): string[] {
    return [...this.channels];
  }
}

// ============= SINGLETON =============
let instance: TelegramCollector | null = null;

export function getTelegramCollector(botToken?: string, channels?: string[]): TelegramCollector {
  if (!instance) {
    instance = new TelegramCollector(botToken, channels);
  }
  return instance;
}
