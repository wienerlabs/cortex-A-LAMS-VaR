/**
 * Discord Sentiment Data Collector
 *
 * Collects messages from Discord channels/guilds for sentiment analysis.
 * Uses Discord Bot API via REST (no gateway/websocket dependency).
 *
 * Features:
 * - Guild channel monitoring via bot token
 * - Message history fetching (last N messages per channel)
 * - Keyword-based sentiment scoring (same lexicon as Telegram)
 * - Volume tracking (24h message count)
 * - Rate limit handling (50 req/sec global)
 * - Result caching (5min TTL)
 * - Graceful degradation on auth/permission failures
 */
import { logger } from '../logger.js';

// ============= TYPES =============

export interface DiscordMessage {
  id: string;
  channel_id: string;
  content: string;
  timestamp: string;
  author: {
    id: string;
    username: string;
    bot: boolean;
  };
  guild_id?: string;
}

export interface DiscordChannelData {
  channelId: string;
  channelName: string;
  guildId: string;
  guildName: string;
  messages: DiscordMessage[];
  messageCount: number;
  sentimentScore: number; // -1 to 1
  volume24h: number;
  timestamp: number;
}

export interface DiscordSentiment {
  token: string;
  channels: DiscordChannelData[];
  totalMessages: number;
  averageSentiment: number; // -1 to 1
  volume24h: number;
  timestamp: number;
}

export interface CachedDiscordResult {
  data: DiscordSentiment;
  timestamp: number;
  token: string;
}

export interface DiscordChannelConfig {
  guildId: string;
  channelId: string;
  guildName?: string;
  channelName?: string;
}

// ============= CONSTANTS =============

const DISCORD_API_BASE = 'https://discord.com/api/v10';
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const RATE_LIMIT_PER_SEC = 50;
const RATE_LIMIT_WINDOW_MS = 1000;
const MAX_MESSAGES_PER_FETCH = 100;
const RETRY_AFTER_DEFAULT_MS = 1000;

const BULLISH_KEYWORDS = [
  'moon', 'bullish', 'buy', 'pump', 'rocket', '🚀', '📈', 'ath', 'breakout',
  'long', 'calls', 'green', 'profit', 'gains', 'winning', 'up', 'surge',
  'wagmi', 'lfg', 'send it', 'based', 'alpha', 'gem', 'fomo',
];

const BEARISH_KEYWORDS = [
  'dump', 'bearish', 'sell', 'crash', 'drop', 'down', '📉', 'short', 'puts',
  'red', 'loss', 'losing', 'fall', 'decline', 'plunge', 'tank',
  'ngmi', 'rekt', 'rug', 'scam', 'dead', 'exit', 'capitulate',
];

// ============= DISCORD COLLECTOR CLASS =============

export class DiscordCollector {
  private botToken: string;
  private channels: DiscordChannelConfig[];
  private cache: Map<string, CachedDiscordResult> = new Map();
  private requestTimestamps: number[] = [];

  constructor(botToken?: string, channels?: DiscordChannelConfig[]) {
    this.botToken = botToken || process.env.DISCORD_BOT_TOKEN || '';
    this.channels = channels || this.parseChannelsFromEnv();

    if (!this.botToken) {
      logger.warn('Discord Bot Token not configured - Discord collector disabled');
    } else {
      logger.info('DiscordCollector initialized', {
        channelCount: this.channels.length,
        channels: this.channels.map(c => `${c.guildId}/${c.channelId}`),
      });
    }
  }

  private parseChannelsFromEnv(): DiscordChannelConfig[] {
    const channelsEnv = process.env.DISCORD_CHANNELS;
    if (!channelsEnv) return [];

    try {
      return JSON.parse(channelsEnv) as DiscordChannelConfig[];
    } catch {
      // Format: "guildId:channelId,guildId:channelId"
      return channelsEnv.split(',').map(entry => {
        const [guildId, channelId] = entry.trim().split(':');
        return { guildId, channelId };
      }).filter(c => c.guildId && c.channelId);
    }
  }

  private canMakeRequest(): boolean {
    const now = Date.now();
    this.requestTimestamps = this.requestTimestamps.filter(
      ts => now - ts < RATE_LIMIT_WINDOW_MS
    );
    return this.requestTimestamps.length < RATE_LIMIT_PER_SEC;
  }

  private async waitForRateLimit(): Promise<void> {
    if (this.canMakeRequest()) return;

    const now = Date.now();
    const oldestTimestamp = Math.min(...this.requestTimestamps);
    const waitTime = RATE_LIMIT_WINDOW_MS - (now - oldestTimestamp) + 50;
    logger.debug('Discord rate limit wait', { waitMs: waitTime });
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }

  private async makeRequest<T>(endpoint: string, retries: number = 2): Promise<T> {
    await this.waitForRateLimit();

    const url = `${DISCORD_API_BASE}${endpoint}`;

    for (let attempt = 0; attempt <= retries; attempt++) {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': `Bot ${this.botToken}`,
          'Content-Type': 'application/json',
          'User-Agent': 'CortexBot (https://cortex-agent.xyz, 1.0)',
        },
      });

      this.requestTimestamps.push(Date.now());

      // Rate limited - wait and retry
      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get('retry-after') || '1', 10);
        const waitMs = (retryAfter || 1) * 1000;
        logger.warn('Discord rate limited', { retryAfter: waitMs, attempt });
        await new Promise(resolve => setTimeout(resolve, waitMs));
        continue;
      }

      if (!response.ok) {
        const errorText = await response.text();
        if (attempt < retries && response.status >= 500) {
          logger.warn('Discord API server error, retrying', {
            status: response.status,
            attempt,
          });
          await new Promise(resolve => setTimeout(resolve, RETRY_AFTER_DEFAULT_MS * (attempt + 1)));
          continue;
        }
        throw new Error(`Discord API error ${response.status}: ${errorText}`);
      }

      return await response.json() as T;
    }

    throw new Error('Discord API request failed after retries');
  }

  private calculateMessageSentiment(text: string): number {
    if (!text) return 0;

    const lowerText = text.toLowerCase();
    let bullishCount = 0;
    let bearishCount = 0;

    for (const keyword of BULLISH_KEYWORDS) {
      if (lowerText.includes(keyword.toLowerCase())) bullishCount++;
    }

    for (const keyword of BEARISH_KEYWORDS) {
      if (lowerText.includes(keyword.toLowerCase())) bearishCount++;
    }

    const total = bullishCount + bearishCount;
    if (total === 0) return 0;

    return (bullishCount - bearishCount) / total;
  }

  async fetchChannelMessages(
    channelConfig: DiscordChannelConfig,
    limit: number = MAX_MESSAGES_PER_FETCH
  ): Promise<DiscordChannelData> {
    if (!this.botToken) {
      throw new Error('Discord Bot Token not configured');
    }

    try {
      // Fetch channel info
      const channel = await this.makeRequest<{
        id: string;
        name: string;
        guild_id: string;
      }>(`/channels/${channelConfig.channelId}`);

      // Fetch guild info for name
      let guildName = channelConfig.guildName || 'Unknown';
      try {
        const guild = await this.makeRequest<{ id: string; name: string }>(
          `/guilds/${channelConfig.guildId}`
        );
        guildName = guild.name;
      } catch {
        // Guild info is optional
      }

      // Fetch recent messages
      const messages = await this.makeRequest<DiscordMessage[]>(
        `/channels/${channelConfig.channelId}/messages?limit=${limit}`
      );

      // Filter out bot messages
      const humanMessages = messages.filter(m => !m.author.bot && m.content);

      // Calculate sentiment
      let totalSentiment = 0;
      let scoredCount = 0;

      for (const msg of humanMessages) {
        if (msg.content) {
          totalSentiment += this.calculateMessageSentiment(msg.content);
          scoredCount++;
        }
      }

      const sentimentScore = scoredCount > 0 ? totalSentiment / scoredCount : 0;

      // 24h volume
      const oneDayAgo = Date.now() - 86400000;
      const volume24h = humanMessages.filter(
        m => new Date(m.timestamp).getTime() > oneDayAgo
      ).length;

      return {
        channelId: channelConfig.channelId,
        channelName: channel.name || channelConfig.channelName || channelConfig.channelId,
        guildId: channelConfig.guildId,
        guildName,
        messages: humanMessages,
        messageCount: humanMessages.length,
        sentimentScore,
        volume24h,
        timestamp: Date.now(),
      };
    } catch (error) {
      logger.error('Discord channel fetch error', {
        channelId: channelConfig.channelId,
        error: (error as Error).message,
      });
      throw error;
    }
  }

  async fetchSentiment(token: string): Promise<DiscordSentiment> {
    // Check cache
    const cached = this.cache.get(token);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      logger.debug('Discord cache hit', { token, age: Date.now() - cached.timestamp });
      return cached.data;
    }

    if (!this.botToken) {
      throw new Error('Discord Bot Token not configured');
    }

    if (this.channels.length === 0) {
      logger.warn('No Discord channels configured');
      return {
        token,
        channels: [],
        totalMessages: 0,
        averageSentiment: 0,
        volume24h: 0,
        timestamp: Date.now(),
      };
    }

    // Fetch from all channels in parallel
    const results = await Promise.all(
      this.channels.map(ch =>
        this.fetchChannelMessages(ch).catch(error => {
          logger.warn('Failed to fetch Discord channel', {
            channelId: ch.channelId,
            error: (error as Error).message,
          });
          return null;
        })
      )
    );

    const channelData = results.filter((d): d is DiscordChannelData => d !== null);

    // Aggregate
    let totalMessages = 0;
    let weightedSentiment = 0;
    let totalVolume24h = 0;

    for (const data of channelData) {
      totalMessages += data.messageCount;
      weightedSentiment += data.sentimentScore * data.messageCount;
      totalVolume24h += data.volume24h;
    }

    const averageSentiment = totalMessages > 0 ? weightedSentiment / totalMessages : 0;

    const sentiment: DiscordSentiment = {
      token,
      channels: channelData,
      totalMessages,
      averageSentiment,
      volume24h: totalVolume24h,
      timestamp: Date.now(),
    };

    // Cache
    this.cache.set(token, { data: sentiment, timestamp: Date.now(), token });

    logger.info('Discord sentiment fetched', {
      token,
      channelCount: channelData.length,
      totalMessages,
      averageSentiment: averageSentiment.toFixed(3),
      volume24h: totalVolume24h,
    });

    return sentiment;
  }

  clearCache(): void {
    this.cache.clear();
    logger.info('Discord cache cleared');
  }

  addChannel(config: DiscordChannelConfig): void {
    const exists = this.channels.some(
      c => c.guildId === config.guildId && c.channelId === config.channelId
    );
    if (!exists) {
      this.channels.push(config);
      logger.info('Discord channel added', { guildId: config.guildId, channelId: config.channelId });
    }
  }

  removeChannel(channelId: string): void {
    this.channels = this.channels.filter(c => c.channelId !== channelId);
    logger.info('Discord channel removed', { channelId });
  }

  getChannels(): DiscordChannelConfig[] {
    return [...this.channels];
  }

  isConfigured(): boolean {
    return !!this.botToken && this.channels.length > 0;
  }
}

// ============= SINGLETON =============

let instance: DiscordCollector | null = null;

export function getDiscordCollector(
  botToken?: string,
  channels?: DiscordChannelConfig[]
): DiscordCollector {
  if (!instance) {
    instance = new DiscordCollector(botToken, channels);
  }
  return instance;
}

export function resetDiscordCollector(): void {
  instance = null;
}
