/**
 * News Monitor Service
 * 
 * Real-time news monitoring with breaking news detection and alerts.
 * Monitors multiple sources for critical news that may impact trading.
 * 
 * Features:
 * - Periodic news checking (configurable interval)
 * - Breaking news detection
 * - Critical keyword alerts
 * - Volume spike detection
 * - Alert callbacks for trading system integration
 */

import { logger } from '../logger.js';
import { getCryptoPanicCollector, type CryptoPanicSentiment } from '../sentiment/cryptopanicCollector.js';
import { getNewsScorer, type NewsImpactScore } from './newsScorer.js';
import { NewsType } from './newsClassifier.js';

// ============= TYPES =============

export interface NewsAlert {
  type: 'BREAKING' | 'CRITICAL' | 'VOLUME_SPIKE' | 'KEYWORD';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  asset: string;
  title: string;
  description: string;
  score: NewsImpactScore;
  timestamp: Date;
  source: string;
  suggestedAction: 'EXIT' | 'REDUCE' | 'HOLD' | 'MONITOR';
}

export interface MonitorConfig {
  /** Interval between checks in milliseconds (default: 60000 = 1 minute) */
  checkIntervalMs: number;
  /** Assets to monitor */
  assets: string[];
  /** Impact threshold for alerts (default: -30) */
  alertThreshold: number;
  /** Critical impact threshold (default: -50) */
  criticalThreshold: number;
  /** Keywords that trigger immediate alerts */
  criticalKeywords: string[];
}

export type AlertCallback = (alert: NewsAlert) => void;

// ============= CONSTANTS =============

const DEFAULT_CONFIG: MonitorConfig = {
  checkIntervalMs: 60000, // 1 minute
  assets: ['BTC', 'ETH', 'SOL'],
  alertThreshold: -30,
  criticalThreshold: -50,
  criticalKeywords: [
    'hack', 'hacked', 'exploit', 'exploited', 'stolen', 'drained',
    'sec', 'lawsuit', 'charges', 'ban', 'banned', 'insolvent', 'bankrupt',
  ],
};

// ============= NEWS MONITOR =============

let monitorInstance: NewsMonitor | null = null;

export class NewsMonitor {
  private config: MonitorConfig;
  private cryptoPanic = getCryptoPanicCollector();
  private scorer = getNewsScorer();
  private intervalId: NodeJS.Timeout | null = null;
  private alertCallbacks: AlertCallback[] = [];
  private lastSeenNews: Map<string, Set<string>> = new Map(); // asset -> set of news titles
  private isRunning = false;

  constructor(config: Partial<MonitorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[NewsMonitor] Initialized', { 
      assets: this.config.assets,
      intervalMs: this.config.checkIntervalMs,
    });
  }

  /**
   * Register a callback for alerts
   */
  onAlert(callback: AlertCallback): void {
    this.alertCallbacks.push(callback);
    logger.debug('[NewsMonitor] Alert callback registered');
  }

  /**
   * Start monitoring news
   */
  start(): void {
    if (this.isRunning) {
      logger.warn('[NewsMonitor] Already running');
      return;
    }

    logger.info('[NewsMonitor] Starting news monitoring', {
      assets: this.config.assets,
      intervalMs: this.config.checkIntervalMs,
    });

    this.isRunning = true;
    
    // Initialize last seen sets for each asset
    for (const asset of this.config.assets) {
      this.lastSeenNews.set(asset, new Set());
    }

    // Run immediately, then on interval
    this.checkAllAssets().catch(err => 
      logger.error('[NewsMonitor] Initial check failed', { error: err.message })
    );

    this.intervalId = setInterval(() => {
      this.checkAllAssets().catch(err =>
        logger.error('[NewsMonitor] Periodic check failed', { error: err.message })
      );
    }, this.config.checkIntervalMs);
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isRunning = false;
    logger.info('[NewsMonitor] Stopped');
  }

  /**
   * Check if monitor is running
   */
  isActive(): boolean {
    return this.isRunning;
  }

  /**
   * Check news for all monitored assets
   */
  private async checkAllAssets(): Promise<void> {
    logger.debug('[NewsMonitor] Checking all assets', { count: this.config.assets.length });

    const checkPromises = this.config.assets.map(asset =>
      this.checkAsset(asset).catch(err => {
        logger.warn('[NewsMonitor] Asset check failed', { asset, error: err.message });
        return [];
      })
    );

    const alertArrays = await Promise.all(checkPromises);
    const allAlerts = alertArrays.flat();

    if (allAlerts.length > 0) {
      logger.info('[NewsMonitor] Alerts generated', { count: allAlerts.length });
    }
  }

  /**
   * Check news for a single asset
   */
  private async checkAsset(asset: string): Promise<NewsAlert[]> {
    const alerts: NewsAlert[] = [];
    // Use fetchPosts which is the actual method on CryptoPanicCollector
    const sentiment = await this.cryptoPanic.fetchPosts(asset);

    if (!sentiment || sentiment.newsCount === 0) {
      return alerts;
    }

    const seenTitles = this.lastSeenNews.get(asset) || new Set();
    const newNews = sentiment.topNews.filter(item => !seenTitles.has(item.title));

    // Process new news items
    for (const newsItem of newNews) {
      const score = this.scorer.scoreNews({
        title: newsItem.title,
        description: newsItem.description,
        publishedAt: newsItem.publishedAt,
        assets: [asset],
      });

      // Check for critical keywords
      const text = `${newsItem.title} ${newsItem.description}`.toLowerCase();
      const matchedKeyword = this.config.criticalKeywords.find(kw => text.includes(kw));

      let alertType: NewsAlert['type'] | null = null;
      let severity: NewsAlert['severity'] = 'LOW';
      let suggestedAction: NewsAlert['suggestedAction'] = 'MONITOR';

      // Determine alert type and severity
      if (matchedKeyword) {
        alertType = 'KEYWORD';
        severity = 'HIGH';
        suggestedAction = 'REDUCE';
      }

      if (score.immediateImpact <= this.config.criticalThreshold) {
        alertType = 'CRITICAL';
        severity = 'CRITICAL';
        suggestedAction = 'EXIT';
      } else if (score.immediateImpact <= this.config.alertThreshold) {
        alertType = alertType || 'BREAKING';
        severity = 'HIGH';
        suggestedAction = 'REDUCE';
      }

      // Check for breaking news (high impact + recent)
      if (!alertType && Math.abs(score.immediateImpact) >= 50) {
        alertType = 'BREAKING';
        severity = 'MEDIUM';
        suggestedAction = score.immediateImpact > 0 ? 'HOLD' : 'REDUCE';
      }

      // Generate alert if criteria met
      if (alertType) {
        const alert: NewsAlert = {
          type: alertType,
          severity,
          asset,
          title: newsItem.title,
          description: newsItem.description,
          score,
          timestamp: new Date(),
          source: 'CryptoPanic',
          suggestedAction,
        };

        alerts.push(alert);
        this.emitAlert(alert);
      }

      // Mark as seen
      seenTitles.add(newsItem.title);
    }

    // Update last seen
    this.lastSeenNews.set(asset, seenTitles);

    // Keep only last 100 titles per asset to prevent memory bloat
    if (seenTitles.size > 100) {
      const titlesArray = Array.from(seenTitles);
      const trimmed = new Set(titlesArray.slice(-100));
      this.lastSeenNews.set(asset, trimmed);
    }

    return alerts;
  }

  /**
   * Emit alert to all registered callbacks
   */
  private emitAlert(alert: NewsAlert): void {
    logger.warn('[NewsMonitor] Alert emitted', {
      type: alert.type,
      severity: alert.severity,
      asset: alert.asset,
      title: alert.title.substring(0, 50),
      impact: alert.score.immediateImpact,
      suggestedAction: alert.suggestedAction,
    });

    for (const callback of this.alertCallbacks) {
      try {
        callback(alert);
      } catch (error) {
        logger.error('[NewsMonitor] Alert callback error', { error });
      }
    }
  }

  /**
   * Manually check for breaking news (for external calls)
   */
  async checkBreakingNews(): Promise<NewsAlert[]> {
    const allAlerts: NewsAlert[] = [];

    for (const asset of this.config.assets) {
      const alerts = await this.checkAsset(asset);
      allAlerts.push(...alerts);
    }

    return allAlerts.filter(a => a.type === 'BREAKING' || a.type === 'CRITICAL');
  }

  /**
   * Update monitored assets
   */
  updateAssets(assets: string[]): void {
    this.config.assets = assets;
    // Initialize tracking for new assets
    for (const asset of assets) {
      if (!this.lastSeenNews.has(asset)) {
        this.lastSeenNews.set(asset, new Set());
      }
    }
    logger.info('[NewsMonitor] Assets updated', { assets });
  }
}

/**
 * Get singleton NewsMonitor instance
 */
export function getNewsMonitor(): NewsMonitor {
  if (!monitorInstance) {
    monitorInstance = new NewsMonitor();
  }
  return monitorInstance;
}

