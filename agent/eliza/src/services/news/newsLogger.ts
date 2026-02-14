/**
 * News Logger Service
 * 
 * Comprehensive logging for news events and trading decisions.
 * Tracks all news, impact scores, and actions taken.
 * 
 * Log Structure:
 * - Timestamp
 * - Source
 * - Asset
 * - News title
 * - Impact score
 * - Classification
 * - Action taken
 */

import * as fs from 'fs';
import * as path from 'path';
import { logger } from '../logger.js';
import { type NewsImpactScore, type TradingAction } from './newsScorer.js';
import { type NewsType } from './newsClassifier.js';

// ============= TYPES =============

export interface NewsLogEntry {
  timestamp: string;
  source: string;
  asset: string;
  title: string;
  description?: string;
  url?: string;
  newsType: NewsType;
  impactScore: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  tradingAction: TradingAction;
  actionTaken?: string;
  positionAffected?: string;
  reasoning?: string;
}

export interface NewsLoggerConfig {
  /** Directory for log files */
  logDir: string;
  /** Enable file logging */
  logToFile: boolean;
  /** Enable console logging */
  logToConsole: boolean;
  /** Maximum log file size in bytes */
  maxFileSizeBytes: number;
  /** Number of log files to keep */
  maxLogFiles: number;
}

// ============= CONSTANTS =============

const DEFAULT_CONFIG: NewsLoggerConfig = {
  logDir: './logs/news',
  logToFile: true,
  logToConsole: true,
  maxFileSizeBytes: 10 * 1024 * 1024, // 10MB
  maxLogFiles: 10,
};

// ============= NEWS LOGGER =============

let loggerInstance: NewsLogger | null = null;

export class NewsLogger {
  private config: NewsLoggerConfig;
  private currentLogFile: string | null = null;
  private logCount = 0;

  constructor(config: Partial<NewsLoggerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    // Create log directory if needed
    if (this.config.logToFile) {
      this.ensureLogDirectory();
    }

    logger.info('[NewsLogger] Initialized', { logDir: this.config.logDir });
  }

  /**
   * Log a news event
   */
  logNews(entry: NewsLogEntry): void {
    const formattedEntry = this.formatEntry(entry);

    if (this.config.logToConsole) {
      this.logToConsole(entry);
    }

    if (this.config.logToFile) {
      this.logToFile(formattedEntry);
    }

    this.logCount++;
  }

  /**
   * Log news with impact score
   */
  logNewsWithScore(
    asset: string,
    title: string,
    score: NewsImpactScore,
    source: string,
    actionTaken?: string
  ): void {
    const entry: NewsLogEntry = {
      timestamp: new Date().toISOString(),
      source,
      asset,
      title,
      newsType: score.classification.type,
      impactScore: score.immediateImpact,
      severity: score.severity,
      tradingAction: score.tradingAction,
      actionTaken,
      reasoning: score.reasoning,
    };

    this.logNews(entry);
  }

  /**
   * Log a critical alert
   */
  logCriticalAlert(
    asset: string,
    title: string,
    score: NewsImpactScore,
    source: string
  ): void {
    logger.error('[NewsLogger] CRITICAL ALERT', {
      asset,
      title: title.substring(0, 100),
      impact: score.immediateImpact,
      action: score.tradingAction,
    });

    this.logNewsWithScore(asset, title, score, source, 'CRITICAL_ALERT_TRIGGERED');
  }

  /**
   * Format entry for file logging
   */
  private formatEntry(entry: NewsLogEntry): string {
    return JSON.stringify(entry) + '\n';
  }

  /**
   * Log to console with appropriate level
   */
  private logToConsole(entry: NewsLogEntry): void {
    const message = `[News] ${entry.asset}: ${entry.title.substring(0, 60)}...`;
    const data = {
      source: entry.source,
      impact: entry.impactScore,
      type: entry.newsType,
      action: entry.tradingAction,
    };

    switch (entry.severity) {
      case 'CRITICAL':
        logger.error(message, data);
        break;
      case 'HIGH':
        logger.warn(message, data);
        break;
      case 'MEDIUM':
        logger.info(message, data);
        break;
      default:
        logger.debug(message, data);
    }
  }

  /**
   * Log to file with rotation
   */
  private logToFile(content: string): void {
    try {
      const logFile = this.getCurrentLogFile();
      fs.appendFileSync(logFile, content);

      // Check file size for rotation
      const stats = fs.statSync(logFile);
      if (stats.size >= this.config.maxFileSizeBytes) {
        this.rotateLogFile();
      }
    } catch (error) {
      logger.error('[NewsLogger] Failed to write to file', { error });
    }
  }

  /**
   * Get current log file path
   */
  private getCurrentLogFile(): string {
    if (!this.currentLogFile) {
      const date = new Date().toISOString().split('T')[0];
      this.currentLogFile = path.join(this.config.logDir, `news_${date}.jsonl`);
    }
    return this.currentLogFile;
  }

  /**
   * Rotate log file when size limit reached
   */
  private rotateLogFile(): void {
    this.currentLogFile = null;
    this.cleanupOldLogs();
  }

  /**
   * Clean up old log files
   */
  private cleanupOldLogs(): void {
    try {
      const files = fs.readdirSync(this.config.logDir)
        .filter(f => f.startsWith('news_') && f.endsWith('.jsonl'))
        .sort()
        .reverse();

      // Remove excess files
      if (files.length > this.config.maxLogFiles) {
        const toDelete = files.slice(this.config.maxLogFiles);
        for (const file of toDelete) {
          fs.unlinkSync(path.join(this.config.logDir, file));
          logger.debug('[NewsLogger] Deleted old log file', { file });
        }
      }
    } catch (error) {
      logger.error('[NewsLogger] Failed to cleanup old logs', { error });
    }
  }

  /**
   * Ensure log directory exists
   */
  private ensureLogDirectory(): void {
    if (!fs.existsSync(this.config.logDir)) {
      fs.mkdirSync(this.config.logDir, { recursive: true });
      logger.debug('[NewsLogger] Created log directory', { dir: this.config.logDir });
    }
  }

  /**
   * Get log statistics
   */
  getStats(): { logCount: number; currentFile: string | null } {
    return {
      logCount: this.logCount,
      currentFile: this.currentLogFile,
    };
  }

  /**
   * Read recent logs
   */
  readRecentLogs(count: number = 100): NewsLogEntry[] {
    try {
      const logFile = this.getCurrentLogFile();
      if (!fs.existsSync(logFile)) return [];

      const content = fs.readFileSync(logFile, 'utf-8');
      const lines = content.trim().split('\n').filter(l => l);
      const recentLines = lines.slice(-count);

      return recentLines.map(line => JSON.parse(line) as NewsLogEntry);
    } catch (error) {
      logger.error('[NewsLogger] Failed to read logs', { error });
      return [];
    }
  }
}

/**
 * Get singleton NewsLogger instance
 */
export function getNewsLogger(): NewsLogger {
  if (!loggerInstance) {
    loggerInstance = new NewsLogger();
  }
  return loggerInstance;
}
