/**
 * File Logger for Perps ML Trading Agent
 * 
 * Logs all trading activity to JSON files for monitoring and analysis:
 * - Trades: logs/perps_trades.json
 * - Predictions: logs/perps_predictions.json  
 * - Errors: logs/perps_errors.json
 */
import * as fs from 'fs';
import * as path from 'path';
import type { PositionSide } from '../../../types/perps.js';
import type { PredictionResult } from './modelLoader.js';
import type { CloseReason, TrackedPosition } from './tradingAgent.js';

// ============= TYPES =============

export interface TradeLogEntry {
  timestamp: string;
  type: 'open' | 'close';
  market: string;
  side: PositionSide;
  sizeUsd: number;
  entryPrice?: number;
  exitPrice?: number;
  fundingRate: number;
  confidence: number;
  closeReason?: CloseReason;
  pnlUsd?: number;
  pnlPercent?: number;
  holdTimeHours?: number;
  orderId?: string;
  venue: string;
  dryRun: boolean;
}

export interface PredictionLogEntry {
  timestamp: string;
  market: string;
  fundingRate: number;
  prediction: number;
  probability: number;
  direction: 'long' | 'short';
  shouldTrade: boolean;
  confidence: number;
  featureCount: number;
}

export interface ErrorLogEntry {
  timestamp: string;
  category: 'trade' | 'prediction' | 'data' | 'connection' | 'system';
  message: string;
  error?: string;
  stack?: string;
  context?: Record<string, unknown>;
}

// ============= FILE LOGGER =============

class PerpsFileLogger {
  private logDir: string;
  private tradesFile: string;
  private predictionsFile: string;
  private errorsFile: string;
  private initialized: boolean = false;

  constructor() {
    this.logDir = process.env.PERPS_LOG_DIR || './logs';
    this.tradesFile = path.join(this.logDir, 'perps_trades.json');
    this.predictionsFile = path.join(this.logDir, 'perps_predictions.json');
    this.errorsFile = path.join(this.logDir, 'perps_errors.json');
  }

  /** Ensure log directory and files exist */
  private ensureInitialized(): void {
    if (this.initialized) return;

    // Create directory
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }

    // Initialize JSON array files if they don't exist
    for (const file of [this.tradesFile, this.predictionsFile, this.errorsFile]) {
      if (!fs.existsSync(file)) {
        fs.writeFileSync(file, '[]', 'utf-8');
      }
    }

    this.initialized = true;
  }

  /** Append entry to JSON array file */
  private appendToFile<T>(filePath: string, entry: T): void {
    this.ensureInitialized();
    
    try {
      // Read existing array
      const content = fs.readFileSync(filePath, 'utf-8');
      const entries: T[] = JSON.parse(content);
      
      // Append and write
      entries.push(entry);
      fs.writeFileSync(filePath, JSON.stringify(entries, null, 2), 'utf-8');
    } catch (err) {
      // If file is corrupted, start fresh
      console.error(`Error writing to ${filePath}, resetting:`, err);
      fs.writeFileSync(filePath, JSON.stringify([entry], null, 2), 'utf-8');
    }
  }

  /** Log a trade (open or close) */
  logTrade(entry: TradeLogEntry): void {
    this.appendToFile(this.tradesFile, entry);
    
    const emoji = entry.type === 'open' ? '📈' : '📉';
    const action = entry.type === 'open' ? 'OPENED' : 'CLOSED';
    const pnl = entry.pnlUsd !== undefined ? ` P&L: $${entry.pnlUsd.toFixed(2)}` : '';
    
    console.log(`${emoji} [TRADE] ${action} ${entry.side.toUpperCase()} ${entry.market} $${entry.sizeUsd}${pnl}`);
  }

  /** Log a prediction */
  logPrediction(entry: PredictionLogEntry): void {
    this.appendToFile(this.predictionsFile, entry);
  }

  /** Log an error */
  logError(entry: ErrorLogEntry): void {
    this.appendToFile(this.errorsFile, entry);
    console.error(`❌ [ERROR] [${entry.category}] ${entry.message}`);
  }

  /** Get recent trades */
  getRecentTrades(limit = 10): TradeLogEntry[] {
    this.ensureInitialized();
    try {
      const content = fs.readFileSync(this.tradesFile, 'utf-8');
      const trades: TradeLogEntry[] = JSON.parse(content);
      return trades.slice(-limit);
    } catch {
      return [];
    }
  }

  /** Get recent predictions */
  getRecentPredictions(limit = 10): PredictionLogEntry[] {
    this.ensureInitialized();
    try {
      const content = fs.readFileSync(this.predictionsFile, 'utf-8');
      const predictions: PredictionLogEntry[] = JSON.parse(content);
      return predictions.slice(-limit);
    } catch {
      return [];
    }
  }

  /** Get recent errors */
  getRecentErrors(limit = 10): ErrorLogEntry[] {
    this.ensureInitialized();
    try {
      const content = fs.readFileSync(this.errorsFile, 'utf-8');
      const errors: ErrorLogEntry[] = JSON.parse(content);
      return errors.slice(-limit);
    } catch {
      return [];
    }
  }
}

// Singleton instance
export const perpsFileLogger = new PerpsFileLogger();

