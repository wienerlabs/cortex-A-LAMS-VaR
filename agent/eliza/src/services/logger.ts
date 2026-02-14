/**
 * Logging Service for Cortex Agent
 * 
 * Structured logging for:
 * - Pool analysis results
 * - Rebalance operations
 * - Errors and warnings
 */
import * as fs from 'fs';
import * as path from 'path';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  category: string;
  message: string;
  data?: Record<string, unknown>;
}

export interface AnalysisLogData {
  pool: string;
  probability: number;
  decision: 'REBALANCE' | 'HOLD';
  shouldTrigger: boolean;
}

export interface RebalanceLogData {
  pool: string;
  action: 'triggered' | 'executed' | 'failed';
  txHash?: string;
  amountIn?: number;
  amountOut?: number;
  simulation?: boolean;
  error?: string;
}

class Logger {
  private logDir: string;
  private logToFile: boolean;
  private logToConsole: boolean;

  constructor() {
    this.logDir = process.env.LOG_DIR || './logs';
    this.logToFile = process.env.LOG_TO_FILE !== 'false';
    this.logToConsole = process.env.LOG_TO_CONSOLE !== 'false';

    // Create log directory if needed
    if (this.logToFile && !fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
  }

  private formatMessage(entry: LogEntry): string {
    const emoji = {
      debug: '🔍',
      info: 'ℹ️',
      warn: '⚠️',
      error: '❌',
    }[entry.level];

    let msg = `${emoji} [${entry.timestamp}] [${entry.category}] ${entry.message}`;
    if (entry.data) {
      msg += ` ${JSON.stringify(entry.data)}`;
    }
    return msg;
  }

  private log(level: LogLevel, category: string, message: string, data?: Record<string, unknown>): void {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      category,
      message,
      data,
    };

    // Console output
    if (this.logToConsole) {
      const formatted = this.formatMessage(entry);
      switch (level) {
        case 'error':
          console.error(formatted);
          break;
        case 'warn':
          console.warn(formatted);
          break;
        default:
          console.log(formatted);
      }
    }

    // File output
    if (this.logToFile) {
      this.writeToFile(entry);
    }
  }

  private writeToFile(entry: LogEntry): void {
    const date = entry.timestamp.split('T')[0];
    const filename = path.join(this.logDir, `cortex-${date}.log`);
    const line = JSON.stringify(entry) + '\n';

    fs.appendFileSync(filename, line);
  }

  // General logging methods
  debug(message: string, data?: Record<string, unknown>): void {
    this.log('debug', 'AGENT', message, data);
  }

  info(message: string, data?: Record<string, unknown>): void {
    this.log('info', 'AGENT', message, data);
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this.log('warn', 'AGENT', message, data);
  }

  error(message: string, data?: Record<string, unknown>): void {
    this.log('error', 'AGENT', message, data);
  }

  // Specialized logging methods
  analysis(data: AnalysisLogData): void {
    const decision = data.decision === 'REBALANCE' ? '🔄' : '✋';
    this.log('info', 'ANALYSIS', `${decision} ${data.pool}: ${(data.probability * 100).toFixed(2)}% - ${data.decision}`, data as unknown as Record<string, unknown>);
  }

  rebalance(data: RebalanceLogData): void {
    const action = {
      triggered: '🚀',
      executed: '✅',
      failed: '❌',
    }[data.action];
    this.log('info', 'REBALANCE', `${action} ${data.pool}: ${data.action}`, data as unknown as Record<string, unknown>);
  }

  api(method: string, endpoint: string, status: number, durationMs: number): void {
    this.log('debug', 'API', `${method} ${endpoint} - ${status} (${durationMs}ms)`);
  }
}

// Singleton instance
export const logger = new Logger();

