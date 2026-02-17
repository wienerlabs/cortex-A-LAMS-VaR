/**
 * Guardian Logger
 * 
 * Specialized logging for Guardian validation system.
 * Logs all validation attempts, blocks, security alerts, and config changes.
 */

import * as fs from 'fs';
import * as path from 'path';

import type {
  GuardianLogEntry,
  GuardianTradeParams,
  GuardianResult,
  SecurityAlert,
} from './types.js';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

class GuardianLogger {
  private logDir: string;
  private logToFile: boolean;
  private logToConsole: boolean;
  private alertsFile: string;

  constructor() {
    this.logDir = process.env.GUARDIAN_LOG_DIR || process.env.LOG_DIR || './logs';
    this.logToFile = process.env.GUARDIAN_LOG_TO_FILE !== 'false';
    this.logToConsole = process.env.GUARDIAN_LOG_TO_CONSOLE !== 'false';
    this.alertsFile = path.join(this.logDir, 'guardian_alerts.json');

    // Create log directory if needed
    if (this.logToFile && !fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
  }

  private formatMessage(level: LogLevel, category: string, message: string, data?: Record<string, unknown>): string {
    const emoji = {
      debug: '🔍',
      info: '🛡️',
      warn: '⚠️',
      error: '❌',
    }[level];

    const timestamp = new Date().toISOString();
    let msg = `${emoji} [${timestamp}] [GUARDIAN:${category}] ${message}`;
    if (data) {
      msg += ` ${JSON.stringify(data)}`;
    }
    return msg;
  }

  private log(level: LogLevel, category: string, message: string, data?: Record<string, unknown>): void {
    // Console output
    if (this.logToConsole) {
      const formatted = this.formatMessage(level, category, message, data);
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
      this.writeToFile(level, category, message, data);
    }
  }

  private writeToFile(level: LogLevel, category: string, message: string, data?: Record<string, unknown>): void {
    const date = new Date().toISOString().split('T')[0];
    const filename = path.join(this.logDir, `guardian-${date}.log`);
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      category,
      message,
      data,
    };
    const line = JSON.stringify(entry) + '\n';

    try {
      fs.appendFileSync(filename, line);
    } catch (error) {
      console.error('[GUARDIAN] Failed to write log:', error);
    }
  }

  // General logging methods
  debug(message: string, data?: Record<string, unknown>): void {
    this.log('debug', 'DEBUG', message, data);
  }

  info(message: string, data?: Record<string, unknown>): void {
    this.log('info', 'INFO', message, data);
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this.log('warn', 'WARN', message, data);
  }

  error(message: string, data?: Record<string, unknown>): void {
    this.log('error', 'ERROR', message, data);
  }

  // Specialized Guardian logging methods

  /**
   * Log validation attempt with result
   */
  logValidation(params: GuardianTradeParams, result: GuardianResult): void {
    const level = result.approved ? 'info' : 'warn';
    const status = result.approved ? '✅ APPROVED' : '🚫 BLOCKED';
    
    const logData = {
      strategy: params.strategy,
      protocol: params.protocol,
      inputMint: params.inputMint,
      outputMint: params.outputMint,
      amountUsd: params.amountInUsd,
      slippageBps: params.slippageBps,
      approved: result.approved,
      blockReason: result.blockReason,
      securityRiskScore: result.securityResult.riskScore,
      threats: result.securityResult.threats,
      validationIssues: result.validationResult.reason,
      sanityIssues: result.sanityResult.issues,
    };

    this.log(level, 'VALIDATION', `${status} ${params.strategy}/${params.protocol || 'unknown'}`, logData);

    // Write detailed entry to guardian log
    if (this.logToFile) {
      const entry: GuardianLogEntry = {
        timestamp: new Date().toISOString(),
        eventType: result.approved ? 'validation' : 'block',
        strategy: params.strategy,
        protocol: params.protocol,
        params,
        result,
        blocked: !result.approved,
        blockReason: result.blockReason,
      };
      this.writeGuardianEntry(entry);
    }
  }

  /**
   * Log security alert
   */
  logSecurityAlert(alert: SecurityAlert): void {
    const levelMap = { low: 'info', medium: 'warn', high: 'error', critical: 'error' } as const;
    const level = levelMap[alert.severity];

    this.log(level, 'SECURITY', `🚨 ${alert.alertType}: ${alert.description}`, {
      severity: alert.severity,
      address: alert.address,
      ...alert.data,
    });

    // Append to alerts file
    if (this.logToFile) {
      this.appendAlert(alert);
    }
  }

  /**
   * Log config change
   */
  logConfigChange(oldConfig: Record<string, unknown>, newConfig: Record<string, unknown>): void {
    const changes: Record<string, { old: unknown; new: unknown }> = {};

    for (const key of Object.keys(newConfig)) {
      if (JSON.stringify(oldConfig[key]) !== JSON.stringify(newConfig[key])) {
        changes[key] = { old: oldConfig[key], new: newConfig[key] };
      }
    }

    if (Object.keys(changes).length > 0) {
      this.log('info', 'CONFIG', 'Guardian configuration changed', { changes });
    }
  }

  /**
   * Log blocked transaction
   */
  logBlocked(params: GuardianTradeParams, reason: string): void {
    this.log('warn', 'BLOCKED', `Transaction blocked: ${reason}`, {
      strategy: params.strategy,
      protocol: params.protocol,
      inputMint: params.inputMint,
      outputMint: params.outputMint,
      amountUsd: params.amountInUsd,
      reason,
    });
  }

  // Private helper methods

  private writeGuardianEntry(entry: GuardianLogEntry): void {
    const date = entry.timestamp.split('T')[0];
    const filename = path.join(this.logDir, `guardian-entries-${date}.jsonl`);
    const line = JSON.stringify(entry) + '\n';

    try {
      fs.appendFileSync(filename, line);
    } catch (error) {
      console.error('[GUARDIAN] Failed to write entry:', error);
    }
  }

  private appendAlert(alert: SecurityAlert): void {
    try {
      let alerts: SecurityAlert[] = [];
      if (fs.existsSync(this.alertsFile)) {
        const content = fs.readFileSync(this.alertsFile, 'utf-8');
        alerts = JSON.parse(content);
      }
      alerts.push(alert);
      // Keep only last 1000 alerts
      if (alerts.length > 1000) {
        alerts = alerts.slice(-1000);
      }
      fs.writeFileSync(this.alertsFile, JSON.stringify(alerts, null, 2));
    } catch (error) {
      console.error('[GUARDIAN] Failed to write alert:', error);
    }
  }

  /**
   * Get recent alerts
   */
  getRecentAlerts(count: number = 50): SecurityAlert[] {
    try {
      if (!fs.existsSync(this.alertsFile)) {
        return [];
      }
      const content = fs.readFileSync(this.alertsFile, 'utf-8');
      const alerts = JSON.parse(content) as SecurityAlert[];
      return alerts.slice(-count);
    } catch {
      return [];
    }
  }

  /**
   * Get validation stats for a time period
   */
  getStats(hours: number = 24): { total: number; approved: number; blocked: number; blockRate: number } {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    const date = cutoff.toISOString().split('T')[0];
    const filename = path.join(this.logDir, `guardian-entries-${date}.jsonl`);

    let total = 0;
    let approved = 0;
    let blocked = 0;

    try {
      if (fs.existsSync(filename)) {
        const lines = fs.readFileSync(filename, 'utf-8').trim().split('\n');
        for (const line of lines) {
          if (!line) continue;
          const entry = JSON.parse(line) as GuardianLogEntry;
          const entryTime = new Date(entry.timestamp);
          if (entryTime >= cutoff) {
            total++;
            if (entry.blocked) {
              blocked++;
            } else {
              approved++;
            }
          }
        }
      }
    } catch {
      // Ignore errors reading stats
    }

    return {
      total,
      approved,
      blocked,
      blockRate: total > 0 ? (blocked / total) * 100 : 0,
    };
  }
}

// Singleton instance
export const guardianLogger = new GuardianLogger();

// Export class for testing
export { GuardianLogger };

