/**
 * PM (Portfolio Manager) Logger
 * 
 * Specialized logging for PM approval workflow decisions.
 * Logs all trade submissions, approvals, rejections, and expirations.
 */

import * as fs from 'fs';
import * as path from 'path';
import { logger } from '../logger.js';
import type { PMLogEntry, PendingTrade, ApprovalStatus } from './types.js';

class PMLogger {
  private logDir: string;
  private logFile: string;

  constructor() {
    this.logDir = path.resolve(process.cwd(), 'logs/pm');
    this.logFile = path.join(this.logDir, `pm-${this.getDateString()}.log`);
    this.ensureLogDir();
  }

  private getDateString(): string {
    return new Date().toISOString().split('T')[0];
  }

  private ensureLogDir(): void {
    try {
      if (!fs.existsSync(this.logDir)) {
        fs.mkdirSync(this.logDir, { recursive: true });
      }
    } catch (error) {
      // Silently fail - will use console logging only
    }
  }

  private writeToFile(entry: PMLogEntry): void {
    try {
      const line = JSON.stringify(entry) + '\n';
      fs.appendFileSync(this.logFile, line);
    } catch (error) {
      // Silently fail - console logging is primary
    }
  }

  /**
   * Log trade submission to approval queue
   */
  logSubmission(trade: PendingTrade): void {
    const entry: PMLogEntry = {
      timestamp: new Date().toISOString(),
      eventType: 'submission',
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      reason: trade.reasoning,
    };

    logger.info('[PM] Trade submitted for approval', {
      tradeId: trade.id,
      strategy: trade.strategy,
      action: trade.action,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      confidence: trade.confidence,
      riskScore: trade.risk.riskScore,
      expiresAt: trade.expiresAt.toISOString(),
    });

    this.writeToFile(entry);
  }

  /**
   * Log trade approval
   */
  logApproval(trade: PendingTrade, approver: string, waitTimeMs: number): void {
    const entry: PMLogEntry = {
      timestamp: new Date().toISOString(),
      eventType: 'approval',
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      approver,
      waitTimeMs,
    };

    logger.info('[PM] Trade APPROVED', {
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      approver,
      waitTimeMs,
    });

    this.writeToFile(entry);
  }

  /**
   * Log trade rejection
   */
  logRejection(trade: PendingTrade, approver: string, reason: string): void {
    const entry: PMLogEntry = {
      timestamp: new Date().toISOString(),
      eventType: 'rejection',
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      approver,
      reason,
    };

    logger.warn('[PM] Trade REJECTED', {
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      approver,
      reason,
    });

    this.writeToFile(entry);
  }

  /**
   * Log trade expiration
   */
  logExpiration(trade: PendingTrade): void {
    const entry: PMLogEntry = {
      timestamp: new Date().toISOString(),
      eventType: 'expiration',
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      reason: 'Approval timeout exceeded',
    };

    logger.warn('[PM] Trade EXPIRED', {
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      submittedAt: trade.submittedAt.toISOString(),
      expiresAt: trade.expiresAt.toISOString(),
    });

    this.writeToFile(entry);
  }

  /**
   * Log auto-approval (when PM offline and auto-approve enabled)
   */
  logAutoApproval(trade: PendingTrade, reason: string): void {
    const entry: PMLogEntry = {
      timestamp: new Date().toISOString(),
      eventType: 'auto_approval',
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      reason,
    };

    logger.info('[PM] Trade AUTO-APPROVED', {
      tradeId: trade.id,
      strategy: trade.strategy,
      asset: trade.asset,
      amountUsd: trade.amountUsd,
      reason,
    });

    this.writeToFile(entry);
  }
}

// Singleton instance
export const pmLogger = new PMLogger();
export { PMLogger };

