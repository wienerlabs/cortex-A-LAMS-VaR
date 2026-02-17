/**
 * PM Approval Queue Service
 * 
 * Manages pending trades awaiting PM approval.
 * Provides queueing, approval, rejection, and wait functionality.
 */

import { randomUUID } from 'crypto';
import { pmLogger } from './logger.js';
import type {
  PendingTrade,
  QueueTradeParams,
  ApprovalResult,
  ApprovalStatus,
  WaitForApprovalOptions,
  ApprovalRules,
} from './types.js';
import { ApprovalStatus as Status } from './types.js';

class ApprovalQueue {
  private queue: Map<string, PendingTrade> = new Map();
  private rules: ApprovalRules | null = null;
  private cleanupIntervalId: NodeJS.Timeout | null = null;

  constructor() {
    // Start background cleanup task
    this.startCleanupTask();
  }

  /**
   * Set approval rules (called by decision engine after config load)
   */
  setRules(rules: ApprovalRules): void {
    this.rules = rules;
  }

  /**
   * Queue a trade for approval
   */
  queueTrade(params: QueueTradeParams): string {
    const tradeId = randomUUID();
    const now = new Date();
    const timeoutSeconds = this.rules?.approvalTimeoutSeconds || 300;
    const expiresAt = new Date(now.getTime() + timeoutSeconds * 1000);

    const trade: PendingTrade = {
      id: tradeId,
      strategy: params.strategy,
      action: params.action,
      asset: params.asset,
      assetMint: params.assetMint,
      amount: params.amount,
      amountUsd: params.amountUsd,
      confidence: params.confidence,
      risk: params.risk,
      reasoning: params.reasoning,
      protocol: params.protocol,
      leverage: params.leverage,
      portfolioPercentage: params.portfolioPercentage,
      submittedAt: now,
      expiresAt,
      status: Status.PENDING,
    };

    this.queue.set(tradeId, trade);
    pmLogger.logSubmission(trade);

    return tradeId;
  }

  /**
   * Get all pending trades
   */
  getPendingTrades(): PendingTrade[] {
    return Array.from(this.queue.values())
      .filter(trade => trade.status === Status.PENDING);
  }

  /**
   * Get trade by ID
   */
  getTrade(tradeId: string): PendingTrade | null {
    return this.queue.get(tradeId) || null;
  }

  /**
   * Get trade status
   */
  getTradeStatus(tradeId: string): ApprovalStatus | null {
    const trade = this.queue.get(tradeId);
    return trade ? trade.status : null;
  }

  /**
   * Approve a trade
   */
  approveTrade(tradeId: string, approver: string): ApprovalResult {
    const trade = this.queue.get(tradeId);
    
    if (!trade) {
      return {
        approved: false,
        tradeId,
        status: Status.REJECTED,
        rejectionReason: 'Trade not found',
      };
    }

    if (trade.status !== Status.PENDING) {
      return {
        approved: false,
        tradeId,
        status: trade.status,
        rejectionReason: `Trade already ${trade.status}`,
      };
    }

    const now = new Date();
    const waitTimeMs = now.getTime() - trade.submittedAt.getTime();

    trade.status = Status.APPROVED;
    trade.approver = approver;
    trade.approvedAt = now;

    pmLogger.logApproval(trade, approver, waitTimeMs);

    return {
      approved: true,
      tradeId,
      status: Status.APPROVED,
      approver,
      approvedAt: now,
      waitTimeMs,
    };
  }

  /**
   * Reject a trade
   */
  rejectTrade(tradeId: string, reason: string, approver: string): ApprovalResult {
    const trade = this.queue.get(tradeId);
    
    if (!trade) {
      return {
        approved: false,
        tradeId,
        status: Status.REJECTED,
        rejectionReason: 'Trade not found',
      };
    }

    if (trade.status !== Status.PENDING) {
      return {
        approved: false,
        tradeId,
        status: trade.status,
        rejectionReason: `Trade already ${trade.status}`,
      };
    }

    trade.status = Status.REJECTED;
    trade.approver = approver;
    trade.rejectionReason = reason;

    pmLogger.logRejection(trade, approver, reason);

    return {
      approved: false,
      tradeId,
      status: Status.REJECTED,
      approver,
      rejectionReason: reason,
    };
  }

  /**
   * Wait for trade approval with polling
   */
  async waitForApproval(
    tradeId: string,
    options?: WaitForApprovalOptions
  ): Promise<ApprovalResult> {
    const trade = this.queue.get(tradeId);

    if (!trade) {
      return {
        approved: false,
        tradeId,
        status: Status.REJECTED,
        rejectionReason: 'Trade not found',
      };
    }

    const timeoutMs = options?.timeoutMs ||
      (this.rules?.approvalTimeoutSeconds || 300) * 1000;
    const pollIntervalMs = options?.pollIntervalMs || 1000;
    const startTime = Date.now();

    return new Promise((resolve) => {
      const checkStatus = () => {
        const currentTrade = this.queue.get(tradeId);

        if (!currentTrade) {
          resolve({
            approved: false,
            tradeId,
            status: Status.REJECTED,
            rejectionReason: 'Trade removed from queue',
          });
          return;
        }

        // Check if approved
        if (currentTrade.status === Status.APPROVED) {
          resolve({
            approved: true,
            tradeId,
            status: Status.APPROVED,
            approver: currentTrade.approver,
            approvedAt: currentTrade.approvedAt,
            waitTimeMs: Date.now() - startTime,
          });
          return;
        }

        // Check if rejected
        if (currentTrade.status === Status.REJECTED) {
          resolve({
            approved: false,
            tradeId,
            status: Status.REJECTED,
            approver: currentTrade.approver,
            rejectionReason: currentTrade.rejectionReason,
            waitTimeMs: Date.now() - startTime,
          });
          return;
        }

        // Check if expired
        if (currentTrade.status === Status.EXPIRED || Date.now() > currentTrade.expiresAt.getTime()) {
          this.expireTrade(tradeId);
          resolve({
            approved: false,
            tradeId,
            status: Status.EXPIRED,
            rejectionReason: 'Approval timeout exceeded',
            waitTimeMs: Date.now() - startTime,
          });
          return;
        }

        // Check if overall timeout exceeded
        if (Date.now() - startTime > timeoutMs) {
          this.expireTrade(tradeId);
          resolve({
            approved: false,
            tradeId,
            status: Status.EXPIRED,
            rejectionReason: 'Wait timeout exceeded',
            waitTimeMs: Date.now() - startTime,
          });
          return;
        }

        // Still pending, poll again
        setTimeout(checkStatus, pollIntervalMs);
      };

      checkStatus();
    });
  }

  /**
   * Auto-approve a trade (when PM offline and auto-approve enabled)
   */
  autoApprove(tradeId: string, reason: string): ApprovalResult {
    const trade = this.queue.get(tradeId);

    if (!trade) {
      return {
        approved: false,
        tradeId,
        status: Status.REJECTED,
        rejectionReason: 'Trade not found',
      };
    }

    if (trade.status !== Status.PENDING) {
      return {
        approved: trade.status === Status.APPROVED || trade.status === Status.AUTO_APPROVED,
        tradeId,
        status: trade.status,
      };
    }

    const now = new Date();
    const waitTimeMs = now.getTime() - trade.submittedAt.getTime();

    trade.status = Status.AUTO_APPROVED;
    trade.approver = 'SYSTEM';
    trade.approvedAt = now;

    pmLogger.logAutoApproval(trade, reason);

    return {
      approved: true,
      tradeId,
      status: Status.AUTO_APPROVED,
      approver: 'SYSTEM',
      approvedAt: now,
      waitTimeMs,
    };
  }

  /**
   * Expire a trade
   */
  private expireTrade(tradeId: string): void {
    const trade = this.queue.get(tradeId);
    if (trade && trade.status === Status.PENDING) {
      trade.status = Status.EXPIRED;
      pmLogger.logExpiration(trade);
    }
  }

  /**
   * Start background cleanup task to expire old trades
   */
  private startCleanupTask(): void {
    this.cleanupIntervalId = setInterval(() => {
      const now = Date.now();
      for (const [tradeId, trade] of this.queue.entries()) {
        if (trade.status === Status.PENDING && now > trade.expiresAt.getTime()) {
          this.expireTrade(tradeId);
        }
        // Remove completed trades older than 1 hour
        if (trade.status !== Status.PENDING &&
            now - trade.submittedAt.getTime() > 3600000) {
          this.queue.delete(tradeId);
        }
      }
    }, 10000); // Check every 10 seconds
  }

  /**
   * Stop cleanup task (for graceful shutdown)
   */
  stopCleanupTask(): void {
    if (this.cleanupIntervalId) {
      clearInterval(this.cleanupIntervalId);
      this.cleanupIntervalId = null;
    }
  }

  /**
   * Get queue size
   */
  getQueueSize(): number {
    return this.queue.size;
  }

  /**
   * Get pending count
   */
  getPendingCount(): number {
    return this.getPendingTrades().length;
  }

  /**
   * Clear all trades from queue (for testing)
   */
  clearAll(): void {
    this.queue.clear();
  }
}

// Singleton instance
export const approvalQueue = new ApprovalQueue();
export { ApprovalQueue };

