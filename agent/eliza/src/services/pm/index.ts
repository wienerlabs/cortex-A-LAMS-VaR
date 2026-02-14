/**
 * PM (Portfolio Manager) Approval Workflow
 * 
 * Human-in-the-loop approval system for critical trades.
 * Exports all PM components for use in executors.
 */

// Types
export type {
  ApprovalStatus,
  TradeRiskInfo,
  PendingTrade,
  ApprovalRules,
  ApprovalResult,
  PMConfig,
  PMLogEntry,
  QueueTradeParams,
  WaitForApprovalOptions,
} from './types.js';

export { ApprovalStatus as ApprovalStatusEnum } from './types.js';

// Logger
export { pmLogger, PMLogger } from './logger.js';

// Approval Queue
export { approvalQueue, ApprovalQueue } from './approvalQueue.js';

// Decision Engine
export { pmDecisionEngine, PMDecisionEngine } from './pmDecisionEngine.js';

