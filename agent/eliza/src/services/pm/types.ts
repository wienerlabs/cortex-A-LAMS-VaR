/**
 * PM (Portfolio Manager) Approval Workflow Types
 * 
 * Type definitions for human-in-the-loop trade approval system.
 * All thresholds are loaded from configuration - no hardcoded values.
 */

// ============= APPROVAL STATUS =============

export enum ApprovalStatus {
  PENDING = 'pending',
  APPROVED = 'approved',
  REJECTED = 'rejected',
  EXPIRED = 'expired',
  AUTO_APPROVED = 'auto_approved',
}

// ============= TRADE RISK INFO =============

export interface TradeRiskInfo {
  volatility: number;        // 24h volatility percentage
  liquidityScore: number;    // 0-100, higher = more liquid
  riskScore: number;         // 0-100, higher = more risky
  marketRegime?: string;     // 'bull' | 'bear' | 'sideways'
}

// ============= PENDING TRADE =============

export interface PendingTrade {
  id: string;
  strategy: 'spot' | 'lp' | 'perps' | 'arbitrage' | 'lending' | 'pumpfun';
  action: 'BUY' | 'SELL' | 'OPEN' | 'CLOSE' | 'DEPOSIT' | 'WITHDRAW' | 'BORROW' | 'REPAY';
  asset: string;
  assetMint?: string;
  amount: number;
  amountUsd: number;
  confidence: number;        // ML model confidence 0-1
  risk: TradeRiskInfo;
  reasoning: string;         // Why the trade was suggested
  submittedAt: Date;
  expiresAt: Date;
  status: ApprovalStatus;
  approver?: string;
  approvedAt?: Date;
  rejectionReason?: string;
  
  // Additional context
  protocol?: string;         // e.g., 'jupiter', 'drift', 'orca'
  leverage?: number;         // For perps
  portfolioPercentage?: number; // % of portfolio this trade represents
}

// ============= APPROVAL RULES =============

export interface ApprovalRules {
  // Size-based thresholds
  minPositionForApproval: number;      // USD amount requiring approval
  minPercentageForApproval: number;    // % of portfolio requiring approval
  
  // Risk-based thresholds
  requireApprovalIfVolatilityAbove: number;   // e.g., 50%
  requireApprovalIfConfidenceBelow: number;   // e.g., 0.6 (60%)
  requireApprovalIfRiskScoreAbove: number;    // e.g., 70
  
  // Strategy-based rules
  alwaysRequireApprovalFor: string[];  // Strategies that always need approval
  neverRequireApprovalFor: string[];   // Strategies that never need approval (auto-approve)
  
  // Timeout settings
  approvalTimeoutSeconds: number;      // How long to wait for approval
  autoApproveWhenPmOffline: boolean;   // If true, auto-approve when PM unavailable
}

// ============= APPROVAL RESULT =============

export interface ApprovalResult {
  approved: boolean;
  tradeId: string;
  status: ApprovalStatus;
  approver?: string;
  approvedAt?: Date;
  rejectionReason?: string;
  waitTimeMs?: number;
}

// ============= PM CONFIGURATION =============

export interface PMConfig {
  enabled: boolean;
  rules: ApprovalRules;
  
  // Notification settings
  notifications: {
    enabled: boolean;
    channels: string[];      // e.g., ['console', 'telegram', 'discord']
  };
  
  // Logging settings
  logging: {
    logAllSubmissions: boolean;
    logApprovals: boolean;
    logRejections: boolean;
    logExpirations: boolean;
  };
}

// ============= PM LOG ENTRY =============

export interface PMLogEntry {
  timestamp: string;
  eventType: 'submission' | 'approval' | 'rejection' | 'expiration' | 'auto_approval';
  tradeId: string;
  strategy: string;
  asset: string;
  amountUsd: number;
  approver?: string;
  reason?: string;
  waitTimeMs?: number;
}

// ============= QUEUE TRADE PARAMS =============

export interface QueueTradeParams {
  strategy: PendingTrade['strategy'];
  action: PendingTrade['action'];
  asset: string;
  assetMint?: string;
  amount: number;
  amountUsd: number;
  confidence: number;
  risk: TradeRiskInfo;
  reasoning: string;
  protocol?: string;
  leverage?: number;
  portfolioPercentage?: number;
}

// ============= WAIT OPTIONS =============

export interface WaitForApprovalOptions {
  timeoutMs?: number;        // Override default timeout
  pollIntervalMs?: number;   // How often to check status
}

