/**
 * Guardian Validator Types
 * 
 * Type definitions for pre-execution transaction validation.
 * All thresholds are loaded from configuration - no hardcoded values.
 */

import type { PublicKey, Transaction, VersionedTransaction } from '@solana/web3.js';

// ============= VALIDATION RESULTS =============

export interface ValidationResult {
  valid: boolean;
  reason?: string;
  details?: Record<string, unknown>;
}

export interface SecurityResult {
  safe: boolean;
  threats: string[];
  riskScore: number;  // 0-100, higher = more risky
}

export interface SanityResult {
  sane: boolean;
  issues: string[];
  warnings: string[];
}

export interface GuardianResult {
  approved: boolean;
  validationResult: ValidationResult;
  securityResult: SecurityResult;
  sanityResult: SanityResult;
  simulationResult?: SimulationResult;
  timestamp: Date;
  executionAllowed: boolean;
  blockReason?: string;
}

// ============= TRADE PARAMETERS =============

export interface GuardianTradeParams {
  // Token addresses
  inputMint: string;
  outputMint: string;
  
  // Amounts
  amountIn: number;
  amountInUsd: number;
  expectedAmountOut?: number;
  
  // Slippage and fees
  slippageBps: number;
  estimatedGasSol?: number;
  priorityFeeLamports?: number;
  
  // Price impact
  priceImpactPct?: number;
  
  // Context
  strategy: 'spot' | 'lp' | 'perps' | 'arbitrage' | 'lending' | 'pumpfun';
  protocol?: string;
  walletAddress: string;
}

// ============= CONFIGURATION =============

export interface GuardianConfig {
  // Global enable/disable
  enabled: boolean;
  
  // Slippage limits
  minSlippagePercent: number;
  maxSlippagePercent: number;
  
  // Position size limits
  minPositionUsd: number;
  maxPositionUsd: number;
  
  // Gas limits
  maxGasSol: number;
  maxPriorityFeeLamports: number;
  
  // Liquidity requirements
  minLiquidityUsd: number;
  
  // Price impact limits
  maxPriceImpactPercent: number;
  
  // Security
  blacklistedAddresses: string[];
  suspiciousPatterns: string[];
  
  // Strategy-specific overrides
  strategyOverrides: {
    [strategy: string]: Partial<GuardianConfig>;
  };
}

// ============= TRANSACTION TYPES =============

export type SolanaTransaction = Transaction | VersionedTransaction;

export interface TransactionInfo {
  transaction: SolanaTransaction;
  instructions?: number;
  signers?: string[];
  recentBlockhash?: string;
  feePayer?: string;
}

// ============= LOGGING TYPES =============

export interface GuardianLogEntry {
  timestamp: string;
  eventType: 'validation' | 'block' | 'security_alert' | 'config_change';
  strategy: string;
  protocol?: string;
  params?: GuardianTradeParams;
  result: GuardianResult | ValidationResult | SecurityResult;
  blocked: boolean;
  blockReason?: string;
}

export interface SecurityAlert {
  timestamp: Date;
  severity: 'low' | 'medium' | 'high' | 'critical';
  alertType: string;
  address?: string;
  description: string;
  data?: Record<string, unknown>;
}

// ============= TOKEN VALIDATION =============

export interface TokenValidation {
  mint: string;
  isValid: boolean;
  isBlacklisted: boolean;
  hasLiquidity: boolean;
  liquidityUsd?: number;
  isHoneypot?: boolean;
  riskFlags: string[];
}

// ============= SIMULATION TYPES =============

export interface SimulationResult {
  success: boolean;
  estimatedOutputAmount?: number;
  estimatedSlippagePct?: number;
  estimatedPriceImpactPct?: number;
  estimatedGasLamports?: number;
  jupiterQuoteId?: string;
  routeDescription?: string;
  warnings: string[];
  error?: string;
  simulationTimeMs: number;
}

export interface JupiterQuoteResponse {
  inputMint: string;
  outputMint: string;
  inAmount: string;
  outAmount: string;
  priceImpactPct: string;
  routePlan: Array<{
    swapInfo: {
      ammKey: string;
      label: string;
      inputMint: string;
      outputMint: string;
      inAmount: string;
      outAmount: string;
      feeAmount: string;
      feeMint: string;
    };
    percent: number;
  }>;
  otherAmountThreshold: string;
  swapMode: string;
  slippageBps: number;
  contextSlot?: number;
}

// ============= CONSTANTS =============

// Solana address validation regex (base58, 32-44 chars)
export const SOLANA_ADDRESS_REGEX = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;

// Well-known token mints for validation
export const KNOWN_STABLE_MINTS = [
  'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC
  'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB', // USDT
  'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm', // PYUSD
];

export const KNOWN_MAJOR_MINTS = [
  'So11111111111111111111111111111111111111112',   // Wrapped SOL
  'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  // mSOL
  'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn', // JitoSOL
];

