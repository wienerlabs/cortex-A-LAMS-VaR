import { PublicKey } from "@solana/web3.js";

export interface AgentConfig {
  rpcUrl: string;
  openaiApiKey?: string;
  heliusApiKey?: string;
}

export interface AgentLimits {
  maxTradeAmountUsd: number;
  dailyTradeLimitUsd: number;
  maxSlippageBps: number;
  allowedActions: AgentActionType[];
  requiresApprovalAboveUsd: number;
  maxPositionSizePercent: number;
}

export type AgentActionType = 
  | "swap"
  | "stake" 
  | "unstake"
  | "rebalance"
  | "deposit_strategy"
  | "withdraw_strategy";

export interface TradeParams {
  inputMint: PublicKey;
  outputMint: PublicKey;
  amountIn: number;
  slippageBps?: number;
  reason?: string;
}

export interface RebalanceParams {
  targetAllocations: TokenAllocation[];
  tolerance: number;
}

export interface TokenAllocation {
  mint: PublicKey;
  targetPercent: number;
}

export interface AgentWalletInfo {
  publicKey: string;
  solBalance: number;
  tokens: TokenBalance[];
  totalValueUsd: number;
}

export interface TokenBalance {
  mint: string;
  symbol: string;
  balance: number;
  valueUsd: number;
}

export interface AgentActivityLog {
  id: string;
  timestamp: Date;
  action: AgentActionType;
  status: "pending" | "success" | "failed";
  params: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  txSignature?: string;
}

export interface SwapResult {
  signature: string;
  inputAmount: number;
  outputAmount: number;
  inputMint: string;
  outputMint: string;
  priceImpact: number;
}

export interface StakeResult {
  signature: string;
  amount: number;
  pool: string;
}

export const DEFAULT_AGENT_LIMITS: AgentLimits = {
  maxTradeAmountUsd: 1000,
  dailyTradeLimitUsd: 10000,
  maxSlippageBps: 100, // 1%
  allowedActions: ["swap", "stake", "unstake", "rebalance"],
  requiresApprovalAboveUsd: 5000,
  maxPositionSizePercent: 50,
};

export const KNOWN_TOKENS = {
  SOL: new PublicKey("So11111111111111111111111111111111111111112"),
  USDC: new PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
  USDT: new PublicKey("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"),
  BONK: new PublicKey("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"),
  JUP: new PublicKey("JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"),
  JITOSOL: new PublicKey("J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn"),
} as const;

