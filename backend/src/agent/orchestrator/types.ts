export type StrategyType = "arbitrage" | "lending" | "lp_provision" | "yield" | "staking";
export type ActionType = "swap" | "stake" | "unstake" | "rebalance" | "deposit" | "withdraw" | "hold";

export interface MLPredictionRequest {
  strategy: StrategyType;
  features: Record<string, number>;
}

export interface MLPredictionResponse {
  prediction_id: string;
  strategy: StrategyType;
  prediction: number;
  confidence: number;
  action: string;
  should_execute: boolean;
  explanation: {
    model_version?: string;
    probability?: number;
    threshold?: number;
    reason?: string;
    [key: string]: unknown;
  };
  timestamp: string;
}

export interface StrategyRecommendation {
  strategy: StrategyType;
  action: ActionType;
  confidence: number;
  expected_profit: number;
  reasoning: string;
}

export interface RecommendationsResponse {
  recommendations: StrategyRecommendation[];
  best_strategy: StrategyType | null;
  timestamp: string;
}

export interface ExecutionRequest {
  strategy: StrategyType;
  action: ActionType;
  params: Record<string, unknown>;
  simulate_only?: boolean;
}

export interface ExecutionResponse {
  execution_id: string;
  strategy: StrategyType;
  action: string;
  state: "SIMULATING" | "AWAITING_APPROVAL" | "EXECUTING" | "COMPLETED" | "FAILED" | "CANCELLED";
  simulation_result?: {
    success: boolean;
    gas_estimate?: number;
    expected_profit?: number;
    slippage_estimate?: number;
  };
  tx_hash?: string;
  error?: string;
  timestamp: string;
}

export interface OrchestratorConfig {
  mlAgentUrl: string;
  autoExecute: boolean;
  minConfidence: number;
  maxTradeAmountUsd: number;
  dailyLimitUsd: number;
  strategies: StrategyType[];
}

export interface OrchestratorDecision {
  id: string;
  timestamp: Date;
  mlPrediction: MLPredictionResponse | null;
  recommendation: StrategyRecommendation | null;
  executionResult: ExecutionResponse | null;
  approved: boolean;
  executed: boolean;
  error?: string;
}

export interface MarketData {
  tokenPrices: Record<string, number>;
  poolData?: Record<string, unknown>;
  lendingRates?: Record<string, { supply: number; borrow: number }>;
  timestamp: Date;
}

export interface FeatureSet {
  strategy: StrategyType;
  features: Record<string, number>;
  source: "live" | "historical";
  timestamp: Date;
}

export const DEFAULT_ORCHESTRATOR_CONFIG: OrchestratorConfig = {
  mlAgentUrl: process.env.ML_AGENT_URL ?? "http://localhost:8000",
  autoExecute: false,
  minConfidence: 0.7,
  maxTradeAmountUsd: 100,
  dailyLimitUsd: 1000,
  strategies: ["arbitrage", "staking", "yield"],
};

