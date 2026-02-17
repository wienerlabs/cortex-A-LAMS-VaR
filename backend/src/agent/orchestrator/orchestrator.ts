import { v4 as uuidv4 } from "uuid";
import type {
  OrchestratorConfig,
  OrchestratorDecision,
  MLPredictionResponse,
  StrategyRecommendation,
  StrategyType,
} from "./types.js";
import { MLAgentClient } from "./ml-client.js";
import { StrategyExecutor, type ExecutionParams, type ExecutionResult } from "./strategy-executor.js";
import { CortexSolanaAgent } from "../solana-agent.js";
import type { AgentConfig, AgentLimits } from "../types.js";
import { DEFAULT_AGENT_LIMITS } from "../types.js";

export interface OrchestratorStatus {
  running: boolean;
  mlAgentAvailable: boolean;
  lastDecision: OrchestratorDecision | null;
  decisionsToday: number;
  executionsToday: number;
  pnlToday: number;
}

export class Orchestrator {
  private config: OrchestratorConfig;
  private mlClient: MLAgentClient;
  private executor: StrategyExecutor;
  private solanaAgent: CortexSolanaAgent | null = null;
  private decisions: OrchestratorDecision[] = [];
  private running = false;

  constructor(config: Partial<OrchestratorConfig> = {}) {
    const defaultConfig: OrchestratorConfig = {
      mlAgentUrl: process.env.ML_AGENT_URL ?? "http://localhost:8000",
      autoExecute: false,
      minConfidence: 0.7,
      maxTradeAmountUsd: 100,
      dailyLimitUsd: 1000,
      strategies: ["arbitrage", "staking", "yield"],
    };
    this.config = { ...defaultConfig, ...config };
    this.mlClient = new MLAgentClient(this.config.mlAgentUrl);
    this.executor = new StrategyExecutor(this.config);
  }

  async initialize(): Promise<void> {
    const privateKey = process.env.AGENT_WALLET_PRIVATE_KEY;
    if (privateKey) {
      const agentConfig: AgentConfig = {
        rpcUrl: process.env.SOLANA_RPC_URL ?? "https://api.devnet.solana.com",
      };
      const limits: AgentLimits = {
        ...DEFAULT_AGENT_LIMITS,
        maxTradeAmountUsd: this.config.maxTradeAmountUsd,
        dailyTradeLimitUsd: this.config.dailyLimitUsd,
      };
      this.solanaAgent = new CortexSolanaAgent(privateKey, agentConfig, limits);
      this.executor.setAgent(this.solanaAgent);
    }
  }

  async getStatus(): Promise<OrchestratorStatus> {
    const mlAvailable = await this.mlClient.isAvailable();
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const todayDecisions = this.decisions.filter((d) => d.timestamp >= today);
    const executions = todayDecisions.filter((d) => d.executed);

    return {
      running: this.running,
      mlAgentAvailable: mlAvailable,
      lastDecision: this.decisions[this.decisions.length - 1] ?? null,
      decisionsToday: todayDecisions.length,
      executionsToday: executions.length,
      pnlToday: 0,
    };
  }

  async evaluateStrategy(
    strategy: StrategyType,
    features: Record<string, number>
  ): Promise<OrchestratorDecision> {
    const decision: OrchestratorDecision = {
      id: uuidv4(),
      timestamp: new Date(),
      mlPrediction: null,
      recommendation: null,
      executionResult: null,
      approved: false,
      executed: false,
    };

    try {
      const prediction = await this.mlClient.predict({ strategy, features });
      decision.mlPrediction = prediction;

      if (prediction.should_execute && prediction.confidence >= this.config.minConfidence) {
        decision.recommendation = {
          strategy,
          action: this.mapPredictionToAction(prediction),
          confidence: prediction.confidence,
          expected_profit: 0,
          reasoning: prediction.explanation.reason ?? "ML model recommendation",
        };
        decision.approved = this.config.autoExecute;
      }
    } catch (error) {
      decision.error = error instanceof Error ? error.message : "Unknown error";
    }

    this.decisions.push(decision);
    return decision;
  }

  async getRecommendations(): Promise<OrchestratorDecision> {
    const decision: OrchestratorDecision = {
      id: uuidv4(),
      timestamp: new Date(),
      mlPrediction: null,
      recommendation: null,
      executionResult: null,
      approved: false,
      executed: false,
    };

    try {
      const recs = await this.mlClient.getRecommendations();
      if (recs.recommendations.length > 0) {
        const best = recs.recommendations[0];
        decision.recommendation = {
          ...best,
          action: best.action as any,
        };
        decision.approved = this.config.autoExecute && best.confidence >= this.config.minConfidence;
      }
    } catch (error) {
      decision.error = error instanceof Error ? error.message : "Unknown error";
    }

    this.decisions.push(decision);
    return decision;
  }

  async executeDecision(
    decisionId: string,
    params: ExecutionParams = {}
  ): Promise<ExecutionResult> {
    const decision = this.decisions.find((d) => d.id === decisionId);
    if (!decision) throw new Error(`Decision ${decisionId} not found`);
    if (!decision.recommendation) throw new Error("No recommendation to execute");
    if (decision.executed) throw new Error("Decision already executed");

    const result = await this.executor.execute(decision.recommendation, params);
    decision.executed = result.success;
    decision.executionResult = {
      execution_id: decisionId,
      strategy: decision.recommendation.strategy,
      action: decision.recommendation.action,
      state: result.success ? "COMPLETED" : "FAILED",
      tx_hash: result.signature,
      error: result.error,
      timestamp: result.timestamp.toISOString(),
    };

    return result;
  }

  private mapPredictionToAction(prediction: MLPredictionResponse): "swap" | "stake" | "hold" {
    const action = prediction.action.toLowerCase();
    if (action.includes("swap") || action.includes("execute")) return "swap";
    if (action.includes("stake")) return "stake";
    return "hold";
  }

  getDecisions(limit = 50): OrchestratorDecision[] {
    return this.decisions.slice(-limit);
  }

  getConfig(): OrchestratorConfig {
    return { ...this.config };
  }

  updateConfig(updates: Partial<OrchestratorConfig>): void {
    this.config = { ...this.config, ...updates };
    this.executor = new StrategyExecutor(this.config, this.solanaAgent ?? undefined);
  }
}

let orchestratorInstance: Orchestrator | null = null;

export const getOrchestrator = (config?: Partial<OrchestratorConfig>): Orchestrator => {
  if (!orchestratorInstance) {
    orchestratorInstance = new Orchestrator(config);
  }
  return orchestratorInstance;
};

