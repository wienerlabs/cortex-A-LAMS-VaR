import { PublicKey } from "@solana/web3.js";
import type { StrategyType, ActionType, StrategyRecommendation, OrchestratorConfig } from "./types.js";
import { CortexSolanaAgent } from "../solana-agent.js";
import { KNOWN_TOKENS } from "../types.js";

export interface ExecutionResult {
  success: boolean;
  action: ActionType;
  signature?: string;
  amountIn?: number;
  amountOut?: number;
  error?: string;
  timestamp: Date;
}

export interface ExecutionParams {
  inputMint?: string;
  outputMint?: string;
  amountUsd?: number;
  slippageBps?: number;
}

export class StrategyExecutor {
  private agent: CortexSolanaAgent | null = null;
  private config: OrchestratorConfig;

  constructor(config: OrchestratorConfig, agent?: CortexSolanaAgent) {
    this.config = config;
    this.agent = agent ?? null;
  }

  setAgent(agent: CortexSolanaAgent): void {
    this.agent = agent;
  }

  async execute(
    recommendation: StrategyRecommendation,
    params: ExecutionParams = {}
  ): Promise<ExecutionResult> {
    const { action, confidence } = recommendation;

    if (confidence < this.config.minConfidence) {
      return {
        success: false,
        action,
        error: `Confidence ${confidence} below threshold ${this.config.minConfidence}`,
        timestamp: new Date(),
      };
    }

    if (action === "hold") {
      return { success: true, action: "hold", timestamp: new Date() };
    }

    if (!this.agent) {
      return {
        success: false,
        action,
        error: "Solana agent not initialized",
        timestamp: new Date(),
      };
    }

    switch (action) {
      case "swap":
        return this.executeSwap(params);
      case "stake":
        return this.executeStake(params);
      case "unstake":
        return this.executeUnstake(params);
      case "rebalance":
        return this.executeRebalance();
      default:
        return {
          success: false,
          action,
          error: `Unknown action: ${action}`,
          timestamp: new Date(),
        };
    }
  }

  private async executeSwap(params: ExecutionParams): Promise<ExecutionResult> {
    const {
      inputMint = KNOWN_TOKENS.SOL.toBase58(),
      outputMint = KNOWN_TOKENS.USDC.toBase58(),
      amountUsd = 10,
      slippageBps = 100,
    } = params;

    if (amountUsd > this.config.maxTradeAmountUsd) {
      return {
        success: false,
        action: "swap",
        error: `Amount $${amountUsd} exceeds max trade limit $${this.config.maxTradeAmountUsd}`,
        timestamp: new Date(),
      };
    }

    try {
      const result = await this.agent!.swap({
        inputMint: new PublicKey(inputMint),
        outputMint: new PublicKey(outputMint),
        amountIn: amountUsd / 200, // Convert USD to SOL estimate
        slippageBps,
      });

      return {
        success: true,
        action: "swap",
        signature: result.signature,
        amountIn: result.inputAmount,
        amountOut: result.outputAmount,
        timestamp: new Date(),
      };
    } catch (error) {
      return {
        success: false,
        action: "swap",
        error: error instanceof Error ? error.message : "Swap failed",
        timestamp: new Date(),
      };
    }
  }

  private async executeStake(params: ExecutionParams): Promise<ExecutionResult> {
    const { amountUsd = 10 } = params;
    const solAmount = amountUsd / 200;

    try {
      const signature = await this.agent!.stakeSOL(solAmount);
      return {
        success: true,
        action: "stake",
        signature,
        amountIn: solAmount,
        timestamp: new Date(),
      };
    } catch (error) {
      return {
        success: false,
        action: "stake",
        error: error instanceof Error ? error.message : "Stake failed",
        timestamp: new Date(),
      };
    }
  }

  private async executeUnstake(params: ExecutionParams): Promise<ExecutionResult> {
    const { amountUsd = 10 } = params;
    const solAmount = amountUsd / 200;

    try {
      const result = await this.agent!.swap({
        inputMint: KNOWN_TOKENS.JITOSOL,
        outputMint: KNOWN_TOKENS.SOL,
        amountIn: solAmount,
        slippageBps: 100,
      });

      return {
        success: true,
        action: "unstake",
        signature: result.signature,
        amountOut: result.outputAmount,
        timestamp: new Date(),
      };
    } catch (error) {
      return {
        success: false,
        action: "unstake",
        error: error instanceof Error ? error.message : "Unstake failed",
        timestamp: new Date(),
      };
    }
  }

  private async executeRebalance(): Promise<ExecutionResult> {
    try {
      const signatures = await this.agent!.rebalance([]);
      return {
        success: true,
        action: "rebalance",
        signature: signatures[0],
        timestamp: new Date(),
      };
    } catch (error) {
      return {
        success: false,
        action: "rebalance",
        error: error instanceof Error ? error.message : "Rebalance failed",
        timestamp: new Date(),
      };
    }
  }
}

