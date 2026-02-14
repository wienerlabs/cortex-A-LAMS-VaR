import { describe, it, expect, vi, beforeEach } from "vitest";
import { StrategyExecutor } from "../strategy-executor.js";
import { CortexSolanaAgent } from "../../solana-agent.js";
import type { OrchestratorConfig, StrategyRecommendation } from "../types.js";

vi.mock("../../solana-agent.js");

describe("StrategyExecutor", () => {
  let executor: StrategyExecutor;
  const config: OrchestratorConfig = {
    mlAgentUrl: "http://localhost:8000",
    autoExecute: false,
    minConfidence: 0.7,
    maxTradeAmountUsd: 100,
    dailyLimitUsd: 1000,
    strategies: ["arbitrage"],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    executor = new StrategyExecutor(config);
  });

  describe("execute", () => {
    it("should reject low confidence recommendations", async () => {
      const rec: StrategyRecommendation = {
        strategy: "arbitrage",
        action: "swap",
        confidence: 0.5,
        expected_profit: 50,
        reasoning: "test",
      };

      const result = await executor.execute(rec);
      expect(result.success).toBe(false);
      expect(result.error).toContain("below threshold");
    });

    it("should execute hold action successfully", async () => {
      const rec: StrategyRecommendation = {
        strategy: "arbitrage",
        action: "hold",
        confidence: 0.8,
        expected_profit: 0,
        reasoning: "No opportunity",
      };

      const result = await executor.execute(rec);
      expect(result.success).toBe(true);
      expect(result.action).toBe("hold");
    });

    it("should reject swap when agent not initialized", async () => {
      const rec: StrategyRecommendation = {
        strategy: "arbitrage",
        action: "swap",
        confidence: 0.8,
        expected_profit: 50,
        reasoning: "Good opportunity",
      };

      const result = await executor.execute(rec, { amountUsd: 50 });
      expect(result.success).toBe(false);
      expect(result.error).toContain("agent not initialized");
    });

    it("should reject swap exceeding max trade limit", async () => {
      const mockAgent = {
        swap: vi.fn().mockResolvedValue({ signature: "sig", inputAmount: 1, outputAmount: 100, inputMint: "", outputMint: "", priceImpact: 0 }),
        stakeSOL: vi.fn(),
        rebalance: vi.fn(),
      } as unknown as CortexSolanaAgent;

      executor.setAgent(mockAgent);

      const rec: StrategyRecommendation = {
        strategy: "arbitrage",
        action: "swap",
        confidence: 0.9,
        expected_profit: 100,
        reasoning: "test",
      };

      const result = await executor.execute(rec, { amountUsd: 200 });
      expect(result.success).toBe(false);
      expect(result.error).toContain("exceeds max trade limit");
    });

    it("should execute swap when agent is set", async () => {
      const mockAgent = {
        swap: vi.fn().mockResolvedValue({
          signature: "sig123",
          inputAmount: 0.25,
          outputAmount: 50,
          inputMint: "SOL",
          outputMint: "USDC",
          priceImpact: 0.01,
        }),
        stakeSOL: vi.fn(),
        rebalance: vi.fn(),
      } as unknown as CortexSolanaAgent;

      executor.setAgent(mockAgent);

      const rec: StrategyRecommendation = {
        strategy: "arbitrage",
        action: "swap",
        confidence: 0.8,
        expected_profit: 50,
        reasoning: "Good opportunity",
      };

      const result = await executor.execute(rec, { amountUsd: 50 });
      expect(result.success).toBe(true);
      expect(result.signature).toBe("sig123");
    });

    it("should execute stake action", async () => {
      const mockAgent = {
        swap: vi.fn(),
        stakeSOL: vi.fn().mockResolvedValue("stake_sig"),
        rebalance: vi.fn(),
      } as unknown as CortexSolanaAgent;

      executor.setAgent(mockAgent);

      const rec: StrategyRecommendation = {
        strategy: "staking",
        action: "stake",
        confidence: 0.8,
        expected_profit: 10,
        reasoning: "Good APY",
      };

      const result = await executor.execute(rec, { amountUsd: 50 });
      expect(result.success).toBe(true);
      expect(result.action).toBe("stake");
      expect(result.signature).toBe("stake_sig");
    });

    it("should execute unstake action via swap", async () => {
      const mockAgent = {
        swap: vi.fn().mockResolvedValue({
          signature: "unstake_sig",
          inputAmount: 0.25,
          outputAmount: 0.24,
          inputMint: "jitoSOL",
          outputMint: "SOL",
          priceImpact: 0.01,
        }),
        stakeSOL: vi.fn(),
        rebalance: vi.fn(),
      } as unknown as CortexSolanaAgent;

      executor.setAgent(mockAgent);

      const rec: StrategyRecommendation = {
        strategy: "staking",
        action: "unstake",
        confidence: 0.8,
        expected_profit: 0,
        reasoning: "Better opportunity elsewhere",
      };

      const result = await executor.execute(rec, { amountUsd: 50 });
      expect(result.success).toBe(true);
      expect(result.action).toBe("unstake");
    });

    it("should handle unknown actions", async () => {
      const mockAgent = {
        swap: vi.fn(),
        stakeSOL: vi.fn(),
        rebalance: vi.fn(),
      } as unknown as CortexSolanaAgent;

      executor.setAgent(mockAgent);

      const rec: StrategyRecommendation = {
        strategy: "arbitrage",
        action: "unknown" as any,
        confidence: 0.8,
        expected_profit: 0,
        reasoning: "test",
      };

      const result = await executor.execute(rec);
      expect(result.success).toBe(false);
      expect(result.error).toContain("Unknown action");
    });
  });
});

