import { describe, it, expect, vi, beforeEach } from "vitest";
import { Orchestrator } from "../orchestrator.js";
import { MLAgentClient } from "../ml-client.js";

vi.mock("../ml-client.js");

describe("Orchestrator", () => {
  let orchestrator: Orchestrator;

  beforeEach(() => {
    vi.clearAllMocks();
    orchestrator = new Orchestrator({
      mlAgentUrl: "http://localhost:8000",
      autoExecute: false,
      minConfidence: 0.7,
      maxTradeAmountUsd: 100,
      dailyLimitUsd: 1000,
      strategies: ["arbitrage"],
    });
  });

  describe("constructor", () => {
    it("should create orchestrator with default config", () => {
      const orch = new Orchestrator();
      expect(orch.getConfig().minConfidence).toBe(0.7);
    });

    it("should merge custom config", () => {
      const config = orchestrator.getConfig();
      expect(config.maxTradeAmountUsd).toBe(100);
      expect(config.strategies).toContain("arbitrage");
    });
  });

  describe("getStatus", () => {
    it("should return orchestrator status", async () => {
      vi.spyOn(MLAgentClient.prototype, "isAvailable").mockResolvedValue(true);

      const status = await orchestrator.getStatus();
      expect(status.running).toBe(false);
      expect(status.decisionsToday).toBe(0);
    });
  });

  describe("evaluateStrategy", () => {
    it("should create decision from ML prediction", async () => {
      const mockPrediction = {
        prediction_id: "pred_123",
        strategy: "arbitrage" as const,
        prediction: 0.85,
        confidence: 0.85,
        action: "execute_swap",
        should_execute: true,
        explanation: { reason: "High confidence opportunity" },
        timestamp: new Date().toISOString(),
      };

      vi.spyOn(MLAgentClient.prototype, "predict").mockResolvedValue(mockPrediction);

      const decision = await orchestrator.evaluateStrategy("arbitrage", {
        spread: 0.5,
        volume: 1000,
      });

      expect(decision.mlPrediction).toEqual(mockPrediction);
      expect(decision.recommendation).not.toBeNull();
      expect(decision.recommendation?.action).toBe("swap");
    });

    it("should not recommend when confidence is low", async () => {
      const mockPrediction = {
        prediction_id: "pred_123",
        strategy: "arbitrage" as const,
        prediction: 0.4,
        confidence: 0.4,
        action: "hold",
        should_execute: false,
        explanation: { reason: "Low confidence" },
        timestamp: new Date().toISOString(),
      };

      vi.spyOn(MLAgentClient.prototype, "predict").mockResolvedValue(mockPrediction);

      const decision = await orchestrator.evaluateStrategy("arbitrage", { spread: 0.1 });

      expect(decision.recommendation).toBeNull();
      expect(decision.approved).toBe(false);
    });

    it("should handle ML agent errors", async () => {
      vi.spyOn(MLAgentClient.prototype, "predict").mockRejectedValue(new Error("Connection refused"));

      const decision = await orchestrator.evaluateStrategy("arbitrage", { spread: 0.5 });

      expect(decision.error).toBe("Connection refused");
      expect(decision.mlPrediction).toBeNull();
    });
  });

  describe("getRecommendations", () => {
    it("should get and process recommendations", async () => {
      const mockRecs = {
        recommendations: [
          { strategy: "arbitrage" as const, action: "swap" as const, confidence: 0.8, expected_profit: 50, reasoning: "test" },
        ],
        best_strategy: "arbitrage" as const,
        timestamp: new Date().toISOString(),
      };

      vi.spyOn(MLAgentClient.prototype, "getRecommendations").mockResolvedValue(mockRecs);

      const decision = await orchestrator.getRecommendations();
      expect(decision.recommendation?.strategy).toBe("arbitrage");
    });
  });

  describe("updateConfig", () => {
    it("should update orchestrator config", () => {
      orchestrator.updateConfig({ minConfidence: 0.9 });
      expect(orchestrator.getConfig().minConfidence).toBe(0.9);
    });
  });

  describe("getDecisions", () => {
    it("should return empty array initially", () => {
      const decisions = orchestrator.getDecisions();
      expect(decisions).toEqual([]);
    });

    it("should limit returned decisions", async () => {
      vi.spyOn(MLAgentClient.prototype, "predict").mockResolvedValue({
        prediction_id: "pred_123",
        strategy: "arbitrage",
        prediction: 0.5,
        confidence: 0.5,
        action: "hold",
        should_execute: false,
        explanation: {},
        timestamp: new Date().toISOString(),
      });

      for (let i = 0; i < 10; i++) {
        await orchestrator.evaluateStrategy("arbitrage", { spread: 0.1 });
      }

      const decisions = orchestrator.getDecisions(5);
      expect(decisions).toHaveLength(5);
    });
  });
});

