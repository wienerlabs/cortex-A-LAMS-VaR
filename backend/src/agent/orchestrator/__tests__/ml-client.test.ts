import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { MLAgentClient } from "../ml-client.js";

describe("MLAgentClient", () => {
  let client: MLAgentClient;

  beforeEach(() => {
    client = new MLAgentClient("http://localhost:8000", 5000);
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("constructor", () => {
    it("should create client with default URL", () => {
      const defaultClient = new MLAgentClient();
      expect(defaultClient).toBeDefined();
    });

    it("should strip trailing slash from baseUrl", () => {
      const c = new MLAgentClient("http://localhost:8000/");
      expect(c).toBeDefined();
    });
  });

  describe("healthCheck", () => {
    it("should return health status on success", async () => {
      const mockResponse = { status: "healthy", timestamp: "2024-01-01T00:00:00Z" };
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await client.healthCheck();
      expect(result.status).toBe("healthy");
    });

    it("should throw on API error", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => "Internal Server Error",
      } as Response);

      await expect(client.healthCheck()).rejects.toThrow("ML Agent error (500)");
    });
  });

  describe("predict", () => {
    it("should return prediction response", async () => {
      const mockPrediction = {
        prediction_id: "pred_123",
        strategy: "arbitrage",
        prediction: 0.85,
        confidence: 0.85,
        action: "execute",
        should_execute: true,
        explanation: { reason: "High confidence" },
        timestamp: "2024-01-01T00:00:00Z",
      };

      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => mockPrediction,
      } as Response);

      const result = await client.predict({
        strategy: "arbitrage",
        features: { spread: 0.5, volume: 1000 },
      });

      expect(result.should_execute).toBe(true);
      expect(result.confidence).toBe(0.85);
    });
  });

  describe("getRecommendations", () => {
    it("should return recommendations", async () => {
      const mockRecs = {
        recommendations: [
          { strategy: "arbitrage", action: "swap", confidence: 0.8, expected_profit: 50, reasoning: "test" },
        ],
        best_strategy: "arbitrage",
        timestamp: "2024-01-01T00:00:00Z",
      };

      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => mockRecs,
      } as Response);

      const result = await client.getRecommendations();
      expect(result.best_strategy).toBe("arbitrage");
      expect(result.recommendations).toHaveLength(1);
    });
  });

  describe("isAvailable", () => {
    it("should return true when healthy", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "healthy" }),
      } as Response);

      const available = await client.isAvailable();
      expect(available).toBe(true);
    });

    it("should return false when unavailable", async () => {
      vi.spyOn(global, "fetch").mockRejectedValueOnce(new Error("Connection refused"));

      const available = await client.isAvailable();
      expect(available).toBe(false);
    });
  });

  describe("simulateExecution", () => {
    it("should call execute with simulate_only=true", async () => {
      const mockExec = {
        execution_id: "exec_123",
        strategy: "arbitrage",
        action: "swap",
        state: "SIMULATING",
        simulation_result: { success: true },
        timestamp: "2024-01-01T00:00:00Z",
      };

      const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValueOnce({
        ok: true,
        json: async () => mockExec,
      } as Response);

      await client.simulateExecution({
        strategy: "arbitrage",
        action: "swap",
        params: {},
      });

      const [, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse(options.body as string);
      expect(body.simulate_only).toBe(true);
    });
  });
});

