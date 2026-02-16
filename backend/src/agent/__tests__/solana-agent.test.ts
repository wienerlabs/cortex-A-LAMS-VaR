import { describe, it, expect, vi, beforeEach } from "vitest";
import { Keypair } from "@solana/web3.js";
import bs58 from "bs58";
import { CortexSolanaAgent } from "../solana-agent.js";
import { DEFAULT_AGENT_LIMITS, KNOWN_TOKENS } from "../types.js";
import type { AgentConfig } from "../types.js";

// Mock fetch for Jupiter API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Helper: mock a successful SOL price response (Jupiter price API)
function mockSolPriceResponse(price: number = 200) {
  mockFetch.mockResolvedValueOnce({
    json: async () => ({
      data: { [KNOWN_TOKENS.SOL.toBase58()]: { price } },
    }),
  });
}

// Generate a valid test keypair
const testKeypair = Keypair.generate();
const TEST_PRIVATE_KEY = bs58.encode(testKeypair.secretKey);
const TEST_CONFIG: AgentConfig = {
  rpcUrl: "https://api.devnet.solana.com",
};

describe("CortexSolanaAgent", () => {
  let agent: CortexSolanaAgent;

  beforeEach(() => {
    vi.clearAllMocks();
    agent = new CortexSolanaAgent(TEST_PRIVATE_KEY, TEST_CONFIG);
  });

  describe("initialization", () => {
    it("should initialize with default limits", () => {
      const limits = agent.getLimits();
      expect(limits.maxTradeAmountUsd).toBe(DEFAULT_AGENT_LIMITS.maxTradeAmountUsd);
      expect(limits.dailyTradeLimitUsd).toBe(DEFAULT_AGENT_LIMITS.dailyTradeLimitUsd);
    });

    it("should initialize with custom limits", () => {
      const customAgent = new CortexSolanaAgent(TEST_PRIVATE_KEY, TEST_CONFIG, {
        ...DEFAULT_AGENT_LIMITS,
        maxTradeAmountUsd: 500,
      });
      expect(customAgent.getLimits().maxTradeAmountUsd).toBe(500);
    });

    it("should return a valid public key", () => {
      const publicKey = agent.getPublicKey();
      expect(publicKey).toBeDefined();
      expect(publicKey.length).toBeGreaterThan(30);
    });
  });

  describe("limits", () => {
    it("should update limits", () => {
      agent.updateLimits({ maxTradeAmountUsd: 2000 });
      expect(agent.getLimits().maxTradeAmountUsd).toBe(2000);
    });

    it("should track daily volume", () => {
      expect(agent.getDailyVolumeUsd()).toBe(0);
    });
  });

  describe("activity log", () => {
    it("should start with empty activity log", () => {
      expect(agent.getActivityLog()).toHaveLength(0);
    });
  });

  describe("getJupiterPrice", () => {
    it("should fetch price from Jupiter API", async () => {
      mockFetch.mockResolvedValueOnce({
        json: async () => ({
          data: {
            [KNOWN_TOKENS.SOL.toBase58()]: { price: 200 },
          },
        }),
      });

      const price = await agent.getJupiterPrice(KNOWN_TOKENS.SOL, KNOWN_TOKENS.USDC);
      expect(price).toBe(200);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("price.jup.ag")
      );
    });

    it("should return 0 on API error", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));
      const price = await agent.getJupiterPrice(KNOWN_TOKENS.SOL, KNOWN_TOKENS.USDC);
      expect(price).toBe(0);
    });
  });

  describe("getTokenPriceUsd", () => {
    it("should try Jupiter first then fallback to CoinGecko", async () => {
      // Jupiter returns 0
      mockFetch.mockResolvedValueOnce({
        json: async () => ({ data: {} }),
      });
      // CoinGecko returns price
      mockFetch.mockResolvedValueOnce({
        json: async () => ({ solana: { usd: 195 } }),
      });

      const price = await agent.getTokenPriceUsd(KNOWN_TOKENS.SOL.toBase58());
      expect(price).toBe(195);
    });
  });

  describe("getMultipleTokenPrices", () => {
    it("should fetch multiple prices in parallel", async () => {
      mockFetch.mockResolvedValue({
        json: async () => ({
          data: { [KNOWN_TOKENS.SOL.toBase58()]: { price: 200 } },
        }),
      });

      const prices = await agent.getMultipleTokenPrices([
        KNOWN_TOKENS.SOL.toBase58(),
        KNOWN_TOKENS.USDC.toBase58(),
      ]);

      expect(Object.keys(prices)).toHaveLength(2);
    });
  });

  describe("swap validation", () => {
    it("should reject swaps exceeding max trade amount", async () => {
      agent.updateLimits({ maxTradeAmountUsd: 100 });
      mockSolPriceResponse(200);

      await expect(
        agent.swap({
          inputMint: KNOWN_TOKENS.SOL,
          outputMint: KNOWN_TOKENS.USDC,
          amountIn: 10, // 10 SOL = $2000 at $200/SOL
        })
      ).rejects.toThrow("exceeds max trade limit");
    });

    it("should reject disallowed actions", async () => {
      agent.updateLimits({ allowedActions: ["stake"] });
      mockSolPriceResponse(200);

      await expect(
        agent.swap({
          inputMint: KNOWN_TOKENS.SOL,
          outputMint: KNOWN_TOKENS.USDC,
          amountIn: 0.1,
        })
      ).rejects.toThrow('Action "swap" is not allowed');
    });

    it("should reject swaps exceeding daily limit", async () => {
      agent.updateLimits({
        maxTradeAmountUsd: 10000,
        dailyTradeLimitUsd: 100
      });
      mockSolPriceResponse(200);

      await expect(
        agent.swap({
          inputMint: KNOWN_TOKENS.SOL,
          outputMint: KNOWN_TOKENS.USDC,
          amountIn: 1, // 1 SOL = $200, exceeds $100 daily
        })
      ).rejects.toThrow("exceed daily limit");
    });
  });

  describe("rebalance", () => {
    it("should reject rebalance with low portfolio value", async () => {
      // Rebalance checks portfolio value before action validation
      await expect(
        agent.rebalance([
          { mint: KNOWN_TOKENS.SOL.toBase58(), targetPercent: 50 },
          { mint: KNOWN_TOKENS.USDC.toBase58(), targetPercent: 50 },
        ])
      ).rejects.toThrow("Portfolio value too low");
    });
  });

  describe("stake validation", () => {
    it("should reject stake if not allowed", async () => {
      agent.updateLimits({ allowedActions: ["swap"] });
      mockSolPriceResponse(200);

      await expect(agent.stakeSOL(1)).rejects.toThrow(
        'Action "stake" is not allowed'
      );
    });

    it("should reject stake exceeding max trade amount", async () => {
      agent.updateLimits({ maxTradeAmountUsd: 100 });
      mockSolPriceResponse(200);

      await expect(agent.stakeSOL(10)).rejects.toThrow("exceeds max trade limit");
    });
  });
});

