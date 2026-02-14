import { PublicKey } from "@solana/web3.js";

const JUPITER_API_URL = "https://quote-api.jup.ag/v6";

interface JupiterQuote {
  inputMint: string;
  inAmount: string;
  outputMint: string;
  outAmount: string;
  otherAmountThreshold: string;
  slippageBps: number;
  priceImpactPct: string;
  routePlan: Array<{
    swapInfo: {
      ammKey: string;
      label: string;
      inputMint: string;
      outputMint: string;
    };
    percent: number;
  }>;
}

const KNOWN_MINTS = {
  USDC: new PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
  USDT: new PublicKey("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"),
  SOL: new PublicKey("So11111111111111111111111111111111111111112"),
  BONK: new PublicKey("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"),
};

describe("Jupiter Integration Tests", () => {
  describe("Jupiter API Connectivity", () => {
    it("can fetch quote for SOL to USDC swap", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.SOL.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("amount", "1000000000"); // 1 SOL
      url.searchParams.set("slippageBps", "50");

      const response = await fetch(url.toString());
      expect(response.ok).toBe(true);

      const quote = await response.json() as JupiterQuote;
      expect(quote.inputMint).toBe(KNOWN_MINTS.SOL.toString());
      expect(quote.outputMint).toBe(KNOWN_MINTS.USDC.toString());
      expect(quote.inAmount).toBe("1000000000");
      expect(Number(quote.outAmount)).toBeGreaterThan(0);
      expect(quote.routePlan).toBeDefined();
      expect(quote.routePlan.length).toBeGreaterThan(0);
    });

    it("can fetch quote for USDC to SOL swap", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.SOL.toString());
      url.searchParams.set("amount", "10000000"); // 10 USDC
      url.searchParams.set("slippageBps", "50");

      const response = await fetch(url.toString());
      expect(response.ok).toBe(true);

      const quote = await response.json() as JupiterQuote;
      expect(quote.inputMint).toBe(KNOWN_MINTS.USDC.toString());
      expect(Number(quote.outAmount)).toBeGreaterThan(0);
    });

    it("can fetch quote for USDC to BONK swap", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.BONK.toString());
      url.searchParams.set("amount", "1000000"); // 1 USDC
      url.searchParams.set("slippageBps", "100");

      const response = await fetch(url.toString());
      expect(response.ok).toBe(true);

      const quote = await response.json() as JupiterQuote;
      expect(Number(quote.outAmount)).toBeGreaterThan(0);
      expect(parseFloat(quote.priceImpactPct)).toBeDefined();
    });
  });

  describe("Quote Validation", () => {
    it("handles invalid mint address gracefully", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", "InvalidMintAddress");
      url.searchParams.set("outputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("amount", "1000000000");

      const response = await fetch(url.toString());
      expect(response.ok).toBe(false);
    });

    it("handles zero amount", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.SOL.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("amount", "0");

      const response = await fetch(url.toString());
      expect(response.ok).toBe(false);
    });

    it("respects slippage tolerance", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.SOL.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("amount", "1000000000");
      url.searchParams.set("slippageBps", "100"); // 1% slippage

      const response = await fetch(url.toString());
      const quote = await response.json() as JupiterQuote;

      expect(quote.slippageBps).toBe(100);
      expect(Number(quote.otherAmountThreshold)).toBeLessThan(Number(quote.outAmount));
    });
  });

  describe("Price Impact Analysis", () => {
    it("calculates price impact for large trades", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.SOL.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("amount", "100000000000"); // 100 SOL
      url.searchParams.set("slippageBps", "100");

      const response = await fetch(url.toString());
      if (response.ok) {
        const quote = await response.json() as JupiterQuote;
        const priceImpact = parseFloat(quote.priceImpactPct);
        expect(priceImpact).toBeGreaterThanOrEqual(0);
        console.log(`Price impact for 100 SOL trade: ${priceImpact}%`);
      }
    });
  });

  describe("Route Plan Analysis", () => {
    it("returns valid route plan with swap info", async () => {
      const url = new URL(`${JUPITER_API_URL}/quote`);
      url.searchParams.set("inputMint", KNOWN_MINTS.SOL.toString());
      url.searchParams.set("outputMint", KNOWN_MINTS.USDC.toString());
      url.searchParams.set("amount", "1000000000");

      const response = await fetch(url.toString());
      const quote = await response.json() as JupiterQuote;

      for (const route of quote.routePlan) {
        expect(route.swapInfo).toBeDefined();
        expect(route.swapInfo.ammKey).toBeDefined();
        expect(route.swapInfo.label).toBeDefined();
        expect(route.percent).toBeGreaterThan(0);
      }
    });
  });
});

