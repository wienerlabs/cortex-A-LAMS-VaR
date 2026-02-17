import { Router, Request, Response } from "express";
import { getAgent, isAgentInitialized } from "../agent/index.js";
import { PublicKey } from "@solana/web3.js";
import { KNOWN_TOKENS } from "../agent/types.js";

const router = Router();

router.get("/status", async (_req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.json({
        initialized: false,
        message: "Agent not initialized",
      });
      return;
    }

    const agent = getAgent();
    const walletInfo = await agent.getWalletInfo();
    const limits = agent.getLimits();
    const dailyVolume = agent.getDailyVolumeUsd();

    res.json({
      initialized: true,
      publicKey: agent.getPublicKey(),
      wallet: walletInfo,
      limits,
      dailyVolumeUsd: dailyVolume,
      remainingDailyLimitUsd: limits.dailyTradeLimitUsd - dailyVolume,
    });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.get("/activity", async (_req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.json({ logs: [] });
      return;
    }

    const agent = getAgent();
    const logs = agent.getActivityLog();

    res.json({ logs });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.post("/swap", async (req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const { inputMint, outputMint, amount, slippageBps, reason } = req.body;

    if (!inputMint || !outputMint || !amount) {
      res.status(400).json({ error: "Missing required fields: inputMint, outputMint, amount" });
      return;
    }

    const agent = getAgent();

    const result = await agent.swap({
      inputMint: new PublicKey(inputMint),
      outputMint: new PublicKey(outputMint),
      amountIn: Number(amount),
      slippageBps: slippageBps ? Number(slippageBps) : undefined,
      reason,
    });

    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.post("/stake", async (req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const { amount } = req.body;

    if (!amount || Number(amount) <= 0) {
      res.status(400).json({ error: "Invalid amount" });
      return;
    }

    const agent = getAgent();
    const signature = await agent.stakeSOL(Number(amount));

    res.json({ success: true, signature });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.get("/price/:base/:quote", async (req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const { base, quote } = req.params;
    const baseMint = KNOWN_TOKENS[base.toUpperCase() as keyof typeof KNOWN_TOKENS];
    const quoteMint = KNOWN_TOKENS[quote.toUpperCase() as keyof typeof KNOWN_TOKENS];

    if (!baseMint || !quoteMint) {
      res.status(400).json({ error: "Unknown token symbol" });
      return;
    }

    const agent = getAgent();
    const price = await agent.getJupiterPrice(baseMint, quoteMint);

    res.json({ base, quote, price });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.post("/rebalance", async (req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const { targetAllocations } = req.body;

    if (!targetAllocations || !Array.isArray(targetAllocations)) {
      res.status(400).json({ error: "targetAllocations array required" });
      return;
    }

    const totalPercent = targetAllocations.reduce(
      (sum: number, a: { targetPercent: number }) => sum + a.targetPercent,
      0
    );
    if (Math.abs(totalPercent - 100) > 1) {
      res.status(400).json({ error: "Target allocations must sum to 100%" });
      return;
    }

    const agent = getAgent();
    const signatures = await agent.rebalance(targetAllocations);

    res.json({ success: true, signatures });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.get("/prices", async (req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const mints = (req.query.mints as string)?.split(",") ?? [];
    if (mints.length === 0) {
      const defaultMints = Object.values(KNOWN_TOKENS).map((pk) => pk.toBase58());
      const agent = getAgent();
      const prices = await agent.getMultipleTokenPrices(defaultMints);
      res.json({ prices });
      return;
    }

    const agent = getAgent();
    const prices = await agent.getMultipleTokenPrices(mints);
    res.json({ prices });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.get("/allocations", async (_req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const agent = getAgent();
    const walletInfo = await agent.getWalletInfo();
    const solPrice = await agent.getTokenPriceUsd(KNOWN_TOKENS.SOL.toBase58());

    const allocations = [
      {
        symbol: "SOL",
        mint: KNOWN_TOKENS.SOL.toBase58(),
        balance: walletInfo.solBalance,
        valueUsd: walletInfo.solBalance * solPrice,
        percent: 0,
      },
      ...walletInfo.tokens.map((t) => ({
        symbol: t.symbol,
        mint: t.mint,
        balance: t.balance,
        valueUsd: t.valueUsd,
        percent: 0,
      })),
    ];

    const totalValue = allocations.reduce((sum, a) => sum + a.valueUsd, 0);
    allocations.forEach((a) => {
      a.percent = totalValue > 0 ? (a.valueUsd / totalValue) * 100 : 0;
    });

    res.json({ allocations, totalValueUsd: totalValue });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.put("/limits", async (req: Request, res: Response) => {
  try {
    if (!isAgentInitialized()) {
      res.status(400).json({ error: "Agent not initialized" });
      return;
    }

    const agent = getAgent();
    agent.updateLimits(req.body);

    res.json({ success: true, limits: agent.getLimits() });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.post("/chat", async (req: Request, res: Response) => {
  try {
    const { message } = req.body;

    if (!message || typeof message !== "string") {
      res.status(400).json({ error: "Message is required" });
      return;
    }

    // Parse intent from natural language
    const intent = parseIntent(message);

    if (!intent) {
      res.json({
        response: "I can help you with: swap tokens, stake SOL, check balance, or rebalance portfolio. Try: 'swap 1 SOL to USDC' or 'check my balance'",
        action: null,
      });
      return;
    }

    res.json({
      response: `I understood: ${intent.action}. ${intent.description}`,
      action: intent.action,
      params: intent.params,
      requiresConfirmation: intent.requiresConfirmation,
    });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

interface ParsedIntent {
  action: string;
  description: string;
  params: Record<string, unknown>;
  requiresConfirmation: boolean;
}

function parseIntent(text: string): ParsedIntent | null {
  const lowerText = text.toLowerCase();

  // Swap pattern
  const swapMatch = lowerText.match(/swap\s+(\d+(?:\.\d+)?)\s+(\w+)\s+(?:to|for|into)\s+(\w+)/);
  if (swapMatch) {
    return {
      action: "swap",
      description: `Swap ${swapMatch[1]} ${swapMatch[2].toUpperCase()} to ${swapMatch[3].toUpperCase()}`,
      params: { amount: parseFloat(swapMatch[1]), from: swapMatch[2], to: swapMatch[3] },
      requiresConfirmation: true,
    };
  }

  // Stake pattern
  const stakeMatch = lowerText.match(/stake\s+(\d+(?:\.\d+)?)\s*(?:sol)?/);
  if (stakeMatch) {
    return {
      action: "stake",
      description: `Stake ${stakeMatch[1]} SOL to JitoSOL`,
      params: { amount: parseFloat(stakeMatch[1]) },
      requiresConfirmation: true,
    };
  }

  // Balance pattern
  if (/balance|portfolio|holdings|how much/i.test(lowerText)) {
    return {
      action: "balance",
      description: "Check wallet balance",
      params: {},
      requiresConfirmation: false,
    };
  }

  // Rebalance pattern
  const rebalanceMatch = lowerText.match(/rebalance/);
  if (rebalanceMatch) {
    const allocations: { token: string; percent: number }[] = [];
    const allocationPattern = /(\d+)%?\s*(\w+)/g;
    let match;
    while ((match = allocationPattern.exec(lowerText)) !== null) {
      if (!["to", "and", "rebalance"].includes(match[2].toLowerCase())) {
        allocations.push({ percent: parseInt(match[1]), token: match[2] });
      }
    }
    return {
      action: "rebalance",
      description: `Rebalance portfolio to ${allocations.map((a) => `${a.percent}% ${a.token.toUpperCase()}`).join(", ")}`,
      params: { allocations },
      requiresConfirmation: true,
    };
  }

  return null;
}

export default router;

