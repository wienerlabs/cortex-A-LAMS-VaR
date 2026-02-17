import { PublicKey, VersionedTransaction } from "@solana/web3.js";
import { AgentWalletManager } from "./wallet-manager.js";
import { convexService } from "../services/convex-client.js";
import type {
  AgentConfig,
  AgentLimits,
  TradeParams,
  SwapResult,
  AgentActivityLog,
  AgentActionType
} from "./types.js";
import { DEFAULT_AGENT_LIMITS, KNOWN_TOKENS } from "./types.js";

interface JupiterQuote {
  outAmount: string;
  priceImpactPct?: string;
  error?: string;
}

interface JupiterSwapResponse {
  swapTransaction?: string;
  error?: string;
}

interface JupiterPriceResponse {
  data?: Record<string, { price: number }>;
}

export class CortexSolanaAgent {
  private walletManager: AgentWalletManager;
  private limits: AgentLimits;
  private activityLog: AgentActivityLog[] = [];
  private dailyVolumeUsd: number = 0;
  private lastResetDate: string = new Date().toISOString().split("T")[0];
  private ownerPublicKey: string;
  private useConvex: boolean;
  private cachedSolPriceUsd: number = 0;

  constructor(
    privateKeyBase58: string,
    config: AgentConfig,
    limits: AgentLimits = DEFAULT_AGENT_LIMITS,
    ownerPublicKey?: string
  ) {
    this.walletManager = new AgentWalletManager(privateKeyBase58, config.rpcUrl);
    this.limits = limits;
    this.ownerPublicKey = ownerPublicKey ?? "";
    this.useConvex = !!process.env.CONVEX_URL;
  }

  getPublicKey(): string {
    return this.walletManager.publicKeyString;
  }

  async getWalletInfo() {
    return this.walletManager.getWalletInfo();
  }

  getActivityLog(): AgentActivityLog[] {
    return [...this.activityLog];
  }

  private async logToConvex(
    action: "swap" | "stake" | "unstake" | "rebalance",
    params: Record<string, unknown>
  ): Promise<string | null> {
    if (!this.useConvex) return null;
    try {
      return await convexService.logActivity({
        agentPublicKey: this.getPublicKey(),
        userPublicKey: this.ownerPublicKey,
        action,
        params,
      });
    } catch {
      return null;
    }
  }

  private async updateConvexStatus(
    activityId: string | null,
    status: "success" | "failed",
    result?: Record<string, unknown>,
    error?: string,
    txSignature?: string
  ): Promise<void> {
    if (!this.useConvex || !activityId) return;
    try {
      await convexService.updateActivityStatus({
        activityId,
        status,
        result,
        error,
        txSignature,
      });
    } catch {
      // Silent fail for logging
    }
  }

  private async recordConvexMetrics(
    successful: boolean,
    volumeUsd: number,
    feesUsd: number = 0,
    pnlUsd: number = 0
  ): Promise<void> {
    if (!this.useConvex) return;
    try {
      await convexService.recordMetrics({
        agentPublicKey: this.getPublicKey(),
        date: new Date().toISOString().split("T")[0],
        trades: 1,
        successful,
        volumeUsd,
        feesUsd,
        pnlUsd,
      });
    } catch {
      // Silent fail for metrics
    }
  }

  private async getSolPriceUsd(): Promise<number> {
    try {
      const price = await this.getTokenPriceUsd(KNOWN_TOKENS.SOL.toBase58());
      if (price > 0) {
        this.cachedSolPriceUsd = price;
        return price;
      }
    } catch {
      // fall through to cache
    }
    if (this.cachedSolPriceUsd > 0) {
      return this.cachedSolPriceUsd;
    }
    throw new Error("Unable to fetch SOL price and no cached price available");
  }

  private resetDailyLimitIfNeeded(): void {
    const today = new Date().toISOString().split("T")[0];
    if (today !== this.lastResetDate) {
      this.dailyVolumeUsd = 0;
      this.lastResetDate = today;
    }
  }

  private validateAction(action: AgentActionType, amountUsd: number): void {
    this.resetDailyLimitIfNeeded();

    if (!this.limits.allowedActions.includes(action)) {
      throw new Error(`Action "${action}" is not allowed`);
    }

    if (amountUsd > this.limits.maxTradeAmountUsd) {
      throw new Error(
        `Amount $${amountUsd} exceeds max trade limit $${this.limits.maxTradeAmountUsd}`
      );
    }

    if (this.dailyVolumeUsd + amountUsd > this.limits.dailyTradeLimitUsd) {
      throw new Error(
        `Trade would exceed daily limit. Current: $${this.dailyVolumeUsd}, Limit: $${this.limits.dailyTradeLimitUsd}`
      );
    }
  }

  private logActivity(
    action: AgentActionType,
    params: Record<string, unknown>,
    status: "pending" | "success" | "failed",
    result?: Record<string, unknown>,
    error?: string,
    txSignature?: string
  ): AgentActivityLog {
    const log: AgentActivityLog = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      action,
      status,
      params,
      result,
      error,
      txSignature,
    };
    this.activityLog.unshift(log);
    if (this.activityLog.length > 100) this.activityLog.pop();
    return log;
  }

  async swap(params: TradeParams): Promise<SwapResult> {
    const estimatedUsd = params.amountIn * (await this.getSolPriceUsd());
    this.validateAction("swap", estimatedUsd);

    const swapParams = {
      inputMint: params.inputMint.toBase58(),
      outputMint: params.outputMint.toBase58(),
      amountIn: params.amountIn,
      reason: params.reason,
    };

    const logEntry = this.logActivity("swap", swapParams, "pending");
    const convexId = await this.logToConvex("swap", swapParams);

    try {
      const slippage = Math.min(params.slippageBps ?? 50, this.limits.maxSlippageBps);

      const result = await this.executeJupiterSwap(
        params.inputMint,
        params.outputMint,
        params.amountIn,
        slippage
      );

      this.dailyVolumeUsd += estimatedUsd;

      const swapResult: SwapResult = {
        signature: result.signature,
        inputAmount: params.amountIn,
        outputAmount: result.outputAmount,
        inputMint: params.inputMint.toBase58(),
        outputMint: params.outputMint.toBase58(),
        priceImpact: result.priceImpact,
      };

      logEntry.status = "success";
      logEntry.txSignature = result.signature;
      logEntry.result = swapResult as unknown as Record<string, unknown>;

      await this.updateConvexStatus(convexId, "success", swapResult as unknown as Record<string, unknown>, undefined, result.signature);
      await this.recordConvexMetrics(true, estimatedUsd);

      return swapResult;
    } catch (error) {
      logEntry.status = "failed";
      logEntry.error = error instanceof Error ? error.message : String(error);
      await this.updateConvexStatus(convexId, "failed", undefined, logEntry.error);
      await this.recordConvexMetrics(false, estimatedUsd);
      throw error;
    }
  }

  private async executeJupiterSwap(
    inputMint: PublicKey,
    outputMint: PublicKey,
    amount: number,
    slippageBps: number
  ): Promise<{ signature: string; outputAmount: number; priceImpact: number }> {
    const quoteUrl = `https://quote-api.jup.ag/v6/quote?inputMint=${inputMint.toBase58()}&outputMint=${outputMint.toBase58()}&amount=${Math.floor(amount * 1e9)}&slippageBps=${slippageBps}`;
    const quoteRes = await fetch(quoteUrl);
    const quote = (await quoteRes.json()) as JupiterQuote;

    if (!quote || quote.error) {
      throw new Error(quote?.error ?? "Failed to get Jupiter quote");
    }

    const swapRes = await fetch("https://quote-api.jup.ag/v6/swap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quoteResponse: quote,
        userPublicKey: this.walletManager.publicKeyString,
        wrapAndUnwrapSol: true,
      }),
    });
    const swapData = (await swapRes.json()) as JupiterSwapResponse;

    if (!swapData.swapTransaction) {
      throw new Error(swapData.error ?? "Failed to get swap transaction");
    }

    const txBuffer = Buffer.from(swapData.swapTransaction, "base64");
    const tx = VersionedTransaction.deserialize(txBuffer);
    tx.sign([this.walletManager.getKeypair()]);

    const connection = this.walletManager.getConnection();
    const signature = await connection.sendTransaction(tx, { skipPreflight: false });
    await connection.confirmTransaction(signature, "confirmed");

    return {
      signature,
      outputAmount: Number(quote.outAmount) / 1e6,
      priceImpact: Number(quote.priceImpactPct ?? 0),
    };
  }

  async stakeSOL(amount: number): Promise<string> {
    const estimatedUsd = amount * (await this.getSolPriceUsd());
    this.validateAction("stake", estimatedUsd);

    const stakeParams = { amount };
    const logEntry = this.logActivity("stake", stakeParams, "pending");
    const convexId = await this.logToConvex("stake", stakeParams);

    try {
      const result = await this.executeJupiterSwap(
        KNOWN_TOKENS.SOL,
        KNOWN_TOKENS.JITOSOL,
        amount,
        50
      );

      this.dailyVolumeUsd += estimatedUsd;
      logEntry.status = "success";
      logEntry.txSignature = result.signature;

      await this.updateConvexStatus(convexId, "success", { signature: result.signature }, undefined, result.signature);
      await this.recordConvexMetrics(true, estimatedUsd);

      return result.signature;
    } catch (error) {
      logEntry.status = "failed";
      logEntry.error = error instanceof Error ? error.message : String(error);
      await this.updateConvexStatus(convexId, "failed", undefined, logEntry.error);
      await this.recordConvexMetrics(false, estimatedUsd);
      throw error;
    }
  }

  async rebalance(targetAllocations: { mint: string; targetPercent: number }[]): Promise<string[]> {
    const rebalanceParams = { targetAllocations };
    const logEntry = this.logActivity("rebalance", rebalanceParams, "pending");
    const convexId = await this.logToConvex("rebalance", rebalanceParams);

    try {
      const walletInfo = await this.walletManager.getWalletInfo();
      const totalValueUsd = walletInfo.totalValueUsd;

      if (totalValueUsd < 10) {
        throw new Error("Portfolio value too low to rebalance");
      }

      const currentAllocations = await this.getCurrentAllocations();
      const trades: string[] = [];
      let totalVolumeUsd = 0;

      for (const target of targetAllocations) {
        const current = currentAllocations.find((a) => a.mint === target.mint);
        const currentPercent = current?.percent ?? 0;
        const diff = target.targetPercent - currentPercent;

        if (Math.abs(diff) < 2) continue;

        const tradeValueUsd = Math.abs(diff / 100) * totalValueUsd;
        this.validateAction("rebalance", tradeValueUsd);
        totalVolumeUsd += tradeValueUsd;

        if (diff > 0) {
          const result = await this.executeJupiterSwap(
            KNOWN_TOKENS.USDC,
            new PublicKey(target.mint),
            tradeValueUsd,
            this.limits.maxSlippageBps
          );
          trades.push(result.signature);
        } else {
          const tokenBalance = current?.balance ?? 0;
          const sellAmount = tokenBalance * (Math.abs(diff) / currentPercent);
          const result = await this.executeJupiterSwap(
            new PublicKey(target.mint),
            KNOWN_TOKENS.USDC,
            sellAmount,
            this.limits.maxSlippageBps
          );
          trades.push(result.signature);
        }
      }

      logEntry.status = "success";
      logEntry.result = { trades };

      await this.updateConvexStatus(convexId, "success", { trades }, undefined, trades[0]);
      await this.recordConvexMetrics(true, totalVolumeUsd);

      return trades;
    } catch (error) {
      logEntry.status = "failed";
      logEntry.error = error instanceof Error ? error.message : String(error);
      await this.updateConvexStatus(convexId, "failed", undefined, logEntry.error);
      await this.recordConvexMetrics(false, 0);
      throw error;
    }
  }

  private async getCurrentAllocations(): Promise<{ mint: string; percent: number; balance: number; valueUsd: number }[]> {
    const walletInfo = await this.walletManager.getWalletInfo();
    const totalValue = walletInfo.totalValueUsd;

    if (totalValue === 0) return [];

    const solPrice = await this.getTokenPriceUsd(KNOWN_TOKENS.SOL.toBase58());
    const solValueUsd = walletInfo.solBalance * solPrice;

    const allocations = [
      {
        mint: KNOWN_TOKENS.SOL.toBase58(),
        percent: (solValueUsd / totalValue) * 100,
        balance: walletInfo.solBalance,
        valueUsd: solValueUsd,
      },
    ];

    for (const token of walletInfo.tokens) {
      allocations.push({
        mint: token.mint,
        percent: (token.valueUsd / totalValue) * 100,
        balance: token.balance,
        valueUsd: token.valueUsd,
      });
    }

    return allocations;
  }

  async getTokenPriceUsd(mintAddress: string): Promise<number> {
    try {
      // Try Jupiter first
      const jupPrice = await this.getJupiterPrice(
        new PublicKey(mintAddress),
        KNOWN_TOKENS.USDC
      );
      if (jupPrice > 0) return jupPrice;

      // Fallback to CoinGecko for major tokens
      const cgIds: Record<string, string> = {
        [KNOWN_TOKENS.SOL.toBase58()]: "solana",
        [KNOWN_TOKENS.BONK.toBase58()]: "bonk",
        [KNOWN_TOKENS.JUP.toBase58()]: "jupiter-exchange-solana",
        [KNOWN_TOKENS.JITOSOL.toBase58()]: "jito-staked-sol",
      };

      const cgId = cgIds[mintAddress];
      if (!cgId) return 0;

      const res = await fetch(
        `https://api.coingecko.com/api/v3/simple/price?ids=${cgId}&vs_currencies=usd`
      );
      const data = (await res.json()) as Record<string, { usd: number }>;
      return data[cgId]?.usd ?? 0;
    } catch {
      return 0;
    }
  }

  async getJupiterPrice(baseMint: PublicKey, quoteMint: PublicKey): Promise<number> {
    try {
      const url = `https://price.jup.ag/v6/price?ids=${baseMint.toBase58()}&vsToken=${quoteMint.toBase58()}`;
      const res = await fetch(url);
      const data = (await res.json()) as JupiterPriceResponse;
      return data.data?.[baseMint.toBase58()]?.price ?? 0;
    } catch {
      return 0;
    }
  }

  async getMultipleTokenPrices(mints: string[]): Promise<Record<string, number>> {
    const prices: Record<string, number> = {};
    await Promise.all(
      mints.map(async (mint) => {
        prices[mint] = await this.getTokenPriceUsd(mint);
      })
    );
    return prices;
  }

  async getBalance(mint?: PublicKey): Promise<number> {
    if (!mint || mint.equals(KNOWN_TOKENS.SOL)) {
      return this.walletManager.getSolBalance();
    }
    const walletInfo = await this.walletManager.getWalletInfo();
    const token = walletInfo.tokens.find((t) => t.mint === mint.toBase58());
    return token?.balance ?? 0;
  }

  getLimits(): AgentLimits {
    return { ...this.limits };
  }

  updateLimits(newLimits: Partial<AgentLimits>): void {
    this.limits = { ...this.limits, ...newLimits };
  }

  getDailyVolumeUsd(): number {
    this.resetDailyLimitIfNeeded();
    return this.dailyVolumeUsd;
  }
}

