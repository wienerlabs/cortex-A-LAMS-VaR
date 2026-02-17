import { CortexSolanaAgent } from "./solana-agent.js";
import { config } from "../config/index.js";
import type { AgentConfig, AgentLimits } from "./types.js";
import { DEFAULT_AGENT_LIMITS } from "./types.js";

let agentInstance: CortexSolanaAgent | null = null;

export function initializeAgent(customLimits?: Partial<AgentLimits>): CortexSolanaAgent {
  const privateKey = process.env.AGENT_WALLET_PRIVATE_KEY;
  
  if (!privateKey) {
    throw new Error("AGENT_WALLET_PRIVATE_KEY environment variable is required");
  }

  const agentConfig: AgentConfig = {
    rpcUrl: process.env.SOLANA_RPC_URL ?? "https://api.mainnet-beta.solana.com",
    openaiApiKey: process.env.OPENAI_API_KEY,
    heliusApiKey: process.env.HELIUS_API_KEY,
  };

  const limits: AgentLimits = {
    ...DEFAULT_AGENT_LIMITS,
    ...customLimits,
  };

  agentInstance = new CortexSolanaAgent(privateKey, agentConfig, limits);
  
  console.log(`🤖 Agent initialized: ${agentInstance.getPublicKey()}`);
  console.log(`📊 Limits: max trade $${limits.maxTradeAmountUsd}, daily $${limits.dailyTradeLimitUsd}`);

  return agentInstance;
}

export function getAgent(): CortexSolanaAgent {
  if (!agentInstance) {
    throw new Error("Agent not initialized. Call initializeAgent() first.");
  }
  return agentInstance;
}

export function isAgentInitialized(): boolean {
  return agentInstance !== null;
}

export { CortexSolanaAgent } from "./solana-agent.js";
export { AgentWalletManager } from "./wallet-manager.js";
export * from "./types.js";

