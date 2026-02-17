import type { Action, Handler, Validator, Memory, State, HandlerCallback, IAgentRuntime } from "../types.js";

function parseBalanceIntent(text: string): { token?: string } | null {
  const balancePatterns = [
    /(?:check|show|get|what(?:'s| is)?)\s*(?:my)?\s*(?:wallet)?\s*balance/i,
    /how much\s+(\w+)?\s*(?:do i have)?/i,
    /balance\s*(?:of)?\s*(\w+)?/i,
  ];
  
  for (const pattern of balancePatterns) {
    const match = text.match(pattern);
    if (match) {
      return { token: match[1]?.toLowerCase() };
    }
  }
  
  return null;
}

const validate: Validator = async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
  const text = message.content?.text || "";
  return parseBalanceIntent(text) !== null;
};

const handler: Handler = async (
  runtime: IAgentRuntime,
  message: Memory,
  _state?: State,
  _options?: { [key: string]: unknown },
  callback?: HandlerCallback
): Promise<unknown> => {
  const text = message.content?.text || "";
  const intent = parseBalanceIntent(text);
  
  if (!intent) {
    await callback?.({
      text: "I couldn't understand. Try: 'check my balance' or 'how much SOL do I have'",
      actions: ["CHECK_BALANCE"],
    });
    return false;
  }
  
  try {
    const cortexAgent = runtime.getSetting("cortexAgent");
    if (!cortexAgent) {
      await callback?.({
        text: "Agent not initialized. Please try again later.",
        actions: ["CHECK_BALANCE"],
      });
      return false;
    }
    
    await callback?.({
      text: "Fetching your wallet balance...",
      actions: ["CHECK_BALANCE"],
    });
    
    return true;
  } catch (error) {
    await callback?.({
      text: `Failed to get balance: ${error instanceof Error ? error.message : "Unknown error"}`,
      actions: ["CHECK_BALANCE"],
    });
    return false;
  }
};

export const balanceAction: Action = {
  name: "CHECK_BALANCE",
  similes: ["BALANCE", "WALLET", "PORTFOLIO", "HOLDINGS"],
  description: "Check wallet balance and token holdings",
  validate,
  handler,
  examples: [
    [
      { name: "{{user}}", content: { text: "check my balance" } },
      { name: "{{agent}}", content: { text: "Your wallet contains: 5.2 SOL, 100 USDC...", actions: ["CHECK_BALANCE"] } },
    ],
    [
      { name: "{{user}}", content: { text: "how much SOL do I have?" } },
      { name: "{{agent}}", content: { text: "You have 5.2 SOL in your wallet.", actions: ["CHECK_BALANCE"] } },
    ],
  ],
};

