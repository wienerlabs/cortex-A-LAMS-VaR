import type { Action, Handler, Validator, Memory, State, HandlerCallback, IAgentRuntime } from "../types.js";

function parseRebalanceIntent(text: string): { allocations: { token: string; percent: number }[] } | null {
  // Match patterns like "rebalance to 50% SOL 50% USDC"
  const rebalancePattern = /rebalance\s+(?:to\s+)?(.+)/i;
  const match = text.match(rebalancePattern);
  
  if (!match) return null;
  
  const allocationsPart = match[1];
  const allocationPattern = /(\d+)%?\s*(\w+)/g;
  const allocations: { token: string; percent: number }[] = [];
  
  let allocationMatch;
  while ((allocationMatch = allocationPattern.exec(allocationsPart)) !== null) {
    allocations.push({
      percent: parseInt(allocationMatch[1]),
      token: allocationMatch[2].toLowerCase(),
    });
  }
  
  if (allocations.length === 0) return null;
  
  const total = allocations.reduce((sum, a) => sum + a.percent, 0);
  if (Math.abs(total - 100) > 5) return null;
  
  return { allocations };
}

const validate: Validator = async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
  const text = message.content?.text || "";
  return parseRebalanceIntent(text) !== null;
};

const handler: Handler = async (
  runtime: IAgentRuntime,
  message: Memory,
  _state?: State,
  _options?: { [key: string]: unknown },
  callback?: HandlerCallback
): Promise<unknown> => {
  const text = message.content?.text || "";
  const intent = parseRebalanceIntent(text);
  
  if (!intent) {
    await callback?.({
      text: "I couldn't understand. Try: 'rebalance to 50% SOL 50% USDC'",
      actions: ["REBALANCE"],
    });
    return false;
  }
  
  try {
    const cortexAgent = runtime.getSetting("cortexAgent");
    if (!cortexAgent) {
      await callback?.({
        text: "Agent not initialized. Please try again later.",
        actions: ["REBALANCE"],
      });
      return false;
    }
    
    const allocationStr = intent.allocations
      .map((a) => `${a.percent}% ${a.token.toUpperCase()}`)
      .join(", ");
    
    await callback?.({
      text: `Rebalancing portfolio to: ${allocationStr}...`,
      actions: ["REBALANCE"],
    });
    
    return true;
  } catch (error) {
    await callback?.({
      text: `Rebalance failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      actions: ["REBALANCE"],
    });
    return false;
  }
};

export const rebalanceAction: Action = {
  name: "REBALANCE",
  similes: ["REBALANCE_PORTFOLIO", "ADJUST_ALLOCATION"],
  description: "Rebalance portfolio to target token allocations",
  validate,
  handler,
  examples: [
    [
      { name: "{{user}}", content: { text: "rebalance to 50% SOL 50% USDC" } },
      { name: "{{agent}}", content: { text: "Rebalancing portfolio to 50% SOL, 50% USDC...", actions: ["REBALANCE"] } },
    ],
    [
      { name: "{{user}}", content: { text: "rebalance 70 sol 30 usdc" } },
      { name: "{{agent}}", content: { text: "Rebalancing portfolio to 70% SOL, 30% USDC...", actions: ["REBALANCE"] } },
    ],
  ],
};

