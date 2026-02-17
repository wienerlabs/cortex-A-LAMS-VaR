import type { Action, Handler, Validator, Memory, State, HandlerCallback, IAgentRuntime } from "../types.js";

function parseStakeIntent(text: string): { amount: number } | null {
  const stakePattern = /stake\s+(\d+(?:\.\d+)?)\s*(?:sol)?/i;
  const match = text.match(stakePattern);
  
  if (!match) return null;
  
  return { amount: parseFloat(match[1]) };
}

const validate: Validator = async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
  const text = message.content?.text || "";
  return parseStakeIntent(text) !== null;
};

const handler: Handler = async (
  runtime: IAgentRuntime,
  message: Memory,
  _state?: State,
  _options?: { [key: string]: unknown },
  callback?: HandlerCallback
): Promise<unknown> => {
  const text = message.content?.text || "";
  const intent = parseStakeIntent(text);
  
  if (!intent) {
    await callback?.({
      text: "I couldn't understand the stake request. Try: 'stake 1 SOL'",
      actions: ["STAKE"],
    });
    return false;
  }
  
  try {
    const cortexAgent = runtime.getSetting("cortexAgent");
    if (!cortexAgent) {
      await callback?.({
        text: "Agent not initialized. Please try again later.",
        actions: ["STAKE"],
      });
      return false;
    }
    
    await callback?.({
      text: `Staking ${intent.amount} SOL to JitoSOL... This may take a moment.`,
      actions: ["STAKE"],
    });
    
    return true;
  } catch (error) {
    await callback?.({
      text: `Stake failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      actions: ["STAKE"],
    });
    return false;
  }
};

export const stakeAction: Action = {
  name: "STAKE",
  similes: ["STAKE_SOL", "DEPOSIT_STAKE"],
  description: "Stake SOL to earn yield via JitoSOL liquid staking",
  validate,
  handler,
  examples: [
    [
      { name: "{{user}}", content: { text: "stake 1 SOL" } },
      { name: "{{agent}}", content: { text: "Staking 1 SOL to JitoSOL...", actions: ["STAKE"] } },
    ],
    [
      { name: "{{user}}", content: { text: "stake 5 sol for yield" } },
      { name: "{{agent}}", content: { text: "Staking 5 SOL to JitoSOL...", actions: ["STAKE"] } },
    ],
  ],
};

