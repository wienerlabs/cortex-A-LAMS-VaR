import type { Action, Handler, Validator, Memory, State, HandlerCallback, IAgentRuntime } from "../types.js";
import { KNOWN_TOKENS } from "../../types.js";

const TOKEN_ALIASES: Record<string, string> = {
  sol: KNOWN_TOKENS.SOL.toBase58(),
  solana: KNOWN_TOKENS.SOL.toBase58(),
  usdc: KNOWN_TOKENS.USDC.toBase58(),
  bonk: KNOWN_TOKENS.BONK.toBase58(),
  jup: KNOWN_TOKENS.JUP.toBase58(),
  jupiter: KNOWN_TOKENS.JUP.toBase58(),
  jitosol: KNOWN_TOKENS.JITOSOL.toBase58(),
};

function parseSwapIntent(text: string): { inputToken: string; outputToken: string; amount: number } | null {
  const swapPattern = /swap\s+(\d+(?:\.\d+)?)\s+(\w+)\s+(?:to|for|into)\s+(\w+)/i;
  const match = text.match(swapPattern);
  
  if (!match) return null;
  
  const amount = parseFloat(match[1]);
  const inputToken = match[2].toLowerCase();
  const outputToken = match[3].toLowerCase();
  
  const inputMint = TOKEN_ALIASES[inputToken];
  const outputMint = TOKEN_ALIASES[outputToken];
  
  if (!inputMint || !outputMint) return null;
  
  return { inputToken: inputMint, outputToken: outputMint, amount };
}

const validate: Validator = async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
  const text = message.content?.text || "";
  return parseSwapIntent(text) !== null;
};

const handler: Handler = async (
  runtime: IAgentRuntime,
  message: Memory,
  _state?: State,
  _options?: { [key: string]: unknown },
  callback?: HandlerCallback
): Promise<unknown> => {
  const text = message.content?.text || "";
  const intent = parseSwapIntent(text);
  
  if (!intent) {
    await callback?.({
      text: "I couldn't understand the swap request. Try: 'swap 1 SOL to USDC'",
      actions: ["SWAP"],
    });
    return false;
  }
  
  try {
    const cortexAgent = runtime.getSetting("cortexAgent");
    if (!cortexAgent) {
      await callback?.({
        text: "Agent not initialized. Please try again later.",
        actions: ["SWAP"],
      });
      return false;
    }
    
    await callback?.({
      text: `Swapping ${intent.amount} tokens... This may take a moment.`,
      actions: ["SWAP"],
    });
    
    return true;
  } catch (error) {
    await callback?.({
      text: `Swap failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      actions: ["SWAP"],
    });
    return false;
  }
};

export const swapAction: Action = {
  name: "SWAP",
  similes: ["TRADE", "EXCHANGE", "CONVERT"],
  description: "Swap one token for another using Jupiter aggregator",
  validate,
  handler,
  examples: [
    [
      { name: "{{user}}", content: { text: "swap 1 SOL to USDC" } },
      { name: "{{agent}}", content: { text: "Swapping 1 SOL to USDC...", actions: ["SWAP"] } },
    ],
    [
      { name: "{{user}}", content: { text: "exchange 100 USDC for BONK" } },
      { name: "{{agent}}", content: { text: "Swapping 100 USDC to BONK...", actions: ["SWAP"] } },
    ],
  ],
};

