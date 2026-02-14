export interface CortexActionContext {
  swap: (params: {
    inputMint: string;
    outputMint: string;
    amountIn: number;
    slippageBps?: number;
  }) => Promise<{ signature: string }>;
  stake: (amount: number) => Promise<string>;
  rebalance: (allocations: { mint: string; targetPercent: number }[]) => Promise<string[]>;
  getBalance: (mint?: string) => Promise<number>;
  getPrice: (mint: string) => Promise<number>;
}

export interface CortexAgentCharacter {
  name: string;
  description: string;
  personality: string[];
  riskTolerance: "low" | "medium" | "high";
  investmentStyle: string;
}

export interface ActionExample {
  name: string;
  content: { text: string; actions?: string[] };
}

export interface Content {
  text?: string;
  actions?: string[];
  [key: string]: unknown;
}

export interface Memory {
  content?: Content;
  [key: string]: unknown;
}

export interface State {
  [key: string]: unknown;
}

export type HandlerCallback = (response: Content) => Promise<unknown>;

export interface IAgentRuntime {
  getSetting: (key: string) => unknown;
  [key: string]: unknown;
}

export type Handler = (
  runtime: IAgentRuntime,
  message: Memory,
  state?: State,
  options?: { [key: string]: unknown },
  callback?: HandlerCallback
) => Promise<unknown>;

export type Validator = (
  runtime: IAgentRuntime,
  message: Memory,
  state?: State
) => Promise<boolean>;

export interface Action {
  name: string;
  similes?: string[];
  description: string;
  examples?: ActionExample[][];
  handler: Handler;
  validate: Validator;
}

export interface Plugin {
  name: string;
  description: string;
  actions: Action[];
  providers: unknown[];
  services: unknown[];
}

