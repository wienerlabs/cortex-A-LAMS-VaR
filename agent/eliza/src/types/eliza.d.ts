/**
 * Eliza Core Types
 * 
 * Type definitions for @elizaos/core components used in Cortex.
 * These are copied/adapted from @elizaos/core for type safety.
 */

export interface UUID extends String {
  readonly __uuid: unique symbol;
}

export interface Content {
  text?: string;
  action?: string;
  source?: string;
  [key: string]: unknown;
}

export interface Memory {
  id?: UUID;
  entityId?: UUID;
  agentId?: UUID;
  roomId?: UUID;
  content?: Content;
  createdAt?: Date;
  [key: string]: unknown;
}

export interface State {
  values?: Record<string, unknown>;
  data?: Record<string, unknown>;
  text?: string;
  [key: string]: unknown;
}

export interface ActionResult {
  text?: string;
  values?: Record<string, unknown>;
  data?: Record<string, unknown>;
  success: boolean;
  error?: string | Error;
}

export type HandlerCallback = (response: Content, memories?: Memory[]) => Promise<Memory[]>;

export interface ActionExample {
  name?: string;
  user?: string;
  content: Content;
}

export interface IAgentRuntime {
  agentId?: UUID;
  getSetting(key: string): string | undefined;
  getMemory?(id: UUID): Promise<Memory | null>;
  createMemory?(memory: Memory): Promise<UUID>;
  // Cache methods for persistent state
  getCache?<T>(key: string): Promise<T | undefined>;
  setCache?<T>(key: string, value: T): Promise<boolean>;
  deleteCache?(key: string): Promise<boolean>;
  [key: string]: unknown;
}

export interface Action {
  name: string;
  description: string;
  similes?: string[];
  examples?: ActionExample[][];
  validate: (runtime: IAgentRuntime, message: Memory, state?: State) => Promise<boolean>;
  handler: (
    runtime: IAgentRuntime,
    message: Memory,
    state?: State,
    options?: Record<string, unknown>,
    callback?: HandlerCallback,
    responses?: Memory[]
  ) => Promise<ActionResult | void | undefined>;
  [key: string]: unknown;
}

export interface Provider {
  name: string;
  description?: string;
  get: (runtime: IAgentRuntime, message: Memory, state: State) => Promise<{
    text?: string;
    values?: Record<string, unknown>;
    data?: Record<string, unknown>;
  }>;
}

export interface Evaluator {
  name: string;
  description: string;
  validate: (runtime: IAgentRuntime, message: Memory, state?: State) => Promise<boolean>;
  handler: (
    runtime: IAgentRuntime,
    message: Memory,
    state?: State,
    options?: Record<string, unknown>,
    callback?: HandlerCallback
  ) => Promise<ActionResult | void | undefined>;
}

export interface Service {
  name: string;
  [key: string]: unknown;
}

export interface Plugin {
  name: string;
  description?: string;
  actions?: Action[];
  providers?: Provider[];
  evaluators?: Evaluator[];
  services?: Service[];
}

