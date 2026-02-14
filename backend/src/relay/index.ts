import type { Address } from "viem";
import { 
  allocateToStrategy, 
  withdrawFromStrategy, 
  harvestStrategy, 
  waitForTransaction 
} from "../services/blockchain.js";

export type ActionType = "allocate" | "withdraw" | "harvest";

export interface RelayRequest {
  actionType: ActionType;
  vaultAddress: Address;
  strategyAddress: Address;
  amount?: bigint;
}

export interface RelayResponse {
  success: boolean;
  txHash?: `0x${string}`;
  error?: string;
}

const actionLimits: Record<ActionType, { maxPerHour: number; cooldownMs: number }> = {
  allocate: { maxPerHour: 10, cooldownMs: 60000 },
  withdraw: { maxPerHour: 10, cooldownMs: 60000 },
  harvest: { maxPerHour: 4, cooldownMs: 900000 },
};

const actionHistory: Map<string, number[]> = new Map();

function checkRateLimit(actionType: ActionType): boolean {
  const now = Date.now();
  const hourAgo = now - 3600000;
  const key = actionType;
  
  const history = actionHistory.get(key) ?? [];
  const recentActions = history.filter((t) => t > hourAgo);
  
  if (recentActions.length >= actionLimits[actionType].maxPerHour) {
    return false;
  }
  
  const lastAction = recentActions[recentActions.length - 1];
  if (lastAction && now - lastAction < actionLimits[actionType].cooldownMs) {
    return false;
  }
  
  recentActions.push(now);
  actionHistory.set(key, recentActions);
  return true;
}

export async function executeRelayRequest(request: RelayRequest): Promise<RelayResponse> {
  if (!checkRateLimit(request.actionType)) {
    return { success: false, error: "Rate limit exceeded" };
  }

  try {
    let txHash: `0x${string}`;

    switch (request.actionType) {
      case "allocate":
        if (!request.amount) {
          return { success: false, error: "Amount required for allocate" };
        }
        txHash = await allocateToStrategy(
          request.vaultAddress,
          request.strategyAddress,
          request.amount
        );
        break;

      case "withdraw":
        if (!request.amount) {
          return { success: false, error: "Amount required for withdraw" };
        }
        txHash = await withdrawFromStrategy(
          request.vaultAddress,
          request.strategyAddress,
          request.amount
        );
        break;

      case "harvest":
        txHash = await harvestStrategy(request.vaultAddress, request.strategyAddress);
        break;

      default:
        return { success: false, error: "Unknown action type" };
    }

    await waitForTransaction(txHash);
    return { success: true, txHash };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return { success: false, error: message };
  }
}

