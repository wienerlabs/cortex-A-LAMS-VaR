import { type Log, decodeEventLog, type Address, type AbiEvent } from "viem";

export const VAULT_EVENTS = {
  Deposit: {
    type: "event",
    name: "Deposit",
    inputs: [
      { indexed: true, name: "sender", type: "address" },
      { indexed: true, name: "owner", type: "address" },
      { indexed: false, name: "assets", type: "uint256" },
      { indexed: false, name: "shares", type: "uint256" },
    ],
  },
  Withdraw: {
    type: "event",
    name: "Withdraw",
    inputs: [
      { indexed: true, name: "sender", type: "address" },
      { indexed: true, name: "receiver", type: "address" },
      { indexed: true, name: "owner", type: "address" },
      { indexed: false, name: "assets", type: "uint256" },
      { indexed: false, name: "shares", type: "uint256" },
    ],
  },
  StrategyHarvested: {
    type: "event",
    name: "StrategyHarvested",
    inputs: [
      { indexed: true, name: "strategy", type: "address" },
      { indexed: false, name: "profit", type: "uint256" },
      { indexed: false, name: "fee", type: "uint256" },
    ],
  },
} as const;

export const VAULT_ABI = Object.values(VAULT_EVENTS) as AbiEvent[];

export interface DecodedDeposit {
  type: "deposit";
  sender: Address;
  owner: Address;
  assets: bigint;
  shares: bigint;
}

export interface DecodedWithdraw {
  type: "withdraw";
  sender: Address;
  receiver: Address;
  owner: Address;
  assets: bigint;
  shares: bigint;
}

export interface DecodedHarvest {
  type: "harvest";
  strategy: Address;
  profit: bigint;
  fee: bigint;
}

export type DecodedEvent = DecodedDeposit | DecodedWithdraw | DecodedHarvest;

export function decodeVaultEvent(log: Log): DecodedEvent | null {
  try {
    const decoded = decodeEventLog({
      abi: VAULT_ABI,
      data: log.data,
      topics: log.topics,
    });

    switch (decoded.eventName) {
      case "Deposit":
        return {
          type: "deposit",
          sender: (decoded.args as { sender: Address }).sender,
          owner: (decoded.args as { owner: Address }).owner,
          assets: (decoded.args as { assets: bigint }).assets,
          shares: (decoded.args as { shares: bigint }).shares,
        };
      case "Withdraw":
        return {
          type: "withdraw",
          sender: (decoded.args as { sender: Address }).sender,
          receiver: (decoded.args as { receiver: Address }).receiver,
          owner: (decoded.args as { owner: Address }).owner,
          assets: (decoded.args as { assets: bigint }).assets,
          shares: (decoded.args as { shares: bigint }).shares,
        };
      case "StrategyHarvested":
        return {
          type: "harvest",
          strategy: (decoded.args as { strategy: Address }).strategy,
          profit: (decoded.args as { profit: bigint }).profit,
          fee: (decoded.args as { fee: bigint }).fee,
        };
      default:
        return null;
    }
  } catch {
    return null;
  }
}

