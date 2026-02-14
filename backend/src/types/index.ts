import { z } from "zod";

export const addressSchema = z.string().regex(/^0x[a-fA-F0-9]{40}$/);
export const uint256Schema = z.string().regex(/^\d+$/);

export const VaultDataSchema = z.object({
  address: addressSchema,
  name: z.string(),
  symbol: z.string(),
  totalAssets: uint256Schema,
  totalShares: uint256Schema,
  sharePrice: uint256Schema,
  state: z.enum(["Active", "Paused", "Emergency"]),
});

export type VaultData = z.infer<typeof VaultDataSchema>;

export const DepositEventSchema = z.object({
  vaultAddress: addressSchema,
  user: addressSchema,
  assets: uint256Schema,
  shares: uint256Schema,
  txHash: z.string(),
  blockNumber: z.number(),
  timestamp: z.date(),
});

export type DepositEvent = z.infer<typeof DepositEventSchema>;

export const WithdrawEventSchema = z.object({
  vaultAddress: addressSchema,
  user: addressSchema,
  assets: uint256Schema,
  shares: uint256Schema,
  txHash: z.string(),
  blockNumber: z.number(),
  timestamp: z.date(),
});

export type WithdrawEvent = z.infer<typeof WithdrawEventSchema>;

export const RelayRequestSchema = z.object({
  actionType: z.enum(["allocate", "withdraw", "harvest"]),
  vaultAddress: addressSchema,
  strategyAddress: addressSchema,
  amount: uint256Schema.optional(),
});

export type RelayRequestInput = z.infer<typeof RelayRequestSchema>;

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

