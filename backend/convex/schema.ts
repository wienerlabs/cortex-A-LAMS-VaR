import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  agentActivity: defineTable({
    agentPublicKey: v.string(),
    userPublicKey: v.string(),
    action: v.union(
      v.literal("swap"),
      v.literal("stake"),
      v.literal("unstake"),
      v.literal("rebalance"),
      v.literal("deposit"),
      v.literal("withdraw")
    ),
    status: v.union(
      v.literal("pending"),
      v.literal("success"),
      v.literal("failed")
    ),
    params: v.any(),
    result: v.optional(v.any()),
    error: v.optional(v.string()),
    txSignature: v.optional(v.string()),
    createdAt: v.number(),
    completedAt: v.optional(v.number()),
  })
    .index("by_agent", ["agentPublicKey"])
    .index("by_user", ["userPublicKey"])
    .index("by_status", ["status"])
    .index("by_created", ["createdAt"]),

  agentConfig: defineTable({
    agentPublicKey: v.string(),
    ownerPublicKey: v.string(),
    maxTradeAmountUsd: v.number(),
    dailyTradeLimitUsd: v.number(),
    maxSlippageBps: v.number(),
    allowedActions: v.array(v.string()),
    requiresApprovalAboveUsd: v.number(),
    isActive: v.boolean(),
    createdAt: v.number(),
    updatedAt: v.number(),
  })
    .index("by_agent", ["agentPublicKey"])
    .index("by_owner", ["ownerPublicKey"]),

  agentMetrics: defineTable({
    agentPublicKey: v.string(),
    date: v.string(),
    totalTrades: v.number(),
    successfulTrades: v.number(),
    failedTrades: v.number(),
    totalVolumeUsd: v.number(),
    totalFeesUsd: v.number(),
    pnlUsd: v.number(),
  })
    .index("by_agent_date", ["agentPublicKey", "date"]),

  pendingApprovals: defineTable({
    agentPublicKey: v.string(),
    userPublicKey: v.string(),
    action: v.string(),
    params: v.any(),
    estimatedValueUsd: v.number(),
    expiresAt: v.number(),
    status: v.union(
      v.literal("pending"),
      v.literal("approved"),
      v.literal("rejected"),
      v.literal("expired")
    ),
    createdAt: v.number(),
    resolvedAt: v.optional(v.number()),
  })
    .index("by_user", ["userPublicKey"])
    .index("by_status", ["status"]),
});

