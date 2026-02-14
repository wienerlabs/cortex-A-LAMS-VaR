import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

export const getConfig = query({
  args: { agentPublicKey: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("agentConfig")
      .withIndex("by_agent", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .first();
  },
});

export const getConfigByOwner = query({
  args: { ownerPublicKey: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("agentConfig")
      .withIndex("by_owner", (q) => q.eq("ownerPublicKey", args.ownerPublicKey))
      .collect();
  },
});

export const createConfig = mutation({
  args: {
    agentPublicKey: v.string(),
    ownerPublicKey: v.string(),
    maxTradeAmountUsd: v.optional(v.number()),
    dailyTradeLimitUsd: v.optional(v.number()),
    maxSlippageBps: v.optional(v.number()),
    allowedActions: v.optional(v.array(v.string())),
    requiresApprovalAboveUsd: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("agentConfig")
      .withIndex("by_agent", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .first();

    if (existing) {
      throw new Error("Config already exists for this agent");
    }

    const now = Date.now();
    return await ctx.db.insert("agentConfig", {
      agentPublicKey: args.agentPublicKey,
      ownerPublicKey: args.ownerPublicKey,
      maxTradeAmountUsd: args.maxTradeAmountUsd ?? 100,
      dailyTradeLimitUsd: args.dailyTradeLimitUsd ?? 1000,
      maxSlippageBps: args.maxSlippageBps ?? 100,
      allowedActions: args.allowedActions ?? ["swap", "stake", "rebalance"],
      requiresApprovalAboveUsd: args.requiresApprovalAboveUsd ?? 500,
      isActive: true,
      createdAt: now,
      updatedAt: now,
    });
  },
});

export const updateConfig = mutation({
  args: {
    agentPublicKey: v.string(),
    ownerPublicKey: v.string(),
    maxTradeAmountUsd: v.optional(v.number()),
    dailyTradeLimitUsd: v.optional(v.number()),
    maxSlippageBps: v.optional(v.number()),
    allowedActions: v.optional(v.array(v.string())),
    requiresApprovalAboveUsd: v.optional(v.number()),
    isActive: v.optional(v.boolean()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("agentConfig")
      .withIndex("by_agent", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .first();

    if (!existing) {
      throw new Error("Config not found");
    }

    if (existing.ownerPublicKey !== args.ownerPublicKey) {
      throw new Error("Unauthorized: not the owner");
    }

    const { agentPublicKey, ownerPublicKey, ...updates } = args;
    const filteredUpdates = Object.fromEntries(
      Object.entries(updates).filter(([, v]) => v !== undefined)
    );

    await ctx.db.patch(existing._id, {
      ...filteredUpdates,
      updatedAt: Date.now(),
    });
  },
});

export const deactivateAgent = mutation({
  args: {
    agentPublicKey: v.string(),
    ownerPublicKey: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("agentConfig")
      .withIndex("by_agent", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .first();

    if (!existing) {
      throw new Error("Config not found");
    }

    if (existing.ownerPublicKey !== args.ownerPublicKey) {
      throw new Error("Unauthorized: not the owner");
    }

    await ctx.db.patch(existing._id, {
      isActive: false,
      updatedAt: Date.now(),
    });
  },
});

