import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

export const logActivity = mutation({
  args: {
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
    params: v.any(),
  },
  handler: async (ctx, args) => {
    const id = await ctx.db.insert("agentActivity", {
      ...args,
      status: "pending",
      createdAt: Date.now(),
    });
    return id;
  },
});

export const updateActivityStatus = mutation({
  args: {
    activityId: v.id("agentActivity"),
    status: v.union(v.literal("success"), v.literal("failed")),
    result: v.optional(v.any()),
    error: v.optional(v.string()),
    txSignature: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { activityId, ...updates } = args;
    await ctx.db.patch(activityId, {
      ...updates,
      completedAt: Date.now(),
    });
  },
});

export const getRecentActivity = query({
  args: {
    agentPublicKey: v.optional(v.string()),
    userPublicKey: v.optional(v.string()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const limit = args.limit ?? 50;
    
    let q = ctx.db.query("agentActivity").order("desc");
    
    if (args.agentPublicKey) {
      q = ctx.db
        .query("agentActivity")
        .withIndex("by_agent", (q) => q.eq("agentPublicKey", args.agentPublicKey!))
        .order("desc");
    } else if (args.userPublicKey) {
      q = ctx.db
        .query("agentActivity")
        .withIndex("by_user", (q) => q.eq("userPublicKey", args.userPublicKey!))
        .order("desc");
    }
    
    return await q.take(limit);
  },
});

export const getActivityById = query({
  args: { activityId: v.id("agentActivity") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.activityId);
  },
});

export const getDailyStats = query({
  args: {
    agentPublicKey: v.string(),
    startOfDay: v.number(),
  },
  handler: async (ctx, args) => {
    const activities = await ctx.db
      .query("agentActivity")
      .withIndex("by_agent", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .filter((q) => q.gte(q.field("createdAt"), args.startOfDay))
      .collect();

    return {
      totalTrades: activities.length,
      successfulTrades: activities.filter((a) => a.status === "success").length,
      failedTrades: activities.filter((a) => a.status === "failed").length,
      pendingTrades: activities.filter((a) => a.status === "pending").length,
    };
  },
});

