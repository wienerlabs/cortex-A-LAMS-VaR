import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

export const createApprovalRequest = mutation({
  args: {
    agentPublicKey: v.string(),
    userPublicKey: v.string(),
    action: v.string(),
    params: v.any(),
    estimatedValueUsd: v.number(),
    expiresInMs: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const expiresIn = args.expiresInMs ?? 24 * 60 * 60 * 1000; // 24 hours default
    const now = Date.now();

    return await ctx.db.insert("pendingApprovals", {
      agentPublicKey: args.agentPublicKey,
      userPublicKey: args.userPublicKey,
      action: args.action,
      params: args.params,
      estimatedValueUsd: args.estimatedValueUsd,
      expiresAt: now + expiresIn,
      status: "pending",
      createdAt: now,
    });
  },
});

export const getPendingApprovals = query({
  args: { userPublicKey: v.string() },
  handler: async (ctx, args) => {
    const now = Date.now();
    const approvals = await ctx.db
      .query("pendingApprovals")
      .withIndex("by_user", (q) => q.eq("userPublicKey", args.userPublicKey))
      .filter((q) => q.eq(q.field("status"), "pending"))
      .collect();

    return approvals.filter((a) => a.expiresAt > now);
  },
});

export const approveRequest = mutation({
  args: {
    approvalId: v.id("pendingApprovals"),
    userPublicKey: v.string(),
  },
  handler: async (ctx, args) => {
    const approval = await ctx.db.get(args.approvalId);
    
    if (!approval) {
      throw new Error("Approval request not found");
    }

    if (approval.userPublicKey !== args.userPublicKey) {
      throw new Error("Unauthorized");
    }

    if (approval.status !== "pending") {
      throw new Error("Request already processed");
    }

    if (approval.expiresAt < Date.now()) {
      await ctx.db.patch(args.approvalId, {
        status: "expired",
        resolvedAt: Date.now(),
      });
      throw new Error("Request expired");
    }

    await ctx.db.patch(args.approvalId, {
      status: "approved",
      resolvedAt: Date.now(),
    });

    return { approved: true, params: approval.params };
  },
});

export const rejectRequest = mutation({
  args: {
    approvalId: v.id("pendingApprovals"),
    userPublicKey: v.string(),
  },
  handler: async (ctx, args) => {
    const approval = await ctx.db.get(args.approvalId);
    
    if (!approval) {
      throw new Error("Approval request not found");
    }

    if (approval.userPublicKey !== args.userPublicKey) {
      throw new Error("Unauthorized");
    }

    if (approval.status !== "pending") {
      throw new Error("Request already processed");
    }

    await ctx.db.patch(args.approvalId, {
      status: "rejected",
      resolvedAt: Date.now(),
    });
  },
});

export const expireOldRequests = mutation({
  args: {},
  handler: async (ctx) => {
    const now = Date.now();
    const expired = await ctx.db
      .query("pendingApprovals")
      .withIndex("by_status", (q) => q.eq("status", "pending"))
      .filter((q) => q.lt(q.field("expiresAt"), now))
      .collect();

    for (const approval of expired) {
      await ctx.db.patch(approval._id, {
        status: "expired",
        resolvedAt: now,
      });
    }

    return { expiredCount: expired.length };
  },
});

