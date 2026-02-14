import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

export const recordMetrics = mutation({
  args: {
    agentPublicKey: v.string(),
    date: v.string(),
    trades: v.number(),
    successful: v.boolean(),
    volumeUsd: v.number(),
    feesUsd: v.number(),
    pnlUsd: v.number(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("agentMetrics")
      .withIndex("by_agent_date", (q) =>
        q.eq("agentPublicKey", args.agentPublicKey).eq("date", args.date)
      )
      .first();

    if (existing) {
      await ctx.db.patch(existing._id, {
        totalTrades: existing.totalTrades + args.trades,
        successfulTrades: existing.successfulTrades + (args.successful ? 1 : 0),
        failedTrades: existing.failedTrades + (args.successful ? 0 : 1),
        totalVolumeUsd: existing.totalVolumeUsd + args.volumeUsd,
        totalFeesUsd: existing.totalFeesUsd + args.feesUsd,
        pnlUsd: existing.pnlUsd + args.pnlUsd,
      });
    } else {
      await ctx.db.insert("agentMetrics", {
        agentPublicKey: args.agentPublicKey,
        date: args.date,
        totalTrades: args.trades,
        successfulTrades: args.successful ? 1 : 0,
        failedTrades: args.successful ? 0 : 1,
        totalVolumeUsd: args.volumeUsd,
        totalFeesUsd: args.feesUsd,
        pnlUsd: args.pnlUsd,
      });
    }
  },
});

export const getMetrics = query({
  args: {
    agentPublicKey: v.string(),
    startDate: v.optional(v.string()),
    endDate: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    let metrics = await ctx.db
      .query("agentMetrics")
      .withIndex("by_agent_date", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .collect();

    if (args.startDate) {
      metrics = metrics.filter((m) => m.date >= args.startDate!);
    }
    if (args.endDate) {
      metrics = metrics.filter((m) => m.date <= args.endDate!);
    }

    return metrics.sort((a, b) => a.date.localeCompare(b.date));
  },
});

export const getAggregatedMetrics = query({
  args: { agentPublicKey: v.string() },
  handler: async (ctx, args) => {
    const metrics = await ctx.db
      .query("agentMetrics")
      .withIndex("by_agent_date", (q) => q.eq("agentPublicKey", args.agentPublicKey))
      .collect();

    if (metrics.length === 0) {
      return {
        totalTrades: 0,
        successfulTrades: 0,
        failedTrades: 0,
        successRate: 0,
        totalVolumeUsd: 0,
        totalFeesUsd: 0,
        totalPnlUsd: 0,
        avgTradeVolumeUsd: 0,
      };
    }

    const totals = metrics.reduce(
      (acc, m) => ({
        totalTrades: acc.totalTrades + m.totalTrades,
        successfulTrades: acc.successfulTrades + m.successfulTrades,
        failedTrades: acc.failedTrades + m.failedTrades,
        totalVolumeUsd: acc.totalVolumeUsd + m.totalVolumeUsd,
        totalFeesUsd: acc.totalFeesUsd + m.totalFeesUsd,
        totalPnlUsd: acc.totalPnlUsd + m.pnlUsd,
      }),
      {
        totalTrades: 0,
        successfulTrades: 0,
        failedTrades: 0,
        totalVolumeUsd: 0,
        totalFeesUsd: 0,
        totalPnlUsd: 0,
      }
    );

    return {
      ...totals,
      successRate: totals.totalTrades > 0
        ? (totals.successfulTrades / totals.totalTrades) * 100
        : 0,
      avgTradeVolumeUsd: totals.totalTrades > 0
        ? totals.totalVolumeUsd / totals.totalTrades
        : 0,
    };
  },
});

