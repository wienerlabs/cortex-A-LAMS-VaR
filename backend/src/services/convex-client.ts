import { ConvexHttpClient } from "convex/browser";

const CONVEX_URL = process.env.CONVEX_URL ?? "";

class ConvexService {
  private client: ConvexHttpClient | null = null;

  private getClient(): ConvexHttpClient {
    if (!this.client) {
      if (!CONVEX_URL) {
        throw new Error("CONVEX_URL environment variable not set");
      }
      this.client = new ConvexHttpClient(CONVEX_URL);
    }
    return this.client;
  }

  async logActivity(params: {
    agentPublicKey: string;
    userPublicKey: string;
    action: "swap" | "stake" | "unstake" | "rebalance" | "deposit" | "withdraw";
    params: Record<string, unknown>;
  }): Promise<string> {
    try {
      const client = this.getClient();
      const result = await client.mutation("agentActivity:logActivity" as never, params as never);
      return result as string;
    } catch (error) {
      console.error("Failed to log activity to Convex:", error);
      throw error;
    }
  }

  async updateActivityStatus(params: {
    activityId: string;
    status: "success" | "failed";
    result?: Record<string, unknown>;
    error?: string;
    txSignature?: string;
  }): Promise<void> {
    try {
      const client = this.getClient();
      await client.mutation("agentActivity:updateActivityStatus" as never, params as never);
    } catch (error) {
      console.error("Failed to update activity status:", error);
      throw error;
    }
  }

  async getRecentActivity(params: {
    agentPublicKey?: string;
    userPublicKey?: string;
    limit?: number;
  }): Promise<unknown[]> {
    try {
      const client = this.getClient();
      return await client.query("agentActivity:getRecentActivity" as never, params as never);
    } catch (error) {
      console.error("Failed to get activity:", error);
      return [];
    }
  }

  async getAgentConfig(agentPublicKey: string): Promise<unknown | null> {
    try {
      const client = this.getClient();
      return await client.query("agentConfig:getConfig" as never, { agentPublicKey } as never);
    } catch (error) {
      console.error("Failed to get agent config:", error);
      return null;
    }
  }

  async createAgentConfig(params: {
    agentPublicKey: string;
    ownerPublicKey: string;
    maxTradeAmountUsd?: number;
    dailyTradeLimitUsd?: number;
    maxSlippageBps?: number;
    allowedActions?: string[];
    requiresApprovalAboveUsd?: number;
  }): Promise<string> {
    const client = this.getClient();
    return await client.mutation("agentConfig:createConfig" as never, params as never);
  }

  async updateAgentConfig(params: {
    agentPublicKey: string;
    ownerPublicKey: string;
    maxTradeAmountUsd?: number;
    dailyTradeLimitUsd?: number;
    maxSlippageBps?: number;
    allowedActions?: string[];
    requiresApprovalAboveUsd?: number;
    isActive?: boolean;
  }): Promise<void> {
    const client = this.getClient();
    await client.mutation("agentConfig:updateConfig" as never, params as never);
  }

  async createApprovalRequest(params: {
    agentPublicKey: string;
    userPublicKey: string;
    action: string;
    params: Record<string, unknown>;
    estimatedValueUsd: number;
    expiresInMs?: number;
  }): Promise<string> {
    const client = this.getClient();
    return await client.mutation("approvals:createApprovalRequest" as never, params as never);
  }

  async getPendingApprovals(userPublicKey: string): Promise<unknown[]> {
    try {
      const client = this.getClient();
      return await client.query("approvals:getPendingApprovals" as never, { userPublicKey } as never);
    } catch (error) {
      console.error("Failed to get pending approvals:", error);
      return [];
    }
  }

  async approveRequest(approvalId: string, userPublicKey: string): Promise<{ approved: boolean; params: unknown }> {
    const client = this.getClient();
    return await client.mutation("approvals:approveRequest" as never, { approvalId, userPublicKey } as never);
  }

  async rejectRequest(approvalId: string, userPublicKey: string): Promise<void> {
    const client = this.getClient();
    await client.mutation("approvals:rejectRequest" as never, { approvalId, userPublicKey } as never);
  }

  async recordMetrics(params: {
    agentPublicKey: string;
    date: string;
    trades: number;
    successful: boolean;
    volumeUsd: number;
    feesUsd: number;
    pnlUsd: number;
  }): Promise<void> {
    try {
      const client = this.getClient();
      await client.mutation("metrics:recordMetrics" as never, params as never);
    } catch (error) {
      console.error("Failed to record metrics:", error);
    }
  }

  async getAggregatedMetrics(agentPublicKey: string): Promise<unknown> {
    try {
      const client = this.getClient();
      return await client.query("metrics:getAggregatedMetrics" as never, { agentPublicKey } as never);
    } catch (error) {
      console.error("Failed to get metrics:", error);
      return null;
    }
  }
}

export const convexService = new ConvexService();

