import { Router, type Request, type Response } from "express";
import { executeRelayRequest } from "../relay/index.js";
import { RelayRequestSchema } from "../types/index.js";
import type { Address } from "viem";
import { prisma } from "../lib/prisma.js";

const router = Router();

router.post("/execute", async (req: Request, res: Response) => {
  try {
    const parsed = RelayRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ 
        success: false, 
        error: "Invalid request", 
        details: parsed.error.flatten() 
      });
      return;
    }

    const { actionType, vaultAddress, strategyAddress, amount } = parsed.data;

    const vault = await prisma.vault.findUnique({
      where: { address: vaultAddress },
    });

    if (!vault) {
      res.status(404).json({ success: false, error: "Vault not found" });
      return;
    }

    const action = await prisma.agentAction.create({
      data: {
        vaultId: vault.id,
        actionType,
        status: "pending",
        payload: { strategyAddress, amount },
      },
    });

    const result = await executeRelayRequest({
      actionType,
      vaultAddress: vaultAddress as Address,
      strategyAddress: strategyAddress as Address,
      amount: amount ? BigInt(amount) : undefined,
    });

    await prisma.agentAction.update({
      where: { id: action.id },
      data: {
        status: result.success ? "completed" : "failed",
        txHash: result.txHash,
        error: result.error,
        executedAt: new Date(),
      },
    });

    if (result.success) {
      res.json({ success: true, data: { txHash: result.txHash, actionId: action.id } });
    } else {
      res.status(400).json({ success: false, error: result.error });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

router.get("/actions", async (req: Request, res: Response) => {
  try {
    const { vaultId, status, limit = "20" } = req.query;

    const where: Record<string, unknown> = {};
    if (typeof vaultId === "string") where.vaultId = vaultId;
    if (typeof status === "string") where.status = status;

    const actions = await prisma.agentAction.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: parseInt(limit as string, 10),
    });

    res.json({ success: true, data: actions });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

router.get("/actions/:id", async (req: Request, res: Response) => {
  try {
    const action = await prisma.agentAction.findUnique({
      where: { id: req.params.id },
    });

    if (!action) {
      res.status(404).json({ success: false, error: "Action not found" });
      return;
    }

    res.json({ success: true, data: action });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

export default router;

