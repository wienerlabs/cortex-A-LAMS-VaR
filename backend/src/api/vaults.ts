import { Router, type Request, type Response } from "express";
import { getVaultData, getStrategyAllocation } from "../services/blockchain.js";
import { addressSchema } from "../types/index.js";
import type { Address } from "viem";
import { prisma } from "../lib/prisma.js";

const router = Router();

router.get("/", async (_req: Request, res: Response) => {
  try {
    const vaults = await prisma.vault.findMany({
      select: {
        id: true,
        address: true,
        name: true,
        symbol: true,
        totalAssets: true,
        sharePrice: true,
        state: true,
      },
    });
    res.json({ success: true, data: vaults });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

router.get("/:address", async (req: Request, res: Response) => {
  try {
    const parsed = addressSchema.safeParse(req.params.address);
    if (!parsed.success) {
      res.status(400).json({ success: false, error: "Invalid address" });
      return;
    }

    const vault = await prisma.vault.findUnique({
      where: { address: parsed.data },
      include: {
        allocations: {
          include: { strategy: true },
        },
      },
    });

    if (!vault) {
      res.status(404).json({ success: false, error: "Vault not found" });
      return;
    }

    const onChainData = await getVaultData(parsed.data as Address);

    res.json({
      success: true,
      data: {
        ...vault,
        onChain: onChainData,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

router.get("/:address/performance", async (req: Request, res: Response) => {
  try {
    const parsed = addressSchema.safeParse(req.params.address);
    if (!parsed.success) {
      res.status(400).json({ success: false, error: "Invalid address" });
      return;
    }

    const vault = await prisma.vault.findUnique({
      where: { address: parsed.data },
    });

    if (!vault) {
      res.status(404).json({ success: false, error: "Vault not found" });
      return;
    }

    const snapshots = await prisma.vaultSnapshot.findMany({
      where: { vaultId: vault.id },
      orderBy: { timestamp: "desc" },
      take: 30,
    });

    const harvests = await prisma.harvest.findMany({
      where: { vaultId: vault.id },
      orderBy: { timestamp: "desc" },
      take: 10,
    });

    res.json({
      success: true,
      data: {
        snapshots,
        harvests,
        totalHarvested: harvests.reduce((sum: bigint, h) => sum + BigInt(h.profit), 0n).toString(),
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

router.get("/:address/deposits", async (req: Request, res: Response) => {
  try {
    const parsed = addressSchema.safeParse(req.params.address);
    if (!parsed.success) {
      res.status(400).json({ success: false, error: "Invalid address" });
      return;
    }

    const vault = await prisma.vault.findUnique({
      where: { address: parsed.data },
    });

    if (!vault) {
      res.status(404).json({ success: false, error: "Vault not found" });
      return;
    }

    const deposits = await prisma.deposit.findMany({
      where: { vaultId: vault.id },
      orderBy: { timestamp: "desc" },
      take: 50,
    });

    res.json({ success: true, data: deposits });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

router.get("/user/:userAddress/history", async (req: Request, res: Response) => {
  try {
    const parsed = addressSchema.safeParse(req.params.userAddress);
    if (!parsed.success) {
      res.status(400).json({ success: false, error: "Invalid address" });
      return;
    }

    const userAddress = parsed.data.toLowerCase();

    const [deposits, withdrawals] = await Promise.all([
      prisma.deposit.findMany({
        where: { user: userAddress },
        orderBy: { timestamp: "desc" },
        take: 50,
        include: { vault: { select: { address: true, name: true, symbol: true } } },
      }),
      prisma.withdrawal.findMany({
        where: { user: userAddress },
        orderBy: { timestamp: "desc" },
        take: 50,
        include: { vault: { select: { address: true, name: true, symbol: true } } },
      }),
    ]);

    const transactions = [
      ...deposits.map((d) => ({ ...d, type: "deposit" as const })),
      ...withdrawals.map((w) => ({ ...w, type: "withdraw" as const })),
    ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    const totalDeposited = deposits.reduce((sum, d) => sum + BigInt(d.assets), 0n);
    const totalWithdrawn = withdrawals.reduce((sum, w) => sum + BigInt(w.assets), 0n);

    res.json({
      success: true,
      data: {
        transactions,
        summary: {
          totalDeposited: totalDeposited.toString(),
          totalWithdrawn: totalWithdrawn.toString(),
          netDeposited: (totalDeposited - totalWithdrawn).toString(),
          depositCount: deposits.length,
          withdrawCount: withdrawals.length,
        },
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({ success: false, error: message });
  }
});

export default router;

