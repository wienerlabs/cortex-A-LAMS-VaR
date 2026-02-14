import { Router, type Request, type Response } from "express";
import { getBlockNumber } from "../services/blockchain.js";

const router = Router();

router.get("/", async (_req: Request, res: Response) => {
  try {
    const blockNumber = await getBlockNumber();
    res.json({
      success: true,
      data: {
        status: "healthy",
        timestamp: new Date().toISOString(),
        blockNumber: blockNumber.toString(),
      },
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      data: {
        status: "unhealthy",
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : "Unknown error",
      },
    });
  }
});

export default router;

