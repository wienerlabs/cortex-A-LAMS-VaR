import { Router, Request, Response } from "express";
import {
  getOrchestrator,
  type StrategyType,
  type ExecutionParams,
} from "../agent/orchestrator/index.js";

const router = Router();

router.get("/status", async (_req: Request, res: Response) => {
  try {
    const orchestrator = getOrchestrator();
    const status = await orchestrator.getStatus();
    res.json(status);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to get orchestrator status",
    });
  }
});

router.get("/config", (_req: Request, res: Response) => {
  const orchestrator = getOrchestrator();
  res.json(orchestrator.getConfig());
});

router.patch("/config", (req: Request, res: Response) => {
  try {
    const orchestrator = getOrchestrator();
    orchestrator.updateConfig(req.body);
    res.json(orchestrator.getConfig());
  } catch (error) {
    res.status(400).json({
      error: error instanceof Error ? error.message : "Failed to update config",
    });
  }
});

router.post("/evaluate", async (req: Request, res: Response) => {
  try {
    const { strategy, features } = req.body as {
      strategy: StrategyType;
      features: Record<string, number>;
    };

    if (!strategy || !features) {
      return res.status(400).json({ error: "strategy and features are required" });
    }

    const orchestrator = getOrchestrator();
    const decision = await orchestrator.evaluateStrategy(strategy, features);
    res.json(decision);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Evaluation failed",
    });
  }
});

router.get("/recommendations", async (_req: Request, res: Response) => {
  try {
    const orchestrator = getOrchestrator();
    const decision = await orchestrator.getRecommendations();
    res.json(decision);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to get recommendations",
    });
  }
});

router.post("/execute/:decisionId", async (req: Request, res: Response) => {
  try {
    const { decisionId } = req.params;
    const params = req.body as ExecutionParams;

    const orchestrator = getOrchestrator();
    const result = await orchestrator.executeDecision(decisionId, params);
    res.json(result);
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Execution failed",
    });
  }
});

router.get("/decisions", (req: Request, res: Response) => {
  const limit = parseInt(req.query.limit as string) || 50;
  const orchestrator = getOrchestrator();
  res.json(orchestrator.getDecisions(limit));
});

router.post("/auto-trade", async (req: Request, res: Response) => {
  try {
    const { strategy, features, params } = req.body as {
      strategy: StrategyType;
      features: Record<string, number>;
      params?: ExecutionParams;
    };

    const orchestrator = getOrchestrator();

    const decision = await orchestrator.evaluateStrategy(strategy, features);

    if (!decision.recommendation || !decision.approved) {
      return res.json({
        executed: false,
        decision,
        reason: decision.error ?? "No approved recommendation",
      });
    }

    const result = await orchestrator.executeDecision(decision.id, params);
    res.json({
      executed: result.success,
      decision,
      result,
    });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : "Auto-trade failed",
    });
  }
});

export default router;

