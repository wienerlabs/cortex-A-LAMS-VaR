import express from "express";
import cors from "cors";
import { config } from "./config/index.js";
import healthRouter from "./api/health.js";
import vaultsRouter from "./api/vaults.js";
import relayRouter from "./api/relay.js";
import solanaRouter from "./api/solana.js";
import agentRouter from "./api/agent.js";
import orchestratorRouter from "./routes/orchestrator.js";
import { initializeAgent, isAgentInitialized } from "./agent/index.js";
import { getOrchestrator } from "./agent/orchestrator/index.js";
import { solanaAuth } from "./middleware/solanaAuth.js";

const app = express();

app.use(cors({ origin: config.CORS_ORIGIN }));
app.use(express.json());

app.use("/api/health", healthRouter);
app.use("/api/vaults", vaultsRouter);
app.use("/api/relay", solanaAuth, relayRouter);
app.use("/api/solana", solanaRouter);
app.use("/api/agent", solanaAuth, agentRouter);
app.use("/api/orchestrator", orchestratorRouter);

app.use((_req, res) => {
  res.status(404).json({ success: false, error: "Not found" });
});

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ success: false, error: "Internal server error" });
});

app.listen(config.PORT, async () => {
  console.log(`Cortex Backend running on port ${config.PORT}`);
  console.log(`Environment: ${config.NODE_ENV}`);
  console.log(`Chain ID: ${config.CHAIN_ID}`);

  // Initialize agent if private key is available
  if (process.env.AGENT_WALLET_PRIVATE_KEY) {
    try {
      initializeAgent();
      console.log("🤖 AI Agent initialized successfully");
    } catch (error) {
      console.warn("⚠️ Failed to initialize agent:", error instanceof Error ? error.message : error);
    }
  } else {
    console.log("ℹ️ Agent not initialized (AGENT_WALLET_PRIVATE_KEY not set)");
  }

  // Initialize orchestrator
  try {
    const orchestrator = getOrchestrator();
    await orchestrator.initialize();
    const status = await orchestrator.getStatus();
    console.log("🎯 Orchestrator initialized");
    console.log(`   ML Agent available: ${status.mlAgentAvailable}`);
  } catch (error) {
    console.warn("⚠️ Orchestrator init warning:", error instanceof Error ? error.message : error);
  }
});

export default app;

