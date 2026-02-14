import "dotenv/config";
import { EventIndexer } from "./indexer.js";
import type { Address } from "viem";
import { prisma } from "../lib/prisma.js";

async function main(): Promise<void> {
  console.log("[Indexer CLI] Starting...");

  const vaults = await prisma.vault.findMany({
    select: { address: true },
  });

  if (vaults.length === 0) {
    console.log("[Indexer CLI] No vaults found in database. Add vaults first.");
    process.exit(0);
  }

  const vaultAddresses = vaults.map((v) => v.address as Address);
  console.log(`[Indexer CLI] Found ${vaultAddresses.length} vaults`);

  const startBlock = process.env.INDEXER_START_BLOCK
    ? BigInt(process.env.INDEXER_START_BLOCK)
    : undefined;

  const indexer = new EventIndexer({ vaultAddresses, startBlock });

  process.on("SIGINT", () => {
    console.log("\n[Indexer CLI] Shutting down...");
    indexer.stop();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    indexer.stop();
    process.exit(0);
  });

  await indexer.start();
}

main().catch((err) => {
  console.error("[Indexer CLI] Fatal error:", err);
  process.exit(1);
});

