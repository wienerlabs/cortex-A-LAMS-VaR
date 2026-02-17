import { Connection, Keypair, PublicKey, clusterApiUrl } from "@solana/web3.js";
import { createMint, getOrCreateAssociatedTokenAccount } from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

const PROGRAM_IDS = {
  cortex: new PublicKey("9qeww9s48ByyxzPuoiwhvYfB9dfG4Q3a1VnJrYcahwo6"),
  staking: new PublicKey("DRYYZQ7Vr6VErWfWYuYjtfgDFT48txe77ijDP9NCVVYA"),
  vault: new PublicKey("7nVZjbjEiJVPzaKBPDWJ5dPVMmcEBduDtiWi9o67nosS"),
  strategy: new PublicKey("WpEYz9eHtu3jhZkLsfgyyFpWR5C7bjswktpso5BWeQS"),
  treasury: new PublicKey("GsMtBFGq3DWFGDqtJ6Fb4LjdB6sBADKL2Lix9xaYV7GS"),
};

async function main() {
  console.log("🚀 Initializing Cortex on Devnet\n");

  const connection = new Connection(clusterApiUrl("devnet"), "confirmed");

  const walletPath = path.join(process.env.HOME!, ".config/solana/id.json");
  const walletKeypair = Keypair.fromSecretKey(
    Uint8Array.from(JSON.parse(fs.readFileSync(walletPath, "utf-8")))
  );

  console.log(`Wallet: ${walletKeypair.publicKey.toBase58()}`);

  const balance = await connection.getBalance(walletKeypair.publicKey);
  console.log(`Balance: ${balance / 1e9} SOL\n`);

  // Create CORTEX token mint
  console.log("Creating CORTEX token mint...");
  const cortexMint = await createMint(
    connection,
    walletKeypair,
    walletKeypair.publicKey,
    walletKeypair.publicKey,
    9
  );
  console.log(`CORTEX Mint: ${cortexMint.toBase58()}\n`);

  // Derive PDAs
  console.log("Deriving PDAs...");
  const [stakingPoolPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("staking_pool")],
    PROGRAM_IDS.staking
  );
  console.log(`Staking Pool PDA: ${stakingPoolPda.toBase58()}`);

  const [treasuryPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("treasury")],
    PROGRAM_IDS.treasury
  );
  console.log(`Treasury PDA: ${treasuryPda.toBase58()}`);

  // Create token accounts owned by authority (wallet)
  console.log("\nCreating token accounts...");
  const stakeVault = await getOrCreateAssociatedTokenAccount(
    connection,
    walletKeypair,
    cortexMint,
    walletKeypair.publicKey
  );
  console.log(`Stake Vault ATA: ${stakeVault.address.toBase58()}`);

  const treasuryVault = await getOrCreateAssociatedTokenAccount(
    connection,
    walletKeypair,
    cortexMint,
    walletKeypair.publicKey
  );
  console.log(`Treasury Vault ATA: ${treasuryVault.address.toBase58()}\n`);

  const deploymentInfo = {
    network: "devnet",
    timestamp: new Date().toISOString(),
    programs: {
      cortex: PROGRAM_IDS.cortex.toBase58(),
      staking: PROGRAM_IDS.staking.toBase58(),
      vault: PROGRAM_IDS.vault.toBase58(),
      strategy: PROGRAM_IDS.strategy.toBase58(),
      treasury: PROGRAM_IDS.treasury.toBase58(),
    },
    tokens: {
      cortexMint: cortexMint.toBase58(),
    },
    pdas: {
      stakingPool: stakingPoolPda.toBase58(),
      treasury: treasuryPda.toBase58(),
    },
    accounts: {
      stakeVault: stakeVault.address.toBase58(),
      treasuryVault: treasuryVault.address.toBase58(),
    },
    authority: walletKeypair.publicKey.toBase58(),
  };

  fs.writeFileSync(
    "deployment-devnet.json",
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("✅ Initialization complete!");
  console.log("Deployment info saved to deployment-devnet.json");
}

main().catch(console.error);

