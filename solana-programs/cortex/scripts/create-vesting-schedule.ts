import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram, SYSVAR_RENT_PUBKEY, Keypair } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID, getAssociatedTokenAddress } from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

const PROGRAM_ID = new PublicKey("5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns");
const TOKEN_MINT = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");

interface VestingConfig {
  beneficiary: PublicKey;
  totalAmount: anchor.BN;
  cliffDuration: anchor.BN;
  vestingDuration: anchor.BN;
  tgeUnlockPercent: number;
  description: string;
}

async function main() {
  const provider = AnchorProvider.env();
  anchor.setProvider(provider);

  console.log("Creating Vesting Schedules...");
  console.log("Program ID:", PROGRAM_ID.toString());
  console.log("Token Mint:", TOKEN_MINT.toString());
  console.log("Authority:", provider.wallet.publicKey.toString());

  const idlPath = path.join(__dirname, "../target/idl/cortex_vesting.json");
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf-8"));
  const program = new Program(idl as anchor.Idl, provider);

  const now = Math.floor(Date.now() / 1000);

  const vestingSchedules: VestingConfig[] = [
    {
      beneficiary: Keypair.generate().publicKey,
      totalAmount: new anchor.BN(20_000_000_000_000_000), // 20M tokens - Team
      cliffDuration: new anchor.BN(7776000), // 90 days
      vestingDuration: new anchor.BN(63072000), // 730 days (2 years)
      tgeUnlockPercent: 0,
      description: "Team Vesting"
    },
    {
      beneficiary: Keypair.generate().publicKey,
      totalAmount: new anchor.BN(5_000_000_000_000_000), // 5M tokens - Advisors
      cliffDuration: new anchor.BN(2592000), // 30 days
      vestingDuration: new anchor.BN(31536000), // 365 days (1 year)
      tgeUnlockPercent: 10,
      description: "Advisors Vesting"
    },
    {
      beneficiary: Keypair.generate().publicKey,
      totalAmount: new anchor.BN(15_000_000_000_000_000), // 15M tokens - Ecosystem
      cliffDuration: new anchor.BN(0), // No cliff
      vestingDuration: new anchor.BN(94608000), // 1095 days (3 years)
      tgeUnlockPercent: 5,
      description: "Ecosystem Vesting"
    }
  ];

  console.log(`\nCreating ${vestingSchedules.length} vesting schedules...\n`);

  for (const config of vestingSchedules) {
    console.log(`\n--- ${config.description} ---`);
    console.log("Beneficiary:", config.beneficiary.toString());
    console.log("Total Amount:", (config.totalAmount.toNumber() / 1_000_000_000).toFixed(2), "tokens");
    console.log("Cliff Duration:", config.cliffDuration.toNumber() / 86400, "days");
    console.log("Vesting Duration:", config.vestingDuration.toNumber() / 86400, "days");
    console.log("TGE Unlock:", config.tgeUnlockPercent, "%");

    const [vestingSchedulePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("vesting"), config.beneficiary.toBuffer()],
      PROGRAM_ID
    );

    const vestingVault = await getAssociatedTokenAddress(
      TOKEN_MINT,
      vestingSchedulePda,
      true
    );

    console.log("Vesting Schedule PDA:", vestingSchedulePda.toString());
    console.log("Vesting Vault:", vestingVault.toString());

    const scheduleInfo = await provider.connection.getAccountInfo(vestingSchedulePda);
    if (scheduleInfo) {
      console.log("⚠️  Vesting schedule already exists, skipping...");
      continue;
    }

    try {
      const startTime = new anchor.BN(now);

      const tx = await program.methods
        .createVestingSchedule(
          config.totalAmount,
          startTime,
          config.cliffDuration,
          config.vestingDuration,
          config.tgeUnlockPercent
        )
        .accounts({
          vestingSchedule: vestingSchedulePda,
          vestingVault: vestingVault,
          tokenMint: TOKEN_MINT,
          beneficiary: config.beneficiary,
          authority: provider.wallet.publicKey,
          systemProgram: SystemProgram.programId,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .rpc();

      console.log("✅ Vesting schedule created!");
      console.log("Transaction signature:", tx);

      await new Promise(resolve => setTimeout(resolve, 1000));

    } catch (error) {
      console.error("❌ Error creating vesting schedule:");
      console.error(error);
    }
  }

  console.log("\n✅ All vesting schedules created!");
  console.log("\nNext steps:");
  console.log("1. Transfer tokens to vesting vaults");
  console.log("2. Beneficiaries can claim tokens according to schedule");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

