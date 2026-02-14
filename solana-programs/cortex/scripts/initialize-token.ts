import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram, SYSVAR_RENT_PUBKEY } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, getAssociatedTokenAddress, createAssociatedTokenAccountInstruction } from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

const PROGRAM_ID = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");

async function main() {
  const provider = AnchorProvider.env();
  anchor.setProvider(provider);

  console.log("Initializing Cortex Token Program...");
  console.log("Program ID:", PROGRAM_ID.toString());
  console.log("Authority:", provider.wallet.publicKey.toString());

  const [tokenDataPda, tokenDataBump] = PublicKey.findProgramAddressSync(
    [Buffer.from("token_data")],
    PROGRAM_ID
  );

  const [mintPda, mintBump] = PublicKey.findProgramAddressSync(
    [Buffer.from("mint")],
    PROGRAM_ID
  );

  console.log("\nPDAs:");
  console.log("Token Data PDA:", tokenDataPda.toString());
  console.log("Mint PDA:", mintPda.toString());

  const tokenDataInfo = await provider.connection.getAccountInfo(tokenDataPda);
  if (tokenDataInfo) {
    console.log("\n✅ Token program already initialized!");
    console.log("Token Data Account:", tokenDataPda.toString());
    console.log("Mint Account:", mintPda.toString());
    return;
  }

  const idlPath = path.join(__dirname, "../target/idl/cortex_token.json");
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf-8"));
  const program = new Program(idl as anchor.Idl, provider);

  console.log("\nInitializing token program...");

  try {
    const tx = await program.methods
      .initialize()
      .accounts({
        tokenData: tokenDataPda,
        mint: mintPda,
        authority: provider.wallet.publicKey,
        systemProgram: SystemProgram.programId,
        tokenProgram: TOKEN_PROGRAM_ID,
        rent: SYSVAR_RENT_PUBKEY,
      })
      .rpc();

    console.log("\n✅ Token program initialized successfully!");
    console.log("Transaction signature:", tx);
    console.log("\nAccounts created:");
    console.log("Token Data:", tokenDataPda.toString());
    console.log("Mint:", mintPda.toString());

    await new Promise(resolve => setTimeout(resolve, 2000));

    const mintInfo = await provider.connection.getAccountInfo(mintPda);
    if (mintInfo) {
      console.log("\n✅ Mint account verified on-chain");
    }

    const treasuryTokenAccount = await getAssociatedTokenAddress(
      mintPda,
      provider.wallet.publicKey
    );

    console.log("\nTreasury Token Account:", treasuryTokenAccount.toString());
    console.log("\nNext steps:");
    console.log("1. Create treasury token account if needed");
    console.log("2. Mint tokens to treasury using mint-to-treasury script");
    console.log("3. Distribute tokens to private sale and vesting programs");

  } catch (error) {
    console.error("\n❌ Error initializing token program:");
    console.error(error);
    throw error;
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

