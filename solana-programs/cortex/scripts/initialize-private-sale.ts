import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram, SYSVAR_RENT_PUBKEY } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID, getAssociatedTokenAddress } from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

const PROGRAM_ID = new PublicKey("Cr3msfrK46Mx4id7thTGeihCwK2dpaXWioEWNYgETJGU");
const TOKEN_MINT = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");

async function main() {
  const provider = AnchorProvider.env();
  anchor.setProvider(provider);

  console.log("Initializing Cortex Private Sale...");
  console.log("Program ID:", PROGRAM_ID.toString());
  console.log("Token Mint:", TOKEN_MINT.toString());
  console.log("Authority:", provider.wallet.publicKey.toString());

  const [salePda, saleBump] = PublicKey.findProgramAddressSync(
    [Buffer.from("sale")],
    PROGRAM_ID
  );

  const saleVault = await getAssociatedTokenAddress(
    TOKEN_MINT,
    salePda,
    true
  );

  console.log("\nPDAs:");
  console.log("Sale PDA:", salePda.toString());
  console.log("Sale Vault:", saleVault.toString());

  const saleInfo = await provider.connection.getAccountInfo(salePda);
  if (saleInfo) {
    console.log("\n✅ Private sale already initialized!");
    console.log("Sale Account:", salePda.toString());
    console.log("Sale Vault:", saleVault.toString());
    return;
  }

  const idlPath = path.join(__dirname, "../target/idl/cortex_private_sale.json");
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf-8"));
  const program = new Program(idl as anchor.Idl, provider);

  const now = Math.floor(Date.now() / 1000);
  const startTime = new anchor.BN(now + 300); // Start in 5 minutes
  const endTime = new anchor.BN(now + 2592000); // End in 30 days

  console.log("\nSale Configuration:");
  console.log("Start Time:", new Date(startTime.toNumber() * 1000).toISOString());
  console.log("End Time:", new Date(endTime.toNumber() * 1000).toISOString());

  try {
    const tx = await program.methods
      .initializeSale(startTime, endTime)
      .accounts({
        sale: salePda,
        saleVault: saleVault,
        tokenMint: TOKEN_MINT,
        authority: provider.wallet.publicKey,
        systemProgram: SystemProgram.programId,
        tokenProgram: TOKEN_PROGRAM_ID,
        associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        rent: SYSVAR_RENT_PUBKEY,
      })
      .rpc();

    console.log("\n✅ Private sale initialized successfully!");
    console.log("Transaction signature:", tx);
    console.log("\nAccounts created:");
    console.log("Sale:", salePda.toString());
    console.log("Sale Vault:", saleVault.toString());

    console.log("\nNext steps:");
    console.log("1. Transfer tokens to sale vault");
    console.log("2. Add users to whitelist using add-to-whitelist script");
    console.log("3. Users can purchase tokens after start time");

  } catch (error) {
    console.error("\n❌ Error initializing private sale:");
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

