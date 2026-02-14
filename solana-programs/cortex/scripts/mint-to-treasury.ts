import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, getAssociatedTokenAddress, createAssociatedTokenAccountInstruction, getAccount } from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

const PROGRAM_ID = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");

async function main() {
  const provider = AnchorProvider.env();
  anchor.setProvider(provider);

  console.log("Minting tokens to treasury...");
  console.log("Program ID:", PROGRAM_ID.toString());
  console.log("Authority:", provider.wallet.publicKey.toString());

  const [tokenDataPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("token_data")],
    PROGRAM_ID
  );

  const [mintPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("mint")],
    PROGRAM_ID
  );

  const treasuryTokenAccount = await getAssociatedTokenAddress(
    mintPda,
    provider.wallet.publicKey
  );

  console.log("\nAccounts:");
  console.log("Token Data PDA:", tokenDataPda.toString());
  console.log("Mint PDA:", mintPda.toString());
  console.log("Treasury Token Account:", treasuryTokenAccount.toString());

  const treasuryAccountInfo = await provider.connection.getAccountInfo(treasuryTokenAccount);
  if (!treasuryAccountInfo) {
    console.log("\nCreating treasury token account...");
    const createAtaIx = createAssociatedTokenAccountInstruction(
      provider.wallet.publicKey,
      treasuryTokenAccount,
      provider.wallet.publicKey,
      mintPda
    );

    const createAtaTx = new anchor.web3.Transaction().add(createAtaIx);
    const createAtaSig = await provider.sendAndConfirm(createAtaTx);
    console.log("Treasury token account created:", createAtaSig);
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  const idlPath = path.join(__dirname, "../target/idl/cortex_token.json");
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf-8"));
  const program = new Program(idl as anchor.Idl, provider);

  const amountToMint = new anchor.BN(10_000_000_000_000_000); // 10M tokens (with 9 decimals)
  console.log("\nMinting amount:", amountToMint.toString(), "raw units");
  console.log("Minting amount:", (amountToMint.toNumber() / 1_000_000_000).toFixed(2), "tokens");

  try {
    const tx = await program.methods
      .mintToTreasury(amountToMint)
      .accounts({
        tokenData: tokenDataPda,
        mint: mintPda,
        treasuryTokenAccount: treasuryTokenAccount,
        authority: provider.wallet.publicKey,
        tokenProgram: TOKEN_PROGRAM_ID,
      })
      .rpc();

    console.log("\n✅ Tokens minted successfully!");
    console.log("Transaction signature:", tx);

    await new Promise(resolve => setTimeout(resolve, 2000));

    const treasuryAccount = await getAccount(provider.connection, treasuryTokenAccount);
    console.log("\nTreasury balance:", treasuryAccount.amount.toString(), "raw units");
    console.log("Treasury balance:", (Number(treasuryAccount.amount) / 1_000_000_000).toFixed(2), "tokens");

  } catch (error) {
    console.error("\n❌ Error minting tokens:");
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

