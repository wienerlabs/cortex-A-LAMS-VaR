import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, getAssociatedTokenAddress, createAssociatedTokenAccountInstruction } from "@solana/spl-token";
import { assert } from "chai";

describe("cortex_token", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const programId = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");
  
  let tokenDataPda: PublicKey;
  let tokenDataBump: number;
  let mintPda: PublicKey;
  let mintBump: number;
  let treasuryTokenAccount: PublicKey;

  before(async () => {
    [tokenDataPda, tokenDataBump] = PublicKey.findProgramAddressSync(
      [Buffer.from("token_data")],
      programId
    );

    [mintPda, mintBump] = PublicKey.findProgramAddressSync(
      [Buffer.from("mint")],
      programId
    );

    treasuryTokenAccount = await getAssociatedTokenAddress(
      mintPda,
      provider.wallet.publicKey
    );
  });

  it("Initializes the token program", async () => {
    const tx = await provider.connection.getTransaction(
      await provider.connection.requestAirdrop(provider.wallet.publicKey, 1000000000),
      { commitment: "confirmed" }
    );

    const initTx = await provider.sendAndConfirm(
      new anchor.web3.Transaction().add(
        await createInitializeInstruction(
          programId,
          tokenDataPda,
          mintPda,
          provider.wallet.publicKey
        )
      )
    );

    console.log("Initialize transaction:", initTx);

    const tokenData = await provider.connection.getAccountInfo(tokenDataPda);
    assert.ok(tokenData !== null, "Token data account should exist");

    const mintInfo = await provider.connection.getAccountInfo(mintPda);
    assert.ok(mintInfo !== null, "Mint account should exist");
  });

  it("Mints tokens to treasury", async () => {
    const amountToMint = new anchor.BN(1_000_000_000_000_000); // 1M tokens

    const mintTx = await provider.sendAndConfirm(
      new anchor.web3.Transaction().add(
        await createMintToTreasuryInstruction(
          programId,
          tokenDataPda,
          mintPda,
          treasuryTokenAccount,
          provider.wallet.publicKey,
          amountToMint
        )
      )
    );

    console.log("Mint transaction:", mintTx);

    const treasuryBalance = await provider.connection.getTokenAccountBalance(treasuryTokenAccount);
    assert.equal(
      treasuryBalance.value.amount,
      amountToMint.toString(),
      "Treasury should have minted tokens"
    );
  });

  it("Fails to mint beyond total supply", async () => {
    const excessAmount = new anchor.BN(100_000_000_000_000_000); // 100M tokens (exceeds limit)

    try {
      await provider.sendAndConfirm(
        new anchor.web3.Transaction().add(
          await createMintToTreasuryInstruction(
            programId,
            tokenDataPda,
            mintPda,
            treasuryTokenAccount,
            provider.wallet.publicKey,
            excessAmount
          )
        )
      );
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.include(error.message, "ExceedsTotalSupply", "Should fail with ExceedsTotalSupply error");
    }
  });

  it("Fails when unauthorized user tries to mint", async () => {
    const unauthorizedUser = Keypair.generate();
    const amountToMint = new anchor.BN(1_000_000_000_000);

    await provider.connection.requestAirdrop(unauthorizedUser.publicKey, 1000000000);
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      await provider.sendAndConfirm(
        new anchor.web3.Transaction().add(
          await createMintToTreasuryInstruction(
            programId,
            tokenDataPda,
            mintPda,
            treasuryTokenAccount,
            unauthorizedUser.publicKey,
            amountToMint
          )
        ),
        [unauthorizedUser]
      );
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.ok(error, "Should fail with unauthorized error");
    }
  });
});

async function createInitializeInstruction(
  programId: PublicKey,
  tokenData: PublicKey,
  mint: PublicKey,
  authority: PublicKey
): Promise<anchor.web3.TransactionInstruction> {
  // This is a placeholder - actual instruction creation would use the IDL
  throw new Error("Implement using actual program IDL");
}

async function createMintToTreasuryInstruction(
  programId: PublicKey,
  tokenData: PublicKey,
  mint: PublicKey,
  treasuryTokenAccount: PublicKey,
  authority: PublicKey,
  amount: anchor.BN
): Promise<anchor.web3.TransactionInstruction> {
  // This is a placeholder - actual instruction creation would use the IDL
  throw new Error("Implement using actual program IDL");
}

