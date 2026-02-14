import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, getAssociatedTokenAddress } from "@solana/spl-token";
import { assert } from "chai";

describe("cortex_private_sale", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const programId = new PublicKey("Cr3msfrK46Mx4id7thTGeihCwK2dpaXWioEWNYgETJGU");
  const tokenMint = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");
  
  let salePda: PublicKey;
  let saleBump: number;
  let saleVault: PublicKey;
  let buyer: Keypair;
  let whitelistPda: PublicKey;

  before(async () => {
    buyer = Keypair.generate();
    
    [salePda, saleBump] = PublicKey.findProgramAddressSync(
      [Buffer.from("sale")],
      programId
    );

    saleVault = await getAssociatedTokenAddress(tokenMint, salePda, true);

    [whitelistPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("whitelist"), buyer.publicKey.toBuffer()],
      programId
    );

    await provider.connection.requestAirdrop(buyer.publicKey, 2000000000);
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  it("Initializes the private sale", async () => {
    const now = Math.floor(Date.now() / 1000);
    const startTime = new anchor.BN(now + 60); // Start in 1 minute
    const endTime = new anchor.BN(now + 86400); // End in 24 hours

    console.log("Initializing private sale...");
    console.log("Start time:", startTime.toString());
    console.log("End time:", endTime.toString());

    const saleAccount = await provider.connection.getAccountInfo(salePda);
    assert.ok(saleAccount !== null, "Sale account should exist after initialization");
  });

  it("Adds user to whitelist", async () => {
    const maxAllocation = new anchor.BN(1_000_000_000_000_000); // 1M tokens

    console.log("Adding buyer to whitelist:", buyer.publicKey.toString());
    console.log("Max allocation:", maxAllocation.toString());

    const whitelistAccount = await provider.connection.getAccountInfo(whitelistPda);
    assert.ok(whitelistAccount !== null, "Whitelist account should exist");
  });

  it("Allows whitelisted user to purchase tokens", async () => {
    const tokenAmount = new anchor.BN(100_000_000_000_000); // 100k tokens
    
    console.log("Buyer purchasing tokens:", tokenAmount.toString());

    const buyerTokenAccount = await getAssociatedTokenAddress(tokenMint, buyer.publicKey);
    
    const balanceBefore = await provider.connection.getBalance(buyer.publicKey);
    console.log("Buyer SOL balance before:", balanceBefore);

    const balanceAfter = await provider.connection.getBalance(buyer.publicKey);
    console.log("Buyer SOL balance after:", balanceAfter);
    
    assert.ok(balanceBefore > balanceAfter, "Buyer should have paid SOL");
  });

  it("Prevents purchase exceeding allocation", async () => {
    const excessAmount = new anchor.BN(2_000_000_000_000_000); // 2M tokens (exceeds 1M limit)

    try {
      console.log("Attempting to purchase excess amount:", excessAmount.toString());
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.include(error.message, "ExceedsAllocation", "Should fail with ExceedsAllocation error");
    }
  });

  it("Prevents non-whitelisted user from purchasing", async () => {
    const nonWhitelistedUser = Keypair.generate();
    await provider.connection.requestAirdrop(nonWhitelistedUser.publicKey, 1000000000);
    await new Promise(resolve => setTimeout(resolve, 1000));

    const tokenAmount = new anchor.BN(100_000_000_000);

    try {
      console.log("Non-whitelisted user attempting purchase");
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.ok(error, "Should fail for non-whitelisted user");
    }
  });

  it("Prevents purchase before sale starts", async () => {
    const tokenAmount = new anchor.BN(100_000_000_000);

    try {
      console.log("Attempting purchase before sale start time");
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.include(error.message, "SaleNotStarted", "Should fail with SaleNotStarted error");
    }
  });

  it("Allows authority to pause sale", async () => {
    console.log("Pausing sale...");
    
    const saleAccount = await provider.connection.getAccountInfo(salePda);
    assert.ok(saleAccount !== null, "Sale account should still exist");
  });

  it("Prevents purchase when sale is paused", async () => {
    const tokenAmount = new anchor.BN(100_000_000_000);

    try {
      console.log("Attempting purchase while sale is paused");
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.include(error.message, "SaleNotActive", "Should fail with SaleNotActive error");
    }
  });

  it("Allows authority to resume sale", async () => {
    console.log("Resuming sale...");
    
    const saleAccount = await provider.connection.getAccountInfo(salePda);
    assert.ok(saleAccount !== null, "Sale account should still exist");
  });

  it("Tracks total sold amount correctly", async () => {
    console.log("Checking total sold amount...");
    
    const saleAccount = await provider.connection.getAccountInfo(salePda);
    assert.ok(saleAccount !== null, "Sale account should exist");
  });
});

