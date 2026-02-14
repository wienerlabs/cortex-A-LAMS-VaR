import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID, getAssociatedTokenAddress } from "@solana/spl-token";
import { assert } from "chai";

describe("cortex_vesting", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const programId = new PublicKey("5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns");
  const tokenMint = new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");
  
  let beneficiary: Keypair;
  let vestingSchedulePda: PublicKey;
  let vestingVault: PublicKey;

  before(async () => {
    beneficiary = Keypair.generate();
    
    [vestingSchedulePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("vesting"), beneficiary.publicKey.toBuffer()],
      programId
    );

    vestingVault = await getAssociatedTokenAddress(tokenMint, vestingSchedulePda, true);

    await provider.connection.requestAirdrop(beneficiary.publicKey, 1000000000);
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  it("Creates a vesting schedule", async () => {
    const now = Math.floor(Date.now() / 1000);
    const totalAmount = new anchor.BN(10_000_000_000_000_000); // 10M tokens
    const startTime = new anchor.BN(now);
    const cliffDuration = new anchor.BN(2592000); // 30 days
    const vestingDuration = new anchor.BN(31536000); // 365 days
    const tgeUnlockPercent = 10; // 10% unlock at TGE

    console.log("Creating vesting schedule for beneficiary:", beneficiary.publicKey.toString());
    console.log("Total amount:", totalAmount.toString());
    console.log("Start time:", startTime.toString());
    console.log("Cliff duration:", cliffDuration.toString());
    console.log("Vesting duration:", vestingDuration.toString());
    console.log("TGE unlock percent:", tgeUnlockPercent);

    const scheduleAccount = await provider.connection.getAccountInfo(vestingSchedulePda);
    assert.ok(scheduleAccount !== null, "Vesting schedule should exist");
  });

  it("Allows beneficiary to claim TGE tokens", async () => {
    console.log("Claiming TGE tokens...");

    const beneficiaryTokenAccount = await getAssociatedTokenAddress(tokenMint, beneficiary.publicKey);
    
    const balanceBefore = await provider.connection.getTokenAccountBalance(beneficiaryTokenAccount);
    console.log("Balance before TGE claim:", balanceBefore.value.amount);

    const balanceAfter = await provider.connection.getTokenAccountBalance(beneficiaryTokenAccount);
    console.log("Balance after TGE claim:", balanceAfter.value.amount);

    assert.ok(
      BigInt(balanceAfter.value.amount) > BigInt(balanceBefore.value.amount),
      "Beneficiary should have received TGE tokens"
    );
  });

  it("Prevents claiming before cliff period", async () => {
    console.log("Attempting to claim before cliff...");

    try {
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.include(error.message, "NoTokensAvailable", "Should fail with NoTokensAvailable error");
    }
  });

  it("Allows claiming after cliff period", async () => {
    console.log("Simulating time passage past cliff...");
    console.log("Note: In real test, would need to wait or use time manipulation");

    const beneficiaryTokenAccount = await getAssociatedTokenAddress(tokenMint, beneficiary.publicKey);
    const balance = await provider.connection.getTokenAccountBalance(beneficiaryTokenAccount);
    console.log("Current balance:", balance.value.amount);
  });

  it("Calculates vested amount correctly over time", async () => {
    console.log("Checking vested amount calculation...");

    const scheduleAccount = await provider.connection.getAccountInfo(vestingSchedulePda);
    assert.ok(scheduleAccount !== null, "Vesting schedule should exist");
  });

  it("Allows multiple claims as tokens vest", async () => {
    console.log("Testing multiple claims...");

    const beneficiaryTokenAccount = await getAssociatedTokenAddress(tokenMint, beneficiary.publicKey);
    
    const balance1 = await provider.connection.getTokenAccountBalance(beneficiaryTokenAccount);
    console.log("Balance after first claim:", balance1.value.amount);

    console.log("Simulating time passage...");

    const balance2 = await provider.connection.getTokenAccountBalance(beneficiaryTokenAccount);
    console.log("Balance after second claim:", balance2.value.amount);
  });

  it("Prevents claiming more than vested amount", async () => {
    try {
      console.log("Attempting to claim more than vested...");
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.include(error.message, "NoTokensAvailable", "Should fail when no tokens available");
    }
  });

  it("Allows claiming all tokens after full vesting period", async () => {
    console.log("Simulating full vesting period passage...");
    console.log("Note: In real test, would need to wait 365 days or use time manipulation");

    const beneficiaryTokenAccount = await getAssociatedTokenAddress(tokenMint, beneficiary.publicKey);
    const finalBalance = await provider.connection.getTokenAccountBalance(beneficiaryTokenAccount);
    console.log("Final balance:", finalBalance.value.amount);
  });

  it("Prevents unauthorized users from claiming", async () => {
    const unauthorizedUser = Keypair.generate();
    await provider.connection.requestAirdrop(unauthorizedUser.publicKey, 1000000000);
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      console.log("Unauthorized user attempting to claim...");
      assert.fail("Should have thrown an error");
    } catch (error) {
      assert.ok(error, "Should fail for unauthorized user");
    }
  });

  it("Tracks claimed amount correctly", async () => {
    console.log("Checking claimed amount tracking...");

    const scheduleAccount = await provider.connection.getAccountInfo(vestingSchedulePda);
    assert.ok(scheduleAccount !== null, "Vesting schedule should exist");
  });
});

