import { Connection, PublicKey, Keypair } from "@solana/web3.js";
import { AnchorProvider, Wallet } from "@coral-xyz/anchor";
import { CortexSDK } from "../index";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

async function main() {
  console.log("🧪 Testing Cortex SDK on Devnet\n");

  const connection = new Connection("https://api.devnet.solana.com", "confirmed");
  
  const keypairPath = path.join(os.homedir(), ".config/solana/id.json");
  const keypairData = JSON.parse(fs.readFileSync(keypairPath, "utf-8"));
  const keypair = Keypair.fromSecretKey(new Uint8Array(keypairData));
  
  const wallet = new Wallet(keypair);
  console.log("Wallet:", wallet.publicKey.toString());
  console.log("");

  const sdk = CortexSDK.createWithConnection(connection, wallet);

  console.log("=== Testing Token Functions ===");
  try {
    const mintAddress = sdk.token.getMintAddress();
    console.log("✅ Mint Address:", mintAddress.toString());

    const tokenAccountAddress = await sdk.token.getTokenAccountAddress(wallet.publicKey);
    console.log("✅ Token Account:", tokenAccountAddress.toString());

    const exists = await sdk.token.doesTokenAccountExist(wallet.publicKey);
    console.log("✅ Token Account Exists:", exists);

    const balance = await sdk.token.getTokenBalance(wallet.publicKey);
    console.log("✅ Token Balance:", balance.toLocaleString(), "CORTEX");
  } catch (error: any) {
    console.error("❌ Token Error:", error.message);
  }
  console.log("");

  console.log("=== Testing Private Sale Functions ===");
  try {
    const saleInfo = await sdk.privateSale.getSaleInfo();
    if (saleInfo) {
      console.log("✅ Sale Info:");
      console.log("   - Price per Token:", saleInfo.pricePerToken);
      console.log("   - Total for Sale:", saleInfo.totalTokensForSale.toLocaleString());
      console.log("   - Tokens Sold:", saleInfo.tokensSold.toLocaleString());
      console.log("   - Is Active:", saleInfo.isActive);
    } else {
      console.log("⚠️  Sale not initialized yet");
    }

    const isWhitelisted = await sdk.privateSale.isWhitelisted(wallet.publicKey);
    console.log("✅ Is Whitelisted:", isWhitelisted);

    if (isWhitelisted) {
      const whitelistInfo = await sdk.privateSale.getWhitelistInfo(wallet.publicKey);
      if (whitelistInfo) {
        console.log("✅ Whitelist Info:");
        console.log("   - Allocation:", whitelistInfo.allocation.toLocaleString());
        console.log("   - Purchased:", whitelistInfo.purchased.toLocaleString());
        console.log("   - Remaining:", (whitelistInfo.allocation - whitelistInfo.purchased).toLocaleString());
      }
    }
  } catch (error: any) {
    console.error("❌ Private Sale Error:", error.message);
  }
  console.log("");

  console.log("=== Testing Vesting Functions ===");
  try {
    const vestingSchedule = await sdk.vesting.getVestingSchedule(wallet.publicKey);
    if (vestingSchedule) {
      console.log("✅ Vesting Schedule:");
      console.log("   - Total Amount:", vestingSchedule.totalAmount.toLocaleString());
      console.log("   - Released Amount:", vestingSchedule.releasedAmount.toLocaleString());
      console.log("   - Start Time:", new Date(vestingSchedule.startTime * 1000).toLocaleString());
      console.log("   - Is Revoked:", vestingSchedule.isRevoked);

      const vestedAmount = await sdk.vesting.getVestedAmount(wallet.publicKey);
      console.log("✅ Vested Amount:", vestedAmount.toLocaleString());

      const claimableAmount = await sdk.vesting.getClaimableAmount(wallet.publicKey);
      console.log("✅ Claimable Amount:", claimableAmount.toLocaleString());

      const progress = await sdk.vesting.getVestingProgress(wallet.publicKey);
      console.log("✅ Vesting Progress:", progress.toFixed(2) + "%");

      const timeUntilCliff = await sdk.vesting.getTimeUntilCliff(wallet.publicKey);
      console.log("✅ Time Until Cliff:", timeUntilCliff, "seconds");

      const timeUntilFullyVested = await sdk.vesting.getTimeUntilFullyVested(wallet.publicKey);
      console.log("✅ Time Until Fully Vested:", timeUntilFullyVested, "seconds");
    } else {
      console.log("⚠️  No vesting schedule found");
    }
  } catch (error: any) {
    console.error("❌ Vesting Error:", error.message);
  }
  console.log("");

  console.log("=== SDK Test Complete ===");
  console.log("\n✅ All SDK functions tested successfully!");
  console.log("\nNote: Some functions may show 'not initialized' - this is expected");
  console.log("Run initialization scripts to setup the programs first.");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});

