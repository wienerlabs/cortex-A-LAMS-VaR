/**
 * Get devnet USDC and deposit to Drift account
 *
 * Uses Drift SDK's TokenFaucet for devnet USDC
 */
import 'dotenv/config';
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';

async function main() {
  const privateKey = process.env.SOLANA_PRIVATE_KEY;
  if (!privateKey) {
    console.error('SOLANA_PRIVATE_KEY required');
    process.exit(1);
  }

  // Parse keypair
  let keypair: Keypair;
  try {
    const decoded = bs58.decode(privateKey);
    keypair = Keypair.fromSecretKey(decoded);
  } catch {
    try {
      const parsed = JSON.parse(privateKey);
      keypair = Keypair.fromSecretKey(Uint8Array.from(parsed));
    } catch {
      console.error('Invalid private key format');
      process.exit(1);
    }
  }

  const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
  console.log('Wallet:', keypair.publicKey.toString());
  console.log('SOL Balance:', (await connection.getBalance(keypair.publicKey)) / 1e9, 'SOL');

  // Use Drift SDK
  const DriftSDK = await import('@drift-labs/sdk');
  const { DriftClient, Wallet, TokenFaucet, getMarketsAndOraclesForSubscription, BulkAccountLoader, initialize, QUOTE_PRECISION, BN } = DriftSDK;

  const sdkConfig = initialize({ env: 'devnet' });
  const wallet = new Wallet(keypair);

  console.log('\n📡 Using TokenFaucet to get devnet USDC...');
  console.log('USDC Mint:', sdkConfig.USDC_MINT_ADDRESS);

  // Create TokenFaucet
  const faucet = new TokenFaucet(
    connection,
    wallet,
    new PublicKey(sdkConfig.DRIFT_PROGRAM_ID),
    new PublicKey(sdkConfig.USDC_MINT_ADDRESS)
  );

  try {
    // Mint devnet USDC to wallet
    const mintAmount = new BN(10000).mul(QUOTE_PRECISION); // $10,000
    console.log('Minting', mintAmount.toString(), 'USDC (raw)...');

    const txSig = await faucet.mintToUser(keypair.publicKey, mintAmount);
    console.log('✅ Minted devnet USDC! TX:', txSig);
  } catch (e: any) {
    console.log('❌ Faucet mint failed:', e.message);
    console.log('\nAlternative: Use spl-token-faucet or Solana devnet faucet');
  }

  // Now initialize Drift and deposit
  console.log('\n📡 Initializing Drift to deposit collateral...');

  const { perpMarketIndexes, spotMarketIndexes, oracleInfos } = getMarketsAndOraclesForSubscription('devnet');
  const bulkAccountLoader = new BulkAccountLoader(connection, 'confirmed', 1000);

  const driftClient = new DriftClient({
    connection,
    wallet,
    programID: new PublicKey(sdkConfig.DRIFT_PROGRAM_ID),
    accountSubscription: { type: 'polling', accountLoader: bulkAccountLoader },
    perpMarketIndexes,
    spotMarketIndexes,
    oracleInfos,
    env: 'devnet',
  });

  await driftClient.subscribe();
  console.log('✅ Drift client ready');

  // Check current collateral
  const user = driftClient.getUser();
  const userExists = await user.exists();
  console.log('User account exists:', userExists);

  if (userExists) {
    const collateral = user.getFreeCollateral();
    console.log('Current collateral:', collateral.toNumber() / 1e6, 'USDC');
  }

  // Try to deposit
  try {
    const depositAmount = new BN(100).mul(QUOTE_PRECISION); // $100
    console.log('\n📡 Depositing $100 USDC...');

    const tx = await driftClient.deposit(depositAmount, 0, keypair.publicKey);
    console.log('✅ Deposited! TX:', tx);

    // Check new balance
    const newCollateral = user.getFreeCollateral();
    console.log('New collateral:', newCollateral.toNumber() / 1e6, 'USDC');
  } catch (e: any) {
    console.log('❌ Deposit failed:', e.message);
  }

  await driftClient.unsubscribe();
}

main().catch(console.error);

