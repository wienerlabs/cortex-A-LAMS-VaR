/**
 * Create a Devnet Test Wallet
 * 
 * Creates a new Solana keypair and requests SOL airdrop on devnet
 */

import { Connection, Keypair, LAMPORTS_PER_SOL } from '@solana/web3.js';
import bs58 from 'bs58';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEVNET_RPC = 'https://api.devnet.solana.com';
const WALLET_FILE = join(__dirname, 'devnet-test-wallet.json');

async function main() {
  console.log('\n=== DEVNET WALLET SETUP ===\n');

  let keypair: Keypair;
  
  // Check if wallet already exists
  if (fs.existsSync(WALLET_FILE)) {
    console.log('📁 Loading existing wallet...');
    const secretKey = JSON.parse(fs.readFileSync(WALLET_FILE, 'utf8'));
    keypair = Keypair.fromSecretKey(Uint8Array.from(secretKey));
  } else {
    console.log('🆕 Creating new wallet...');
    keypair = Keypair.generate();
    fs.writeFileSync(WALLET_FILE, JSON.stringify(Array.from(keypair.secretKey)));
    console.log(`   Wallet saved to: ${WALLET_FILE}`);
  }

  const publicKey = keypair.publicKey.toBase58();
  console.log(`\n🔑 Wallet Address: ${publicKey}`);
  
  // Export format for environment variable
  const privateKeyBase58 = bs58.encode(keypair.secretKey);
  console.log(`\n📋 Private Key (base58 - use for SOLANA_PRIVATE_KEY):`);
  console.log(`   ${privateKeyBase58}`);

  // Connect to devnet
  const connection = new Connection(DEVNET_RPC, 'confirmed');
  
  // Check current balance
  const balance = await connection.getBalance(keypair.publicKey);
  console.log(`\n💰 Current Balance: ${balance / LAMPORTS_PER_SOL} SOL`);

  // Request airdrop if low balance
  if (balance < 1 * LAMPORTS_PER_SOL) {
    console.log('\n🚿 Requesting SOL airdrop...');
    try {
      const signature = await connection.requestAirdrop(
        keypair.publicKey,
        2 * LAMPORTS_PER_SOL // Request 2 SOL
      );
      
      console.log(`   Airdrop tx: ${signature}`);
      
      // Wait for confirmation
      console.log('   Waiting for confirmation...');
      await connection.confirmTransaction(signature, 'confirmed');
      
      const newBalance = await connection.getBalance(keypair.publicKey);
      console.log(`   ✅ New Balance: ${newBalance / LAMPORTS_PER_SOL} SOL`);
    } catch (error) {
      console.log(`   ❌ Airdrop failed: ${error instanceof Error ? error.message : error}`);
      console.log('   Try again later or use https://faucet.solana.com');
    }
  }

  // Print instructions
  console.log('\n=== NEXT STEPS ===');
  console.log(`\n1. Set environment variables:`);
  console.log(`   export SOLANA_PRIVATE_KEY="${privateKeyBase58}"`);
  console.log(`   export SOLANA_RPC_URL="${DEVNET_RPC}"`);
  console.log(`\n2. Run integration test:`);
  console.log(`   npx ts-node src/services/perps/tests/perpsIntegrationTest.ts`);
  console.log(`\n3. View wallet on explorer:`);
  console.log(`   https://explorer.solana.com/address/${publicKey}?cluster=devnet`);
}

main().catch(console.error);

