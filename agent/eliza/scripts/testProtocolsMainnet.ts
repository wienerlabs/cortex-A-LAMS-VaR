#!/usr/bin/env npx tsx
/**
 * Test Protocol Initialization on Mainnet (Read-Only)
 * 
 * This script tests if MarginFi, Kamino, and Solend can initialize on mainnet
 * WITHOUT executing any transactions (read-only mode).
 */

import { Keypair } from '@solana/web3.js';
import bs58 from 'bs58';
import { MarginFiLendingClient } from '../src/services/lending/marginfiClient.js';
import { KaminoLendingClient } from '../src/services/lending/kaminoClient.js';
import { SolendLendingClient } from '../src/services/lending/solendClient.js';

const MAINNET_RPC = 'https://api.mainnet-beta.solana.com';

// Generate a temporary keypair (won't be used for transactions)
const tempKeypair = Keypair.generate();
const tempPrivateKey = bs58.encode(tempKeypair.secretKey);

console.log('\n╔═══════════════════════════════════════════════════════════╗');
console.log('║  🧪 MAINNET PROTOCOL INITIALIZATION TEST (READ-ONLY)     ║');
console.log('╚═══════════════════════════════════════════════════════════╝\n');

console.log('📋 Test Configuration:');
console.log(`   Network: MAINNET (read-only)`);
console.log(`   Temp Wallet: ${tempKeypair.publicKey.toBase58()}`);
console.log(`   ⚠️  No real transactions will be executed\n`);

async function testMarginFi() {
  console.log('─'.repeat(60));
  console.log('🔵 Testing MarginFi...\n');
  
  try {
    const client = new MarginFiLendingClient({
      rpcUrl: MAINNET_RPC,
      privateKey: tempPrivateKey,
      environment: 'production',
    });
    
    console.log('   ⏳ Initializing...');
    await client.initialize();
    
    console.log('   ✅ MarginFi initialized successfully!');
    console.log(`   📊 Account: ${client.accountAddress?.toBase58() || 'N/A'}`);
    
    // Try to fetch APYs
    const apys = await client.getAPYs();
    console.log(`   📈 Available markets: ${apys.length}`);
    if (apys.length > 0) {
      console.log(`   💰 Sample APY (${apys[0].asset}): ${apys[0].supplyAPY.toFixed(2)}%`);
    }
    
    return true;
  } catch (error: any) {
    console.log('   ❌ MarginFi failed to initialize');
    console.log(`   📝 Error: ${error.message}`);
    return false;
  }
}

async function testKamino() {
  console.log('\n─'.repeat(60));
  console.log('🟢 Testing Kamino...\n');
  
  try {
    const client = new KaminoLendingClient({
      rpcUrl: MAINNET_RPC,
      privateKey: tempPrivateKey,
    });
    
    console.log('   ⏳ Initializing...');
    await client.initialize();
    
    console.log('   ✅ Kamino initialized successfully!');
    console.log(`   📊 Obligation: ${client.obligationAddress?.toString() || 'None (will create on first action)'}`);
    
    // Try to fetch APYs
    const apys = await client.getAPYs();
    console.log(`   📈 Available markets: ${apys.length}`);
    if (apys.length > 0) {
      console.log(`   💰 Sample APY (${apys[0].asset}): ${apys[0].supplyAPY.toFixed(2)}%`);
    }
    
    return true;
  } catch (error: any) {
    console.log('   ❌ Kamino failed to initialize');
    console.log(`   📝 Error: ${error.message}`);
    return false;
  }
}

async function testSolend() {
  console.log('\n─'.repeat(60));
  console.log('🟡 Testing Solend...\n');
  
  try {
    const client = new SolendLendingClient({
      rpcUrl: MAINNET_RPC,
      privateKey: tempPrivateKey,
    });
    
    console.log('   ⏳ Initializing...');
    await client.initialize();
    
    console.log('   ✅ Solend initialized successfully!');
    
    // Try to fetch APYs
    const apys = await client.getAPYs();
    console.log(`   📈 Available markets: ${apys.length}`);
    if (apys.length > 0) {
      console.log(`   💰 Sample APY (${apys[0].asset}): ${apys[0].supplyAPY.toFixed(2)}%`);
    }
    
    return true;
  } catch (error: any) {
    console.log('   ❌ Solend failed to initialize');
    console.log(`   📝 Error: ${error.message}`);
    return false;
  }
}

async function main() {
  const results = {
    marginfi: false,
    kamino: false,
    solend: false,
  };
  
  results.marginfi = await testMarginFi();
  results.kamino = await testKamino();
  results.solend = await testSolend();
  
  console.log('\n' + '═'.repeat(60));
  console.log('📊 RESULTS SUMMARY');
  console.log('═'.repeat(60) + '\n');
  
  console.log(`   MarginFi: ${results.marginfi ? '✅ WORKING' : '❌ FAILED'}`);
  console.log(`   Kamino:   ${results.kamino ? '✅ WORKING' : '❌ FAILED'}`);
  console.log(`   Solend:   ${results.solend ? '✅ WORKING' : '❌ FAILED'}`);
  
  const successCount = Object.values(results).filter(Boolean).length;
  console.log(`\n   Total: ${successCount}/3 protocols initialized successfully`);
  
  if (successCount === 3) {
    console.log('\n   🎉 All protocols are working on mainnet!');
  } else if (successCount > 0) {
    console.log('\n   ⚠️  Some protocols failed - check errors above');
  } else {
    console.log('\n   ❌ All protocols failed - may be RPC or network issues');
  }
  
  console.log('\n' + '═'.repeat(60) + '\n');
}

main().catch(console.error);

