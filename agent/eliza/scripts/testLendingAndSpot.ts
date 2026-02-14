#!/usr/bin/env npx tsx
/**
 * Test Lending and Spot Execution
 * 
 * Tests the newly implemented lending and spot trading executors:
 * 1. Initialize CRTXAgent with test configuration
 * 2. Run one trading cycle
 * 3. Check if executors are initialized
 * 4. Check if opportunities are detected
 * 5. Verify execution flow
 * 
 * Usage:
 *   # DRY RUN (safe, no real transactions)
 *   npx tsx scripts/testLendingAndSpot.ts
 * 
 *   # LIVE TEST on devnet (requires devnet wallet with SOL/USDC)
 *   npx tsx scripts/testLendingAndSpot.ts --live
 * 
 * Environment Variables:
 *   SOLANA_RPC_URL       - Solana RPC endpoint (default: devnet)
 *   SOLANA_PRIVATE_KEY   - Wallet private key (base58) for live testing
 *   TRADING_MODE         - 'NORMAL' or 'AGGRESSIVE' (default: NORMAL)
 */

import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env from agent/ directory (parent of eliza/)
config({ path: resolve(__dirname, '../../.env') });

import { CRTXAgent } from '../src/agents/crtxAgent.js';
import { logger } from '../src/services/logger.js';

async function main() {
  const args = process.argv.slice(2);
  const liveMode = args.includes('--live');
  
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  🧪 LENDING & SPOT EXECUTION TEST                        ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
  
  console.log('📋 Test Configuration:');
  console.log(`   Mode: ${liveMode ? '🔴 LIVE' : '📝 DRY RUN'}`);
  console.log(`   Network: ${process.env.SOLANA_RPC_URL || 'devnet (default)'}`);
  console.log(`   Trading Mode: ${process.env.TRADING_MODE || 'NORMAL (default)'}`);
  console.log('');
  
  if (liveMode && !process.env.SOLANA_PRIVATE_KEY) {
    console.error('❌ ERROR: SOLANA_PRIVATE_KEY required for live mode');
    console.error('   Set it in .env file or environment variable');
    process.exit(1);
  }
  
  try {
    // Set devnet RPC if not specified
    if (!process.env.SOLANA_RPC_URL) {
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';
      console.log('ℹ️  Using devnet RPC (default)');
    }
    
    // Set NORMAL mode if not specified
    if (!process.env.TRADING_MODE) {
      process.env.TRADING_MODE = 'NORMAL';
      console.log('ℹ️  Using NORMAL trading mode (default)\n');
    }
    
    console.log('🤖 Initializing CRTXAgent...\n');
    
    // Create agent with test configuration
    const agent = new CRTXAgent({
      portfolioValueUsd: 10000,
      minConfidence: 0.5,  // Lower threshold to find opportunities easily
      minRiskAdjustedReturn: 0.5,  // Lower threshold for testing
      dryRun: !liveMode,
      volatility24h: 0.05,
      solanaRpcUrl: process.env.SOLANA_RPC_URL,
      solanaPrivateKey: process.env.SOLANA_PRIVATE_KEY,
    });
    
    console.log('✅ Agent initialized successfully!\n');

    // Wait a bit for async executor initialization to complete
    console.log('⏳ Waiting for executors to initialize...');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Check executor initialization
    console.log('🔍 Checking executor initialization...');
    const hasLendingExecutor = (agent as any).lendingExecutor !== null;
    const hasSpotExecutor = (agent as any).spotExecutor !== null;
    
    console.log(`   Lending Executor: ${hasLendingExecutor ? '✅ Initialized' : '❌ Not initialized'}`);
    console.log(`   Spot Executor: ${hasSpotExecutor ? '✅ Initialized' : '❌ Not initialized'}`);
    console.log('');
    
    if (!hasLendingExecutor && !liveMode) {
      console.log('ℹ️  Lending executor not initialized (wallet required)');
    }
    if (!hasSpotExecutor && !liveMode) {
      console.log('ℹ️  Spot executor not initialized (wallet required)');
    }
    console.log('');
    
    // Run one trading cycle
    console.log('🔄 Running trading cycle...\n');
    console.log('─'.repeat(60));
    
    const result = await agent.run();
    
    console.log('─'.repeat(60));
    console.log('');
    
    // Display results
    if (result) {
      console.log('✅ OPPORTUNITY FOUND AND PROCESSED:\n');
      console.log(`   Type: ${result.type}`);
      console.log(`   Name: ${result.name}`);
      console.log(`   Expected Return: +${result.expectedReturn.toFixed(2)}%`);
      console.log(`   Risk-Adjusted Return: +${result.riskAdjustedReturn.toFixed(2)}%`);
      console.log(`   Confidence: ${(result.confidence * 100).toFixed(0)}%`);
      console.log(`   Risk Score: ${result.riskScore}/10`);
      
      if (result.warnings && result.warnings.length > 0) {
        console.log(`   ⚠️  Warnings: ${result.warnings.join(', ')}`);
      }
      
      console.log('');
      
      // Check if it was lending or spot
      if (result.type === 'lending') {
        console.log('📊 LENDING EXECUTION TEST:');
        console.log(`   ${liveMode ? '✅ Real transaction executed' : '📝 Dry run - no real transaction'}`);
      } else if (result.type === 'spot') {
        console.log('📊 SPOT EXECUTION TEST:');
        console.log(`   ${liveMode ? '✅ Real transaction executed' : '📝 Dry run - no real transaction'}`);
      }
    } else {
      console.log('ℹ️  NO OPPORTUNITIES FOUND\n');
      console.log('   This is normal if:');
      console.log('   - Market conditions don\'t meet criteria');
      console.log('   - No profitable opportunities available');
      console.log('   - API rate limits reached');
      console.log('');
      console.log('   Try:');
      console.log('   - Lowering minConfidence further (e.g., 0.3)');
      console.log('   - Running multiple times');
      console.log('   - Checking API keys in .env');
    }

    console.log('');

    // Show risk state
    const state = agent.getRiskState();
    console.log('📊 FINAL RISK STATE:');
    console.log(`   Daily PnL: ${state.dailyPnL.toFixed(2)}%`);
    console.log(`   Daily Trades: ${state.dailyTradeCount}`);
    console.log('');

    console.log('✅ TEST COMPLETED SUCCESSFULLY!\n');

    if (!liveMode) {
      console.log('💡 NEXT STEPS:');
      console.log('   1. Review the logs above');
      console.log('   2. If executors initialized: ✅ Implementation working');
      console.log('   3. If opportunities found: ✅ Detection working');
      console.log('   4. For live test: npx tsx scripts/testLendingAndSpot.ts --live');
      console.log('      (Make sure you have devnet SOL and USDC first!)');
      console.log('');
    } else {
      console.log('💡 LIVE TEST RESULTS:');
      console.log('   Check transaction signatures above');
      console.log('   Verify on Solana Explorer:');
      console.log('   https://explorer.solana.com/?cluster=devnet');
      console.log('');
    }

  } catch (error: any) {
    logger.error('[TEST] Fatal error during test', {
      error: error.message,
      stack: error.stack,
    });
    console.error('\n❌ TEST FAILED:', error.message);
    console.error('\nStack trace:', error.stack);
    console.error('');
    console.error('🔍 TROUBLESHOOTING:');
    console.error('   1. Check .env file has required API keys');
    console.error('   2. Check SOLANA_RPC_URL is accessible');
    console.error('   3. Check SOLANA_PRIVATE_KEY format (base58)');
    console.error('   4. Check network connectivity');
    console.error('');
    process.exit(1);
  }
}

// Run the test
main().catch((error) => {
  console.error('\n❌ Unhandled error:', error);
  process.exit(1);
});

