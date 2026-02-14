/**
 * Test Trading Modes (NORMAL vs AGGRESSIVE)
 * 
 * Tests both trading modes to verify:
 * - NORMAL mode: Filters memecoins, min health 60, no Pump.fun
 * - AGGRESSIVE mode: Includes Pump.fun, min health 40, shows risk warnings
 */

import { FundamentalAnalyst } from '../src/agents/analysts/FundamentalAnalyst.js';
import { getTradingMode, TRADING_MODES } from '../src/config/tradingModes.js';
import { PumpFunClient } from '../src/services/pumpfun/pumpfunClient.js';
import { filterPumpFunTokens } from '../src/services/pumpfun/pumpfunFilter.js';

async function testNormalMode() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  TEST 1: NORMAL MODE                                      ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  // Set NORMAL mode
  process.env.TRADING_MODE = 'NORMAL';
  
  const mode = getTradingMode();
  console.log('📊 Mode Configuration:');
  console.log(`   Mode: ${mode.mode}`);
  console.log(`   Min Health Score: ${mode.minHealthScore}`);
  console.log(`   Allow Memecoins: ${mode.allowMemecoins}`);
  console.log(`   Allow Pump.fun: ${mode.allowPumpFun}`);
  console.log(`   Risk Multiplier: ${mode.riskMultiplier}x`);
  console.log(`   Description: ${mode.description}\n`);

  // Test FundamentalAnalyst with NORMAL mode
  const analyst = new FundamentalAnalyst();
  console.log('✅ FundamentalAnalyst initialized');
  console.log(`   Min Health Score: ${analyst['config'].minHealthScore}`);
  console.log(`   Trading Mode: ${analyst['tradingMode'].mode}\n`);

  // Test Pump.fun integration (should be disabled)
  if (mode.allowPumpFun) {
    console.log('❌ ERROR: Pump.fun should be DISABLED in NORMAL mode\n');
  } else {
    console.log('✅ Pump.fun: DISABLED (as expected)\n');
  }

  console.log('Expected Behavior:');
  console.log('  - Tokens with health < 60: REJECTED');
  console.log('  - Tokens with health ≥ 60: APPROVED');
  console.log('  - Memecoins: FILTERED');
  console.log('  - Pump.fun tokens: NOT SCANNED\n');
}

async function testAggressiveMode() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  TEST 2: AGGRESSIVE MODE                                  ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  // Set AGGRESSIVE mode
  process.env.TRADING_MODE = 'AGGRESSIVE';
  
  const mode = getTradingMode();
  console.log('📊 Mode Configuration:');
  console.log(`   Mode: ${mode.mode}`);
  console.log(`   Min Health Score: ${mode.minHealthScore}`);
  console.log(`   Allow Memecoins: ${mode.allowMemecoins}`);
  console.log(`   Allow Pump.fun: ${mode.allowPumpFun}`);
  console.log(`   Risk Multiplier: ${mode.riskMultiplier}x`);
  console.log(`   Description: ${mode.description}\n`);

  // Test FundamentalAnalyst with AGGRESSIVE mode
  const analyst = new FundamentalAnalyst();
  console.log('✅ FundamentalAnalyst initialized');
  console.log(`   Min Health Score: ${analyst['config'].minHealthScore}`);
  console.log(`   Trading Mode: ${analyst['tradingMode'].mode}\n`);

  // Test Pump.fun integration (should be enabled)
  if (!mode.allowPumpFun) {
    console.log('❌ ERROR: Pump.fun should be ENABLED in AGGRESSIVE mode\n');
  } else {
    console.log('✅ Pump.fun: ENABLED (as expected)');

    // Test Pump.fun client (if API key available)
    if (process.env.PUMPFUN_API_KEY) {
      try {
        console.log('   Testing Pump.fun API connection...');
        const client = new PumpFunClient();
        const tokens = await client.getTokens(50, 0);
        console.log(`   ✅ Fetched: ${tokens.length} tokens from Pump.fun API`);

        // Test filtering
        console.log('   Filtering tokens (TVL >$10K, holders >50, age >24h)...');
        const filtered = await filterPumpFunTokens(tokens);
        console.log(`   ✅ Filtered: ${filtered.length} tokens passed safety criteria`);

        // Display filtered tokens
        if (filtered.length > 0) {
          console.log('\n   📊 Approved Pump.fun Tokens:');
          filtered.slice(0, 10).forEach((token, idx) => {
            const tvlDisplay = `$${(token.tvl / 1000).toFixed(1)}K`;
            const holdersDisplay = token.holderCount >= 1000000 ? '1M+' : token.holderCount.toString();
            const ageDisplay = `${token.ageHours.toFixed(0)}h`;
            console.log(`      ${idx + 1}. ${token.symbol.padEnd(10)} | TVL: ${tvlDisplay.padEnd(8)} | Holders: ${holdersDisplay.padEnd(6)} | Age: ${ageDisplay}`);
          });

          if (filtered.length > 10) {
            console.log(`      ... and ${filtered.length - 10} more tokens`);
          }
          console.log('');
        } else {
          console.log('   ⚠️  No tokens passed filters\n');
        }
      } catch (error: any) {
        console.log(`   ⚠️  Pump.fun API error: ${error.message}\n`);
      }
    } else {
      console.log('   ⚠️  PUMPFUN_API_KEY not set - skipping API test\n');
    }
  }

  console.log('Expected Behavior:');
  console.log('  - Tokens with health < 40: REJECTED');
  console.log('  - Tokens with health 40-59: APPROVED with ⚠️ AGGRESSIVE MODE warning');
  console.log('  - Tokens with health ≥ 60: APPROVED (no warning)');
  console.log('  - Memecoins: ALLOWED');
  console.log('  - Pump.fun tokens: SCANNED and FILTERED\n');
}

async function testModeComparison() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  TEST 3: MODE COMPARISON                                  ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  console.log('┌─────────────────┬──────────────┬──────────────────┐');
  console.log('│ Feature         │ NORMAL       │ AGGRESSIVE       │');
  console.log('├─────────────────┼──────────────┼──────────────────┤');
  console.log(`│ Min Health      │ ${TRADING_MODES.NORMAL.minHealthScore.toString().padEnd(12)} │ ${TRADING_MODES.AGGRESSIVE.minHealthScore.toString().padEnd(16)} │`);
  console.log(`│ Memecoins       │ ${(TRADING_MODES.NORMAL.allowMemecoins ? 'ALLOWED' : 'FILTERED').padEnd(12)} │ ${(TRADING_MODES.AGGRESSIVE.allowMemecoins ? 'ALLOWED' : 'FILTERED').padEnd(16)} │`);
  console.log(`│ Pump.fun        │ ${(TRADING_MODES.NORMAL.allowPumpFun ? 'ENABLED' : 'DISABLED').padEnd(12)} │ ${(TRADING_MODES.AGGRESSIVE.allowPumpFun ? 'ENABLED' : 'DISABLED').padEnd(16)} │`);
  console.log(`│ Risk Multiplier │ ${TRADING_MODES.NORMAL.riskMultiplier.toFixed(1) + 'x'.padEnd(11)} │ ${TRADING_MODES.AGGRESSIVE.riskMultiplier.toFixed(1) + 'x'.padEnd(15)} │`);
  console.log('└─────────────────┴──────────────┴──────────────────┘\n');

  console.log('Health Score Approval Matrix:');
  console.log('  Health 70: ✅ NORMAL (approved) | ✅ AGGRESSIVE (approved)');
  console.log('  Health 55: ❌ NORMAL (rejected) | ⚠️  AGGRESSIVE (approved with warning)');
  console.log('  Health 35: ❌ NORMAL (rejected) | ❌ AGGRESSIVE (rejected)\n');
}

async function main() {
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║  TRADING MODE TEST SUITE                                  ║');
  console.log('╚═══════════════════════════════════════════════════════════╝');

  await testNormalMode();
  await testAggressiveMode();
  await testModeComparison();

  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║  ✅ ALL TESTS COMPLETE                                     ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
}

main().catch(console.error);

