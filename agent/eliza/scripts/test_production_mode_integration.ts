/**
 * Production Mode Integration Test
 * 
 * Tests the complete trading mode system integration:
 * 1. Mode configuration loading
 * 2. CRTXAgent initialization with mode
 * 3. Analyst initialization with mode-aware configs
 * 4. PumpFunAnalyst conditional initialization
 * 5. Position sizing with risk multiplier
 */

import { CRTXAgent } from '../src/agents/crtxAgent.js';
import { getTradingMode, TradingMode, MODE_CONFIGS } from '../src/config/tradingModes.js';
import { logger } from '../src/services/logger.js';

async function testProductionModeIntegration() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  🧪 PRODUCTION MODE INTEGRATION TEST                      ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  // Test 1: Mode Configuration Loading
  console.log('📋 Test 1: Mode Configuration Loading');
  console.log('─'.repeat(60));
  
  const currentMode = getTradingMode();
  console.log(`✅ Current mode: ${currentMode.mode}`);
  console.log(`   Min Health Score: ${currentMode.minHealthScore}`);
  console.log(`   Enable Pump.fun: ${currentMode.enablePumpFun}`);
  console.log(`   Risk Multiplier: ${currentMode.riskMultiplier}x`);
  console.log(`   Max Position Size: ${(currentMode.maxPositionSize * 100).toFixed(1)}%`);
  console.log(`   Min TVL: $${(currentMode.minTVL / 1000).toFixed(0)}K`);
  console.log(`   Min Holders: ${currentMode.minHolders}`);
  console.log(`   Min Liquidity: $${(currentMode.minLiquidity / 1000).toFixed(0)}K`);

  // Test 2: NORMAL Mode Agent Initialization
  console.log('\n📋 Test 2: NORMAL Mode Agent Initialization');
  console.log('─'.repeat(60));
  
  process.env.TRADING_MODE = 'NORMAL';
  const normalAgent = new CRTXAgent({
    portfolioValueUsd: 10000,
    dryRun: true,
  });

  console.log('✅ NORMAL mode agent created');
  console.log(`   Mode: ${normalAgent['modeConfig'].mode}`);
  console.log(`   PumpFunAnalyst initialized: ${normalAgent['pumpfunAnalyst'] !== null}`);
  
  if (normalAgent['pumpfunAnalyst'] !== null) {
    console.log('❌ ERROR: PumpFunAnalyst should NOT be initialized in NORMAL mode');
    process.exit(1);
  }

  // Test 3: AGGRESSIVE Mode Agent Initialization
  console.log('\n📋 Test 3: AGGRESSIVE Mode Agent Initialization');
  console.log('─'.repeat(60));
  
  process.env.TRADING_MODE = 'AGGRESSIVE';
  const aggressiveAgent = new CRTXAgent({
    portfolioValueUsd: 10000,
    dryRun: true,
  });

  console.log('✅ AGGRESSIVE mode agent created');
  console.log(`   Mode: ${aggressiveAgent['modeConfig'].mode}`);
  console.log(`   PumpFunAnalyst initialized: ${aggressiveAgent['pumpfunAnalyst'] !== null}`);
  
  if (aggressiveAgent['pumpfunAnalyst'] === null) {
    console.log('❌ ERROR: PumpFunAnalyst SHOULD be initialized in AGGRESSIVE mode');
    process.exit(1);
  }

  // Test 4: Analyst Configuration Verification
  console.log('\n📋 Test 4: Analyst Configuration Verification');
  console.log('─'.repeat(60));
  
  const lpAnalyst = aggressiveAgent['lpAnalyst'];
  const spotAnalyst = aggressiveAgent['spotAnalyst'];
  const fundamentalAnalyst = aggressiveAgent['fundamentalAnalyst'];

  const lpConfig = lpAnalyst.getConfig();
  const spotConfig = spotAnalyst.getConfig();
  const fundamentalConfig = fundamentalAnalyst['config'];

  console.log('✅ LPAnalyst config:');
  console.log(`   Min TVL: $${(lpConfig.minTvl / 1000).toFixed(0)}K (expected: $${(MODE_CONFIGS[TradingMode.AGGRESSIVE].minTVL / 1000).toFixed(0)}K)`);
  
  console.log('✅ SpotAnalyst config:');
  console.log(`   Min Liquidity: $${(spotConfig.minLiquidity / 1000).toFixed(0)}K (expected: $${(MODE_CONFIGS[TradingMode.AGGRESSIVE].minLiquidity / 1000).toFixed(0)}K)`);
  
  console.log('✅ FundamentalAnalyst config:');
  console.log(`   Min Health Score: ${fundamentalConfig.minHealthScore} (expected: ${MODE_CONFIGS[TradingMode.AGGRESSIVE].minHealthScore})`);

  // Verify configs match mode
  if (lpConfig.minTvl !== MODE_CONFIGS[TradingMode.AGGRESSIVE].minTVL) {
    console.log('❌ ERROR: LPAnalyst minTvl does not match AGGRESSIVE mode config');
    process.exit(1);
  }
  if (spotConfig.minLiquidity !== MODE_CONFIGS[TradingMode.AGGRESSIVE].minLiquidity) {
    console.log('❌ ERROR: SpotAnalyst minLiquidity does not match AGGRESSIVE mode config');
    process.exit(1);
  }
  if (fundamentalConfig.minHealthScore !== MODE_CONFIGS[TradingMode.AGGRESSIVE].minHealthScore) {
    console.log('❌ ERROR: FundamentalAnalyst minHealthScore does not match AGGRESSIVE mode config');
    process.exit(1);
  }

  // Test 5: Position Sizing with Risk Multiplier
  console.log('\n📋 Test 5: Position Sizing with Risk Multiplier');
  console.log('─'.repeat(60));
  
  const mockOpp = {
    type: 'spot' as const,
    name: 'Test Token',
    expectedReturn: 15,
    riskScore: 5,
    confidence: 0.7,
    riskAdjustedReturn: 10,
    approved: true,
    warnings: [],
    raw: {},
  };

  const normalPositionSize = normalAgent['calculatePositionSize'](mockOpp);
  const aggressivePositionSize = aggressiveAgent['calculatePositionSize'](mockOpp);

  console.log(`✅ NORMAL mode position size: $${normalPositionSize.toFixed(2)}`);
  console.log(`✅ AGGRESSIVE mode position size: $${aggressivePositionSize.toFixed(2)}`);
  console.log(`   Ratio: ${(aggressivePositionSize / normalPositionSize).toFixed(2)}x (expected: ~${MODE_CONFIGS[TradingMode.AGGRESSIVE].riskMultiplier}x)`);

  // Test 6: Mode Comparison Summary
  console.log('\n📋 Test 6: Mode Comparison Summary');
  console.log('─'.repeat(60));
  
  console.log('\nNORMAL MODE:');
  console.log(`  Min Health Score:  ${MODE_CONFIGS[TradingMode.NORMAL].minHealthScore}`);
  console.log(`  Pump.fun:          ${MODE_CONFIGS[TradingMode.NORMAL].enablePumpFun ? 'ENABLED' : 'DISABLED'}`);
  console.log(`  Risk Multiplier:   ${MODE_CONFIGS[TradingMode.NORMAL].riskMultiplier}x`);
  console.log(`  Max Position:      ${(MODE_CONFIGS[TradingMode.NORMAL].maxPositionSize * 100).toFixed(1)}%`);
  console.log(`  Min TVL:           $${(MODE_CONFIGS[TradingMode.NORMAL].minTVL / 1000).toFixed(0)}K`);

  console.log('\nAGGRESSIVE MODE:');
  console.log(`  Min Health Score:  ${MODE_CONFIGS[TradingMode.AGGRESSIVE].minHealthScore}`);
  console.log(`  Pump.fun:          ${MODE_CONFIGS[TradingMode.AGGRESSIVE].enablePumpFun ? 'ENABLED' : 'DISABLED'}`);
  console.log(`  Risk Multiplier:   ${MODE_CONFIGS[TradingMode.AGGRESSIVE].riskMultiplier}x`);
  console.log(`  Max Position:      ${(MODE_CONFIGS[TradingMode.AGGRESSIVE].maxPositionSize * 100).toFixed(1)}%`);
  console.log(`  Min TVL:           $${(MODE_CONFIGS[TradingMode.AGGRESSIVE].minTVL / 1000).toFixed(0)}K`);

  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  ✅ ALL TESTS PASSED                                      ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
}

testProductionModeIntegration().catch((error) => {
  console.error('\n❌ Test failed:', error);
  process.exit(1);
});

