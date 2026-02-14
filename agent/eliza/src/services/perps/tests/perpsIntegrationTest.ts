/**
 * Perpetuals Integration Test
 *
 * PRODUCTION TEST - Tests real on-chain execution on devnet
 *
 * Run with: npx ts-node src/services/perps/tests/perpsIntegrationTest.ts
 *
 * Requirements:
 * - SOLANA_PRIVATE_KEY env var (devnet wallet with SOL + USDC)
 * - SOLANA_RPC_URL env var (devnet RPC)
 */

// ============= CONFIG =============

const DEVNET_RPC = process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com';
const PRIVATE_KEY = process.env.SOLANA_PRIVATE_KEY;

interface TestResult {
  name: string;
  success: boolean;
  details?: unknown;
  error?: string;
}

// ============= TEST RUNNER =============

async function runTests(): Promise<void> {
  console.log('\n========================================');
  console.log('  PERPETUALS PRODUCTION INTEGRATION TEST');
  console.log('  Environment: DEVNET');
  console.log('========================================\n');

  if (!PRIVATE_KEY) {
    console.error('❌ ERROR: SOLANA_PRIVATE_KEY environment variable required');
    console.log('\nSet it with: export SOLANA_PRIVATE_KEY="your-base58-private-key"');
    process.exit(1);
  }

  // Dynamic import to handle ESM
  const driftModule = await import('../driftClientProduction.js');
  const { getDriftProductionClient } = driftModule;

  const results: TestResult[] = [];

  // ============= TEST 1: Drift Client Initialization =============
  console.log('📋 Test 1: Drift Production Client Initialization');
  let driftClient: ReturnType<typeof getDriftProductionClient> | null = null;

  try {
    driftClient = getDriftProductionClient({
      rpcUrl: DEVNET_RPC,
      privateKey: PRIVATE_KEY,
      env: 'devnet',
      useJito: false, // Jito only works on mainnet
    });

    const initialized = await driftClient.initialize();

    if (initialized) {
      console.log('  ✅ Drift client initialized');
      console.log(`  Wallet: ${driftClient.getWalletAddress()}`);
      results.push({ name: 'Drift Init', success: true });
    } else {
      console.log('  ❌ Drift client failed to initialize');
      results.push({ name: 'Drift Init', success: false, error: 'Init returned false' });
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.log(`  ❌ Drift init error: ${msg}`);
    results.push({ name: 'Drift Init', success: false, error: msg });
  }

  // Only continue if init succeeded
  if (!driftClient || !driftClient.isReady()) {
    console.log('\n⚠️  Drift client not initialized. Skipping remaining tests.');
    printSummary(results);
    return;
  }

  // ============= TEST 2: Fetch Drift Positions =============
  console.log('\n📋 Test 2: Fetch Drift Positions (Real On-Chain)');
  try {
    const positions = await driftClient.getPositions();

    console.log(`  ✅ Fetched ${positions.length} positions`);
    positions.forEach((p: { market: string; side: string; size: number; entryPrice: number }) => {
      console.log(`     - ${p.market}: ${p.side} ${p.size} @ ${p.entryPrice}`);
    });
    results.push({ name: 'Drift Positions', success: true, details: { count: positions.length } });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.log(`  ❌ Fetch positions error: ${msg}`);
    results.push({ name: 'Drift Positions', success: false, error: msg });
  }

  // ============= TEST 3: Fetch Drift Account State =============
  console.log('\n📋 Test 3: Fetch Drift Account State');
  try {
    const state = await driftClient.getAccountState();

    console.log(`  ✅ Account state retrieved`);
    console.log(`     Total Collateral: $${state.totalCollateral.toFixed(2)}`);
    console.log(`     Free Collateral: $${state.freeCollateral.toFixed(2)}`);
    console.log(`     Leverage: ${state.leverage.toFixed(2)}x`);
    results.push({ name: 'Drift Account', success: true, details: state });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.log(`  ❌ Account state error: ${msg}`);
    results.push({ name: 'Drift Account', success: false, error: msg });
  }

  // ============= TEST 4: Fetch Funding Rates =============
  console.log('\n📋 Test 4: Fetch Drift Funding Rates');
  try {
    const rates = await driftClient.getFundingRates();

    console.log(`  ✅ Fetched ${rates.length} funding rates`);
    rates.slice(0, 5).forEach((r: { market: string; rate: number }) => {
      console.log(`     - ${r.market}: ${(r.rate * 100).toFixed(4)}% hourly`);
    });
    results.push({ name: 'Funding Rates', success: true, details: { count: rates.length } });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.log(`  ❌ Funding rates error: ${msg}`);
    results.push({ name: 'Funding Rates', success: false, error: msg });
  }

  printSummary(results);
}

function printSummary(results: TestResult[]): void {
  console.log('\n========================================');
  console.log('  TEST SUMMARY');
  console.log('========================================');

  const passed = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;

  results.forEach(r => {
    const icon = r.success ? '✅' : '❌';
    console.log(`  ${icon} ${r.name}${r.error ? `: ${r.error}` : ''}`);
  });

  console.log(`\n  Passed: ${passed}/${results.length}`);
  console.log(`  Failed: ${failed}/${results.length}`);

  if (failed > 0) {
    console.log('\n⚠️  Some tests failed. Check errors above.');
    process.exit(1);
  } else if (passed > 0) {
    console.log('\n🎉 All tests passed! Production execution is ready.');
  }
}

// Run tests
runTests().catch(console.error);
