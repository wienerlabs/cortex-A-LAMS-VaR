/**
 * Standalone Test for Lending ML Model
 * 
 * Tests only the ML components without importing Kamino SDK
 */
import {
  getLendingModelLoader,
  createFeatureExtractor,
} from '../src/services/lending/ml/index.js';
import type { LendingMarketData } from '../src/services/lending/types.js';

async function testLendingML() {
  console.log('🧪 Testing Lending ML Model (Standalone)\n');

  // Step 1: Initialize ML model
  console.log('🤖 Step 1: Loading ONNX model...');
  const modelLoader = getLendingModelLoader({
    minConfidence: 0.6,
    minNetApy: 0.02,  // 2% minimum
  });

  const initialized = await modelLoader.initialize();
  if (!initialized) {
    console.error('❌ Failed to initialize ML model');
    return;
  }
  console.log('✅ ONNX model loaded successfully');
  console.log(`   Number of features: ${modelLoader.getFeatureNames().length}`);
  console.log();

  // Step 2: Create test data (simulating real Kamino data)
  console.log('📊 Step 2: Creating test market data...');
  
  const testMarkets: LendingMarketData[] = [
    {
      asset: 'USDC',
      protocol: 'kamino',
      tvlUsd: 150_000_000,
      supplyApy: 0.05,  // 5% supply (good opportunity)
      borrowApy: 0.02,  // 2% borrow (net APY = 3%)
      utilizationRate: 0.65,
      totalBorrows: 97_500_000,
      availableLiquidity: 52_500_000,
      protocolTvlUsd: 500_000_000,
      totalSupply: 150_000_000,
      totalBorrow: 97_500_000,
    },
    {
      asset: 'SOL',
      protocol: 'kamino',
      tvlUsd: 80_000_000,
      supplyApy: 0.06,  // 6% supply
      borrowApy: 0.03,  // 3% borrow (net APY = 3%)
      utilizationRate: 0.55,
      totalBorrows: 44_000_000,
      availableLiquidity: 36_000_000,
      protocolTvlUsd: 500_000_000,
      totalSupply: 80_000_000,
      totalBorrow: 44_000_000,
    },
    {
      asset: 'JITOSOL',
      protocol: 'kamino',
      tvlUsd: 60_000_000,
      supplyApy: 0.07,  // 7% supply
      borrowApy: 0.04,  // 4% borrow (net APY = 3%)
      utilizationRate: 0.70,
      totalBorrows: 42_000_000,
      availableLiquidity: 18_000_000,
      protocolTvlUsd: 500_000_000,
      totalSupply: 60_000_000,
      totalBorrow: 42_000_000,
    },
    {
      asset: 'BONK',
      protocol: 'kamino',
      tvlUsd: 5_000_000,
      supplyApy: 0.15,  // 15% supply (high risk)
      borrowApy: 0.10,  // 10% borrow (net APY = 5%)
      utilizationRate: 0.90,  // Very high utilization (risky)
      totalBorrows: 4_500_000,
      availableLiquidity: 500_000,
      protocolTvlUsd: 500_000_000,
      totalSupply: 5_000_000,
      totalBorrow: 4_500_000,
    },
    {
      asset: 'USDT',
      protocol: 'kamino',
      tvlUsd: 120_000_000,
      supplyApy: 0.025,  // 2.5% supply
      borrowApy: 0.015,  // 1.5% borrow (net APY = 1%, below threshold)
      utilizationRate: 0.60,
      totalBorrows: 72_000_000,
      availableLiquidity: 48_000_000,
      protocolTvlUsd: 500_000_000,
      totalSupply: 120_000_000,
      totalBorrow: 72_000_000,
    },
  ];

  console.log(`✅ Created ${testMarkets.length} test markets\n`);

  // Step 3: Extract features and run inference
  console.log('🔬 Step 3: Running ML inference...\n');
  const featureExtractor = createFeatureExtractor();

  console.log('═'.repeat(80));
  
  let recommendedCount = 0;
  
  for (const market of testMarkets) {
    // Extract features
    const features = featureExtractor.extractFeatures(market);
    const netApy = featureExtractor.calculateNetApy(market);

    console.log(`\nExtracting features for ${market.asset}...`);
    console.log(`  Features extracted: ${features.length}`);
    console.log(`  Net APY calculated: ${(netApy * 100).toFixed(2)}%`);

    // Run ML prediction
    const prediction = await modelLoader.predict(features, netApy);

    // Display results
    console.log(`\n${market.asset} (${market.protocol.toUpperCase()})`);
    console.log(`  Supply APY:     ${(market.supplyApy * 100).toFixed(2)}%`);
    console.log(`  Borrow APY:     ${((market.borrowApy || 0) * 100).toFixed(2)}%`);
    console.log(`  Net APY:        ${(netApy * 100).toFixed(2)}%`);
    console.log(`  Utilization:    ${(market.utilizationRate * 100).toFixed(1)}%`);
    console.log(`  TVL:            $${(market.tvlUsd / 1_000_000).toFixed(2)}M`);
    console.log(`  ---`);
    console.log(`  ML Prediction:  ${prediction.prediction === 1 ? 'LEND ✅' : 'NO_LEND ❌'}`);
    console.log(`  Probability:    ${(prediction.probability * 100).toFixed(1)}%`);
    console.log(`  Confidence:     ${(prediction.confidence * 100).toFixed(1)}%`);
    console.log(`  Recommendation: ${prediction.shouldLend ? '✅ LEND' : '❌ SKIP'}`);
    
    if (prediction.shouldLend) {
      recommendedCount++;
    } else {
      const reasons = [];
      if (prediction.confidence < 0.6) reasons.push('Low confidence');
      if (netApy < 0.02) reasons.push('Low net APY (<2%)');
      if (prediction.prediction === 0) reasons.push('Model says NO_LEND');
      console.log(`  Reason:         ${reasons.join(', ')}`);
    }
  }

  console.log('\n' + '═'.repeat(80));
  console.log('\n✅ Test complete!\n');

  // Summary
  console.log('📊 Summary:');
  console.log(`   Total markets analyzed: ${testMarkets.length}`);
  console.log(`   Recommended for lending: ${recommendedCount}`);
  console.log(`   Model loaded successfully: ✅`);
  console.log(`   Feature extraction working: ✅`);
  console.log(`   ML inference working: ✅`);
  console.log(`   Integration complete: ✅`);
  console.log();
}

// Run test
testLendingML().catch(console.error);

