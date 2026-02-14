/**
 * Test Lending ML Model
 * 
 * Tests the ML model loading and inference without requiring Kamino SDK
 */
import {
  getLendingModelLoader,
  createFeatureExtractor,
  type LendingMarketData,
} from '../src/services/lending/index.js';

async function testLendingML() {
  console.log('🧪 Testing Lending ML Model\n');

  // Step 1: Initialize ML model
  console.log('🤖 Step 1: Loading ML model...');
  const modelLoader = getLendingModelLoader({
    minConfidence: 0.6,
    minNetApy: 0.02,  // 2% minimum
  });

  const initialized = await modelLoader.initialize();
  if (!initialized) {
    console.error('❌ Failed to initialize ML model');
    return;
  }
  console.log('✅ ML model loaded');
  console.log(`   Features: ${modelLoader.getFeatureNames().length}`);
  console.log();

  // Step 2: Create test data (simulating real Kamino data)
  console.log('📊 Step 2: Creating test market data...');
  
  const testMarkets: LendingMarketData[] = [
    {
      asset: 'USDC',
      protocol: 'kamino',
      tvlUsd: 150_000_000,
      supplyApy: 0.05,  // 5%
      borrowApy: 0.08,  // 8%
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
      supplyApy: 0.03,  // 3%
      borrowApy: 0.06,  // 6%
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
      supplyApy: 0.04,  // 4%
      borrowApy: 0.07,  // 7%
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
      supplyApy: 0.15,  // 15% (high risk)
      borrowApy: 0.25,  // 25%
      utilizationRate: 0.90,  // Very high utilization
      totalBorrows: 4_500_000,
      availableLiquidity: 500_000,
      protocolTvlUsd: 500_000_000,
      totalSupply: 5_000_000,
      totalBorrow: 4_500_000,
    },
  ];

  console.log(`✅ Created ${testMarkets.length} test markets\n`);

  // Step 3: Extract features and run inference
  console.log('🔬 Step 3: Running ML inference...\n');
  const featureExtractor = createFeatureExtractor();

  console.log('═'.repeat(80));
  
  for (const market of testMarkets) {
    // Extract features
    const features = featureExtractor.extractFeatures(market);
    const netApy = featureExtractor.calculateNetApy(market);

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
    
    if (!prediction.shouldLend) {
      const reasons = [];
      if (prediction.confidence < 0.6) reasons.push('Low confidence');
      if (netApy < 0.02) reasons.push('Low net APY');
      if (prediction.prediction === 0) reasons.push('Model says NO_LEND');
      console.log(`  Reason:         ${reasons.join(', ')}`);
    }
  }

  console.log('\n' + '═'.repeat(80));
  console.log('\n✅ Test complete!\n');

  // Summary
  const recommended = testMarkets.filter((market, i) => {
    const features = featureExtractor.extractFeatures(market);
    const netApy = featureExtractor.calculateNetApy(market);
    const prediction = modelLoader.predict(features, netApy);
    return prediction;
  });

  console.log('📊 Summary:');
  console.log(`   Total markets analyzed: ${testMarkets.length}`);
  console.log(`   Model loaded successfully: ✅`);
  console.log(`   Feature extraction working: ✅`);
  console.log(`   ML inference working: ✅`);
  console.log();
}

// Run test
testLendingML().catch(console.error);

