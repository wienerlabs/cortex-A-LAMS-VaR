/**
 * Test Lending Strategy Integration
 * 
 * Tests the complete lending strategy flow:
 * 1. Load ONNX model
 * 2. Get real-time data from Kamino
 * 3. Extract features
 * 4. Run ML inference
 * 5. Display recommendations
 */
import { Connection, Keypair } from '@solana/web3.js';
import bs58 from 'bs58';
import dotenv from 'dotenv';
import { KaminoLendingClient } from '../src/services/lending/kaminoClient.js';
import {
  getLendingModelLoader,
  createFeatureExtractor,
  type LendingMarketData,
} from '../src/services/lending/index.js';

dotenv.config();

const RPC_URL = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';
const PRIVATE_KEY = process.env.SOLANA_PRIVATE_KEY || '';

async function testLendingIntegration() {
  console.log('🧪 Testing Lending Strategy Integration\n');

  // Step 1: Initialize Kamino client
  console.log('📡 Step 1: Connecting to Kamino...');
  const kaminoClient = new KaminoLendingClient({
    rpcUrl: RPC_URL,
    privateKey: PRIVATE_KEY,
  });

  try {
    await kaminoClient.initialize();
    console.log('✅ Kamino client initialized\n');
  } catch (error) {
    console.error('❌ Failed to initialize Kamino client:', error);
    return;
  }

  // Step 2: Get real-time APY data
  console.log('📊 Step 2: Fetching real-time APY data...');
  const apys = await kaminoClient.getAPYs();
  console.log(`✅ Fetched ${apys.length} lending markets\n`);

  if (apys.length === 0) {
    console.error('❌ No APY data available');
    return;
  }

  // Display sample data
  console.log('Sample markets:');
  apys.slice(0, 5).forEach(apy => {
    console.log(`  ${apy.asset}: Supply ${apy.supplyAPY.toFixed(2)}%, Borrow ${apy.borrowAPY.toFixed(2)}%, Util ${apy.utilization.toFixed(1)}%`);
  });
  console.log();

  // Step 3: Initialize ML model
  console.log('🤖 Step 3: Loading ML model...');
  const modelLoader = getLendingModelLoader({
    minConfidence: 0.6,
    minNetApy: 0.02,  // 2% minimum
  });

  const initialized = await modelLoader.initialize();
  if (!initialized) {
    console.error('❌ Failed to initialize ML model');
    return;
  }
  console.log('✅ ML model loaded\n');

  // Step 4: Extract features and run inference
  console.log('🔬 Step 4: Running ML inference on all markets...');
  const featureExtractor = createFeatureExtractor();
  const recommendations: Array<{
    asset: string;
    supplyApy: number;
    borrowApy: number;
    netApy: number;
    utilization: number;
    tvl: number;
    prediction: any;
  }> = [];

  for (const apy of apys) {
    // Convert to market data format
    const marketData: LendingMarketData = {
      asset: apy.asset,
      protocol: 'kamino',
      tvlUsd: apy.totalDeposits,
      supplyApy: apy.supplyAPY / 100,  // Convert from percentage
      borrowApy: apy.borrowAPY / 100,
      utilizationRate: apy.utilization / 100,
      totalBorrows: apy.totalBorrows,
      availableLiquidity: apy.totalDeposits - apy.totalBorrows,
      protocolTvlUsd: apy.totalDeposits,
      totalSupply: apy.totalDeposits,
      totalBorrow: apy.totalBorrows,
    };

    // Extract features
    const features = featureExtractor.extractFeatures(marketData);
    const netApy = featureExtractor.calculateNetApy(marketData);

    // Run ML prediction
    const prediction = await modelLoader.predict(features, netApy);

    if (prediction.shouldLend) {
      recommendations.push({
        asset: apy.asset,
        supplyApy: apy.supplyAPY,
        borrowApy: apy.borrowAPY,
        netApy: netApy * 100,
        utilization: apy.utilization,
        tvl: apy.totalDeposits,
        prediction,
      });
    }
  }

  // Step 5: Display results
  console.log(`✅ Analysis complete: ${recommendations.length} opportunities found\n`);

  if (recommendations.length === 0) {
    console.log('❌ No lending opportunities meet the criteria (>2% net APY, >60% confidence)\n');
  } else {
    console.log('🎯 Recommended Lending Opportunities:\n');
    console.log('═'.repeat(80));
    
    // Sort by net APY
    recommendations.sort((a, b) => b.netApy - a.netApy);
    
    recommendations.slice(0, 10).forEach((rec, i) => {
      console.log(`\n${i + 1}. ${rec.asset}`);
      console.log(`   Supply APY:    ${rec.supplyApy.toFixed(2)}%`);
      console.log(`   Borrow APY:    ${rec.borrowApy.toFixed(2)}%`);
      console.log(`   Net APY:       ${rec.netApy.toFixed(2)}%`);
      console.log(`   Utilization:   ${rec.utilization.toFixed(1)}%`);
      console.log(`   TVL:           $${(rec.tvl / 1_000_000).toFixed(2)}M`);
      console.log(`   ML Confidence: ${(rec.prediction.confidence * 100).toFixed(1)}%`);
      console.log(`   ML Probability: ${(rec.prediction.probability * 100).toFixed(1)}%`);
    });
    
    console.log('\n' + '═'.repeat(80));
  }

  console.log('\n✅ Test complete!\n');
}

// Run test
testLendingIntegration().catch(console.error);

