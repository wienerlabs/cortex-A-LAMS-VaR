#!/usr/bin/env npx tsx
/**
 * Test script for Perps ML Model Loader
 * 
 * Run with: npx tsx src/services/perps/ml/testModelLoader.ts
 */
import { getPerpsModelLoader, NUM_FEATURES, FEATURE_NAMES } from './modelLoader.js';

async function main() {
  console.log('='.repeat(60));
  console.log('  PERPS ML MODEL LOADER TEST');
  console.log('='.repeat(60));

  // Initialize model
  const model = getPerpsModelLoader({ minConfidence: 0.6 });
  
  console.log('\n📦 Loading model...');
  const initialized = await model.initialize();
  
  if (!initialized) {
    console.error('❌ Failed to initialize model');
    process.exit(1);
  }
  
  console.log('✅ Model loaded successfully');
  console.log(`   Features: ${NUM_FEATURES}`);

  // Create test feature vector (simulating high funding rate scenario)
  console.log('\n🧪 Running inference test...');
  
  // Simulate a high funding rate scenario (>0.25%)
  const testFeatures = new Array(NUM_FEATURES).fill(0);
  
  // Set some key features
  const featureMap: Record<string, number> = {
    'funding_rate': 0.003,        // 0.3% funding rate (above threshold)
    'funding_rate_raw': 0.003,
    'funding_mean_1h': 0.003,
    'funding_mean_4h': 0.0028,
    'funding_mean_8h': 0.0026,
    'funding_mean_24h': 0.0025,
    'funding_zscore': 2.5,
    'volatility_24h': 0.05,
    'hour': 12,
    'day_of_week': 2,
    'hour_sin': Math.sin(12 * 2 * Math.PI / 24),
    'hour_cos': Math.cos(12 * 2 * Math.PI / 24),
    'dow_sin': Math.sin(2 * 2 * Math.PI / 7),
    'dow_cos': Math.cos(2 * 2 * Math.PI / 7),
  };

  // Fill in the features
  FEATURE_NAMES.forEach((name, idx) => {
    if (name in featureMap) {
      testFeatures[idx] = featureMap[name];
    }
  });

  const fundingRate = 0.003; // 0.3%
  
  try {
    const result = await model.predict(testFeatures, fundingRate);
    
    console.log('\n📊 Prediction Result:');
    console.log(`   Prediction: ${result.prediction === 1 ? 'TRADE' : 'NO_TRADE'}`);
    console.log(`   Probability: ${(result.probability * 100).toFixed(2)}%`);
    console.log(`   Confidence: ${(result.confidence * 100).toFixed(2)}%`);
    console.log(`   Should Trade: ${result.shouldTrade ? 'YES ✅' : 'NO ❌'}`);
    console.log(`   Direction: ${result.direction || 'N/A'}`);
    
    // Test with low funding rate
    console.log('\n🧪 Testing low funding rate scenario...');
    const lowFundingFeatures = [...testFeatures];
    lowFundingFeatures[0] = 0.001; // 0.1% funding rate (below threshold)
    
    const lowResult = await model.predict(lowFundingFeatures, 0.001);
    console.log('\n📊 Low Funding Result:');
    console.log(`   Prediction: ${lowResult.prediction === 1 ? 'TRADE' : 'NO_TRADE'}`);
    console.log(`   Probability: ${(lowResult.probability * 100).toFixed(2)}%`);
    console.log(`   Should Trade: ${lowResult.shouldTrade ? 'YES ✅' : 'NO ❌'}`);
    
    console.log('\n✅ All tests passed!');
    
  } catch (error) {
    console.error('❌ Inference failed:', error);
    process.exit(1);
  }
}

main().catch(console.error);

