/**
 * Test SpotAnalyst with Mock Data
 * Verifies ML model loading and analysis pipeline
 */

import { SpotAnalyst } from '../src/agents/analysts/SpotAnalyst.js';
import type { ApprovedToken } from '../src/services/trading/tokenWhitelist.js';
import type { TokenMarketData } from '../src/services/spot/ml/featureExtractor.js';

async function testSpotAnalyst() {
  console.log('\n' + '='.repeat(60));
  console.log('SPOT ANALYST TEST');
  console.log('='.repeat(60) + '\n');

  // [1/4] Initialize SpotAnalyst
  console.log('[1/4] Initializing SpotAnalyst...');
  const analyst = new SpotAnalyst({
    minConfidence: 0.60,
    portfolioValueUsd: 10_000,
    volatility24h: 0.05,
    verbose: true,
  });

  // Wait for ML model to load
  await new Promise(resolve => setTimeout(resolve, 2000));

  // [2/4] Create mock token
  console.log('\n[2/4] Creating mock token...');
  const mockToken: ApprovedToken = {
    symbol: 'JUP',
    address: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    marketCap: 1_500_000_000,
    liquidity: 50_000_000,
    volume24h: 8_000_000,
    holders: 125_000,
    age: 450,
    tier: 1,
  };

  // [3/4] Create mock market data
  console.log('[3/4] Creating mock market data...');
  const now = Date.now();
  const mockMarketData: TokenMarketData = {
    // OHLCV data
    prices: Array.from({ length: 200 }, (_, i) => 1.2 + Math.sin(i / 10) * 0.1),
    volumes: Array.from({ length: 200 }, (_, i) => 5_000_000 + Math.random() * 1_000_000),
    timestamps: Array.from({ length: 200 }, (_, i) => now - (199 - i) * 24 * 60 * 60 * 1000),

    // Current data
    currentPrice: 1.25,
    currentVolume: 8_000_000,

    // SOL data
    solPrices: Array.from({ length: 200 }, (_, i) => 180 + Math.sin(i / 15) * 10),
    currentSolPrice: 180,

    // Sentiment data (optional)
    sentimentScore: 0.65,
    socialVolume: 1200,
    newsSentiment: 0.70,
    influencerMentions: 15,

    // Fundamental data
    marketCap: 1_500_000_000,
    liquidity: 50_000_000,
    holders: 125_000,
    tokenAge: 450,
    topHolderShare: 0.15,
    whaleActivity: 0.05,
    sectorPerformance: 0.08,
  };

  // Create market data map
  const marketDataMap = new Map<string, TokenMarketData>();
  marketDataMap.set(mockToken.address, mockMarketData);

  // [4/4] Run analysis
  console.log('\n[4/4] Running analysis...\n');
  try {
    const results = await analyst.analyze({
      tokens: [mockToken],
      marketData: marketDataMap,
      volatility24h: 0.05,
    });

    if (results.length === 0) {
      console.log('\n⚠️  No opportunities found (this is OK for testing)\n');
      return;
    }

    const result = results[0];
    console.log('\n' + '='.repeat(60));
    console.log('ANALYSIS RESULT');
    console.log('='.repeat(60));
    console.log(`Token: ${result.token.symbol}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log(`Expected Return: ${(result.expectedReturn * 100).toFixed(1)}%`);
    console.log(`Risk Score: ${result.riskScore}/10`);
    console.log(`Risk-Adjusted Return: ${(result.riskAdjustedReturn * 100).toFixed(1)}%`);
    console.log(`Approved: ${result.approved ? '✅' : '❌'}`);
    if (result.rejectReason) {
      console.log(`Reject Reason: ${result.rejectReason}`);
    }
    console.log(`\nWarnings: ${result.warnings.length > 0 ? result.warnings.join(', ') : 'None'}`);
    console.log(`\nML Probability: ${(Number(result.raw.mlProbability) * 100).toFixed(1)}%`);
    console.log(`Rule Score: ${Number(result.raw.ruleScore).toFixed(1)}`);
    console.log('='.repeat(60) + '\n');

    console.log('✅ Test completed successfully!\n');
  } catch (error) {
    console.error('\n❌ Test failed:', error);
    process.exit(1);
  }
}

// Run test
testSpotAnalyst().catch(console.error);
