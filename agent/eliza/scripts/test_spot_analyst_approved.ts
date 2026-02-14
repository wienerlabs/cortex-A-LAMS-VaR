/**
 * Test SpotAnalyst with Strong Signals to Get Approval
 */

import { SpotAnalyst } from '../src/agents/analysts/SpotAnalyst.js';
import type { ApprovedToken } from '../src/services/trading/types.js';
import type { TokenMarketData } from '../src/services/spot/ml/featureExtractor.js';

async function testSpotAnalystApproved() {
  console.log('\n============================================================');
  console.log('SPOT ANALYST TEST - STRONG SIGNALS FOR APPROVAL');
  console.log('============================================================\n');

  // [1/4] Initialize analyst with LOWER threshold for testing
  console.log('[1/4] Initializing SpotAnalyst with lower threshold...');
  const analyst = new SpotAnalyst({
    minConfidence: 0.6,
    portfolioValueUsd: 10_000,
    volatility24h: 0.05,
    verbose: true,
    minConfidenceThreshold: 0.25,  // Lower threshold to 25% for testing
    minLiquidity: 200_000,
  });

  // [2/4] Create mock token with strong fundamentals
  console.log('[2/4] Creating mock token with strong fundamentals...');
  const mockToken: ApprovedToken = {
    symbol: 'SOL',
    address: 'So11111111111111111111111111111111111111112',
    marketCap: 85_000_000_000,  // $85B
    liquidity: 500_000_000,      // $500M
    volume24h: 2_500_000_000,    // $2.5B
    holders: 5_000_000,
    age: 1200,  // 3+ years
    tier: 1,    // Tier 1 token
    dexes: ['JUPITER', 'RAYDIUM', 'ORCA'],
    verified: true,
    approvedAt: Date.now(),
  };

  // [3/4] Create market data with STRONG BUY signals
  console.log('[3/4] Creating market data with strong buy signals...');
  const now = Date.now();
  
  // Create price data showing a dip (buy opportunity)
  const prices: number[] = [];
  for (let i = 0; i < 200; i++) {
    if (i < 150) {
      prices.push(150 + Math.sin(i / 20) * 10);  // Normal range
    } else {
      prices.push(150 - (i - 150) * 0.5);  // Recent dip
    }
  }
  
  const mockMarketData: TokenMarketData = {
    // OHLCV data showing recent dip
    prices,
    volumes: Array.from({ length: 200 }, () => 2_000_000_000 + Math.random() * 500_000_000),
    timestamps: Array.from({ length: 200 }, (_, i) => now - (199 - i) * 24 * 60 * 60 * 1000),

    // Current data - at a dip
    currentPrice: 135.50,  // Down from recent highs
    currentVolume: 2_800_000_000,  // High volume
    
    // Sentiment - VERY POSITIVE
    sentimentScore: 0.75,  // Strong positive sentiment
    socialVolume: 150_000,
    influencerMentions: 25,
    
    // SOL market context - BULLISH
    solPrices: Array.from({ length: 200 }, (_, i) => 140 + Math.sin(i / 15) * 8),
    currentSolPrice: 148,
    
    // Fundamental data - STRONG
    marketCap: 85_000_000_000,
    liquidity: 500_000_000,
    holders: 5_000_000,
    tokenAge: 1200,
    topHolderShare: 0.08,  // Well distributed
    whaleActivity: 0.15,   // Some whale buying
    sectorPerformance: 0.12,  // Sector doing well
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
      volatility24h: 0.04,  // Low volatility
    });

    if (results.length === 0) {
      console.log('\n⚠️  No opportunities found\n');
      return;
    }

    const result = results[0];
    
    console.log('\n============================================================');
    console.log('ANALYSIS RESULT');
    console.log('============================================================');
    console.log(`Token: ${result.token.symbol}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log(`Expected Return: ${(result.expectedReturn * 100).toFixed(1)}%`);
    console.log(`Risk Score: ${result.riskScore}/10`);
    console.log(`Risk-Adjusted Return: ${(result.riskAdjustedReturn * 100).toFixed(1)}%`);
    console.log(`Approved: ${result.approved ? '✅' : '❌'}`);
    if (result.rejectReason) {
      console.log(`Reject Reason: ${result.rejectReason}`);
    }
    console.log('\nTrading Info:');
    console.log(`  Direction: ${result.trading.direction}`);
    console.log(`  Leverage: ${result.trading.leverage}x`);
    console.log(`  Entry: $${result.trading.entryPrice.toFixed(2)}`);
    console.log(`  TP1: $${result.trading.tp1.price.toFixed(2)} (+${result.trading.tp1.percentage}%)`);
    console.log(`  TP2: $${result.trading.tp2.price.toFixed(2)} (+${result.trading.tp2.percentage}%)`);
    console.log(`  TP3: $${result.trading.tp3.price.toFixed(2)} (+${result.trading.tp3.percentage}%)`);
    console.log(`  Stop Loss: $${result.trading.stopLoss.price.toFixed(2)} (${result.trading.stopLoss.percentage}%)`);
    console.log(`  Position Size: $${result.trading.positionSizeUsd.toFixed(2)}`);
    console.log(`  Expected Return: $${result.trading.expectedReturnUsd.toFixed(2)}`);
    console.log(`  Max Loss: $${result.trading.maxLossUsd.toFixed(2)}`);
    console.log(`  R/R Ratio: ${result.trading.riskRewardRatio.toFixed(1)}:1`);
    console.log('============================================================\n');

    console.log('✅ Test completed successfully!\n');
  } catch (error: any) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

testSpotAnalystApproved();

