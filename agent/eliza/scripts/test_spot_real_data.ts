/**
 * Test SpotAnalyst with Real Market Data (Fast Version)
 * Directly fetches market data for approved tokens and runs SpotAnalyst
 */

import dotenv from 'dotenv';
import { SpotAnalyst } from '../src/agents/analysts/SpotAnalyst.js';
import { fetchSpotTokens } from '../src/services/marketScanner/spotScanner.js';

dotenv.config();

async function testSpotRealData() {
  console.log('\n============================================================');
  console.log('SPOT ANALYST TEST - REAL MARKET DATA');
  console.log('============================================================\n');

  // [1/3] Initialize SpotAnalyst
  console.log('[1/3] Initializing SpotAnalyst...');
  const analyst = new SpotAnalyst({
    minConfidence: 0.6,
    portfolioValueUsd: 10_000,
    volatility24h: 0.05,
    verbose: true,
    minConfidenceThreshold: 0.45,  // 45% threshold (production setting)
    minLiquidity: 200_000,
  });

  // [2/3] Fetch real spot tokens with market data
  console.log('\n[2/3] Fetching real spot tokens from Solana network...');
  console.log('(This will discover tokens from Birdeye and fetch real market data)\n');

  const spotTokens = await fetchSpotTokens();

  if (spotTokens.length === 0) {
    console.log('❌ No spot tokens found. Cannot run analysis.\n');
    console.log('Make sure BIRDEYE_API_KEY and HELIUS_API_KEY are set in .env\n');
    return;
  }

  console.log(`\n✅ Found ${spotTokens.length} tokens with real market data\n`);

  // [3/3] Run SpotAnalyst
  console.log('[3/3] Running SpotAnalyst...\n');
  console.log('='.repeat(80));

  // Initialize ML model
  await analyst.initialize();

  try {
    // Build market data map from spotTokens
    const marketDataMap = new Map();
    spotTokens.forEach((token: any) => {
      if (token.marketData) {
        marketDataMap.set(token.address, token.marketData);
      }
    });

    const results = await analyst.analyze({
      tokens: spotTokens,
      marketData: marketDataMap,
      volatility24h: 0.05,
    });

    console.log('\n' + '='.repeat(80));
    console.log('✅ Analysis complete!\n');

    // Show approved opportunities in detail
    const approved = results.filter(r => r.approved);
    if (approved.length > 0) {
      console.log(`\n🎯 APPROVED OPPORTUNITIES (${approved.length}):\n`);
      approved.forEach((result, index) => {
        console.log(`\n[${ index + 1}] ${result.token.symbol}`);
        console.log('─'.repeat(60));
        console.log(`Address:          ${result.token.address}`);
        console.log(`Price:            $${result.trading.entryPrice.toFixed(6)}`);
        console.log(`Market Cap:       $${(result.token.marketCap / 1_000_000).toFixed(2)}M`);
        console.log(`Liquidity:        $${(result.token.liquidity / 1_000_000).toFixed(2)}M`);
        console.log(`Volume 24h:       $${(result.token.volume24h / 1_000_000).toFixed(2)}M`);
        console.log('');
        console.log(`Direction:        ${result.trading.direction}`);
        console.log(`Leverage:         ${result.trading.leverage}x`);
        console.log(`Confidence:       ${(result.confidence * 100).toFixed(1)}%`);
        console.log(`Position Size:    $${result.trading.positionSizeUsd.toFixed(2)}`);
        console.log('');
        console.log(`ENTRY:            $${result.trading.entryPrice.toFixed(6)}`);
        console.log(`TP1 (+12%):       $${result.trading.tp1.price.toFixed(6)} → Exit 40%`);
        console.log(`TP2 (+25%):       $${result.trading.tp2.price.toFixed(6)} → Exit 35%`);
        console.log(`TP3 (+40%):       $${result.trading.tp3.price.toFixed(6)} → Exit 25%`);
        console.log(`STOP LOSS (-8%):  $${result.trading.stopLoss.price.toFixed(6)}`);
        console.log('');
        console.log(`Expected Return:  $${result.trading.expectedReturnUsd.toFixed(2)}`);
        console.log(`Max Loss:         $${result.trading.maxLossUsd.toFixed(2)}`);
        console.log(`Risk Score:       ${result.riskScore}/10`);
        console.log(`R/R Ratio:        ${result.trading.riskRewardRatio.toFixed(1)}:1`);
        console.log('─'.repeat(60));
      });
    } else {
      console.log('\n⚠️  No opportunities approved at 45% threshold\n');
    }

  } catch (error: any) {
    console.error('\n❌ Analysis failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

testSpotRealData();

