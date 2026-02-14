#!/usr/bin/env npx tsx
/**
 * Test the historical data loader
 */
import { loadHistoricalFundingRates, getAvailableMarkets } from '../src/services/perps/ml/index.js';

async function test() {
  console.log('Testing historical data loader...\n');
  
  const markets = await getAvailableMarkets();
  console.log('Available markets:', markets);
  console.log('');
  
  for (const market of markets) {
    const data = await loadHistoricalFundingRates(undefined, market, 168);
    console.log(`${market}: ${data.length} data points`);
    if (data.length > 0) {
      console.log(`  First: ${data[0].timestamp.toISOString()}, rate: ${(data[0].fundingRate * 100).toFixed(4)}%`);
      console.log(`  Last:  ${data[data.length-1].timestamp.toISOString()}, rate: ${(data[data.length-1].fundingRate * 100).toFixed(4)}%`);
    }
    console.log('');
  }
  
  console.log('✅ Historical data loader working!');
}

test().catch(err => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});

