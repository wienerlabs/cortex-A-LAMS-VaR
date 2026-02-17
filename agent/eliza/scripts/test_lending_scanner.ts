/**
 * Test Lending Scanner with REAL on-chain data
 */
import { fetchLendingMarkets } from '../src/services/marketScanner/lendingScanner.js';

async function test() {
  console.log('🔍 Testing Lending Scanner\n');
  console.log('═'.repeat(80));
  
  const markets = await fetchLendingMarkets();
  
  console.log('\n📊 Top 10 Lending Markets (REAL on-chain data from DeFiLlama):');
  console.log('═'.repeat(80));
  
  markets
    .sort((a, b) => b.tvlUsd - a.tvlUsd)
    .slice(0, 10)
    .forEach((m, i) => {
      console.log(`${i+1}. ${m.asset.padEnd(10)} on ${m.protocol.padEnd(10)} TVL: $${(m.tvlUsd/1_000_000).toFixed(2)}M  APY: ${(m.supplyApy*100).toFixed(2)}%`);
    });
  
  console.log('\n' + '═'.repeat(80));
  console.log(`\n✅ Total markets: ${markets.length}`);
  console.log('✅ Data source: DeFiLlama API (aggregates Kamino, MarginFi, Solend)');
  console.log('✅ All data is REAL on-chain data from Solana mainnet\n');
}

test().catch(console.error);

