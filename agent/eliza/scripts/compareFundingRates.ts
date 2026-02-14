#!/usr/bin/env npx tsx
/**
 * Compare Funding Rates: Drift vs Adrena
 *
 * Shows side-by-side comparison for common markets:
 * - SOL-PERP
 * - BTC-PERP
 * - ETH-PERP (Drift only)
 *
 * IMPORTANT NOTES:
 * - Drift funding rate is hourly (paid every hour)
 * - Adrena borrow rate is annual (APR)
 * - Drift's API /contracts returns hourly rate as decimal (e.g., 0.001248577 = 0.1248% per hour)
 * - Adrena's API returns annual rate as decimal (e.g., 0.0080052 = 0.80% APR)
 */

async function main() {
  console.log('\n🔄 Comparing Funding Rates: Drift vs Adrena\n');
  console.log('='.repeat(80));
  console.log('NOTE: Drift = true funding rate (hourly). Adrena = borrow rate (annual).');
  console.log('These are fundamentally different costs - compare with caution.');
  console.log('='.repeat(80));

  // Import clients
  const { DriftClient } = await import('../src/services/perps/driftClient.js');
  const { AdrenaClient } = await import('../src/services/perps/adrenaClient.js');

  // Initialize clients
  const driftClient = new DriftClient({
    rpcUrl: 'https://api.mainnet-beta.solana.com',
    env: 'mainnet-beta',
  });

  const adrenaClient = new AdrenaClient({
    rpcUrl: 'https://api.mainnet-beta.solana.com',
  });

  await Promise.all([
    driftClient.initialize(),
    adrenaClient.initialize(),
  ]);

  // Fetch funding rates from both venues
  console.log('\nFetching funding rates from both venues...\n');

  const [driftRates, adrenaRates] = await Promise.all([
    driftClient.getFundingRates(),
    adrenaClient.getFundingRates(),
  ]);

  // Markets to compare
  const marketsToCompare = ['SOL-PERP', 'BTC-PERP', 'ETH-PERP'];

  // Build comparison table
  console.log('┌' + '─'.repeat(78) + '┐');
  console.log('│' + ' FUNDING RATE COMPARISON '.padStart(52).padEnd(78) + '│');
  console.log('├' + '─'.repeat(14) + '┬' + '─'.repeat(30) + '┬' + '─'.repeat(30) + '┬' + '─'.repeat(2) + '┤');
  console.log('│ Market       │ Drift                        │ Adrena                       │  │');
  console.log('│              │ Hourly       APR             │ Hourly       APR             │Δ │');
  console.log('├' + '─'.repeat(14) + '┼' + '─'.repeat(30) + '┼' + '─'.repeat(30) + '┼' + '─'.repeat(2) + '┤');

  const results: {
    market: string;
    driftHourly: number;
    driftApr: number;
    adrenaHourly: number;
    adrenaApr: number;
    spread: number;
  }[] = [];

  for (const market of marketsToCompare) {
    const driftRate = driftRates.find(r => r.market === market);
    const adrenaRate = adrenaRates.find(r => r.market === market);

    const driftHourly = driftRate?.rate || 0;
    const driftApr = driftRate?.annualizedRate || 0;
    const adrenaHourly = adrenaRate?.rate || 0;
    const adrenaApr = adrenaRate?.annualizedRate || 0;

    // Spread = difference in APR (absolute)
    const spread = Math.abs(driftApr - adrenaApr);

    results.push({ market, driftHourly, driftApr, adrenaHourly, adrenaApr, spread });

    // Format values
    const driftHourlyStr = driftHourly !== 0 
      ? `${(driftHourly * 100).toFixed(6)}%` 
      : 'N/A';
    const driftAprStr = driftApr !== 0 
      ? `${(driftApr * 100).toFixed(2)}%` 
      : 'N/A';
    const adrenaHourlyStr = adrenaHourly !== 0 
      ? `${(adrenaHourly * 100).toFixed(6)}%` 
      : 'N/A';
    const adrenaAprStr = adrenaApr !== 0 
      ? `${(adrenaApr * 100).toFixed(2)}%` 
      : 'N/A';

    // Determine which is better for longs (lower funding = better)
    let indicator = ' ';
    if (driftApr !== 0 && adrenaApr !== 0) {
      indicator = driftApr < adrenaApr ? 'D' : (adrenaApr < driftApr ? 'A' : '=');
    } else if (driftApr !== 0) {
      indicator = 'D';
    } else if (adrenaApr !== 0) {
      indicator = 'A';
    }

    console.log(
      `│ ${market.padEnd(12)} │ ${driftHourlyStr.padEnd(12)} ${driftAprStr.padEnd(15)} │ ${adrenaHourlyStr.padEnd(12)} ${adrenaAprStr.padEnd(15)} │${indicator} │`
    );
  }

  console.log('└' + '─'.repeat(14) + '┴' + '─'.repeat(30) + '┴' + '─'.repeat(30) + '┴' + '─'.repeat(2) + '┘');

  // Summary
  console.log('\n📊 SUMMARY\n');
  console.log('Legend: D = Drift better for longs, A = Adrena better for longs, = = Equal\n');

  for (const r of results) {
    if (r.driftApr === 0 && r.adrenaApr === 0) {
      console.log(`${r.market}: No data available`);
      continue;
    }

    if (r.driftApr === 0) {
      console.log(`${r.market}: Only available on Adrena (${(r.adrenaApr * 100).toFixed(2)}% APR)`);
      continue;
    }

    if (r.adrenaApr === 0) {
      console.log(`${r.market}: Only available on Drift (${(r.driftApr * 100).toFixed(2)}% APR)`);
      continue;
    }

    const spreadBps = r.spread * 10000;
    const better = r.driftApr < r.adrenaApr ? 'Drift' : (r.adrenaApr < r.driftApr ? 'Adrena' : 'Equal');
    
    console.log(`${r.market}:`);
    console.log(`  • Drift:  ${(r.driftApr * 100).toFixed(2)}% APR`);
    console.log(`  • Adrena: ${(r.adrenaApr * 100).toFixed(2)}% APR`);
    console.log(`  • Spread: ${spreadBps.toFixed(1)} bps (${(r.spread * 100).toFixed(2)}%)`);
    console.log(`  • Better for longs: ${better}`);
    
    if (spreadBps > 100) {
      console.log(`  ⚡ ARBITRAGE OPPORTUNITY: ${spreadBps.toFixed(0)} bps spread!`);
    }
    console.log();
  }

  // Show additional Adrena markets not on Drift
  console.log('\n📌 Additional Adrena Markets:\n');
  for (const rate of adrenaRates) {
    if (!marketsToCompare.includes(rate.market)) {
      console.log(`  ${rate.market}: ${(rate.annualizedRate * 100).toFixed(2)}% APR`);
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('Done!\n');
}

main().catch(console.error);

