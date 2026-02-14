#!/usr/bin/env npx tsx
/**
 * Debug Funding Rates
 * Tests raw API responses from Drift and Adrena
 */

const DRIFT_DATA_API = 'https://data.api.drift.trade';
const ADRENA_API = 'https://datapi.adrena.xyz';

// Drift precision constants
const FUNDING_RATE_PRECISION = 1e9;
const PRICE_PRECISION = 1e6;

async function testDriftFundingRates() {
  console.log('\n========== DRIFT FUNDING RATES ==========\n');

  // Test 1: Funding rates with marketIndex (required param)
  console.log('1. Testing Drift fundingRates?marketIndex=0 (SOL-PERP)...');
  try {
    const response = await fetch(`${DRIFT_DATA_API}/fundingRates?marketIndex=0`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json() as { fundingRates?: any[] };
      console.log(`   Response type: ${typeof data}`);
      console.log(`   Has fundingRates array: ${!!data.fundingRates}`);

      // Get the latest funding rate
      const fundingRates = data.fundingRates || [];
      console.log(`   Funding rates count: ${fundingRates.length}`);

      if (fundingRates.length > 0) {
        const latest = fundingRates[0];
        const fundingRateRaw = parseInt(latest.fundingRate || '0');
        const fundingRateParsed = fundingRateRaw / FUNDING_RATE_PRECISION;
        const hourlyRate = fundingRateParsed * 100; // as percentage
        const annualized = hourlyRate * 24 * 365;

        console.log(`\n   Latest funding rate:`);
        console.log(`     Raw value: ${latest.fundingRate}`);
        console.log(`     Parsed (decimal): ${fundingRateParsed.toFixed(10)}`);
        console.log(`     Hourly rate: ${hourlyRate.toFixed(6)}%`);
        console.log(`     Annualized: ${annualized.toFixed(2)}%`);
        console.log(`     Timestamp: ${latest.ts}`);
        console.log(`     Oracle Price TWAP: ${parseInt(latest.oraclePriceTwap) / PRICE_PRECISION}`);
        console.log(`     Mark Price TWAP: ${parseInt(latest.markPriceTwap) / PRICE_PRECISION}`);
      }
    } else {
      const text = await response.text();
      console.log(`   Error response: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 2: Funding rates with marketName
  console.log('\n\n2. Testing Drift fundingRates?marketName=SOL-PERP...');
  try {
    const response = await fetch(`${DRIFT_DATA_API}/fundingRates?marketName=SOL-PERP`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Records found: ${Array.isArray(data) ? data.length : 'N/A'}`);

      const samples = Array.isArray(data) ? data.slice(0, 2) : [];
      for (const fr of samples) {
        console.log(`\n   ${JSON.stringify(fr, null, 2).slice(0, 400)}`);
      }
    } else {
      const text = await response.text();
      console.log(`   Error response: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 3: Get contracts (markets info) - this has pre-calculated funding_rate
  console.log('\n\n3. Testing Drift /contracts (has pre-calculated funding_rate)...');
  try {
    const response = await fetch(`${DRIFT_DATA_API}/contracts`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json() as { contracts?: any[] };
      const contracts = data.contracts || [];
      console.log(`   Contracts count: ${contracts.length}`);

      // Find SOL-PERP
      const solContract = contracts.find((c: any) =>
        c.ticker_id === 'SOL-PERP' || c.contract_index === 0
      );

      if (solContract) {
        console.log(`\n   SOL-PERP contract:`);
        console.log(`     Ticker: ${solContract.ticker_id}`);
        console.log(`     Last Price: ${solContract.last_price}`);
        console.log(`     Index Price: ${solContract.index_price}`);
        console.log(`     Funding Rate (from API): ${solContract.funding_rate}`);
        console.log(`     Open Interest: ${solContract.open_interest}`);

        // Calculate annualized from the funding_rate
        const fundingRate = parseFloat(solContract.funding_rate || '0');
        const annualized = fundingRate * 24 * 365 * 100;
        console.log(`     Funding Rate (parsed): ${fundingRate}`);
        console.log(`     Annualized: ${annualized.toFixed(2)}%`);
      }

      // Show a few more contracts
      console.log(`\n   Sample contracts with funding rates:`);
      for (const c of contracts.slice(0, 5)) {
        console.log(`     ${c.ticker_id}: funding_rate=${c.funding_rate}`);
      }
    } else {
      const text = await response.text();
      console.log(`   Error response: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 4: Try perpMarkets endpoint
  console.log('\n\n4. Testing Drift /perpMarkets...');
  try {
    const response = await fetch(`${DRIFT_DATA_API}/perpMarkets`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json() as any;
      console.log(`   Response type: ${typeof data}`);
      console.log(`   Is array: ${Array.isArray(data)}`);

      const markets = Array.isArray(data) ? data : data.markets || [];
      console.log(`   Markets count: ${markets.length}`);

      if (markets.length > 0) {
        const solMarket = markets.find((m: any) =>
          m.symbol === 'SOL-PERP' || m.marketIndex === 0
        );
        if (solMarket) {
          console.log(`\n   SOL-PERP market:`);
          console.log(`     ${JSON.stringify(solMarket, null, 2).slice(0, 1000)}`);
        } else {
          console.log(`   First market: ${JSON.stringify(markets[0], null, 2).slice(0, 500)}`);
        }
      }
    } else {
      const text = await response.text();
      console.log(`   Error response: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }
}

async function testAdrenaAPI() {
  console.log('\n\n========== ADRENA API ==========\n');

  // Test 1: Last trading prices (this works based on the client code)
  console.log('1. Testing Adrena /last-trading-prices...');
  try {
    const response = await fetch(`${ADRENA_API}/last-trading-prices`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 600)}`);
    } else {
      const text = await response.text();
      console.log(`   Error response: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 2: Custody info with borrow_rate (from Swagger docs)
  console.log('\n\n2. Testing Adrena /custodyinfo?borrow_rate=true&limit=1...');
  try {
    const response = await fetch(`${ADRENA_API}/custodyinfo?borrow_rate=true&limit=1`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json() as { success?: boolean; data?: { borrow_rate?: Record<string, string[]> } };
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 1200)}`);

      // Parse borrow rates - these are already annual rates (APR)
      // Per Adrena docs: SOL/WBTC: 0% ~ 80.5% APR, BONK: 0% ~ 150.7% APR
      if (data.success && data.data?.borrow_rate) {
        console.log('\n   Parsed borrow rates (already annual APR):');
        for (const [custody, rates] of Object.entries(data.data.borrow_rate)) {
          const rate = parseFloat((rates as string[])[0] || '0');
          const aprPercent = rate * 100;
          console.log(`     ${custody.slice(0, 8)}...: ${rate} (${aprPercent.toFixed(2)}% APR)`);
        }
      }
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 3: Pool info with main-pool
  console.log('\n\n3. Testing Adrena /poolinfo?pool_name=main-pool&limit=1...');
  try {
    const response = await fetch(`${ADRENA_API}/poolinfo?pool_name=main-pool&limit=1`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 800)}`);
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 4: Custody info with all fields
  console.log('\n\n4. Testing Adrena /custodyinfo with all fields...');
  try {
    const response = await fetch(`${ADRENA_API}/custodyinfo?pool_name=main-pool&limit=1`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 1500)}`);
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 5: Check if there's a funding rate endpoint
  console.log('\n\n5. Testing Adrena /funding-rates...');
  try {
    const response = await fetch(`${ADRENA_API}/funding-rates`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 800)}`);
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 6: Check positions endpoint
  console.log('\n\n6. Testing Adrena /positions...');
  try {
    const response = await fetch(`${ADRENA_API}/positions?limit=1`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 800)}`);
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 7: Check custody info with price to get symbol mapping
  console.log('\n\n7. Testing Adrena /custodyinfo with price and borrow_rate...');
  try {
    const response = await fetch(`${ADRENA_API}/custodyinfo?pool_name=main-pool&borrow_rate=true&price=true&limit=1`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 2000)}`);
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }

  // Test 8: Check poolinfo with open interest
  console.log('\n\n8. Testing Adrena /poolinfo with open interest...');
  try {
    const response = await fetch(`${ADRENA_API}/poolinfo?pool_name=main-pool&open_interest_long_usd=true&open_interest_short_usd=true&limit=1`);
    console.log(`   Status: ${response.status}`);

    if (response.ok) {
      const data = await response.json();
      console.log(`   Response: ${JSON.stringify(data, null, 2).slice(0, 1000)}`);
    } else {
      const text = await response.text();
      console.log(`   Error: ${text.slice(0, 300)}`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }
}

async function testAdrenaClient() {
  console.log('\n\n========== ADRENA CLIENT TEST ==========\n');

  try {
    // Import and test the actual client
    const { AdrenaClient } = await import('../src/services/perps/adrenaClient.js');

    const client = new AdrenaClient({
      rpcUrl: 'https://api.mainnet-beta.solana.com',
    });

    await client.initialize();

    console.log('1. Testing AdrenaClient.getMarketStats()...');
    const stats = await client.getMarketStats();
    console.log(`   Found ${stats.length} markets:`);
    for (const s of stats) {
      console.log(`   ${s.market}:`);
      console.log(`     Mark Price: $${s.markPrice.toFixed(2)}`);
      console.log(`     Funding Rate (hourly): ${(s.fundingRate * 100).toFixed(4)}%`);
      console.log(`     Funding Rate (APR): ${(s.fundingRateApr * 100).toFixed(2)}%`);
    }

    console.log('\n2. Testing AdrenaClient.getFundingRates()...');
    const rates = await client.getFundingRates();
    console.log(`   Found ${rates.length} funding rates:`);
    for (const r of rates) {
      console.log(`   ${r.market}: rate=${r.rate.toFixed(6)}, annualized=${(r.annualizedRate * 100).toFixed(2)}%`);
    }
  } catch (error) {
    console.log(`   Error: ${error}`);
  }
}

async function main() {
  console.log('🔍 Debugging Funding Rate APIs\n');
  console.log('Testing raw API responses to identify why rates return 0...\n');

  await testDriftFundingRates();
  await testAdrenaAPI();
  await testAdrenaClient();

  console.log('\n\n========== DONE ==========\n');
}

main().catch(console.error);

