/**
 * Test Pump.fun Live API Integration
 * 
 * Fetches real tokens from Pump.fun API and displays filtered results
 */
import dotenv from 'dotenv';
dotenv.config();
import { PumpFunClient } from '../src/services/pumpfun/pumpfunClient.js';
import { filterPumpFunTokens } from '../src/services/pumpfun/pumpfunFilter.js';

async function main() {
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║  PUMP.FUN LIVE API TEST                                   ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  // Check API key
  if (!process.env.PUMPFUN_API_KEY) {
    console.log('❌ ERROR: PUMPFUN_API_KEY not set in .env file');
    console.log('   Please add your Pump.fun API key to agent/eliza/.env\n');
    process.exit(1);
  }

  console.log('✅ PUMPFUN_API_KEY found');
  console.log(`   Key: ${process.env.PUMPFUN_API_KEY.substring(0, 20)}...\n`);

  try {
    // Initialize client
    console.log('📡 Connecting to Pump.fun API...');
    const client = new PumpFunClient();

    // Fetch tokens with pagination (API limits to ~30 per request)
    console.log('🔍 Fetching 300 latest tokens (using pagination)...\n');

    const allTokens = [];
    const batchSize = 30;
    const totalToFetch = 300;

    for (let offset = 0; offset < totalToFetch; offset += batchSize) {
      console.log(`   Fetching batch ${Math.floor(offset / batchSize) + 1}/${Math.ceil(totalToFetch / batchSize)} (offset: ${offset})...`);
      const batch = await client.getTokens(batchSize, offset);

      if (batch.length === 0) {
        console.log(`   No more tokens available (stopped at ${allTokens.length} tokens)\n`);
        break;
      }

      allTokens.push(...batch);

      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const tokens = allTokens;
    console.log(`✅ Fetched: ${tokens.length} tokens from Pump.fun\n`);

    // Debug: Show first token structure
    if (tokens.length > 0) {
      console.log('🔍 DEBUG: First token structure:');
      console.log(JSON.stringify(tokens[0], null, 2));
      console.log('\n');
    }

    // Display sample tokens
    console.log('📊 Sample Tokens (first 5):');
    tokens.slice(0, 5).forEach((token, idx) => {
      console.log(`   ${idx + 1}. ${token.ticker} - ${token.name}`);
      console.log(`      Mint: ${token.coinMint}`);
      console.log(`      Market Cap: $${(token.marketCap / 1000).toFixed(1)}K`);
      console.log(`      Volume: $${token.volume.toFixed(2)}`);
      console.log(`      Holders: ${token.numHolders}`);
      console.log(`      Created: ${new Date(token.creationTime).toLocaleString()}`);
      console.log('');
    });

    // Filter tokens
    console.log('🔍 Filtering tokens (TVL >$10K, holders >50, age >24h)...\n');
    const filtered = await filterPumpFunTokens(tokens);

    console.log(`✅ Filtered: ${filtered.length} tokens passed safety criteria\n`);

    if (filtered.length > 0) {
      console.log('╔═══════════════════════════════════════════════════════════╗');
      console.log('║  APPROVED PUMP.FUN TOKENS                                 ║');
      console.log('╚═══════════════════════════════════════════════════════════╝\n');

      filtered.forEach((token, idx) => {
        const tvlDisplay = `$${(token.tvl / 1000).toFixed(1)}K`;
        const holdersDisplay = token.holderCount >= 1000000 ? '1M+' : token.holderCount.toLocaleString();
        const ageDisplay = `${token.ageHours.toFixed(0)}h`;
        const mcapDisplay = `$${(token.marketCap / 1000).toFixed(1)}K`;

        console.log(`${(idx + 1).toString().padStart(2)}. ${token.symbol.padEnd(12)} | ${token.name.substring(0, 20).padEnd(20)}`);
        console.log(`    TVL: ${tvlDisplay.padEnd(10)} | Market Cap: ${mcapDisplay.padEnd(10)} | Holders: ${holdersDisplay.padEnd(8)} | Age: ${ageDisplay}`);
        console.log(`    Mint: ${token.mint}`);
        
        if (token.twitter) console.log(`    Twitter: ${token.twitter}`);
        if (token.telegram) console.log(`    Telegram: ${token.telegram}`);
        if (token.website) console.log(`    Website: ${token.website}`);
        
        console.log('');
      });

      console.log(`\n✅ Total approved tokens: ${filtered.length}`);
      console.log(`   These tokens are ready for AGGRESSIVE mode trading\n`);
    } else {
      console.log('⚠️  No tokens passed the safety filters');
      console.log('   This could mean:');
      console.log('   - All tokens are too new (<24h)');
      console.log('   - All tokens have low TVL (<$10K)');
      console.log('   - All tokens have few holders (<50)');
      console.log('   - Try again later when more tokens mature\n');
    }

  } catch (error: any) {
    console.log(`\n❌ ERROR: ${error.message}`);
    console.log(`   Stack: ${error.stack}\n`);
    process.exit(1);
  }

  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║  ✅ TEST COMPLETE                                          ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
}

main().catch(console.error);

