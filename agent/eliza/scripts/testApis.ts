#!/usr/bin/env npx tsx
/**
 * Test all API integrations
 */

import { config } from 'dotenv';
import { resolve } from 'path';

// Load .env
config({ path: resolve(process.cwd(), '.env') });

async function testBirdeye() {
  console.log('\n=== BIRDEYE TEST ===');
  const apiKey = process.env.BIRDEYE_API_KEY;
  console.log('API Key:', apiKey ? `SET (${apiKey.substring(0, 10)}...)` : 'NOT SET');
  
  if (!apiKey) return;
  
  const address = 'So11111111111111111111111111111111111111112';
  const url = `https://public-api.birdeye.so/defi/multi_price?list_address=${address}`;
  
  try {
    const resp = await fetch(url, {
      headers: { 'X-API-KEY': apiKey, 'x-chain': 'solana' }
    });
    const data = await resp.json();
    console.log('Status:', resp.status);
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));
  } catch (e) {
    console.error('Error:', e);
  }
}

async function testJupiter() {
  console.log('\n=== JUPITER TEST ===');
  const address = 'So11111111111111111111111111111111111111112';
  const url = `https://api.jup.ag/price/v2?ids=${address}`;
  
  try {
    const resp = await fetch(url);
    const data = await resp.json();
    console.log('Status:', resp.status);
    console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));
  } catch (e) {
    console.error('Error:', e);
  }
}

async function testCoinGecko() {
  console.log('\n=== COINGECKO TEST ===');
  const apiKey = process.env.COINGECKO_API_KEY;
  console.log('API Key:', apiKey ? 'SET' : 'NOT SET');
  
  const url = 'https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd';
  const headers: Record<string, string> = {};
  if (apiKey) headers['x-cg-demo-api-key'] = apiKey;
  
  try {
    const resp = await fetch(url, { headers });
    const data = await resp.json();
    console.log('Status:', resp.status);
    console.log('Response:', JSON.stringify(data, null, 2));
  } catch (e) {
    console.error('Error:', e);
  }
}

async function testDexScreener() {
  console.log('\n=== DEXSCREENER TEST ===');
  const address = 'So11111111111111111111111111111111111111112';
  const url = `https://api.dexscreener.com/latest/dex/tokens/${address}`;
  
  try {
    const resp = await fetch(url);
    const data = await resp.json() as { pairs: Array<{ priceUsd: string; dexId: string }> };
    console.log('Status:', resp.status);
    console.log('Pairs:', data.pairs?.length || 0);
    if (data.pairs?.[0]) {
      console.log('First pair:', data.pairs[0].dexId, '$' + data.pairs[0].priceUsd);
    }
  } catch (e) {
    console.error('Error:', e);
  }
}

async function main() {
  console.log('API Integration Tests\n');
  console.log('Environment loaded from:', resolve(process.cwd(), '.env'));
  
  await testBirdeye();
  await testJupiter();
  await testCoinGecko();
  await testDexScreener();
  
  console.log('\n=== DONE ===');
}

main().catch(console.error);

