#!/usr/bin/env npx tsx
/**
 * Test new API integrations - Raydium, Orca, Meteora
 */

import { config } from 'dotenv';
import { resolve } from 'path';

// Load .env
config({ path: resolve(process.cwd(), '.env') });

async function fetchWithTimeout(url: string, timeoutMs = 20000): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { signal: controller.signal });
    clearTimeout(id);
    return resp;
  } catch (e) {
    clearTimeout(id);
    throw e;
  }
}

async function testRaydium() {
  console.log('\n=== RAYDIUM TEST ===');
  const start = Date.now();
  try {
    const resp = await fetchWithTimeout('https://api.raydium.io/v2/main/pairs', 25000);
    console.log('Status:', resp.status);
    const pairs = await resp.json() as Array<{ name: string; liquidity: number; price: number }>;
    console.log('Total pairs:', pairs.length);
    // Find SOL/USDC
    const solUsdc = pairs.find(p => p.name === 'SOL/USDC');
    if (solUsdc) {
      console.log('SOL/USDC:', '$' + solUsdc.price?.toFixed(2), 'Liq:', '$' + (solUsdc.liquidity / 1e6).toFixed(2) + 'M');
    }
    console.log('Time:', Date.now() - start, 'ms');
  } catch (e) {
    console.error('Error:', e);
    console.log('Time:', Date.now() - start, 'ms');
  }
}

async function testOrca() {
  console.log('\n=== ORCA TEST ===');
  const start = Date.now();
  try {
    const resp = await fetchWithTimeout('https://api.orca.so/v1/whirlpool/list', 25000);
    console.log('Status:', resp.status);
    const data = await resp.json() as { whirlpools: Array<{ tokenA: { symbol: string }; tokenB: { symbol: string }; price: number; tvl: number }> };
    console.log('Total pools:', data.whirlpools?.length);
    // Find SOL/USDC
    const solUsdc = data.whirlpools?.find(p => p.tokenA?.symbol === 'SOL' && p.tokenB?.symbol === 'USDC');
    if (solUsdc) {
      console.log('SOL/USDC:', '$' + solUsdc.price?.toFixed(2), 'TVL:', '$' + (solUsdc.tvl / 1e6).toFixed(2) + 'M');
    }
    console.log('Time:', Date.now() - start, 'ms');
  } catch (e) {
    console.error('Error:', e);
    console.log('Time:', Date.now() - start, 'ms');
  }
}

async function testMeteora() {
  console.log('\n=== METEORA TEST ===');
  const start = Date.now();
  try {
    const resp = await fetchWithTimeout('https://dlmm-api.meteora.ag/pair/all', 25000);
    console.log('Status:', resp.status);
    const pairs = await resp.json() as Array<{ name: string; liquidity: string; current_price: number }>;
    console.log('Total pairs:', pairs.length);
    // Find SOL pair
    const solPair = pairs.find(p => p.name?.includes('SOL') && parseFloat(p.liquidity) > 10000);
    if (solPair) {
      console.log('First SOL pair:', solPair.name, 'Liq:', '$' + (parseFloat(solPair.liquidity) / 1e3).toFixed(1) + 'K');
    }
    console.log('Time:', Date.now() - start, 'ms');
  } catch (e) {
    console.error('Error:', e);
    console.log('Time:', Date.now() - start, 'ms');
  }
}

async function main() {
  console.log('New API Integration Tests\n');
  
  // Run sequentially to avoid overwhelming
  await testRaydium();
  await testOrca();
  await testMeteora();
  
  console.log('\n=== DONE ===');
}

main().catch(console.error);

