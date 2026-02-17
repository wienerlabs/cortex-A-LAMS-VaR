#!/usr/bin/env ts-node
/**
 * Collect current Kamino lending data using the official Kamino SDK
 * This provides complete data including borrow APY and utilization
 * which is not available in DeFiLlama
 */

import { Connection, PublicKey } from '@solana/web3.js';
import { KaminoMarket } from '@kamino-finance/klend-sdk';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// ES module __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Solana RPC endpoint
const RPC_ENDPOINT = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';

// Kamino Main Market address
const KAMINO_MAIN_MARKET = new PublicKey('7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF');

// Output directory
const DATA_DIR = path.join(__dirname, '..', '..', 'data', 'lending');

interface KaminoReserveData {
  timestamp: string;
  protocol: string;
  asset: string;
  supply_apy: number;
  borrow_apy: number;
  utilization_rate: number;
  tvl_usd: number;
  total_supply: number;
  total_borrows: number;
  available_liquidity: number;
  mint_address: string;
}

async function collectKaminoData(): Promise<KaminoReserveData[]> {
  console.log('\n📊 Collecting Kamino lending data using SDK...');
  
  try {
    // Connect to Solana
    const connection = new Connection(RPC_ENDPOINT, 'confirmed');
    console.log('✅ Connected to Solana RPC');
    
    // Load Kamino market
    const market = await KaminoMarket.load(
      connection,
      KAMINO_MAIN_MARKET,
      undefined,
      undefined,
      undefined
    );
    console.log('✅ Loaded Kamino market');
    
    // Get current slot
    const slot = BigInt(await connection.getSlot());
    console.log(`✅ Current slot: ${slot}`);
    
    // Collect data for all reserves
    const reserves: KaminoReserveData[] = [];
    const marketReserves = market.getReserves();
    
    console.log(`\n📈 Processing ${marketReserves.length} reserves...`);
    
    for (const reserve of marketReserves) {
      try {
        const asset = reserve.getTokenSymbol();
        const supplyAPY = reserve.totalSupplyAPY(slot) * 100;
        const borrowAPY = reserve.totalBorrowAPY(slot) * 100;
        const utilization = reserve.calculateUtilizationRatio() * 100;
        const totalSupply = reserve.getTotalSupply().toNumber();
        const totalBorrows = reserve.getBorrowedAmount().toNumber();
        const availableLiquidity = reserve.getLiquidityAvailableAmount().toNumber();
        
        reserves.push({
          timestamp: new Date().toISOString(),
          protocol: 'kamino',
          asset,
          supply_apy: supplyAPY,
          borrow_apy: borrowAPY,
          utilization_rate: utilization,
          tvl_usd: 0, // Would need price oracle to calculate
          total_supply: totalSupply,
          total_borrows: totalBorrows,
          available_liquidity: availableLiquidity,
          mint_address: reserve.getLiquidityMint().toString(),
        });
        
        console.log(`  ✅ ${asset}: Supply APY=${supplyAPY.toFixed(2)}%, Borrow APY=${borrowAPY.toFixed(2)}%, Util=${utilization.toFixed(2)}%`);
      } catch (error) {
        console.error(`  ❌ Error processing reserve: ${error}`);
      }
    }
    
    console.log(`\n✅ Collected data for ${reserves.length} reserves`);
    return reserves;
    
  } catch (error) {
    console.error('❌ Error collecting Kamino data:', error);
    throw error;
  }
}

async function saveToCSV(data: KaminoReserveData[], filename?: string): Promise<void> {
  if (!filename) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    filename = `kamino_sdk_snapshot_${timestamp}.csv`;
  }
  
  const filepath = path.join(DATA_DIR, filename);
  
  // Ensure directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  
  // Convert to CSV
  const headers = Object.keys(data[0]).join(',');
  const rows = data.map(row => Object.values(row).join(','));
  const csv = [headers, ...rows].join('\n');
  
  fs.writeFileSync(filepath, csv);
  console.log(`\n💾 Saved ${data.length} rows to ${filepath}`);
  console.log(`   File size: ${(fs.statSync(filepath).size / 1024).toFixed(2)} KB`);
}

async function main() {
  try {
    const data = await collectKaminoData();
    await saveToCSV(data);
    console.log('\n✅ Data collection complete!');
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Data collection failed:', error);
    process.exit(1);
  }
}

main();

