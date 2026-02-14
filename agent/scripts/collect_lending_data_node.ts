/**
 * Real On-Chain Lending Data Collector (Node.js/TypeScript)
 *
 * This script collects REAL on-chain data from Kamino Finance API
 * and exports to CSV for Python training.
 *
 * NO MOCKS, NO PLACEHOLDERS, NO DUMMY DATA - 100% REAL ON-CHAIN DATA
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Kamino API endpoint
const KAMINO_API = 'https://api.kamino.finance';

// Supported assets (from lending_params.yaml)
const SUPPORTED_ASSETS = ['SOL', 'USDC', 'USDT', 'mSOL', 'stSOL', 'jitoSOL', 'bSOL', 'JUP', 'JLP', 'BONK', 'WIF'];

interface LendingDataPoint {
    timestamp: string;
    protocol: string;
    asset: string;
    supply_apy: number;
    borrow_apy: number;
    utilization_rate: number;
    total_supply: number;
    total_borrow: number;
    available_liquidity: number;
    protocol_tvl_usd: number;
    market_name?: string;
}

/**
 * Collect data from Kamino Finance using their public API
 */
async function collectKaminoData(): Promise<LendingDataPoint[]> {
    console.log('\n📊 Collecting Kamino REAL on-chain data...');
    const dataPoints: LendingDataPoint[] = [];

    try {
        // Use Kamino API (REAL API - documented at https://github.com/Kamino-Finance/kamino-api-docs)
        const response = await fetch(`${KAMINO_API}/kamino-market?env=mainnet-beta`);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Kamino API error ${response.status}: ${errorText}`);
        }

        const markets = await response.json();

        if (!Array.isArray(markets)) {
            throw new Error('Unexpected response format from Kamino API');
        }

        for (const market of markets) {
            const marketName = market.name || 'Unknown';
            const reserves = market.reserves || [];
            const marketTvl = parseFloat(market.totalSupplyUsd || '0');

            for (const reserve of reserves) {
                const assetSymbol = reserve.symbol || 'UNKNOWN';

                // Only collect data for supported assets
                if (!SUPPORTED_ASSETS.includes(assetSymbol)) {
                    continue;
                }

                const supplyApy = parseFloat(reserve.supplyApy || '0');
                const borrowApy = parseFloat(reserve.borrowApy || '0');
                const totalSupply = parseFloat(reserve.totalSupply || '0');
                const totalBorrow = parseFloat(reserve.totalBorrow || '0');
                const utilization = totalSupply > 0 ? totalBorrow / totalSupply : 0;

                dataPoints.push({
                    timestamp: new Date().toISOString(),
                    protocol: 'kamino',
                    asset: assetSymbol,
                    supply_apy: supplyApy,
                    borrow_apy: borrowApy,
                    utilization_rate: utilization,
                    total_supply: totalSupply,
                    total_borrow: totalBorrow,
                    available_liquidity: totalSupply - totalBorrow,
                    protocol_tvl_usd: marketTvl,
                    market_name: marketName,
                });

                console.log(`   ✓ ${assetSymbol}: APY=${(supplyApy * 100).toFixed(2)}%, Util=${(utilization * 100).toFixed(1)}%, TVL=$${marketTvl.toLocaleString()}`);
            }
        }

        console.log(`   ✅ Collected ${dataPoints.length} data points from Kamino`);
    } catch (error) {
        console.error(`   ✗ Kamino collection error:`, error);
    }

    return dataPoints;
}

/**
 * Collect data from Solend using their public API
 */
async function collectSolendData(): Promise<LendingDataPoint[]> {
    console.log('\n📊 Collecting Solend REAL on-chain data...');
    const dataPoints: LendingDataPoint[] = [];

    try {
        // Get all markets
        const marketsResponse = await fetch('https://api.solend.fi/v1/markets?scope=all');

        if (!marketsResponse.ok) {
            throw new Error(`Solend API error: ${marketsResponse.status}`);
        }

        const marketsData = await marketsResponse.json();
        const markets = marketsData.results || [];

        // Focus on the main market
        const mainMarket = markets.find((m: any) => m.isPrimary) || markets[0];

        if (!mainMarket) {
            console.log('   ⚠️  No markets found');
            return dataPoints;
        }

        console.log(`   📍 Using market: ${mainMarket.name} (${mainMarket.address})`);

        // Note: Solend API requires specific reserve IDs, which we don't have yet
        // For now, we'll skip Solend and focus on Kamino which has a better API
        console.log('   ⚠️  Solend API requires reserve IDs - skipping for now');
        console.log('   💡 Will add Solend support once we have reserve addresses');

    } catch (error) {
        console.error(`   ✗ Solend collection error:`, error);
    }

    return dataPoints;
}

/**
 * Main data collection function
 */
async function main() {
    console.log('🚀 Starting REAL on-chain lending data collection...');
    console.log(`📡 Kamino API: ${KAMINO_API}`);
    console.log(`📋 Supported assets: ${SUPPORTED_ASSETS.join(', ')}`);

    // Collect data from all protocols
    const allData: LendingDataPoint[] = [];

    const kaminoData = await collectKaminoData();
    allData.push(...kaminoData);

    const solendData = await collectSolendData();
    allData.push(...solendData);


    if (allData.length === 0) {
        console.error('\n❌ No data collected! Check API endpoints and network connection.');
        process.exit(1);
    }

    console.log(`\n✅ Collected ${allData.length} data points from ${new Set(allData.map(d => d.protocol)).size} protocol(s)`);

    // Save to CSV
    const outputDir = path.join(__dirname, '..', 'data', 'lending');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    const outputFile = path.join(outputDir, `lending_real_${timestamp}_${Date.now()}.csv`);

    // Create directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    // Convert to CSV
    const headers = Object.keys(allData[0]).join(',');
    const rows = allData.map(d => Object.values(d).map(v =>
        typeof v === 'string' && v.includes(',') ? `"${v}"` : v
    ).join(','));
    const csv = [headers, ...rows].join('\n');

    fs.writeFileSync(outputFile, csv);
    console.log(`💾 Saved to: ${outputFile}`);
    console.log(`📊 Total rows: ${allData.length}`);
    console.log(`\n✨ Data collection complete!`);
}

// Run the script
main().catch((error) => {
    console.error('\n❌ Fatal error:', error);
    process.exit(1);
});

