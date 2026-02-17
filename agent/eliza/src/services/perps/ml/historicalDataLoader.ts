/**
 * Historical Data Loader for Perps ML Agent
 * 
 * Loads historical funding rate data from CSV files to warm up feature buffers.
 * This allows the agent to start making predictions immediately without waiting
 * 168 hours for buffer to fill.
 * 
 * Data source: agent/training/perps/data/funding_rates_latest.csv
 */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../../logger.js';
import type { FundingDataPoint } from './featureExtractor.js';

// Get directory of current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Default path to historical data - try multiple locations
// agent/eliza/src/services/perps/ml -> ../training/perps/data (relative to eliza root)
function findDataFile(): string {
  const candidates = [
    // From agent/eliza root (most common)
    path.resolve(process.cwd(), '../training/perps/data/funding_rates_latest.csv'),
    // From cortex root
    path.resolve(process.cwd(), 'agent/training/perps/data/funding_rates_latest.csv'),
    // Relative to this file (6 levels up from ml/)
    path.resolve(__dirname, '../../../../../../training/perps/data/funding_rates_latest.csv'),
    // Alternative relative paths
    path.resolve(__dirname, '../../../../../training/perps/data/funding_rates_latest.csv'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0]; // Return first candidate as default
}

const DEFAULT_DATA_PATH = findDataFile();

/** CSV row from funding_rates_latest.csv */
interface FundingRateCSVRow {
  market: string;
  timestamp: string;
  slot: string;
  funding_rate_raw: string;
  funding_rate_pct: string;
  oracle_twap: string;
  mark_twap: string;
  cumulative_funding_rate_long: string;
  cumulative_funding_rate_short: string;
  datetime: string;
}

/**
 * Load historical funding rate data from CSV
 * 
 * @param csvPath - Path to the CSV file (defaults to training data)
 * @param market - Market to filter (e.g., 'SOL-PERP')
 * @param minPoints - Minimum data points required (default: 168 for 1 week)
 * @returns Array of FundingDataPoint sorted by timestamp ascending
 */
export async function loadHistoricalFundingRates(
  csvPath: string = DEFAULT_DATA_PATH,
  market: string = 'SOL-PERP',
  minPoints: number = 168
): Promise<FundingDataPoint[]> {
  // Check if file exists
  if (!fs.existsSync(csvPath)) {
    // Try alternative paths
    const altPaths = [
      path.resolve(process.cwd(), 'agent/training/perps/data/funding_rates_latest.csv'),
      path.resolve(process.cwd(), '../training/perps/data/funding_rates_latest.csv'),
      path.resolve(__dirname, '../../../../../../../training/perps/data/funding_rates_latest.csv'),
    ];
    
    for (const altPath of altPaths) {
      if (fs.existsSync(altPath)) {
        csvPath = altPath;
        break;
      }
    }
    
    if (!fs.existsSync(csvPath)) {
      logger.warn('Historical data file not found', { csvPath, tried: altPaths });
      return [];
    }
  }

  logger.info('Loading historical funding rates', { csvPath, market });

  // Read and parse CSV
  const content = fs.readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  
  if (lines.length < 2) {
    logger.warn('CSV file is empty or has no data rows');
    return [];
  }

  // Parse header
  const header = lines[0].split(',');
  const colIndex: Record<string, number> = {};
  header.forEach((col, idx) => colIndex[col.trim()] = idx);

  // Parse data rows
  const dataPoints: FundingDataPoint[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const rowMarket = values[colIndex['market']];
    
    // Filter by market
    if (rowMarket !== market) continue;
    
    try {
      const timestamp = new Date(values[colIndex['datetime']]);
      const fundingRatePct = parseFloat(values[colIndex['funding_rate_pct']]);
      const fundingRateRaw = parseFloat(values[colIndex['funding_rate_raw']]);
      const oraclePrice = parseFloat(values[colIndex['oracle_twap']]);
      const cumFundingLong = parseFloat(values[colIndex['cumulative_funding_rate_long']]);
      const cumFundingShort = parseFloat(values[colIndex['cumulative_funding_rate_short']]);

      if (isNaN(timestamp.getTime()) || isNaN(fundingRatePct)) {
        continue; // Skip invalid rows
      }

      dataPoints.push({
        timestamp,
        fundingRate: fundingRatePct,  // Use percentage rate
        fundingRateRaw,
        oraclePrice: oraclePrice || undefined,
        cumFundingLong: cumFundingLong || undefined,
        cumFundingShort: cumFundingShort || undefined,
      });
    } catch (e) {
      // Skip malformed rows
      continue;
    }
  }

  // Sort by timestamp ascending
  dataPoints.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Take last N points (most recent)
  const result = dataPoints.slice(-Math.max(minPoints, 200));

  logger.info('Loaded historical data', {
    market,
    totalRows: dataPoints.length,
    returned: result.length,
    oldestDate: result[0]?.timestamp.toISOString(),
    newestDate: result[result.length - 1]?.timestamp.toISOString(),
  });

  if (result.length < minPoints) {
    logger.warn('Insufficient historical data', { 
      required: minPoints, 
      available: result.length 
    });
  }

  return result;
}

/**
 * Get available markets from CSV
 */
export async function getAvailableMarkets(
  csvPath: string = DEFAULT_DATA_PATH
): Promise<string[]> {
  if (!fs.existsSync(csvPath)) return [];
  
  const content = fs.readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  const markets = new Set<string>();
  
  for (let i = 1; i < lines.length; i++) {
    const market = lines[i].split(',')[0];
    if (market && market !== 'market') markets.add(market);
  }
  
  return Array.from(markets).sort();
}

