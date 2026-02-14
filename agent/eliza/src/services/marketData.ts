/**
 * Market Data Service
 * 
 * Provides dynamic market data instead of hardcoded values:
 * - Live SOL price from Binance
 * - Real volatility calculation from price history
 * - Dynamic gas fee estimation
 * 
 * Created: 2026-01-07
 * Reason: Replace hardcoded values (SOL=$200, volatility=5%, gas=0.002)
 */

import { fetchBinancePrices } from './marketScanner/cexFetcher.js';
import { logger } from './logger.js';

// Cache for market data to avoid excessive API calls
interface MarketDataCache {
  solPrice: number;
  solPriceTimestamp: number;
  volatility24h: number;
  volatilityTimestamp: number;
}

const cache: MarketDataCache = {
  solPrice: 200, // Fallback default
  solPriceTimestamp: 0,
  volatility24h: 0.05, // Fallback default (5%)
  volatilityTimestamp: 0,
};

// Cache TTL: 30 seconds for price, 5 minutes for volatility
const PRICE_CACHE_TTL = 30 * 1000;
const VOLATILITY_CACHE_TTL = 5 * 60 * 1000;

// Default gas fee in SOL (configurable)
export const DEFAULT_GAS_SOL = 0.002;

/**
 * Get live SOL price from Binance
 * Returns cached value if fresh, otherwise fetches new price
 */
export async function getSolPrice(): Promise<number> {
  const now = Date.now();
  
  // Return cached if fresh
  if (now - cache.solPriceTimestamp < PRICE_CACHE_TTL) {
    return cache.solPrice;
  }
  
  try {
    const prices = await fetchBinancePrices(['SOL']);
    const solPrice = prices.find(p => p.symbol === 'SOL');
    
    if (solPrice && solPrice.price > 0) {
      cache.solPrice = solPrice.price;
      cache.solPriceTimestamp = now;
      logger.debug('[MarketData] SOL price updated', { price: solPrice.price });
      return solPrice.price;
    }
  } catch (error) {
    logger.warn('[MarketData] Failed to fetch SOL price, using cached', { 
      cached: cache.solPrice,
      error: error instanceof Error ? error.message : 'Unknown'
    });
  }
  
  return cache.solPrice;
}

/**
 * Calculate gas fee in USD using live SOL price
 * @param gasSol - Gas amount in SOL (default: 0.002)
 */
export async function getGasFeeUsd(gasSol: number = DEFAULT_GAS_SOL): Promise<number> {
  const solPrice = await getSolPrice();
  return gasSol * solPrice;
}

/**
 * Calculate volatility from price history
 * Uses standard deviation of returns method
 * 
 * @param prices - Array of historical prices (most recent last)
 * @returns Volatility as decimal (e.g., 0.05 for 5%)
 */
export function calculateVolatilityFromPrices(prices: number[]): number {
  if (prices.length < 2) return cache.volatility24h;
  
  // Calculate returns
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] > 0) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
  }
  
  if (returns.length === 0) return cache.volatility24h;
  
  // Calculate standard deviation
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const squaredDiffs = returns.map(r => Math.pow(r - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  
  // Annualize if hourly data (multiply by sqrt(24) for daily, sqrt(8760) for annual)
  // For 24h volatility from hourly data, just return the daily volatility
  const volatility = stdDev * Math.sqrt(24); // Daily volatility from hourly returns
  
  return Math.max(0.001, Math.min(0.50, volatility)); // Clamp between 0.1% and 50%
}

/**
 * Get 24h volatility for SOL
 * Fetches from cache or calculates from price history
 */
export async function getVolatility24h(priceHistory?: number[]): Promise<number> {
  const now = Date.now();
  
  // Return cached if fresh and no new data provided
  if (!priceHistory && now - cache.volatilityTimestamp < VOLATILITY_CACHE_TTL) {
    return cache.volatility24h;
  }
  
  // If price history provided, calculate
  if (priceHistory && priceHistory.length >= 2) {
    const volatility = calculateVolatilityFromPrices(priceHistory);
    cache.volatility24h = volatility;
    cache.volatilityTimestamp = now;
    logger.debug('[MarketData] Volatility updated', { 
      volatility: `${(volatility * 100).toFixed(2)}%`,
      dataPoints: priceHistory.length
    });
    return volatility;
  }
  
  // Fallback to cached
  return cache.volatility24h;
}

/**
 * Update volatility cache with new price data
 * Call this after fetching CEX prices in market scanner
 */
export function updateVolatilityCache(volatility: number): void {
  if (volatility > 0 && volatility < 1) {
    cache.volatility24h = volatility;
    cache.volatilityTimestamp = Date.now();
  }
}

/**
 * Get all cached market data (for debugging/logging)
 */
export function getMarketDataCache(): MarketDataCache {
  return { ...cache };
}

/**
 * Force refresh all market data
 */
export async function refreshMarketData(): Promise<{ solPrice: number; volatility24h: number }> {
  // Clear cache timestamps to force refresh
  cache.solPriceTimestamp = 0;
  cache.volatilityTimestamp = 0;
  
  const solPrice = await getSolPrice();
  
  return { solPrice, volatility24h: cache.volatility24h };
}

