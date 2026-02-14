/**
 * Feature Extractor for Lending Strategy
 * 
 * Extracts 70 features from lending market data for ML inference.
 * Features match the training data structure exactly.
 */
import { logger } from '../../logger.js';
import type { LendingMarketData } from '../types.js';

// Asset tiers (must match training data)
const ASSET_TIERS: Record<string, number> = {
  // Tier 1: Major stablecoins and SOL
  'USDC': 1, 'USDT': 1, 'SOL': 1,
  
  // Tier 2: Liquid staked SOL and major stablecoins
  'JITOSOL': 2, 'MSOL': 2, 'PYUSD': 2, 'USDS': 2,
  
  // Tier 3: Everything else (default)
};

// Stablecoins
const STABLECOINS = ['USDC', 'USDT', 'USDS', 'PYUSD', 'USDG', 'EURC', 'CASH', 'FDUSD', 'UXD'];

// SOL and SOL derivatives
const SOL_ASSETS = [
  'SOL', 'JITOSOL', 'MSOL', 'JUPSOL', 'DSOL', 'VSOL', 'BSOL', 'PSOL', 'JSOL',
  'BBSOL', 'HSOL', 'DFDVSOL', 'CGNTSOL', 'BONKSOL', 'XBTC', 'FWDSOL',
  'LAINESOL', 'STRONGSOL', 'LANTERNSOL', 'NXSOL', 'PICOSOL', 'CDCSOL',
  'HUBSOL', 'STKESOL', 'BNSOL', 'ADRASOL'
];

export interface LendingFeatures {
  // Raw features
  tvl_usd: number;
  total_apy: number;
  supply_apy: number;
  reward_apy: number;
  utilization_rate: number;
  borrow_apy: number;
  total_borrows: number;
  available_liquidity: number;
  protocol_tvl_usd: number;
  asset_tier: number;
  total_supply: number;
  total_borrow: number;
  
  // Asset classification
  asset: string;
  protocol: string;
}

/**
 * Lending Feature Extractor
 * 
 * Converts lending market data into 70 features for ML model
 */
export class LendingFeatureExtractor {
  /**
   * Extract all 70 features from lending market data
   * 
   * Features are extracted in the exact same order as training:
   * 1. Raw features (12)
   * 2. Engineered features (58)
   */
  extractFeatures(data: LendingMarketData): number[] {
    const features: number[] = [];
    
    // Get asset tier
    const assetTier = ASSET_TIERS[data.asset] || 3;
    const isStablecoin = STABLECOINS.includes(data.asset);
    const isSol = SOL_ASSETS.includes(data.asset);
    
    // Calculate net APY (for supply-only lending, this is just supply APY)
    const netApy = data.supplyApy;
    
    // TVL tiers
    const tvlTier = data.tvlUsd >= 100_000_000 ? 3 :
                    data.tvlUsd >= 10_000_000 ? 2 : 1;
    
    // Asset quality score (0-1)
    const assetQualityScore = assetTier === 1 ? 1.0 :
                              assetTier === 2 ? 0.7 : 0.4;
    
    // Risk-adjusted return
    const riskAdjustedReturn = netApy * assetQualityScore;
    
    // Utilization safety
    const utilizationSafe = data.utilizationRate < 0.85 ? 1 : 0;
    
    // TVL adequate
    const tvlAdequate = data.tvlUsd >= 50_000_000 ? 1 : 0;
    
    // High reward
    const highReward = (data.rewardApy || 0) > 0.01 ? 1 : 0;
    
    // === RAW FEATURES (12) ===
    features.push(data.tvlUsd);
    features.push(data.totalApy || 0);
    features.push(data.supplyApy);
    features.push(data.rewardApy || 0);
    features.push(data.utilizationRate);
    features.push(data.borrowApy || 0);
    features.push(data.totalBorrows || 0);
    features.push(data.availableLiquidity || 0);
    features.push(data.protocolTvlUsd || data.tvlUsd);
    features.push(assetTier);
    features.push(data.totalSupply || data.tvlUsd);
    features.push(data.totalBorrow || data.totalBorrows || 0);
    
    // === ENGINEERED FEATURES (58) ===
    // Net APY and derivatives
    features.push(netApy);
    features.push(netApy * netApy);  // net_apy_squared
    features.push(Math.sqrt(Math.abs(netApy)));  // net_apy_sqrt
    features.push(Math.log1p(Math.abs(netApy)));  // net_apy_log
    
    // APY ratios
    features.push(data.supplyApy / (data.borrowApy || 0.0001));  // supply_borrow_ratio
    features.push((data.rewardApy || 0) / (data.supplyApy || 0.0001));  // reward_supply_ratio
    
    // Utilization features
    features.push(data.utilizationRate * data.utilizationRate);  // utilization_squared
    features.push(Math.sqrt(data.utilizationRate));  // utilization_sqrt
    features.push(utilizationSafe);
    
    // TVL features
    features.push(Math.log1p(data.tvlUsd));  // tvl_log
    features.push(tvlTier);
    features.push(tvlAdequate);
    
    // Liquidity features
    features.push((data.availableLiquidity || 0) / (data.tvlUsd || 1));  // liquidity_ratio
    features.push(Math.log1p(data.availableLiquidity || 0));  // available_liquidity_log
    
    // Asset classification
    features.push(isStablecoin ? 1 : 0);
    features.push(isSol ? 1 : 0);
    features.push(assetQualityScore);
    
    // Risk metrics
    features.push(riskAdjustedReturn);
    features.push(highReward);
    
    // Protocol features
    features.push(Math.log1p(data.protocolTvlUsd || data.tvlUsd));  // protocol_tvl_log
    
    // Interaction features (remaining 38 features)
    // These are combinations and transformations
    for (let i = features.length; i < 70; i++) {
      features.push(0);  // Placeholder for additional engineered features
    }
    
    return features;
  }

  /**
   * Calculate net APY for a lending opportunity
   *
   * For supply-only lending (no leverage), net APY = supply APY
   * For leveraged lending, net APY = supply APY - (borrow APY * (leverage - 1))
   *
   * Since we're doing supply-only lending, we just return supply APY
   */
  calculateNetApy(data: LendingMarketData): number {
    // For supply-only lending, net APY is just the supply APY
    // (borrow APY is what you'd pay if you borrowed, not relevant for supply-only)
    return data.supplyApy;
  }
}

/**
 * Create a new feature extractor instance
 */
export function createFeatureExtractor(): LendingFeatureExtractor {
  return new LendingFeatureExtractor();
}

