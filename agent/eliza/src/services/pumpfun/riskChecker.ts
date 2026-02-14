/**
 * PumpFun Risk Checker
 * 
 * Performs PumpFun-specific risk analysis to detect potential rug pulls,
 * suspicious token distributions, and other red flags.
 */

import { logger } from '../logger.js';
import type { PumpFunToken } from './pumpfunClient.js';
import { loadPumpFunConfig, type PumpFunConfig } from './configLoader.js';

// ============= TYPES =============

export interface PumpFunRiskCheck {
  isRugPull: boolean;
  creatorTokensPct: number;
  topHoldersPct: number;
  liquidityUsd: number;
  volumeUsd24h: number;
  holderCount: number;
  ageHours: number;
  graduated: boolean;
  riskScore: number; // 0-100, higher = more risky
  riskFlags: string[];
  warnings: string[];
}

export interface RiskThresholds {
  maxCreatorPct: number;
  maxTop10Pct: number;
  minLiquidityUsd: number;
  minVolumeUsd24h: number;
  minHolders: number;
  minAgeHours: number;
  maxSniperPct: number;
}

// ============= DEFAULT THRESHOLDS =============

const DEFAULT_THRESHOLDS: RiskThresholds = {
  maxCreatorPct: 20,
  maxTop10Pct: 60,
  minLiquidityUsd: 10000,
  minVolumeUsd24h: 5000,
  minHolders: 50,
  minAgeHours: 1,
  maxSniperPct: 30,
};

// ============= RISK CHECKER =============

/**
 * Check PumpFun token for risk factors
 */
export async function checkPumpFunRisks(
  token: PumpFunToken,
  customThresholds?: Partial<RiskThresholds>
): Promise<PumpFunRiskCheck> {
  // Load config or use defaults
  let thresholds: RiskThresholds;
  try {
    const config = loadPumpFunConfig();
    thresholds = {
      maxCreatorPct: config.riskThresholds.maxCreatorPct,
      maxTop10Pct: config.riskThresholds.maxTop10Pct,
      minLiquidityUsd: config.riskThresholds.minLiquidityUsd,
      minVolumeUsd24h: config.riskThresholds.minVolume24h,
      minHolders: config.riskThresholds.minHolders,
      minAgeHours: config.riskThresholds.minAgeHours,
      maxSniperPct: config.riskThresholds.maxSniperPct,
      ...customThresholds,
    };
  } catch {
    thresholds = { ...DEFAULT_THRESHOLDS, ...customThresholds };
  }

  const riskFlags: string[] = [];
  const warnings: string[] = [];
  let riskScore = 0;

  // Calculate basic metrics
  const creatorTokensPct = token.devHoldingsPercentage;
  const topHoldersPct = token.topHoldersPercentage;
  const liquidityUsd = token.marketCap; // Use market cap as proxy for liquidity
  const volumeUsd24h = token.volume;
  const holderCount = token.numHolders;
  const ageMs = Date.now() - token.creationTime;
  const ageHours = ageMs / (1000 * 60 * 60);
  const graduated = token.graduationDate !== null;
  const sniperPct = token.sniperOwnedPercentage;

  // ============= RUG PULL DETECTION =============

  // Check 1: Creator holds too many tokens
  if (creatorTokensPct > thresholds.maxCreatorPct) {
    riskFlags.push(`Creator holds ${creatorTokensPct.toFixed(1)}% (max: ${thresholds.maxCreatorPct}%)`);
    riskScore += 30;
  } else if (creatorTokensPct > thresholds.maxCreatorPct * 0.7) {
    warnings.push(`Creator holds ${creatorTokensPct.toFixed(1)}% (approaching limit)`);
    riskScore += 10;
  }

  // Check 2: Top holders concentration
  if (topHoldersPct > thresholds.maxTop10Pct) {
    riskFlags.push(`Top holders own ${topHoldersPct.toFixed(1)}% (max: ${thresholds.maxTop10Pct}%)`);
    riskScore += 25;
  } else if (topHoldersPct > thresholds.maxTop10Pct * 0.8) {
    warnings.push(`Top holders own ${topHoldersPct.toFixed(1)}% (high concentration)`);
    riskScore += 8;
  }

  // Check 3: Low liquidity
  if (liquidityUsd < thresholds.minLiquidityUsd) {
    riskFlags.push(`Low liquidity: $${(liquidityUsd / 1000).toFixed(1)}K (min: $${(thresholds.minLiquidityUsd / 1000).toFixed(0)}K)`);
    riskScore += 20;
  }

  // Check 4: Low volume
  if (volumeUsd24h < thresholds.minVolumeUsd24h) {
    riskFlags.push(`Low 24h volume: $${(volumeUsd24h / 1000).toFixed(1)}K (min: $${(thresholds.minVolumeUsd24h / 1000).toFixed(0)}K)`);
    riskScore += 15;
  }

  // Check 5: Few holders
  if (holderCount < thresholds.minHolders) {
    riskFlags.push(`Low holder count: ${holderCount} (min: ${thresholds.minHolders})`);
    riskScore += 15;
  }

  // Check 6: Too new
  if (ageHours < thresholds.minAgeHours) {
    riskFlags.push(`Token too new: ${ageHours.toFixed(1)}h (min: ${thresholds.minAgeHours}h)`);
    riskScore += 20;
  }

  // Check 7: High sniper ownership
  if (sniperPct > thresholds.maxSniperPct) {
    riskFlags.push(`High sniper ownership: ${sniperPct.toFixed(1)}% (max: ${thresholds.maxSniperPct}%)`);
    riskScore += 25;
  } else if (sniperPct > thresholds.maxSniperPct * 0.7) {
    warnings.push(`Sniper ownership: ${sniperPct.toFixed(1)}%`);
    riskScore += 10;
  }

  // Check 8: Social media red flags
  if (token.twitterReuseCount > 2) {
    warnings.push(`Twitter account reused ${token.twitterReuseCount} times`);
    riskScore += 10;
  }

  // Check 9: Mayhem mode (high volatility indicator)
  if (token.isMayhemMode) {
    warnings.push('Token in Mayhem mode (high volatility)');
    riskScore += 5;
  }

  // Check 10: Buy/sell ratio imbalance
  const buyRatio = token.buyTransactions / (token.sellTransactions + 1);
  if (buyRatio < 0.3) {
    riskFlags.push(`Heavy sell pressure: ${token.sellTransactions} sells vs ${token.buyTransactions} buys`);
    riskScore += 20;
  } else if (buyRatio < 0.5) {
    warnings.push(`Sell pressure detected: ${buyRatio.toFixed(2)} buy/sell ratio`);
    riskScore += 10;
  }

  // ============= POSITIVE SIGNALS =============

  // Graduated tokens are safer
  if (graduated) {
    riskScore = Math.max(0, riskScore - 15);
  }

  // Has social presence
  if (token.hasSocial) {
    riskScore = Math.max(0, riskScore - 5);
  }

  // More holders = lower risk
  if (holderCount > 500) {
    riskScore = Math.max(0, riskScore - 10);
  }

  // Older tokens = lower risk
  if (ageHours > 72) {
    riskScore = Math.max(0, riskScore - 10);
  }

  // Cap risk score at 100
  riskScore = Math.min(100, riskScore);

  // Determine if this is a rug pull risk
  const isRugPull = riskScore >= 50 || riskFlags.length >= 3;

  logger.info('[PumpFunRiskChecker] Risk assessment complete', {
    token: token.ticker,
    mint: token.coinMint,
    isRugPull,
    riskScore,
    riskFlags: riskFlags.length,
    warnings: warnings.length,
  });

  return {
    isRugPull,
    creatorTokensPct,
    topHoldersPct,
    liquidityUsd,
    volumeUsd24h,
    holderCount,
    ageHours,
    graduated,
    riskScore,
    riskFlags,
    warnings,
  };
}

/**
 * Quick check if token passes minimum safety requirements
 */
export function isTokenSafe(token: PumpFunToken, thresholds?: Partial<RiskThresholds>): boolean {
  const t = { ...DEFAULT_THRESHOLDS, ...thresholds };

  // Quick checks without full analysis
  if (token.devHoldingsPercentage > t.maxCreatorPct) return false;
  if (token.topHoldersPercentage > t.maxTop10Pct) return false;
  if (token.marketCap < t.minLiquidityUsd) return false;
  if (token.numHolders < t.minHolders) return false;
  if (token.sniperOwnedPercentage > t.maxSniperPct) return false;

  const ageHours = (Date.now() - token.creationTime) / (1000 * 60 * 60);
  if (ageHours < t.minAgeHours) return false;

  return true;
}

