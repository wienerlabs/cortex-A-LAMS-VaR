import { logger } from '../logger.js';
import type { SolanaAgentKitService } from './index.js';

export interface TokenSafetyResult {
  safe: boolean;
  riskScore: number; // 0-100, higher = riskier
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  flags: string[];
  warnings: string[];
  source: 'kit_rugcheck' | 'dexscreener_fallback' | 'unavailable';
  data: {
    liquidityUsd?: number;
    volume24h?: number;
    pairCount?: number;
    fdv?: number;
    rugCheckScore?: number;
  };
}

const RISK_THRESHOLDS = {
  minLiquidityUsd: 30_000,
  minVolume24h: 10_000,
  minPairCount: 1,
  maxRugCheckScore: 50, // Kit rugCheck: higher = riskier
} as const;

/**
 * Fetch token safety data via Kit rugCheck, falling back to DexScreener REST.
 */
export async function checkTokenSafety(
  tokenAddress: string,
  kitService: SolanaAgentKitService | null,
): Promise<TokenSafetyResult> {
  // Try Kit rugCheck first
  if (kitService?.isInitialized() && kitService.hasAction('rugCheck')) {
    try {
      const result = await kitService.execute('rugCheck', { mint: tokenAddress });
      if (result.success && result.data) {
        return parseKitRugCheck(result.data, tokenAddress);
      }
      logger.warn('[TokenSafety] Kit rugCheck returned failure, falling back to DexScreener', {
        token: tokenAddress,
        error: result.error,
      });
    } catch (error) {
      logger.warn('[TokenSafety] Kit rugCheck threw, falling back to DexScreener', {
        token: tokenAddress,
        error: String(error),
      });
    }
  }

  // Fallback: DexScreener REST API
  return fetchDexScreenerSafety(tokenAddress);
}

function parseKitRugCheck(data: unknown, tokenAddress: string): TokenSafetyResult {
  const d = data as Record<string, unknown>;
  const score = typeof d.score === 'number' ? d.score : 50;
  const risks = Array.isArray(d.risks) ? (d.risks as string[]) : [];

  const flags: string[] = [];
  const warnings: string[] = [];

  // Kit scores: 0 = safe, 100 = dangerous
  const riskScore = Math.min(100, Math.max(0, score));

  for (const risk of risks) {
    if (riskScore >= 50) {
      flags.push(risk);
    } else {
      warnings.push(risk);
    }
  }

  const riskLevel = riskLevelFromScore(riskScore);

  logger.info('[TokenSafety] Kit rugCheck result', {
    token: tokenAddress,
    riskScore,
    riskLevel,
    flags: flags.length,
  });

  return {
    safe: riskScore < RISK_THRESHOLDS.maxRugCheckScore,
    riskScore,
    riskLevel,
    flags,
    warnings,
    source: 'kit_rugcheck',
    data: { rugCheckScore: riskScore },
  };
}

async function fetchDexScreenerSafety(tokenAddress: string): Promise<TokenSafetyResult> {
  try {
    const resp = await fetch(
      `https://api.dexscreener.com/latest/dex/tokens/${tokenAddress}`,
      { signal: AbortSignal.timeout(8000) },
    );

    if (!resp.ok) {
      logger.warn('[TokenSafety] DexScreener API error', { status: resp.status });
      return unavailableResult();
    }

    const json = (await resp.json()) as { pairs?: DexScreenerPair[] };
    const pairs = json.pairs;

    if (!pairs || pairs.length === 0) {
      return {
        safe: false,
        riskScore: 80,
        riskLevel: 'HIGH',
        flags: ['No trading pairs found on DexScreener'],
        warnings: [],
        source: 'dexscreener_fallback',
        data: { pairCount: 0 },
      };
    }

    // Use highest-liquidity pair
    const bestPair = pairs.reduce((best, pair) =>
      (pair.liquidity?.usd ?? 0) > (best.liquidity?.usd ?? 0) ? pair : best,
    );

    return evaluateDexScreenerPair(bestPair, pairs.length, tokenAddress);
  } catch (error) {
    logger.warn('[TokenSafety] DexScreener fetch failed', { error: String(error) });
    return unavailableResult();
  }
}

interface DexScreenerPair {
  liquidity?: { usd?: number };
  volume?: { h24?: number };
  fdv?: number;
  pairCreatedAt?: number;
}

function evaluateDexScreenerPair(
  pair: DexScreenerPair,
  pairCount: number,
  tokenAddress: string,
): TokenSafetyResult {
  const flags: string[] = [];
  const warnings: string[] = [];
  let riskScore = 0;

  const liquidityUsd = pair.liquidity?.usd ?? 0;
  const volume24h = pair.volume?.h24 ?? 0;
  const fdv = pair.fdv ?? 0;

  // Liquidity check
  if (liquidityUsd < RISK_THRESHOLDS.minLiquidityUsd) {
    flags.push(`Low liquidity: $${(liquidityUsd / 1000).toFixed(1)}K`);
    riskScore += 30;
  } else if (liquidityUsd < RISK_THRESHOLDS.minLiquidityUsd * 3) {
    warnings.push(`Moderate liquidity: $${(liquidityUsd / 1000).toFixed(1)}K`);
    riskScore += 10;
  }

  // Volume check
  if (volume24h < RISK_THRESHOLDS.minVolume24h) {
    flags.push(`Low 24h volume: $${(volume24h / 1000).toFixed(1)}K`);
    riskScore += 25;
  } else if (volume24h < RISK_THRESHOLDS.minVolume24h * 5) {
    warnings.push(`Moderate 24h volume: $${(volume24h / 1000).toFixed(1)}K`);
    riskScore += 8;
  }

  // FDV vs liquidity ratio (high FDV with low liquidity = red flag)
  if (fdv > 0 && liquidityUsd > 0) {
    const fdvToLiqRatio = fdv / liquidityUsd;
    if (fdvToLiqRatio > 100) {
      flags.push(`FDV/Liquidity ratio extremely high: ${fdvToLiqRatio.toFixed(0)}x`);
      riskScore += 20;
    } else if (fdvToLiqRatio > 50) {
      warnings.push(`High FDV/Liquidity ratio: ${fdvToLiqRatio.toFixed(0)}x`);
      riskScore += 10;
    }
  }

  // Token age check
  if (pair.pairCreatedAt) {
    const ageHours = (Date.now() - pair.pairCreatedAt) / (1000 * 60 * 60);
    if (ageHours < 24) {
      flags.push(`Very new token: ${ageHours.toFixed(0)}h old`);
      riskScore += 15;
    } else if (ageHours < 72) {
      warnings.push(`New token: ${ageHours.toFixed(0)}h old`);
      riskScore += 5;
    }
  }

  riskScore = Math.min(100, riskScore);
  const riskLevel = riskLevelFromScore(riskScore);

  logger.info('[TokenSafety] DexScreener safety result', {
    token: tokenAddress,
    riskScore,
    riskLevel,
    liquidityUsd,
    volume24h,
  });

  return {
    safe: riskScore < RISK_THRESHOLDS.maxRugCheckScore,
    riskScore,
    riskLevel,
    flags,
    warnings,
    source: 'dexscreener_fallback',
    data: { liquidityUsd, volume24h, pairCount, fdv },
  };
}

function riskLevelFromScore(score: number): TokenSafetyResult['riskLevel'] {
  if (score >= 70) return 'CRITICAL';
  if (score >= 50) return 'HIGH';
  if (score >= 25) return 'MEDIUM';
  return 'LOW';
}

function unavailableResult(): TokenSafetyResult {
  return {
    safe: true, // Don't block trades when safety data unavailable
    riskScore: 0,
    riskLevel: 'LOW',
    flags: [],
    warnings: ['Token safety data unavailable — proceed with caution'],
    source: 'unavailable',
    data: {},
  };
}
