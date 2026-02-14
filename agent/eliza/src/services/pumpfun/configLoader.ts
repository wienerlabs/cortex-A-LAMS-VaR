/**
 * PumpFun Configuration Loader
 * 
 * Loads PumpFun trading configuration from YAML file.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { logger } from '../logger.js';

// ============= TYPES =============

export interface PumpFunRiskThresholds {
  minLiquidityUsd: number;
  minVolume24h: number;
  minHolders: number;
  minAgeHours: number;
  maxCreatorPct: number;
  maxTop10Pct: number;
  maxSniperPct: number;
}

export interface ExitStrategyConfig {
  takeProfitLevels: number[];  // % gains to take profit
  stopLossPct: number;
  trailingStopPct: number;
  autoExitTriggers: {
    liquidityDropPct: number;
    topHolderDumpPct: number;
  };
}

export interface PumpFunConfig {
  enabled: boolean;
  maxPositionSol: number;
  maxConcurrentPositions: number;
  useJito: boolean;
  jitoTipLamports: number;
  riskThresholds: PumpFunRiskThresholds;
  exitStrategy: ExitStrategyConfig;
}

// ============= DEFAULT CONFIG =============

const DEFAULT_CONFIG: PumpFunConfig = {
  enabled: true,
  maxPositionSol: 0.5,
  maxConcurrentPositions: 3,
  useJito: true,
  jitoTipLamports: 10000,
  riskThresholds: {
    minLiquidityUsd: 10000,
    minVolume24h: 5000,
    minHolders: 50,
    minAgeHours: 1,
    maxCreatorPct: 20,
    maxTop10Pct: 60,
    maxSniperPct: 30,
  },
  exitStrategy: {
    takeProfitLevels: [100, 200, 500],
    stopLossPct: 50,
    trailingStopPct: 30,
    autoExitTriggers: {
      liquidityDropPct: 50,
      topHolderDumpPct: 20,
    },
  },
};

// ============= LOADER =============

let cachedConfig: PumpFunConfig | null = null;
let lastLoadTime = 0;
const CACHE_TTL_MS = 60000; // 1 minute

/**
 * Load PumpFun configuration from YAML file
 */
export function loadPumpFunConfig(): PumpFunConfig {
  // Return cached config if fresh
  if (cachedConfig && Date.now() - lastLoadTime < CACHE_TTL_MS) {
    return cachedConfig;
  }

  const possiblePaths = [
    path.resolve(process.cwd(), 'config/pumpfun_params.yaml'),
    path.resolve(process.cwd(), '../config/pumpfun_params.yaml'),
    path.resolve(__dirname, '../../../../config/pumpfun_params.yaml'),
  ];

  for (const configPath of possiblePaths) {
    try {
      if (fs.existsSync(configPath)) {
        const content = fs.readFileSync(configPath, 'utf-8');
        const parsed = yaml.load(content) as Record<string, any>;
        cachedConfig = transformConfig(parsed);
        lastLoadTime = Date.now();
        logger.info('[PumpFunConfig] Loaded from', { path: configPath });
        return cachedConfig;
      }
    } catch (error: any) {
      logger.warn('[PumpFunConfig] Failed to load from', { path: configPath, error: error.message });
    }
  }

  logger.warn('[PumpFunConfig] Using defaults - no config file found');
  cachedConfig = DEFAULT_CONFIG;
  lastLoadTime = Date.now();
  return cachedConfig;
}

/**
 * Transform snake_case YAML to camelCase config
 */
function transformConfig(parsed: Record<string, any>): PumpFunConfig {
  return {
    enabled: parsed.enabled ?? DEFAULT_CONFIG.enabled,
    maxPositionSol: parsed.max_position_sol ?? parsed.maxPositionSol ?? DEFAULT_CONFIG.maxPositionSol,
    maxConcurrentPositions: parsed.max_concurrent_positions ?? parsed.maxConcurrentPositions ?? DEFAULT_CONFIG.maxConcurrentPositions,
    useJito: parsed.use_jito ?? parsed.useJito ?? DEFAULT_CONFIG.useJito,
    jitoTipLamports: parsed.jito_tip_lamports ?? parsed.jitoTipLamports ?? DEFAULT_CONFIG.jitoTipLamports,
    riskThresholds: transformRiskThresholds(parsed.risk_thresholds || parsed.riskThresholds || {}),
    exitStrategy: transformExitStrategy(parsed.exit_strategy || parsed.exitStrategy || {}),
  };
}

function transformRiskThresholds(parsed: Record<string, any>): PumpFunRiskThresholds {
  return {
    minLiquidityUsd: parsed.min_liquidity_usd ?? parsed.minLiquidityUsd ?? DEFAULT_CONFIG.riskThresholds.minLiquidityUsd,
    minVolume24h: parsed.min_volume_24h ?? parsed.minVolume24h ?? DEFAULT_CONFIG.riskThresholds.minVolume24h,
    minHolders: parsed.min_holders ?? parsed.minHolders ?? DEFAULT_CONFIG.riskThresholds.minHolders,
    minAgeHours: parsed.min_age_hours ?? parsed.minAgeHours ?? DEFAULT_CONFIG.riskThresholds.minAgeHours,
    maxCreatorPct: parsed.max_creator_pct ?? parsed.maxCreatorPct ?? DEFAULT_CONFIG.riskThresholds.maxCreatorPct,
    maxTop10Pct: parsed.max_top10_pct ?? parsed.maxTop10Pct ?? DEFAULT_CONFIG.riskThresholds.maxTop10Pct,
    maxSniperPct: parsed.max_sniper_pct ?? parsed.maxSniperPct ?? DEFAULT_CONFIG.riskThresholds.maxSniperPct,
  };
}

function transformExitStrategy(parsed: Record<string, any>): ExitStrategyConfig {
  const triggers = parsed.auto_exit_triggers || parsed.autoExitTriggers || {};
  return {
    takeProfitLevels: parsed.take_profit_levels ?? parsed.takeProfitLevels ?? DEFAULT_CONFIG.exitStrategy.takeProfitLevels,
    stopLossPct: parsed.stop_loss_pct ?? parsed.stopLossPct ?? DEFAULT_CONFIG.exitStrategy.stopLossPct,
    trailingStopPct: parsed.trailing_stop_pct ?? parsed.trailingStopPct ?? DEFAULT_CONFIG.exitStrategy.trailingStopPct,
    autoExitTriggers: {
      liquidityDropPct: triggers.liquidity_drop_pct ?? triggers.liquidityDropPct ?? DEFAULT_CONFIG.exitStrategy.autoExitTriggers.liquidityDropPct,
      topHolderDumpPct: triggers.top_holder_dump_pct ?? triggers.topHolderDumpPct ?? DEFAULT_CONFIG.exitStrategy.autoExitTriggers.topHolderDumpPct,
    },
  };
}

/**
 * Get default config (for testing or fallback)
 */
export function getDefaultConfig(): PumpFunConfig {
  return { ...DEFAULT_CONFIG };
}

