/**
 * Benchmarks Config Loader
 *
 * Loads and parses the benchmarks_config.yaml file.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import { BenchmarksConfig } from './benchmarkTypes.js';

// ============= DEFAULT CONFIG =============

const DEFAULT_CONFIG: BenchmarksConfig = {
  enabled: true,
  check_interval_hours: 24,
  models: {
    perps: { min_precision: 0.70, min_recall: 0.50, min_sharpe: 1.5, min_win_rate: 0.60, min_roc_auc: 0.75 },
    spot: { min_precision: 0.75, min_recall: 0.55, min_sharpe: 1.8, min_win_rate: 0.65, min_roc_auc: 0.78 },
    lp: { min_precision: 0.75, min_recall: 0.60, min_sharpe: 1.2, min_win_rate: 0.70, min_roc_auc: 0.75 },
    lending: { min_precision: 0.65, min_recall: 0.50, min_sharpe: 1.0, min_win_rate: 0.60, min_roc_auc: 0.70 },
    arbitrage: { min_precision: 0.90, min_recall: 0.40, min_sharpe: 2.0, min_win_rate: 0.85, min_roc_auc: 0.90 },
  },
  system: {
    max_daily_drawdown: 0.05,
    max_weekly_drawdown: 0.10,
    max_monthly_drawdown: 0.15,
    min_daily_return: -0.02,
    min_weekly_return: -0.05,
    min_monthly_return: 0.02,
    target_annual_alpha: 0.05,
    target_annual_sharpe: 1.5,
    max_consecutive_losses: 5,
  },
  strategies: {
    lp_rebalancing: { min_monthly_trades: 5, min_win_rate: 0.70, max_avg_loss: 0.03 },
    spot_trading: { min_monthly_trades: 10, min_win_rate: 0.65, max_avg_loss: 0.05 },
    arbitrage: { min_monthly_trades: 20, min_win_rate: 0.85, max_avg_loss: 0.02 },
    perps: { min_monthly_trades: 15, min_win_rate: 0.60, max_avg_loss: 0.04 },
    lending: { min_monthly_trades: 5, min_win_rate: 0.70, max_avg_loss: 0.02 },
  },
  alerts: {
    warning_threshold: 0.90,
    critical_threshold: 1.00,
    pause_trading_on_critical: true,
    resume_requires_manual_approval: true,
    notification_channels: ['log'],
    alert_cooldown_hours: 4,
  },
};

// ============= CONFIG LOADER =============

let cachedConfig: BenchmarksConfig | null = null;

/**
 * Load benchmarks configuration from YAML file
 */
export async function loadBenchmarksConfig(): Promise<BenchmarksConfig> {
  if (cachedConfig) {
    return cachedConfig;
  }

  try {
    // Get the directory of this file
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    
    // Config path relative to project root
    const configPath = path.resolve(__dirname, '../../../../config/benchmarks_config.yaml');

    if (!fs.existsSync(configPath)) {
      logger.warn('[ConfigLoader] Config file not found, using defaults', { configPath });
      cachedConfig = DEFAULT_CONFIG;
      return cachedConfig;
    }

    const fileContent = fs.readFileSync(configPath, 'utf8');
    const parsed = yaml.load(fileContent) as Partial<BenchmarksConfig>;

    // Merge with defaults to ensure all fields exist
    cachedConfig = mergeConfig(DEFAULT_CONFIG, parsed);

    logger.info('[ConfigLoader] Loaded benchmarks config', {
      enabled: cachedConfig.enabled,
      checkInterval: cachedConfig.check_interval_hours,
      modelCount: Object.keys(cachedConfig.models).length,
      strategyCount: Object.keys(cachedConfig.strategies).length,
    });

    return cachedConfig;
  } catch (error) {
    logger.error('[ConfigLoader] Failed to load config', { error });
    cachedConfig = DEFAULT_CONFIG;
    return cachedConfig;
  }
}

/**
 * Merge partial config with defaults
 */
function mergeConfig(defaults: BenchmarksConfig, partial: Partial<BenchmarksConfig>): BenchmarksConfig {
  return {
    enabled: partial.enabled ?? defaults.enabled,
    check_interval_hours: partial.check_interval_hours ?? defaults.check_interval_hours,
    models: { ...defaults.models, ...partial.models },
    system: { ...defaults.system, ...partial.system },
    strategies: { ...defaults.strategies, ...partial.strategies },
    alerts: { ...defaults.alerts, ...partial.alerts },
  };
}

/**
 * Reload configuration (clears cache)
 */
export function reloadBenchmarksConfig(): void {
  cachedConfig = null;
  logger.info('[ConfigLoader] Config cache cleared');
}

/**
 * Get cached config or null
 */
export function getCachedConfig(): BenchmarksConfig | null {
  return cachedConfig;
}

