/**
 * Trading Mode Configuration - Production System
 *
 * NORMAL: Conservative trading with established tokens only
 * AGGRESSIVE: Higher risk tolerance with memecoin/Pump.fun support
 */

export enum TradingMode {
  NORMAL = 'NORMAL',
  AGGRESSIVE = 'AGGRESSIVE'
}

export interface ModeConfig {
  mode: TradingMode;
  minHealthScore: number;
  enablePumpFun: boolean;
  riskMultiplier: number;
  maxPositionSize: number;
  minTVL: number;
  minHolders: number;
  minLiquidity: number;
  description: string;
}

export const MODE_CONFIGS: Record<TradingMode, ModeConfig> = {
  [TradingMode.NORMAL]: {
    mode: TradingMode.NORMAL,
    minHealthScore: 60,
    enablePumpFun: false,
    riskMultiplier: 1.0,
    maxPositionSize: 0.05,      // 5% max position
    minTVL: 100_000,             // $100K minimum TVL
    minHolders: 100,             // 100+ holders
    minLiquidity: 200_000,       // $200K minimum liquidity
    description: 'Conservative trading with established tokens only (health ≥ 60)',
  },
  [TradingMode.AGGRESSIVE]: {
    mode: TradingMode.AGGRESSIVE,
    minHealthScore: 40,
    enablePumpFun: true,
    riskMultiplier: 1.5,
    maxPositionSize: 0.10,      // 10% max position
    minTVL: 10_000,              // $10K minimum TVL
    minHolders: 50,              // 50+ holders
    minLiquidity: 50_000,        // $50K minimum liquidity
    description: 'Higher risk tolerance with memecoins and Pump.fun tokens (health ≥ 40)',
  },
};

// Legacy type alias for backward compatibility
export type TradingModeConfig = ModeConfig;

/**
 * Get current trading mode from environment variable
 * Defaults to NORMAL if not set or invalid
 */
export function getTradingMode(): ModeConfig {
  const modeEnv = (process.env.TRADING_MODE || 'NORMAL').toUpperCase();
  const mode = (modeEnv === 'AGGRESSIVE' ? TradingMode.AGGRESSIVE : TradingMode.NORMAL);
  return MODE_CONFIGS[mode];
}

/**
 * Check if current mode allows a specific feature
 */
export function isModeFeatureEnabled(feature: 'memecoins' | 'pumpfun'): boolean {
  const mode = getTradingMode();
  if (feature === 'pumpfun') return mode.enablePumpFun;
  // Memecoins are allowed in AGGRESSIVE mode (same as Pump.fun)
  return mode.enablePumpFun;
}

/**
 * Get risk-adjusted position size based on health score and mode
 */
export function getRiskAdjustedSize(
  baseSize: number,
  healthScore: number,
  modeConfig?: ModeConfig
): number {
  const mode = modeConfig || getTradingMode();

  // Reduce position size for low-health tokens
  let sizeMultiplier = 1.0;

  if (healthScore < 50) {
    sizeMultiplier = 0.5;  // 50% reduction for health < 50
  } else if (healthScore < 60) {
    sizeMultiplier = 0.75; // 25% reduction for health < 60
  }

  const adjustedSize = baseSize * sizeMultiplier * mode.riskMultiplier;

  // Cap at maxPositionSize
  return Math.min(adjustedSize, mode.maxPositionSize);
}

