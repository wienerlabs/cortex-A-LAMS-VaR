import { logger } from '../logger.js';
import { getPortfolioManager } from '../portfolioManager.js';
import type { LPPosition } from '../portfolioManager.js';

/**
 * LP Position Monitor Configuration
 */
export interface LPMonitorConfig {
  // Stop loss as percentage of capital (e.g., 0.05 = 5% loss)
  stopLossPct: number;
  
  // Take profit levels
  takeProfit1Pct: number;  // First take profit (e.g., 0.10 = 10% profit)
  takeProfit2Pct: number;  // Second take profit (e.g., 0.20 = 20% profit)
  
  // Partial exit sizes
  tp1ExitPct: number;  // Exit 50% at TP1
  tp2ExitPct: number;  // Exit 50% of remaining at TP2
  
  // Time-based exits
  maxHoldDaysLosing: number;  // Max days to hold losing position
  maxHoldDaysFlat: number;    // Max days to hold flat position
  
  // Impermanent loss threshold
  maxImpermanentLossPct: number;  // Max IL before exit (e.g., 0.05 = 5%)
  
  // Monitoring interval
  checkIntervalMs: number;  // How often to check positions (default: 5 minutes)
}

const DEFAULT_CONFIG: LPMonitorConfig = {
  stopLossPct: 0.05,           // 5% stop loss
  takeProfit1Pct: 0.10,        // 10% take profit
  takeProfit2Pct: 0.20,        // 20% take profit
  tp1ExitPct: 0.50,            // Exit 50% at TP1
  tp2ExitPct: 0.50,            // Exit 50% at TP2
  maxHoldDaysLosing: 7,        // 7 days max for losing positions
  maxHoldDaysFlat: 14,         // 14 days max for flat positions
  maxImpermanentLossPct: 0.08, // 8% max impermanent loss
  checkIntervalMs: 5 * 60 * 1000, // 5 minutes
};

export interface ExitDecision {
  shouldExit: boolean;
  reason?: string;
  exitType?: 'stop_loss' | 'tp1' | 'tp2' | 'time_based' | 'impermanent_loss';
  exitPercentage?: number;  // Percentage of position to exit (100 = full exit)
}

/**
 * LP Position Monitor
 * Monitors open LP positions and determines when to exit based on:
 * - Stop loss
 * - Take profit levels
 * - Time-based exits
 * - Impermanent loss
 */
export class LPPositionMonitor {
  private config: LPMonitorConfig;
  private portfolioManager = getPortfolioManager();

  constructor(config: Partial<LPMonitorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    logger.info('[LPPositionMonitor] Initialized', {
      stopLossPct: this.config.stopLossPct,
      takeProfit1Pct: this.config.takeProfit1Pct,
      takeProfit2Pct: this.config.takeProfit2Pct,
      maxHoldDaysLosing: this.config.maxHoldDaysLosing,
      maxImpermanentLossPct: this.config.maxImpermanentLossPct,
    });
  }

  /**
   * Check if a position should be exited
   */
  shouldExit(position: LPPosition): ExitDecision {
    const pnlPct = this.calculatePnLPercent(position);
    const daysHeld = this.calculateDaysHeld(position);
    const ilPct = this.calculateImpermanentLossPct(position);

    // 1. Check Stop Loss
    if (pnlPct <= -this.config.stopLossPct) {
      return {
        shouldExit: true,
        reason: `Stop loss hit: ${(pnlPct * 100).toFixed(2)}% (max: -${(this.config.stopLossPct * 100).toFixed(0)}%)`,
        exitType: 'stop_loss',
        exitPercentage: 100,
      };
    }

    // 2. Check Impermanent Loss
    if (ilPct >= this.config.maxImpermanentLossPct) {
      return {
        shouldExit: true,
        reason: `Impermanent loss too high: ${(ilPct * 100).toFixed(2)}% (max: ${(this.config.maxImpermanentLossPct * 100).toFixed(0)}%)`,
        exitType: 'impermanent_loss',
        exitPercentage: 100,
      };
    }

    // 3. Check Take Profit 2 (higher level)
    if (pnlPct >= this.config.takeProfit2Pct) {
      return {
        shouldExit: true,
        reason: `Take profit 2 hit: +${(pnlPct * 100).toFixed(2)}%`,
        exitType: 'tp2',
        exitPercentage: this.config.tp2ExitPct * 100,
      };
    }

    // 4. Check Take Profit 1
    if (pnlPct >= this.config.takeProfit1Pct) {
      return {
        shouldExit: true,
        reason: `Take profit 1 hit: +${(pnlPct * 100).toFixed(2)}%`,
        exitType: 'tp1',
        exitPercentage: this.config.tp1ExitPct * 100,
      };
    }

    // 5. Time-based exits
    const timeBasedExit = this.checkTimeBasedExit(position, pnlPct, daysHeld);
    if (timeBasedExit.shouldExit) {
      return timeBasedExit;
    }

    return { shouldExit: false };
  }

  /**
   * Check time-based exit conditions
   */
  private checkTimeBasedExit(position: LPPosition, pnlPct: number, daysHeld: number): ExitDecision {
    // Max hold losing position
    if (pnlPct < 0 && daysHeld >= this.config.maxHoldDaysLosing) {
      return {
        shouldExit: true,
        reason: `Max hold time for losing position (${daysHeld} days)`,
        exitType: 'time_based',
        exitPercentage: 100,
      };
    }

    // Max hold flat position (±3%)
    if (Math.abs(pnlPct) <= 0.03 && daysHeld >= this.config.maxHoldDaysFlat) {
      return {
        shouldExit: true,
        reason: `Max hold time for flat position (${daysHeld} days)`,
        exitType: 'time_based',
        exitPercentage: 100,
      };
    }

    return { shouldExit: false };
  }

  /**
   * Calculate PnL percentage
   */
  private calculatePnLPercent(position: LPPosition): number {
    return (position.currentValueUsd - position.capitalUsd) / position.capitalUsd;
  }

  /**
   * Calculate days held
   */
  private calculateDaysHeld(position: LPPosition): number {
    const msHeld = Date.now() - position.entryTime;
    return msHeld / (1000 * 60 * 60 * 24);
  }

  /**
   * Calculate impermanent loss percentage
   */
  private calculateImpermanentLossPct(position: LPPosition): number {
    return Math.abs(position.impermanentLossUsd) / position.capitalUsd;
  }

  /**
   * Get monitoring interval
   */
  getCheckInterval(): number {
    return this.config.checkIntervalMs;
  }
}

