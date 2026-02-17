/**
 * Lending Position Monitor
 * 
 * Monitors lending positions and determines when to exit based on:
 * - APY drops (take profit when APY drops significantly)
 * - Protocol health degradation
 * - Time-based exits
 * - Better opportunity elsewhere
 */

import { logger } from '../logger.js';

export interface LendingPositionMonitorConfig {
  minApyPct: number;           // Minimum APY to keep position (e.g., 0.02 = 2%)
  apyDropThresholdPct: number; // Exit if APY drops by this much from entry (e.g., 0.5 = 50%)
  maxHoldDays: number;         // Maximum days to hold a lending position
  minHealthFactor: number;     // Minimum protocol health factor
  checkIntervalMs: number;     // How often to check positions
}

export interface TrackedLendingPosition {
  id: string;
  protocol: string;
  asset: string;
  depositedUsd: number;
  entryApy: number;
  currentApy: number;
  entryTime: number;
  currentValueUsd: number;
  earnedUsd: number;
  healthFactor: number;
  status: 'open' | 'closed';
}

export interface LendingExitDecision {
  shouldExit: boolean;
  reason?: string;
  exitType?: 'apy_drop' | 'low_apy' | 'time_based' | 'health_risk' | 'take_profit';
  exitPercentage?: number;
}

const DEFAULT_CONFIG: LendingPositionMonitorConfig = {
  minApyPct: 0.02,             // 2% minimum APY
  apyDropThresholdPct: 0.5,    // 50% APY drop from entry
  maxHoldDays: 30,             // 30 days max
  minHealthFactor: 1.2,        // Minimum health factor
  checkIntervalMs: 10 * 60 * 1000, // Check every 10 minutes
};

export class LendingPositionMonitor {
  private config: LendingPositionMonitorConfig;

  constructor(config: Partial<LendingPositionMonitorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    logger.info('[LendingPositionMonitor] Initialized', {
      minApyPct: (this.config.minApyPct * 100).toFixed(1) + '%',
      apyDropThreshold: (this.config.apyDropThresholdPct * 100).toFixed(0) + '%',
      maxHoldDays: this.config.maxHoldDays,
      minHealthFactor: this.config.minHealthFactor,
    });
  }

  /**
   * Determine if a lending position should be exited
   */
  shouldExit(position: TrackedLendingPosition): LendingExitDecision {
    const daysHeld = this.calculateDaysHeld(position);
    const pnlPct = this.calculatePnLPercent(position);
    const apyDropPct = this.calculateApyDrop(position);

    // 1. Check Health Factor (protocol risk)
    if (position.healthFactor < this.config.minHealthFactor) {
      return {
        shouldExit: true,
        reason: `Health factor too low: ${position.healthFactor.toFixed(2)} < ${this.config.minHealthFactor}`,
        exitType: 'health_risk',
        exitPercentage: 100,
      };
    }

    // 2. Check APY Drop from Entry
    if (apyDropPct >= this.config.apyDropThresholdPct) {
      return {
        shouldExit: true,
        reason: `APY dropped ${(apyDropPct * 100).toFixed(0)}% from entry (${(position.entryApy * 100).toFixed(2)}% → ${(position.currentApy * 100).toFixed(2)}%)`,
        exitType: 'apy_drop',
        exitPercentage: 100,
      };
    }

    // 3. Check Minimum APY
    if (position.currentApy < this.config.minApyPct) {
      return {
        shouldExit: true,
        reason: `APY too low: ${(position.currentApy * 100).toFixed(2)}% < ${(this.config.minApyPct * 100).toFixed(0)}% minimum`,
        exitType: 'low_apy',
        exitPercentage: 100,
      };
    }

    // 4. Time-based Exit
    if (daysHeld >= this.config.maxHoldDays) {
      return {
        shouldExit: true,
        reason: `Max hold time reached: ${daysHeld.toFixed(0)} days`,
        exitType: 'time_based',
        exitPercentage: 100,
      };
    }

    // 5. Take Profit - if earned significant amount, consider partial exit
    if (pnlPct >= 0.05 && daysHeld >= 7) { // 5% profit after 7 days
      return {
        shouldExit: true,
        reason: `Take profit: +${(pnlPct * 100).toFixed(2)}% earned after ${daysHeld.toFixed(0)} days`,
        exitType: 'take_profit',
        exitPercentage: 50, // Partial exit
      };
    }

    return { shouldExit: false };
  }

  private calculateDaysHeld(position: TrackedLendingPosition): number {
    return (Date.now() - position.entryTime) / (24 * 60 * 60 * 1000);
  }

  private calculatePnLPercent(position: TrackedLendingPosition): number {
    if (position.depositedUsd === 0) return 0;
    return position.earnedUsd / position.depositedUsd;
  }

  private calculateApyDrop(position: TrackedLendingPosition): number {
    if (position.entryApy === 0) return 0;
    return (position.entryApy - position.currentApy) / position.entryApy;
  }

  getCheckInterval(): number {
    return this.config.checkIntervalMs;
  }
}

