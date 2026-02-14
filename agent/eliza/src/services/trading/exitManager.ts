/**
 * Exit Manager
 * Manages take profit levels, trailing stops, and stop losses
 */

import type { ExitLevels, SpotPosition, ApprovedToken } from './types.js';
import { logger } from '../logger.js';

export class ExitManager {
  /**
   * Calculate exit levels for a position
   */
  calculateExitLevels(
    entryPrice: number,
    entrySize: number,
    tokenTier: 1 | 2 | 3
  ): ExitLevels {
    // Production exit levels with tier-based stop losses

    // Take Profit Levels
    const tp1 = {
      price: entryPrice * 1.12,  // +12% (production)
      percentage: 0.12,
      size: entrySize * 0.40,    // Exit 40%
    };

    const tp2 = {
      price: entryPrice * 1.25,  // +25%
      percentage: 0.25,
      size: entrySize * 0.35,    // Exit 35%
    };

    const tp3 = {
      price: entryPrice * 1.40,  // +40%
      percentage: 0.40,
      size: entrySize * 0.25,    // Exit 25%
    };

    // Stop Loss - tier-based (8%/10%/12%) production values
    const stopLossPercentage = this.getStopLossPercentage(tokenTier);
    const stopLoss = {
      price: entryPrice * (1 - stopLossPercentage),
      percentage: stopLossPercentage,
    };

    return {
      tp1,
      tp2,
      tp3,
      stopLoss,
      trailingStop: null, // Activated after TP1
    };
  }

  /**
   * Get stop loss percentage based on token tier
   */
  private getStopLossPercentage(tier: 1 | 2 | 3): number {
    switch (tier) {
      case 1: return 0.08;  // 8% for tier 1
      case 2: return 0.10;  // 10% for tier 2
      case 3: return 0.12;  // 12% for tier 3
    }
  }

  /**
   * Update trailing stop
   */
  updateTrailingStop(
    position: SpotPosition,
    currentPrice: number
  ): ExitLevels {
    const exitLevels = { ...position.exitLevels };

    // Only activate trailing stop after TP1 is hit
    if (!position.tp1Hit) {
      return exitLevels;
    }

    // Initial trailing stop distance: 8%
    let trailingDistance = 0.08;

    // Tighten by 0.5% per day, minimum 4%
    const daysHeld = position.daysHeld;
    trailingDistance = Math.max(0.04, 0.08 - (daysHeld * 0.005));

    // Calculate trailing stop price
    const trailingStopPrice = currentPrice * (1 - trailingDistance);

    // Initialize or update trailing stop
    if (!exitLevels.trailingStop) {
      exitLevels.trailingStop = {
        distance: trailingDistance,
        price: trailingStopPrice,
      };
    } else {
      // Only move trailing stop up, never down
      if (trailingStopPrice > exitLevels.trailingStop.price) {
        exitLevels.trailingStop = {
          distance: trailingDistance,
          price: trailingStopPrice,
        };
      }
    }

    return exitLevels;
  }

  /**
   * Move stop loss to breakeven
   */
  moveStopToBreakeven(position: SpotPosition): ExitLevels {
    const exitLevels = { ...position.exitLevels };
    
    // Move to breakeven at +7% profit
    if (position.pnlPercent >= 0.07) {
      exitLevels.stopLoss = {
        price: position.entryPrice,
        percentage: 0,
      };

      logger.info('[ExitManager] Stop loss moved to breakeven', {
        symbol: position.token.symbol,
        entryPrice: position.entryPrice,
        currentPnl: (position.pnlPercent * 100).toFixed(2) + '%',
      });
    }

    return exitLevels;
  }

  /**
   * Check if position should be exited
   */
  shouldExit(position: SpotPosition, currentPrice: number): {
    shouldExit: boolean;
    reason?: string;
    exitType?: 'tp1' | 'tp2' | 'tp3' | 'stop_loss' | 'trailing_stop' | 'time_based';
    exitSize?: number;
  } {
    // Check TP1
    if (!position.tp1Hit && currentPrice >= position.exitLevels.tp1.price) {
      return {
        shouldExit: true,
        reason: 'TP1 hit (+12%)',
        exitType: 'tp1',
        exitSize: position.exitLevels.tp1.size,
      };
    }

    // Check TP2
    if (position.tp1Hit && !position.tp2Hit && currentPrice >= position.exitLevels.tp2.price) {
      return {
        shouldExit: true,
        reason: 'TP2 hit (+25%)',
        exitType: 'tp2',
        exitSize: position.exitLevels.tp2.size,
      };
    }

    // Check TP3
    if (position.tp2Hit && !position.tp3Hit && currentPrice >= position.exitLevels.tp3.price) {
      return {
        shouldExit: true,
        reason: 'TP3 hit (+40%)',
        exitType: 'tp3',
        exitSize: position.exitLevels.tp3.size,
      };
    }

    // Check stop loss
    if (currentPrice <= position.exitLevels.stopLoss.price) {
      return {
        shouldExit: true,
        reason: `Stop loss hit (${(position.exitLevels.stopLoss.percentage * 100).toFixed(0)}%)`,
        exitType: 'stop_loss',
        exitSize: position.remainingSize,
      };
    }

    // Check trailing stop
    if (position.exitLevels.trailingStop && currentPrice <= position.exitLevels.trailingStop.price) {
      return {
        shouldExit: true,
        reason: `Trailing stop hit (${(position.exitLevels.trailingStop.distance * 100).toFixed(1)}%)`,
        exitType: 'trailing_stop',
        exitSize: position.remainingSize,
      };
    }

    // Time-based exits
    const timeBasedExit = this.checkTimeBasedExit(position);
    if (timeBasedExit.shouldExit) {
      return timeBasedExit;
    }

    return { shouldExit: false };
  }

  /**
   * Check time-based exit conditions
   */
  private checkTimeBasedExit(position: SpotPosition): {
    shouldExit: boolean;
    reason?: string;
    exitType?: 'time_based';
    exitSize?: number;
  } {
    // Max hold losing position: 4 days
    if (position.pnlPercent < 0 && position.daysHeld >= 4) {
      return {
        shouldExit: true,
        reason: 'Max hold time for losing position (4 days)',
        exitType: 'time_based',
        exitSize: position.remainingSize,
      };
    }

    // Max hold flat position (±5%): 7 days
    if (Math.abs(position.pnlPercent) <= 0.05 && position.daysHeld >= 7) {
      return {
        shouldExit: true,
        reason: 'Max hold time for flat position (7 days)',
        exitType: 'time_based',
        exitSize: position.remainingSize,
      };
    }

    return { shouldExit: false };
  }
}

