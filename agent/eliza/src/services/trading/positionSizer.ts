/**
 * Position Sizing Calculator
 * Calculates position size based on volatility and confidence
 */

import type { PositionSize, ApprovedToken, RiskLimits } from './types.js';
import { logger } from '../logger.js';

export const DEFAULT_RISK_LIMITS: RiskLimits = {
  maxPositionSize: 2500,           // $2,500
  maxPerToken: 0.12,               // 12% of spot capital
  maxSpotAllocation: 0.35,         // 35% of total capital
  maxConcurrentPositions: 4,
  maxEntriesPerDay: 2,
  minTimeBetweenEntries: 6,        // hours
  dailyLossLimit: 0.06,            // 6% of spot allocation
  weeklyLossLimit: 0.12,           // 12% of spot allocation
  maxConsecutiveLosses: 3,
  maxCorrelatedPositions: 2,
  correlationThreshold: 0.75,
  maxSOLBetaExposure: 0.50,        // 50%
};

export class PositionSizer {
  private limits: RiskLimits;

  constructor(limits: RiskLimits = DEFAULT_RISK_LIMITS) {
    this.limits = limits;
  }

  /**
   * Calculate position size
   */
  calculatePositionSize(
    token: ApprovedToken,
    confidence: number,
    volatility: number,
    spotCapital: number
  ): PositionSize {
    // Base size
    const baseSize = this.limits.maxPositionSize;

    // Volatility adjustment
    const volatilityMultiplier = this.getVolatilityMultiplier(volatility);

    // Confidence adjustment
    const confidenceMultiplier = this.getConfidenceMultiplier(confidence);

    // Calculate final size
    let finalSize = baseSize * volatilityMultiplier * confidenceMultiplier;

    // Apply max per token limit
    const maxPerToken = spotCapital * this.limits.maxPerToken;
    finalSize = Math.min(finalSize, maxPerToken);

    // Apply minimum position size
    const minPosition = 15; // $15 minimum for spot trading
    if (finalSize < minPosition) {
      finalSize = 0; // Too small, don't enter
    }

    const result: PositionSize = {
      baseSize,
      volatilityMultiplier,
      confidenceMultiplier,
      finalSize,
      sizeUsd: finalSize,
    };

    logger.info('[PositionSizer] Position size calculated', {
      symbol: token.symbol,
      baseSize,
      volatility: (volatility * 100).toFixed(2) + '%',
      volatilityMultiplier,
      confidence: (confidence * 100).toFixed(1) + '%',
      confidenceMultiplier,
      finalSize: finalSize.toFixed(0),
    });

    return result;
  }

  /**
   * Get volatility multiplier
   */
  private getVolatilityMultiplier(volatility: number): number {
    if (volatility < 0.05) return 1.0;   // < 5% daily: 1.0x
    if (volatility < 0.08) return 0.8;   // 5-8% daily: 0.8x
    if (volatility < 0.12) return 0.6;   // 8-12% daily: 0.6x
    return 0.4;                          // > 12% daily: 0.4x
  }

  /**
   * Get confidence multiplier
   */
  private getConfidenceMultiplier(confidence: number): number {
    if (confidence < 0.45) return 0;     // No entry
    if (confidence < 0.55) return 0.5;   // 50% position
    if (confidence < 0.70) return 0.75;  // 75% position
    return 1.0;                          // 100% position
  }

  /**
   * Check if position size is within limits
   */
  checkLimits(
    positionSize: number,
    spotCapital: number,
    currentPositions: number,
    currentAllocation: number
  ): { allowed: boolean; reason?: string } {
    // Check max concurrent positions
    if (currentPositions >= this.limits.maxConcurrentPositions) {
      return {
        allowed: false,
        reason: `Max concurrent positions reached (${this.limits.maxConcurrentPositions})`,
      };
    }

    // Check max spot allocation
    const newAllocation = (currentAllocation + positionSize) / spotCapital;
    if (newAllocation > this.limits.maxSpotAllocation) {
      return {
        allowed: false,
        reason: `Would exceed max spot allocation (${(this.limits.maxSpotAllocation * 100).toFixed(0)}%)`,
      };
    }

    // Check minimum position size
    if (positionSize < 15) {
      return {
        allowed: false,
        reason: 'Position size too small (< $15)',
      };
    }

    return { allowed: true };
  }

  /**
   * Get risk limits
   */
  getLimits(): RiskLimits {
    return { ...this.limits };
  }

  /**
   * Update risk limits
   */
  updateLimits(newLimits: Partial<RiskLimits>): void {
    this.limits = { ...this.limits, ...newLimits };
    logger.info('[PositionSizer] Risk limits updated', { limits: this.limits });
  }
}

