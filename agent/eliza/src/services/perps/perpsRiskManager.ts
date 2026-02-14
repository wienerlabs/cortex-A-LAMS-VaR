/**
 * Perpetual Futures Risk Manager
 * 
 * Manages risk for leveraged perpetual positions:
 * - Liquidation distance monitoring
 * - Margin ratio tracking
 * - Position sizing based on risk parameters
 * - Multi-venue risk aggregation
 */
import { logger } from '../logger.js';
import type {
  PerpsPosition,
  PerpsVenue,
  PositionSide,
  RiskMetrics,
} from '../../types/perps.js';

// ============= RISK PARAMETERS =============

export interface PerpsRiskConfig {
  // Maximum leverage allowed
  maxLeverage: number;
  
  // Minimum liquidation distance (% from current price)
  minLiquidationDistance: number;
  
  // Maximum position size as % of portfolio
  maxPositionSizePercent: number;
  
  // Maximum total exposure across all positions
  maxTotalExposure: number;
  
  // Minimum margin ratio before warning
  minMarginRatio: number;
  
  // Auto-deleverage threshold
  autoDeleverageThreshold: number;
  
  // Maximum funding rate to pay (annualized)
  maxFundingRateApr: number;
  
  // Stop loss percentage
  defaultStopLossPercent: number;
  
  // Take profit percentage
  defaultTakeProfitPercent: number;
}

export const DEFAULT_PERPS_RISK_CONFIG: PerpsRiskConfig = {
  maxLeverage: 5,
  minLiquidationDistance: 0.15,        // 15% from liquidation
  maxPositionSizePercent: 0.25,        // 25% of portfolio per position
  maxTotalExposure: 2.0,               // 2x total portfolio exposure
  minMarginRatio: 0.20,                // 20% margin ratio warning
  autoDeleverageThreshold: 0.10,       // 10% from liquidation triggers deleverage
  maxFundingRateApr: 0.50,             // 50% APR max funding
  defaultStopLossPercent: 0.05,        // 5% stop loss
  defaultTakeProfitPercent: 0.10,      // 10% take profit
};

// ============= RISK ASSESSMENT =============

export interface RiskAssessment {
  approved: boolean;
  riskScore: number;           // 0-100, higher = riskier
  warnings: string[];
  blockers: string[];
  recommendations: string[];
  adjustedParams?: {
    leverage?: number;
    size?: number;
    stopLoss?: number;
    takeProfit?: number;
  };
}

export interface PositionRiskStatus {
  position: PerpsPosition;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  liquidationDistance: number;
  marginRatio: number;
  unrealizedPnlPercent: number;
  fundingCostPercent: number;
  timeToLiquidation?: number;  // Estimated hours at current funding rate
  action?: 'hold' | 'reduce' | 'close' | 'add_margin';
}

// ============= PERPS RISK MANAGER =============

export class PerpsRiskManager {
  private config: PerpsRiskConfig;
  private positions: Map<string, PerpsPosition> = new Map();
  private portfolioValue: number = 0;

  constructor(config: Partial<PerpsRiskConfig> = {}) {
    this.config = { ...DEFAULT_PERPS_RISK_CONFIG, ...config };
    logger.info('PerpsRiskManager initialized', { config: this.config });
  }

  /**
   * Update portfolio value for risk calculations
   */
  setPortfolioValue(value: number): void {
    this.portfolioValue = value;
  }

  /**
   * Update tracked positions
   */
  updatePositions(positions: PerpsPosition[]): void {
    this.positions.clear();
    for (const pos of positions) {
      const key = `${pos.venue}-${pos.market}`;
      this.positions.set(key, pos);
    }
  }

  /**
   * Assess risk for a new position
   */
  assessNewPosition(params: {
    venue: PerpsVenue;
    market: string;
    side: PositionSide;
    size: number;
    entryPrice: number;
    leverage: number;
    collateral: number;
    fundingRateApr?: number;
  }): RiskAssessment {
    const {
      venue, market, side, size, entryPrice, leverage, collateral, fundingRateApr = 0
    } = params;

    const assessment: RiskAssessment = {
      approved: true,
      riskScore: 0,
      warnings: [],
      blockers: [],
      recommendations: [],
    };

    // Calculate position metrics
    const notional = size * entryPrice;
    const positionSizePercent = this.portfolioValue > 0 
      ? notional / this.portfolioValue 
      : 1;

    // Calculate liquidation price
    const maintenanceMarginRatio = 0.05;
    const maintenanceMargin = notional * maintenanceMarginRatio;
    const maxLoss = collateral - maintenanceMargin;
    const priceMove = maxLoss / size;
    const liquidationPrice = side === 'long'
      ? Math.max(0, entryPrice - priceMove)
      : entryPrice + priceMove;
    
    const liquidationDistance = Math.abs(entryPrice - liquidationPrice) / entryPrice;

    // Risk Score Calculation (0-100)
    let riskScore = 0;

    // Leverage risk (0-30 points)
    riskScore += Math.min(30, (leverage / this.config.maxLeverage) * 30);

    // Liquidation distance risk (0-30 points)
    const liqDistRisk = 1 - (liquidationDistance / this.config.minLiquidationDistance);
    riskScore += Math.max(0, Math.min(30, liqDistRisk * 30));

    // Position size risk (0-20 points)
    riskScore += Math.min(20, (positionSizePercent / this.config.maxPositionSizePercent) * 20);

    // Funding rate risk (0-20 points)
    if (fundingRateApr > 0) {
      riskScore += Math.min(20, (fundingRateApr / this.config.maxFundingRateApr) * 20);
    }

    assessment.riskScore = Math.round(riskScore);

    // Check blockers
    if (leverage > this.config.maxLeverage) {
      assessment.blockers.push(
        `Leverage ${leverage}x exceeds maximum ${this.config.maxLeverage}x`
      );
      assessment.adjustedParams = { leverage: this.config.maxLeverage };
    }

    if (liquidationDistance < this.config.autoDeleverageThreshold) {
      assessment.blockers.push(
        `Liquidation distance ${(liquidationDistance * 100).toFixed(1)}% is below threshold`
      );
    }

    if (positionSizePercent > this.config.maxPositionSizePercent) {
      assessment.blockers.push(
        `Position size ${(positionSizePercent * 100).toFixed(1)}% exceeds max ${(this.config.maxPositionSizePercent * 100).toFixed(0)}%`
      );
      const adjustedSize = (this.portfolioValue * this.config.maxPositionSizePercent) / entryPrice;
      assessment.adjustedParams = { ...assessment.adjustedParams, size: adjustedSize };
    }

    // Check warnings
    if (liquidationDistance < this.config.minLiquidationDistance) {
      assessment.warnings.push(
        `Liquidation distance ${(liquidationDistance * 100).toFixed(1)}% is below recommended ${(this.config.minLiquidationDistance * 100).toFixed(0)}%`
      );
    }

    if (fundingRateApr > this.config.maxFundingRateApr) {
      assessment.warnings.push(
        `Funding rate ${(fundingRateApr * 100).toFixed(1)}% APR is high`
      );
    }

    // Recommendations
    if (leverage > 3) {
      assessment.recommendations.push('Consider using stop-loss orders');
    }

    if (riskScore > 70) {
      assessment.recommendations.push('High risk position - consider reducing size or leverage');
    }

    // Set stop loss and take profit recommendations
    assessment.adjustedParams = {
      ...assessment.adjustedParams,
      stopLoss: side === 'long'
        ? entryPrice * (1 - this.config.defaultStopLossPercent)
        : entryPrice * (1 + this.config.defaultStopLossPercent),
      takeProfit: side === 'long'
        ? entryPrice * (1 + this.config.defaultTakeProfitPercent)
        : entryPrice * (1 - this.config.defaultTakeProfitPercent),
    };

    // Final approval
    assessment.approved = assessment.blockers.length === 0;

    logger.info('Position risk assessment', {
      venue, market, side, leverage,
      riskScore: assessment.riskScore,
      approved: assessment.approved,
      liquidationDistance: `${(liquidationDistance * 100).toFixed(1)}%`,
      warnings: assessment.warnings.length,
      blockers: assessment.blockers.length,
    });

    return assessment;
  }

  /**
   * Assess risk status of an existing position
   */
  assessPositionRisk(position: PerpsPosition): PositionRiskStatus {
    const { side, size, entryPrice, markPrice, collateral, leverage, venue, market } = position;

    // Calculate metrics
    const notional = size * markPrice;
    const unrealizedPnl = side === 'long'
      ? (markPrice - entryPrice) * size
      : (entryPrice - markPrice) * size;
    const unrealizedPnlPercent = unrealizedPnl / collateral;

    // Calculate current margin ratio
    const equity = collateral + unrealizedPnl;
    const marginRatio = equity / notional;

    // Calculate liquidation distance
    const maintenanceMarginRatio = 0.05;
    const maintenanceMargin = notional * maintenanceMarginRatio;
    const maxLoss = equity - maintenanceMargin;
    const priceMove = maxLoss / size;
    const liquidationPrice = side === 'long'
      ? Math.max(0, markPrice - priceMove)
      : markPrice + priceMove;
    const liquidationDistance = Math.abs(markPrice - liquidationPrice) / markPrice;

    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'critical';
    let action: 'hold' | 'reduce' | 'close' | 'add_margin' | undefined;

    if (liquidationDistance < this.config.autoDeleverageThreshold) {
      riskLevel = 'critical';
      action = 'close';
    } else if (liquidationDistance < this.config.minLiquidationDistance) {
      riskLevel = 'high';
      action = 'reduce';
    } else if (marginRatio < this.config.minMarginRatio) {
      riskLevel = 'medium';
      action = 'add_margin';
    } else {
      riskLevel = 'low';
      action = 'hold';
    }

    return {
      position,
      riskLevel,
      liquidationDistance,
      marginRatio,
      unrealizedPnlPercent,
      fundingCostPercent: 0, // Would need funding data
      action,
    };
  }

  /**
   * Get aggregate risk metrics for all positions
   */
  getAggregateRiskMetrics(): RiskMetrics {
    const positions = Array.from(this.positions.values());

    if (positions.length === 0) {
      return {
        totalExposure: 0,
        totalCollateral: 0,
        totalUnrealizedPnl: 0,
        averageLeverage: 0,
        worstLiquidationDistance: 1,
        positionCount: 0,
        riskLevel: 'low',
      };
    }

    let totalExposure = 0;
    let totalCollateral = 0;
    let totalUnrealizedPnl = 0;
    let weightedLeverage = 0;
    let worstLiquidationDistance = 1;

    for (const pos of positions) {
      const notional = pos.size * pos.markPrice;
      totalExposure += notional;
      totalCollateral += pos.collateral;
      totalUnrealizedPnl += pos.unrealizedPnl;
      weightedLeverage += pos.leverage * notional;

      const status = this.assessPositionRisk(pos);
      if (status.liquidationDistance < worstLiquidationDistance) {
        worstLiquidationDistance = status.liquidationDistance;
      }
    }

    const averageLeverage = totalExposure > 0 ? weightedLeverage / totalExposure : 0;

    // Determine overall risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'critical';
    if (worstLiquidationDistance < this.config.autoDeleverageThreshold) {
      riskLevel = 'critical';
    } else if (worstLiquidationDistance < this.config.minLiquidationDistance) {
      riskLevel = 'high';
    } else if (averageLeverage > this.config.maxLeverage * 0.8) {
      riskLevel = 'medium';
    } else {
      riskLevel = 'low';
    }

    return {
      totalExposure,
      totalCollateral,
      totalUnrealizedPnl,
      averageLeverage,
      worstLiquidationDistance,
      positionCount: positions.length,
      riskLevel,
    };
  }

  /**
   * Calculate optimal position size based on risk parameters
   */
  calculateOptimalPositionSize(params: {
    entryPrice: number;
    leverage: number;
    stopLossPercent?: number;
    riskPerTradePercent?: number;
  }): { size: number; collateral: number; notional: number } {
    const {
      entryPrice,
      leverage,
      stopLossPercent = this.config.defaultStopLossPercent,
      riskPerTradePercent = 0.02, // 2% risk per trade
    } = params;

    // Risk amount in USD
    const riskAmount = this.portfolioValue * riskPerTradePercent;

    // Position size based on stop loss
    const size = riskAmount / (entryPrice * stopLossPercent);
    const notional = size * entryPrice;
    const collateral = notional / leverage;

    // Check against max position size
    const maxNotional = this.portfolioValue * this.config.maxPositionSizePercent;
    if (notional > maxNotional) {
      const adjustedSize = maxNotional / entryPrice;
      return {
        size: adjustedSize,
        collateral: maxNotional / leverage,
        notional: maxNotional,
      };
    }

    return { size, collateral, notional };
  }

  /**
   * Get positions that need attention
   */
  getPositionsNeedingAttention(): PositionRiskStatus[] {
    const positions = Array.from(this.positions.values());
    return positions
      .map(pos => this.assessPositionRisk(pos))
      .filter(status => status.riskLevel !== 'low')
      .sort((a, b) => {
        const riskOrder = { critical: 0, high: 1, medium: 2, low: 3 };
        return riskOrder[a.riskLevel] - riskOrder[b.riskLevel];
      });
  }
}

// ============= SINGLETON =============

let riskManagerInstance: PerpsRiskManager | null = null;

export function getPerpsRiskManager(config?: Partial<PerpsRiskConfig>): PerpsRiskManager {
  if (!riskManagerInstance) {
    riskManagerInstance = new PerpsRiskManager(config);
  }
  return riskManagerInstance;
}

export function resetPerpsRiskManager(): void {
  riskManagerInstance = null;
}
