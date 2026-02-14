/**
 * Model Performance Monitor
 *
 * Tracks ML model performance metrics over time and detects degradation.
 * Uses rolling windows to calculate live metrics from prediction outcomes.
 */

import { logger } from '../logger.js';
import { modelRegistry, type ModelMetrics } from './modelRegistry.js';

// ============= TYPES =============

export interface PredictionOutcome {
  modelName: string;
  modelVersion: string;
  prediction: number;
  confidence: number;
  actual: boolean;
  profitLoss?: number;
  timestamp: Date;
}

export interface RollingMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  winRate: number;
  sharpe: number;
  totalPredictions: number;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  avgConfidence: number;
  windowStart: Date;
  windowEnd: Date;
}

export interface ModelPerformance {
  modelName: string;
  modelVersion: string;
  currentMetrics: RollingMetrics;
  baselineMetrics: ModelMetrics;
  degradation: {
    precision: number;
    recall: number;
    sharpe: number;
    winRate: number;
  };
  needsRetraining: boolean;
  reason?: string;
  lastChecked: Date;
}

export interface PerformanceThresholds {
  precisionDropThreshold: number;
  recallDropThreshold: number;
  sharpeDropThreshold: number;
  winRateDropThreshold: number;
  minPredictionsForEvaluation: number;
}

// ============= CONSTANTS =============

const DEFAULT_THRESHOLDS: PerformanceThresholds = {
  precisionDropThreshold: 0.10,  // 10% drop triggers retraining
  recallDropThreshold: 0.10,
  sharpeDropThreshold: 0.20,
  winRateDropThreshold: 0.15,
  minPredictionsForEvaluation: 50,
};

// ============= PERFORMANCE MONITOR CLASS =============

class ModelPerformanceMonitor {
  private outcomes: Map<string, PredictionOutcome[]> = new Map();
  private thresholds: PerformanceThresholds;
  private maxOutcomesPerModel = 10000;

  constructor(thresholds: Partial<PerformanceThresholds> = {}) {
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...thresholds };
    logger.info('[PerformanceMonitor] Initialized', { thresholds: this.thresholds });
  }

  /**
   * Track a prediction outcome (called when we know the actual result)
   */
  trackPrediction(outcome: PredictionOutcome): void {
    const key = outcome.modelName;
    const outcomes = this.outcomes.get(key) || [];
    
    outcomes.push(outcome);
    
    // Trim to max size (keep most recent)
    if (outcomes.length > this.maxOutcomesPerModel) {
      outcomes.splice(0, outcomes.length - this.maxOutcomesPerModel);
    }
    
    this.outcomes.set(key, outcomes);
    
    logger.debug('[PerformanceMonitor] Tracked prediction outcome', {
      modelName: outcome.modelName,
      prediction: outcome.prediction,
      actual: outcome.actual,
      correct: (outcome.prediction === 1) === outcome.actual,
    });
  }

  /**
   * Calculate rolling metrics for a model over a window of trades
   */
  calculateRollingMetrics(modelName: string, windowSize: number = 100): RollingMetrics | null {
    const outcomes = this.outcomes.get(modelName) || [];
    
    if (outcomes.length < this.thresholds.minPredictionsForEvaluation) {
      logger.debug('[PerformanceMonitor] Insufficient data for metrics', {
        modelName,
        available: outcomes.length,
        required: this.thresholds.minPredictionsForEvaluation,
      });
      return null;
    }

    // Get last N outcomes
    const window = outcomes.slice(-windowSize);
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    let totalConfidence = 0;
    const profitLosses: number[] = [];

    for (const o of window) {
      const predicted = o.prediction === 1;
      const actual = o.actual;
      
      if (predicted && actual) tp++;
      else if (predicted && !actual) fp++;
      else if (!predicted && !actual) tn++;
      else fn++;
      
      totalConfidence += o.confidence;
      if (o.profitLoss !== undefined) {
        profitLosses.push(o.profitLoss);
      }
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const winRate = window.length > 0 ? (tp + tn) / window.length : 0;
    
    // Calculate Sharpe ratio from profit/loss
    const sharpe = this.calculateSharpe(profitLosses);

    return {
      precision,
      recall,
      f1Score,
      winRate,
      sharpe,
      totalPredictions: window.length,
      truePositives: tp,
      falsePositives: fp,
      trueNegatives: tn,
      falseNegatives: fn,
      avgConfidence: totalConfidence / window.length,
      windowStart: window[0]?.timestamp || new Date(),
      windowEnd: window[window.length - 1]?.timestamp || new Date(),
    };
  }

  /**
   * Calculate Sharpe ratio from profit/loss array
   */
  private calculateSharpe(profitLosses: number[]): number {
    if (profitLosses.length < 2) return 0;

    const mean = profitLosses.reduce((a, b) => a + b, 0) / profitLosses.length;
    const variance = profitLosses.reduce((sum, pl) => sum + Math.pow(pl - mean, 2), 0) / profitLosses.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return mean > 0 ? 3 : 0;  // Cap at 3 if no variance

    // Annualized Sharpe (assuming daily returns, 252 trading days)
    return (mean / stdDev) * Math.sqrt(252);
  }

  /**
   * Check model performance and determine if retraining is needed
   */
  async checkModelPerformance(modelName: string, windowSize: number = 100): Promise<ModelPerformance> {
    const activeVersion = modelRegistry.getActiveVersion(modelName);
    const currentMetrics = this.calculateRollingMetrics(modelName, windowSize);

    const baselineMetrics: ModelMetrics = activeVersion?.metrics || {
      precision: 0.7,
      recall: 0.7,
      f1Score: 0.7,
      rocAuc: 0.75,
      sharpe: 1.0,
    };

    // If no current metrics, return with no degradation
    if (!currentMetrics) {
      return {
        modelName,
        modelVersion: activeVersion?.version || 'unknown',
        currentMetrics: this.getEmptyMetrics(),
        baselineMetrics,
        degradation: { precision: 0, recall: 0, sharpe: 0, winRate: 0 },
        needsRetraining: false,
        reason: 'Insufficient prediction data for evaluation',
        lastChecked: new Date(),
      };
    }

    // Calculate degradation (positive = worse, negative = better)
    const degradation = {
      precision: baselineMetrics.precision - currentMetrics.precision,
      recall: baselineMetrics.recall - currentMetrics.recall,
      sharpe: (baselineMetrics.sharpe || 1) - currentMetrics.sharpe,
      winRate: 0.7 - currentMetrics.winRate,  // Assume 70% baseline win rate
    };

    // Determine if retraining is needed
    let needsRetraining = false;
    let reason: string | undefined;

    if (degradation.precision > this.thresholds.precisionDropThreshold) {
      needsRetraining = true;
      reason = `Precision dropped by ${(degradation.precision * 100).toFixed(1)}%`;
    } else if (degradation.recall > this.thresholds.recallDropThreshold) {
      needsRetraining = true;
      reason = `Recall dropped by ${(degradation.recall * 100).toFixed(1)}%`;
    } else if (degradation.sharpe > this.thresholds.sharpeDropThreshold) {
      needsRetraining = true;
      reason = `Sharpe ratio dropped by ${degradation.sharpe.toFixed(2)}`;
    } else if (degradation.winRate > this.thresholds.winRateDropThreshold) {
      needsRetraining = true;
      reason = `Win rate dropped by ${(degradation.winRate * 100).toFixed(1)}%`;
    }

    const result: ModelPerformance = {
      modelName,
      modelVersion: activeVersion?.version || 'unknown',
      currentMetrics,
      baselineMetrics,
      degradation,
      needsRetraining,
      reason,
      lastChecked: new Date(),
    };

    logger.info('[PerformanceMonitor] Performance check complete', {
      modelName,
      needsRetraining,
      reason,
      precision: currentMetrics.precision.toFixed(3),
      recall: currentMetrics.recall.toFixed(3),
      winRate: currentMetrics.winRate.toFixed(3),
    });

    return result;
  }

  /**
   * Get empty metrics structure
   */
  private getEmptyMetrics(): RollingMetrics {
    return {
      precision: 0,
      recall: 0,
      f1Score: 0,
      winRate: 0,
      sharpe: 0,
      totalPredictions: 0,
      truePositives: 0,
      falsePositives: 0,
      trueNegatives: 0,
      falseNegatives: 0,
      avgConfidence: 0,
      windowStart: new Date(),
      windowEnd: new Date(),
    };
  }

  /**
   * Get all tracked outcomes for a model
   */
  getOutcomes(modelName: string): PredictionOutcome[] {
    return this.outcomes.get(modelName) || [];
  }

  /**
   * Clear outcomes for a model (after retraining)
   */
  clearOutcomes(modelName: string): void {
    this.outcomes.delete(modelName);
    logger.info('[PerformanceMonitor] Cleared outcomes', { modelName });
  }

  /**
   * Update thresholds
   */
  updateThresholds(thresholds: Partial<PerformanceThresholds>): void {
    this.thresholds = { ...this.thresholds, ...thresholds };
    logger.info('[PerformanceMonitor] Thresholds updated', { thresholds: this.thresholds });
  }
}

// ============= SINGLETON =============

let performanceMonitorInstance: ModelPerformanceMonitor | null = null;

export function getPerformanceMonitor(thresholds?: Partial<PerformanceThresholds>): ModelPerformanceMonitor {
  if (!performanceMonitorInstance) {
    performanceMonitorInstance = new ModelPerformanceMonitor(thresholds);
  }
  return performanceMonitorInstance;
}

export const performanceMonitor = getPerformanceMonitor();

export { ModelPerformanceMonitor };

