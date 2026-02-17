/**
 * Model Validator
 *
 * Validates new models before deployment by comparing their metrics
 * against the current production model. Ensures new models meet
 * minimum improvement thresholds and don't regress on key metrics.
 */

import { logger } from '../logger.js';
import type { ModelMetrics, ModelVersion } from './modelRegistry.js';
import type { RetrainingResult } from './retrainingExecutor.js';

// ============= TYPES =============

export interface MetricComparison {
  metric: string;
  oldValue: number;
  newValue: number;
  delta: number;
  deltaPercent: number;
  improved: boolean;
}

export interface ValidationResult {
  isValid: boolean;
  isBetter: boolean;
  metricsComparison: MetricComparison[];
  recommendation: 'DEPLOY' | 'KEEP_OLD' | 'REVIEW_NEEDED';
  reason: string;
  validatedAt: Date;
}

export interface ValidationConfig {
  minImprovementThreshold: number;   // Minimum improvement to consider "better" (e.g., 0.02 = 2%)
  maxRegressionThreshold: number;    // Max allowed regression on any metric (e.g., -0.05 = -5%)
  criticalMetrics: string[];          // Metrics that must not regress
  weightedMetrics: Record<string, number>; // Weights for overall score calculation
}

// ============= CONSTANTS =============

const DEFAULT_CONFIG: ValidationConfig = {
  minImprovementThreshold: 0.02,    // 2% improvement required
  maxRegressionThreshold: -0.05,    // -5% max regression allowed
  criticalMetrics: ['precision', 'sharpe'],
  weightedMetrics: {
    precision: 0.30,
    recall: 0.20,
    f1Score: 0.20,
    rocAuc: 0.15,
    sharpe: 0.15,
  },
};

// ============= MODEL VALIDATOR CLASS =============

class ModelValidator {
  private config: ValidationConfig;

  constructor(config: Partial<ValidationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[ModelValidator] Initialized', { config: this.config });
  }

  /**
   * Validate a new model against the current production model
   */
  async validateNewModel(
    oldVersion: ModelVersion | null,
    newResult: RetrainingResult
  ): Promise<ValidationResult> {
    logger.info('[ModelValidator] Validating new model', {
      oldVersion: oldVersion?.version,
      newVersion: newResult.version,
    });

    const oldMetrics = oldVersion?.metrics || this.getDefaultBaseline();
    const newMetrics = newResult.metrics;

    // Compare each metric
    const comparisons = this.compareMetrics(oldMetrics, newMetrics);

    // Check for critical regressions
    const criticalRegressions = this.checkCriticalRegressions(comparisons);
    
    // Calculate overall improvement score
    const overallScore = this.calculateOverallScore(comparisons);

    // Determine if new model is valid and better
    const isValid = criticalRegressions.length === 0;
    const isBetter = isValid && overallScore >= this.config.minImprovementThreshold;

    // Generate recommendation
    let recommendation: ValidationResult['recommendation'];
    let reason: string;

    if (!isValid) {
      recommendation = 'KEEP_OLD';
      reason = `Critical regressions detected: ${criticalRegressions.join(', ')}`;
    } else if (isBetter) {
      recommendation = 'DEPLOY';
      reason = `New model shows ${(overallScore * 100).toFixed(1)}% overall improvement`;
    } else if (overallScore > 0) {
      recommendation = 'REVIEW_NEEDED';
      reason = `Marginal improvement of ${(overallScore * 100).toFixed(1)}% - manual review recommended`;
    } else {
      recommendation = 'KEEP_OLD';
      reason = `No significant improvement (${(overallScore * 100).toFixed(1)}% change)`;
    }

    const result: ValidationResult = {
      isValid,
      isBetter,
      metricsComparison: comparisons,
      recommendation,
      reason,
      validatedAt: new Date(),
    };

    logger.info('[ModelValidator] Validation complete', {
      isValid,
      isBetter,
      recommendation,
      overallScore: overallScore.toFixed(4),
    });

    return result;
  }

  /**
   * Compare old and new metrics
   */
  private compareMetrics(oldMetrics: ModelMetrics, newMetrics: ModelMetrics): MetricComparison[] {
    const comparisons: MetricComparison[] = [];
    const metricsToCompare = ['precision', 'recall', 'f1Score', 'rocAuc', 'sharpe'];

    for (const metric of metricsToCompare) {
      const oldValue = (oldMetrics as any)[metric] ?? 0;
      const newValue = (newMetrics as any)[metric] ?? 0;
      const delta = newValue - oldValue;
      const deltaPercent = oldValue !== 0 ? delta / oldValue : (newValue > 0 ? 1 : 0);

      comparisons.push({
        metric,
        oldValue,
        newValue,
        delta,
        deltaPercent,
        improved: delta > 0,
      });
    }

    return comparisons;
  }

  /**
   * Check for critical metric regressions
   */
  private checkCriticalRegressions(comparisons: MetricComparison[]): string[] {
    const regressions: string[] = [];

    for (const comp of comparisons) {
      // Check if it's a critical metric
      if (this.config.criticalMetrics.includes(comp.metric)) {
        if (comp.deltaPercent < this.config.maxRegressionThreshold) {
          regressions.push(`${comp.metric}: ${(comp.deltaPercent * 100).toFixed(1)}%`);
        }
      }
    }

    return regressions;
  }

  /**
   * Calculate overall weighted improvement score
   */
  private calculateOverallScore(comparisons: MetricComparison[]): number {
    let totalWeight = 0;
    let weightedScore = 0;

    for (const comp of comparisons) {
      const weight = this.config.weightedMetrics[comp.metric] || 0;
      if (weight > 0) {
        totalWeight += weight;
        weightedScore += weight * comp.deltaPercent;
      }
    }

    return totalWeight > 0 ? weightedScore / totalWeight : 0;
  }

  /**
   * Get default baseline metrics when no previous model exists
   */
  private getDefaultBaseline(): ModelMetrics {
    return {
      precision: 0.65,
      recall: 0.60,
      f1Score: 0.62,
      rocAuc: 0.70,
      sharpe: 0.8,
    };
  }

  /**
   * Validate model meets minimum production requirements
   */
  validateMinimumRequirements(metrics: ModelMetrics): { valid: boolean; issues: string[] } {
    const issues: string[] = [];
    const minimums = {
      precision: 0.50,
      recall: 0.40,
      f1Score: 0.45,
      rocAuc: 0.55,
    };

    for (const [metric, minValue] of Object.entries(minimums)) {
      const actualValue = (metrics as any)[metric];
      if (actualValue !== undefined && actualValue < minValue) {
        issues.push(`${metric} (${actualValue.toFixed(3)}) below minimum (${minValue})`);
      }
    }

    return {
      valid: issues.length === 0,
      issues,
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ValidationConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('[ModelValidator] Config updated', { config: this.config });
  }
}

// ============= SINGLETON =============

let validatorInstance: ModelValidator | null = null;

export function getModelValidator(config?: Partial<ValidationConfig>): ModelValidator {
  if (!validatorInstance) {
    validatorInstance = new ModelValidator(config);
  }
  return validatorInstance;
}

export const modelValidator = getModelValidator();

export { ModelValidator };

