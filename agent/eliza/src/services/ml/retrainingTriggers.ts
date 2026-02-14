/**
 * Retraining Trigger Engine
 *
 * Evaluates multiple trigger conditions to determine if a model needs retraining:
 * - Performance degradation (precision, recall, sharpe drops)
 * - Time-based (days since last training)
 * - Data drift (PSI, KS statistics)
 * - Manual triggers
 */

import { logger } from '../logger.js';
import { modelRegistry } from './modelRegistry.js';
import { getPerformanceMonitor } from './performanceMonitor.js';

// ============= TYPES =============

export type TriggerType = 'PERFORMANCE' | 'TIME' | 'DRIFT' | 'MANUAL';
export type TriggerSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface RetrainingTrigger {
  type: TriggerType;
  severity: TriggerSeverity;
  shouldRetrain: boolean;
  reason: string;
  details?: Record<string, unknown>;
  timestamp: Date;
}

export interface DriftResult {
  psiScore: number;
  ksStatistic: number;
  driftedFeatures: string[];
  hasDrift: boolean;
}

export interface TriggerConfig {
  performance: {
    enabled: boolean;
    precisionDropThreshold: number;
    recallDropThreshold: number;
    sharpeDropThreshold: number;
    windowTrades: number;
  };
  time: {
    enabled: boolean;
    maxDaysSinceTraining: number;
  };
  drift: {
    enabled: boolean;
    psiThreshold: number;
    ksThreshold: number;
  };
}

// ============= CONSTANTS =============

const DEFAULT_CONFIG: TriggerConfig = {
  performance: {
    enabled: true,
    precisionDropThreshold: 0.10,
    recallDropThreshold: 0.10,
    sharpeDropThreshold: 0.20,
    windowTrades: 100,
  },
  time: {
    enabled: true,
    maxDaysSinceTraining: 30,
  },
  drift: {
    enabled: true,
    psiThreshold: 0.2,
    ksThreshold: 0.05,
  },
};

// ============= DRIFT DETECTOR =============

class DriftDetector {
  private featureDistributions: Map<string, Map<string, number[]>> = new Map();

  /**
   * Record feature values for drift detection
   */
  recordFeatures(modelName: string, features: Record<string, number>): void {
    let modelFeatures = this.featureDistributions.get(modelName);
    if (!modelFeatures) {
      modelFeatures = new Map();
      this.featureDistributions.set(modelName, modelFeatures);
    }

    for (const [name, value] of Object.entries(features)) {
      const values = modelFeatures.get(name) || [];
      values.push(value);
      // Keep last 10000 values
      if (values.length > 10000) {
        values.splice(0, values.length - 10000);
      }
      modelFeatures.set(name, values);
    }
  }

  /**
   * Check for data drift using PSI and KS statistics
   */
  async checkDrift(modelName: string): Promise<DriftResult> {
    const modelFeatures = this.featureDistributions.get(modelName);
    
    if (!modelFeatures || modelFeatures.size === 0) {
      return {
        psiScore: 0,
        ksStatistic: 0,
        driftedFeatures: [],
        hasDrift: false,
      };
    }

    let totalPsi = 0;
    let maxKs = 0;
    const driftedFeatures: string[] = [];
    let featureCount = 0;

    for (const [featureName, values] of modelFeatures) {
      if (values.length < 200) continue;  // Need enough data
      
      // Split into reference (first half) and current (second half)
      const midpoint = Math.floor(values.length / 2);
      const reference = values.slice(0, midpoint);
      const current = values.slice(midpoint);

      // Calculate PSI
      const psi = this.calculatePSI(reference, current);
      totalPsi += psi;

      // Calculate KS statistic
      const ks = this.calculateKS(reference, current);
      maxKs = Math.max(maxKs, ks);

      if (psi > 0.1 || ks > 0.05) {
        driftedFeatures.push(featureName);
      }

      featureCount++;
    }

    const avgPsi = featureCount > 0 ? totalPsi / featureCount : 0;

    return {
      psiScore: avgPsi,
      ksStatistic: maxKs,
      driftedFeatures,
      hasDrift: avgPsi > DEFAULT_CONFIG.drift.psiThreshold || maxKs > DEFAULT_CONFIG.drift.ksThreshold,
    };
  }

  /**
   * Calculate Population Stability Index (PSI)
   */
  private calculatePSI(reference: number[], current: number[], bins: number = 10): number {
    const refMin = Math.min(...reference);
    const refMax = Math.max(...reference);
    const binWidth = (refMax - refMin) / bins || 1;

    const refHist = new Array(bins).fill(0);
    const curHist = new Array(bins).fill(0);

    for (const val of reference) {
      const bin = Math.min(Math.floor((val - refMin) / binWidth), bins - 1);
      refHist[bin]++;
    }
    for (const val of current) {
      const bin = Math.min(Math.max(Math.floor((val - refMin) / binWidth), 0), bins - 1);
      curHist[bin]++;
    }

    // Normalize to proportions
    const refTotal = reference.length;
    const curTotal = current.length;

    let psi = 0;
    for (let i = 0; i < bins; i++) {
      const refProp = Math.max(refHist[i] / refTotal, 0.0001);
      const curProp = Math.max(curHist[i] / curTotal, 0.0001);
      psi += (curProp - refProp) * Math.log(curProp / refProp);
    }

    return psi;
  }

  /**
   * Calculate Kolmogorov-Smirnov statistic
   */
  private calculateKS(reference: number[], current: number[]): number {
    const refSorted = [...reference].sort((a, b) => a - b);
    const curSorted = [...current].sort((a, b) => a - b);

    const allValues = [...new Set([...refSorted, ...curSorted])].sort((a, b) => a - b);

    let maxDiff = 0;
    for (const val of allValues) {
      const refCdf = refSorted.filter(v => v <= val).length / refSorted.length;
      const curCdf = curSorted.filter(v => v <= val).length / curSorted.length;
      maxDiff = Math.max(maxDiff, Math.abs(refCdf - curCdf));
    }

    return maxDiff;
  }

  /**
   * Clear feature distributions for a model
   */
  clearDistributions(modelName: string): void {
    this.featureDistributions.delete(modelName);
  }
}

// ============= RETRAINING TRIGGER ENGINE =============

class RetrainingTriggerEngine {
  private config: TriggerConfig;
  private performanceMonitor = getPerformanceMonitor();
  private driftDetector = new DriftDetector();
  private manualTriggers: Map<string, RetrainingTrigger> = new Map();

  constructor(config: Partial<TriggerConfig> = {}) {
    this.config = {
      performance: { ...DEFAULT_CONFIG.performance, ...config.performance },
      time: { ...DEFAULT_CONFIG.time, ...config.time },
      drift: { ...DEFAULT_CONFIG.drift, ...config.drift },
    };
    logger.info('[RetrainingTriggers] Initialized', { config: this.config });
  }

  /**
   * Evaluate all triggers for a model
   */
  async evaluateTriggers(modelName: string): Promise<RetrainingTrigger[]> {
    const triggers: RetrainingTrigger[] = [];

    // 1. Performance degradation trigger
    if (this.config.performance.enabled) {
      const perfTrigger = await this.evaluatePerformanceTrigger(modelName);
      if (perfTrigger) triggers.push(perfTrigger);
    }

    // 2. Time-based trigger
    if (this.config.time.enabled) {
      const timeTrigger = await this.evaluateTimeTrigger(modelName);
      if (timeTrigger) triggers.push(timeTrigger);
    }

    // 3. Data drift trigger
    if (this.config.drift.enabled) {
      const driftTrigger = await this.evaluateDriftTrigger(modelName);
      if (driftTrigger) triggers.push(driftTrigger);
    }

    // 4. Manual trigger
    const manualTrigger = this.manualTriggers.get(modelName);
    if (manualTrigger) {
      triggers.push(manualTrigger);
      this.manualTriggers.delete(modelName);  // Clear after evaluation
    }

    logger.info('[RetrainingTriggers] Evaluated triggers', {
      modelName,
      triggerCount: triggers.length,
      types: triggers.map(t => t.type),
    });

    return triggers;
  }

  /**
   * Evaluate performance degradation trigger
   */
  private async evaluatePerformanceTrigger(modelName: string): Promise<RetrainingTrigger | null> {
    const perf = await this.performanceMonitor.checkModelPerformance(
      modelName,
      this.config.performance.windowTrades
    );

    if (!perf.needsRetraining) return null;

    // Determine severity based on degradation magnitude
    let severity: TriggerSeverity = 'MEDIUM';
    const maxDegradation = Math.max(
      perf.degradation.precision,
      perf.degradation.recall,
      perf.degradation.sharpe / 2
    );

    if (maxDegradation > 0.25) severity = 'CRITICAL';
    else if (maxDegradation > 0.15) severity = 'HIGH';
    else if (maxDegradation > 0.10) severity = 'MEDIUM';
    else severity = 'LOW';

    return {
      type: 'PERFORMANCE',
      severity,
      shouldRetrain: true,
      reason: perf.reason || 'Performance degradation detected',
      details: {
        degradation: perf.degradation,
        currentMetrics: perf.currentMetrics,
      },
      timestamp: new Date(),
    };
  }

  /**
   * Evaluate time-based trigger
   */
  private async evaluateTimeTrigger(modelName: string): Promise<RetrainingTrigger | null> {
    const activeVersion = modelRegistry.getActiveVersion(modelName);
    if (!activeVersion) return null;

    const daysSinceTraining = this.getDaysSinceDate(activeVersion.trainedAt);

    if (daysSinceTraining <= this.config.time.maxDaysSinceTraining) return null;

    // Severity based on how overdue
    let severity: TriggerSeverity = 'MEDIUM';
    if (daysSinceTraining > this.config.time.maxDaysSinceTraining * 2) severity = 'HIGH';
    if (daysSinceTraining > this.config.time.maxDaysSinceTraining * 3) severity = 'CRITICAL';

    return {
      type: 'TIME',
      severity,
      shouldRetrain: true,
      reason: `${daysSinceTraining} days since last training (max: ${this.config.time.maxDaysSinceTraining})`,
      details: {
        daysSinceTraining,
        trainedAt: activeVersion.trainedAt,
        maxDays: this.config.time.maxDaysSinceTraining,
      },
      timestamp: new Date(),
    };
  }

  /**
   * Evaluate data drift trigger
   */
  private async evaluateDriftTrigger(modelName: string): Promise<RetrainingTrigger | null> {
    const drift = await this.driftDetector.checkDrift(modelName);

    if (!drift.hasDrift) return null;

    // Severity based on drift magnitude
    let severity: TriggerSeverity = 'MEDIUM';
    if (drift.psiScore > 0.4 || drift.ksStatistic > 0.15) severity = 'CRITICAL';
    else if (drift.psiScore > 0.3 || drift.ksStatistic > 0.10) severity = 'HIGH';

    return {
      type: 'DRIFT',
      severity,
      shouldRetrain: true,
      reason: `Data drift detected (PSI: ${drift.psiScore.toFixed(3)}, KS: ${drift.ksStatistic.toFixed(3)})`,
      details: {
        psiScore: drift.psiScore,
        ksStatistic: drift.ksStatistic,
        driftedFeatures: drift.driftedFeatures,
      },
      timestamp: new Date(),
    };
  }

  /**
   * Add a manual retraining trigger
   */
  addManualTrigger(modelName: string, reason: string): void {
    this.manualTriggers.set(modelName, {
      type: 'MANUAL',
      severity: 'HIGH',
      shouldRetrain: true,
      reason: `Manual trigger: ${reason}`,
      timestamp: new Date(),
    });
    logger.info('[RetrainingTriggers] Manual trigger added', { modelName, reason });
  }

  /**
   * Record features for drift detection
   */
  recordFeatures(modelName: string, features: Record<string, number>): void {
    this.driftDetector.recordFeatures(modelName, features);
  }

  /**
   * Get days since a date
   */
  private getDaysSinceDate(date: Date): number {
    const now = new Date();
    const diffMs = now.getTime() - new Date(date).getTime();
    return Math.floor(diffMs / (1000 * 60 * 60 * 24));
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<TriggerConfig>): void {
    if (config.performance) {
      this.config.performance = { ...this.config.performance, ...config.performance };
    }
    if (config.time) {
      this.config.time = { ...this.config.time, ...config.time };
    }
    if (config.drift) {
      this.config.drift = { ...this.config.drift, ...config.drift };
    }
    logger.info('[RetrainingTriggers] Config updated', { config: this.config });
  }

  /**
   * Get drift detector for external access
   */
  getDriftDetector(): DriftDetector {
    return this.driftDetector;
  }
}

// ============= SINGLETON =============

let triggerEngineInstance: RetrainingTriggerEngine | null = null;

export function getRetrainingTriggerEngine(config?: Partial<TriggerConfig>): RetrainingTriggerEngine {
  if (!triggerEngineInstance) {
    triggerEngineInstance = new RetrainingTriggerEngine(config);
  }
  return triggerEngineInstance;
}

export const retrainingTriggerEngine = getRetrainingTriggerEngine();

export { RetrainingTriggerEngine, DriftDetector };

