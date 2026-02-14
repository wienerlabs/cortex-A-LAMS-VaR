/**
 * Model Version Comparison Utilities
 * 
 * Provides utilities for comparing model versions, analyzing prediction logs,
 * and generating version comparison reports.
 */
import { modelRegistry, type PredictionLog } from './modelRegistry.js';
import { logger } from '../logger.js';

/**
 * Performance summary for a model version
 */
export interface VersionPerformance {
  modelName: string;
  version: string;
  trainedAt: Date;
  deployedAt?: Date;
  metrics: {
    precision?: number;
    recall?: number;
    f1Score?: number;
    rocAuc?: number;
    ece?: number;
  };
  predictionStats: {
    totalPredictions: number;
    avgConfidence: number;
    avgLatencyMs: number;
    positiveRate: number;
  };
  status: string;
}

/**
 * Comparison report between two versions
 */
export interface ComparisonReport {
  modelName: string;
  baseVersion: VersionPerformance;
  compareVersion: VersionPerformance;
  metricsDelta: Record<string, number>;
  recommendation: 'keep_base' | 'upgrade' | 'needs_more_data';
  analysis: string;
}

/**
 * Get performance summary for a model version
 */
export function getVersionPerformance(modelName: string, version: string): VersionPerformance | null {
  const modelVersion = modelRegistry.getVersion(modelName, version);
  if (!modelVersion) {
    return null;
  }

  // Get prediction stats from logs
  const logs = modelRegistry.getPredictionLogs(modelName, version);
  const predictionStats = computePredictionStats(logs);

  return {
    modelName,
    version,
    trainedAt: modelVersion.trainedAt,
    deployedAt: modelVersion.deployedAt,
    metrics: {
      precision: modelVersion.metrics.precision,
      recall: modelVersion.metrics.recall,
      f1Score: modelVersion.metrics.f1Score,
      rocAuc: modelVersion.metrics.rocAuc,
      ece: modelVersion.metrics.ece,
    },
    predictionStats,
    status: modelVersion.status,
  };
}

/**
 * Compute prediction statistics from logs
 */
function computePredictionStats(logs: PredictionLog[]): VersionPerformance['predictionStats'] {
  if (logs.length === 0) {
    return {
      totalPredictions: 0,
      avgConfidence: 0,
      avgLatencyMs: 0,
      positiveRate: 0,
    };
  }

  const totalPredictions = logs.length;
  const avgConfidence = logs.reduce((sum, l) => sum + l.confidence, 0) / totalPredictions;
  const avgLatencyMs = logs.reduce((sum, l) => sum + (l.latencyMs || 0), 0) / totalPredictions;
  const positiveRate = logs.filter(l => l.prediction === 1).length / totalPredictions;

  return {
    totalPredictions,
    avgConfidence,
    avgLatencyMs,
    positiveRate,
  };
}

/**
 * Compare two model versions and generate a report
 */
export function compareModelVersions(
  modelName: string,
  baseVersion: string,
  compareVersion: string
): ComparisonReport | null {
  const basePerfomance = getVersionPerformance(modelName, baseVersion);
  const comparePerfomance = getVersionPerformance(modelName, compareVersion);

  if (!basePerfomance || !comparePerfomance) {
    logger.warn('Cannot compare versions - one or both not found', {
      modelName,
      baseVersion,
      compareVersion,
    });
    return null;
  }

  // Calculate deltas for each metric
  const metricsDelta: Record<string, number> = {};
  const baseMetrics = basePerfomance.metrics;
  const compareMetrics = comparePerfomance.metrics;

  for (const key of Object.keys(baseMetrics) as (keyof typeof baseMetrics)[]) {
    const baseVal = baseMetrics[key] ?? 0;
    const compareVal = compareMetrics[key] ?? 0;
    metricsDelta[key] = compareVal - baseVal;
  }

  // Determine recommendation
  let recommendation: ComparisonReport['recommendation'];
  let analysis: string;

  const precisionDelta = metricsDelta.precision ?? 0;
  const recallDelta = metricsDelta.recall ?? 0;
  const rocAucDelta = metricsDelta.rocAuc ?? 0;

  if (comparePerfomance.predictionStats.totalPredictions < 100) {
    recommendation = 'needs_more_data';
    analysis = `Insufficient prediction data for ${compareVersion}. Collect more predictions before making a decision.`;
  } else if (precisionDelta > 0.02 && recallDelta >= -0.05 && rocAucDelta >= 0) {
    recommendation = 'upgrade';
    analysis = `${compareVersion} shows improved precision (+${(precisionDelta * 100).toFixed(1)}%) with acceptable recall. Recommend upgrading.`;
  } else if (precisionDelta < -0.05 || recallDelta < -0.1) {
    recommendation = 'keep_base';
    analysis = `${compareVersion} shows degraded metrics. Keep current ${baseVersion}.`;
  } else {
    recommendation = 'needs_more_data';
    analysis = `Metrics are similar. Collect more data to make a confident decision.`;
  }

  return {
    modelName,
    baseVersion: basePerfomance,
    compareVersion: comparePerfomance,
    metricsDelta,
    recommendation,
    analysis,
  };
}

/**
 * Get summary of all models and their active versions
 */
export function getModelsSummary(): Record<string, VersionPerformance | null> {
  const summary: Record<string, VersionPerformance | null> = {};
  const registrySummary = modelRegistry.getSummary();

  for (const modelName of Object.keys(registrySummary)) {
    const activeVersion = modelRegistry.getActiveVersion(modelName);
    if (activeVersion) {
      summary[modelName] = getVersionPerformance(modelName, activeVersion.version);
    } else {
      summary[modelName] = null;
    }
  }

  return summary;
}

