/**
 * ML Services Module
 *
 * Exports calibration, model registry, and other ML-related services.
 */

export {
  CalibrationService,
  getCalibrationService,
  calibrationService,
  type PlattParameters,
  type CalibrationConfig,
  type CalibrationBin,
  type CalibrationEvaluation,
} from './calibration.js';

export {
  ModelRegistry,
  modelRegistry,
  type ModelVersion,
  type ModelMetrics,
  type PredictionLog,
  type VersionComparison,
} from './modelRegistry.js';

export {
  getVersionPerformance,
  compareModelVersions,
  getModelsSummary,
  type VersionPerformance,
  type ComparisonReport,
} from './versionComparison.js';

// Performance Monitoring
export {
  ModelPerformanceMonitor,
  getPerformanceMonitor,
  performanceMonitor,
  type PredictionOutcome,
  type RollingMetrics,
  type ModelPerformance,
  type PerformanceThresholds,
} from './performanceMonitor.js';

// Retraining Triggers
export {
  RetrainingTriggerEngine,
  DriftDetector,
  getRetrainingTriggerEngine,
  retrainingTriggerEngine,
  type TriggerType,
  type TriggerSeverity,
  type RetrainingTrigger,
  type DriftResult,
  type TriggerConfig,
} from './retrainingTriggers.js';

// Retraining Scheduler
export {
  RetrainingScheduler,
  getRetrainingScheduler,
  retrainingScheduler,
  type SchedulerConfig,
  type RetrainingJob,
  type SchedulerStatus,
} from './retrainingScheduler.js';

// Retraining Executor
export {
  RetrainingExecutor,
  getRetrainingExecutor,
  retrainingExecutor,
  type RetrainingResult,
  type TrainingScriptResult,
  type ExecutorConfig,
} from './retrainingExecutor.js';

// Model Validator
export {
  ModelValidator,
  getModelValidator,
  modelValidator,
  type MetricComparison,
  type ValidationResult,
  type ValidationConfig,
} from './modelValidator.js';

// Notification Service
export {
  NotificationService,
  getNotificationService,
  notificationService,
  type NotificationType,
  type NotificationChannel,
  type Notification,
  type NotificationConfig,
  type NotificationResult,
} from './notificationService.js';
