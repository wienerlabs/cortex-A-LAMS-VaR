/**
 * Retraining Scheduler
 *
 * Orchestrates automated model retraining based on trigger evaluations.
 * Runs periodic checks (default: 24 hours) and initiates retraining when needed.
 */

import { logger } from '../logger.js';
import { modelRegistry } from './modelRegistry.js';
import { getRetrainingTriggerEngine, type RetrainingTrigger } from './retrainingTriggers.js';
import { getRetrainingExecutor, type RetrainingResult } from './retrainingExecutor.js';
import { getModelValidator, type ValidationResult } from './modelValidator.js';
import { getNotificationService, type NotificationType } from './notificationService.js';

// ============= TYPES =============

export interface SchedulerConfig {
  enabled: boolean;
  checkIntervalMs: number;
  models: string[];
  maxConcurrentRetraining: number;
  autoDeployOnSuccess: boolean;
}

export interface RetrainingJob {
  modelName: string;
  triggers: RetrainingTrigger[];
  startedAt: Date;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: RetrainingResult;
  validation?: ValidationResult;
  error?: string;
}

export interface SchedulerStatus {
  isRunning: boolean;
  lastCheckAt: Date | null;
  nextCheckAt: Date | null;
  activeJobs: Map<string, RetrainingJob>;
  completedJobs: RetrainingJob[];
}

// ============= CONSTANTS =============

const DEFAULT_CONFIG: SchedulerConfig = {
  enabled: true,
  checkIntervalMs: 24 * 60 * 60 * 1000,  // 24 hours
  models: ['perps', 'spot', 'lp', 'lending', 'arbitrage'],
  maxConcurrentRetraining: 1,
  autoDeployOnSuccess: true,
};

// ============= RETRAINING SCHEDULER CLASS =============

class RetrainingScheduler {
  private config: SchedulerConfig;
  private triggerEngine = getRetrainingTriggerEngine();
  private executor = getRetrainingExecutor();
  private validator = getModelValidator();
  private notificationService = getNotificationService();
  
  private intervalId: NodeJS.Timeout | null = null;
  private activeJobs: Map<string, RetrainingJob> = new Map();
  private completedJobs: RetrainingJob[] = [];
  private lastCheckAt: Date | null = null;
  private isRunning = false;

  constructor(config: Partial<SchedulerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[RetrainingScheduler] Initialized', { 
      checkIntervalHours: this.config.checkIntervalMs / (60 * 60 * 1000),
      models: this.config.models,
    });
  }

  /**
   * Start the scheduler
   */
  start(): void {
    if (this.intervalId) {
      logger.warn('[RetrainingScheduler] Already running');
      return;
    }

    if (!this.config.enabled) {
      logger.info('[RetrainingScheduler] Disabled by config');
      return;
    }

    this.isRunning = true;
    logger.info('[RetrainingScheduler] Starting', {
      intervalMs: this.config.checkIntervalMs,
      models: this.config.models,
    });

    // Run immediately
    this.checkAllModels().catch(error => {
      logger.error('[RetrainingScheduler] Initial check failed', { error: String(error) });
    });

    // Then run periodically
    this.intervalId = setInterval(() => {
      this.checkAllModels().catch(error => {
        logger.error('[RetrainingScheduler] Periodic check failed', { error: String(error) });
      });
    }, this.config.checkIntervalMs);
  }

  /**
   * Stop the scheduler
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isRunning = false;
    logger.info('[RetrainingScheduler] Stopped');
  }

  /**
   * Check all models for retraining needs
   */
  async checkAllModels(): Promise<void> {
    this.lastCheckAt = new Date();
    logger.info('[RetrainingScheduler] Checking all models', { models: this.config.models });

    for (const modelName of this.config.models) {
      // Skip if already retraining
      if (this.activeJobs.has(modelName)) {
        logger.info('[RetrainingScheduler] Skipping - retraining in progress', { modelName });
        continue;
      }

      // Check concurrent limit
      if (this.activeJobs.size >= this.config.maxConcurrentRetraining) {
        logger.info('[RetrainingScheduler] Max concurrent retraining reached', {
          active: this.activeJobs.size,
          max: this.config.maxConcurrentRetraining,
        });
        break;
      }

      try {
        await this.checkModel(modelName);
      } catch (error) {
        logger.error('[RetrainingScheduler] Model check failed', { 
          modelName, 
          error: String(error),
        });
      }
    }
  }

  /**
   * Check a single model for retraining needs
   */
  private async checkModel(modelName: string): Promise<void> {
    const triggers = await this.triggerEngine.evaluateTriggers(modelName);

    if (triggers.length === 0) {
      logger.debug('[RetrainingScheduler] No triggers for model', { modelName });
      return;
    }

    // Evaluate trigger severity
    const criticalTriggers = triggers.filter(t => t.severity === 'CRITICAL');
    const highTriggers = triggers.filter(t => t.severity === 'HIGH');
    const shouldRetrain = criticalTriggers.length > 0 || highTriggers.length >= 2;

    if (!shouldRetrain) {
      logger.info('[RetrainingScheduler] Triggers present but not severe enough', {
        modelName,
        critical: criticalTriggers.length,
        high: highTriggers.length,
      });
      return;
    }

    logger.warn('[RetrainingScheduler] Retraining triggered', {
      modelName,
      triggers: triggers.map(t => ({ type: t.type, severity: t.severity, reason: t.reason })),
    });

    await this.initiateRetraining(modelName, triggers);
  }

  /**
   * Initiate retraining for a model
   */
  async initiateRetraining(modelName: string, triggers: RetrainingTrigger[]): Promise<void> {
    const job: RetrainingJob = {
      modelName,
      triggers,
      startedAt: new Date(),
      status: 'running',
    };

    this.activeJobs.set(modelName, job);

    try {
      // 1. Notify PM that retraining started
      await this.notificationService.notify({
        type: 'RETRAINING_STARTED',
        modelName,
        message: `Retraining initiated for ${modelName}`,
        details: {
          triggers: triggers.map(t => ({ type: t.type, severity: t.severity, reason: t.reason })),
        },
      });

      // 2. Execute retraining
      logger.info('[RetrainingScheduler] Executing retraining', { modelName });
      const result = await this.executor.retrain(modelName);
      job.result = result;

      if (!result.success) {
        throw new Error(result.error || 'Retraining failed');
      }

      // 3. Validate new model
      logger.info('[RetrainingScheduler] Validating new model', { modelName });
      const oldVersion = modelRegistry.getActiveVersion(modelName) ?? null;
      const validation = await this.validator.validateNewModel(oldVersion, result);
      job.validation = validation;

      // 4. Deploy if better (and auto-deploy enabled)
      if (validation.isBetter && this.config.autoDeployOnSuccess) {
        logger.info('[RetrainingScheduler] Deploying new model', { modelName });
        await this.deployNewModel(modelName, result);

        await this.notificationService.notify({
          type: 'NEW_MODEL_DEPLOYED',
          modelName,
          message: `New model deployed for ${modelName}`,
          details: {
            version: result.version,
            metrics: result.metrics,
            improvement: validation.metricsComparison,
          },
        });
      } else if (!validation.isBetter) {
        logger.warn('[RetrainingScheduler] New model not better, keeping old', {
          modelName,
          recommendation: validation.recommendation,
        });

        await this.notificationService.notify({
          type: 'RETRAINING_COMPLETED',
          modelName,
          message: `Retraining completed but new model not deployed (not better)`,
          details: {
            validation,
          },
        });
      }

      job.status = 'completed';

    } catch (error) {
      job.status = 'failed';
      job.error = String(error);

      logger.error('[RetrainingScheduler] Retraining failed', {
        modelName,
        error: String(error),
      });

      await this.notificationService.notify({
        type: 'RETRAINING_FAILED',
        modelName,
        message: `Retraining failed for ${modelName}: ${String(error)}`,
        details: { error: String(error) },
      });

    } finally {
      this.activeJobs.delete(modelName);
      this.completedJobs.push(job);

      // Keep only last 100 completed jobs
      if (this.completedJobs.length > 100) {
        this.completedJobs.splice(0, this.completedJobs.length - 100);
      }
    }
  }

  /**
   * Deploy a new model version
   */
  private async deployNewModel(modelName: string, result: RetrainingResult): Promise<void> {
    const newVersion = {
      modelName,
      version: result.version,
      onnxPath: result.modelPath,
      metadataPath: result.metadataPath,
      calibrationPath: result.calibrationPath,
      trainedAt: new Date(),
      deployedAt: new Date(),
      metrics: result.metrics,
      status: 'active' as const,
    };

    modelRegistry.addVersion(newVersion);
    logger.info('[RetrainingScheduler] New model version deployed', {
      modelName,
      version: result.version,
    });
  }

  /**
   * Manually trigger retraining for a model
   */
  async triggerManualRetraining(modelName: string, reason: string): Promise<void> {
    this.triggerEngine.addManualTrigger(modelName, reason);

    // Run check immediately
    await this.checkModel(modelName);
  }

  /**
   * Get scheduler status
   */
  getStatus(): SchedulerStatus {
    return {
      isRunning: this.isRunning,
      lastCheckAt: this.lastCheckAt,
      nextCheckAt: this.lastCheckAt
        ? new Date(this.lastCheckAt.getTime() + this.config.checkIntervalMs)
        : null,
      activeJobs: new Map(this.activeJobs),
      completedJobs: [...this.completedJobs],
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SchedulerConfig>): void {
    const wasRunning = this.isRunning;

    if (wasRunning) {
      this.stop();
    }

    this.config = { ...this.config, ...config };
    logger.info('[RetrainingScheduler] Config updated', { config: this.config });

    if (wasRunning && this.config.enabled) {
      this.start();
    }
  }
}

// ============= SINGLETON =============

let schedulerInstance: RetrainingScheduler | null = null;

export function getRetrainingScheduler(config?: Partial<SchedulerConfig>): RetrainingScheduler {
  if (!schedulerInstance) {
    schedulerInstance = new RetrainingScheduler(config);
  }
  return schedulerInstance;
}

export const retrainingScheduler = getRetrainingScheduler();

export { RetrainingScheduler };

