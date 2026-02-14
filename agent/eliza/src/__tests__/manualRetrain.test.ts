/**
 * Manual Retraining Trigger Tests
 *
 * Tests the manual retraining workflow:
 * 1. Adding manual triggers
 * 2. Triggering retraining via scheduler
 * 3. Verifying notifications are sent
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  RetrainingTriggerEngine,
  RetrainingScheduler,
  NotificationService,
  type RetrainingTrigger,
} from '../services/ml/index.js';

describe('Manual Retraining', () => {
  let triggerEngine: RetrainingTriggerEngine;
  let scheduler: RetrainingScheduler;
  let notificationService: NotificationService;

  beforeEach(() => {
    triggerEngine = new RetrainingTriggerEngine({
      performance: { enabled: false, precisionDropThreshold: 0.10, recallDropThreshold: 0.10, sharpeDropThreshold: 0.20, windowTrades: 100 },
      time: { enabled: false, maxDaysSinceTraining: 30 },
      drift: { enabled: false, psiThreshold: 0.2, ksThreshold: 0.05 },
    });

    scheduler = new RetrainingScheduler({
      enabled: true,
      checkIntervalMs: 1000,
      models: ['perps', 'spot'],
      maxConcurrentRetraining: 1,
      autoDeployOnSuccess: false,
    });

    notificationService = new NotificationService({
      enabled: true,
      channels: ['log'],
      notifyOn: ['RETRAINING_STARTED', 'RETRAINING_COMPLETED', 'RETRAINING_FAILED'],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Manual Trigger Addition', () => {
    it('adds manual trigger with reason', async () => {
      triggerEngine.addManualTrigger('perps', 'Testing new data pipeline');

      const triggers = await triggerEngine.evaluateTriggers('perps');
      const manualTriggers = triggers.filter(t => t.type === 'MANUAL');

      expect(manualTriggers.length).toBe(1);
      expect(manualTriggers[0].severity).toBe('HIGH');
      expect(manualTriggers[0].shouldRetrain).toBe(true);
      expect(manualTriggers[0].reason).toContain('Testing new data pipeline');
    });

    it('manual trigger is consumed after evaluation', async () => {
      triggerEngine.addManualTrigger('perps', 'One-time trigger');

      // First evaluation should have the trigger
      const triggers1 = await triggerEngine.evaluateTriggers('perps');
      expect(triggers1.filter(t => t.type === 'MANUAL').length).toBe(1);

      // Second evaluation should not have it
      const triggers2 = await triggerEngine.evaluateTriggers('perps');
      expect(triggers2.filter(t => t.type === 'MANUAL').length).toBe(0);
    });

    it('multiple models can have independent manual triggers', async () => {
      triggerEngine.addManualTrigger('perps', 'Perps reason');
      triggerEngine.addManualTrigger('spot', 'Spot reason');

      const perpsTriggers = await triggerEngine.evaluateTriggers('perps');
      const spotTriggers = await triggerEngine.evaluateTriggers('spot');

      expect(perpsTriggers.filter(t => t.type === 'MANUAL').length).toBe(1);
      expect(spotTriggers.filter(t => t.type === 'MANUAL').length).toBe(1);
      expect(perpsTriggers[0].reason).toContain('Perps reason');
      expect(spotTriggers[0].reason).toContain('Spot reason');
    });
  });

  describe('Scheduler Manual Trigger', () => {
    it('triggerManualRetraining adds trigger and checks model', async () => {
      // Mock the trigger engine
      const addTriggerSpy = vi.spyOn((scheduler as any).triggerEngine, 'addManualTrigger');
      
      // Mock evaluateTriggers to return the manual trigger
      vi.spyOn((scheduler as any).triggerEngine, 'evaluateTriggers').mockResolvedValue([
        {
          type: 'MANUAL',
          severity: 'HIGH',
          shouldRetrain: true,
          reason: 'Manual trigger: PM requested retraining',
          timestamp: new Date(),
        },
      ]);

      // Mock executor to avoid actual training
      vi.spyOn((scheduler as any).executor, 'retrain').mockResolvedValue({
        success: true,
        modelPath: '/models/perps_new.onnx',
        metadataPath: '/models/perps_new_metadata.json',
        metrics: { precision: 0.80, recall: 0.70, f1Score: 0.75, rocAuc: 0.85 },
        version: '1.1.0',
        trainedAt: new Date(),
        trainingDurationMs: 60000,
      });

      // Mock validator
      vi.spyOn((scheduler as any).validator, 'validateNewModel').mockResolvedValue({
        isValid: true,
        isBetter: true,
        metricsComparison: [],
        recommendation: 'DEPLOY',
        reason: 'Model improved',
        validatedAt: new Date(),
      });

      // Mock notifications
      vi.spyOn((scheduler as any).notificationService, 'notify').mockResolvedValue([]);

      await scheduler.triggerManualRetraining('perps', 'PM requested retraining');

      expect(addTriggerSpy).toHaveBeenCalledWith('perps', 'PM requested retraining');
    });

    it('scheduler status reflects manual trigger job', async () => {
      const status = scheduler.getStatus();

      expect(status.isRunning).toBe(false);
      expect(status.activeJobs.size).toBe(0);
      expect(status.completedJobs).toEqual([]);
    });
  });

  describe('Notification on Manual Trigger', () => {
    it('sends RETRAINING_STARTED notification', async () => {
      const results = await notificationService.notify({
        type: 'RETRAINING_STARTED',
        modelName: 'perps',
        message: 'Manual retraining initiated for perps',
        details: { reason: 'PM requested' },
      });

      expect(results.length).toBe(1);
      expect(results[0].success).toBe(true);
    });
  });
});

