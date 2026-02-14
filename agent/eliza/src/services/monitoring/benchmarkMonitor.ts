/**
 * Benchmark Monitor
 *
 * Automated monitoring service that:
 * - Periodically checks all benchmarks
 * - Pauses trading on critical violations
 * - Sends notifications to configured channels
 * - Tracks violation history
 */

import { logger } from '../logger.js';
import { notificationService } from '../ml/notificationService.js';
import { benchmarkChecker, TradeRecord } from './benchmarkChecker.js';
import {
  BenchmarksConfig,
  BenchmarkViolation,
  BenchmarkStatus,
  MetricStatusInfo,
  MetricStatus,
} from './benchmarkTypes.js';
import { loadBenchmarksConfig } from './configLoader.js';

// ============= BENCHMARK MONITOR CLASS =============

class BenchmarkMonitor {
  private config: BenchmarksConfig | null = null;
  private checkInterval: NodeJS.Timeout | null = null;
  private tradingPaused = false;
  private pauseReason: string | undefined;
  private violationHistory: BenchmarkViolation[] = [];
  private lastCheck: Date | null = null;
  private lastAlertTime: Map<string, Date> = new Map();

  constructor() {
    logger.info('[BenchmarkMonitor] Initialized');
  }

  /**
   * Start the benchmark monitoring service
   */
  async start(): Promise<void> {
    this.config = await loadBenchmarksConfig();

    if (!this.config.enabled) {
      logger.info('[BenchmarkMonitor] Benchmarks disabled in config');
      return;
    }

    const intervalMs = this.config.check_interval_hours * 60 * 60 * 1000;

    // Set up periodic check
    this.checkInterval = setInterval(async () => {
      await this.runBenchmarkCheck();
    }, intervalMs);

    // Run immediately on start
    await this.runBenchmarkCheck();

    logger.info('[BenchmarkMonitor] Started', {
      checkIntervalHours: this.config.check_interval_hours,
    });
  }

  /**
   * Stop the monitoring service
   */
  stop(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    logger.info('[BenchmarkMonitor] Stopped');
  }

  /**
   * Run a benchmark check
   */
  async runBenchmarkCheck(): Promise<BenchmarkViolation[]> {
    logger.info('[BenchmarkMonitor] Running benchmark check');

    const violations = await benchmarkChecker.checkAllBenchmarks();
    this.lastCheck = new Date();

    if (violations.length === 0) {
      logger.info('[BenchmarkMonitor] All benchmarks met ✓');
      return [];
    }

    // Store violations
    this.violationHistory.push(...violations);
    this.trimViolationHistory();

    // Log violations
    for (const violation of violations) {
      if (violation.severity === 'CRITICAL') {
        logger.error('[BenchmarkMonitor] CRITICAL violation', {
          type: violation.type,
          metric: violation.metric,
          current: violation.current,
          benchmark: violation.benchmark,
          message: violation.message,
        });
      } else {
        logger.warn('[BenchmarkMonitor] WARNING', {
          type: violation.type,
          metric: violation.metric,
          message: violation.message,
        });
      }
    }

    // Handle critical violations
    const criticalViolations = violations.filter(v => v.severity === 'CRITICAL');
    if (criticalViolations.length > 0 && this.config?.alerts.pause_trading_on_critical) {
      await this.pauseTrading(criticalViolations);
    }

    // Send notifications (with cooldown)
    await this.sendNotifications(violations);

    return violations;
  }

  /**
   * Pause all trading due to benchmark violations
   */
  private async pauseTrading(violations: BenchmarkViolation[]): Promise<void> {
    if (this.tradingPaused) {
      logger.debug('[BenchmarkMonitor] Trading already paused');
      return;
    }

    this.tradingPaused = true;
    this.pauseReason = violations.map(v => v.message).join('; ');

    logger.error('[BenchmarkMonitor] PAUSING TRADING', {
      violationCount: violations.length,
      reasons: violations.map(v => v.message),
    });

    // Notify about pause
    await notificationService.notify({
      type: 'PERFORMANCE_DEGRADATION',
      modelName: 'system',
      message: `Trading paused due to ${violations.length} critical benchmark violation(s)`,
      details: {
        violations: violations.map(v => ({
          type: v.type,
          metric: v.metric,
          current: v.current,
          benchmark: v.benchmark,
        })),
      },
    });
  }

  /**
   * Resume trading (requires manual approval if configured)
   */
  async resumeTrading(approvedBy?: string): Promise<boolean> {
    if (!this.tradingPaused) {
      logger.info('[BenchmarkMonitor] Trading not paused');
      return true;
    }

    if (this.config?.alerts.resume_requires_manual_approval && !approvedBy) {
      logger.warn('[BenchmarkMonitor] Manual approval required to resume trading');
      return false;
    }

    this.tradingPaused = false;
    this.pauseReason = undefined;

    logger.info('[BenchmarkMonitor] Trading resumed', { approvedBy });

    await notificationService.notify({
      type: 'RETRAINING_COMPLETED',  // Reuse existing notification type
      modelName: 'system',
      message: `Trading resumed by ${approvedBy || 'system'}`,
    });

    return true;
  }

  /**
   * Check if trading is paused
   */
  isTradingPaused(): boolean {
    return this.tradingPaused;
  }

  /**
   * Get pause reason
   */
  getPauseReason(): string | undefined {
    return this.pauseReason;
  }

  /**
   * Send notifications with cooldown
   */
  private async sendNotifications(violations: BenchmarkViolation[]): Promise<void> {
    const channels = this.config?.alerts.notification_channels || ['log'];
    const cooldownMs = (this.config?.alerts.alert_cooldown_hours || 4) * 60 * 60 * 1000;

    for (const violation of violations) {
      const key = `${violation.type}:${violation.metric}:${violation.modelOrStrategy || 'system'}`;
      const lastAlert = this.lastAlertTime.get(key);

      // Check cooldown
      if (lastAlert && Date.now() - lastAlert.getTime() < cooldownMs) {
        logger.debug('[BenchmarkMonitor] Skipping notification (cooldown)', { key });
        continue;
      }

      // Send notification
      for (const channel of channels) {
        if (channel === 'log') {
          // Already logged above
          continue;
        }

        await notificationService.notify({
          type: 'PERFORMANCE_DEGRADATION',
          modelName: violation.modelOrStrategy || 'system',
          message: violation.message,
          details: {
            type: violation.type,
            metric: violation.metric,
            current: violation.current,
            benchmark: violation.benchmark,
            severity: violation.severity,
          },
        });
      }

      this.lastAlertTime.set(key, new Date());
    }
  }

  /**
   * Trim violation history to last 1000 entries
   */
  private trimViolationHistory(): void {
    const maxHistory = 1000;
    if (this.violationHistory.length > maxHistory) {
      this.violationHistory = this.violationHistory.slice(-maxHistory);
    }
  }

  /**
   * Get current benchmark status
   */
  async getStatus(): Promise<BenchmarkStatus> {
    const config = this.config || await loadBenchmarksConfig();
    const systemMetrics = await benchmarkChecker.getSystemMetrics();

    // Build model status
    const models: BenchmarkStatus['models'] = {};
    for (const modelName of Object.keys(config.models)) {
      const benchmarks = config.models[modelName];
      const performance = await (await import('../ml/performanceMonitor.js')).performanceMonitor.checkModelPerformance(modelName);
      const metrics = performance.currentMetrics;

      models[modelName] = {
        precision: this.createMetricStatus(metrics.precision, benchmarks.min_precision),
        recall: this.createMetricStatus(metrics.recall, benchmarks.min_recall),
        sharpe: this.createMetricStatus(metrics.sharpe, benchmarks.min_sharpe),
        winRate: this.createMetricStatus(metrics.winRate, benchmarks.min_win_rate),
      };
    }

    // Build system status
    const system: BenchmarkStatus['system'] = {
      dailyDrawdown: this.createMetricStatus(
        systemMetrics.dailyDrawdown, config.system.max_daily_drawdown, true
      ),
      weeklyDrawdown: this.createMetricStatus(
        systemMetrics.weeklyDrawdown, config.system.max_weekly_drawdown, true
      ),
      monthlyDrawdown: this.createMetricStatus(
        systemMetrics.monthlyDrawdown, config.system.max_monthly_drawdown, true
      ),
      dailyReturn: this.createMetricStatus(systemMetrics.dailyReturn, config.system.min_daily_return),
      weeklyReturn: this.createMetricStatus(systemMetrics.weeklyReturn, config.system.min_weekly_return),
      monthlyReturn: this.createMetricStatus(systemMetrics.monthlyReturn, config.system.min_monthly_return),
      consecutiveLosses: this.createMetricStatus(
        systemMetrics.consecutiveLosses, config.system.max_consecutive_losses, true
      ),
    };

    // Build strategy status
    const strategies: BenchmarkStatus['strategies'] = {};
    for (const strategyName of Object.keys(config.strategies)) {
      const benchmarks = config.strategies[strategyName];
      const metrics = await benchmarkChecker.getStrategyMetrics(strategyName);

      strategies[strategyName] = {
        monthlyTrades: this.createMetricStatus(metrics.monthlyTrades, benchmarks.min_monthly_trades),
        winRate: this.createMetricStatus(metrics.winRate, benchmarks.min_win_rate),
        avgLoss: this.createMetricStatus(metrics.avgLoss, benchmarks.max_avg_loss, true),
      };
    }

    return {
      models,
      system,
      strategies,
      tradingPaused: this.tradingPaused,
      pauseReason: this.pauseReason,
      lastCheck: this.lastCheck || new Date(),
      violations: this.violationHistory.slice(-50),  // Last 50 violations
    };
  }

  /**
   * Create metric status info
   */
  private createMetricStatus(
    current: number,
    benchmark: number,
    isMaxLimit = false
  ): MetricStatusInfo {
    let status: MetricStatus;

    if (isMaxLimit) {
      // For max limits (drawdown, losses), current should be < benchmark
      status = current > benchmark ? 'FAIL' : current > benchmark * 0.9 ? 'WARNING' : 'PASS';
    } else {
      // For min limits, current should be > benchmark
      status = current < benchmark ? 'FAIL' : current < benchmark * 1.1 ? 'WARNING' : 'PASS';
    }

    return {
      current,
      benchmark,
      status,
      deviation: Math.abs(current - benchmark),
    };
  }

  /**
   * Track a trade for benchmark calculation
   */
  trackTrade(trade: TradeRecord): void {
    benchmarkChecker.trackTrade(trade);
  }

  /**
   * Get violation history
   */
  getViolationHistory(): BenchmarkViolation[] {
    return [...this.violationHistory];
  }

  /**
   * Clear violation history
   */
  clearViolationHistory(): void {
    this.violationHistory = [];
    logger.info('[BenchmarkMonitor] Violation history cleared');
  }
}

// ============= SINGLETON =============

let benchmarkMonitorInstance: BenchmarkMonitor | null = null;

export function getBenchmarkMonitor(): BenchmarkMonitor {
  if (!benchmarkMonitorInstance) {
    benchmarkMonitorInstance = new BenchmarkMonitor();
  }
  return benchmarkMonitorInstance;
}

export const benchmarkMonitor = getBenchmarkMonitor();

export { BenchmarkMonitor };

