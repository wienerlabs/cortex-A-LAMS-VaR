/**
 * Benchmark Checker
 *
 * Validates current performance metrics against defined benchmarks.
 * Checks models, system, and strategy performance.
 * Returns violations when metrics fall below thresholds.
 */

import { logger } from '../logger.js';
import { performanceMonitor } from '../ml/performanceMonitor.js';
import {
  BenchmarksConfig,
  BenchmarkViolation,
  ViolationSeverity,
  SystemMetrics,
  StrategyMetrics,
} from './benchmarkTypes.js';
import { loadBenchmarksConfig } from './configLoader.js';

// ============= BENCHMARK CHECKER CLASS =============

class BenchmarkChecker {
  private config: BenchmarksConfig | null = null;
  private tradeHistory: TradeRecord[] = [];
  private maxHistorySize = 10000;

  constructor() {
    logger.info('[BenchmarkChecker] Initialized');
  }

  /**
   * Load configuration from YAML file
   */
  async loadConfig(): Promise<void> {
    this.config = await loadBenchmarksConfig();
    if (this.config) {
      logger.info('[BenchmarkChecker] Config loaded', {
        models: Object.keys(this.config.models),
        strategies: Object.keys(this.config.strategies),
      });
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): BenchmarksConfig | null {
    return this.config;
  }

  /**
   * Check all benchmarks and return violations
   */
  async checkAllBenchmarks(): Promise<BenchmarkViolation[]> {
    if (!this.config) {
      await this.loadConfig();
    }

    if (!this.config?.enabled) {
      logger.debug('[BenchmarkChecker] Benchmarks disabled');
      return [];
    }

    const violations: BenchmarkViolation[] = [];

    // 1. Check model benchmarks
    const modelViolations = await this.checkModelBenchmarks();
    violations.push(...modelViolations);

    // 2. Check system benchmarks
    const systemViolations = await this.checkSystemBenchmarks();
    violations.push(...systemViolations);

    // 3. Check strategy benchmarks
    const strategyViolations = await this.checkStrategyBenchmarks();
    violations.push(...strategyViolations);

    logger.info('[BenchmarkChecker] Check complete', {
      totalViolations: violations.length,
      critical: violations.filter(v => v.severity === 'CRITICAL').length,
      warning: violations.filter(v => v.severity === 'WARNING').length,
    });

    return violations;
  }

  /**
   * Check model benchmarks against current performance
   */
  async checkModelBenchmarks(): Promise<BenchmarkViolation[]> {
    if (!this.config) return [];

    const violations: BenchmarkViolation[] = [];

    for (const [modelName, benchmarks] of Object.entries(this.config.models)) {
      const performance = await performanceMonitor.checkModelPerformance(modelName);

      // Skip if no data available
      if (performance.currentMetrics.totalPredictions === 0) {
        logger.debug('[BenchmarkChecker] No data for model', { modelName });
        continue;
      }

      const metrics = performance.currentMetrics;

      // Check precision
      if (metrics.precision < benchmarks.min_precision) {
        violations.push(this.createViolation(
          'MODEL', 'precision', metrics.precision, benchmarks.min_precision,
          `${modelName} precision`, modelName
        ));
      }

      // Check recall
      if (metrics.recall < benchmarks.min_recall) {
        violations.push(this.createViolation(
          'MODEL', 'recall', metrics.recall, benchmarks.min_recall,
          `${modelName} recall`, modelName
        ));
      }

      // Check Sharpe ratio
      if (metrics.sharpe < benchmarks.min_sharpe) {
        violations.push(this.createViolation(
          'MODEL', 'sharpe', metrics.sharpe, benchmarks.min_sharpe,
          `${modelName} Sharpe ratio`, modelName
        ));
      }

      // Check win rate
      if (metrics.winRate < benchmarks.min_win_rate) {
        violations.push(this.createViolation(
          'MODEL', 'win_rate', metrics.winRate, benchmarks.min_win_rate,
          `${modelName} win rate`, modelName
        ));
      }
    }

    return violations;
  }

  /**
   * Check system-wide benchmarks
   */
  async checkSystemBenchmarks(): Promise<BenchmarkViolation[]> {
    if (!this.config) return [];

    const violations: BenchmarkViolation[] = [];
    const systemMetrics = await this.getSystemMetrics();
    const benchmarks = this.config.system;

    // Check daily drawdown
    if (systemMetrics.dailyDrawdown > benchmarks.max_daily_drawdown) {
      violations.push(this.createViolation(
        'SYSTEM', 'daily_drawdown', systemMetrics.dailyDrawdown,
        benchmarks.max_daily_drawdown, 'Daily drawdown'
      ));
    }

    // Check weekly drawdown
    if (systemMetrics.weeklyDrawdown > benchmarks.max_weekly_drawdown) {
      violations.push(this.createViolation(
        'SYSTEM', 'weekly_drawdown', systemMetrics.weeklyDrawdown,
        benchmarks.max_weekly_drawdown, 'Weekly drawdown'
      ));
    }

    // Check monthly drawdown
    if (systemMetrics.monthlyDrawdown > benchmarks.max_monthly_drawdown) {
      violations.push(this.createViolation(
        'SYSTEM', 'monthly_drawdown', systemMetrics.monthlyDrawdown,
        benchmarks.max_monthly_drawdown, 'Monthly drawdown'
      ));
    }

    // Check daily return (min return = max loss)
    if (systemMetrics.dailyReturn < benchmarks.min_daily_return) {
      violations.push(this.createViolation(
        'SYSTEM', 'daily_return', systemMetrics.dailyReturn,
        benchmarks.min_daily_return, 'Daily return'
      ));
    }

    // Check consecutive losses
    if (systemMetrics.consecutiveLosses > benchmarks.max_consecutive_losses) {
      violations.push(this.createViolation(
        'SYSTEM', 'consecutive_losses', systemMetrics.consecutiveLosses,
        benchmarks.max_consecutive_losses, 'Consecutive losses'
      ));
    }

    return violations;
  }

  /**
   * Check strategy-specific benchmarks
   */
  async checkStrategyBenchmarks(): Promise<BenchmarkViolation[]> {
    if (!this.config) return [];

    const violations: BenchmarkViolation[] = [];

    for (const [strategyName, benchmarks] of Object.entries(this.config.strategies)) {
      const metrics = await this.getStrategyMetrics(strategyName);

      // Skip if no trades
      if (metrics.totalTrades === 0) {
        logger.debug('[BenchmarkChecker] No trades for strategy', { strategyName });
        continue;
      }

      // Check monthly trades
      if (metrics.monthlyTrades < benchmarks.min_monthly_trades) {
        violations.push(this.createViolation(
          'STRATEGY', 'monthly_trades', metrics.monthlyTrades,
          benchmarks.min_monthly_trades, `${strategyName} monthly trades`, strategyName
        ));
      }

      // Check win rate
      if (metrics.winRate < benchmarks.min_win_rate) {
        violations.push(this.createViolation(
          'STRATEGY', 'win_rate', metrics.winRate,
          benchmarks.min_win_rate, `${strategyName} win rate`, strategyName
        ));
      }

      // Check average loss (current should be LESS than max)
      if (metrics.avgLoss > benchmarks.max_avg_loss) {
        violations.push(this.createViolation(
          'STRATEGY', 'avg_loss', metrics.avgLoss,
          benchmarks.max_avg_loss, `${strategyName} average loss`, strategyName
        ));
      }
    }

    return violations;
  }

  /**
   * Create a benchmark violation record
   */
  private createViolation(
    type: 'MODEL' | 'SYSTEM' | 'STRATEGY',
    metric: string,
    current: number,
    benchmark: number,
    label: string,
    modelOrStrategy?: string
  ): BenchmarkViolation {
    const deviation = type === 'SYSTEM' && metric.includes('drawdown')
      ? current - benchmark  // For max limits, deviation = current - limit
      : benchmark - current; // For min limits, deviation = limit - current

    const severity = this.getSeverity(current, benchmark, type, metric);

    const message = type === 'SYSTEM' && (metric.includes('drawdown') || metric === 'consecutive_losses')
      ? `${label} (${this.formatValue(current, metric)}) exceeds limit (${this.formatValue(benchmark, metric)})`
      : `${label} (${this.formatValue(current, metric)}) below benchmark (${this.formatValue(benchmark, metric)})`;

    return {
      type,
      severity,
      metric,
      current,
      benchmark,
      deviation: Math.abs(deviation),
      message,
      timestamp: new Date(),
      modelOrStrategy,
    };
  }

  /**
   * Determine violation severity
   */
  private getSeverity(
    current: number,
    benchmark: number,
    type: 'MODEL' | 'SYSTEM' | 'STRATEGY',
    metric: string
  ): ViolationSeverity {
    if (!this.config) return 'CRITICAL';

    // For drawdowns and consecutive losses, current > benchmark is always critical
    if (type === 'SYSTEM' && (metric.includes('drawdown') || metric === 'consecutive_losses')) {
      return 'CRITICAL';
    }

    // For min thresholds, check ratio
    const ratio = current / benchmark;
    if (ratio < this.config.alerts.critical_threshold) return 'CRITICAL';
    if (ratio < this.config.alerts.warning_threshold) return 'WARNING';
    return 'WARNING';
  }

  /**
   * Format metric value for display
   */
  private formatValue(value: number, metric: string): string {
    if (metric.includes('drawdown') || metric.includes('return') || metric.includes('rate') || metric === 'avg_loss') {
      return `${(value * 100).toFixed(1)}%`;
    }
    if (metric === 'consecutive_losses' || metric === 'monthly_trades') {
      return value.toString();
    }
    return value.toFixed(2);
  }

  /**
   * Get system-wide metrics from trade history
   */
  async getSystemMetrics(): Promise<SystemMetrics> {
    const now = new Date();
    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const oneMonthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    // Filter trades by time period
    const dailyTrades = this.tradeHistory.filter(t => t.timestamp >= oneDayAgo);
    const weeklyTrades = this.tradeHistory.filter(t => t.timestamp >= oneWeekAgo);
    const monthlyTrades = this.tradeHistory.filter(t => t.timestamp >= oneMonthAgo);

    return {
      dailyDrawdown: this.calculateDrawdown(dailyTrades),
      weeklyDrawdown: this.calculateDrawdown(weeklyTrades),
      monthlyDrawdown: this.calculateDrawdown(monthlyTrades),
      dailyReturn: this.calculateReturn(dailyTrades),
      weeklyReturn: this.calculateReturn(weeklyTrades),
      monthlyReturn: this.calculateReturn(monthlyTrades),
      consecutiveLosses: this.calculateConsecutiveLosses(),
      annualAlpha: 0, // Calculated separately with benchmark comparison
      annualSharpe: this.calculateSharpe(monthlyTrades),
    };
  }

  /**
   * Get strategy-specific metrics
   */
  async getStrategyMetrics(strategyName: string): Promise<StrategyMetrics> {
    const now = new Date();
    const oneMonthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const strategyTrades = this.tradeHistory.filter(
      t => t.strategy === strategyName && t.timestamp >= oneMonthAgo
    );

    const winningTrades = strategyTrades.filter(t => t.profitLoss > 0);
    const losingTrades = strategyTrades.filter(t => t.profitLoss < 0);

    return {
      monthlyTrades: strategyTrades.length,
      winRate: strategyTrades.length > 0 ? winningTrades.length / strategyTrades.length : 0,
      avgLoss: losingTrades.length > 0
        ? Math.abs(losingTrades.reduce((sum, t) => sum + t.profitLoss, 0) / losingTrades.length)
        : 0,
      avgProfit: winningTrades.length > 0
        ? winningTrades.reduce((sum, t) => sum + t.profitLoss, 0) / winningTrades.length
        : 0,
      totalTrades: strategyTrades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
    };
  }

  /**
   * Track a completed trade
   */
  trackTrade(trade: TradeRecord): void {
    this.tradeHistory.push(trade);

    // Trim history
    if (this.tradeHistory.length > this.maxHistorySize) {
      this.tradeHistory.splice(0, this.tradeHistory.length - this.maxHistorySize);
    }

    logger.debug('[BenchmarkChecker] Trade tracked', {
      strategy: trade.strategy,
      profitLoss: trade.profitLoss,
    });
  }

  /**
   * Calculate drawdown from trades
   */
  private calculateDrawdown(trades: TradeRecord[]): number {
    if (trades.length === 0) return 0;

    let peak = 0;
    let maxDrawdown = 0;
    let cumulative = 0;

    for (const trade of trades) {
      cumulative += trade.profitLoss;
      peak = Math.max(peak, cumulative);
      const drawdown = peak - cumulative;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    // Return as percentage (assuming profitLoss is already in %)
    return maxDrawdown;
  }

  /**
   * Calculate total return from trades
   */
  private calculateReturn(trades: TradeRecord[]): number {
    if (trades.length === 0) return 0;
    return trades.reduce((sum, t) => sum + t.profitLoss, 0);
  }

  /**
   * Calculate consecutive losses
   */
  private calculateConsecutiveLosses(): number {
    let maxConsecutive = 0;
    let currentConsecutive = 0;

    for (const trade of this.tradeHistory) {
      if (trade.profitLoss < 0) {
        currentConsecutive++;
        maxConsecutive = Math.max(maxConsecutive, currentConsecutive);
      } else {
        currentConsecutive = 0;
      }
    }

    return maxConsecutive;
  }

  /**
   * Calculate Sharpe ratio from trades
   */
  private calculateSharpe(trades: TradeRecord[]): number {
    if (trades.length < 2) return 0;

    const returns = trades.map(t => t.profitLoss);
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return mean > 0 ? 3 : 0;

    // Annualized (assuming daily trades)
    return (mean / stdDev) * Math.sqrt(252);
  }

  /**
   * Get trade history
   */
  getTradeHistory(): TradeRecord[] {
    return [...this.tradeHistory];
  }

  /**
   * Clear trade history
   */
  clearTradeHistory(): void {
    this.tradeHistory = [];
    logger.info('[BenchmarkChecker] Trade history cleared');
  }
}

// ============= TRADE RECORD TYPE =============

export interface TradeRecord {
  id: string;
  strategy: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  profitLoss: number;  // As percentage (e.g., 0.02 = 2%)
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

// ============= SINGLETON =============

let benchmarkCheckerInstance: BenchmarkChecker | null = null;

export function getBenchmarkChecker(): BenchmarkChecker {
  if (!benchmarkCheckerInstance) {
    benchmarkCheckerInstance = new BenchmarkChecker();
  }
  return benchmarkCheckerInstance;
}

export const benchmarkChecker = getBenchmarkChecker();

export { BenchmarkChecker };

