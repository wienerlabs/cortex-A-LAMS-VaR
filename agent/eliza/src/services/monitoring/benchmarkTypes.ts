/**
 * Benchmark Types
 *
 * Type definitions for the performance benchmarks monitoring system.
 * These types define the structure of benchmark configuration,
 * violations, and status tracking.
 */

// ============= VIOLATION TYPES =============

export type ViolationType = 'MODEL' | 'SYSTEM' | 'STRATEGY';
export type ViolationSeverity = 'WARNING' | 'CRITICAL';
export type MetricStatus = 'PASS' | 'WARNING' | 'FAIL';

export interface BenchmarkViolation {
  type: ViolationType;
  severity: ViolationSeverity;
  metric: string;
  current: number;
  benchmark: number;
  deviation: number;  // How far below benchmark (positive = worse)
  message: string;
  timestamp: Date;
  modelOrStrategy?: string;
}

// ============= MODEL BENCHMARKS =============

export interface ModelBenchmarks {
  min_precision: number;
  min_recall: number;
  min_sharpe: number;
  min_win_rate: number;
  min_roc_auc?: number;
}

export interface ModelBenchmarksConfig {
  [modelName: string]: ModelBenchmarks;
}

// ============= SYSTEM BENCHMARKS =============

export interface SystemBenchmarks {
  max_daily_drawdown: number;
  max_weekly_drawdown: number;
  max_monthly_drawdown: number;
  min_daily_return: number;
  min_weekly_return: number;
  min_monthly_return: number;
  target_annual_alpha: number;
  target_annual_sharpe: number;
  max_consecutive_losses: number;
}

// ============= STRATEGY BENCHMARKS =============

export interface StrategyBenchmarks {
  min_monthly_trades: number;
  min_win_rate: number;
  max_avg_loss: number;
  min_avg_profit?: number;
}

export interface StrategyBenchmarksConfig {
  [strategyName: string]: StrategyBenchmarks;
}

// ============= ALERT CONFIG =============

export type NotificationChannel = 'log' | 'email' | 'slack';

export interface AlertConfig {
  warning_threshold: number;
  critical_threshold: number;
  pause_trading_on_critical: boolean;
  resume_requires_manual_approval: boolean;
  notification_channels: NotificationChannel[];
  alert_cooldown_hours: number;
}

// ============= FULL CONFIG =============

export interface BenchmarksConfig {
  enabled: boolean;
  check_interval_hours: number;
  models: ModelBenchmarksConfig;
  system: SystemBenchmarks;
  strategies: StrategyBenchmarksConfig;
  alerts: AlertConfig;
}

// ============= METRIC STATUS =============

export interface MetricStatusInfo {
  current: number;
  benchmark: number;
  status: MetricStatus;
  deviation?: number;
}

// ============= BENCHMARK STATUS =============

export interface ModelBenchmarkStatus {
  precision: MetricStatusInfo;
  recall: MetricStatusInfo;
  sharpe: MetricStatusInfo;
  winRate: MetricStatusInfo;
  rocAuc?: MetricStatusInfo;
}

export interface SystemBenchmarkStatus {
  dailyDrawdown: MetricStatusInfo;
  weeklyDrawdown: MetricStatusInfo;
  monthlyDrawdown: MetricStatusInfo;
  dailyReturn: MetricStatusInfo;
  weeklyReturn: MetricStatusInfo;
  monthlyReturn: MetricStatusInfo;
  consecutiveLosses: MetricStatusInfo;
}

export interface StrategyBenchmarkStatus {
  monthlyTrades: MetricStatusInfo;
  winRate: MetricStatusInfo;
  avgLoss: MetricStatusInfo;
  avgProfit?: MetricStatusInfo;
}

export interface BenchmarkStatus {
  models: { [modelName: string]: ModelBenchmarkStatus };
  system: SystemBenchmarkStatus;
  strategies: { [strategyName: string]: StrategyBenchmarkStatus };
  tradingPaused: boolean;
  pauseReason?: string;
  lastCheck: Date;
  violations: BenchmarkViolation[];
}

// ============= SYSTEM METRICS =============

export interface SystemMetrics {
  dailyDrawdown: number;
  weeklyDrawdown: number;
  monthlyDrawdown: number;
  dailyReturn: number;
  weeklyReturn: number;
  monthlyReturn: number;
  consecutiveLosses: number;
  annualAlpha: number;
  annualSharpe: number;
}

// ============= STRATEGY METRICS =============

export interface StrategyMetrics {
  monthlyTrades: number;
  winRate: number;
  avgLoss: number;
  avgProfit: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
}

