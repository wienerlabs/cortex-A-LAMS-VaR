/**
 * Stress Testing & Monitoring Module
 * 
 * Exports all stress testing components for testing system behavior
 * under extreme market conditions.
 * 
 * Features:
 * 1. Stress Scenarios (Flash crash, Black swan, Liquidity crisis, MEV, Network)
 * 2. Stress Test Runner (Simulates scenarios and validates responses)
 * 3. Stress Monitor (Real-time detection of stress conditions)
 * 4. Historical Event Replay (LUNA, FTX, COVID crash)
 */

// Stress scenario types and loader
export {
  type StressSeverity,
  type ExpectedResponse,
  type FlashCrashScenario,
  type BlackSwanScenario,
  type LiquidityCrisisScenario,
  type MevAttackScenario,
  type NetworkCongestionScenario,
  type StressScenario,
  type ScenarioType,
  type StressTestResult,
  type HistoricalEvent,
  loadStressScenarios,
  loadHistoricalEvents,
  getScenario,
  getAvailableScenarios,
  getHistoricalEvent,
  clearScenarioCache,
} from './stressScenarios.js';

// Stress test runner
export {
  StressTestRunner,
  getStressTestRunner,
  resetStressTestRunner,
} from './stressTestRunner.js';

// Real-time stress monitor
export {
  type MarketConditions,
  type StressAlert,
  type StressMonitorConfig,
  DEFAULT_STRESS_MONITOR_CONFIG,
  StressMonitor,
  getStressMonitor,
  resetStressMonitor,
} from './stressMonitor.js';

// Benchmark types
export {
  type BenchmarksConfig,
  type ModelBenchmarks,
  type SystemBenchmarks,
  type StrategyBenchmarks,
  type AlertConfig,
  type BenchmarkViolation,
  type ViolationSeverity,
  type SystemMetrics,
  type StrategyMetrics,
  type MetricStatus,
  type MetricStatusInfo,
  type BenchmarkStatus,
} from './benchmarkTypes.js';

// Benchmark config loader
export {
  loadBenchmarksConfig,
  reloadBenchmarksConfig,
  getCachedConfig,
} from './configLoader.js';

// Benchmark checker
export {
  benchmarkChecker,
  getBenchmarkChecker,
  BenchmarkChecker,
  type TradeRecord,
} from './benchmarkChecker.js';

// Benchmark monitor
export {
  benchmarkMonitor,
  getBenchmarkMonitor,
  BenchmarkMonitor,
} from './benchmarkMonitor.js';
