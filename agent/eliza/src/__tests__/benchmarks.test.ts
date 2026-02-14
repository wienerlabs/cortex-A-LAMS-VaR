/**
 * Performance Benchmarks Tests
 *
 * Tests for the benchmark monitoring system:
 * 1. Benchmark checker (model, system, strategy checks)
 * 2. Benchmark monitor (pause/resume, notifications)
 * 3. Config loading
 * 4. Risk manager integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  BenchmarkChecker,
  BenchmarkMonitor,
  loadBenchmarksConfig,
  reloadBenchmarksConfig,
  type TradeRecord,
  type BenchmarkViolation,
} from '../services/monitoring/index.js';
import { performanceMonitor } from '../services/ml/performanceMonitor.js';

// ============= TEST SETUP =============

describe('Performance Benchmarks', () => {
  let benchmarkChecker: BenchmarkChecker;
  let benchmarkMonitor: BenchmarkMonitor;

  beforeEach(() => {
    // Create fresh instances
    benchmarkChecker = new BenchmarkChecker();
    benchmarkMonitor = new BenchmarkMonitor();
    reloadBenchmarksConfig();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    benchmarkChecker.clearTradeHistory();
    benchmarkMonitor.clearViolationHistory();
    benchmarkMonitor.stop();
  });

  // ============= CONFIG LOADING TESTS =============

  describe('Config Loading', () => {
    it('loads benchmarks config with defaults', async () => {
      const config = await loadBenchmarksConfig();

      expect(config).toBeDefined();
      expect(config.enabled).toBeDefined();
      expect(config.check_interval_hours).toBeGreaterThan(0);
      expect(config.models).toBeDefined();
      expect(config.system).toBeDefined();
      expect(config.strategies).toBeDefined();
      expect(config.alerts).toBeDefined();
    });

    it('has model benchmarks for core models', async () => {
      const config = await loadBenchmarksConfig();

      expect(config.models.perps).toBeDefined();
      expect(config.models.perps.min_precision).toBeGreaterThan(0);
      expect(config.models.perps.min_sharpe).toBeGreaterThan(0);
    });

    it('has system benchmarks with reasonable defaults', async () => {
      const config = await loadBenchmarksConfig();

      expect(config.system.max_daily_drawdown).toBeLessThan(0.5);
      expect(config.system.max_consecutive_losses).toBeGreaterThan(0);
    });
  });

  // ============= BENCHMARK CHECKER TESTS =============

  describe('Benchmark Checker', () => {
    it('tracks trades correctly', () => {
      const trade: TradeRecord = {
        id: 'trade-001',
        strategy: 'perps',
        symbol: 'SOL-PERP',
        side: 'BUY',
        profitLoss: 0.02,  // 2% profit
        timestamp: new Date(),
      };

      benchmarkChecker.trackTrade(trade);
      const history = benchmarkChecker.getTradeHistory();

      expect(history.length).toBe(1);
      expect(history[0].profitLoss).toBe(0.02);
    });

    it('calculates system metrics from trades', async () => {
      // Add some trades
      for (let i = 0; i < 10; i++) {
        benchmarkChecker.trackTrade({
          id: `trade-${i}`,
          strategy: 'spot_trading',
          symbol: 'SOL/USDC',
          side: i % 2 === 0 ? 'BUY' : 'SELL',
          profitLoss: i % 3 === 0 ? -0.01 : 0.015,  // Some wins, some losses
          timestamp: new Date(),
        });
      }

      const metrics = await benchmarkChecker.getSystemMetrics();

      expect(metrics).toBeDefined();
      expect(metrics.dailyDrawdown).toBeGreaterThanOrEqual(0);
      expect(metrics.consecutiveLosses).toBeGreaterThanOrEqual(0);
    });

    it('calculates strategy metrics correctly', async () => {
      // Add trades for a specific strategy
      for (let i = 0; i < 5; i++) {
        benchmarkChecker.trackTrade({
          id: `arb-${i}`,
          strategy: 'arbitrage',
          symbol: 'SOL/USDC',
          side: 'BUY',
          profitLoss: i < 4 ? 0.01 : -0.005,  // 4 wins, 1 loss
          timestamp: new Date(),
        });
      }

      const metrics = await benchmarkChecker.getStrategyMetrics('arbitrage');

      expect(metrics.totalTrades).toBe(5);
      expect(metrics.winRate).toBe(0.8);  // 4/5
      expect(metrics.winningTrades).toBe(4);
      expect(metrics.losingTrades).toBe(1);
    });

    it('detects model benchmark violations', async () => {
      await benchmarkChecker.loadConfig();

      // Mock performanceMonitor to return degraded metrics
      vi.spyOn(performanceMonitor, 'checkModelPerformance').mockResolvedValue({
        modelName: 'perps',
        modelVersion: '1.0.0',
        currentMetrics: {
          precision: 0.50,  // Below 0.70 threshold
          recall: 0.40,     // Below 0.50 threshold
          f1Score: 0.45,
          winRate: 0.45,    // Below 0.60 threshold
          sharpe: 1.0,      // Below 1.5 threshold
          totalPredictions: 100,
          truePositives: 50,
          trueNegatives: 50,
          falsePositives: 50,
          falseNegatives: 50,
          avgConfidence: 0.6,
          windowStart: new Date(Date.now() - 86400000),
          windowEnd: new Date(),
        },
        baselineMetrics: { precision: 0.75, recall: 0.65, f1Score: 0.70, rocAuc: 0.80, sharpe: 1.8 },
        degradation: { precision: 0.25, recall: 0.25, sharpe: 0.8, winRate: 0.15 },
        needsRetraining: true,
        reason: 'Performance degraded',
        lastChecked: new Date(),
      });

      const violations = await benchmarkChecker.checkModelBenchmarks();

      expect(violations.length).toBeGreaterThan(0);
      expect(violations.some(v => v.metric === 'precision')).toBe(true);
    });

    it('returns no violations when all metrics pass', async () => {
      await benchmarkChecker.loadConfig();

      // Mock performanceMonitor to return good metrics
      vi.spyOn(performanceMonitor, 'checkModelPerformance').mockResolvedValue({
        modelName: 'perps',
        modelVersion: '1.0.0',
        currentMetrics: {
          precision: 0.85,  // Above 0.70 threshold
          recall: 0.70,     // Above 0.50 threshold
          f1Score: 0.77,
          winRate: 0.75,    // Above 0.60 threshold
          sharpe: 2.0,      // Above 1.5 threshold
          totalPredictions: 100,
          truePositives: 75,
          trueNegatives: 10,
          falsePositives: 10,
          falseNegatives: 5,
          avgConfidence: 0.8,
          windowStart: new Date(Date.now() - 86400000),
          windowEnd: new Date(),
        },
        baselineMetrics: { precision: 0.75, recall: 0.65, f1Score: 0.70, rocAuc: 0.80, sharpe: 1.8 },
        degradation: { precision: -0.10, recall: -0.05, sharpe: -0.2, winRate: 0 },
        needsRetraining: false,
        reason: 'Performance OK',
        lastChecked: new Date(),
      });

      const violations = await benchmarkChecker.checkModelBenchmarks();

      expect(violations.filter(v => v.modelOrStrategy === 'perps').length).toBe(0);
    });
  });

  // ============= BENCHMARK MONITOR TESTS =============

  describe('Benchmark Monitor', () => {
    it('starts and stops correctly', async () => {
      // Start should not throw
      await benchmarkMonitor.start();
      expect(benchmarkMonitor.isTradingPaused()).toBe(false);

      benchmarkMonitor.stop();
    });

    it('resumes trading with approval', async () => {
      // Manually set paused state
      (benchmarkMonitor as any).tradingPaused = true;
      (benchmarkMonitor as any).pauseReason = 'Test pause';
      (benchmarkMonitor as any).config = await loadBenchmarksConfig();

      expect(benchmarkMonitor.isTradingPaused()).toBe(true);

      const success = await benchmarkMonitor.resumeTrading('test-user');

      expect(success).toBe(true);
      expect(benchmarkMonitor.isTradingPaused()).toBe(false);
    });

    it('tracks trades through monitor', () => {
      const trade: TradeRecord = {
        id: 'trade-via-monitor',
        strategy: 'lp_rebalancing',
        symbol: 'SOL/USDC',
        side: 'BUY',
        profitLoss: 0.03,
        timestamp: new Date(),
      };

      benchmarkMonitor.trackTrade(trade);
      // Trade should be tracked in the checker
    });

    it('returns status with all sections', async () => {
      // Initialize with some data
      (benchmarkMonitor as any).config = await loadBenchmarksConfig();
      (benchmarkMonitor as any).lastCheck = new Date();

      const status = await benchmarkMonitor.getStatus();

      expect(status.models).toBeDefined();
      expect(status.system).toBeDefined();
      expect(status.strategies).toBeDefined();
      expect(status.tradingPaused).toBeDefined();
      expect(status.lastCheck).toBeInstanceOf(Date);
    });

    it('stores violations in history', async () => {
      (benchmarkMonitor as any).config = await loadBenchmarksConfig();

      const mockViolation: BenchmarkViolation = {
        type: 'MODEL',
        severity: 'WARNING',
        metric: 'precision',
        current: 0.65,
        benchmark: 0.70,
        deviation: 0.05,
        message: 'Test violation',
        timestamp: new Date(),
        modelOrStrategy: 'perps',
      };

      (benchmarkMonitor as any).violationHistory.push(mockViolation);

      const history = benchmarkMonitor.getViolationHistory();

      expect(history.length).toBe(1);
      expect(history[0].message).toBe('Test violation');
    });

    it('clears violation history', () => {
      (benchmarkMonitor as any).violationHistory = [
        { message: 'violation1' },
        { message: 'violation2' },
      ];

      benchmarkMonitor.clearViolationHistory();

      expect(benchmarkMonitor.getViolationHistory().length).toBe(0);
    });
  });

  // ============= INTEGRATION TESTS =============

  describe('Integration', () => {
    it('complete benchmark check flow', async () => {
      await benchmarkChecker.loadConfig();

      // Mock performanceMonitor with no data
      vi.spyOn(performanceMonitor, 'checkModelPerformance').mockResolvedValue({
        modelName: 'perps',
        modelVersion: '1.0.0',
        currentMetrics: {
          precision: 0,
          recall: 0,
          f1Score: 0,
          winRate: 0,
          sharpe: 0,
          totalPredictions: 0,  // No predictions = skip
          truePositives: 0,
          trueNegatives: 0,
          falsePositives: 0,
          falseNegatives: 0,
          avgConfidence: 0,
          windowStart: new Date(Date.now() - 86400000),
          windowEnd: new Date(),
        },
        baselineMetrics: { precision: 0, recall: 0, f1Score: 0, rocAuc: 0 },
        degradation: { precision: 0, recall: 0, sharpe: 0, winRate: 0 },
        needsRetraining: false,
        reason: 'No data',
        lastChecked: new Date(),
      });

      // Should run without errors
      const violations = await benchmarkChecker.checkAllBenchmarks();

      expect(Array.isArray(violations)).toBe(true);
    });

    it('consecutive losses detection', async () => {
      // Add a streak of losing trades
      for (let i = 0; i < 7; i++) {
        benchmarkChecker.trackTrade({
          id: `loss-${i}`,
          strategy: 'perps',
          symbol: 'BTC-PERP',
          side: 'SELL',
          profitLoss: -0.01,  // All losses
          timestamp: new Date(),
        });
      }

      const metrics = await benchmarkChecker.getSystemMetrics();

      expect(metrics.consecutiveLosses).toBe(7);
    });

    it('drawdown calculation', async () => {
      // Add trades that create a drawdown
      benchmarkChecker.trackTrade({
        id: 'win-1',
        strategy: 'spot_trading',
        symbol: 'SOL/USDC',
        side: 'BUY',
        profitLoss: 0.05,  // 5% win
        timestamp: new Date(),
      });

      benchmarkChecker.trackTrade({
        id: 'loss-1',
        strategy: 'spot_trading',
        symbol: 'SOL/USDC',
        side: 'SELL',
        profitLoss: -0.08,  // 8% loss (drawdown from peak)
        timestamp: new Date(),
      });

      const metrics = await benchmarkChecker.getSystemMetrics();

      // Drawdown should be 8% (the loss from peak)
      expect(metrics.dailyDrawdown).toBeGreaterThan(0);
    });
  });
});
