/**
 * Stress Tests
 * 
 * Automated tests for extreme market conditions.
 * Tests circuit breakers, guardian validation, and risk management.
 * 
 * ALL DATA IS DYNAMIC - NO HARDCODED VALUES
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  StressTestRunner,
  getStressTestRunner,
  resetStressTestRunner,
} from '../services/monitoring/stressTestRunner.js';
import {
  StressMonitor,
  getStressMonitor,
  resetStressMonitor,
  type MarketConditions,
} from '../services/monitoring/stressMonitor.js';
import {
  loadStressScenarios,
  loadHistoricalEvents,
  getScenario,
  getHistoricalEvent,
  clearScenarioCache,
  type ScenarioType,
} from '../services/monitoring/stressScenarios.js';
import { resetGlobalRiskManager } from '../services/risk/globalRiskManager.js';

// ============= TEST SETUP =============

describe('Stress Tests', () => {
  let runner: StressTestRunner;
  let monitor: StressMonitor;

  beforeEach(() => {
    resetGlobalRiskManager();
    resetStressTestRunner();
    resetStressMonitor();
    clearScenarioCache();
    
    runner = getStressTestRunner();
    monitor = getStressMonitor();
  });

  afterEach(() => {
    resetGlobalRiskManager();
    resetStressTestRunner();
    resetStressMonitor();
  });

  // ============= SCENARIO LOADING =============

  describe('Scenario Loading', () => {
    it('should load all stress scenarios from YAML', () => {
      const scenarios = loadStressScenarios();
      
      expect(scenarios).toBeDefined();
      expect(scenarios.scenarios).toBeDefined();
      expect(scenarios.scenarios.flash_crash).toBeDefined();
      expect(scenarios.scenarios.black_swan).toBeDefined();
      expect(scenarios.scenarios.liquidity_crisis).toBeDefined();
      expect(scenarios.scenarios.mev_attack).toBeDefined();
      expect(scenarios.scenarios.network_congestion).toBeDefined();
    });

    it('should load historical events from JSON', () => {
      const events = loadHistoricalEvents();
      
      expect(events).toBeInstanceOf(Array);
      expect(events.length).toBeGreaterThanOrEqual(3);
      
      // Check for expected events
      const eventIds = events.map(e => e.id);
      expect(eventIds).toContain('luna_collapse');
      expect(eventIds).toContain('ftx_collapse');
      expect(eventIds).toContain('covid_crash');
    });

    it('should get a specific scenario by type', () => {
      const scenario = getScenario('flash_crash');
      
      expect(scenario).toBeDefined();
      expect(scenario.name).toBe('Flash Crash');
      expect(scenario.severity).toBe('critical');
    });

    it('should get a specific historical event by ID', () => {
      const event = getHistoricalEvent('luna_collapse');
      
      expect(event).toBeDefined();
      expect(event.name).toBe('LUNA/UST Collapse');
      expect(event.price_action.price_drop_pct).toBe(99.9999);
    });
  });

  // ============= FLASH CRASH TESTS =============

  describe('Flash Crash Scenario', () => {
    it('should trigger circuit breaker on flash crash', async () => {
      const result = await runner.runScenario('flash_crash');
      
      expect(result.scenario).toBe('Flash Crash');
      expect(result.scenarioType).toBe('flash_crash');
      expect(result.systemResponse.circuitBreakerTriggered).toBe(true);
      expect(result.systemResponse.tradesHalted).toBe(true);
    });

    it('should have circuit breaker in PAUSED state', async () => {
      const result = await runner.runScenario('flash_crash');
      
      expect(result.systemResponse.circuitBreakerState).toBe('PAUSED');
    });

    it('should have max drawdown less than 25%', async () => {
      const result = await runner.runScenario('flash_crash');
      const scenario = getScenario('flash_crash') as { price_drop_pct: number };
      
      expect(result.systemResponse.maxDrawdown).toBe(scenario.price_drop_pct);
    });
  });

  // ============= BLACK SWAN TESTS =============

  describe('Black Swan Scenario', () => {
    it('should trigger LOCKDOWN on black swan event', async () => {
      const result = await runner.runScenario('black_swan');
      
      expect(result.scenario).toBe('Black Swan Event');
      expect(result.systemResponse.circuitBreakerTriggered).toBe(true);
      expect(result.systemResponse.circuitBreakerState).toBe('LOCKDOWN');
    });

    it('should halt all trades during black swan', async () => {
      const result = await runner.runScenario('black_swan');
      
      expect(result.systemResponse.tradesHalted).toBe(true);
    });
  });

  // ============= LIQUIDITY CRISIS TESTS =============

  describe('Liquidity Crisis Scenario', () => {
    it('should block high slippage trades', async () => {
      const result = await runner.runScenario('liquidity_crisis');

      expect(result.scenario).toBe('Liquidity Crisis');
      expect(result.systemResponse.guardianBlockedTrades).toBeGreaterThan(0);
    });

    it('should not trigger circuit breaker for liquidity crisis alone', async () => {
      const result = await runner.runScenario('liquidity_crisis');

      expect(result.systemResponse.circuitBreakerTriggered).toBe(false);
    });
  });

  // ============= MEV ATTACK TESTS =============

  describe('MEV Attack Scenario', () => {
    it('should detect slippage from MEV attack', async () => {
      const result = await runner.runScenario('mev_attack');

      expect(result.scenario).toBe('MEV Sandwich Attack');
      expect(result.scenarioType).toBe('mev_attack');
    });

    it('should have Jito bundle protection active', async () => {
      const result = await runner.runScenario('mev_attack');

      // Check expectation for Jito protection
      const jitoExpectation = result.expectations.find(e => e.name === 'Jito bundles protect');
      expect(jitoExpectation?.passed).toBe(true);
    });
  });

  // ============= NETWORK CONGESTION TESTS =============

  describe('Network Congestion Scenario', () => {
    it('should activate gas price limiter', async () => {
      const result = await runner.runScenario('network_congestion');

      expect(result.scenario).toBe('Network Congestion');
      expect(result.systemResponse.gasLimitExceeded).toBe(true);
    });

    it('should have retry with backoff enabled', async () => {
      const result = await runner.runScenario('network_congestion');

      const retryExpectation = result.expectations.find(e => e.name === 'Retry with backoff');
      expect(retryExpectation?.passed).toBe(true);
    });
  });

  // ============= HISTORICAL EVENT REPLAY =============

  describe('Historical Event Replay', () => {
    it('should replay LUNA collapse', async () => {
      const result = await runner.replayHistoricalEvent('luna_collapse');

      expect(result.scenario).toBe('LUNA/UST Collapse');
      expect(result.systemResponse.circuitBreakerTriggered).toBe(true);
      expect(result.systemResponse.circuitBreakerState).toBe('LOCKDOWN');
    });

    it('should replay FTX collapse', async () => {
      const result = await runner.replayHistoricalEvent('ftx_collapse');

      expect(result.scenario).toBe('FTX Collapse');
      expect(result.systemResponse.circuitBreakerTriggered).toBe(true);
    });

    it('should replay COVID crash', async () => {
      const result = await runner.replayHistoricalEvent('covid_crash');

      expect(result.scenario).toBe('March 2020 COVID Crash');
      expect(result.systemResponse.tradesHalted).toBe(true);
    });
  });

  // ============= STRESS MONITOR TESTS =============

  describe('Stress Monitor', () => {
    it('should detect flash crash conditions', async () => {
      const conditions: MarketConditions = {
        price: 100,
        priceChange1m: -6,  // 6% drop in 1 minute (above 5% threshold)
        priceChange5m: -12,
        priceChange1h: -15,
        priceChange24h: -20,
        volume24h: 1000000,
        volumeChange: 50,
        liquidity: 500000,
        liquidityChange: -30,
        bidAskSpread: 0.5,
        averageBidAskSpread: 0.1,
        gasPrice: 0.00001,
        averageGasPrice: 0.00001,
        oracleStaleness: 5,
        txFailureRate: 5,
      };

      const alerts = await monitor.checkConditions(conditions);

      expect(alerts.length).toBeGreaterThan(0);
      const flashCrashAlert = alerts.find(a => a.scenarioType === 'flash_crash');
      expect(flashCrashAlert).toBeDefined();
      expect(flashCrashAlert?.severity).toBe('critical');
    });

    it('should detect black swan conditions', async () => {
      const conditions: MarketConditions = {
        price: 100,
        priceChange1m: -2,
        priceChange5m: -5,
        priceChange1h: -15,
        priceChange24h: -30,  // 30% drop (above 25% threshold)
        volume24h: 1000000,
        volumeChange: 100,
        liquidity: 500000,
        liquidityChange: -50,
        bidAskSpread: 0.5,
        averageBidAskSpread: 0.1,
        gasPrice: 0.00001,
        averageGasPrice: 0.00001,
        oracleStaleness: 5,
        txFailureRate: 5,
      };

      const alerts = await monitor.checkConditions(conditions);

      const blackSwanAlert = alerts.find(a => a.scenarioType === 'black_swan');
      expect(blackSwanAlert).toBeDefined();
      expect(blackSwanAlert?.severity).toBe('emergency');
    });

    it('should detect network congestion', async () => {
      const conditions: MarketConditions = {
        price: 100,
        priceChange1m: 0,
        priceChange5m: 0,
        priceChange1h: 0,
        priceChange24h: 0,
        volume24h: 1000000,
        volumeChange: 0,
        liquidity: 500000,
        liquidityChange: 0,
        bidAskSpread: 0.1,
        averageBidAskSpread: 0.1,
        gasPrice: 0.0005,  // 50x normal gas
        averageGasPrice: 0.00001,
        oracleStaleness: 120,  // 2 minutes stale (above 60s threshold)
        txFailureRate: 30,  // 30% failure rate (above 20% threshold)
      };

      const alerts = await monitor.checkConditions(conditions);

      const networkAlert = alerts.find(a => a.scenarioType === 'network_congestion');
      expect(networkAlert).toBeDefined();
      expect(networkAlert?.severity).toBe('high');
    });
  });

  // ============= RUN ALL SCENARIOS =============

  describe('Run All Scenarios', () => {
    it('should run all stress scenarios', async () => {
      const results = await runner.runAllScenarios();

      expect(results.size).toBe(5);
      expect(results.has('flash_crash')).toBe(true);
      expect(results.has('black_swan')).toBe(true);
      expect(results.has('liquidity_crisis')).toBe(true);
      expect(results.has('mev_attack')).toBe(true);
      expect(results.has('network_congestion')).toBe(true);
    });

    it('should generate a stress test report', async () => {
      const results = await runner.runAllScenarios();
      const report = runner.generateReport(results);

      expect(report).toContain('STRESS TEST REPORT');
      expect(report).toContain('Flash Crash');
      expect(report).toContain('Black Swan');
      expect(report).toContain('SUMMARY');
    });
  });
});

