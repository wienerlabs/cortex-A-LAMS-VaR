/**
 * Stress Test Scenarios - Types and Loader
 * 
 * Defines interfaces for stress test scenarios and provides
 * utilities for loading scenarios from YAML configuration.
 */
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { parse as parseYaml } from 'yaml';
import { logger } from '../logger.js';
import type { CircuitBreakerState } from '../risk/types.js';

// ============= SCENARIO TYPES =============

export type StressSeverity = 'high' | 'critical' | 'emergency';

export interface ExpectedResponse {
  circuit_breaker_triggered?: boolean;
  circuit_breaker_state?: CircuitBreakerState;
  trades_halted?: boolean;
  new_positions_blocked?: boolean;
  max_acceptable_drawdown_pct?: number;
  guardian_blocks_trades?: boolean;
  guardian_blocks_high_slippage?: boolean;
  max_slippage_threshold_pct?: number;
  system_lockdown?: boolean;
  manual_intervention_required?: boolean;
  positions_evaluated_for_emergency_close?: boolean;
  position_size_reduction_pct?: number;
  high_liquidity_assets_only?: boolean;
  jito_bundles_protect?: boolean;
  slippage_detection_catches_attack?: boolean;
  trade_reverted_if_slippage_exceeded?: boolean;
  max_acceptable_slippage_pct?: number;
  gas_price_limiter_active?: boolean;
  max_gas_sol?: number;
  timeout_handling_seconds?: number;
  retry_with_backoff?: boolean;
  max_retries?: number;
  trades_halted_if_gas_exceeded?: boolean;
  all_trading_halted?: boolean;
  emergency_exit_only?: boolean;
}

export interface FlashCrashScenario {
  name: string;
  description: string;
  severity: StressSeverity;
  price_drop_pct: number;
  drop_duration_seconds: number;
  recovery_pct: number;
  recovery_duration_seconds: number;
  liquidity_drain_pct: number;
  bid_ask_spread_multiplier: number;
  slippage_multiplier: number;
  expected_response: ExpectedResponse;
}

export interface BlackSwanScenario {
  name: string;
  description: string;
  severity: StressSeverity;
  sigma_move: number;
  price_change_pct: number;
  duration_hours: number;
  cascade_effect: boolean;
  correlated_assets_drop_pct: number;
  protocol_failures: number;
  liquidity_drain_pct: number;
  oracle_staleness_seconds: number;
  expected_response: ExpectedResponse;
}

export interface LiquidityCrisisScenario {
  name: string;
  description: string;
  severity: StressSeverity;
  available_liquidity_pct: number;
  bid_ask_multiplier: number;
  slippage_multiplier: number;
  order_book_depth_pct: number;
  market_order_rejection_rate: number;
  expected_response: ExpectedResponse;
}

export interface MevAttackScenario {
  name: string;
  description: string;
  severity: StressSeverity;
  front_run_pct: number;
  back_run_pct: number;
  victim_slippage_pct: number;
  attack_frequency: number;
  expected_response: ExpectedResponse;
}

export interface NetworkCongestionScenario {
  name: string;
  description: string;
  severity: StressSeverity;
  gas_price_multiplier: number;
  confirmation_delay_seconds: number;
  tx_failure_rate_pct: number;
  rpc_timeout_rate_pct: number;
  block_production_delay_seconds: number;
  expected_response: ExpectedResponse;
}

export type StressScenario =
  | FlashCrashScenario
  | BlackSwanScenario
  | LiquidityCrisisScenario
  | MevAttackScenario
  | NetworkCongestionScenario;

export type ScenarioType = 'flash_crash' | 'black_swan' | 'liquidity_crisis' | 'mev_attack' | 'network_congestion';

export interface StressTestResult {
  scenario: string;
  scenarioType: ScenarioType;
  severity: StressSeverity;
  timestamp: Date;
  durationMs: number;
  systemResponse: {
    circuitBreakerTriggered: boolean;
    circuitBreakerState: CircuitBreakerState;
    tradesHalted: boolean;
    positionsFlattened: boolean;
    maxDrawdown: number;
    guardianBlockedTrades: number;
    gasLimitExceeded: boolean;
  };
  expectations: {
    name: string;
    expected: boolean | number | string;
    actual: boolean | number | string;
    passed: boolean;
  }[];
  failures: string[];
  passed: boolean;
}

export interface HistoricalEvent {
  id: string;
  name: string;
  date: string;
  end_date: string;
  duration_days: number;
  description: string;
  price_action: {
    asset: string;
    start_price_usd: number;
    end_price_usd: number;
    price_drop_pct: number;
    max_hourly_drop_pct: number;
  };
  market_impact: {
    btc_drop_pct: number;
    eth_drop_pct: number;
    sol_drop_pct?: number;
    total_market_cap_loss_billion: number;
    contagion_assets: string[];
  };
  expected_system_behavior: ExpectedResponse & {
    recovery_time_hours?: number;
    sol_positions_evaluated?: boolean;
    emergency_exit_triggered?: boolean;
  };
}

interface ScenariosConfig {
  scenarios: {
    flash_crash: FlashCrashScenario;
    black_swan: BlackSwanScenario;
    liquidity_crisis: LiquidityCrisisScenario;
    mev_attack: MevAttackScenario;
    network_congestion: NetworkCongestionScenario;
  };
  combined_scenarios: {
    [key: string]: {
      name: string;
      description: string;
      severity: StressSeverity;
      components: ScenarioType[];
      expected_response: ExpectedResponse;
    };
  };
}

interface HistoricalEventsConfig {
  events: HistoricalEvent[];
}

// ============= SCENARIO LOADER =============

let scenariosCache: ScenariosConfig | null = null;
let historicalEventsCache: HistoricalEventsConfig | null = null;

/**
 * Get the path to the testing directory
 */
function getTestingPath(): string {
  const currentDir = dirname(fileURLToPath(import.meta.url));
  // Navigate from agent/eliza/src/services/monitoring to agent/testing
  // Path: monitoring -> services -> src -> eliza -> agent -> testing (4 levels up)
  return join(currentDir, '..', '..', '..', '..', 'testing');
}

/**
 * Load stress scenarios from YAML configuration
 */
export function loadStressScenarios(): ScenariosConfig {
  if (scenariosCache) {
    return scenariosCache;
  }

  try {
    const testingPath = getTestingPath();
    const scenariosPath = join(testingPath, 'stress_scenarios.yaml');
    const content = readFileSync(scenariosPath, 'utf-8');
    scenariosCache = parseYaml(content) as ScenariosConfig;
    logger.info('Loaded stress scenarios', {
      scenarioCount: Object.keys(scenariosCache.scenarios).length
    });
    return scenariosCache;
  } catch (error) {
    logger.error('Failed to load stress scenarios', { error });
    throw new Error(`Failed to load stress scenarios: ${error}`);
  }
}

/**
 * Load historical stress events from JSON
 */
export function loadHistoricalEvents(): HistoricalEvent[] {
  if (historicalEventsCache) {
    return historicalEventsCache.events;
  }

  try {
    const testingPath = getTestingPath();
    const eventsPath = join(testingPath, 'historical_stress_events.json');
    const content = readFileSync(eventsPath, 'utf-8');
    historicalEventsCache = JSON.parse(content) as HistoricalEventsConfig;
    logger.info('Loaded historical stress events', {
      eventCount: historicalEventsCache.events.length
    });
    return historicalEventsCache.events;
  } catch (error) {
    logger.error('Failed to load historical events', { error });
    throw new Error(`Failed to load historical events: ${error}`);
  }
}

/**
 * Get a specific stress scenario by type
 */
export function getScenario(type: ScenarioType): StressScenario {
  const config = loadStressScenarios();
  const scenario = config.scenarios[type];
  if (!scenario) {
    throw new Error(`Unknown scenario type: ${type}`);
  }
  return scenario;
}

/**
 * Get all available scenario types
 */
export function getAvailableScenarios(): ScenarioType[] {
  const config = loadStressScenarios();
  return Object.keys(config.scenarios) as ScenarioType[];
}

/**
 * Get a historical event by ID
 */
export function getHistoricalEvent(id: string): HistoricalEvent {
  const events = loadHistoricalEvents();
  const event = events.find(e => e.id === id);
  if (!event) {
    throw new Error(`Unknown historical event: ${id}`);
  }
  return event;
}

/**
 * Clear cached scenarios (useful for testing)
 */
export function clearScenarioCache(): void {
  scenariosCache = null;
  historicalEventsCache = null;
}

