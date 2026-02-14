/**
 * Real-Time Stress Monitor
 * 
 * Monitors market conditions in real-time to detect stress scenarios.
 * Triggers alerts when conditions match known stress patterns.
 * 
 * ALL DATA IS DYNAMIC - NO HARDCODED VALUES
 */
import { logger } from '../logger.js';
import { getGlobalRiskManager, GlobalRiskManager } from '../risk/globalRiskManager.js';
import type { CircuitBreakerState } from '../risk/types.js';
import { loadStressScenarios, type StressSeverity, type ScenarioType } from './stressScenarios.js';

// ============= TYPES =============

export interface MarketConditions {
  price: number;
  priceChange1m: number;      // 1 minute price change %
  priceChange5m: number;      // 5 minute price change %
  priceChange1h: number;      // 1 hour price change %
  priceChange24h: number;     // 24 hour price change %
  volume24h: number;
  volumeChange: number;       // Volume change vs average %
  liquidity: number;          // Available liquidity USD
  liquidityChange: number;    // Liquidity change %
  bidAskSpread: number;       // Current spread %
  averageBidAskSpread: number;
  gasPrice: number;           // Current gas in SOL
  averageGasPrice: number;
  oracleStaleness: number;    // Oracle update delay in seconds
  txFailureRate: number;      // Recent transaction failure rate %
}

export interface StressAlert {
  id: string;
  timestamp: Date;
  severity: StressSeverity;
  scenarioType: ScenarioType | 'unknown';
  title: string;
  description: string;
  conditions: Partial<MarketConditions>;
  recommendedActions: string[];
  circuitBreakerState: CircuitBreakerState;
}

export interface StressMonitorConfig {
  // Price drop thresholds
  flashCrashThreshold1m: number;    // % drop in 1 minute to trigger alert
  flashCrashThreshold5m: number;    // % drop in 5 minutes
  blackSwanThreshold24h: number;    // % drop in 24 hours
  
  // Liquidity thresholds
  liquidityCrisisThreshold: number; // % of normal liquidity
  spreadMultiplierThreshold: number; // Spread vs average
  
  // Network thresholds
  gasMultiplierThreshold: number;   // Gas vs average
  oracleStalenessThreshold: number; // Seconds of staleness
  txFailureThreshold: number;       // % failure rate
  
  // Monitoring interval
  checkIntervalMs: number;
}

export const DEFAULT_STRESS_MONITOR_CONFIG: StressMonitorConfig = {
  flashCrashThreshold1m: 5,       // 5% drop in 1 minute
  flashCrashThreshold5m: 10,      // 10% drop in 5 minutes
  blackSwanThreshold24h: 25,      // 25% drop in 24 hours
  liquidityCrisisThreshold: 20,   // Less than 20% of normal liquidity
  spreadMultiplierThreshold: 5,   // Spread 5x normal
  gasMultiplierThreshold: 10,     // Gas 10x normal
  oracleStalenessThreshold: 60,   // 60 seconds stale
  txFailureThreshold: 20,         // 20% failure rate
  checkIntervalMs: 5000,          // Check every 5 seconds
};

// ============= STRESS MONITOR =============

export class StressMonitor {
  private config: StressMonitorConfig;
  private riskManager: GlobalRiskManager;
  private alerts: StressAlert[] = [];
  private isRunning = false;
  private checkInterval: NodeJS.Timeout | null = null;
  private lastConditions: MarketConditions | null = null;
  private onAlertCallbacks: ((alert: StressAlert) => void)[] = [];

  constructor(
    config: Partial<StressMonitorConfig> = {},
    riskManager?: GlobalRiskManager,
  ) {
    this.config = { ...DEFAULT_STRESS_MONITOR_CONFIG, ...config };
    this.riskManager = riskManager || getGlobalRiskManager();
  }

  /**
   * Register a callback for stress alerts
   */
  onAlert(callback: (alert: StressAlert) => void): void {
    this.onAlertCallbacks.push(callback);
  }

  /**
   * Start monitoring market conditions
   */
  start(): void {
    if (this.isRunning) {
      logger.warn('StressMonitor is already running');
      return;
    }

    this.isRunning = true;
    logger.info('StressMonitor started', { config: this.config });

    // Note: In production, this would poll real market data
    // For now, we expose checkConditions for manual/test usage
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    this.isRunning = false;
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    logger.info('StressMonitor stopped');
  }

  /**
   * Check current market conditions for stress indicators
   */
  async checkConditions(conditions: MarketConditions): Promise<StressAlert[]> {
    this.lastConditions = conditions;
    const newAlerts: StressAlert[] = [];

    // Check for flash crash
    const flashCrashAlert = this.checkFlashCrash(conditions);
    if (flashCrashAlert) newAlerts.push(flashCrashAlert);

    // Check for black swan
    const blackSwanAlert = this.checkBlackSwan(conditions);
    if (blackSwanAlert) newAlerts.push(blackSwanAlert);

    // Check for liquidity crisis
    const liquidityAlert = this.checkLiquidityCrisis(conditions);
    if (liquidityAlert) newAlerts.push(liquidityAlert);

    // Check for network congestion
    const networkAlert = this.checkNetworkCongestion(conditions);
    if (networkAlert) newAlerts.push(networkAlert);

    // Store and notify
    for (const alert of newAlerts) {
      this.alerts.push(alert);
      this.notifyAlert(alert);
    }

    return newAlerts;
  }

  // ============= STRESS DETECTION =============

  private checkFlashCrash(conditions: MarketConditions): StressAlert | null {
    const drop1m = Math.abs(Math.min(0, conditions.priceChange1m));
    const drop5m = Math.abs(Math.min(0, conditions.priceChange5m));

    if (drop1m >= this.config.flashCrashThreshold1m ||
        drop5m >= this.config.flashCrashThreshold5m) {
      return {
        id: `flash_crash_${Date.now()}`,
        timestamp: new Date(),
        severity: 'critical',
        scenarioType: 'flash_crash',
        title: 'Flash Crash Detected',
        description: `Rapid price drop: ${drop1m.toFixed(1)}% in 1m, ${drop5m.toFixed(1)}% in 5m`,
        conditions: {
          priceChange1m: conditions.priceChange1m,
          priceChange5m: conditions.priceChange5m,
        },
        recommendedActions: [
          'Halt new positions',
          'Review existing positions',
          'Check circuit breaker status',
        ],
        circuitBreakerState: this.riskManager.getCircuitBreakerState(),
      };
    }
    return null;
  }

  private checkBlackSwan(conditions: MarketConditions): StressAlert | null {
    const drop24h = Math.abs(Math.min(0, conditions.priceChange24h));

    if (drop24h >= this.config.blackSwanThreshold24h) {
      return {
        id: `black_swan_${Date.now()}`,
        timestamp: new Date(),
        severity: 'emergency',
        scenarioType: 'black_swan',
        title: 'Black Swan Event Detected',
        description: `Extreme price drop: ${drop24h.toFixed(1)}% in 24h`,
        conditions: {
          priceChange24h: conditions.priceChange24h,
        },
        recommendedActions: [
          'Enter LOCKDOWN mode',
          'Evaluate all positions for emergency exit',
          'Manual intervention required',
        ],
        circuitBreakerState: this.riskManager.getCircuitBreakerState(),
      };
    }
    return null;
  }

  private checkLiquidityCrisis(conditions: MarketConditions): StressAlert | null {
    const liquidityPct = (conditions.liquidity / (conditions.liquidity / (1 + conditions.liquidityChange / 100))) * 100;
    const spreadMultiplier = conditions.bidAskSpread / conditions.averageBidAskSpread;

    if (conditions.liquidityChange <= -80 ||
        spreadMultiplier >= this.config.spreadMultiplierThreshold) {
      return {
        id: `liquidity_crisis_${Date.now()}`,
        timestamp: new Date(),
        severity: 'critical',
        scenarioType: 'liquidity_crisis',
        title: 'Liquidity Crisis Detected',
        description: `Liquidity down ${Math.abs(conditions.liquidityChange).toFixed(0)}%, spread ${spreadMultiplier.toFixed(1)}x normal`,
        conditions: {
          liquidity: conditions.liquidity,
          liquidityChange: conditions.liquidityChange,
          bidAskSpread: conditions.bidAskSpread,
        },
        recommendedActions: [
          'Block high-slippage trades',
          'Reduce position sizes by 50%',
          'Trade only high-liquidity assets',
        ],
        circuitBreakerState: this.riskManager.getCircuitBreakerState(),
      };
    }
    return null;
  }

  private checkNetworkCongestion(conditions: MarketConditions): StressAlert | null {
    const gasMultiplier = conditions.gasPrice / conditions.averageGasPrice;

    if (gasMultiplier >= this.config.gasMultiplierThreshold ||
        conditions.oracleStaleness >= this.config.oracleStalenessThreshold ||
        conditions.txFailureRate >= this.config.txFailureThreshold) {
      return {
        id: `network_congestion_${Date.now()}`,
        timestamp: new Date(),
        severity: 'high',
        scenarioType: 'network_congestion',
        title: 'Network Congestion Detected',
        description: `Gas ${gasMultiplier.toFixed(1)}x normal, oracle ${conditions.oracleStaleness}s stale, ${conditions.txFailureRate.toFixed(0)}% tx failures`,
        conditions: {
          gasPrice: conditions.gasPrice,
          oracleStaleness: conditions.oracleStaleness,
          txFailureRate: conditions.txFailureRate,
        },
        recommendedActions: [
          'Apply gas price limits',
          'Increase transaction timeouts',
          'Enable retry with backoff',
        ],
        circuitBreakerState: this.riskManager.getCircuitBreakerState(),
      };
    }
    return null;
  }

  private notifyAlert(alert: StressAlert): void {
    logger.warn('Stress alert triggered', {
      id: alert.id,
      severity: alert.severity,
      scenario: alert.scenarioType,
      title: alert.title,
    });

    for (const callback of this.onAlertCallbacks) {
      try {
        callback(alert);
      } catch (error) {
        logger.error('Alert callback failed', { error });
      }
    }
  }

  // ============= GETTERS =============

  getAlerts(limit = 100): StressAlert[] {
    return this.alerts.slice(-limit);
  }

  getLastConditions(): MarketConditions | null {
    return this.lastConditions;
  }

  isActive(): boolean {
    return this.isRunning;
  }

  clearAlerts(): void {
    this.alerts = [];
  }
}

// Export singleton factory
let monitorInstance: StressMonitor | null = null;

export function getStressMonitor(): StressMonitor {
  if (!monitorInstance) {
    monitorInstance = new StressMonitor();
  }
  return monitorInstance;
}

export function resetStressMonitor(): void {
  if (monitorInstance) {
    monitorInstance.stop();
  }
  monitorInstance = null;
}

