/**
 * P&L Attribution Engine
 *
 * Breaks down realized P&L across multiple dimensions:
 *   - Strategy:  spot | lp | perps | arbitrage | lending | pumpfun
 *   - Component: sentiment | risk_model | guardian | debate | execution
 *   - Time:      hourly | daily | weekly | monthly
 *   - Token:     per-asset attribution
 *
 * Each closed trade carries an attribution vector that records which
 * components influenced the decision and by how much. The engine
 * aggregates these vectors into human-readable reports and exposes
 * them via getters for the dashboard / API layer.
 *
 * Persistence: JSON file (append-only log + periodic snapshot).
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============= TYPES =============

export type StrategyType = 'spot' | 'lp' | 'perps' | 'arbitrage' | 'lending' | 'pumpfun';
export type ComponentType = 'sentiment' | 'risk_model' | 'guardian' | 'debate' | 'execution';
export type TimeBucket = 'hourly' | 'daily' | 'weekly' | 'monthly';

export interface ComponentInfluence {
  component: ComponentType;
  weight: number;       // 0-1, how much this component influenced the decision
  signal: number;       // -1 to 1, what the component recommended
  correct: boolean;     // Was the component's signal direction aligned with outcome?
}

export interface AttributedTrade {
  id: string;
  timestamp: number;
  token: string;
  strategy: StrategyType;
  direction: 'long' | 'short' | 'buy' | 'sell';
  sizeUsd: number;
  pnlUsd: number;
  pnlPct: number;
  holdTimeMs: number;
  // Attribution breakdown
  components: ComponentInfluence[];
  // Guardian/debate context
  guardianApproved: boolean;
  debateDecision?: string;
  debateConfidence?: number;
  riskScore?: number;
  sentimentScore?: number;
  // Execution quality
  slippageBps?: number;
  gasCostUsd?: number;
  venue: string;
}

export interface StrategyPnL {
  strategy: StrategyType;
  totalPnlUsd: number;
  tradeCount: number;
  winCount: number;
  lossCount: number;
  winRate: number;
  avgPnlUsd: number;
  maxWinUsd: number;
  maxLossUsd: number;
  profitFactor: number;
  sharpeRatio: number;
  avgHoldTimeMs: number;
}

export interface ComponentPnL {
  component: ComponentType;
  attributedPnlUsd: number;
  tradeCount: number;
  avgWeight: number;
  accuracyRate: number;        // % of times signal direction matched outcome
  avgSignalStrength: number;   // avg absolute signal value
  contributionPct: number;     // % of total P&L attributed to this component
}

export interface TokenPnL {
  token: string;
  totalPnlUsd: number;
  tradeCount: number;
  winRate: number;
  avgPnlUsd: number;
  bestStrategy: StrategyType;
  worstStrategy: StrategyType;
}

export interface TimePeriodPnL {
  period: string;     // ISO date or hour key
  bucket: TimeBucket;
  totalPnlUsd: number;
  tradeCount: number;
  winRate: number;
  byStrategy: Record<string, number>;
  byComponent: Record<string, number>;
}

export interface PnLAttributionReport {
  generatedAt: number;
  periodStart: number;
  periodEnd: number;
  // Totals
  totalPnlUsd: number;
  totalTrades: number;
  overallWinRate: number;
  overallSharpe: number;
  // Breakdowns
  byStrategy: StrategyPnL[];
  byComponent: ComponentPnL[];
  byToken: TokenPnL[];
  byTime: TimePeriodPnL[];
  // Top performers
  topWinningTrades: AttributedTrade[];
  topLosingTrades: AttributedTrade[];
  // Execution quality
  avgSlippageBps: number;
  totalGasCostUsd: number;
}

// ============= CONSTANTS =============

const DATA_DIR = process.env.PNL_ATTRIBUTION_DIR || path.join(__dirname, '../../../data');
const LOG_FILE = 'pnl_attribution_log.jsonl';
const SNAPSHOT_FILE = 'pnl_attribution_snapshot.json';

// ============= P&L ATTRIBUTION ENGINE =============

export class PnLAttributionEngine {
  private trades: AttributedTrade[] = [];
  private logPath: string;
  private snapshotPath: string;
  private dirty: boolean = false;

  constructor(dataDir?: string) {
    const dir = dataDir || DATA_DIR;
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    this.logPath = path.join(dir, LOG_FILE);
    this.snapshotPath = path.join(dir, SNAPSHOT_FILE);
    this.loadSnapshot();
    logger.info('[PnLAttribution] Initialized', {
      trades: this.trades.length,
      logPath: this.logPath,
    });
  }

  // ============= RECORDING =============

  /**
   * Record a closed trade with component-level attribution.
   */
  recordTrade(trade: AttributedTrade): void {
    this.trades.push(trade);
    this.dirty = true;

    // Append to log file
    this.appendLog(trade);

    logger.info('[PnLAttribution] Trade recorded', {
      id: trade.id,
      token: trade.token,
      strategy: trade.strategy,
      pnlUsd: trade.pnlUsd.toFixed(4),
      components: trade.components.length,
    });

    // Auto-snapshot every 50 trades
    if (this.trades.length % 50 === 0) {
      this.saveSnapshot();
    }
  }

  /**
   * Build attribution vector from pre-trade signals.
   * Call this before executing a trade to capture signal state.
   */
  buildAttributionVector(signals: {
    sentimentScore?: number;
    sentimentWeight?: number;
    riskModelScore?: number;
    riskModelWeight?: number;
    guardianApproved?: boolean;
    guardianScore?: number;
    debateDecision?: string;
    debateConfidence?: number;
    executionQuality?: number;
  }): ComponentInfluence[] {
    const components: ComponentInfluence[] = [];

    if (signals.sentimentScore !== undefined) {
      components.push({
        component: 'sentiment',
        weight: signals.sentimentWeight ?? 0.2,
        signal: signals.sentimentScore,
        correct: false, // Set after trade closes
      });
    }

    if (signals.riskModelScore !== undefined) {
      components.push({
        component: 'risk_model',
        weight: signals.riskModelWeight ?? 0.25,
        signal: signals.riskModelScore,
        correct: false,
      });
    }

    if (signals.guardianScore !== undefined) {
      components.push({
        component: 'guardian',
        weight: 0.25,
        signal: signals.guardianApproved ? (1 - signals.guardianScore / 100) : -(signals.guardianScore / 100),
        correct: false,
      });
    }

    if (signals.debateConfidence !== undefined) {
      components.push({
        component: 'debate',
        weight: 0.2,
        signal: signals.debateDecision === 'approve'
          ? signals.debateConfidence
          : -signals.debateConfidence,
        correct: false,
      });
    }

    if (signals.executionQuality !== undefined) {
      components.push({
        component: 'execution',
        weight: 0.1,
        signal: signals.executionQuality,
        correct: false,
      });
    }

    // Normalize weights to sum to 1
    const totalWeight = components.reduce((s, c) => s + c.weight, 0);
    if (totalWeight > 0) {
      for (const c of components) {
        c.weight /= totalWeight;
      }
    }

    return components;
  }

  /**
   * Finalize attribution: mark which components were correct based on trade outcome.
   */
  finalizeAttribution(components: ComponentInfluence[], pnlUsd: number): ComponentInfluence[] {
    const isWin = pnlUsd > 0;
    return components.map(c => ({
      ...c,
      correct: isWin ? c.signal > 0 : c.signal < 0,
    }));
  }

  // ============= STRATEGY BREAKDOWN =============

  getStrategyBreakdown(since?: number): StrategyPnL[] {
    const filtered = since
      ? this.trades.filter(t => t.timestamp >= since)
      : this.trades;

    const byStrategy = new Map<StrategyType, AttributedTrade[]>();
    for (const trade of filtered) {
      const existing = byStrategy.get(trade.strategy) || [];
      existing.push(trade);
      byStrategy.set(trade.strategy, existing);
    }

    const result: StrategyPnL[] = [];
    for (const [strategy, trades] of byStrategy) {
      const wins = trades.filter(t => t.pnlUsd > 0);
      const losses = trades.filter(t => t.pnlUsd <= 0);
      const totalPnl = trades.reduce((s, t) => s + t.pnlUsd, 0);
      const grossWin = wins.reduce((s, t) => s + t.pnlUsd, 0);
      const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnlUsd, 0));

      // Simplified Sharpe (daily returns approximation)
      const returns = trades.map(t => t.pnlPct);
      const avgReturn = returns.length > 0
        ? returns.reduce((a, b) => a + b, 0) / returns.length
        : 0;
      const stdDev = this.calcStdDev(returns);

      result.push({
        strategy,
        totalPnlUsd: totalPnl,
        tradeCount: trades.length,
        winCount: wins.length,
        lossCount: losses.length,
        winRate: trades.length > 0 ? wins.length / trades.length : 0,
        avgPnlUsd: trades.length > 0 ? totalPnl / trades.length : 0,
        maxWinUsd: wins.length > 0 ? Math.max(...wins.map(t => t.pnlUsd)) : 0,
        maxLossUsd: losses.length > 0 ? Math.min(...losses.map(t => t.pnlUsd)) : 0,
        profitFactor: grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0,
        sharpeRatio: stdDev > 0 ? avgReturn / stdDev : 0,
        avgHoldTimeMs: trades.length > 0
          ? trades.reduce((s, t) => s + t.holdTimeMs, 0) / trades.length
          : 0,
      });
    }

    return result.sort((a, b) => b.totalPnlUsd - a.totalPnlUsd);
  }

  // ============= COMPONENT BREAKDOWN =============

  getComponentBreakdown(since?: number): ComponentPnL[] {
    const filtered = since
      ? this.trades.filter(t => t.timestamp >= since)
      : this.trades;

    const totalPnl = filtered.reduce((s, t) => s + t.pnlUsd, 0);

    const componentMap = new Map<ComponentType, {
      attributedPnl: number;
      tradeCount: number;
      totalWeight: number;
      correctCount: number;
      totalSignalAbs: number;
    }>();

    for (const trade of filtered) {
      for (const comp of trade.components) {
        const existing = componentMap.get(comp.component) || {
          attributedPnl: 0,
          tradeCount: 0,
          totalWeight: 0,
          correctCount: 0,
          totalSignalAbs: 0,
        };

        // Attributed P&L: trade P&L * component weight
        existing.attributedPnl += trade.pnlUsd * comp.weight;
        existing.tradeCount += 1;
        existing.totalWeight += comp.weight;
        if (comp.correct) existing.correctCount += 1;
        existing.totalSignalAbs += Math.abs(comp.signal);

        componentMap.set(comp.component, existing);
      }
    }

    const result: ComponentPnL[] = [];
    for (const [component, data] of componentMap) {
      result.push({
        component,
        attributedPnlUsd: data.attributedPnl,
        tradeCount: data.tradeCount,
        avgWeight: data.tradeCount > 0 ? data.totalWeight / data.tradeCount : 0,
        accuracyRate: data.tradeCount > 0 ? data.correctCount / data.tradeCount : 0,
        avgSignalStrength: data.tradeCount > 0 ? data.totalSignalAbs / data.tradeCount : 0,
        contributionPct: totalPnl !== 0 ? (data.attributedPnl / totalPnl) * 100 : 0,
      });
    }

    return result.sort((a, b) => b.attributedPnlUsd - a.attributedPnlUsd);
  }

  // ============= TOKEN BREAKDOWN =============

  getTokenBreakdown(since?: number): TokenPnL[] {
    const filtered = since
      ? this.trades.filter(t => t.timestamp >= since)
      : this.trades;

    const byToken = new Map<string, AttributedTrade[]>();
    for (const trade of filtered) {
      const existing = byToken.get(trade.token) || [];
      existing.push(trade);
      byToken.set(trade.token, existing);
    }

    const result: TokenPnL[] = [];
    for (const [token, trades] of byToken) {
      const wins = trades.filter(t => t.pnlUsd > 0);
      const totalPnl = trades.reduce((s, t) => s + t.pnlUsd, 0);

      // Best/worst strategy
      const stratPnl = new Map<StrategyType, number>();
      for (const t of trades) {
        stratPnl.set(t.strategy, (stratPnl.get(t.strategy) || 0) + t.pnlUsd);
      }
      let bestStrategy: StrategyType = 'spot';
      let worstStrategy: StrategyType = 'spot';
      let bestPnl = -Infinity;
      let worstPnl = Infinity;
      for (const [strat, pnl] of stratPnl) {
        if (pnl > bestPnl) { bestPnl = pnl; bestStrategy = strat; }
        if (pnl < worstPnl) { worstPnl = pnl; worstStrategy = strat; }
      }

      result.push({
        token,
        totalPnlUsd: totalPnl,
        tradeCount: trades.length,
        winRate: trades.length > 0 ? wins.length / trades.length : 0,
        avgPnlUsd: trades.length > 0 ? totalPnl / trades.length : 0,
        bestStrategy,
        worstStrategy,
      });
    }

    return result.sort((a, b) => b.totalPnlUsd - a.totalPnlUsd);
  }

  // ============= TIME BREAKDOWN =============

  getTimeBreakdown(bucket: TimeBucket = 'daily', since?: number): TimePeriodPnL[] {
    const filtered = since
      ? this.trades.filter(t => t.timestamp >= since)
      : this.trades;

    const byPeriod = new Map<string, AttributedTrade[]>();
    for (const trade of filtered) {
      const key = this.getTimeBucketKey(trade.timestamp, bucket);
      const existing = byPeriod.get(key) || [];
      existing.push(trade);
      byPeriod.set(key, existing);
    }

    const result: TimePeriodPnL[] = [];
    for (const [period, trades] of byPeriod) {
      const wins = trades.filter(t => t.pnlUsd > 0);
      const totalPnl = trades.reduce((s, t) => s + t.pnlUsd, 0);

      // Strategy breakdown for this period
      const byStrategy: Record<string, number> = {};
      for (const t of trades) {
        byStrategy[t.strategy] = (byStrategy[t.strategy] || 0) + t.pnlUsd;
      }

      // Component breakdown for this period
      const byComponent: Record<string, number> = {};
      for (const t of trades) {
        for (const c of t.components) {
          byComponent[c.component] = (byComponent[c.component] || 0) + t.pnlUsd * c.weight;
        }
      }

      result.push({
        period,
        bucket,
        totalPnlUsd: totalPnl,
        tradeCount: trades.length,
        winRate: trades.length > 0 ? wins.length / trades.length : 0,
        byStrategy,
        byComponent,
      });
    }

    return result.sort((a, b) => a.period.localeCompare(b.period));
  }

  // ============= FULL REPORT =============

  generateReport(since?: number): PnLAttributionReport {
    const periodStart = since || (this.trades.length > 0 ? this.trades[0].timestamp : Date.now());
    const periodEnd = Date.now();
    const filtered = since
      ? this.trades.filter(t => t.timestamp >= since)
      : this.trades;

    const totalPnl = filtered.reduce((s, t) => s + t.pnlUsd, 0);
    const wins = filtered.filter(t => t.pnlUsd > 0);
    const returns = filtered.map(t => t.pnlPct);
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const stdDev = this.calcStdDev(returns);

    // Slippage and gas
    const tradesWithSlippage = filtered.filter(t => t.slippageBps !== undefined);
    const avgSlippage = tradesWithSlippage.length > 0
      ? tradesWithSlippage.reduce((s, t) => s + (t.slippageBps || 0), 0) / tradesWithSlippage.length
      : 0;
    const totalGas = filtered.reduce((s, t) => s + (t.gasCostUsd || 0), 0);

    // Top trades
    const sorted = [...filtered].sort((a, b) => b.pnlUsd - a.pnlUsd);

    return {
      generatedAt: Date.now(),
      periodStart,
      periodEnd,
      totalPnlUsd: totalPnl,
      totalTrades: filtered.length,
      overallWinRate: filtered.length > 0 ? wins.length / filtered.length : 0,
      overallSharpe: stdDev > 0 ? avgReturn / stdDev : 0,
      byStrategy: this.getStrategyBreakdown(since),
      byComponent: this.getComponentBreakdown(since),
      byToken: this.getTokenBreakdown(since),
      byTime: this.getTimeBreakdown('daily', since),
      topWinningTrades: sorted.slice(0, 5),
      topLosingTrades: sorted.slice(-5).reverse(),
      avgSlippageBps: avgSlippage,
      totalGasCostUsd: totalGas,
    };
  }

  // ============= PERSISTENCE =============

  private appendLog(trade: AttributedTrade): void {
    try {
      const dir = path.dirname(this.logPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.appendFileSync(this.logPath, JSON.stringify(trade) + '\n');
    } catch (err) {
      logger.error('[PnLAttribution] Failed to append log', { error: String(err) });
    }
  }

  saveSnapshot(): void {
    try {
      const dir = path.dirname(this.snapshotPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      const tmpPath = this.snapshotPath + '.tmp';
      fs.writeFileSync(tmpPath, JSON.stringify(this.trades, null, 2));
      fs.renameSync(tmpPath, this.snapshotPath);
      this.dirty = false;
      logger.info('[PnLAttribution] Snapshot saved', { trades: this.trades.length });
    } catch (err) {
      logger.error('[PnLAttribution] Failed to save snapshot', { error: String(err) });
    }
  }

  private loadSnapshot(): void {
    // Prefer snapshot for fast startup
    if (fs.existsSync(this.snapshotPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.snapshotPath, 'utf-8'));
        if (Array.isArray(data)) {
          this.trades = data;
          return;
        }
      } catch {
        logger.warn('[PnLAttribution] Failed to load snapshot, trying log replay');
      }
    }

    // Fall back to log replay
    if (fs.existsSync(this.logPath)) {
      try {
        const content = fs.readFileSync(this.logPath, 'utf-8');
        const lines = content.trim().split('\n').filter(Boolean);
        for (const line of lines) {
          try {
            this.trades.push(JSON.parse(line));
          } catch {
            // skip malformed lines
          }
        }
      } catch {
        logger.warn('[PnLAttribution] Failed to load log file');
      }
    }
  }

  // ============= HELPERS =============

  private getTimeBucketKey(timestamp: number, bucket: TimeBucket): string {
    const d = new Date(timestamp);
    switch (bucket) {
      case 'hourly':
        return `${d.toISOString().slice(0, 13)}:00Z`;
      case 'daily':
        return d.toISOString().slice(0, 10);
      case 'weekly': {
        // ISO week start (Monday)
        const day = d.getDay();
        const diff = d.getDate() - day + (day === 0 ? -6 : 1);
        const monday = new Date(d.getTime());
        monday.setDate(diff);
        return `W${monday.toISOString().slice(0, 10)}`;
      }
      case 'monthly':
        return d.toISOString().slice(0, 7);
    }
  }

  private calcStdDev(values: number[]): number {
    if (values.length < 2) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }

  getTradeCount(): number {
    return this.trades.length;
  }

  getTrades(limit?: number): AttributedTrade[] {
    if (limit) return this.trades.slice(-limit);
    return [...this.trades];
  }
}

// ============= SINGLETON =============

let instance: PnLAttributionEngine | null = null;

export function getPnLAttributionEngine(dataDir?: string): PnLAttributionEngine {
  if (!instance) {
    instance = new PnLAttributionEngine(dataDir);
  }
  return instance;
}

export function resetPnLAttributionEngine(): void {
  instance = null;
}
