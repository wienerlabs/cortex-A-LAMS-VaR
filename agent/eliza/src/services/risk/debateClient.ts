/**
 * Adversarial Debate HTTP Client
 *
 * Bridges the TypeScript runtime (ElizaOS) to the Python adversarial debate
 * pipeline. Provides debate execution and outcome circuit breaker management.
 *
 * The Python API runs at CORTEX_API_URL (default http://localhost:8000) and
 * exposes:
 *   POST /guardian/debate                — run standalone debate
 *   POST /guardian/assess                — full assessment with optional debate
 *   POST /guardian/trade-outcome/strategy — record trade outcome for circuit breakers
 *   GET  /guardian/circuit-breakers/outcomes — outcome-based circuit breaker states
 */

import { logger } from '../logger.js';
import type { DebateResult, OutcomeCircuitBreakerStatus } from './types.js';

// ============= CONFIGURATION =============

const DEFAULT_API_URL = process.env.CORTEX_API_URL || 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = parseInt(process.env.DEBATE_TIMEOUT_MS || '10000', 10);

interface DebateClientConfig {
  apiUrl: string;
  timeoutMs: number;
}

// ============= DEBATE CLIENT =============

export class DebateClient {
  private config: DebateClientConfig;
  private abortController: AbortController | null = null;

  constructor(config?: Partial<DebateClientConfig>) {
    this.config = {
      apiUrl: config?.apiUrl ?? DEFAULT_API_URL,
      timeoutMs: config?.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    };
    logger.info('[DebateClient] Initialized', {
      apiUrl: this.config.apiUrl,
      timeoutMs: this.config.timeoutMs,
    });
  }

  /**
   * Run a standalone adversarial debate for a trade proposal.
   */
  async runDebate(params: {
    token: string;
    direction: string;
    trade_size_usd: number;
    strategy?: string;
    run_debate?: boolean;
  }): Promise<DebateResult> {
    const url = `${this.config.apiUrl}/guardian/debate`;
    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: params.token,
          direction: params.direction,
          trade_size_usd: params.trade_size_usd,
          strategy: params.strategy ?? 'spot',
          run_debate: true,
        }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Debate API error ${response.status}: ${text}`);
      }

      return (await response.json()) as DebateResult;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Debate request timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Record a trade outcome for strategy-specific circuit breakers.
   *
   * Feeds the outcome-based breakers:
   * - LP: 3 consecutive IL → pause
   * - Arb: 5 consecutive failed executions → pause
   * - Perp: 2 consecutive stop-losses → pause
   */
  async recordTradeOutcome(params: {
    strategy: string;
    success: boolean;
    pnl?: number;
    loss_type?: string;
    details?: string;
  }): Promise<OutcomeCircuitBreakerStatus> {
    const url = new URL(`${this.config.apiUrl}/guardian/trade-outcome/strategy`);
    url.searchParams.set('strategy', params.strategy);
    url.searchParams.set('success', String(params.success));
    if (params.pnl !== undefined) url.searchParams.set('pnl', String(params.pnl));
    if (params.loss_type) url.searchParams.set('loss_type', params.loss_type);
    if (params.details) url.searchParams.set('details', params.details);

    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url.toString(), {
        method: 'POST',
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Outcome record API error ${response.status}: ${text}`);
      }

      return (await response.json()) as OutcomeCircuitBreakerStatus;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Outcome record timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Get outcome-based circuit breaker states.
   */
  async getOutcomeBreakerStates(): Promise<{
    outcome_breakers: OutcomeCircuitBreakerStatus[];
    timestamp: number;
  }> {
    const url = `${this.config.apiUrl}/guardian/circuit-breakers/outcomes`;
    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'GET',
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Outcome breakers API error ${response.status}: ${text}`);
      }

      return (await response.json()) as {
        outcome_breakers: OutcomeCircuitBreakerStatus[];
        timestamp: number;
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Outcome breakers request timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Check if a strategy is blocked by outcome circuit breakers.
   */
  async isStrategyBlocked(strategy: string): Promise<boolean> {
    try {
      const states = await this.getOutcomeBreakerStates();
      const breaker = states.outcome_breakers.find(b => b.strategy === strategy);
      return (breaker?.state === 'open') || false;
    } catch (error) {
      logger.warn('[DebateClient] Failed to check strategy block status', {
        strategy,
        error: error instanceof Error ? error.message : error,
      });
      return false;
    }
  }

  /**
   * Cancel any in-flight requests.
   */
  cancel(): void {
    this.abortController?.abort();
  }
}

// ============= SINGLETON =============

let instance: DebateClient | null = null;

export function getDebateClient(config?: Partial<DebateClientConfig>): DebateClient {
  if (!instance) {
    instance = new DebateClient(config);
  }
  return instance;
}

export function resetDebateClient(): void {
  instance?.cancel();
  instance = null;
}
