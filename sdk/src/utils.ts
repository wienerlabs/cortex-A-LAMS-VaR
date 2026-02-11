import {
  retry,
  circuitBreaker,
  timeout,
  wrap,
  handleAll,
  ExponentialBackoff,
  ConsecutiveBreaker,
  TimeoutStrategy,
  TaskCancelledError,
  BrokenCircuitError,
} from "cockatiel";
import type { IPolicy } from "cockatiel";
import { CircuitBreakerOpenError, RiskEngineTimeout } from "./errors";

export interface ResilienceConfig {
  timeoutMs: number;
  maxRetries: number;
  retryBaseDelay: number;
  cbThreshold: number;
  cbResetMs: number;
}

export function createResiliencePolicy(config: ResilienceConfig): IPolicy {
  const timeoutPolicy = timeout(config.timeoutMs, TimeoutStrategy.Aggressive);
  const retryPolicy = retry(handleAll, {
    maxAttempts: config.maxRetries,
    backoff: new ExponentialBackoff({ initialDelay: config.retryBaseDelay }),
  });
  const cbPolicy = circuitBreaker(handleAll, {
    halfOpenAfter: config.cbResetMs,
    breaker: new ConsecutiveBreaker(config.cbThreshold),
  });
  return wrap(retryPolicy, cbPolicy, timeoutPolicy);
}

export async function executeWithResilience<T>(
  policy: IPolicy,
  fn: (signal: AbortSignal) => Promise<T>,
  url: string,
  timeoutMs: number,
): Promise<T> {
  try {
    return await policy.execute(({ signal }) => fn(signal));
  } catch (err) {
    if (err instanceof TaskCancelledError) {
      throw new RiskEngineTimeout(url, timeoutMs);
    }
    if (err instanceof BrokenCircuitError) {
      throw new CircuitBreakerOpenError(0);
    }
    throw err;
  }
}

