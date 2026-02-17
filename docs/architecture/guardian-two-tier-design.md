# Guardian Two-Tier Design

## Overview

The Cortex protocol uses **two separate Guardian systems** that operate at different
layers of the execution pipeline. This is an intentional architectural decision, not
duplication.

## Tier 1: TypeScript GuardianValidator (Agent-Side)

**Location:** `agent/eliza/src/agents/crtxAgent.ts` (GuardianValidator class)

**Purpose:** Fast, local pre-filter that runs inside the TS agent process before any
network call is made.

**Checks:**
- Portfolio concentration limits (max % per token)
- Daily volume caps (USD)
- Per-trade size caps
- Cooldown timers between trades
- Basic sanity checks (valid mint, valid direction)

**Characteristics:**
- Runs synchronously in the agent event loop
- No network calls, no model inference
- Sub-millisecond latency
- No shared state with the Python engine
- Stateless across restarts (limits reset)

## Tier 2: Python cortex.guardian (Risk Engine-Side)

**Location:** `cortex/guardian.py`

**Purpose:** Comprehensive 6-component risk scoring engine with ML model outputs,
circuit breakers, Kelly sizing, and adversarial debate.

**Components (weights):**
- A-LAMS VaR: 25% (regime-conditional per-strategy VaR)
- EVT tail risk: 20% (extreme value theory)
- SVJ jump risk: 15% (stochastic volatility with jumps)
- Hawkes clustering: 15% (self-exciting event intensity)
- Regime state: 15% (MSM Markov-switching model)
- News sentiment: 10% (NLP-based market signal)

**Characteristics:**
- Called via HTTP (`POST /api/v1/guardian/assess`)
- 50-500ms latency depending on cache state
- Requires calibrated models (MSM, EVT, SVJ, Hawkes)
- Feeds circuit breakers (score-based + outcome-based)
- Computes Kelly-optimal position size
- Optionally triggers adversarial debate for large trades
- State persisted to Redis (Kelly history, debate outcomes, CB state)

## Data Flow

```
Agent decides to trade
       |
       v
[Tier 1] GuardianValidator.validateTrade()
       |  (local, <1ms)
       |  Rejected? --> stop, log reason
       v
[Tier 2] POST /api/v1/guardian/assess
       |  (network, 50-500ms)
       |  Rejected? --> stop, log veto_reasons
       v
Trade execution via Jupiter Swap API
       |
       v
Record outcome to /trade-outcome
  --> Kelly stats, debate priors, circuit breakers
```

## Why Two Tiers?

1. **Latency**: The TS validator catches obvious violations (over daily limit, too
   large a position) without waiting for a network round-trip to the Python engine.

2. **Availability**: If the Python risk engine is down, the TS validator still
   prevents the agent from exceeding basic limits.

3. **Separation of Concerns**: The TS agent owns portfolio-level constraints (how
   much to risk). The Python engine owns market-level risk assessment (how dangerous
   is this trade right now).

4. **No Shared State**: By design, the two tiers share no state. The TS validator
   doesn't know the current EVT score, and the Python engine doesn't know the agent's
   daily volume counter. This prevents coupling and makes each tier independently
   testable.

## Important Notes

- Both tiers must approve for a trade to execute
- The Python tier can reduce position size via `recommended_size`
- Circuit breaker trips in Python block ALL trades for a strategy, regardless of TS approval
- The TS tier's limits are configurable via `PUT /api/agent/limits` (backend)
- The Python tier's thresholds come from `cortex/config.py` environment variables
