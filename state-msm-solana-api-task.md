# State: MSM-VaR Solana DeFi Extension

## Task
Extend the MSM-VaR (Markov Switching Multifractal) volatility model to:
1. Support Solana DeFi data sources (Birdeye, Drift, Raydium)
2. Expose the model via a FastAPI REST API with WebSocket streaming

## Context
Part of **Cortex** — a multi-agent autonomous DeFi trading system on Solana.

## Files Created

### `solana_data_adapter.py`
- `get_token_ohlcv()` — Birdeye API OHLCV data → DataFrame [Open,High,Low,Close,Volume]
- `get_funding_rates()` — Drift Protocol perp funding rates
- `get_liquidity_metrics()` — Raydium pool TVL/volume/APR
- `ohlcv_to_returns()` — log-returns in % (MSM pipeline compatible)
- Requires: `BIRDEYE_API_KEY` env var, `httpx` package

### `api/models.py`
- Pydantic models: CalibrateRequest/Response, RegimeResponse, VaRResponse,
  VolatilityForecastResponse, BacktestSummaryResponse, TailProbResponse,
  RegimeStreamMessage, ErrorResponse
- `get_regime_name()` helper for state→label mapping

### `api/routes.py`
- `POST /calibrate` — calibrate MSM model for a token
- `GET /regime/current` — current regime state + probabilities
- `GET /var/{confidence}` — VaR at given confidence level
- `GET /volatility/forecast` — next-period σ forecast
- `GET /backtest/summary` — Kupiec + Christoffersen test results
- `GET /tail-probs` — tail probability analysis
- `WebSocket /stream/regime` — real-time regime streaming (5s interval)
- In-memory model store keyed by token symbol

### `api/main.py`
- FastAPI app with CORS, lifespan, health check
- Routes mounted at `/api/v1`
- Run: `uvicorn api.main:app --reload`

## Key Technical Details
- MSM model: K states (default 5), Bayesian filtering
- σ_{t|t-1} = forecast (out-of-sample), σ_t = filtered (in-sample)
- VaR = z_alpha × σ_forecast where z_alpha = norm.ppf(0.05) ≈ -1.645
- Returns format: 100 × diff(log(close)) as pd.Series with DatetimeIndex
- Calibration methods: mle, grid, empirical, hybrid
- Backtesting: Kupiec Unconditional Coverage + Christoffersen Independence

## Dependencies Needed (not yet installed)
```
fastapi
uvicorn[standard]
httpx
pydantic>=2.0
websockets
```

## Status: COMPLETE
All files created and verified. No IDE errors (only unresolved imports due to missing deps).

