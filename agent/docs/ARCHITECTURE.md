# Cortex Agent Architecture

## Overview

Cortex uses a **dual-language architecture** with clear separation of concerns:

- **Python**: Offline processing (training, backtesting, optimization)
- **TypeScript**: Online production (inference, execution, monitoring)

## Execution Flow

### Training Pipeline (Python)

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE - Python                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Data    │───▶│ Features │───▶│ XGBoost  │              │
│  │Collection│    │ (61 dim) │    │ Training │              │
│  └──────────┘    └──────────┘    └────┬─────┘              │
│                                       │                     │
│                                       ▼                     │
│                              ┌──────────────┐               │
│                              │  ONNX Export │               │
│                              └──────┬───────┘               │
│                                     │                       │
└─────────────────────────────────────┼───────────────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  model.onnx  │
                              └──────────────┘
```

### Production Pipeline (TypeScript)

```
┌─────────────────────────────────────────────────────────────┐
│                    ONLINE - TypeScript                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  ONNX    │───▶│   Risk   │───▶│  Jito    │              │
│  │Inference │    │ Manager  │    │ Execute  │              │
│  └──────────┘    └────┬─────┘    └────┬─────┘              │
│                       │               │                     │
│                       ▼               ▼                     │
│               ┌───────────────────────────┐                 │
│               │     Eliza Framework       │                 │
│               │  (Actions, State, Memory) │                 │
│               └───────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Python Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Data Collectors | `src/data/collectors/` | Birdeye, Jupiter API data |
| Feature Engineering | `src/features/` | 61 features for ML |
| Model Training | `src/models/training/` | XGBoost + ONNX export |
| Backtesting | `scripts/` | Historical performance |
| Risk Analyzer | `scripts/risk_analyzer.py` | Calculate optimal limits |
| Optimization | `scripts/optimize_*.py` | Threshold tuning |

### TypeScript Components

| Component | Location | Purpose |
|-----------|----------|---------|
| ONNX Inference | `eliza/src/services/` | Real-time predictions |
| Risk Manager | `eliza/src/services/riskManager.ts` | Position sizing, limits |
| Jito Service | `eliza/src/services/jitoService.ts` | MEV protection |
| Pool Monitor | `eliza/src/services/monitor.ts` | Market data |
| Execute Action | `eliza/src/actions/executeRebalance.ts` | Trade execution |
| Analyze Action | `eliza/src/actions/analyzePool.ts` | Pool analysis |

## Deprecation Notes

### Removed: Python Executor (2026-01-06)

**File**: `src/execution/solana_executor.py`

**Reason**: TypeScript execution is production-ready with:
- Full Jupiter integration
- Jito MEV protection
- Risk management
- Eliza state management

**Migration**: Use `eliza/src/actions/executeRebalance.ts` for all trade execution.

### Kept: State Machine

**File**: `src/execution/state_machine.py`

**Reason**: Generic state machine pattern, useful for backtesting simulation.

## Data Flow

```
1. Training (Daily/Weekly)
   Python → Train → ONNX → Deploy

2. Inference (Real-time)
   Market Data → ONNX → Prediction → Risk Check → Execute

3. Risk Management
   Backtest Analysis → Risk Limits → RiskManager → Trade Filter
```

## Configuration Files

| File | Purpose |
|------|---------|
| `eliza/config/risk.json` | Risk parameters |
| `models/lp_rebalancer/metadata/optimization_report.json` | Training results |
| `models/lp_rebalancer/metadata/risk_analysis.json` | Risk calculations |

## API Endpoints (Python - Optional)

FastAPI server for internal testing:
- `GET /health` - Health check
- `POST /predict` - Model inference
- `GET /metrics` - Prometheus metrics

**Note**: Production inference uses TypeScript ONNX runtime directly.

