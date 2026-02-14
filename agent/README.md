# Cortex Agent

AI-powered DeFi agent for Solana LP rebalancing and arbitrage.

## Architecture

### Python (Offline - Training & Analysis)
- **Data Collection**: Birdeye, Jupiter, Helius APIs
- **Feature Engineering**: 61 features for ML models
- **Model Training**: XGBoost with hyperparameter tuning
- **Backtesting**: Historical performance analysis
- **Optimization**: Threshold tuning, cost modeling
- **Export**: ONNX model for production inference

### TypeScript (Online - Production)
- **ONNX Inference**: Real-time model predictions
- **Risk Management**: Data-driven position sizing, daily limits
- **Trade Execution**: Jupiter aggregator integration
- **MEV Protection**: Jito bundles for sandwich attack prevention
- **State Management**: Eliza framework integration
- **Monitoring**: Performance tracking, drift detection

## Directory Structure

```
agent/
├── src/                    # Python - Offline
│   ├── data/               # Data collectors & processors
│   ├── features/           # Feature engineering
│   ├── models/             # Training & ONNX export
│   ├── inference/          # Python ONNX runtime (testing)
│   ├── execution/          # State machine (deprecated executor)
│   ├── monitoring/         # Drift detection, metrics
│   └── api/                # FastAPI endpoints
│
├── eliza/                  # TypeScript - Production
│   ├── src/
│   │   ├── actions/        # Eliza actions (analyze, rebalance)
│   │   ├── services/       # Core services
│   │   │   ├── jitoService.ts    # MEV protection
│   │   │   ├── riskManager.ts    # Risk management
│   │   │   └── monitor.ts        # Pool monitoring
│   │   └── providers/      # Data providers
│   └── config/
│       └── risk.json       # Risk parameters
│
├── models/                 # Trained models
│   └── lp_rebalancer/
│       ├── model.onnx      # Production model
│       └── metadata/       # Training reports
│
├── data/                   # Training data
│   └── lp_rebalancer/
│       └── features/       # Feature CSVs
│
└── scripts/                # Utility scripts
    └── risk_analyzer.py    # Risk limit calculator
```

## Quick Start

### Training (Python)
```bash
cd agent
pip install -r requirements.txt

# Collect data
python -m src.data.collectors.birdeye_collector

# Train model
python -m src.models.training.train_xgboost

# Optimize
python scripts/optimize_lp_rebalancer.py
```

### Production (TypeScript)
```bash
cd agent/eliza
npm install
npm run build

# Start agent
npm start
```

## Configuration

### Environment Variables
```bash
# Solana
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_PRIVATE_KEY=<base58-encoded>

# APIs
BIRDEYE_API_KEY=<key>
JUPITER_API_URL=https://quote-api.jup.ag/v6

# Mode
SIMULATION_MODE=true  # Set false for live trading
```

### Risk Parameters
See `eliza/config/risk.json` for data-driven risk limits:
- Max position: 3.7% (Half-Kelly)
- Max daily loss: 5%
- Max daily trades: 2
- Cooldown: 24h

## Recent Changes

### 2026-01-06
- ✅ Removed Python executor (TypeScript is production-ready)
- ✅ Added Jito MEV protection
- ✅ Implemented data-driven risk management
- ✅ Alpha improved: -0.15% → +9.62%

## License

MIT

