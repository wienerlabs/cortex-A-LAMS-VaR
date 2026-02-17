# Solana Cross-DEX Arbitrage Pipeline

## Overview

This document describes the Solana cross-DEX arbitrage pipeline that identifies and executes profitable trading opportunities between Raydium and Orca DEXes.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Collectors │────▶│ Feature Engineer │────▶│  XGBoost Model  │
│  (Birdeye/Jupiter)│     │  (Solana-specific)│     │  (Arbitrage)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  API Endpoints  │◀────│ Inference Engine │◀────│ Execution Engine│
│  (FastAPI)      │     │ (Real-time)      │     │ (Simulation)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Components

### 1. Data Collectors

#### BirdeyeCollector
- Fetches real-time token prices from Birdeye API
- Provides historical OHLCV data
- Monitors priority fees and network conditions

#### JupiterCollector
- Gets swap quotes from Jupiter aggregator
- Compares routes across DEXes (Raydium, Orca, etc.)
- Calculates price impact and slippage

### 2. Feature Engineering

#### SolanaFeatureEngineer
Creates Solana-specific features:
- `priority_fee_normalized`: Normalized priority fee (0-1)
- `tx_cost_pct`: Transaction cost as percentage of trade
- `slot_time_deviation`: Block time deviation
- Technical indicators (RSI, volatility, volume MA)

#### CrossDexFeatureEngineer
Creates cross-DEX spread features:
- `spread_abs`: Absolute price difference
- `spread_pct`: Percentage spread
- `spread_zscore`: Z-score of spread
- `raydium_premium` / `orca_premium`: DEX-specific premiums

### 3. Model Training

#### SolanaArbitrageModel
XGBoost classifier trained to predict profitable arbitrage:
- Binary classification (profitable/not profitable)
- Features: spread, fees, volatility, volume
- Outputs probability of profitable trade

Training command:
```bash
python -m src.models.training.solana_trainer \
    --data-path data/solana_training.parquet \
    --output-dir models/solana
```

### 4. Inference Engine

#### SolanaArbitrageInference
Real-time prediction engine:
- Loads trained XGBoost or ONNX model
- Calculates expected profit/loss
- Provides execution recommendations

Key methods:
- `predict()`: Get probability of profitable trade
- `predict_with_decision()`: Get execution recommendation
- `calculate_expected_profit()`: Detailed profit breakdown

### 5. Execution Engine

#### SolanaExecutor
Transaction execution (simulation mode):
- Jupiter swap execution
- Priority fee optimization
- Arbitrage simulation for backtesting

## API Endpoints

### Prediction
```
POST /api/v1/solana/predict
{
    "token_address": "So11111111111111111111111111111111111111112",
    "trade_size_usd": 10000,
    "min_confidence": 0.65
}
```

### Market Data
```
GET /api/v1/solana/market
```

### DEX Spreads
```
GET /api/v1/solana/spreads?amount_usd=10000
```

### Simulation
```
POST /api/v1/solana/simulate?spread_pct=0.3&trade_size_usd=10000
```

## Configuration

Environment variables:
```bash
BIRDEYE_API_KEY=your_api_key
HELIUS_RPC_URL=https://mainnet.helius-rpc.com/?api-key=xxx
SOLANA_PRIVATE_KEY=your_private_key  # For execution
```

Chain parameters (in `config.py`):
```python
SOLANA_CHAIN_PARAMS = {
    'raydium_fee_pct': 0.0025,      # 0.25%
    'orca_fee_pct': 0.003,          # 0.30%
    'base_tx_fee_lamports': 5000,
    'priority_fee_lamports': 50000,
    'sol_mint': 'So11111111111111111111111111111111111111112',
    'usdc_mint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
}
```

## Profit Calculation

Net profit = Spread - (DEX fees + TX fees + Slippage)

Example for $10,000 trade with 0.3% spread:
- Gross profit: 0.30% = $30
- Raydium fee: 0.25% = $25
- Orca fee: 0.30% = $30
- TX fees: ~$0.02
- Slippage: ~0.1% = $10
- **Net profit: -$35** (not profitable)

Minimum profitable spread: ~0.65% for $10k trades

## Testing

Run tests:
```bash
pytest tests/test_solana_pipeline.py -v
pytest tests/test_solana_api.py -v
```

## Future Improvements

1. **Real execution**: Implement actual Solana transaction signing
2. **MEV protection**: Add Jito bundles for MEV protection
3. **Multi-token**: Extend to other token pairs
4. **Latency optimization**: WebSocket connections for faster data

