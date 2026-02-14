"""
Solana Cross-DEX Arbitrage API Endpoints.

Endpoints for:
- Real-time arbitrage predictions
- DEX spread monitoring
- Execution recommendations
- Market data
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import structlog

from ...inference import SolanaArbitrageInference
from ...data.collectors import BirdeyeCollector, JupiterCollector
from ...config import settings, SOLANA_CHAIN_PARAMS

logger = structlog.get_logger()
router = APIRouter(prefix="/solana", tags=["solana"])

# Global instances (lazy loaded)
_solana_model: SolanaArbitrageInference | None = None
_birdeye: BirdeyeCollector | None = None
_jupiter: JupiterCollector | None = None


def get_solana_model() -> SolanaArbitrageInference:
    """Get or create Solana inference model."""
    global _solana_model
    if _solana_model is None:
        _solana_model = SolanaArbitrageInference()
        try:
            _solana_model.load()
            logger.info("Solana arbitrage model loaded")
        except FileNotFoundError:
            logger.warning("Solana model not found - predictions unavailable")
    return _solana_model


def get_birdeye() -> BirdeyeCollector:
    """Get Birdeye collector."""
    global _birdeye
    if _birdeye is None:
        _birdeye = BirdeyeCollector(api_key=settings.birdeye_api_key)
    return _birdeye


def get_jupiter() -> JupiterCollector:
    """Get Jupiter collector."""
    global _jupiter
    if _jupiter is None:
        _jupiter = JupiterCollector()
    return _jupiter


# Request/Response Models

class SolanaPredictionRequest(BaseModel):
    """Request for Solana arbitrage prediction."""
    token_address: str | None = Field(None, description="Token to analyze (default: SOL)")
    trade_size_usd: float = Field(10000, description="Trade size in USD")
    min_confidence: float = Field(0.65, description="Minimum confidence threshold")


class SpreadInfo(BaseModel):
    """DEX spread information."""
    spread_pct: float
    raydium_price: float | None
    orca_price: float | None
    best_buy_dex: str
    best_sell_dex: str


class CostBreakdown(BaseModel):
    """Cost breakdown for arbitrage."""
    dex_fees_pct: float
    tx_fee_pct: float
    slippage_pct: float
    total_cost_pct: float


class SolanaPredictionResponse(BaseModel):
    """Solana arbitrage prediction response."""
    prediction_id: str
    execute: bool
    probability: float
    confidence_level: str
    spread: SpreadInfo
    costs: CostBreakdown
    net_profit_pct: float
    net_profit_usd: float
    reasoning: str
    timestamp: str


class MarketDataResponse(BaseModel):
    """Current Solana market data."""
    sol_price_usd: float
    current_slot: int
    avg_priority_fee_lamports: float
    priority_fee_usd: float
    timestamp: str


# Endpoints

@router.post("/predict", response_model=SolanaPredictionResponse)
async def predict_arbitrage(request: SolanaPredictionRequest) -> SolanaPredictionResponse:
    """
    Get arbitrage prediction for Solana cross-DEX opportunity.
    
    Analyzes Raydium vs Orca spread and returns execution recommendation.
    """
    logger.info("Solana prediction request", token=request.token_address)
    
    try:
        model = get_solana_model()
        birdeye = get_birdeye()
        jupiter = get_jupiter()
        
        # Get current market data
        token = request.token_address or SOLANA_CHAIN_PARAMS['sol_mint']
        usdc = SOLANA_CHAIN_PARAMS['usdc_mint']
        
        # Fetch prices
        birdeye_data = await birdeye.fetch_latest()
        sol_price = birdeye_data.get('prices', {}).get('SOL', {}).get('price_usd', 200)
        
        # Get DEX comparison from Jupiter
        amount_lamports = int((request.trade_size_usd / sol_price) * 1e9)
        dex_comparison = await jupiter.compare_dex_routes(token, usdc, amount_lamports)
        
        spread_pct = dex_comparison.get('spread_pct', 0)
        dex_routes = dex_comparison.get('dex_routes', {})
        
        # Calculate profit
        profit_calc = model.calculate_expected_profit(
            spread_pct=spread_pct,
            trade_size_usd=request.trade_size_usd,
            sol_price=sol_price
        )
        
        # Determine execution
        execute = (
            profit_calc['profitable'] and
            profit_calc['net_profit_pct'] > 0.01
        )
        
        # Confidence level
        if profit_calc['net_profit_pct'] > 0.1:
            confidence_level = "high"
        elif profit_calc['net_profit_pct'] > 0.05:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return SolanaPredictionResponse(
            prediction_id=f"sol_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            execute=execute,
            probability=min(profit_calc['net_profit_pct'] * 10, 1.0),
            confidence_level=confidence_level,
            spread=SpreadInfo(
                spread_pct=spread_pct,
                raydium_price=dex_routes.get('raydium', {}).get('out_amount'),
                orca_price=dex_routes.get('orca', {}).get('out_amount'),
                best_buy_dex=dex_comparison.get('best_dex', 'unknown'),
                best_sell_dex='orca' if dex_comparison.get('best_dex') == 'raydium' else 'raydium'
            ),
            costs=CostBreakdown(
                dex_fees_pct=profit_calc['dex_fees_pct'],
                tx_fee_pct=profit_calc['tx_fee_pct'],
                slippage_pct=profit_calc['slippage_pct'],
                total_cost_pct=profit_calc['total_cost_pct']
            ),
            net_profit_pct=profit_calc['net_profit_pct'],
            net_profit_usd=profit_calc['net_profit_usd'],
            reasoning="Profitable opportunity" if execute else "Costs exceed spread",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Solana prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market", response_model=MarketDataResponse)
async def get_market_data() -> MarketDataResponse:
    """
    Get current Solana market data.

    Returns SOL price, slot, and priority fee information.
    """
    try:
        birdeye = get_birdeye()
        data = await birdeye.fetch_latest()

        sol_price = data.get('prices', {}).get('SOL', {}).get('price_usd', 200)
        priority_fee = data.get('avg_priority_fee_lamports', 50000)

        return MarketDataResponse(
            sol_price_usd=sol_price,
            current_slot=data.get('slot', 0),
            avg_priority_fee_lamports=priority_fee,
            priority_fee_usd=(priority_fee / 1e9) * sol_price,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error("Market data fetch failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spreads")
async def get_dex_spreads(
    token: str | None = None,
    amount_usd: float = 10000
) -> dict[str, Any]:
    """
    Get current DEX spreads for a token.

    Compares prices across Raydium, Orca, and other DEXes.
    """
    try:
        jupiter = get_jupiter()
        birdeye = get_birdeye()

        # Get SOL price for conversion
        birdeye_data = await birdeye.fetch_latest()
        sol_price = birdeye_data.get('prices', {}).get('SOL', {}).get('price_usd', 200)

        token_mint = token or SOLANA_CHAIN_PARAMS['sol_mint']
        usdc_mint = SOLANA_CHAIN_PARAMS['usdc_mint']

        # Convert USD to lamports
        amount_lamports = int((amount_usd / sol_price) * 1e9)

        # Get comparison
        comparison = await jupiter.compare_dex_routes(
            token_mint, usdc_mint, amount_lamports
        )

        return {
            "token": token_mint,
            "amount_usd": amount_usd,
            "sol_price": sol_price,
            "spread_pct": comparison.get('spread_pct', 0),
            "best_dex": comparison.get('best_dex'),
            "dex_routes": comparison.get('dex_routes', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Spread fetch failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info() -> dict[str, Any]:
    """
    Get Solana arbitrage model information.

    Returns model type, metrics, and feature information.
    """
    try:
        model = get_solana_model()
        return {
            "model": model.model_info,
            "chain_params": {
                "raydium_fee_pct": SOLANA_CHAIN_PARAMS['raydium_fee_pct'],
                "orca_fee_pct": SOLANA_CHAIN_PARAMS['orca_fee_pct'],
                "base_tx_fee_lamports": SOLANA_CHAIN_PARAMS['base_tx_fee_lamports'],
                "priority_fee_lamports": SOLANA_CHAIN_PARAMS['priority_fee_lamports']
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "model": {"loaded": False, "error": str(e)},
            "chain_params": SOLANA_CHAIN_PARAMS,
            "timestamp": datetime.utcnow().isoformat()
        }


@router.post("/simulate")
async def simulate_arbitrage(
    spread_pct: float,
    trade_size_usd: float = 10000,
    sol_price: float | None = None
) -> dict[str, Any]:
    """
    Simulate arbitrage execution.

    Calculates expected profit/loss for given spread.
    """
    try:
        model = get_solana_model()

        # Get current SOL price if not provided
        if sol_price is None:
            birdeye = get_birdeye()
            data = await birdeye.fetch_latest()
            sol_price = data.get('prices', {}).get('SOL', {}).get('price_usd', 200)

        result = model.calculate_expected_profit(
            spread_pct=spread_pct,
            trade_size_usd=trade_size_usd,
            sol_price=sol_price
        )

        return {
            "simulation": result,
            "input": {
                "spread_pct": spread_pct,
                "trade_size_usd": trade_size_usd,
                "sol_price": sol_price
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Simulation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

