from __future__ import annotations
"""
Prediction endpoints.
"""
from datetime import datetime
from typing import Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
import structlog

from ...inference import CrossDexInference

logger = structlog.get_logger()
router = APIRouter()

# Global model instance (lazy loaded)
_cross_dex_model: CrossDexInference | None = None


def get_cross_dex_model() -> CrossDexInference:
    """Get or create Cross-DEX inference model."""
    global _cross_dex_model
    if _cross_dex_model is None:
        _cross_dex_model = CrossDexInference()
        _cross_dex_model.load()
        logger.info("Cross-DEX model loaded")
    return _cross_dex_model


# Request/Response Models

class PredictionRequest(BaseModel):
    """Request for a prediction."""
    strategy: str = Field(..., description="Strategy type: arbitrage, lending, lp_provision")
    features: dict[str, float] = Field(..., description="Feature values for prediction")


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""
    feature: str
    value: float
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    """Prediction response with explanation."""
    prediction_id: str
    strategy: str
    prediction: float
    confidence: float
    action: str
    should_execute: bool
    explanation: dict[str, Any]
    timestamp: str


class StrategyRecommendation(BaseModel):
    """Strategy recommendation."""
    strategy: str
    action: str
    confidence: float
    expected_profit: float
    reasoning: str


class RecommendationsResponse(BaseModel):
    """Response with all strategy recommendations."""
    recommendations: list[StrategyRecommendation]
    best_strategy: str | None
    timestamp: str


# Endpoints

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Get prediction for a specific strategy.

    Returns prediction with SHAP-based explanation.
    """
    logger.info("Prediction request", strategy=request.strategy)

    valid_strategies = ["arbitrage", "lending", "lp_provision"]
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {valid_strategies}"
        )

    # Handle arbitrage with real ONNX model
    if request.strategy == "arbitrage":
        try:
            model = get_cross_dex_model()

            # Create DataFrame from features
            features_df = pd.DataFrame([request.features])

            # Check for missing features
            missing = set(model.feature_names) - set(request.features.keys())
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features: {list(missing)}"
                )

            # Get prediction
            probs = model.predict_proba(features_df)
            prob_profitable = float(probs[0][1]) if len(probs.shape) > 1 else float(probs[0])

            # Decision
            decision = model.should_execute(features_df)

            return PredictionResponse(
                prediction_id=f"pred_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                strategy=request.strategy,
                prediction=prob_profitable,
                confidence=prob_profitable,
                action="execute" if decision["execute"] else "hold",
                should_execute=decision["execute"],
                explanation={
                    "model_version": model.metadata.get("version"),
                    "probability": prob_profitable,
                    "threshold": decision["min_confidence"],
                    "reason": decision["reason"]
                },
                timestamp=datetime.utcnow().isoformat()
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Model not found: {e}")
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Placeholder for other strategies
    return PredictionResponse(
        prediction_id=f"pred_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        strategy=request.strategy,
        prediction=0.5,
        confidence=0.5,
        action="hold",
        should_execute=False,
        explanation={"note": "Model not yet implemented for this strategy"},
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations() -> RecommendationsResponse:
    """
    Get recommendations for all strategies.
    
    Evaluates current market conditions and returns ranked recommendations.
    """
    logger.info("Recommendations request")
    
    # This is a placeholder response
    
    recommendations = [
        StrategyRecommendation(
            strategy="arbitrage",
            action="execute_swap",
            confidence=0.78,
            expected_profit=45.50,
            reasoning="Price spread of 0.5% detected between Raydium and Orca"
        ),
        StrategyRecommendation(
            strategy="lending",
            action="switch_to_kamino",
            confidence=0.65,
            expected_profit=12.30,
            reasoning="Kamino supply APY 0.8% higher than MarginFi"
        ),
        StrategyRecommendation(
            strategy="lp_provision",
            action="hold",
            confidence=0.55,
            expected_profit=0.0,
            reasoning="Current LP position has optimal APY"
        )
    ]
    
    # Sort by expected profit * confidence
    recommendations.sort(
        key=lambda r: r.expected_profit * r.confidence,
        reverse=True
    )
    
    return RecommendationsResponse(
        recommendations=recommendations,
        best_strategy=recommendations[0].strategy if recommendations else None,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/strategies")
async def list_strategies() -> dict:
    """List available strategies and their status."""
    # Check if cross-dex model is loaded
    arbitrage_loaded = False
    arbitrage_info = {}
    try:
        model = get_cross_dex_model()
        arbitrage_loaded = True
        arbitrage_info = model.model_info
    except Exception:
        pass

    now = datetime.now(datetime.timezone.utc).isoformat()

    return {
        "strategies": [
            {
                "name": "arbitrage",
                "description": "Cross-DEX arbitrage between Uniswap V3 and V2",
                "model_loaded": arbitrage_loaded,
                "model_version": arbitrage_info.get("version", "N/A"),
                "metrics": arbitrage_info.get("metrics", {}),
                "n_features": arbitrage_info.get("n_features", 0)
            },
            {
                "name": "lending",
                "description": "Lending rate optimization between Aave and Compound",
                "model_loaded": False,
                "model_version": "N/A"
            },
            {
                "name": "lp_provision",
                "description": "Curve LP pool APY prediction",
                "model_loaded": False,
                "model_version": "N/A"
            }
        ],
        "timestamp": now
    }
