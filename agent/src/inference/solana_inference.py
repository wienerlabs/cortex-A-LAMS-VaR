"""
Solana Cross-DEX Arbitrage Inference Engine.

Real-time inference for Raydium vs Orca arbitrage:
- Uses trained XGBoost model (or ONNX for production)
- Integrates with Solana data collectors
- Provides execution recommendations
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from .onnx_runtime import ONNXInference
from ..models.arbitrage import SolanaArbitrageModel
from ..features import SolanaFeatureEngineer, CrossDexFeatureEngineer
from ..config import SOLANA_CHAIN_PARAMS

logger = structlog.get_logger()

# Default paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "solana"


class SolanaArbitrageInference:
    """
    Real-time inference for Solana cross-DEX arbitrage.
    
    Supports both:
    - XGBoost model (for development/testing)
    - ONNX model (for production deployment)
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        use_onnx: bool = False
    ):
        self.model_path = Path(model_path) if model_path else None
        self.use_onnx = use_onnx
        
        self.model: SolanaArbitrageModel | None = None
        self.onnx_engine: ONNXInference | None = None
        self.metadata: dict[str, Any] = {}
        self.feature_names: list[str] = []
        
        # Feature engineers
        self.base_engineer = SolanaFeatureEngineer()
        self.cross_dex_engineer = CrossDexFeatureEngineer()
        
        # Chain parameters
        self.chain_params = SOLANA_CHAIN_PARAMS
        
        self.logger = logger.bind(component="solana_inference")
    
    def load(self, model_path: str | Path | None = None) -> None:
        """Load model for inference."""
        path = Path(model_path) if model_path else self.model_path
        
        if path is None:
            # Find latest model
            path = self._find_latest_model()
        
        if self.use_onnx:
            self._load_onnx(path)
        else:
            self._load_xgboost(path)
        
        self.logger.info("Model loaded", path=str(path), use_onnx=self.use_onnx)
    
    def _load_xgboost(self, path: Path) -> None:
        """Load XGBoost model."""
        self.model = SolanaArbitrageModel()
        self.model.load(path)
        self.feature_names = self.model.feature_names
        self.metadata = {
            "type": "xgboost",
            "metrics": self.model.metrics,
            "features": self.feature_names
        }
    
    def _load_onnx(self, path: Path) -> None:
        """Load ONNX model."""
        onnx_path = path.with_suffix(".onnx")
        meta_path = path.with_suffix(".meta.json")
        
        self.onnx_engine = ONNXInference(onnx_path)
        self.onnx_engine.load()
        
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
    
    def _find_latest_model(self) -> Path:
        """Find the latest trained model."""
        model_files = list(MODEL_DIR.glob("solana_arbitrage_*.json"))
        if not model_files:
            raise FileNotFoundError(f"No models found in {MODEL_DIR}")
        
        # Sort by timestamp in filename
        model_files.sort(reverse=True)
        return model_files[0].with_suffix("")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict arbitrage profitability.
        
        Args:
            features: DataFrame with market features
            
        Returns:
            Probability array (0-1)
        """
        X = self._prepare_features(features)
        
        if self.use_onnx and self.onnx_engine:
            probs = self.onnx_engine.predict_proba(X)
            return probs[:, 1] if len(probs.shape) > 1 else probs
        elif self.model:
            return self.model.predict(X)
        else:
            raise RuntimeError("No model loaded")
    
    def predict_with_decision(
        self,
        features: pd.DataFrame,
        min_confidence: float = 0.65,
        max_priority_fee: int | None = None
    ) -> dict[str, Any]:
        """
        Predict with execution decision.
        
        Returns comprehensive decision including:
        - Probability
        - Execute recommendation
        - DEX routing
        - Cost estimates
        """
        X = self._prepare_features(features)
        
        if self.model:
            return self.model.should_execute(
                X,
                min_confidence=min_confidence,
                max_priority_fee_lamports=max_priority_fee
            )
        
        # Fallback for ONNX
        prob = self.predict(features)[0]
        return {
            "execute": prob >= min_confidence,
            "probability": float(prob),
            "reason": "High confidence" if prob >= min_confidence else "Low confidence"
        }
    
    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for inference."""
        # If raw data, apply feature engineering
        if 'rsi' not in features.columns:
            features = self.base_engineer.engineer_features(features)
            features = self.cross_dex_engineer.engineer_features(features)

        # Select only required features
        if self.feature_names:
            available = [c for c in self.feature_names if c in features.columns]
            return features[available]

        return features

    async def predict_realtime(
        self,
        birdeye_collector,
        jupiter_collector,
        token_address: str | None = None
    ) -> dict[str, Any]:
        """
        Real-time prediction using live data from collectors.

        Args:
            birdeye_collector: BirdeyeCollector instance
            jupiter_collector: JupiterCollector instance
            token_address: Token to analyze (default: SOL)

        Returns:
            Prediction with current market data
        """
        from ..data.collectors import BirdeyeCollector, JupiterCollector

        token = token_address or self.chain_params['sol_mint']
        usdc = self.chain_params['usdc_mint']

        # Fetch current data
        birdeye_data = await birdeye_collector.fetch_latest()

        # Get Jupiter quote for spread calculation
        amount = 1_000_000_000  # 1 SOL in lamports
        jupiter_quote = await jupiter_collector.get_quote(token, usdc, amount)

        # Get DEX comparison
        dex_comparison = await jupiter_collector.compare_dex_routes(token, usdc, amount)

        # Build feature row
        current_time = datetime.utcnow()
        feature_row = {
            'datetime': current_time,
            'sol_price': birdeye_data.get('prices', {}).get('SOL', {}).get('price_usd', 0),
            'spread_abs': dex_comparison.get('spread_pct', 0),
            'price_impact_pct': float(jupiter_quote.get('price_impact_pct', 0)),
            'priority_fee_lamports': birdeye_data.get('avg_priority_fee_lamports', 50000),
        }

        # Add DEX-specific prices if available
        dex_routes = dex_comparison.get('dex_routes', {})
        if 'raydium' in dex_routes:
            feature_row['raydium_price'] = float(dex_routes['raydium'].get('out_amount', 0))
        if 'orca' in dex_routes:
            feature_row['orca_price'] = float(dex_routes['orca'].get('out_amount', 0))

        # Create DataFrame and predict
        df = pd.DataFrame([feature_row])

        decision = self.predict_with_decision(df)

        return {
            **decision,
            "timestamp": current_time.isoformat(),
            "market_data": feature_row,
            "dex_routes": dex_routes
        }

    def batch_predict(
        self,
        data: pd.DataFrame,
        threshold: float = 0.65
    ) -> pd.DataFrame:
        """
        Batch prediction for backtesting.

        Args:
            data: Historical data
            threshold: Probability threshold

        Returns:
            DataFrame with predictions added
        """
        df = data.copy()

        # Prepare features
        X = self._prepare_features(df)

        # Get predictions
        probs = self.predict(X)

        df['prediction_prob'] = probs
        df['prediction'] = (probs >= threshold).astype(int)

        return df

    def calculate_expected_profit(
        self,
        spread_pct: float,
        trade_size_usd: float = 10000,
        sol_price: float = 200
    ) -> dict[str, float]:
        """
        Calculate expected profit for an arbitrage opportunity.

        Args:
            spread_pct: Current spread percentage
            trade_size_usd: Trade size in USD
            sol_price: Current SOL price

        Returns:
            Profit breakdown
        """
        # DEX fees
        raydium_fee = self.chain_params['raydium_fee_pct']
        orca_fee = self.chain_params['orca_fee_pct']
        total_dex_fee = (raydium_fee + orca_fee) * 100  # Convert to %

        # Transaction fees
        tx_fee_sol = (
            self.chain_params['base_tx_fee_lamports'] +
            self.chain_params['priority_fee_lamports']
        ) / 1e9 * 2  # Two transactions
        tx_fee_usd = tx_fee_sol * sol_price
        tx_fee_pct = (tx_fee_usd / trade_size_usd) * 100

        # Slippage estimate (0.1% for $10k trade)
        slippage_pct = 0.1

        # Calculate profit
        gross_profit_pct = spread_pct
        total_cost_pct = total_dex_fee + tx_fee_pct + slippage_pct
        net_profit_pct = gross_profit_pct - total_cost_pct
        net_profit_usd = (net_profit_pct / 100) * trade_size_usd

        return {
            "gross_profit_pct": round(gross_profit_pct, 4),
            "dex_fees_pct": round(total_dex_fee, 4),
            "tx_fee_pct": round(tx_fee_pct, 4),
            "slippage_pct": round(slippage_pct, 4),
            "total_cost_pct": round(total_cost_pct, 4),
            "net_profit_pct": round(net_profit_pct, 4),
            "net_profit_usd": round(net_profit_usd, 2),
            "profitable": net_profit_pct > 0
        }

    @property
    def model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "type": "onnx" if self.use_onnx else "xgboost",
            "loaded": self.model is not None or self.onnx_engine is not None,
            "n_features": len(self.feature_names),
            "metrics": self.metadata.get("metrics", {}),
            "chain": "solana"
        }

