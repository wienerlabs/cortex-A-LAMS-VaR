"""
Cross-DEX Arbitrage Inference Engine.

Uses ONNX model trained on Uniswap V3 vs V2 data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from .onnx_runtime import ONNXInference

logger = structlog.get_logger()

# Default paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
ONNX_MODEL_PATH = MODEL_DIR / "cross_dex_arbitrage.onnx"
METADATA_PATH = MODEL_DIR / "cross_dex_metadata.json"


class CrossDexInference:
    """
    Cross-DEX Arbitrage inference using ONNX model.
    
    Predicts profitability of V3 vs V2 arbitrage opportunities.
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None
    ):
        self.model_path = Path(model_path) if model_path else ONNX_MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        
        self.onnx_engine: ONNXInference | None = None
        self.metadata: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.logger = logger.bind(component="cross_dex_inference")
    
    def load(self) -> None:
        """Load model and metadata."""
        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("features", [])
            self.logger.info(
                "Metadata loaded",
                version=self.metadata.get("version"),
                features=len(self.feature_names),
                accuracy=self.metadata.get("metrics", {}).get("accuracy")
            )
        else:
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        
        # Load ONNX model
        self.onnx_engine = ONNXInference(self.model_path)
        self.onnx_engine.load()
        
        self.logger.info("Cross-DEX model loaded", path=str(self.model_path))
    
    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict arbitrage profitability.
        
        Args:
            features: DataFrame with feature columns or numpy array
            
        Returns:
            Binary predictions (0=not profitable, 1=profitable)
        """
        if self.onnx_engine is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        X = self._prepare_features(features)
        return self.onnx_engine.predict(X)
    
    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Get probability of profitable arbitrage.
        
        Returns:
            Probabilities array (n_samples, 2) - [not_profitable, profitable]
        """
        if self.onnx_engine is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        X = self._prepare_features(features)
        return self.onnx_engine.predict_proba(X)
    
    def should_execute(
        self,
        features: pd.DataFrame,
        min_confidence: float = 0.7,
        min_spread_pct: float = 0.5
    ) -> dict[str, Any]:
        """
        Determine if arbitrage should be executed.
        
        Args:
            features: Current market features
            min_confidence: Minimum prediction probability
            min_spread_pct: Minimum spread percentage
            
        Returns:
            Decision dict with execute flag and reasoning
        """
        probs = self.predict_proba(features)
        prob_profitable = float(probs[0][1]) if len(probs.shape) > 1 else float(probs[0])
        
        # Get spread if available
        spread = features.get("spread_abs", pd.Series([0])).iloc[0] if isinstance(features, pd.DataFrame) else 0
        
        execute = prob_profitable >= min_confidence and spread >= min_spread_pct
        
        return {
            "execute": execute,
            "probability": prob_profitable,
            "spread_pct": spread,
            "min_confidence": min_confidence,
            "reason": "Profitable opportunity" if execute else "Below threshold"
        }
    
    def _prepare_features(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Prepare features for inference."""
        if isinstance(features, pd.DataFrame):
            # Ensure correct column order
            if self.feature_names:
                missing = set(self.feature_names) - set(features.columns)
                if missing:
                    raise ValueError(f"Missing features: {missing}")
                X = features[self.feature_names].values
            else:
                X = features.values
        else:
            X = features
        
        return X.astype(np.float32)
    
    @property
    def model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "type": self.metadata.get("model_type", "unknown"),
            "version": self.metadata.get("version", "unknown"),
            "training_date": self.metadata.get("training_date"),
            "metrics": self.metadata.get("metrics", {}),
            "n_features": len(self.feature_names),
            "features": self.feature_names
        }

