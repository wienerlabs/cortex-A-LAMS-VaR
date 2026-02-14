from __future__ import annotations
"""
SHAP Explainer for XGBoost Models.

Provides interpretable explanations for model predictions.
"""
from typing import Any
import numpy as np
import pandas as pd
import shap
import structlog

logger = structlog.get_logger()


class SHAPExplainer:
    """
    SHAP-based explainer for XGBoost models.
    
    Provides:
    - Feature importance explanations
    - Individual prediction explanations
    - Visualization data for UI
    """
    
    def __init__(self, model: Any):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained XGBoost model
        """
        self.model = model
        self.explainer: shap.TreeExplainer | None = None
        self.feature_names: list[str] = []
        self.logger = logger.bind(component="shap_explainer")
    
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the SHAP explainer.
        
        Args:
            X: Training data for background distribution
        """
        self.feature_names = list(X.columns)
        
        # Use TreeExplainer for XGBoost (fast)
        self.explainer = shap.TreeExplainer(self.model)
        
        self.logger.info(
            "SHAP explainer fitted",
            features=len(self.feature_names)
        )
    
    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Explain predictions for samples.
        
        Args:
            X: Samples to explain
            top_k: Number of top features to include
            
        Returns:
            List of explanation dicts for each sample
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class (returns list of arrays)
        if isinstance(shap_values, list):
            # Use positive class for binary classification
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        explanations = []
        
        for i in range(len(X)):
            sample_shap = shap_values[i]
            sample_features = X.iloc[i]
            
            # Get top contributing features
            feature_contributions = []
            sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
            
            for idx in sorted_indices[:top_k]:
                feature_name = self.feature_names[idx]
                contribution = sample_shap[idx]
                value = sample_features.iloc[idx]
                
                feature_contributions.append({
                    "feature": feature_name,
                    "value": float(value),
                    "contribution": float(contribution),
                    "direction": "positive" if contribution > 0 else "negative"
                })
            
            # Base value (expected value)
            base_value = float(self.explainer.expected_value)
            if isinstance(self.explainer.expected_value, np.ndarray):
                base_value = float(self.explainer.expected_value[1])  # Positive class
            
            explanations.append({
                "base_value": base_value,
                "prediction_contribution": float(np.sum(sample_shap)),
                "top_features": feature_contributions,
                "all_shap_values": {
                    name: float(val)
                    for name, val in zip(self.feature_names, sample_shap)
                }
            })
        
        return explanations
    
    def get_global_importance(self) -> dict[str, float]:
        """
        Get global feature importance from SHAP values.
        
        Returns:
            Dict of feature -> mean absolute SHAP value
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not fitted")
        
        # This would require stored SHAP values from training data
        # For now, return model's feature importance
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        
        return {}
    
    def generate_reasoning(
        self,
        explanation: dict[str, Any],
        prediction: float,
        strategy: str
    ) -> str:
        """
        Generate human-readable reasoning from SHAP explanation.
        
        Args:
            explanation: SHAP explanation dict
            prediction: Model prediction
            strategy: Strategy type
            
        Returns:
            Human-readable reasoning string
        """
        top_features = explanation.get("top_features", [])
        
        if not top_features:
            return "No explanation available."
        
        # Build reasoning
        lines = []
        
        if strategy == "arbitrage":
            lines.append(f"Arbitrage opportunity detected with {prediction:.1%} confidence.")
        elif strategy == "lending":
            lines.append(f"Lending recommendation with {prediction:.1%} confidence.")
        else:
            lines.append(f"Prediction: {prediction:.4f}")
        
        lines.append("\nKey factors:")
        
        for feat in top_features[:3]:
            direction = "increases" if feat["direction"] == "positive" else "decreases"
            lines.append(
                f"  • {feat['feature']} = {feat['value']:.4f} "
                f"({direction} prediction by {abs(feat['contribution']):.4f})"
            )
        
        return "\n".join(lines)
