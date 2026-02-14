"""
Solana Cross-DEX Arbitrage XGBoost Model.

Optimized for Raydium vs Orca arbitrage on Solana:
- Lower cost thresholds (cheap tx fees)
- Faster execution (400ms slots)
- Different feature importance (no gas, priority fees instead)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import structlog
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ..base import BaseModel
from ...config import ARBITRAGE_PARAMS, SOLANA_CHAIN_PARAMS

logger = structlog.get_logger()


# Solana-optimized parameters
SOLANA_ARBITRAGE_PARAMS = {
    **ARBITRAGE_PARAMS,
    # Slightly different hyperparameters for Solana's faster data
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    # Solana-specific
    'min_child_weight': 3,
    'scale_pos_weight': 2.0,  # Handle class imbalance
}


class SolanaArbitrageModel(BaseModel):
    """
    XGBoost classifier for Solana cross-DEX arbitrage.
    
    Predicts whether a Raydium vs Orca price spread will be profitable after:
    - DEX trading fees (0.25% + 0.30%)
    - Transaction fees (~0.00025 SOL)
    - Slippage
    
    Key differences from Ethereum arbitrage:
    - Much lower cost threshold (0.1% vs 0.5%)
    - More frequent opportunities
    - Priority fee instead of gas price
    """
    
    def __init__(self, params: dict | None = None):
        super().__init__(
            name="solana_arbitrage",
            params=params or SOLANA_ARBITRAGE_PARAMS
        )
        self.chain_params = SOLANA_CHAIN_PARAMS
        
        # Solana-specific thresholds
        self.min_profit_threshold = self.chain_params.get('min_profit_threshold', 0.001)
        self.max_priority_fee = self.chain_params.get('priority_fee_lamports', 50000)
    
    def create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier optimized for Solana arbitrage."""
        return xgb.XGBClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", 8),
            learning_rate=self.params.get("learning_rate", 0.03),
            subsample=self.params.get("subsample", 0.85),
            colsample_bytree=self.params.get("colsample_bytree", 0.85),
            gamma=self.params.get("gamma", 0.05),
            reg_alpha=self.params.get("reg_alpha", 0.05),
            reg_lambda=self.params.get("reg_lambda", 0.5),
            min_child_weight=self.params.get("min_child_weight", 3),
            scale_pos_weight=self.params.get("scale_pos_weight", 2.0),
            objective="binary:logistic",
            eval_metric=["auc", "logloss"],
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=self.training_config.get("early_stopping_rounds", 30),
            use_label_encoder=False,
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict arbitrage opportunity probability.
        
        Returns:
            Array of probabilities (0-1) for profitable arbitrage.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Predict with confidence levels.
        
        Returns DataFrame with:
        - probability: Raw prediction probability
        - prediction: Binary prediction
        - confidence: Confidence level (low/medium/high)
        """
        probs = self.predict(X)
        
        result = pd.DataFrame({
            'probability': probs,
            'prediction': (probs >= threshold).astype(int),
        })
        
        # Confidence levels
        result['confidence'] = pd.cut(
            probs,
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        return result
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, float]:
        """
        Evaluate model with Solana-specific metrics.
        
        Includes profit-weighted metrics since Solana allows
        more frequent but smaller trades.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_prob = self.predict(X)
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Standard metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0,
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
        
        # Profit-relevant metrics
        # False positives are costly (execute unprofitable trade)
        # False negatives are missed opportunities
        metrics["fp_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["fn_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    def should_execute(
        self,
        features: pd.DataFrame,
        min_confidence: float = 0.65,
        max_priority_fee_lamports: int | None = None
    ) -> dict:
        """
        Determine if arbitrage should be executed on Solana.

        Args:
            features: Current market features
            min_confidence: Minimum prediction probability
            max_priority_fee_lamports: Maximum acceptable priority fee

        Returns:
            Dict with decision and reasoning.
        """
        if self.model is None:
            return {"execute": False, "reason": "Model not loaded"}

        max_fee = max_priority_fee_lamports or self.max_priority_fee

        # Get prediction
        prob = self.predict(features)[0]

        # Check priority fee if available
        priority_fee = features.get("priority_fee_lamports", pd.Series([0])).iloc[0]

        # Check spread if available
        net_spread = features.get("net_spread", pd.Series([0])).iloc[0]

        # Decision logic
        if priority_fee > max_fee:
            return {
                "execute": False,
                "reason": f"Priority fee too high: {priority_fee} > {max_fee} lamports",
                "probability": prob,
                "priority_fee": priority_fee,
                "net_spread": net_spread
            }

        if prob < min_confidence:
            return {
                "execute": False,
                "reason": f"Low confidence: {prob:.2f} < {min_confidence}",
                "probability": prob,
                "priority_fee": priority_fee,
                "net_spread": net_spread
            }

        if net_spread < self.min_profit_threshold * 100:
            return {
                "execute": False,
                "reason": f"Spread too low: {net_spread:.3f}% < {self.min_profit_threshold*100}%",
                "probability": prob,
                "priority_fee": priority_fee,
                "net_spread": net_spread
            }

        return {
            "execute": True,
            "reason": "High confidence arbitrage opportunity",
            "probability": prob,
            "priority_fee": priority_fee,
            "net_spread": net_spread,
            "recommended_dex": self._get_recommended_dex(features)
        }

    def _get_recommended_dex(self, features: pd.DataFrame) -> dict:
        """
        Get recommended DEX for buy/sell based on features.

        Returns which DEX to buy from and which to sell to.
        """
        # Check if we have DEX-specific data
        if 'raydium_price' in features.columns and 'orca_price' in features.columns:
            raydium_price = features['raydium_price'].iloc[0]
            orca_price = features['orca_price'].iloc[0]

            if raydium_price < orca_price:
                return {
                    "buy_from": "raydium",
                    "sell_to": "orca",
                    "spread_pct": (orca_price - raydium_price) / raydium_price * 100
                }
            else:
                return {
                    "buy_from": "orca",
                    "sell_to": "raydium",
                    "spread_pct": (raydium_price - orca_price) / orca_price * 100
                }

        # Default based on buy_dex column if available
        if 'buy_dex' in features.columns:
            buy_dex = features['buy_dex'].iloc[0]
            sell_dex = 'orca' if buy_dex == 'raydium' else 'raydium'
            return {"buy_from": buy_dex, "sell_to": sell_dex}

        return {"buy_from": "unknown", "sell_to": "unknown"}

    def get_feature_importance_by_category(self) -> dict[str, list]:
        """
        Get feature importance grouped by category.

        Categories:
        - spread: Spread-related features
        - price: Price and return features
        - volume: Volume features
        - technical: Technical indicators
        - time: Time-based features
        - cost: Cost-related features
        """
        categories = {
            'spread': [],
            'price': [],
            'volume': [],
            'technical': [],
            'time': [],
            'cost': [],
            'other': []
        }

        for feature, importance in self.feature_importance.items():
            if 'spread' in feature.lower():
                categories['spread'].append((feature, importance))
            elif any(x in feature.lower() for x in ['return', 'price', 'momentum']):
                categories['price'].append((feature, importance))
            elif 'volume' in feature.lower():
                categories['volume'].append((feature, importance))
            elif any(x in feature.lower() for x in ['rsi', 'macd', 'bb_', 'sma', 'ema']):
                categories['technical'].append((feature, importance))
            elif any(x in feature.lower() for x in ['hour', 'day', 'session', 'weekend']):
                categories['time'].append((feature, importance))
            elif any(x in feature.lower() for x in ['cost', 'fee', 'profit']):
                categories['cost'].append((feature, importance))
            else:
                categories['other'].append((feature, importance))

        # Sort each category by importance
        for cat in categories:
            categories[cat].sort(key=lambda x: x[1], reverse=True)

        return categories

