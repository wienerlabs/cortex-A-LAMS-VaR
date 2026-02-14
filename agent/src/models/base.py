from __future__ import annotations
"""
Base model class for all XGBoost models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import structlog
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)

from ..config import TRAINING_CONFIG

logger = structlog.get_logger()


class BaseModel(ABC):
    """
    Abstract base class for XGBoost DeFi models.
    
    All strategy models inherit from this class.
    """
    
    def __init__(
        self,
        name: str,
        params: dict[str, Any],
        training_config: dict[str, Any] | None = None
    ):
        self.name = name
        self.params = params
        self.training_config = training_config or TRAINING_CONFIG
        self.model: xgb.XGBClassifier | xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.feature_importance: dict[str, float] = {}
        self.metrics: dict[str, float] = {}
        self.logger = logger.bind(model=name)
    
    @abstractmethod
    def create_model(self) -> xgb.XGBClassifier | xgb.XGBRegressor:
        """Create the XGBoost model with parameters."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data."""
        pass
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None
    ) -> dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Optional validation set (X_val, y_val)
            
        Returns:
            Training metrics
        """
        self.logger.info(
            "Starting training",
            samples=len(X),
            features=X.shape[1]
        )
        
        self.feature_names = list(X.columns)
        self.model = self.create_model()
        
        # Prepare eval set for early stopping
        eval_sets = [(X, y)]
        if eval_set:
            eval_sets.append(eval_set)
        
        # Train with early stopping
        self.model.fit(
            X, y,
            eval_set=eval_sets,
            verbose=self.training_config.get("verbose_eval", 10)
        )
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Calculate metrics
        if eval_set:
            self.metrics = self.evaluate(eval_set[0], eval_set[1])
        else:
            self.metrics = self.evaluate(X, y)
        
        self.logger.info("Training complete", metrics=self.metrics)
        return self.metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int | None = None
    ) -> dict[str, list[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            
        Returns:
            Dict of metrics for each fold
        """
        n_splits = n_splits or self.training_config.get("cv_folds", 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_metrics: dict[str, list[float]] = {}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.train(X_train, y_train, eval_set=(X_val, y_val))
            
            metrics = self.evaluate(X_val, y_val)
            
            for metric, value in metrics.items():
                if metric not in fold_metrics:
                    fold_metrics[metric] = []
                fold_metrics[metric].append(value)
        
        # Log average metrics
        avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
        self.logger.info("Cross-validation complete", avg_metrics=avg_metrics)
        
        return fold_metrics
    
    @abstractmethod
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, float]:
        """Evaluate model on data."""
        pass
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Saves:
        - XGBoost model (.json)
        - Metadata (.meta.json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save")

        # Save XGBoost model using booster
        self.model.get_booster().save_model(str(path.with_suffix(".json")))

        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            """Convert numpy types to Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Save metadata
        metadata = {
            "name": self.name,
            "params": convert_to_python_types(self.params),
            "feature_names": self.feature_names,
            "feature_importance": convert_to_python_types(self.feature_importance),
            "metrics": convert_to_python_types(self.metrics),
        }

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info("Model saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """
        Load model from disk.
        """
        path = Path(path)

        # Load XGBoost model
        model_path = path.with_suffix(".json")
        self.model = self.create_model()
        self.model.load_model(str(model_path))

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            self.feature_names = metadata.get("feature_names", [])
            self.feature_importance = metadata.get("feature_importance", {})
            self.metrics = metadata.get("metrics", {})

        self.logger.info("Model loaded", path=str(path))

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        items = list(self.feature_importance.items())
        return items[:n]
