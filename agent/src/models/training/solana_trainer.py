"""
Solana Cross-DEX Arbitrage Training Pipeline.

End-to-end training for Raydium vs Orca arbitrage model:
1. Load collected data
2. Feature engineering
3. Train/val/test split (time-based)
4. Model training with early stopping
5. Evaluation and model saving
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
import structlog
from sklearn.model_selection import train_test_split

from ..base import BaseModel
from ..arbitrage import SolanaArbitrageModel
from ...features import SolanaFeatureEngineer, CrossDexFeatureEngineer
from ...config import TRAINING_CONFIG

logger = structlog.get_logger()


class SolanaArbitrageTrainer:
    """
    Training pipeline for Solana cross-DEX arbitrage model.
    
    Optimized for:
    - Time-series data (no random splits)
    - Class imbalance handling
    - Solana-specific features
    """
    
    def __init__(
        self,
        model_dir: str | Path = "models/solana",
        training_config: dict[str, Any] | None = None
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_config = training_config or TRAINING_CONFIG
        
        # Feature engineers
        self.base_feature_engineer = SolanaFeatureEngineer()
        self.cross_dex_engineer = CrossDexFeatureEngineer()
        
        self.logger = logger.bind(component="solana_trainer")
    
    def load_data(self, data_path: str | Path) -> pd.DataFrame:
        """
        Load collected cross-DEX data.
        
        Supports parquet and CSV formats.
        """
        data_path = Path(data_path)
        
        if data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path, parse_dates=['datetime'])
        else:
            raise ValueError(f"Unsupported format: {data_path.suffix}")
        
        self.logger.info("Data loaded", rows=len(df), columns=list(df.columns))
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        create_labels: bool = True
    ) -> pd.DataFrame:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Raw data
            create_labels: Whether to create training labels
        
        Returns:
            DataFrame with all features
        """
        self.logger.info("Engineering features", input_rows=len(df))
        
        # Apply base Solana features
        df = self.base_feature_engineer.engineer_features(df)
        
        # Apply cross-DEX specific features
        df = self.cross_dex_engineer.engineer_features(df)
        
        # Create labels if needed
        if create_labels:
            df = self.cross_dex_engineer.create_labels(df, lookahead=1)
        
        # Drop any remaining NaN
        df = df.dropna()
        
        self.logger.info("Features engineered", output_rows=len(df))
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (no shuffling for time series).
        
        Args:
            df: Full dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
        
        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        self.logger.info(
            "Data split",
            train=len(train_df),
            val=len(val_df),
            test=len(test_df)
        )
        
        return train_df, val_df, test_df
    
    def get_feature_columns(self) -> list[str]:
        """Get list of feature columns for training."""
        base_features = self.base_feature_engineer.get_feature_names()
        cross_dex_features = self.cross_dex_engineer.get_feature_names()
        
        # Combine and deduplicate
        all_features = list(set(base_features + cross_dex_features))
        return all_features
    
    def prepare_xy(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        label_col: str = 'label'
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X (features) and y (labels) for training.
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns()

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols]
        y = df[label_col]

        return X, y

    def train(
        self,
        data_path: str | Path,
        params: dict[str, Any] | None = None,
        save: bool = True
    ) -> tuple[SolanaArbitrageModel, dict[str, Any]]:
        """
        Full training pipeline.

        Args:
            data_path: Path to collected data
            params: Optional model parameters
            save: Whether to save the model

        Returns:
            (trained_model, results_dict)
        """
        self.logger.info("Starting Solana arbitrage training pipeline")

        # 1. Load data
        df = self.load_data(data_path)

        # 2. Feature engineering
        df = self.prepare_features(df, create_labels=True)

        # 3. Split data
        train_df, val_df, test_df = self.split_data(df)

        # 4. Prepare X, y
        X_train, y_train = self.prepare_xy(train_df)
        X_val, y_val = self.prepare_xy(val_df)
        X_test, y_test = self.prepare_xy(test_df)

        self.logger.info(
            "Training data prepared",
            features=X_train.shape[1],
            train_samples=len(X_train),
            positive_rate=y_train.mean()
        )

        # 5. Create and train model
        model = SolanaArbitrageModel(params=params)
        train_metrics = model.train(X_train, y_train, eval_set=(X_val, y_val))

        # 6. Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)

        self.logger.info("Training complete", test_metrics=test_metrics)

        # 7. Save model
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"solana_arbitrage_{timestamp}"
            model.save(model_path)
            self.logger.info("Model saved", path=str(model_path))

        # 8. Compile results
        results = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": model.get_top_features(20),
            "data_stats": {
                "total_samples": len(df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "positive_rate": float(y_train.mean()),
                "features_used": list(X_train.columns)
            }
        }

        return model, results

    def cross_validate(
        self,
        data_path: str | Path,
        n_splits: int = 5,
        params: dict[str, Any] | None = None
    ) -> dict[str, list[float]]:
        """
        Time-series cross-validation.

        Uses expanding window approach for time series.
        """
        self.logger.info("Starting cross-validation", n_splits=n_splits)

        # Load and prepare data
        df = self.load_data(data_path)
        df = self.prepare_features(df, create_labels=True)
        X, y = self.prepare_xy(df)

        # Create model and cross-validate
        model = SolanaArbitrageModel(params=params)
        fold_metrics = model.cross_validate(X, y, n_splits=n_splits)

        # Log summary
        avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in fold_metrics.items()}

        self.logger.info(
            "Cross-validation complete",
            avg_auc=avg_metrics.get('auc', 0),
            std_auc=std_metrics.get('auc', 0)
        )

        return fold_metrics

    def evaluate_model(
        self,
        model: SolanaArbitrageModel,
        test_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Comprehensive model evaluation.

        Returns metrics and analysis for production readiness.
        """
        # Prepare test data
        test_df = self.prepare_features(test_data, create_labels=True)
        X_test, y_test = self.prepare_xy(test_df)

        # Get predictions
        predictions = model.predict_with_confidence(X_test)

        # Standard metrics
        metrics = model.evaluate(X_test, y_test)

        # Additional analysis
        analysis = {
            "metrics": metrics,
            "prediction_distribution": {
                "mean_probability": float(predictions['probability'].mean()),
                "std_probability": float(predictions['probability'].std()),
                "high_confidence_pct": float((predictions['probability'] > 0.7).mean()),
            },
            "feature_importance": model.get_feature_importance_by_category(),
            "production_ready": self._check_production_ready(metrics)
        }

        return analysis

    def _check_production_ready(self, metrics: dict[str, float]) -> dict[str, Any]:
        """
        Check if model meets production thresholds.
        """
        thresholds = {
            "auc": 0.65,
            "precision": 0.60,
            "recall": 0.50,
            "f1": 0.55
        }

        checks = {}
        for metric, threshold in thresholds.items():
            value = metrics.get(metric, 0)
            checks[metric] = {
                "value": value,
                "threshold": threshold,
                "passed": value >= threshold
            }

        all_passed = all(c["passed"] for c in checks.values())

        return {
            "ready": all_passed,
            "checks": checks
        }

