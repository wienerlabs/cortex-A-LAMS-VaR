"""
Solana Lending Strategy Training Pipeline.

End-to-end training for lending protocol selection model:
1. Load collected lending data
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

from ..base import BaseModel
from .lending_model import LendingModel
from ...features.lending_features import LendingFeatureEngineer
from ...config import TRAINING_CONFIG

# MLflow tracking import
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "training"))
    from mlflow_tracking import MLflowTracker, create_tracker
    MLFLOW_TRACKING_AVAILABLE = True
except ImportError:
    MLFLOW_TRACKING_AVAILABLE = False

logger = structlog.get_logger()


class LendingTrainer:
    """
    Training pipeline for Solana lending strategy model.
    
    Optimized for:
    - Time-series data (no random splits)
    - Conservative lending decisions
    - Multi-protocol comparison
    """
    
    def __init__(
        self,
        model_dir: str | Path = "models/lending",
        training_config: dict[str, Any] | None = None
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_config = training_config or TRAINING_CONFIG
        
        # Feature engineer
        self.feature_engineer = LendingFeatureEngineer()
        
        self.logger = logger.bind(component="lending_trainer")
    
    def load_data(self, data_path: str | Path) -> pd.DataFrame:
        """
        Load collected lending data.
        
        Expected columns:
        - timestamp
        - protocol (marginfi, kamino, solend)
        - asset (SOL, USDC, etc.)
        - supply_apy
        - borrow_apy
        - utilization_rate
        - protocol_tvl_usd
        - total_supply
        - total_borrow
        - available_liquidity
        """
        self.logger.info("Loading data", path=str(data_path))
        
        df = pd.read_csv(data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(
            "Data loaded",
            rows=len(df),
            protocols=df['protocol'].nunique(),
            assets=df['asset'].nunique(),
            date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        create_labels: bool = True
    ) -> pd.DataFrame:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Raw lending data
            create_labels: Whether to create training labels
        
        Returns:
            DataFrame with all features
        """
        self.logger.info("Engineering features", input_rows=len(df))
        
        # Apply lending feature engineering
        df = self.feature_engineer.engineer_features(df)
        
        # Create confidence scores
        df = self.feature_engineer.create_confidence_score(df)
        
        # Create labels if needed
        if create_labels:
            df = self._create_labels(df)
        
        # Drop any remaining NaN
        df = df.dropna()
        
        self.logger.info("Features engineered", output_rows=len(df))
        return df
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary labels for lending decisions.
        
        Label = 1 (LEND) if:
        - Net APY >= 2% (minimum threshold)
        - Utilization < 85% (safety)
        - Protocol TVL >= $50M (liquidity)
        - Asset is Tier 1 or Tier 2
        
        Label = 0 (NO_LEND) otherwise
        """
        df = df.copy()
        
        # Calculate net APY (supply APY - borrow APY if leveraged)
        df['net_apy'] = df['supply_apy']
        
        # Create label based on multiple conditions
        df['label'] = (
            (df['net_apy'] >= 0.02) &  # 2% minimum APY
            (df['utilization_rate'] < 0.85) &  # Safe utilization
            (df['protocol_tvl_usd'] >= 50_000_000) &  # $50M minimum TVL
            (df.get('asset_tier', 3) <= 2)  # Tier 1 or 2 assets only
        ).astype(int)
        
        self.logger.info(
            "Labels created",
            positive_rate=df['label'].mean(),
            total_samples=len(df)
        )

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based train/val/test split.

        Important: No random shuffling for time-series data!

        Args:
            df: Full dataset (must be sorted by timestamp)
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        self.logger.info(
            "Data split",
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_positive_rate=train_df['label'].mean() if 'label' in train_df else None
        )

        return train_df, val_df, test_df

    def prepare_xy(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and labels.

        Excludes non-feature columns like timestamp, protocol, asset, label.
        """
        # Columns to exclude from features
        exclude_cols = [
            'timestamp', 'protocol', 'asset', 'label',
            'market_name', 'confidence_level'  # Categorical columns
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df['label'].copy()

        return X, y

    def train(
        self,
        data_path: str | Path,
        params: dict[str, Any] | None = None,
        save: bool = True
    ) -> tuple[LendingModel, dict[str, Any]]:
        """
        Full training pipeline with MLflow tracking.

        Args:
            data_path: Path to collected lending data
            params: Optional model parameters
            save: Whether to save the model

        Returns:
            (trained_model, results_dict)
        """
        # Generate version tag
        version_tag = f"v{datetime.now().strftime('%Y%m%d.%H%M%S')}"

        # Initialize MLflow tracker
        mlflow_tracker = None
        mlflow_run_context = None
        if MLFLOW_TRACKING_AVAILABLE:
            try:
                mlflow_tracker = create_tracker("lending")
                self.logger.info("MLflow tracking enabled", version=version_tag)
            except Exception as e:
                self.logger.warning("MLflow initialization failed", error=str(e))

        self.logger.info("Starting lending strategy training pipeline", version=version_tag)

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

        # Start MLflow run
        if mlflow_tracker:
            try:
                mlflow_run_context = mlflow_tracker.start_run(
                    run_name=version_tag,
                    tags={"model_type": "lending"}
                )
                mlflow_run_context.__enter__()

                # Log hyperparameters
                mlflow_tracker.log_params({
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "n_features": X_train.shape[1],
                    **(params or {})
                })
            except Exception as e:
                self.logger.warning("MLflow run start failed", error=str(e))

        # 5. Create and train model
        model = LendingModel(params=params)
        train_metrics = model.train(X_train, y_train, eval_set=(X_val, y_val))

        # 6. Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)

        self.logger.info("Training complete", test_metrics=test_metrics)

        # 7. Save model
        model_path = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"lending_model_{timestamp}"
            model.save(model_path)
            self.logger.info("Model saved", path=str(model_path))

        # 8. MLflow: Log metrics and artifacts
        if mlflow_tracker and mlflow_run_context:
            try:
                # Log test metrics
                mlflow_tracker.log_metrics({
                    "precision": test_metrics.get("precision", 0),
                    "recall": test_metrics.get("recall", 0),
                    "f1_score": test_metrics.get("f1", 0),
                    "roc_auc": test_metrics.get("roc_auc", 0),
                    "accuracy": test_metrics.get("accuracy", 0),
                })

                # Log model if saved
                if model_path:
                    mlflow_tracker.log_artifact(str(model_path))

                # Register model
                model_version = mlflow_tracker.register_model("lending-model", stage="Production")
                if model_version:
                    self.logger.info("MLflow: Registered model", version=model_version)

                # End run
                mlflow_run_context.__exit__(None, None, None)
            except Exception as e:
                self.logger.warning("MLflow artifact logging failed", error=str(e))
                try:
                    mlflow_run_context.__exit__(None, None, None)
                except:
                    pass

        # 9. Compile results
        results = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": model.feature_importance,
            "model_path": str(model_path) if save else None,
            "version": version_tag,
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
        model = LendingModel(params=params)
        fold_metrics = model.cross_validate(X, y, n_splits=n_splits)

        # Log summary
        avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in fold_metrics.items()}

        self.logger.info(
            "Cross-validation complete",
            avg_metrics=avg_metrics,
            std_metrics=std_metrics
        )

        return fold_metrics


