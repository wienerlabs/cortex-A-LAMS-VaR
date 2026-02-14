"""
XGBoost Model for LP Rebalancing Decisions

Predicts whether to STAY in or EXIT a liquidity pool.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Calibration import
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
    from calibration import ModelCalibrator, fit_and_save_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

# Regime validation import
try:
    from regime_validation import RegimeValidator, generate_regime_metadata
    REGIME_VALIDATION_AVAILABLE = True
except ImportError:
    REGIME_VALIDATION_AVAILABLE = False

# Regime detection import
try:
    analysis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "analysis")
    sys.path.insert(0, analysis_path)
    from regimeDetector import label_data_with_regimes
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False


@dataclass
class PredictionResult:
    """Result of a stay/exit prediction."""
    pool_address: str
    pool_name: str
    stay_probability: float
    exit_probability: float
    recommendation: str  # "STAY", "EXIT", or "HOLD"
    confidence: float
    expected_apy_change: float
    
    def to_dict(self) -> Dict:
        return {
            "pool_address": self.pool_address,
            "pool_name": self.pool_name,
            "stay_probability": self.stay_probability,
            "exit_probability": self.exit_probability,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "expected_apy_change": self.expected_apy_change,
        }


class LPRebalancerModel:
    """XGBoost model for LP rebalancing decisions."""
    
    def __init__(self, model_path: Optional[Path] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("xgboost and sklearn required. Install with: pip install xgboost scikit-learn")
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.model_path = model_path
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # Thresholds
        self.exit_threshold = 0.4
        self.stay_threshold = 0.6
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def prepare_labels(self, df: pd.DataFrame, forward_hours: int = 24) -> pd.Series:
        """
        Generate labels based on future APY change.
        
        Label = 1 (STAY) if APY increases in next 24h
        Label = 0 (EXIT) if APY decreases in next 24h
        """
        # Shift APY to get future value
        future_apy = df.groupby("pool_address")["apy_current"].shift(-forward_hours)
        current_apy = df["apy_current"]
        
        # Label: 1 if future > current (stay), 0 if future < current (exit)
        labels = (future_apy > current_apy).astype(int)
        
        return labels
    
    def train(self, features_df: pd.DataFrame, 
              test_size: float = 0.2,
              validation_size: float = 0.1) -> Dict:
        """Train the XGBoost model."""
        
        # Prepare labels
        labels = self.prepare_labels(features_df)
        
        # Remove rows with NaN labels (last 24h of each pool)
        valid_mask = ~labels.isna()
        X = features_df[valid_mask].drop(columns=["timestamp", "pool_address"], errors="ignore")
        y = labels[valid_mask]
        
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Time series - no shuffle
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, shuffle=False
        )
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=10,
            random_state=42,
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        # Fit probability calibration (Platt Scaling)
        if CALIBRATION_AVAILABLE:
            try:
                print("\n📊 Fitting Platt Scaling calibration...")
                y_val_proba = self.model.predict_proba(X_val)[:, 1]

                # Save to local calibration directory
                model_dir = Path(__file__).parent
                calibration_dir = model_dir / "calibration"
                calibration_dir.mkdir(exist_ok=True)

                params = fit_and_save_calibration(
                    predictions=y_val_proba,
                    true_labels=np.array(y_val),
                    model_name="lp_rebalancer",
                    output_dir=str(calibration_dir),
                    num_bins=10
                )

                # Also copy to main calibration directory for TypeScript service
                main_calibration_dir = model_dir.parent.parent / "eliza" / "models" / "calibration"
                main_calibration_dir.mkdir(parents=True, exist_ok=True)

                import shutil
                src = calibration_dir / "lp_rebalancer_calibration.json"
                dst = main_calibration_dir / "lp_rebalancer_calibration.json"
                if src.exists():
                    shutil.copy(src, dst)
                    print(f"  Calibration copied to {dst}")

                # Add calibration metrics
                metrics["calibration"] = {
                    "ece_before": params.ece_before_calibration,
                    "ece_after": params.ece_after_calibration,
                    "brier_before": params.brier_score_before,
                    "brier_after": params.brier_score_after,
                    "is_well_calibrated": params.ece_after_calibration < 0.10,
                }
            except Exception as e:
                print(f"  ⚠️ Calibration failed: {e}")

        # Regime-specific validation
        if REGIME_VALIDATION_AVAILABLE and REGIME_DETECTION_AVAILABLE:
            try:
                print("\n📊 Regime-specific validation...")

                # Check if features_df has a price column for regime detection
                price_col = None
                for col in ['apy_current', 'price', 'close']:
                    if col in features_df.columns:
                        price_col = col
                        break

                if price_col:
                    # Label test data with regimes
                    test_data = features_df[valid_mask].iloc[-len(X_test):]
                    labeled_data = label_data_with_regimes(test_data, price_column=price_col)

                    if 'regime' in labeled_data.columns:
                        test_regimes = labeled_data['regime'].reset_index(drop=True)

                        validator = RegimeValidator(
                            performance_drop_threshold=0.20,
                            min_samples_per_regime=30,  # Lower threshold for LP data
                        )

                        result = validator.validate(
                            model=self.model,
                            X_test=X_test.values,
                            y_test=y_test.values,
                            regimes=test_regimes,
                            model_name="lp_rebalancer",
                            returns=None,
                        )

                        # Add regime metrics to results
                        metrics["regime_performance"] = {
                            regime: m.to_dict()
                            for regime, m in result.regime_metrics.items()
                        }
                        metrics["regime_flags"] = result.flags
                        metrics["regime_warnings"] = result.warnings

                        if result.flags:
                            print("  🚨 REGIME PERFORMANCE FLAGS:")
                            for flag in result.flags:
                                print(f"    - {flag}")
                        else:
                            print("  ✅ No significant regime performance drops (>20%)")
            except Exception as e:
                print(f"  ⚠️ Regime validation failed: {e}")

        return metrics

    def predict(self, features: pd.DataFrame) -> List[PredictionResult]:
        """Predict stay/exit for each pool."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = features.drop(columns=["timestamp", "pool_address", "pool_name"], errors="ignore")
        
        # Get probabilities
        probas = self.model.predict_proba(X)
        
        results = []
        for i, row in features.iterrows():
            stay_prob = probas[i][1]
            exit_prob = probas[i][0]
            
            # Determine recommendation
            if stay_prob > self.stay_threshold:
                recommendation = "STAY"
                confidence = stay_prob
            elif stay_prob < self.exit_threshold:
                recommendation = "EXIT"
                confidence = exit_prob
            else:
                recommendation = "HOLD"
                confidence = 0.5
            
            results.append(PredictionResult(
                pool_address=row.get("pool_address", ""),
                pool_name=row.get("pool_name", ""),
                stay_probability=stay_prob,
                exit_probability=exit_prob,
                recommendation=recommendation,
                confidence=confidence,
                expected_apy_change=0.0,  # Could be enhanced with regression
            ))
        
        return results
    
    def save(self, path: Path):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(str(path))
        
        # Save metadata
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
            }, f)
    
    def load(self, path: Path):
        """Load model from disk."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.feature_importance = meta.get("feature_importance", {})

