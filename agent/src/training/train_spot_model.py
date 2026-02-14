"""
Spot Trading XGBoost Model Training
Trains binary classifier for spot trading entry signals
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import json
import os
import shutil
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Calibration import
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "training"))
    from calibration import ModelCalibrator, fit_and_save_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("Calibration module not available")

# MLflow tracking import
try:
    from mlflow_tracking import MLflowTracker, create_tracker
    MLFLOW_TRACKING_AVAILABLE = True
except ImportError:
    MLFLOW_TRACKING_AVAILABLE = False
    print("MLflow tracking not available. Run: pip install mlflow")

# Regime validation import
try:
    training_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "training")
    sys.path.insert(0, training_path)
    from regime_validation import RegimeValidator, generate_regime_metadata
    REGIME_VALIDATION_AVAILABLE = True
except ImportError:
    REGIME_VALIDATION_AVAILABLE = False
    print("Regime validation not available")

# Regime detection import
try:
    analysis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
    sys.path.insert(0, analysis_path)
    from regimeDetector import label_data_with_regimes, RegimeConfig
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False
    print("Regime detection not available")

from features.spot_trading_features import SpotTradingFeatureExtractor
from training.spot_label_generator import SpotLabelGenerator


class SpotModelTrainer:
    """Train XGBoost model for spot trading"""
    
    def __init__(self, output_dir: str = 'agent/eliza/src/models/spot'):
        self.output_dir = output_dir
        self.feature_extractor = SpotTradingFeatureExtractor()
        self.label_generator = SpotLabelGenerator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self, data_path: str):
        """
        Train XGBoost model with MLflow tracking.

        Args:
            data_path: Path to CSV file with historical data
        """
        # Generate version tag
        version_tag = f"v{datetime.now().strftime('%Y%m%d.%H%M%S')}"

        # Initialize MLflow tracker
        mlflow_tracker = None
        mlflow_run_context = None
        if MLFLOW_TRACKING_AVAILABLE:
            try:
                mlflow_tracker = create_tracker("spot")
                print(f"\n📊 MLflow tracking enabled - Version: {version_tag}")
            except Exception as e:
                print(f"⚠️ MLflow initialization failed: {e}")

        print(f"\n{'='*60}")
        print(f"SPOT TRADING MODEL TRAINING")
        print(f"  Version: {version_tag}")
        print(f"{'='*60}\n")
        
        # 1. Load data
        print("[1/6] Loading data...")
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} rows")
        
        # 2. Generate features
        print("\n[2/6] Generating features...")
        features_df = self.feature_extractor.extract_features(df)
        print(f"  Generated {len(features_df.columns)} features")
        
        # 3. Generate labels
        print("\n[3/6] Generating labels...")
        labeled_df = self.label_generator.generate_labels(df)
        
        # Merge features with labels
        features_df = features_df.loc[labeled_df.index]
        features_df['label'] = labeled_df['label']
        
        # Fill NaN and inf values
        print(f"  NaN counts before fill: {features_df.isna().sum().sum()}")
        features_df = features_df.replace([np.inf, -np.inf], 0)
        features_df = features_df.fillna(0)
        print(f"  Final dataset: {len(features_df)} rows")

        # Label data with market regimes
        regimes = None
        if REGIME_DETECTION_AVAILABLE:
            try:
                print("\n  📊 Detecting market regimes...")
                # Detect price column
                price_col = None
                for col in ['close', 'price', 'close_price', 'oracle_twap']:
                    if col in df.columns:
                        price_col = col
                        break

                if price_col:
                    labeled_with_regimes = label_data_with_regimes(df, price_column=price_col)
                    if 'regime' in labeled_with_regimes.columns:
                        # Align with features_df
                        regimes = labeled_with_regimes['regime'].loc[features_df.index].reset_index(drop=True)
                        regime_counts = regimes.value_counts()
                        print(f"    BULL: {regime_counts.get('BULL', 0)}, BEAR: {regime_counts.get('BEAR', 0)}, SIDEWAYS: {regime_counts.get('SIDEWAYS', 0)}")
                else:
                    print("    ⚠️ No price column found for regime detection")
            except Exception as e:
                print(f"    ⚠️ Regime detection failed: {e}")

        # 4. Split data (time-based, not random)
        print("\n[4/6] Splitting data...")
        split_idx = int(len(features_df) * 0.8)
        
        train_df = features_df.iloc[:split_idx]
        test_df = features_df.iloc[split_idx:]
        
        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        
        print(f"  Train: {len(X_train)} rows")
        print(f"  Test: {len(X_test)} rows")
        print(f"  Train BUY%: {(y_train == 1).sum() / len(y_train) * 100:.1f}%")
        print(f"  Test BUY%: {(y_test == 1).sum() / len(y_test) * 100:.1f}%")
        
        # 5. Train XGBoost
        print("\n[5/6] Training XGBoost model...")

        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        # Start MLflow run before training
        if mlflow_tracker:
            try:
                mlflow_run_context = mlflow_tracker.start_run(
                    run_name=version_tag,
                    tags={"model_type": "spot"}
                )
                mlflow_run_context.__enter__()

                # Log hyperparameters
                mlflow_tracker.log_params({
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "n_features": len(X_train.columns),
                    **params,
                })
                print("📊 MLflow: Logged hyperparameters")
            except Exception as e:
                print(f"⚠️ MLflow run start failed: {e}")

        model = xgb.XGBClassifier(**params)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 6. Evaluate
        print("\n[6/6] Evaluating model...")
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n  Accuracy: {accuracy * 100:.2f}%")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['NO_BUY', 'BUY']))
        
        print("\n  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")
        
        # Feature importance
        print("\n  Top 10 Features:")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        for i, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        # Fit probability calibration (Platt Scaling)
        calibration_metrics = {}
        if CALIBRATION_AVAILABLE:
            try:
                print("\n  📊 Fitting Platt Scaling calibration...")

                # Save to local calibration directory
                calibration_dir = os.path.join(self.output_dir, "calibration")
                os.makedirs(calibration_dir, exist_ok=True)

                params = fit_and_save_calibration(
                    predictions=y_pred_proba,
                    true_labels=np.array(y_test),
                    model_name="spot_model",
                    output_dir=calibration_dir,
                    num_bins=10
                )

                # Also copy to main calibration directory for TypeScript service
                main_calibration_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(self.output_dir))),
                    "eliza/models/calibration"
                )
                os.makedirs(main_calibration_dir, exist_ok=True)

                src = os.path.join(calibration_dir, "spot_model_calibration.json")
                dst = os.path.join(main_calibration_dir, "spot_model_calibration.json")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"  Calibration copied to {dst}")

                calibration_metrics = {
                    "ece_before": params.ece_before_calibration,
                    "ece_after": params.ece_after_calibration,
                    "brier_before": params.brier_score_before,
                    "brier_after": params.brier_score_after,
                    "is_well_calibrated": params.ece_after_calibration < 0.10,
                }
            except Exception as e:
                print(f"  ⚠️ Calibration failed: {e}")

        # Regime-specific validation
        regime_metrics = {}
        regime_flags = []
        regime_warnings = []
        if REGIME_VALIDATION_AVAILABLE and regimes is not None:
            try:
                print("\n  📊 Regime-specific validation...")
                # Get test regimes (matching test set indices)
                test_regimes = regimes.iloc[split_idx:].reset_index(drop=True)

                validator = RegimeValidator(
                    performance_drop_threshold=0.20,
                    min_samples_per_regime=50,
                )

                result = validator.validate(
                    model=model,
                    X_test=X_test.values,
                    y_test=y_test.values,
                    regimes=test_regimes,
                    model_name="spot_trading_model",
                    returns=None,
                )

                # Extract results
                for regime, metrics in result.regime_metrics.items():
                    regime_metrics[regime] = metrics.to_dict()
                regime_flags = result.flags
                regime_warnings = result.warnings

            except Exception as e:
                print(f"  ⚠️ Regime validation failed: {e}")

        # 7. Save model
        print(f"\n[7/7] Saving model to {self.output_dir}...")
        
        # Save XGBoost model
        model_path = os.path.join(self.output_dir, 'spot_model.json')
        model.save_model(model_path)
        print(f"  Saved XGBoost model: {model_path}")
        
        # Save feature names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(list(X_train.columns), f, indent=2)
        print(f"  Saved feature names: {feature_names_path}")
        
        # Check for regime issues
        meets_targets = True
        if regime_flags:
            print("\n  🚨 REGIME PERFORMANCE FLAGS:")
            for flag in regime_flags:
                print(f"    - {flag}")
            meets_targets = False
        else:
            print("\n  ✅ No significant regime performance drops (>20%)")

        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'version': version_tag,
            'n_features': len(X_train.columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'params': params,
            'calibration': calibration_metrics,
            'regime_performance': regime_metrics,
            'regime_flags': regime_flags,
            'regime_warnings': regime_warnings,
            'meets_targets': meets_targets,
        }

        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

        # MLflow: Log metrics and artifacts
        if mlflow_tracker and mlflow_run_context:
            try:
                # Log test metrics
                mlflow_tracker.log_metrics({
                    "accuracy": accuracy,
                    "roc_auc": roc_auc,
                })

                # Log calibration metrics
                if calibration_metrics:
                    mlflow_tracker.log_metrics({
                        "ece_before": calibration_metrics.get("ece_before", 0),
                        "ece_after": calibration_metrics.get("ece_after", 0),
                    })

                # Log model artifacts
                if os.path.exists(model_path):
                    model_info = mlflow_tracker.log_model(
                        model_path=model_path,
                        metadata_path=metadata_path,
                        version=version_tag,
                    )
                    print(f"📊 MLflow: Logged model artifacts - {model_info.get('version', 'unknown')}")

                # Register model in MLflow registry
                model_version = mlflow_tracker.register_model("spot-model", stage="Production")
                if model_version:
                    print(f"📊 MLflow: Registered as version {model_version}")

                # End MLflow run
                mlflow_run_context.__exit__(None, None, None)
                print("📊 MLflow: Run completed successfully")
            except Exception as e:
                print(f"⚠️ MLflow artifact logging failed: {e}")
                try:
                    mlflow_run_context.__exit__(None, None, None)
                except:
                    pass

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*60}\n")

        return model, accuracy, roc_auc


if __name__ == '__main__':
    trainer = SpotModelTrainer()
    trainer.train('agent/data/spot/spot_training_data.csv')

