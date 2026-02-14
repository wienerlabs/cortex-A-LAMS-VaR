#!/usr/bin/env python3
"""
Cross-DEX Arbitrage Model Training

Training pipeline for arbitrage profitability prediction:
- XGBoost classifier
- TimeSeriesSplit validation
- Platt Scaling calibration
- ONNX export

Target metrics:
- Precision > 85% (avoid false positives = losing trades)
- Recall > 80% (catch profitable opportunities)
- ROC-AUC > 0.90

Usage:
    python train_arbitrage_model.py --data ./data/cross_dex_data.csv --output ./models
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

try:
    import onnx
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. Install with: pip install onnx onnxmltools")


# Feature columns expected by the model (27 features)
FEATURE_COLS = [
    'spread_ma_12', 'spread_std_12', 'spread_ma_24', 'spread_std_24',
    'spread_ma_48', 'spread_std_48', 'spread_change', 'spread_pct_change',
    'total_volume', 'volume_ma_12', 'volume_ma_24', 'volume_ma_48', 'volume_ratio',
    'v3_volume', 'v2_volume', 'v3_price', 'v2_price',
    'price_ma_12', 'price_volatility',
    'gas_gwei', 'gas_cost_usd', 'slippage', 'dex_fees_pct', 'gas_cost_pct',
    'hour', 'day_of_week', 'is_weekend'
]

TARGET_COL = 'profitable'


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess training data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Profitable: {df[TARGET_COL].sum():,} ({df[TARGET_COL].mean()*100:.1f}%)")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and target."""
    # Handle missing columns with defaults
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            print(f"  Warning: Missing column {col}, filled with 0")
    
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    print("Training XGBoost model...")
    
    # XGBoost parameters optimized for arbitrage
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'auc',
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    print(f"  Best iteration: {model.best_iteration}")
    
    return model


def calibrate_model(model: xgb.XGBClassifier, X_cal: np.ndarray, 
                    y_cal: np.ndarray) -> LogisticRegression:
    """Platt Scaling calibration."""
    print("Calibrating model (Platt Scaling)...")
    
    proba = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(proba, y_cal)
    
    return calibrator


def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray,
                   y_test: np.ndarray) -> Dict:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
    }

    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['auc']:.4f}")

    return metrics


def export_onnx(model: xgb.XGBClassifier, output_path: str):
    """Export model to ONNX format."""
    if not ONNX_AVAILABLE:
        print("ONNX export skipped (onnxmltools not available)")
        return

    print(f"Exporting to ONNX: {output_path}")

    # Get booster and save feature names
    booster = model.get_booster()
    original_names = booster.feature_names
    booster.feature_names = [f'f{i}' for i in range(len(FEATURE_COLS))]

    # Convert to ONNX
    initial_type = [('input', FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_type)

    # Restore feature names
    booster.feature_names = original_names

    # Save model
    onnx.save_model(onnx_model, output_path)
    print(f"  ONNX model saved: {output_path}")


def save_metadata(metrics: Dict, output_dir: str, best_iter: int):
    """Save model metadata."""
    metadata = {
        'model_type': 'cross_dex_arbitrage_v3_vs_v2',
        'version': '2.0.0',
        'training_date': datetime.now().isoformat(),
        'features': FEATURE_COLS,
        'metrics': metrics,
        'best_iteration': best_iter,
    }

    metadata_path = os.path.join(output_dir, 'arbitrage_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata_path}")
    return metadata_path


def save_calibration(calibrator: LogisticRegression, output_dir: str):
    """Save calibration parameters."""
    calibration = {
        'type': 'platt_scaling',
        'slope': float(calibrator.coef_[0][0]),
        'intercept': float(calibrator.intercept_[0]),
    }

    cal_path = os.path.join(output_dir, 'arbitrage_calibration.json')
    with open(cal_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibration saved: {cal_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Arbitrage ML Model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='./models', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    df = load_data(args.data)
    X, y = prepare_features(df)

    # Time-series split
    n_samples = len(X)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Calibrate
    calibrator = calibrate_model(model, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Export
    onnx_path = os.path.join(args.output, 'cross_dex_arbitrage.onnx')
    export_onnx(model, onnx_path)

    # Save metadata
    metadata_path = save_metadata(metrics, args.output, model.best_iteration)

    # Save calibration
    save_calibration(calibrator, args.output)

    # Print final JSON for auto-retraining executor
    print(json.dumps({
        'success': True,
        'model_path': onnx_path,
        'metadata_path': metadata_path,
        'metrics': metrics
    }))


if __name__ == '__main__':
    main()

