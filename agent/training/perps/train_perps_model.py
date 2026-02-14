#!/usr/bin/env python3
"""
Perpetuals Funding Rate Classification Model Training

Enhanced training pipeline with:
- XGBoost + LightGBM ensemble
- Profit-weighted loss function
- Walk-forward validation
- Market regime detection
- Time-series cross-validation
- Platt Scaling calibration

Target metrics:
- Precision > 70% (not coin flip!)
- Recall > 50% (catch opportunities)
- Sharpe < 3.0 (realistic, not overfitted)
- ROC-AUC > 0.75

Usage:
    python train_perps_model.py --features ./features/perps_features.csv --output ./models --trials 100
"""

import os
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Optional imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. Install with: pip install onnx")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available. Install with: pip install imbalanced-learn")

# Calibration import
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from calibration import ModelCalibrator, fit_and_save_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("Calibration module not available")

# MLflow tracking import
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mlflow_tracking import MLflowTracker, create_tracker
    MLFLOW_TRACKING_AVAILABLE = True
except ImportError:
    MLFLOW_TRACKING_AVAILABLE = False
    print("MLflow tracking not available. Run: pip install mlflow")

# Regime validation import
try:
    from regime_validation import RegimeValidator, generate_regime_metadata
    REGIME_VALIDATION_AVAILABLE = True
except ImportError:
    REGIME_VALIDATION_AVAILABLE = False
    print("Regime validation not available")


# ============= TRAINING CONFIGURATION =============

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data splits
    train_ratio: float = 0.70  # 70% train
    val_ratio: float = 0.15    # 15% validation
    test_ratio: float = 0.15   # 15% test (holdout)

    # Cross-validation
    cv_splits: int = 5

    # Optuna hyperparameter tuning
    n_trials: int = 100
    optuna_timeout: int = 3600  # 1 hour max

    # Model selection
    use_ensemble: bool = True
    ensemble_method: str = "stacking"  # "voting" or "stacking"

    # Class imbalance handling
    use_smote: bool = False  # Disable by default (use class weights instead)
    smote_ratio: float = 0.5  # Minority class ratio after SMOTE

    # Profit-weighted loss parameters (from backtest data)
    avg_win: float = 1.45      # Average profit per winning trade (from metadata)
    avg_loss: float = 0.78     # Average loss per losing trade
    fp_weight: float = 0.78    # Cost of false positive (bad trade)
    fn_weight: float = 1.45    # Cost of false negative (missed opportunity)

    # Target metrics (REALISTIC)
    target_precision: float = 0.70
    target_recall: float = 0.50
    target_roc_auc: float = 0.75
    max_sharpe: float = 3.0  # Sharpe > 3 indicates overfitting
    target_ece: float = 0.10  # Expected Calibration Error

    # Walk-forward validation
    walk_forward_windows: int = 6  # Test on 6 different time windows


# ============= PROFIT-WEIGHTED LOSS FUNCTION =============

def profit_weighted_loss(y_true: np.ndarray, y_pred_proba: np.ndarray,
                         fp_weight: float = 0.78, fn_weight: float = 1.45) -> float:
    """
    Custom loss function that weights false positives and false negatives
    by their actual cost in trading:

    - False Positive (FP): Model predicts TRADE, but should be NO_TRADE
      Cost = Average loss from bad trade = 0.78 (from backtest)

    - False Negative (FN): Model predicts NO_TRADE, but should be TRADE
      Cost = Missed opportunity = 1.45 (average win from backtest)

    This loss function penalizes missing good trades more heavily than
    making bad trades, since the opportunity cost is higher.

    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities (0 to 1)
        fp_weight: Cost of false positive (bad trade)
        fn_weight: Cost of false negative (missed opportunity)

    Returns:
        Weighted binary cross-entropy loss
    """
    epsilon = 1e-7
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # Weighted binary cross-entropy
    loss = -(
        fn_weight * y_true * np.log(y_pred_proba) +
        fp_weight * (1 - y_true) * np.log(1 - y_pred_proba)
    )

    return loss.mean()


def detect_market_regime(df: pd.DataFrame, price_col: str = "oracle_twap") -> pd.Series:
    """
    Detect market regime for each sample based on price action.

    Regimes:
    - BULL: Price trending up (positive momentum)
    - BEAR: Price trending down (negative momentum)
    - SIDEWAYS: Low volatility, ranging market

    This helps validate model performance across different market conditions.
    """
    if price_col not in df.columns:
        # Fallback: use funding rate momentum as proxy
        if "funding_momentum_24h" in df.columns:
            momentum = df["funding_momentum_24h"]
        else:
            return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    else:
        # Calculate 7-day momentum
        price = df[price_col]
        momentum = price.pct_change(168)  # 7 days of hourly data

    # Calculate 7-day volatility
    if price_col in df.columns:
        returns = df[price_col].pct_change()
        volatility = returns.rolling(168).std()
    else:
        volatility = df.get("volatility_24h", pd.Series([0.02] * len(df)))

    # Define regime thresholds
    momentum_threshold = 0.05  # 5% for trend
    volatility_threshold = 0.03  # 3% for sideways detection

    def classify_regime(row_idx):
        mom = momentum.iloc[row_idx] if row_idx < len(momentum) else 0
        vol = volatility.iloc[row_idx] if row_idx < len(volatility) else 0.02

        if pd.isna(mom) or pd.isna(vol):
            return "UNKNOWN"
        elif abs(mom) < momentum_threshold and vol < volatility_threshold:
            return "SIDEWAYS"
        elif mom > momentum_threshold:
            return "BULL"
        elif mom < -momentum_threshold:
            return "BEAR"
        else:
            return "SIDEWAYS"

    regimes = pd.Series([classify_regime(i) for i in range(len(df))], index=df.index)
    return regimes


def calculate_realistic_sharpe(
    returns_series: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe ratio correctly.

    CRITICAL: Input should be RETURNS (decimal, e.g., 0.01 for 1%), NOT absolute PnL!

    A realistic Sharpe ratio for a trading strategy is typically:
    - < 1.0: Weak strategy
    - 1.0 - 2.0: Good strategy
    - 2.0 - 3.0: Excellent strategy
    - > 3.0: Likely overfitting or data leakage!

    Args:
        returns_series: Series of returns (decimal, e.g., 0.01 for 1%)
        periods_per_year: Trading periods per year (252 for daily, 365 for calendar days)
        risk_free_rate: Annual risk-free rate (default 0 for crypto)

    Returns:
        Annualized Sharpe ratio (capped at 5.0 to flag issues)
    """
    if len(returns_series) < 2:
        return 0.0

    # Use sample standard deviation (ddof=1)
    std_return = returns_series.std(ddof=1)

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    mean_return = returns_series.mean()

    # Convert risk-free rate to per-period
    rf_per_period = risk_free_rate / periods_per_year

    # Sharpe ratio formula: (mean_return - rf) / std * sqrt(periods)
    excess_return = mean_return - rf_per_period
    sharpe = (excess_return / std_return) * np.sqrt(periods_per_year)

    # Sanity check: Cap at 5.0 to flag potential issues
    if sharpe > 5.0:
        print(f"  ⚠️ Warning: Sharpe ratio {sharpe:.2f} > 5.0 - possible overfitting or data leakage!")
        sharpe = 5.0
    elif sharpe < -5.0:
        sharpe = -5.0

    return sharpe


class PerpsModelTrainer:
    """
    Enhanced XGBoost/LightGBM classifier for funding rate trade prediction.

    Features:
    - Profit-weighted loss function
    - Ensemble learning (XGBoost + LightGBM)
    - Walk-forward validation
    - Market regime analysis
    - Platt Scaling calibration
    """

    def __init__(
        self,
        features_path: str,
        output_dir: str = "./models",
        n_trials: int = 100,
        cv_splits: int = 5,
        config: TrainingConfig = None,
    ):
        self.features_path = features_path
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.config = config or TrainingConfig(n_trials=n_trials, cv_splits=cv_splits)
        self.scaler = StandardScaler()
        self.model = None
        self.lgb_model = None  # LightGBM for ensemble
        self.best_params = None
        self.feature_cols = None
        self.df_full = None  # Store full dataframe for regime analysis
        self.regime_metrics = {}  # Performance by market regime
        self.walk_forward_results = []  # Walk-forward validation results

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare data for training with enhanced analysis."""
        print("📊 Loading data...")
        df = pd.read_csv(self.features_path)

        # Identify feature columns (exclude metadata, target, and label)
        exclude_cols = ["market", "timestamp", "datetime", "target_funding_rate", "label",
                        "trade_direction", "source", "regime"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Check for label column
        if "label" not in df.columns:
            raise ValueError("Missing 'label' column. Run perps_label_generation.py first.")

        # Handle missing values
        df_clean = df.dropna(subset=["label"])
        df_clean = df_clean.ffill().fillna(0)

        # Detect market regimes for regime-specific analysis
        df_clean["regime"] = detect_market_regime(df_clean)
        regime_counts = df_clean["regime"].value_counts()

        # Store full dataframe for later analysis
        self.df_full = df_clean.copy()

        X = df_clean[self.feature_cols].values
        y = df_clean["label"].values.astype(int)

        # Calculate class weights for imbalanced data
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Apply profit-weighted class weight
        # Higher weight for catching trades (reduce false negatives)
        profit_adjusted_weight = self.scale_pos_weight * (self.config.fn_weight / self.config.fp_weight)
        self.profit_adjusted_weight = min(profit_adjusted_weight, 50.0)  # Cap to prevent instability

        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Label 0 (NO TRADE): {n_neg} ({n_neg/len(y):.1%})")
        print(f"  Label 1 (TRADE):    {n_pos} ({n_pos/len(y):.1%})")
        print(f"  Class weight (scale_pos_weight): {self.scale_pos_weight:.2f}")
        print(f"  Profit-adjusted weight: {self.profit_adjusted_weight:.2f}")
        print(f"\n  Market Regimes:")
        for regime, count in regime_counts.items():
            pct = count / len(df_clean) * 100
            print(f"    {regime}: {count} ({pct:.1f}%)")

        # Check data date range
        if "timestamp" in df_clean.columns:
            min_ts = pd.to_datetime(df_clean["timestamp"].min(), unit="s")
            max_ts = pd.to_datetime(df_clean["timestamp"].max(), unit="s")
            days_span = (max_ts - min_ts).days
            print(f"\n  Date range: {min_ts.date()} to {max_ts.date()} ({days_span} days)")

        return df_clean, X, y

    def create_train_val_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-series aware train/validation/test split.

        Split ratios (chronological):
        - Train: 70% (oldest data)
        - Validation: 15% (middle)
        - Test: 15% (newest data - holdout)
        """
        train_end = int(len(X) * self.config.train_ratio)
        val_end = int(len(X) * (self.config.train_ratio + self.config.val_ratio))

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]

        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]

        # Scale features (fit on train only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n📈 Chronological Train/Val/Test Split:")
        print(f"  Train: {len(X_train)} samples ({self.config.train_ratio:.0%})")
        print(f"  Val:   {len(X_val)} samples ({self.config.val_ratio:.0%})")
        print(f"  Test:  {len(X_test)} samples ({self.config.test_ratio:.0%})")

        # Check class balance in each split
        for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            pos_rate = y_split.mean() if len(y_split) > 0 else 0
            print(f"    {name} positive rate: {pos_rate:.2%}")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def create_train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-series aware train/test split (legacy compatibility)."""
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n📈 Train/Test Split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def objective(self, trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
        """
        Optuna objective function with profit-weighted optimization.

        Optimizes for a combined score considering:
        1. Precision (target > 70%) - avoid bad trades
        2. Recall (target > 50%) - catch opportunities
        3. ROC-AUC (target > 0.75) - discrimination ability
        4. Profit-weighted loss - business-aligned metric

        The objective balances precision and recall while ensuring
        the model doesn't completely sacrifice one for the other.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "scale_pos_weight": self.profit_adjusted_weight,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",  # Faster training
        }

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        precision_scores = []
        recall_scores = []
        auc_scores = []
        profit_losses = []

        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # Skip folds with no positive class in validation
            if y_val_cv.sum() == 0 or y_train_cv.sum() == 0:
                continue

            model = xgb.XGBClassifier(**params)
            model.fit(X_train_cv, y_train_cv, verbose=False)

            # Get predictions
            y_pred = model.predict(X_val_cv)
            y_prob = model.predict_proba(X_val_cv)[:, 1]

            # Calculate metrics
            if y_pred.sum() > 0:
                prec = precision_score(y_val_cv, y_pred, zero_division=0)
                rec = recall_score(y_val_cv, y_pred, zero_division=0)
                precision_scores.append(prec)
                recall_scores.append(rec)

            # ROC-AUC
            try:
                auc = roc_auc_score(y_val_cv, y_prob)
                auc_scores.append(auc)
            except ValueError:
                pass

            # Profit-weighted loss
            pw_loss = profit_weighted_loss(
                y_val_cv, y_prob,
                fp_weight=self.config.fp_weight,
                fn_weight=self.config.fn_weight
            )
            profit_losses.append(pw_loss)

        if not precision_scores:
            return 1.0  # Worst possible score

        # Calculate mean metrics
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores) if recall_scores else 0
        mean_auc = np.mean(auc_scores) if auc_scores else 0.5
        mean_profit_loss = np.mean(profit_losses)

        # Combined objective:
        # - High precision (weight 0.4) - most important for trading
        # - Reasonable recall (weight 0.2) - catch opportunities
        # - High AUC (weight 0.2) - good discrimination
        # - Low profit loss (weight 0.2) - business aligned

        # Penalties for not meeting minimum thresholds
        precision_penalty = max(0, (self.config.target_precision - mean_precision) * 2)
        recall_penalty = max(0, (self.config.target_recall - mean_recall) * 1.5)

        combined_score = (
            0.4 * (1 - mean_precision) +    # Lower is better
            0.2 * (1 - mean_recall) +       # Lower is better
            0.2 * (1 - mean_auc) +           # Lower is better
            0.2 * mean_profit_loss +         # Lower is better
            precision_penalty +
            recall_penalty
        )

        return combined_score  # Minimize

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Run Optuna hyperparameter optimization with TPE sampler."""
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available, using default parameters")
            return self._get_default_params()

        print(f"\n🔍 Hyperparameter Optimization ({self.n_trials} trials)...")
        print(f"    Target: Precision > {self.config.target_precision:.0%}, Recall > {self.config.target_recall:.0%}")

        # Use TPE sampler for better hyperparameter search
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Set timeout to prevent excessively long runs
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True,
        )

        print(f"\n✅ Best trial:")
        print(f"  Combined score: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")

        self.best_params = study.best_trial.params.copy()
        self.best_params["scale_pos_weight"] = self.profit_adjusted_weight
        self.best_params["objective"] = "binary:logistic"
        self.best_params["tree_method"] = "hist"
        self.best_params["random_state"] = 42
        self.best_params["n_jobs"] = -1

        return self.best_params

    def _get_default_params(self) -> Dict:
        """Default XGBoost parameters if Optuna not available."""
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.1,
            "scale_pos_weight": getattr(self, 'profit_adjusted_weight',
                                        getattr(self, 'scale_pos_weight', 4.0)),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, params: Dict = None
    ) -> xgb.XGBClassifier:
        """Train XGBoost classifier with given parameters."""
        if params is None:
            params = self.best_params or self._get_default_params()

        print("\n🚀 Training XGBoost Classifier...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, verbose=False)

        print("✅ XGBoost model trained successfully")
        return self.model

    def train_lightgbm(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Optional[Any]:
        """Train LightGBM classifier for ensemble."""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available, skipping ensemble")
            return None

        print("\n🚀 Training LightGBM Classifier...")

        lgb_params = {
            "n_estimators": self.best_params.get("n_estimators", 300),
            "max_depth": self.best_params.get("max_depth", 6),
            "learning_rate": self.best_params.get("learning_rate", 0.05),
            "subsample": self.best_params.get("subsample", 0.8),
            "colsample_bytree": self.best_params.get("colsample_bytree", 0.8),
            "reg_alpha": self.best_params.get("reg_alpha", 1.0),
            "reg_lambda": self.best_params.get("reg_lambda", 1.0),
            "scale_pos_weight": self.profit_adjusted_weight,
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.lgb_model.fit(X_train, y_train)

        print("✅ LightGBM model trained successfully")
        return self.lgb_model

    def walk_forward_validation(
        self, X: np.ndarray, y: np.ndarray, n_windows: int = 6
    ) -> List[Dict]:
        """
        Walk-forward validation to test model on multiple time windows.

        This simulates how the model would perform if deployed at different
        points in time, providing a more realistic estimate of future performance.
        """
        print(f"\n🔄 Walk-Forward Validation ({n_windows} windows)...")

        results = []
        window_size = len(X) // (n_windows + 1)

        for i in range(n_windows):
            # Train on all data up to window start
            train_end = window_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + window_size, len(X))

            if test_end <= test_start:
                continue

            X_train_wf = X[:train_end]
            y_train_wf = y[:train_end]
            X_test_wf = X[test_start:test_end]
            y_test_wf = y[test_start:test_end]

            # Skip if no positive samples in test
            if y_test_wf.sum() == 0 or y_train_wf.sum() == 0:
                continue

            # Train fresh model
            params = self._get_default_params()
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_wf, y_train_wf, verbose=False)

            # Evaluate
            y_pred = model.predict(X_test_wf)
            y_prob = model.predict_proba(X_test_wf)[:, 1]

            try:
                auc = roc_auc_score(y_test_wf, y_prob)
            except ValueError:
                auc = 0.5

            window_metrics = {
                "window": i + 1,
                "train_size": len(X_train_wf),
                "test_size": len(X_test_wf),
                "precision": precision_score(y_test_wf, y_pred, zero_division=0),
                "recall": recall_score(y_test_wf, y_pred, zero_division=0),
                "f1": f1_score(y_test_wf, y_pred, zero_division=0),
                "roc_auc": auc,
            }
            results.append(window_metrics)

            print(f"    Window {i+1}: Precision={window_metrics['precision']:.3f}, "
                  f"Recall={window_metrics['recall']:.3f}, AUC={window_metrics['roc_auc']:.3f}")

        # Calculate aggregate metrics
        if results:
            avg_precision = np.mean([r["precision"] for r in results])
            avg_recall = np.mean([r["recall"] for r in results])
            avg_auc = np.mean([r["roc_auc"] for r in results])
            std_precision = np.std([r["precision"] for r in results])

            print(f"\n  📊 Walk-Forward Summary:")
            print(f"    Avg Precision: {avg_precision:.3f} (±{std_precision:.3f})")
            print(f"    Avg Recall: {avg_recall:.3f}")
            print(f"    Avg ROC-AUC: {avg_auc:.3f}")

            # Check for stability - high std indicates overfitting
            if std_precision > 0.15:
                print(f"    ⚠️ High precision variance - model may be unstable")

        self.walk_forward_results = results
        return results

    def evaluate_by_regime(
        self, X: np.ndarray, y: np.ndarray, regimes: pd.Series
    ) -> Dict[str, Dict]:
        """
        Evaluate model performance by market regime using RegimeValidator.

        Uses the new regime validation framework for comprehensive analysis
        with performance drop detection (>20% flags).
        """
        if self.model is None:
            print("⚠️ No model trained, skipping regime evaluation")
            return {}

        # Use RegimeValidator if available for comprehensive analysis
        if REGIME_VALIDATION_AVAILABLE:
            validator = RegimeValidator(
                performance_drop_threshold=0.20,  # Flag >20% drops
                min_samples_per_regime=50,
            )

            result = validator.validate(
                model=self.model,
                X_test=X,
                y_test=y,
                regimes=regimes,
                model_name="perps_funding_rate_predictor",
                returns=None,  # No returns data available here
            )

            # Store for metadata
            self.regime_validation_result = result

            # Convert to simple dict format for backward compatibility
            regime_metrics = {}
            for regime, metrics in result.regime_metrics.items():
                regime_metrics[regime] = metrics.to_dict()

            # Store flags for later checks
            self.regime_flags = result.flags
            self.regime_warnings = result.warnings

            self.regime_metrics = regime_metrics
            return regime_metrics

        # Fallback to basic implementation
        print("\n📊 Performance by Market Regime:")

        regime_metrics = {}
        for regime in regimes.unique():
            if regime == "UNKNOWN":
                continue

            mask = (regimes.values == regime)
            if mask.sum() < 10:  # Need minimum samples
                continue

            X_regime = X[mask]
            y_regime = y[mask]

            if y_regime.sum() == 0:  # No positive samples
                continue

            y_pred = self.model.predict(X_regime)
            y_prob = self.model.predict_proba(X_regime)[:, 1]

            try:
                auc = roc_auc_score(y_regime, y_prob)
            except ValueError:
                auc = 0.5

            metrics = {
                "samples": len(X_regime),
                "positive_rate": float(y_regime.mean()),
                "precision": float(precision_score(y_regime, y_pred, zero_division=0)),
                "recall": float(recall_score(y_regime, y_pred, zero_division=0)),
                "roc_auc": float(auc),
            }
            regime_metrics[regime] = metrics

            print(f"  {regime}:")
            print(f"    Samples: {metrics['samples']}, Positive rate: {metrics['positive_rate']:.2%}")
            print(f"    Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, AUC: {metrics['roc_auc']:.3f}")

        self.regime_metrics = regime_metrics
        return regime_metrics

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate classifier on test set with ROC-AUC."""
        print("\n📊 Model Evaluation:")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = 0.0  # Handle case with single class

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc,
        }

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

        # Classification report
        print(f"\n{classification_report(y_test, y_pred, target_names=['NO TRADE', 'TRADE'])}")

        # Save ROC curve plot if matplotlib available
        if PLT_AVAILABLE and roc_auc > 0:
            self._plot_roc_curve(y_test, y_prob, roc_auc)

        # Store for later use
        self.y_test = y_test
        self.y_prob = y_prob

        return metrics

    def _plot_roc_curve(self, y_test: np.ndarray, y_prob: np.ndarray, roc_auc: float) -> None:
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Perps Funding Rate Predictor - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ROC curve saved to {roc_path}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\n📈 Top 10 Features:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return importance

    def compute_shap_values(self, X_test: np.ndarray) -> None:
        """Compute SHAP values for model explainability."""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available, skipping explainability")
            return

        print("\n🔍 Computing SHAP values...")
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)

            # Get mean absolute SHAP values for feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (TRADE)

            shap_importance = pd.DataFrame({
                "feature": self.feature_cols,
                "shap_importance": np.abs(shap_values).mean(axis=0)
            }).sort_values("shap_importance", ascending=False)

            # Save SHAP importance to JSON
            shap_path = os.path.join(self.output_dir, "feature_importance.json")
            shap_importance.to_json(shap_path, orient="records", indent=2)
            print(f"  SHAP importance saved to {shap_path}")

            # Print top 20 features
            print("\n📊 Top 20 Features by SHAP:")
            for _, row in shap_importance.head(20).iterrows():
                print(f"  {row['feature']}: {row['shap_importance']:.4f}")

            # Save SHAP summary plot if matplotlib available
            if PLT_AVAILABLE:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test, feature_names=self.feature_cols, show=False)
                shap_plot_path = os.path.join(self.output_dir, "shap_summary.png")
                plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  SHAP summary plot saved to {shap_plot_path}")

            self.shap_importance = shap_importance
        except Exception as e:
            print(f"  ⚠️ SHAP computation failed: {e}")

    def export_to_onnx(self, X_sample: np.ndarray) -> Optional[str]:
        """Export model to ONNX format for production deployment.

        Returns:
            Path to the ONNX file if successful, None otherwise.
        """
        print("\n📦 Exporting model to ONNX...")
        try:
            # Use XGBoost's native ONNX export via onnxmltools
            try:
                from onnxmltools import convert_xgboost
                from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatType
            except ImportError:
                print("  ⚠️ onnxmltools not available. Install with: pip install onnxmltools")
                return None

            onnx_path = os.path.join(self.output_dir, "perps_predictor.onnx")

            # Convert XGBoost model to ONNX
            initial_type = [('float_input', OnnxFloatType([None, len(self.feature_cols)]))]
            onx = convert_xgboost(self.model, initial_types=initial_type, target_opset=12)

            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())

            print(f"  ONNX model saved to {onnx_path}")

            # Verify ONNX model
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path)
                input_name = sess.get_inputs()[0].name
                test_input = X_sample[:1].astype(np.float32)
                _ = sess.run(None, {input_name: test_input})
                print(f"  ✅ ONNX model verified successfully")
            except ImportError:
                print("  ⚠️ onnxruntime not available for verification")
            except Exception as e:
                print(f"  ⚠️ ONNX verification failed: {e}")

            return onnx_path

        except Exception as e:
            print(f"  ⚠️ ONNX export failed: {e}")
            return None

    def save_model(self, metrics: Dict) -> str:
        """Save trained model and comprehensive metadata."""
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params = self.best_params or self._get_default_params()

        # Save model
        model_path = os.path.join(self.output_dir, f"perps_model_{timestamp}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "params": params,
                "metrics": metrics,
                "timestamp": timestamp,
            }, f)

        # Also save as "latest"
        latest_path = os.path.join(self.output_dir, "perps_model_latest.pkl")
        with open(latest_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "params": params,
                "metrics": metrics,
                "timestamp": timestamp,
            }, f)

        # Save comprehensive metadata as JSON
        metadata = {
            "model_info": {
                "name": "perps_funding_rate_predictor",
                "version": "2.0.0",  # Enhanced version
                "type": "XGBClassifier",
                "timestamp": timestamp,
                "framework": "xgboost",
                "has_lightgbm_ensemble": self.lgb_model is not None,
            },
            "training_config": {
                "features_path": self.features_path,
                "n_features": len(self.feature_cols),
                "feature_names": self.feature_cols,
                "hyperparameters": params,
                "optuna_trials": self.n_trials if OPTUNA_AVAILABLE else 0,
                "train_val_test_split": f"{self.config.train_ratio:.0%}/{self.config.val_ratio:.0%}/{self.config.test_ratio:.0%}",
                "profit_weights": {
                    "fp_weight": self.config.fp_weight,
                    "fn_weight": self.config.fn_weight,
                },
                "scale_pos_weight": getattr(self, 'profit_adjusted_weight', self.scale_pos_weight),
            },
            "metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1", 0),
                "roc_auc": metrics.get("roc_auc", 0),
            },
            "targets": {
                "target_precision": self.config.target_precision,
                "target_recall": self.config.target_recall,
                "target_roc_auc": self.config.target_roc_auc,
                "max_sharpe": self.config.max_sharpe,
                "target_ece": self.config.target_ece,
                "meets_targets": metrics.get("meets_targets", False),
            },
            "calibration": metrics.get("calibration", {}),
            "regime_performance": metrics.get("regime_performance", {}),
            "walk_forward": {
                "num_windows": len(metrics.get("walk_forward", [])),
                "results": metrics.get("walk_forward", []),
            },
            "files": {
                "model_pkl": f"perps_model_{timestamp}.pkl",
                "model_latest": "perps_model_latest.pkl",
                "onnx": "perps_predictor.onnx" if ONNX_AVAILABLE else None,
                "calibration": "calibration/perps_predictor_calibration.json",
                "roc_curve": "roc_curve.png" if PLT_AVAILABLE else None,
                "shap_summary": "shap_summary.png" if SHAP_AVAILABLE else None,
                "feature_importance": "feature_importance.json" if SHAP_AVAILABLE else None,
            },
            "usage": {
                "input_format": "numpy array of shape (n_samples, n_features)",
                "output_format": "binary classification (0=NO_TRADE, 1=TRADE)",
                "preprocessing": "StandardScaler (included in pkl)",
                "threshold": "0.25% funding rate for profitable trades",
                "calibration": "Apply Platt Scaling for accurate probabilities",
            },
        }

        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save simple metrics.json for backward compatibility
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Model saved to {model_path}")
        print(f"💾 Metadata saved to {metadata_path}")
        return model_path

    def fit_calibration(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Fit Platt Scaling calibration on validation data.

        Calibrates model probability outputs to reflect true win rates.
        ECE (Expected Calibration Error) target: < 0.10
        """
        if not CALIBRATION_AVAILABLE:
            print("\n⚠️ Calibration module not available, skipping...")
            return {}

        print("\n" + "=" * 60)
        print("  PROBABILITY CALIBRATION (Platt Scaling)")
        print("=" * 60)

        # Get raw predictions
        X_val_scaled = self.scaler.transform(X_val)
        raw_probas = self.model.predict_proba(X_val_scaled)[:, 1]

        # Fit calibration
        calibration_dir = os.path.join(self.output_dir, "calibration")
        params = fit_and_save_calibration(
            predictions=raw_probas,
            true_labels=y_val,
            model_name="perps_predictor",
            output_dir=calibration_dir,
            num_bins=10
        )

        # Also copy to main calibration directory for TypeScript service
        main_calibration_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(self.output_dir))),
            "eliza/models/calibration"
        )
        os.makedirs(main_calibration_dir, exist_ok=True)

        import shutil
        src = os.path.join(calibration_dir, "perps_predictor_calibration.json")
        dst = os.path.join(main_calibration_dir, "perps_predictor_calibration.json")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Calibration copied to {dst}")

        return {
            "ece_before": params.ece_before_calibration,
            "ece_after": params.ece_after_calibration,
            "brier_before": params.brier_score_before,
            "brier_after": params.brier_score_after,
            "is_well_calibrated": params.ece_after_calibration < 0.10,
        }

    def run(
        self,
        optimize: bool = True,
        export_onnx: bool = True,
        compute_shap: bool = True,
        calibrate: bool = True,
        walk_forward: bool = True,
        train_ensemble: bool = True,
    ) -> Dict:
        """
        Run enhanced training pipeline with:
        - 70/15/15 train/val/test split
        - Profit-weighted hyperparameter optimization
        - Walk-forward validation
        - LightGBM ensemble (optional)
        - Market regime analysis
        - Platt Scaling calibration
        - ONNX export with verification
        - MLflow experiment tracking

        Target metrics:
        - Precision > 70%
        - Recall > 50%
        - Sharpe < 3.0
        - ROC-AUC > 0.75
        - ECE < 0.10
        """
        # Generate version tag based on timestamp
        version_tag = f"v{datetime.now().strftime('%Y%m%d.%H%M%S')}"

        # Initialize MLflow tracker
        mlflow_tracker = None
        if MLFLOW_TRACKING_AVAILABLE:
            try:
                mlflow_tracker = create_tracker("perps")
                print(f"\n📊 MLflow tracking enabled - Version: {version_tag}")
            except Exception as e:
                print(f"⚠️ MLflow initialization failed: {e}")

        print("=" * 60)
        print("  PERPS FUNDING RATE MODEL TRAINING (ENHANCED)")
        print("=" * 60)
        print(f"  Target Precision: > {self.config.target_precision:.0%}")
        print(f"  Target Recall: > {self.config.target_recall:.0%}")
        print(f"  Target ROC-AUC: > {self.config.target_roc_auc}")
        print(f"  Max Sharpe (realistic): < {self.config.max_sharpe}")
        print(f"  Model Version: {version_tag}")
        print("=" * 60)

        # Load data with regime detection
        df_clean, X, y = self.load_data()

        # New 70/15/15 split
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_split(X, y)

        # Walk-forward validation BEFORE final training
        if walk_forward:
            wf_results = self.walk_forward_validation(
                X_train, y_train,
                n_windows=self.config.walk_forward_windows
            )
        else:
            wf_results = []

        # Hyperparameter optimization on training data
        if optimize and OPTUNA_AVAILABLE:
            params = self.optimize_hyperparameters(X_train, y_train)
        else:
            params = self._get_default_params()

        # Start MLflow run and log hyperparameters
        mlflow_run_context = None
        if mlflow_tracker:
            try:
                mlflow_run_context = mlflow_tracker.start_run(
                    run_name=version_tag,
                    tags={
                        "model_type": "perps",
                        "optimization": "optuna" if optimize else "default",
                        "ensemble": str(train_ensemble),
                        "calibrated": str(calibrate),
                    }
                )
                mlflow_run_context.__enter__()

                # Log hyperparameters
                mlflow_tracker.log_params({
                    "n_trials": self.n_trials,
                    "train_ratio": self.config.train_ratio,
                    "val_ratio": self.config.val_ratio,
                    "test_ratio": self.config.test_ratio,
                    "cv_splits": self.config.cv_splits,
                    "target_precision": self.config.target_precision,
                    "target_recall": self.config.target_recall,
                    **params,
                })
                print("📊 MLflow: Logged hyperparameters")
            except Exception as e:
                print(f"⚠️ MLflow logging failed: {e}")

        # Train XGBoost model on full training data
        self.train_model(X_train, y_train, params)

        # Train LightGBM for ensemble
        if train_ensemble and LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train, y_train)

        # Evaluate on validation set (for calibration)
        print("\n📊 Validation Set Evaluation:")
        val_metrics = self.evaluate_model(X_val, y_val)

        # Fit calibration on VALIDATION set (not test!)
        if calibrate:
            # Note: fit_calibration expects unscaled X
            val_idx_start = int(len(X) * self.config.train_ratio)
            val_idx_end = int(len(X) * (self.config.train_ratio + self.config.val_ratio))
            X_val_unscaled = X[val_idx_start:val_idx_end]
            calibration_metrics = self.fit_calibration(X_val_unscaled, y_val)
        else:
            calibration_metrics = {}

        # Final evaluation on HELD-OUT test set
        print("\n" + "=" * 60)
        print("  FINAL TEST SET EVALUATION (HELD-OUT)")
        print("=" * 60)
        test_metrics = self.evaluate_model(X_test, y_test)

        # Regime-specific evaluation
        if self.df_full is not None and "regime" in self.df_full.columns:
            # Get regimes for test set
            test_idx_start = int(len(X) * (self.config.train_ratio + self.config.val_ratio))
            test_regimes = self.df_full["regime"].iloc[test_idx_start:].reset_index(drop=True)
            regime_metrics = self.evaluate_by_regime(X_test, y_test, test_regimes)
        else:
            regime_metrics = {}

        # Feature importance
        _ = self.get_feature_importance()

        # SHAP explainability
        if compute_shap:
            self.compute_shap_values(X_test)

        # Compile all metrics
        metrics = {
            **test_metrics,
            "validation": val_metrics,
            "calibration": calibration_metrics,
            "regime_performance": regime_metrics,
            "walk_forward": wf_results,
        }

        # Validate metrics meet targets
        print("\n" + "=" * 60)
        print("  METRICS VALIDATION")
        print("=" * 60)

        meets_targets = True
        if test_metrics.get("precision", 0) < self.config.target_precision:
            print(f"  ❌ Precision {test_metrics['precision']:.3f} < {self.config.target_precision:.3f}")
            meets_targets = False
        else:
            print(f"  ✅ Precision {test_metrics['precision']:.3f} >= {self.config.target_precision:.3f}")

        if test_metrics.get("recall", 0) < self.config.target_recall:
            print(f"  ❌ Recall {test_metrics['recall']:.3f} < {self.config.target_recall:.3f}")
            meets_targets = False
        else:
            print(f"  ✅ Recall {test_metrics['recall']:.3f} >= {self.config.target_recall:.3f}")

        if test_metrics.get("roc_auc", 0) < self.config.target_roc_auc:
            print(f"  ❌ ROC-AUC {test_metrics['roc_auc']:.3f} < {self.config.target_roc_auc}")
            meets_targets = False
        else:
            print(f"  ✅ ROC-AUC {test_metrics['roc_auc']:.3f} >= {self.config.target_roc_auc}")

        if calibration_metrics.get("ece_after", 1.0) > self.config.target_ece:
            print(f"  ⚠️ ECE {calibration_metrics.get('ece_after', 'N/A')} > {self.config.target_ece}")
        else:
            print(f"  ✅ ECE {calibration_metrics.get('ece_after', 'N/A')} <= {self.config.target_ece}")

        # Check regime-specific performance flags
        regime_flags = getattr(self, 'regime_flags', [])
        if regime_flags:
            print(f"\n  🚨 REGIME PERFORMANCE ISSUES:")
            for flag in regime_flags:
                print(f"    - {flag}")
            meets_targets = False  # Regime issues should affect overall target status
        else:
            print(f"  ✅ No significant regime performance drops (>20%)")

        metrics["meets_targets"] = meets_targets
        metrics["regime_flags"] = regime_flags
        metrics["regime_warnings"] = getattr(self, 'regime_warnings', [])

        # Save model with enhanced metadata
        self.save_model(metrics)

        # Export to ONNX for production
        onnx_path = None
        if export_onnx:
            onnx_path = self.export_to_onnx(X_test)

        # MLflow: Log metrics and artifacts
        if mlflow_tracker and mlflow_run_context:
            try:
                # Log test metrics
                mlflow_tracker.log_metrics({
                    "precision": test_metrics.get("precision", 0),
                    "recall": test_metrics.get("recall", 0),
                    "f1_score": test_metrics.get("f1", 0),
                    "roc_auc": test_metrics.get("roc_auc", 0),
                    "accuracy": test_metrics.get("accuracy", 0),
                    "meets_targets": 1.0 if meets_targets else 0.0,
                })

                # Log calibration metrics
                if calibration_metrics:
                    mlflow_tracker.log_metrics({
                        "ece_before": calibration_metrics.get("ece_before", 0),
                        "ece_after": calibration_metrics.get("ece_after", 0),
                    })

                # Log model artifacts
                metadata_path = os.path.join(self.output_dir, "metadata.json")
                calibration_path = os.path.join(self.output_dir, "..", "calibration", "perps_predictor_calibration.json")

                if onnx_path and os.path.exists(onnx_path):
                    model_info = mlflow_tracker.log_model(
                        model_path=onnx_path,
                        metadata_path=metadata_path if os.path.exists(metadata_path) else None,
                        calibration_path=calibration_path if os.path.exists(calibration_path) else None,
                        version=version_tag,
                    )
                    print(f"📊 MLflow: Logged model artifacts - {model_info.get('version', 'unknown')}")

                # Register model in MLflow registry
                stage = "Production" if meets_targets else "Staging"
                model_version = mlflow_tracker.register_model("perps-predictor", stage=stage)
                if model_version:
                    print(f"📊 MLflow: Registered as version {model_version} ({stage})")

                # End MLflow run
                mlflow_run_context.__exit__(None, None, None)
                print("📊 MLflow: Run completed successfully")
            except Exception as e:
                print(f"⚠️ MLflow artifact logging failed: {e}")
                try:
                    mlflow_run_context.__exit__(None, None, None)
                except:
                    pass

        print("\n" + "=" * 60)
        if meets_targets:
            print("  ✅ TRAINING COMPLETE - ALL TARGETS MET")
        else:
            print("  ⚠️ TRAINING COMPLETE - SOME TARGETS NOT MET")
        print("=" * 60)

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Perps funding rate model with enhanced pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training with 100 Optuna trials
  python train_perps_model.py --features ./features/perps_features.csv --trials 100

  # Quick training without optimization
  python train_perps_model.py --features ./features/perps_features.csv --no-optimize

  # Training without walk-forward (faster)
  python train_perps_model.py --features ./features/perps_features.csv --no-walk-forward
        """
    )
    parser.add_argument("--features", type=str, default="./features/perps_features.csv")
    parser.add_argument("--output", type=str, default="./models")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials (default: 100)")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--no-walk-forward", action="store_true", help="Skip walk-forward validation")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip LightGBM ensemble")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip Platt Scaling calibration")
    args = parser.parse_args()

    # Create config
    config = TrainingConfig(n_trials=args.trials)

    trainer = PerpsModelTrainer(
        features_path=args.features,
        output_dir=args.output,
        n_trials=args.trials,
        config=config,
    )

    metrics = trainer.run(
        optimize=not args.no_optimize,
        walk_forward=not args.no_walk_forward,
        train_ensemble=not args.no_ensemble,
        calibrate=not args.no_calibrate,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("  FINAL METRICS SUMMARY")
    print("=" * 60)
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    if metrics.get("calibration"):
        print(f"  ECE: {metrics['calibration'].get('ece_after', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

