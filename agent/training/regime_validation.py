#!/usr/bin/env python3
"""
Regime-Specific Model Validation

Tests models across different market regimes (BULL, BEAR, SIDEWAYS) to
identify performance gaps. Flags if performance drops >20% in any regime.

Key features:
- Per-regime precision, recall, F1, ROC-AUC, Sharpe
- Cross-regime performance comparison
- Regime-specific warnings and flags
- Metadata generation for regime performance

Usage:
    from regime_validation import RegimeValidator
    validator = RegimeValidator()
    results = validator.validate(model, X_test, y_test, regimes)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# ML imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RegimeMetrics:
    """Metrics for a single market regime."""
    regime: str
    samples: int
    positive_rate: float  # % of positive labels in this regime
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    sharpe: Optional[float] = None
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "sharpe": round(self.sharpe, 4) if self.sharpe else None,
            "samples": self.samples,
            "positive_rate": round(self.positive_rate, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
        }


@dataclass
class RegimeValidationResult:
    """Complete regime validation results."""
    model_name: str
    validation_timestamp: str
    overall_metrics: Dict[str, float]
    regime_metrics: Dict[str, RegimeMetrics]
    warnings: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    performance_drop_threshold: float = 0.20  # 20% drop triggers flag
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "validation_timestamp": self.validation_timestamp,
            "overall_metrics": self.overall_metrics,
            "regime_performance": {
                regime: metrics.to_dict()
                for regime, metrics in self.regime_metrics.items()
            },
            "warnings": self.warnings,
            "flags": self.flags,
            "performance_drop_threshold": self.performance_drop_threshold,
        }
    
    def has_regime_issues(self) -> bool:
        """Check if any regime has significant performance issues."""
        return len(self.flags) > 0


class RegimeValidator:
    """
    Validates model performance across market regimes.
    
    Evaluates on BULL, BEAR, and SIDEWAYS markets separately,
    identifies performance gaps, and flags concerning patterns.
    """
    
    def __init__(
        self,
        performance_drop_threshold: float = 0.20,
        min_samples_per_regime: int = 50,
    ):
        """
        Args:
            performance_drop_threshold: Flag if any regime drops more than this vs overall
            min_samples_per_regime: Minimum samples needed to evaluate a regime
        """
        self.performance_drop_threshold = performance_drop_threshold
        self.min_samples_per_regime = min_samples_per_regime
    
    def validate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regimes: pd.Series,
        model_name: str = "model",
        returns: Optional[pd.Series] = None,
    ) -> RegimeValidationResult:
        """
        Validate model across all market regimes.
        
        Args:
            model: Trained sklearn-compatible model with predict/predict_proba
            X_test: Test features
            y_test: Test labels
            regimes: Series of regime labels ('BULL', 'BEAR', 'SIDEWAYS')
            model_name: Name for the model
            returns: Optional series of returns for Sharpe calculation
            
        Returns:
            RegimeValidationResult with per-regime metrics
        """
        print(f"\n{'='*60}")
        print(f"  REGIME-SPECIFIC VALIDATION: {model_name}")
        print(f"{'='*60}")
        
        # Calculate overall metrics first
        overall_metrics = self._calculate_metrics(model, X_test, y_test, returns)
        
        print(f"\n  📊 Overall Metrics:")
        print(f"    Precision: {overall_metrics['precision']:.4f}")
        print(f"    Recall:    {overall_metrics['recall']:.4f}")
        print(f"    F1:        {overall_metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:   {overall_metrics['roc_auc']:.4f}")
        
        # Validate each regime
        regime_metrics = {}
        warnings = []
        flags = []

        print(f"\n  📊 Per-Regime Metrics:")

        for regime in ["BULL", "BEAR", "SIDEWAYS"]:
            mask = regimes.values == regime
            n_samples = mask.sum()

            if n_samples < self.min_samples_per_regime:
                warnings.append(
                    f"{regime}: Only {n_samples} samples (min: {self.min_samples_per_regime})"
                )
                continue

            X_regime = X_test[mask]
            y_regime = y_test[mask]
            returns_regime = returns.iloc[mask] if returns is not None else None

            metrics = self._calculate_regime_metrics(
                model, X_regime, y_regime, regime, returns_regime
            )
            regime_metrics[regime] = metrics

            # Print regime metrics
            print(f"\n    {regime}:")
            print(f"      Samples: {metrics.samples} (positive rate: {metrics.positive_rate:.1%})")
            print(f"      Precision: {metrics.precision:.4f}")
            print(f"      Recall:    {metrics.recall:.4f}")
            print(f"      F1:        {metrics.f1_score:.4f}")
            print(f"      ROC-AUC:   {metrics.roc_auc:.4f}")
            if metrics.sharpe is not None:
                print(f"      Sharpe:    {metrics.sharpe:.2f}")

            # Check for significant performance drops
            self._check_performance_drop(
                regime, metrics, overall_metrics, warnings, flags
            )

        # Create result
        result = RegimeValidationResult(
            model_name=model_name,
            validation_timestamp=datetime.now().isoformat(),
            overall_metrics=overall_metrics,
            regime_metrics=regime_metrics,
            warnings=warnings,
            flags=flags,
            performance_drop_threshold=self.performance_drop_threshold,
        )

        # Print warnings and flags
        if warnings:
            print(f"\n  ⚠️ Warnings:")
            for w in warnings:
                print(f"    - {w}")

        if flags:
            print(f"\n  🚨 REGIME PERFORMANCE FLAGS:")
            for f in flags:
                print(f"    - {f}")
        else:
            print(f"\n  ✅ No significant regime performance issues detected")

        return result

    def _calculate_metrics(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        y_pred = model.predict(X)

        try:
            y_prob = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_prob)
        except (ValueError, AttributeError):
            roc_auc = 0.0

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc),
        }

        # Calculate Sharpe if returns provided
        if returns is not None:
            sharpe = self._calculate_sharpe(returns, y_pred)
            metrics["sharpe"] = sharpe

        return metrics

    def _calculate_regime_metrics(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        regime: str,
        returns: Optional[pd.Series] = None,
    ) -> RegimeMetrics:
        """Calculate metrics for a specific regime."""
        y_pred = model.predict(X)

        try:
            y_prob = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_prob)
        except (ValueError, AttributeError):
            roc_auc = 0.0

        sharpe = None
        avg_return = 0.0
        if returns is not None:
            sharpe = self._calculate_sharpe(returns, y_pred)
            trade_mask = y_pred == 1
            if trade_mask.sum() > 0:
                avg_return = float(returns.iloc[trade_mask].mean())

        # Calculate win rate from predictions and actual outcomes
        trade_mask = y_pred == 1
        total_trades = int(trade_mask.sum())
        winning_trades = int((y[trade_mask] == 1).sum()) if total_trades > 0 else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return RegimeMetrics(
            regime=regime,
            samples=len(y),
            positive_rate=float(y.mean()),
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1_score=float(f1_score(y, y_pred, zero_division=0)),
            roc_auc=float(roc_auc),
            sharpe=sharpe,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=win_rate,
            avg_return=avg_return,
        )

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        predictions: np.ndarray,
        periods_per_year: int = 365,
    ) -> float:
        """Calculate Sharpe ratio for strategy returns."""
        # Filter returns for trades (predictions == 1)
        trade_mask = predictions == 1
        if trade_mask.sum() < 2:
            return 0.0

        strategy_returns = returns.iloc[trade_mask]

        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

        # Cap at reasonable values
        return float(np.clip(sharpe, -5.0, 5.0))

    def _check_performance_drop(
        self,
        regime: str,
        regime_metrics: RegimeMetrics,
        overall_metrics: Dict[str, float],
        warnings: List[str],
        flags: List[str],
    ) -> None:
        """Check for significant performance drops in a regime."""
        # Check precision drop
        if overall_metrics["precision"] > 0:
            precision_drop = (
                (overall_metrics["precision"] - regime_metrics.precision)
                / overall_metrics["precision"]
            )
            if precision_drop > self.performance_drop_threshold:
                flags.append(
                    f"{regime}: Precision dropped {precision_drop:.1%} "
                    f"({overall_metrics['precision']:.3f} → {regime_metrics.precision:.3f})"
                )

        # Check recall drop
        if overall_metrics["recall"] > 0:
            recall_drop = (
                (overall_metrics["recall"] - regime_metrics.recall)
                / overall_metrics["recall"]
            )
            if recall_drop > self.performance_drop_threshold:
                flags.append(
                    f"{regime}: Recall dropped {recall_drop:.1%} "
                    f"({overall_metrics['recall']:.3f} → {regime_metrics.recall:.3f})"
                )

        # Check ROC-AUC drop
        if overall_metrics["roc_auc"] > 0.5:  # Only if better than random
            auc_drop = (
                (overall_metrics["roc_auc"] - regime_metrics.roc_auc)
                / (overall_metrics["roc_auc"] - 0.5)  # Relative to random baseline
            )
            if auc_drop > self.performance_drop_threshold:
                flags.append(
                    f"{regime}: ROC-AUC dropped {auc_drop:.1%} "
                    f"({overall_metrics['roc_auc']:.3f} → {regime_metrics.roc_auc:.3f})"
                )

        # Check for near-random performance
        if regime_metrics.roc_auc < 0.55:
            warnings.append(
                f"{regime}: Near-random AUC ({regime_metrics.roc_auc:.3f})"
            )


def validate_by_regime(
    model: Any,
    data: pd.DataFrame,
    feature_columns: List[str],
    label_column: str = "label",
    regime_column: str = "regime",
    returns_column: Optional[str] = None,
    model_name: str = "model",
) -> RegimeValidationResult:
    """
    Convenience function to validate model by regime.

    Args:
        model: Trained model
        data: DataFrame with features, labels, and regime column
        feature_columns: List of feature column names
        label_column: Name of label column
        regime_column: Name of regime column
        returns_column: Optional returns column for Sharpe
        model_name: Name of the model

    Returns:
        RegimeValidationResult
    """
    X = data[feature_columns].values
    y = data[label_column].values
    regimes = data[regime_column]
    returns = data[returns_column] if returns_column else None

    validator = RegimeValidator()
    return validator.validate(model, X, y, regimes, model_name, returns)


def generate_regime_metadata(
    result: RegimeValidationResult,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate metadata JSON for regime performance.

    Args:
        result: RegimeValidationResult from validation
        output_path: Optional path to save JSON

    Returns:
        Metadata dictionary
    """
    metadata = result.to_dict()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n💾 Regime metadata saved to: {output_path}")

    return metadata

