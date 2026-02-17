from __future__ import annotations
"""
Model Drift Detection.

Monitors for data and concept drift in production.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
import structlog

from .metrics import MODEL_DRIFT_SCORE, FEATURE_DRIFT

logger = structlog.get_logger()


@dataclass
class DriftResult:
    """Result of drift detection."""
    strategy: str
    drift_detected: bool
    drift_score: float
    feature_drifts: dict[str, float]
    timestamp: datetime
    recommendation: str


class DriftDetector:
    """
    Detects model and data drift.
    
    Uses:
    - Population Stability Index (PSI) for feature drift
    - Kolmogorov-Smirnov test for distribution shift
    - Prediction distribution monitoring
    """
    
    # Thresholds
    PSI_THRESHOLD = 0.2  # PSI > 0.2 indicates significant drift
    KS_THRESHOLD = 0.05  # p-value < 0.05 indicates drift
    
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.reference_data: pd.DataFrame | None = None
        self.reference_predictions: np.ndarray | None = None
        self.logger = logger.bind(component="drift_detector", strategy=strategy)
    
    def set_reference(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray
    ) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            X: Reference feature data (typically training data)
            predictions: Reference predictions
        """
        self.reference_data = X.copy()
        self.reference_predictions = predictions.copy()
        self.logger.info(
            "Reference data set",
            samples=len(X),
            features=X.shape[1]
        )
    
    def detect_drift(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray
    ) -> DriftResult:
        """
        Detect drift in new data.
        
        Args:
            X: New feature data
            predictions: New predictions
            
        Returns:
            DriftResult with drift scores
        """
        if self.reference_data is None:
            raise RuntimeError("Reference data not set. Call set_reference() first.")
        
        # Calculate feature drift (PSI)
        feature_drifts = {}
        for col in X.columns:
            if col in self.reference_data.columns:
                psi = self._calculate_psi(
                    self.reference_data[col].values,
                    X[col].values
                )
                feature_drifts[col] = psi
                
                # Update Prometheus metric
                FEATURE_DRIFT.labels(
                    strategy=self.strategy,
                    feature=col
                ).set(psi)
        
        # Calculate prediction drift (KS test)
        ks_stat, ks_pvalue = stats.ks_2samp(
            self.reference_predictions,
            predictions
        )
        
        # Overall drift score (weighted average)
        avg_feature_drift = np.mean(list(feature_drifts.values())) if feature_drifts else 0
        drift_score = 0.6 * avg_feature_drift + 0.4 * ks_stat
        
        # Update Prometheus metric
        MODEL_DRIFT_SCORE.labels(strategy=self.strategy).set(drift_score)
        
        # Determine if drift is significant
        drift_detected = (
            drift_score > self.PSI_THRESHOLD or
            ks_pvalue < self.KS_THRESHOLD
        )
        
        # Generate recommendation
        if drift_detected:
            if avg_feature_drift > self.PSI_THRESHOLD:
                recommendation = "Feature distribution has shifted. Consider retraining."
            else:
                recommendation = "Prediction distribution has shifted. Monitor closely."
        else:
            recommendation = "No significant drift detected."
        
        result = DriftResult(
            strategy=self.strategy,
            drift_detected=drift_detected,
            drift_score=drift_score,
            feature_drifts=feature_drifts,
            timestamp=datetime.utcnow(),
            recommendation=recommendation
        )
        
        self.logger.info(
            "Drift detection complete",
            drift_detected=drift_detected,
            drift_score=drift_score
        )
        
        return result
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI = Σ (current% - reference%) * ln(current% / reference%)
        """
        # Handle edge cases
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Create bins from reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate percentages
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)
        
        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)
        
        # Avoid division by zero
        ref_pct = np.clip(ref_pct, 0.0001, 1)
        cur_pct = np.clip(cur_pct, 0.0001, 1)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def get_drifted_features(
        self,
        feature_drifts: dict[str, float],
        threshold: float | None = None
    ) -> list[str]:
        """
        Get list of features with significant drift.

        Args:
            feature_drifts: Dict of feature -> PSI score
            threshold: PSI threshold (default: PSI_THRESHOLD)

        Returns:
            List of feature names with drift
        """
        threshold = threshold or self.PSI_THRESHOLD
        return [
            feature for feature, psi in feature_drifts.items()
            if psi > threshold
        ]


class DriftMonitor:
    """
    Monitors drift across all strategies.
    """

    def __init__(self):
        self.detectors: dict[str, DriftDetector] = {}
        self.logger = logger.bind(component="drift_monitor")

    def register_strategy(
        self,
        strategy: str,
        reference_X: pd.DataFrame,
        reference_predictions: np.ndarray
    ) -> None:
        """Register a strategy for drift monitoring."""
        detector = DriftDetector(strategy)
        detector.set_reference(reference_X, reference_predictions)
        self.detectors[strategy] = detector

        self.logger.info("Strategy registered for drift monitoring", strategy=strategy)

    def check_all(
        self,
        data: dict[str, tuple[pd.DataFrame, np.ndarray]]
    ) -> dict[str, DriftResult]:
        """
        Check drift for all registered strategies.

        Args:
            data: Dict of strategy -> (features, predictions)

        Returns:
            Dict of strategy -> DriftResult
        """
        results = {}

        for strategy, (X, predictions) in data.items():
            if strategy in self.detectors:
                results[strategy] = self.detectors[strategy].detect_drift(X, predictions)

        return results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of drift status across all strategies."""
        return {
            "strategies_monitored": list(self.detectors.keys()),
            "total_strategies": len(self.detectors)
        }
