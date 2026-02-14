#!/usr/bin/env python3
"""
Model Calibration Module - Platt Scaling Implementation

Calibrates ML model probability outputs to reflect true win rates.
Uses Platt Scaling (logistic regression on model outputs) to transform
raw probabilities into calibrated probabilities.

Problem: Model says 80% confidence, actual win rate might be 60%
Solution: Platt Scaling learns the mapping from raw → calibrated probabilities

ECE (Expected Calibration Error) target: < 0.10
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Optional sklearn import for Platt scaling
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available. Install with: pip install scikit-learn")


@dataclass
class PlattParameters:
    """Platt Scaling parameters for probability calibration."""
    A: float  # Slope parameter
    B: float  # Intercept parameter
    model_name: str
    fitted_at: str
    validation_samples: int
    ece_before_calibration: float
    ece_after_calibration: float
    brier_score_before: float
    brier_score_after: float


@dataclass
class CalibrationEvaluation:
    """Calibration quality metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    calibration_curve: List[Dict]
    is_well_calibrated: bool  # ECE < 0.10


class ModelCalibrator:
    """
    Model Calibration using Platt Scaling.
    
    Fits logistic regression on model outputs to calibrate probabilities.
    """
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.params: Optional[PlattParameters] = None
        self.calibrator: Optional[LogisticRegression] = None
    
    def fit(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        model_name: str
    ) -> PlattParameters:
        """
        Fit Platt Scaling on validation data.
        
        Args:
            predictions: Raw model probabilities (0-1)
            true_labels: Actual binary labels (0 or 1)
            model_name: Name of the model for saving
            
        Returns:
            Fitted Platt parameters
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for calibration")
        
        n = len(predictions)
        if n < 10:
            raise ValueError("Need at least 10 samples for calibration")
        
        print(f"\n📊 Fitting Platt Scaling for {model_name}...")
        print(f"  Samples: {n}")
        print(f"  Positive rate: {np.mean(true_labels):.2%}")
        
        # Evaluate ECE before calibration
        ece_before = self.calculate_ece(predictions, true_labels)
        brier_before = self.calculate_brier_score(predictions, true_labels)
        
        print(f"  ECE before: {ece_before:.4f}")
        print(f"  Brier before: {brier_before:.4f}")
        
        # Fit logistic regression on log-odds
        # Clip predictions to avoid log(0)
        epsilon = 1e-7
        clipped = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Reshape for sklearn
        X = clipped.reshape(-1, 1)
        y = true_labels
        
        # Fit calibrator
        self.calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.calibrator.fit(X, y)
        
        # Extract Platt parameters (A = coef, B = intercept)
        A = float(self.calibrator.coef_[0][0])
        B = float(self.calibrator.intercept_[0])
        
        # Get calibrated predictions
        calibrated = self.calibrator.predict_proba(X)[:, 1]
        
        # Evaluate ECE after calibration
        ece_after = self.calculate_ece(calibrated, true_labels)
        brier_after = self.calculate_brier_score(calibrated, true_labels)
        
        print(f"  ECE after: {ece_after:.4f}")
        print(f"  Brier after: {brier_after:.4f}")
        print(f"  ECE improvement: {ece_before - ece_after:.4f}")
        
        self.params = PlattParameters(
            A=A,
            B=B,
            model_name=model_name,
            fitted_at=datetime.now().isoformat(),
            validation_samples=n,
            ece_before_calibration=ece_before,
            ece_after_calibration=ece_after,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
        )
        
        return self.params
    
    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if self.calibrator is None:
            return raw_proba

        epsilon = 1e-7
        clipped = np.clip(raw_proba, epsilon, 1 - epsilon)
        X = clipped.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]

    def calculate_ece(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures how well predicted probabilities match actual outcomes.
        Lower is better. Target: < 0.10
        """
        n = len(predictions)
        if n == 0:
            return 0.0

        bins = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0

        for i in range(self.num_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if np.sum(mask) > 0:
                avg_pred = np.mean(predictions[mask])
                avg_actual = np.mean(true_labels[mask])
                ece += (np.sum(mask) / n) * np.abs(avg_pred - avg_actual)

        return float(ece)

    def calculate_brier_score(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        """
        Calculate Brier Score.

        Mean squared error between predictions and outcomes.
        Lower is better. Range: 0-1
        """
        return float(np.mean((predictions - true_labels) ** 2))

    def evaluate(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> CalibrationEvaluation:
        """Evaluate calibration quality."""
        ece = self.calculate_ece(predictions, true_labels)
        brier = self.calculate_brier_score(predictions, true_labels)

        # Calculate calibration curve
        bins = np.linspace(0, 1, self.num_bins + 1)
        curve = []
        mce = 0.0

        for i in range(self.num_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if np.sum(mask) > 0:
                avg_pred = np.mean(predictions[mask])
                avg_actual = np.mean(true_labels[mask])
                error = np.abs(avg_pred - avg_actual)
                mce = max(mce, error)

                curve.append({
                    "bin_start": float(bins[i]),
                    "bin_end": float(bins[i + 1]),
                    "mean_predicted": float(avg_pred),
                    "mean_actual": float(avg_actual),
                    "count": int(np.sum(mask)),
                })

        return CalibrationEvaluation(
            ece=ece,
            mce=mce,
            brier_score=brier,
            calibration_curve=curve,
            is_well_calibrated=ece < 0.10,
        )

    def save(self, output_dir: str) -> str:
        """Save calibration parameters to JSON file."""
        if self.params is None:
            raise ValueError("No calibration fitted. Call fit() first.")

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.params.model_name}_calibration.json")

        with open(filepath, "w") as f:
            json.dump(asdict(self.params), f, indent=2)

        print(f"  Calibration saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "ModelCalibrator":
        """Load calibration parameters from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        calibrator = cls()
        calibrator.params = PlattParameters(**data)

        # Reconstruct sklearn calibrator from parameters
        if SKLEARN_AVAILABLE:
            calibrator.calibrator = LogisticRegression()
            calibrator.calibrator.coef_ = np.array([[calibrator.params.A]])
            calibrator.calibrator.intercept_ = np.array([calibrator.params.B])
            calibrator.calibrator.classes_ = np.array([0, 1])

        return calibrator


def fit_and_save_calibration(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    model_name: str,
    output_dir: str,
    num_bins: int = 10
) -> PlattParameters:
    """
    Convenience function to fit and save calibration in one step.

    Args:
        predictions: Raw model probabilities
        true_labels: Actual binary labels
        model_name: Name of the model
        output_dir: Directory to save calibration
        num_bins: Number of bins for ECE calculation

    Returns:
        Fitted Platt parameters
    """
    calibrator = ModelCalibrator(num_bins=num_bins)
    params = calibrator.fit(predictions, true_labels, model_name)
    calibrator.save(output_dir)
    return params

