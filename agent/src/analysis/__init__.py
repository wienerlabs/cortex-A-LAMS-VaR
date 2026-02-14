"""
Analysis module for market regime detection and validation.
"""
from .regimeDetector import (
    RegimeDetector,
    MarketRegime,
    RegimeConfig,
    detect_regime,
    label_data_with_regimes,
)

__all__ = [
    'RegimeDetector',
    'MarketRegime',
    'RegimeConfig',
    'detect_regime',
    'label_data_with_regimes',
]

