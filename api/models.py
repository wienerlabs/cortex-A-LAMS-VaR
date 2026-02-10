from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CalibrationMethod(str, Enum):
    MLE = "mle"
    GRID = "grid"
    EMPIRICAL = "empirical"
    HYBRID = "hybrid"


class DataSource(str, Enum):
    SOLANA = "solana"
    YFINANCE = "yfinance"


class CalibrateRequest(BaseModel):
    token: str = Field(..., description="Token symbol (SOL, RAY) or mint address, or yfinance ticker")
    data_source: DataSource = DataSource.SOLANA
    start_date: str = Field(..., description="ISO date string, e.g. '2025-01-01'")
    end_date: str = Field(..., description="ISO date string, e.g. '2026-02-10'")
    num_states: int = Field(5, ge=2, le=10)
    method: CalibrationMethod = CalibrationMethod.MLE
    target_var_breach: float = Field(0.05, gt=0.0, lt=1.0)
    interval: str = Field("1D", description="Candle interval for Solana data")


class CalibrationMetrics(BaseModel):
    var_breach_rate: float
    vol_correlation: float
    log_likelihood: float
    aic: float
    bic: float


class CalibrateResponse(BaseModel):
    token: str
    method: str
    num_states: int
    sigma_low: float
    sigma_high: float
    p_stay: float
    sigma_states: list[float]
    metrics: CalibrationMetrics
    calibrated_at: datetime


class RegimeResponse(BaseModel):
    timestamp: datetime
    regime_state: int = Field(..., description="Most probable state index (1-based)")
    regime_name: str = Field(..., description="Human-readable regime label")
    regime_probabilities: list[float]
    volatility_filtered: float
    volatility_forecast: float
    var_95: float
    transition_matrix: list[list[float]]


class VaRResponse(BaseModel):
    timestamp: datetime
    confidence: float
    var_value: float
    sigma_forecast: float
    z_alpha: float
    regime_probabilities: list[float]


class VolatilityForecastResponse(BaseModel):
    timestamp: datetime
    sigma_forecast: float
    sigma_filtered: float
    regime_probabilities: list[float]
    sigma_states: list[float]


class BacktestSummaryResponse(BaseModel):
    token: str
    num_observations: int
    var_alpha: float
    breach_count: int
    breach_rate: float
    kupiec_lr: Optional[float]
    kupiec_pvalue: Optional[float]
    kupiec_pass: bool
    christoffersen_lr: Optional[float]
    christoffersen_pvalue: Optional[float]
    christoffersen_pass: bool


class TailProbResponse(BaseModel):
    l1_threshold: float
    p1_day: float
    horizon_probs: dict[int, float]
    distribution: str


class RegimeStreamMessage(BaseModel):
    timestamp: datetime
    regime_state: int
    regime_name: str
    regime_probabilities: list[float]
    volatility_forecast: float
    var_95: float


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "INTERNAL_ERROR"


REGIME_NAMES: dict[int, str] = {
    1: "Very Low Vol",
    2: "Low Vol",
    3: "Normal",
    4: "High Vol",
    5: "Crisis",
}


def get_regime_name(state_idx: int, num_states: int) -> str:
    """Map 1-based state index to human-readable name."""
    if num_states in (4, 5, 6):
        return REGIME_NAMES.get(state_idx, f"State {state_idx}")
    return f"State {state_idx}/{num_states}"

