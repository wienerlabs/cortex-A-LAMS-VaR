"""DX-Research Task 5: Ising Cascade Detection.

DX Terminal finding: When all 36K agents herd in the same direction, cascading
failures follow within minutes. The Ising model from statistical physics provides
the right framework for detecting this:

  - Spins (+1 / -1) = agent signals (bullish / bearish)
  - Coupling constant J = how strongly agents influence each other
  - External field h = market-level bias (regime, Hawkes intensity)
  - Magnetization M = |mean spin| → net alignment / herding measure
  - Temperature T = volatility / disorder → low T = strong herding

When magnetization approaches 1.0 (all aligned), the system is near a phase
transition — a cascade/contagion event is imminent.

Three input layers:
  1. Stigmergy consensus: agent-level spin alignment
  2. Hawkes intensity: event clustering / contagion speed
  3. Regime state: environmental susceptibility (crisis = low temperature)

Output: cascade_score 0–100 for Guardian integration + debate evidence.
"""
from __future__ import annotations

__all__ = [
    "CascadeResult",
    "detect_cascade",
    "get_cascade_score",
]

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from cortex.config import (
    ISING_CASCADE_COUPLING_STRENGTH,
    ISING_CASCADE_CRITICAL_THRESHOLD,
    ISING_CASCADE_ENABLED,
)

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Result of Ising cascade detection."""
    magnetization: float         # |M| ∈ [0, 1] — net alignment of agents
    cascade_score: float         # 0–100 Guardian-compatible score
    cascade_risk: str            # low / medium / high / critical
    herding_direction: str       # bullish / bearish / neutral
    effective_temperature: float  # T_eff — lower = more susceptible
    susceptibility: float        # χ = dM/dh — how responsive to perturbation
    components: dict[str, float] = field(default_factory=dict)
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnetization": round(self.magnetization, 4),
            "cascade_score": round(self.cascade_score, 2),
            "cascade_risk": self.cascade_risk,
            "herding_direction": self.herding_direction,
            "effective_temperature": round(self.effective_temperature, 4),
            "susceptibility": round(self.susceptibility, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "ts": self.ts,
        }


def _compute_magnetization(
    stigmergy_conviction: float,
    stigmergy_direction: str,
    num_sources: int,
    swarm_active: bool,
) -> tuple[float, str]:
    """Compute agent magnetization from stigmergy consensus.

    Magnetization M ∈ [-1, +1] where:
      +1 = all agents bullish (aligned spins up)
      -1 = all agents bearish (aligned spins down)
       0 = no consensus (paramagnetic / disordered)
    """
    if num_sources == 0:
        return 0.0, "neutral"

    # Base magnetization from conviction
    m = stigmergy_conviction

    # Source count factor: more sources aligned = stronger magnetization
    # Scaled logarithmically to avoid linear explosion
    source_factor = min(1.0, math.log1p(num_sources) / math.log1p(10))
    m *= source_factor

    # Swarm amplification: when swarm is active, coupling intensifies
    if swarm_active:
        m = min(1.0, m * ISING_CASCADE_COUPLING_STRENGTH)

    direction = stigmergy_direction if stigmergy_direction != "neutral" else "neutral"
    return min(1.0, m), direction


def _compute_effective_temperature(
    regime: int,
    max_regimes: int = 5,
    hawkes_intensity_ratio: float = 1.0,
) -> float:
    """Compute effective temperature from market conditions.

    Low temperature → strong coupling → easier cascade formation.
    High temperature → disorder → agents act independently.

    regime=0 (calm) → high T, regime=4 (crisis) → low T
    High Hawkes intensity → lower T (contagion amplifies coupling)
    """
    # Regime contribution: crisis regime lowers temperature
    regime_normalized = regime / max(max_regimes - 1, 1)  # 0.0 (calm) → 1.0 (crisis)
    t_regime = 1.0 - 0.6 * regime_normalized  # 1.0 → 0.4

    # Hawkes intensity: elevated intensity reduces temperature
    # intensity_ratio = λ(t)/μ — typically 1.0 (baseline), can spike to 3+
    hawkes_factor = max(0.3, 1.0 / max(hawkes_intensity_ratio, 0.5))
    t_hawkes = hawkes_factor

    # Combined temperature (geometric mean for multiplicative interaction)
    t_eff = math.sqrt(t_regime * t_hawkes)
    return max(0.1, min(2.0, t_eff))


def _compute_susceptibility(magnetization: float, temperature: float) -> float:
    """Compute magnetic susceptibility χ — how responsive the system is.

    Near critical point (M → 1, T → T_c), susceptibility diverges.
    χ = (1 - M²) / T  (mean-field approximation)
    """
    m_sq = magnetization ** 2
    chi = (1.0 - m_sq) / max(temperature, 0.01)

    # Cap susceptibility to avoid divergence
    return min(100.0, chi)


def _cascade_score(
    magnetization: float,
    temperature: float,
    susceptibility: float,
) -> float:
    """Compute cascade risk score 0–100 from Ising parameters.

    Score formula:
      base = magnetization × 60 (herding alignment: max 60 points)
      temp_bonus = (1 - T_eff) × 25 (environmental susceptibility: max 25 points)
      chi_bonus = min(χ / 10, 1) × 15 (near-criticality: max 15 points)
    """
    base = magnetization * 60.0
    temp_bonus = max(0.0, (1.0 - temperature)) * 25.0
    chi_bonus = min(susceptibility / 10.0, 1.0) * 15.0
    return min(100.0, base + temp_bonus + chi_bonus)


def _risk_level(score: float) -> str:
    """Map cascade score to risk level."""
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def detect_cascade(
    stigmergy_conviction: float = 0.0,
    stigmergy_direction: str = "neutral",
    num_sources: int = 0,
    swarm_active: bool = False,
    regime: int = 0,
    max_regimes: int = 5,
    hawkes_intensity_ratio: float = 1.0,
) -> CascadeResult:
    """Detect cascade/herding risk using Ising model framework.

    Args:
        stigmergy_conviction: Consensus conviction from pheromone board [0, 1]
        stigmergy_direction: Consensus direction (bullish/bearish/neutral)
        num_sources: Number of unique agent signal sources
        swarm_active: Whether swarm threshold is exceeded
        regime: Current MSM regime index (0=calm, K-1=crisis)
        max_regimes: Total number of MSM regime states
        hawkes_intensity_ratio: λ(t)/μ from Hawkes model (1.0 = baseline)

    Returns:
        CascadeResult with cascade_score, magnetization, temperature, etc.
    """
    magnetization, direction = _compute_magnetization(
        stigmergy_conviction, stigmergy_direction, num_sources, swarm_active,
    )

    temperature = _compute_effective_temperature(
        regime, max_regimes, hawkes_intensity_ratio,
    )

    susceptibility = _compute_susceptibility(magnetization, temperature)
    score = _cascade_score(magnetization, temperature, susceptibility)
    risk = _risk_level(score)

    return CascadeResult(
        magnetization=magnetization,
        cascade_score=score,
        cascade_risk=risk,
        herding_direction=direction,
        effective_temperature=temperature,
        susceptibility=susceptibility,
        components={
            "magnetization_contribution": magnetization * 60.0,
            "temperature_contribution": max(0.0, (1.0 - temperature)) * 25.0,
            "susceptibility_contribution": min(susceptibility / 10.0, 1.0) * 15.0,
            "stigmergy_conviction": stigmergy_conviction,
            "num_sources": float(num_sources),
            "regime": float(regime),
            "hawkes_intensity_ratio": hawkes_intensity_ratio,
        },
    )


def get_cascade_score(
    token: str,
    regime: int = 0,
    hawkes_intensity_ratio: float = 1.0,
) -> CascadeResult:
    """Convenience: compute cascade score using live stigmergy data.

    Pulls consensus from the global pheromone board for the given token
    and combines with regime + Hawkes data for full Ising assessment.
    """
    if not ISING_CASCADE_ENABLED:
        return CascadeResult(
            magnetization=0.0,
            cascade_score=0.0,
            cascade_risk="low",
            herding_direction="neutral",
            effective_temperature=1.0,
            susceptibility=1.0,
        )

    # Pull stigmergy consensus
    try:
        from cortex.stigmergy import get_consensus
        consensus = get_consensus(token)
        return detect_cascade(
            stigmergy_conviction=consensus.conviction,
            stigmergy_direction=consensus.direction,
            num_sources=consensus.num_sources,
            swarm_active=consensus.swarm_active,
            regime=regime,
            hawkes_intensity_ratio=hawkes_intensity_ratio,
        )
    except Exception:
        logger.debug("Ising cascade detection failed for %s", token, exc_info=True)
        return CascadeResult(
            magnetization=0.0,
            cascade_score=0.0,
            cascade_risk="low",
            herding_direction="neutral",
            effective_temperature=1.0,
            susceptibility=1.0,
        )
