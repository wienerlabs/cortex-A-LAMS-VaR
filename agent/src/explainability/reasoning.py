from __future__ import annotations
"""
Reasoning Metadata Generator.

Creates structured reasoning for agent decisions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import structlog

logger = structlog.get_logger()


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    action: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReasoningMetadata:
    """Complete reasoning metadata for a decision."""
    decision_id: str
    strategy: str
    action: str
    confidence: float
    steps: list[ReasoningStep] = field(default_factory=list)
    shap_explanation: dict[str, Any] | None = None
    market_context: dict[str, Any] = field(default_factory=dict)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_step(
        self,
        action: str,
        description: str,
        data: dict[str, Any] | None = None
    ) -> None:
        """Add a reasoning step."""
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            action=action,
            description=description,
            data=data or {}
        )
        self.steps.append(step)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "strategy": self.strategy,
            "action": self.action,
            "confidence": self.confidence,
            "steps": [
                {
                    "step_number": s.step_number,
                    "action": s.action,
                    "description": s.description,
                    "data": s.data,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in self.steps
            ],
            "shap_explanation": self.shap_explanation,
            "market_context": self.market_context,
            "risk_assessment": self.risk_assessment,
            "created_at": self.created_at.isoformat()
        }


class ReasoningGenerator:
    """
    Generates structured reasoning for agent decisions.
    
    Creates audit trail of:
    - Data collection steps
    - Feature computation
    - Model inference
    - Decision logic
    - Risk assessment
    """
    
    def __init__(self):
        self.logger = logger.bind(component="reasoning_generator")
    
    def create_reasoning(
        self,
        decision_id: str,
        strategy: str,
        action: str,
        confidence: float
    ) -> ReasoningMetadata:
        """Create a new reasoning metadata object."""
        return ReasoningMetadata(
            decision_id=decision_id,
            strategy=strategy,
            action=action,
            confidence=confidence
        )
    
    def add_data_collection_step(
        self,
        reasoning: ReasoningMetadata,
        sources: list[str],
        data_summary: dict[str, Any]
    ) -> None:
        """Add data collection step."""
        reasoning.add_step(
            action="data_collection",
            description=f"Collected data from {len(sources)} sources",
            data={
                "sources": sources,
                "summary": data_summary
            }
        )
    
    def add_feature_computation_step(
        self,
        reasoning: ReasoningMetadata,
        feature_count: int,
        key_features: dict[str, float]
    ) -> None:
        """Add feature computation step."""
        reasoning.add_step(
            action="feature_computation",
            description=f"Computed {feature_count} features",
            data={
                "feature_count": feature_count,
                "key_features": key_features
            }
        )
    
    def add_inference_step(
        self,
        reasoning: ReasoningMetadata,
        model_name: str,
        prediction: float,
        shap_explanation: dict[str, Any] | None = None
    ) -> None:
        """Add model inference step."""
        reasoning.add_step(
            action="model_inference",
            description=f"Ran inference with {model_name} model",
            data={
                "model": model_name,
                "prediction": prediction
            }
        )
        
        if shap_explanation:
            reasoning.shap_explanation = shap_explanation
    
    def add_risk_assessment_step(
        self,
        reasoning: ReasoningMetadata,
        risk_score: float,
        risk_factors: dict[str, Any]
    ) -> None:
        """Add risk assessment step."""
        reasoning.add_step(
            action="risk_assessment",
            description=f"Assessed risk: {risk_score:.2f}",
            data={
                "risk_score": risk_score,
                "factors": risk_factors
            }
        )
        
        reasoning.risk_assessment = {
            "score": risk_score,
            "factors": risk_factors
        }

    def add_decision_step(
        self,
        reasoning: ReasoningMetadata,
        decision: str,
        rationale: str
    ) -> None:
        """Add final decision step."""
        reasoning.add_step(
            action="decision",
            description=rationale,
            data={
                "decision": decision,
                "confidence": reasoning.confidence
            }
        )

    def set_market_context(
        self,
        reasoning: ReasoningMetadata,
        gas_price: float,
        eth_price: float,
        market_conditions: dict[str, Any]
    ) -> None:
        """Set market context."""
        reasoning.market_context = {
            "gas_price_gwei": gas_price,
            "eth_price_usd": eth_price,
            "conditions": market_conditions,
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_summary(self, reasoning: ReasoningMetadata) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Decision: {reasoning.action}",
            f"Strategy: {reasoning.strategy}",
            f"Confidence: {reasoning.confidence:.1%}",
            "",
            "Reasoning Chain:"
        ]

        for step in reasoning.steps:
            lines.append(f"  {step.step_number}. {step.action}: {step.description}")

        if reasoning.risk_assessment:
            lines.append("")
            lines.append(f"Risk Score: {reasoning.risk_assessment.get('score', 'N/A')}")

        return "\n".join(lines)
