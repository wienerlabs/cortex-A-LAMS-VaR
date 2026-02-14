from __future__ import annotations
"""
Strategy Selector - Chooses best strategy based on market conditions.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import structlog

from .onnx_runtime import ONNXInference
from ..data.storage import FeatureCache
from ..config import settings

logger = structlog.get_logger()


@dataclass
class StrategyDecision:
    """Decision output from strategy selector."""
    strategy: str
    action: str
    confidence: float
    expected_profit: float
    gas_estimate: float
    reasoning: dict[str, Any]
    should_execute: bool


class StrategySelector:
    """
    Selects the best DeFi strategy based on current market conditions.
    
    Evaluates:
    - Arbitrage opportunities
    - Lending rate differentials
    - LP provision opportunities
    
    Returns the strategy with highest expected risk-adjusted return.
    """
    
    def __init__(
        self,
        arbitrage_model: ONNXInference | None = None,
        lending_model: ONNXInference | None = None,
        lp_model: ONNXInference | None = None,
        cache: FeatureCache | None = None
    ):
        self.arbitrage_model = arbitrage_model
        self.lending_model = lending_model
        self.lp_model = lp_model
        self.cache = cache
        self.logger = logger.bind(component="strategy_selector")
    
    async def evaluate_all(
        self,
        features: dict[str, pd.DataFrame]
    ) -> list[StrategyDecision]:
        """
        Evaluate all strategies and return ranked decisions.
        
        Args:
            features: Dict of strategy -> features DataFrame
            
        Returns:
            List of StrategyDecision sorted by expected profit
        """
        decisions = []
        
        # Evaluate arbitrage
        if "arbitrage" in features and self.arbitrage_model:
            decision = await self._evaluate_arbitrage(features["arbitrage"])
            if decision:
                decisions.append(decision)
        
        # Evaluate lending
        if "lending" in features and self.lending_model:
            decision = await self._evaluate_lending(features["lending"])
            if decision:
                decisions.append(decision)
        
        # Evaluate LP
        if "lp" in features and self.lp_model:
            decision = await self._evaluate_lp(features["lp"])
            if decision:
                decisions.append(decision)
        
        # Sort by expected profit (risk-adjusted)
        decisions.sort(key=lambda d: d.expected_profit * d.confidence, reverse=True)
        
        return decisions
    
    async def select_best(
        self,
        features: dict[str, pd.DataFrame],
        min_confidence: float = 0.6
    ) -> StrategyDecision | None:
        """
        Select the best strategy to execute.
        
        Args:
            features: Dict of strategy -> features
            min_confidence: Minimum confidence threshold
            
        Returns:
            Best StrategyDecision or None if no good opportunity
        """
        decisions = await self.evaluate_all(features)
        
        if not decisions:
            return None
        
        # Filter by confidence
        valid = [d for d in decisions if d.confidence >= min_confidence]
        
        if not valid:
            self.logger.info(
                "No strategies meet confidence threshold",
                threshold=min_confidence,
                best_confidence=decisions[0].confidence if decisions else 0
            )
            return None
        
        best = valid[0]
        self.logger.info(
            "Selected strategy",
            strategy=best.strategy,
            action=best.action,
            confidence=best.confidence,
            expected_profit=best.expected_profit
        )
        
        return best
    
    async def _evaluate_arbitrage(
        self,
        features: pd.DataFrame
    ) -> StrategyDecision | None:
        """Evaluate arbitrage opportunity on Solana DEXs."""
        if self.arbitrage_model is None:
            return None

        try:
            # Get prediction
            X = features.values.astype(np.float32)
            probs = self.arbitrage_model.predict_proba(X)

            # Get probability of profitable trade
            prob = probs[0][1] if len(probs.shape) > 1 else probs[0]

            # Get spread from features
            spread = features.get("spread_abs", pd.Series([0])).iloc[0]

            # Estimate profit for Solana
            trade_size = 10000  # $10k trade
            gross_profit = spread * trade_size

            # Solana gas costs are negligible (~0.000055 SOL = ~$0.01)
            # For $10k trade this is 0.0001% - effectively zero
            gas_cost = 0.01  # ~$0.01 per Solana transaction
            expected_profit = (gross_profit - gas_cost) * prob

            return StrategyDecision(
                strategy="arbitrage",
                action="execute_swap",
                confidence=float(prob),
                expected_profit=expected_profit,
                gas_estimate=gas_cost,
                reasoning={
                    "spread": spread,
                    "gross_profit": gross_profit,
                    "solana_tx_fee_usd": gas_cost,
                    "probability": prob
                },
                should_execute=prob > 0.7 and expected_profit > 10
            )
        except Exception as e:
            self.logger.error("Arbitrage evaluation failed", error=str(e))
            return None

    async def _evaluate_lending(
        self,
        features: pd.DataFrame
    ) -> StrategyDecision | None:
        """Evaluate lending opportunity."""
        if self.lending_model is None:
            return None

        try:
            X = features.values.astype(np.float32)
            probs = self.lending_model.predict_proba(X)

            # Get best protocol (0=MarginFi, 1=Kamino, 2=Solend, 3=Hold)
            best_class = np.argmax(probs[0])
            confidence = probs[0][best_class]

            protocols = {0: "marginfi", 1: "kamino", 2: "solend", 3: "hold"}
            protocol = protocols[best_class]

            # Get APY differential
            apy_diff = features.get("supply_apy_diff", pd.Series([0])).iloc[0]

            # Estimate profit (annual, for $100k position)
            position_size = 100000
            expected_profit = abs(apy_diff) * position_size / 100 * confidence

            return StrategyDecision(
                strategy="lending",
                action=f"switch_to_{protocol}" if protocol != "hold" else "hold",
                confidence=float(confidence),
                expected_profit=expected_profit / 365,  # Daily
                gas_estimate=50,  # Approximate gas cost
                reasoning={
                    "best_protocol": protocol,
                    "apy_diff": apy_diff,
                    "annual_profit": expected_profit
                },
                should_execute=protocol != "hold" and abs(apy_diff) > 0.5
            )
        except Exception as e:
            self.logger.error("Lending evaluation failed", error=str(e))
            return None

    async def _evaluate_lp(
        self,
        features: pd.DataFrame
    ) -> StrategyDecision | None:
        """Evaluate LP provision opportunity."""
        if self.lp_model is None:
            return None

        try:
            X = features.values.astype(np.float32)
            predicted_apy = self.lp_model.predict(X)[0]

            # Get current APY
            current_apy = features.get("total_apy", pd.Series([0])).iloc[0]

            # Confidence based on prediction vs current
            apy_improvement = predicted_apy - current_apy
            confidence = min(0.9, max(0.3, 0.5 + apy_improvement / 10))

            # Estimate profit (daily, for $50k position)
            position_size = 50000
            expected_profit = predicted_apy * position_size / 100 / 365

            return StrategyDecision(
                strategy="lp_provision",
                action="provide_liquidity" if apy_improvement > 0 else "hold",
                confidence=float(confidence),
                expected_profit=expected_profit,
                gas_estimate=100,  # LP operations are more expensive
                reasoning={
                    "predicted_apy": predicted_apy,
                    "current_apy": current_apy,
                    "apy_improvement": apy_improvement
                },
                should_execute=apy_improvement > 1.0
            )
        except Exception as e:
            self.logger.error("LP evaluation failed", error=str(e))
            return None
