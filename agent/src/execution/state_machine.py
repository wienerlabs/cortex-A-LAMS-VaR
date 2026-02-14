from __future__ import annotations
"""
Execution State Machine.

Manages multi-step DeFi transaction execution.
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import asyncio
import structlog

logger = structlog.get_logger()


class ExecutionState(Enum):
    """States in the execution pipeline."""
    IDLE = auto()
    ANALYZING = auto()
    SIMULATING = auto()
    AWAITING_APPROVAL = auto()
    EXECUTING = auto()
    CONFIRMING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ExecutionContext:
    """Context for execution pipeline."""
    id: str
    strategy: str
    action: str
    params: dict[str, Any]
    state: ExecutionState = ExecutionState.IDLE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    simulation_result: dict[str, Any] | None = None
    tx_hash: str | None = None
    error: str | None = None
    gas_used: int = 0
    profit_realized: float = 0.0


class ExecutionStateMachine:
    """
    State machine for multi-step DeFi execution.
    
    Flow:
    1. IDLE -> ANALYZING: Analyze opportunity
    2. ANALYZING -> SIMULATING: Run Hardhat simulation
    3. SIMULATING -> AWAITING_APPROVAL: Wait for approval (if required)
    4. AWAITING_APPROVAL -> EXECUTING: Execute transaction
    5. EXECUTING -> CONFIRMING: Wait for confirmation
    6. CONFIRMING -> COMPLETED: Transaction confirmed
    
    Any state can transition to FAILED or CANCELLED.
    """
    
    # Valid state transitions
    TRANSITIONS = {
        ExecutionState.IDLE: [ExecutionState.ANALYZING, ExecutionState.CANCELLED],
        ExecutionState.ANALYZING: [ExecutionState.SIMULATING, ExecutionState.FAILED, ExecutionState.CANCELLED],
        ExecutionState.SIMULATING: [ExecutionState.AWAITING_APPROVAL, ExecutionState.EXECUTING, ExecutionState.FAILED, ExecutionState.CANCELLED],
        ExecutionState.AWAITING_APPROVAL: [ExecutionState.EXECUTING, ExecutionState.CANCELLED],
        ExecutionState.EXECUTING: [ExecutionState.CONFIRMING, ExecutionState.FAILED],
        ExecutionState.CONFIRMING: [ExecutionState.COMPLETED, ExecutionState.FAILED],
        ExecutionState.COMPLETED: [],
        ExecutionState.FAILED: [],
        ExecutionState.CANCELLED: [],
    }
    
    def __init__(
        self,
        require_approval: bool = True,
        max_retries: int = 3
    ):
        self.require_approval = require_approval
        self.max_retries = max_retries
        self.contexts: dict[str, ExecutionContext] = {}
        self.state_handlers: dict[ExecutionState, Callable] = {}
        self.logger = logger.bind(component="state_machine")
    
    def register_handler(
        self,
        state: ExecutionState,
        handler: Callable
    ) -> None:
        """Register a handler for a state."""
        self.state_handlers[state] = handler
    
    def create_execution(
        self,
        execution_id: str,
        strategy: str,
        action: str,
        params: dict[str, Any]
    ) -> ExecutionContext:
        """Create a new execution context."""
        context = ExecutionContext(
            id=execution_id,
            strategy=strategy,
            action=action,
            params=params
        )
        self.contexts[execution_id] = context
        self.logger.info(
            "Execution created",
            id=execution_id,
            strategy=strategy,
            action=action
        )
        return context
    
    def transition(
        self,
        execution_id: str,
        new_state: ExecutionState,
        **kwargs
    ) -> bool:
        """
        Transition execution to a new state.
        
        Returns:
            True if transition was successful
        """
        if execution_id not in self.contexts:
            self.logger.error("Execution not found", id=execution_id)
            return False
        
        context = self.contexts[execution_id]
        current_state = context.state
        
        # Check if transition is valid
        if new_state not in self.TRANSITIONS[current_state]:
            self.logger.error(
                "Invalid state transition",
                id=execution_id,
                from_state=current_state.name,
                to_state=new_state.name
            )
            return False
        
        # Update context
        context.state = new_state
        context.updated_at = datetime.utcnow()
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        self.logger.info(
            "State transition",
            id=execution_id,
            from_state=current_state.name,
            to_state=new_state.name
        )
        
        return True
    
    async def run(self, execution_id: str) -> ExecutionContext:
        """
        Run the execution pipeline.
        
        Returns:
            Final execution context
        """
        if execution_id not in self.contexts:
            raise ValueError(f"Execution not found: {execution_id}")
        
        context = self.contexts[execution_id]
        
        try:
            # Run through states
            await self._run_state(context, ExecutionState.ANALYZING)
            await self._run_state(context, ExecutionState.SIMULATING)
            
            if self.require_approval:
                self.transition(execution_id, ExecutionState.AWAITING_APPROVAL)
                # Wait for external approval
                return context
            
            await self._run_state(context, ExecutionState.EXECUTING)
            await self._run_state(context, ExecutionState.CONFIRMING)
            
            self.transition(execution_id, ExecutionState.COMPLETED)
            
        except Exception as e:
            self.transition(execution_id, ExecutionState.FAILED, error=str(e))
            self.logger.error("Execution failed", id=execution_id, error=str(e))

        return context

    async def _run_state(
        self,
        context: ExecutionContext,
        state: ExecutionState
    ) -> None:
        """Run handler for a state."""
        if not self.transition(context.id, state):
            raise RuntimeError(f"Failed to transition to {state.name}")

        if state in self.state_handlers:
            handler = self.state_handlers[state]
            await handler(context)

    def approve(self, execution_id: str) -> bool:
        """Approve an execution waiting for approval."""
        if execution_id not in self.contexts:
            return False

        context = self.contexts[execution_id]
        if context.state != ExecutionState.AWAITING_APPROVAL:
            return False

        return self.transition(execution_id, ExecutionState.EXECUTING)

    def cancel(self, execution_id: str) -> bool:
        """Cancel an execution."""
        if execution_id not in self.contexts:
            return False

        context = self.contexts[execution_id]
        if context.state in [ExecutionState.COMPLETED, ExecutionState.FAILED]:
            return False

        return self.transition(execution_id, ExecutionState.CANCELLED)

    def get_context(self, execution_id: str) -> ExecutionContext | None:
        """Get execution context."""
        return self.contexts.get(execution_id)

    def get_active_executions(self) -> list[ExecutionContext]:
        """Get all active (non-terminal) executions."""
        terminal_states = {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED
        }
        return [
            ctx for ctx in self.contexts.values()
            if ctx.state not in terminal_states
        ]
