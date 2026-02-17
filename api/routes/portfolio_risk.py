"""Portfolio risk management endpoints — positions, drawdown, limits."""

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.models import (
    DrawdownResponse,
    PortfolioLimitsResponse,
    PositionItem,
    PositionsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio-risk"])


class UpdatePositionRequest(BaseModel):
    token: str
    size_usd: float = Field(..., gt=0)
    direction: str = Field(..., pattern="^(long|short)$")
    entry_price: float = Field(default=0.0, ge=0)


class ClosePositionRequest(BaseModel):
    token: str
    pnl: float


class SetPortfolioValueRequest(BaseModel):
    value: float = Field(..., gt=0)


@router.get("/portfolio/positions", response_model=PositionsResponse, summary="List positions")
def get_positions():
    """Return all open positions with total exposure and portfolio value."""
    from cortex.portfolio_risk import get_portfolio_value, get_positions as _get

    positions = _get()
    total = sum(p["size_usd"] for p in positions)
    return PositionsResponse(
        positions=[PositionItem(**p) for p in positions],
        total_exposure_usd=round(total, 2),
        portfolio_value=get_portfolio_value(),
        timestamp=time.time(),
    )


@router.post("/portfolio/positions", summary="Update position")
def update_position(req: UpdatePositionRequest):
    """Add or update a position in the portfolio tracker."""
    from cortex.portfolio_risk import update_position as _update
    _update(token=req.token, size_usd=req.size_usd, direction=req.direction, entry_price=req.entry_price)
    return {"status": "updated", "token": req.token}


@router.post("/portfolio/positions/close", summary="Close position")
def close_position(req: ClosePositionRequest):
    """Close a position and record realized PnL."""
    from cortex.portfolio_risk import close_position as _close
    _close(token=req.token, pnl=req.pnl)
    return {"status": "closed", "token": req.token, "pnl": req.pnl}


@router.post("/portfolio/value", summary="Set portfolio value")
def set_portfolio_value(req: SetPortfolioValueRequest):
    """Set the total portfolio value used for drawdown and exposure calculations."""
    from cortex.portfolio_risk import set_portfolio_value as _set
    _set(req.value)
    return {"status": "set", "portfolio_value": req.value}


@router.get("/portfolio/drawdown", response_model=DrawdownResponse, summary="Get drawdown")
def get_drawdown():
    """Return daily and weekly drawdown metrics with limit breach status."""
    from cortex.portfolio_risk import get_drawdown as _get
    return DrawdownResponse(**_get())


@router.get("/portfolio/limits", response_model=PortfolioLimitsResponse, summary="Check portfolio limits")
def get_limits(token: str = "SOL"):
    """Check drawdown limits and correlation exposure constraints for a token."""
    from cortex.portfolio_risk import check_limits
    return check_limits(token)

