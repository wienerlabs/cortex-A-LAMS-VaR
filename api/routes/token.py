"""Token info endpoint — fetches metadata from Birdeye for the Token Card UI."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models import TokenInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["token"])


@router.get("/token/info/{address}", response_model=TokenInfoResponse)
def get_token_info(address: str):
    """Fetch token metadata (name, symbol, logo, price, market cap, etc.)."""
    from cortex.data.solana import get_token_metadata

    try:
        data = get_token_metadata(address)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Token metadata fetch failed for %s", address)
        raise HTTPException(status_code=502, detail=f"Birdeye API error: {exc}")

    return TokenInfoResponse(**data, timestamp=datetime.now(timezone.utc))

