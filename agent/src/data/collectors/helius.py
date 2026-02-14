"""
Helius collector for Solana blockchain data.

Helius provides:
- Enhanced RPC with DAS (Digital Asset Standard) API
- Transaction history and parsing
- Webhooks for real-time events
- NFT and token metadata
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
import structlog

from .base import BaseCollector, CollectorConfig

logger = structlog.get_logger()


class HeliusCollector(BaseCollector):
    """
    Collector for Solana data via Helius API.
    
    Fetches:
    - Transaction history for accounts
    - Parsed transaction data
    - Token balances and metadata
    - DeFi swap events
    """
    
    def __init__(self, api_key: str, rpc_url: str | None = None):
        self.api_key = api_key
        self.rpc_url = rpc_url or f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.api_base = "https://api.helius.xyz/v0"
        
        config = CollectorConfig(
            name="helius",
            base_url=self.api_base,
            api_key=api_key
        )
        super().__init__(config)
    
    async def fetch_latest(self) -> dict[str, Any]:
        """Fetch latest Solana network data."""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "chain": "solana"
        }
        
        # Get recent transactions for monitored pools
        try:
            # Get slot info via RPC
            slot_response = await self._make_request(
                "POST",
                self.rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSlot",
                    "params": []
                }
            )
            result["current_slot"] = slot_response.get("result", 0)
            
            # Get recent priority fees
            fees_response = await self._make_request(
                "POST",
                self.rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getRecentPrioritizationFees",
                    "params": []
                }
            )
            fees = fees_response.get("result", [])
            if fees:
                avg_fee = sum(f.get("prioritizationFee", 0) for f in fees) / len(fees)
                result["avg_priority_fee_lamports"] = avg_fee
                result["priority_fee_sol"] = avg_fee / 1e9
            
        except Exception as e:
            self.logger.error("Failed to fetch slot info", error=str(e))
        
        return result
    
    async def fetch_transaction_history(
        self,
        address: str,
        before: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Fetch parsed transaction history for an address.
        
        Args:
            address: Solana address (base58)
            before: Signature to start before (for pagination)
            limit: Number of transactions to fetch
        """
        params = {"api-key": self.api_key}
        url = f"{self.api_base}/addresses/{address}/transactions"
        
        query_params = {"limit": limit}
        if before:
            query_params["before"] = before
        
        try:
            response = await self._make_request(
                "GET",
                url,
                params={**params, **query_params}
            )
            return response if isinstance(response, list) else []
        except Exception as e:
            self.logger.error("Failed to fetch transaction history", error=str(e))
            return []
    
    async def fetch_historical(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m"
    ) -> list[dict[str, Any]]:
        """Fetch historical data - delegated to Birdeye for OHLCV."""
        # Helius is mainly for transactions, not historical OHLCV
        self.logger.warning("Historical OHLCV should use Birdeye, not Helius")
        return []
    
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate Helius API response."""
        if not response:
            return False
        if "error" in response:
            self.logger.error("Helius API error", error=response["error"])
            return False
        return True
    
    async def get_swap_events(
        self,
        pool_address: str,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get recent swap events for a DEX pool."""
        transactions = await self.fetch_transaction_history(pool_address, limit=limit)
        
        swaps = []
        for tx in transactions:
            # Filter for swap transactions
            if tx.get("type") == "SWAP":
                swaps.append({
                    "signature": tx.get("signature"),
                    "timestamp": tx.get("timestamp"),
                    "source": tx.get("source"),  # DEX name
                    "fee_payer": tx.get("feePayer"),
                    "token_transfers": tx.get("tokenTransfers", []),
                    "native_transfers": tx.get("nativeTransfers", []),
                })
        
        return swaps

