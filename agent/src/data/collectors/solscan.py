"""
Solscan collector for Solana blockchain explorer data.

Solscan provides:
- Block information
- Transaction details
- Account data
- Token information
- Fee statistics
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
import structlog

from .base import BaseCollector, CollectorConfig

logger = structlog.get_logger()


class SolscanCollector(BaseCollector):
    """
    Collector for Solana blockchain data via Solscan Pro API.
    
    Fetches:
    - Current block/slot information
    - Transaction fee statistics
    - Account transaction history
    - Token transfer events
    
    Similar role to Etherscan on Ethereum.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pro-api.solscan.io/v2.0"
        
        config = CollectorConfig(
            name="solscan",
            base_url=self.base_url,
            api_key=api_key
        )
        super().__init__(config)
        
        self.headers = {"token": api_key}
    
    async def _make_solscan_request(
        self,
        endpoint: str,
        params: dict | None = None
    ) -> dict[str, Any]:
        """Make request to Solscan API with proper headers."""
        url = f"{self.base_url}{endpoint}"
        
        response = await self.client.get(
            url,
            params=params or {},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def fetch_latest(self) -> dict[str, Any]:
        """Fetch latest Solana network statistics."""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "chain": "solana"
        }
        
        try:
            # Get chain info
            chain_info = await self._make_solscan_request("/chaininfo")
            
            if chain_info.get("success"):
                data = chain_info.get("data", {})
                result["chain_info"] = {
                    "slot": data.get("slot"),
                    "epoch": data.get("epoch"),
                    "block_height": data.get("blockHeight"),
                    "tps": data.get("tps"),
                    "sol_price": data.get("solPrice"),
                    "total_supply": data.get("totalSupply"),
                    "circulating_supply": data.get("circulatingSupply")
                }
                
                # Extract SOL price for cost calculations
                result["sol_price_usd"] = data.get("solPrice", 0)
                result["current_slot"] = data.get("slot", 0)
                
        except Exception as e:
            self.logger.error("Failed to fetch chain info", error=str(e))
        
        return result
    
    async def fetch_account_transactions(
        self,
        address: str,
        limit: int = 50,
        before_hash: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Fetch transaction history for an account.
        
        Args:
            address: Solana account address
            limit: Number of transactions (max 50)
            before_hash: Transaction signature for pagination
        """
        params = {
            "address": address,
            "limit": min(limit, 50)
        }
        if before_hash:
            params["before"] = before_hash
        
        try:
            response = await self._make_solscan_request(
                "/account/transactions",
                params=params
            )
            
            if response.get("success"):
                return response.get("data", [])
            return []
            
        except Exception as e:
            self.logger.error("Failed to fetch account transactions", error=str(e))
            return []
    
    async def fetch_transaction_detail(
        self,
        signature: str
    ) -> dict[str, Any]:
        """Get detailed information about a transaction."""
        try:
            response = await self._make_solscan_request(
                "/transaction",
                params={"tx": signature}
            )
            
            if response.get("success"):
                return response.get("data", {})
            return {}
            
        except Exception as e:
            self.logger.error("Failed to fetch transaction detail", error=str(e))
            return {}
    
    async def fetch_historical(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m"
    ) -> list[dict[str, Any]]:
        """Solscan doesn't provide historical OHLCV - use Birdeye."""
        self.logger.warning("Historical OHLCV should use Birdeye, not Solscan")
        return []

    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate Solscan API response."""
        return response.get("success", False)

    async def fetch_token_info(
        self,
        token_address: str
    ) -> dict[str, Any]:
        """Get token metadata and statistics."""
        try:
            response = await self._make_solscan_request(
                "/token/meta",
                params={"address": token_address}
            )

            if response.get("success"):
                data = response.get("data", {})
                return {
                    "address": token_address,
                    "symbol": data.get("symbol"),
                    "name": data.get("name"),
                    "decimals": data.get("decimals"),
                    "supply": data.get("supply"),
                    "holder_count": data.get("holder"),
                    "price_usd": data.get("priceUsdt")
                }
            return {}

        except Exception as e:
            self.logger.error("Failed to fetch token info", error=str(e))
            return {}

    async def fetch_defi_activities(
        self,
        address: str,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Fetch DeFi activities (swaps, LP, etc.) for an account.

        Useful for tracking DEX interactions.
        """
        try:
            response = await self._make_solscan_request(
                "/account/defi/activities",
                params={
                    "address": address,
                    "limit": min(limit, 50)
                }
            )

            if response.get("success"):
                return response.get("data", [])
            return []

        except Exception as e:
            self.logger.error("Failed to fetch DeFi activities", error=str(e))
            return []

    def estimate_transaction_cost(
        self,
        sol_price_usd: float,
        priority_fee_lamports: int = 50000,
        compute_units: int = 200000
    ) -> dict[str, float]:
        """
        Estimate transaction cost in SOL and USD.

        Solana transaction cost components:
        - Base fee: 5000 lamports (0.000005 SOL)
        - Priority fee: variable (for faster inclusion)
        - Compute units: variable based on tx complexity

        Args:
            sol_price_usd: Current SOL price
            priority_fee_lamports: Priority fee in lamports
            compute_units: Compute units for the transaction

        Returns:
            Dict with cost estimates
        """
        # Base transaction fee
        base_fee_lamports = 5000  # 0.000005 SOL

        # Total cost in lamports
        total_lamports = base_fee_lamports + priority_fee_lamports

        # Convert to SOL
        cost_sol = total_lamports / 1e9

        # Convert to USD
        cost_usd = cost_sol * sol_price_usd

        return {
            "base_fee_lamports": base_fee_lamports,
            "priority_fee_lamports": priority_fee_lamports,
            "total_lamports": total_lamports,
            "cost_sol": round(cost_sol, 9),
            "cost_usd": round(cost_usd, 6),
            "sol_price_usd": sol_price_usd
        }

    def is_fee_acceptable(
        self,
        priority_fee_lamports: int,
        max_fee_lamports: int = 500000  # 0.0005 SOL
    ) -> bool:
        """
        Check if current priority fee is acceptable.

        Solana fees are generally very low, but during congestion
        priority fees can spike.
        """
        return priority_fee_lamports <= max_fee_lamports

