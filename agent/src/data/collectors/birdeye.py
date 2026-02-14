"""
Birdeye collector for Solana DEX data.

Birdeye provides:
- Token prices across all Solana DEXs
- OHLCV historical data (1m to 1W intervals)
- Pool/pair information
- Volume and liquidity data
- Multi-DEX aggregated prices
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
import structlog

from .base import BaseCollector, CollectorConfig

logger = structlog.get_logger()

# Birdeye interval mapping
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W"
}


class BirdeyeCollector(BaseCollector):
    """
    Collector for Solana DEX data via Birdeye API.
    
    Primary source for:
    - Historical OHLCV data (365+ days)
    - Real-time token prices
    - DEX-specific pool data
    - Volume and liquidity metrics
    """
    
    # Popular Solana token addresses
    TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://public-api.birdeye.so"
        
        config = CollectorConfig(
            name="birdeye",
            base_url=self.base_url,
            api_key=api_key
        )
        super().__init__(config)
        
        self.headers = {
            "X-API-KEY": api_key,
            "x-chain": "solana"
        }
    
    async def _make_birdeye_request(
        self,
        endpoint: str,
        params: dict | None = None
    ) -> dict[str, Any]:
        """Make request to Birdeye API with proper headers."""
        url = f"{self.base_url}{endpoint}"
        
        # Override default request to include headers
        response = await self.client.get(
            url,
            params=params or {},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def fetch_latest(self) -> dict[str, Any]:
        """Fetch latest prices for monitored tokens."""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "chain": "solana",
            "prices": {}
        }
        
        for name, address in self.TOKENS.items():
            try:
                data = await self._make_birdeye_request(
                    "/defi/price",
                    params={"address": address}
                )
                if data.get("success"):
                    result["prices"][name] = {
                        "address": address,
                        "price_usd": data["data"]["value"],
                        "update_time": data["data"].get("updateUnixTime")
                    }
            except Exception as e:
                self.logger.error(f"Failed to fetch {name} price", error=str(e))
        
        return result
    
    async def fetch_ohlcv(
        self,
        token_address: str,
        interval: str = "5m",
        time_from: int | None = None,
        time_to: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Fetch OHLCV data for a token.
        
        Args:
            token_address: Solana token mint address
            interval: Time interval (1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W)
            time_from: Unix timestamp start
            time_to: Unix timestamp end
        """
        birdeye_interval = INTERVAL_MAP.get(interval, "5m")
        
        # Default to last 24 hours
        if time_to is None:
            time_to = int(datetime.utcnow().timestamp())
        if time_from is None:
            time_from = time_to - 86400  # 24 hours
        
        try:
            data = await self._make_birdeye_request(
                "/defi/ohlcv",
                params={
                    "address": token_address,
                    "type": birdeye_interval,
                    "time_from": time_from,
                    "time_to": time_to
                }
            )
            
            if data.get("success") and data.get("data"):
                return data["data"].get("items", [])
            return []
            
        except Exception as e:
            self.logger.error("Failed to fetch OHLCV", error=str(e))
            return []
    
    async def fetch_historical(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m"
    ) -> list[dict[str, Any]]:
        """Fetch historical OHLCV data for SOL/USDC."""
        return await self.fetch_ohlcv(
            token_address=self.TOKENS["SOL"],
            interval=interval,
            time_from=int(start_time.timestamp()),
            time_to=int(end_time.timestamp())
        )
    
    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate Birdeye API response."""
        return response.get("success", False)

    async def fetch_token_price_multi_dex(
        self,
        token_address: str
    ) -> dict[str, Any]:
        """
        Fetch token price from multiple DEXs.

        Returns prices from Raydium, Orca, and other DEXs.
        """
        try:
            data = await self._make_birdeye_request(
                "/defi/multi_price",
                params={"list_address": token_address}
            )

            if data.get("success"):
                return data.get("data", {})
            return {}

        except Exception as e:
            self.logger.error("Failed to fetch multi-DEX prices", error=str(e))
            return {}

    async def fetch_pair_overview(
        self,
        pair_address: str
    ) -> dict[str, Any]:
        """
        Fetch detailed pair/pool information.

        Returns:
            Liquidity, volume, price, and trade data for a pool.
        """
        try:
            data = await self._make_birdeye_request(
                "/defi/pair_overview",
                params={"address": pair_address}
            )

            if data.get("success"):
                return data.get("data", {})
            return {}

        except Exception as e:
            self.logger.error("Failed to fetch pair overview", error=str(e))
            return {}

    async def fetch_trades(
        self,
        pair_address: str,
        limit: int = 100,
        tx_type: str = "all"  # "all", "buy", "sell"
    ) -> list[dict[str, Any]]:
        """
        Fetch recent trades for a pair.

        Args:
            pair_address: DEX pool address
            limit: Number of trades to fetch (max 100)
            tx_type: Filter by trade type
        """
        try:
            data = await self._make_birdeye_request(
                "/defi/txs/pair",
                params={
                    "address": pair_address,
                    "limit": min(limit, 100),
                    "tx_type": tx_type
                }
            )

            if data.get("success"):
                return data.get("data", {}).get("items", [])
            return []

        except Exception as e:
            self.logger.error("Failed to fetch trades", error=str(e))
            return []

    async def fetch_pool_liquidity(
        self,
        pool_address: str
    ) -> dict[str, float]:
        """Get current liquidity for a pool."""
        overview = await self.fetch_pair_overview(pool_address)

        return {
            "liquidity_usd": overview.get("liquidity", 0),
            "volume_24h": overview.get("v24hUSD", 0),
            "price": overview.get("price", 0),
            "price_change_24h": overview.get("v24hChangePercent", 0)
        }

