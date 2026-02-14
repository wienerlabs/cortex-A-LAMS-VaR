"""
DefiLlama Collector - Reliable free API for DeFi data.

DefiLlama provides:
- TVL data for all protocols
- Yields/APY for lending and LP
- Historical prices
- No API key required!
"""
import aiohttp
import structlog
from datetime import datetime, timedelta
from typing import Any

from .base import BaseCollector

logger = structlog.get_logger()


class DefiLlamaCollector(BaseCollector):
    """
    Collector for DefiLlama API.
    
    Free, reliable, no API key needed.
    https://defillama.com/docs/api
    """
    
    BASE_URL = "https://api.llama.fi"
    YIELDS_URL = "https://yields.llama.fi"
    COINS_URL = "https://coins.llama.fi"
    
    def __init__(self):
        super().__init__()
        self.session = None
        self.logger = logger.bind(collector="defillama")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_latest(self) -> dict[str, Any]:
        """Fetch latest DeFi data."""
        return await self.fetch_all_data()
    
    async def fetch_all_data(self) -> dict[str, Any]:
        """Fetch comprehensive DeFi data."""
        import asyncio
        
        results = await asyncio.gather(
            self.fetch_yields(),
            self.fetch_prices(),
            self.fetch_tvl_protocols(),
            return_exceptions=True
        )
        
        return {
            "yields": results[0] if not isinstance(results[0], Exception) else [],
            "prices": results[1] if not isinstance(results[1], Exception) else {},
            "protocols": results[2] if not isinstance(results[2], Exception) else []
        }
    
    async def fetch_yields(self) -> list[dict]:
        """Fetch lending and LP yields."""
        session = await self._get_session()
        url = f"{self.YIELDS_URL}/pools"
        
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                pools = data.get("data", [])
                
                # Filter for major protocols
                major_protocols = [
                    "aave-v3", "compound-v3", "curve-dex",
                    "uniswap-v3", "lido", "convex-finance"
                ]
                
                filtered = [
                    p for p in pools 
                    if p.get("project") in major_protocols
                    and p.get("chain") == "Ethereum"
                ]
                
                self.logger.info(f"Fetched {len(filtered)} yield pools")
                return filtered
            else:
                self.logger.error(f"Failed to fetch yields: {resp.status}")
                return []
    
    async def fetch_prices(self, coins: list[str] = None) -> dict:
        """Fetch current token prices."""
        if coins is None:
            coins = [
                "ethereum:0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "ethereum:0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                "ethereum:0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "ethereum:0x6B175474E89094C44Da98b954EescdeCB5BE33D8",  # DAI
            ]
        
        session = await self._get_session()
        coins_str = ",".join(coins)
        url = f"{self.COINS_URL}/prices/current/{coins_str}"
        
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("coins", {})
            return {}
    
    async def fetch_tvl_protocols(self) -> list[dict]:
        """Fetch TVL for major protocols."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/protocols"
        
        async with session.get(url) as resp:
            if resp.status == 200:
                protocols = await resp.json()
                
                # Filter top protocols
                major = ["aave", "compound", "curve", "uniswap", "lido"]
                filtered = [
                    {"name": p["name"], "tvl": p["tvl"], "category": p.get("category")}
                    for p in protocols
                    if any(m in p["name"].lower() for m in major)
                ][:20]
                
                return filtered
            return []
    
    async def fetch_historical_prices(
        self, 
        coin: str,
        start_timestamp: int,
        search_width: int = 3600
    ) -> dict:
        """Fetch historical price for a coin."""
        session = await self._get_session()
        url = f"{self.COINS_URL}/prices/historical/{start_timestamp}/{coin}"
        
        params = {"searchWidth": search_width}
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}

