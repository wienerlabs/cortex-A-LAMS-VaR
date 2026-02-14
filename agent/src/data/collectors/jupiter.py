"""
Jupiter collector for Solana DEX aggregation.

Jupiter is the #1 DEX aggregator on Solana:
- Aggregates liquidity from all major DEXs
- Provides best swap routes
- Real-time price quotes
- No additional fees (uses underlying DEX fees)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
import structlog

from .base import BaseCollector, CollectorConfig

logger = structlog.get_logger()


class JupiterCollector(BaseCollector):
    """
    Collector for Jupiter DEX aggregator on Solana.
    
    Uses Jupiter for:
    - Best swap routes across DEXs
    - Real-time price quotes
    - Slippage estimation
    - Route comparison (Raydium vs Orca vs others)
    """
    
    # Common token mints
    TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    }
    
    def __init__(self):
        self.quote_url = "https://quote-api.jup.ag/v6"
        self.price_url = "https://price.jup.ag/v6"
        
        config = CollectorConfig(
            name="jupiter",
            base_url=self.quote_url,
            api_key=None  # Jupiter doesn't require API key
        )
        super().__init__(config)
    
    async def fetch_latest(self) -> dict[str, Any]:
        """Fetch latest prices from Jupiter."""
        result = {
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
            "chain": "solana",
            "source": "jupiter",
            "prices": {}
        }
        
        # Get SOL price in USDC
        try:
            sol_price = await self.get_price(
                self.TOKENS["SOL"],
                self.TOKENS["USDC"]
            )
            result["prices"]["SOL_USDC"] = sol_price
        except Exception as e:
            self.logger.error("Failed to fetch SOL price", error=str(e))
        
        return result
    
    async def get_price(
        self,
        input_mint: str,
        output_mint: str = None
    ) -> dict[str, Any]:
        """
        Get token price from Jupiter Price API.
        
        Args:
            input_mint: Token mint address to price
            output_mint: Quote token (default: USDC)
        """
        output_mint = output_mint or self.TOKENS["USDC"]
        
        try:
            response = await self._make_request(
                "GET",
                f"{self.price_url}/price",
                params={
                    "ids": input_mint,
                    "vsToken": output_mint
                }
            )
            
            data = response.get("data", {}).get(input_mint, {})
            return {
                "price": data.get("price", 0),
                "mint": input_mint,
                "vs_token": output_mint
            }
            
        except Exception as e:
            self.logger.error("Failed to get price", error=str(e))
            return {"price": 0, "mint": input_mint}
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> dict[str, Any]:
        """
        Get swap quote from Jupiter.
        
        Args:
            input_mint: Input token mint
            output_mint: Output token mint
            amount: Amount in smallest units (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
        
        Returns:
            Quote with routes, output amount, and price impact
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.quote_url}/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": str(amount),
                    "slippageBps": slippage_bps,
                    "onlyDirectRoutes": False,
                    "asLegacyTransaction": False
                }
            )
            
            return {
                "input_mint": input_mint,
                "output_mint": output_mint,
                "in_amount": response.get("inAmount"),
                "out_amount": response.get("outAmount"),
                "price_impact_pct": response.get("priceImpactPct"),
                "route_plan": response.get("routePlan", []),
                "slippage_bps": slippage_bps,
                "other_amount_threshold": response.get("otherAmountThreshold")
            }
            
        except Exception as e:
            self.logger.error("Failed to get quote", error=str(e))
            return {}
    
    async def fetch_historical(
        self,
        start_time: datetime,  # noqa: ARG002
        end_time: datetime,  # noqa: ARG002
        interval: str = "5m"  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Jupiter doesn't provide historical data - use Birdeye."""
        _ = start_time, end_time, interval  # Unused - Jupiter doesn't support historical
        self.logger.warning("Historical data should use Birdeye, not Jupiter")
        return []

    def validate_response(self, response: dict[str, Any]) -> bool:
        """Validate Jupiter API response."""
        return bool(response) and "error" not in response

    async def compare_dex_routes(
        self,
        input_mint: str,
        output_mint: str,
        amount: int
    ) -> dict[str, Any]:
        """
        Compare routes across different DEXs.

        Returns best route from each DEX for arbitrage analysis.
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.quote_url}/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": str(amount),
                    "slippageBps": 50,
                    "onlyDirectRoutes": True  # Direct routes only
                }
            )

            route_plan = response.get("routePlan", [])

            # Extract DEX-specific data
            dex_routes = {}
            for route in route_plan:
                swap_info = route.get("swapInfo", {})
                amm_key = swap_info.get("ammKey", "unknown")
                label = swap_info.get("label", "Unknown DEX")

                # Determine DEX type from label
                dex_name = self._extract_dex_name(label)

                if dex_name not in dex_routes:
                    dex_routes[dex_name] = {
                        "label": label,
                        "amm_key": amm_key,
                        "in_amount": swap_info.get("inAmount"),
                        "out_amount": swap_info.get("outAmount"),
                        "fee_amount": swap_info.get("feeAmount"),
                        "fee_mint": swap_info.get("feeMint")
                    }

            return {
                "input_mint": input_mint,
                "output_mint": output_mint,
                "amount": amount,
                "dex_routes": dex_routes,
                "best_output": response.get("outAmount"),
                "price_impact_pct": response.get("priceImpactPct")
            }

        except Exception as e:
            self.logger.error("Failed to compare DEX routes", error=str(e))
            return {}

    def _extract_dex_name(self, label: str) -> str:
        """Extract standardized DEX name from Jupiter label."""
        label_lower = label.lower()

        if "raydium" in label_lower:
            return "raydium"
        elif "orca" in label_lower or "whirlpool" in label_lower:
            return "orca"
        elif "phoenix" in label_lower:
            return "phoenix"
        elif "meteora" in label_lower:
            return "meteora"
        elif "lifinity" in label_lower:
            return "lifinity"
        elif "openbook" in label_lower:
            return "openbook"
        else:
            return label.lower().replace(" ", "_")

    async def calculate_arbitrage_spread(
        self,
        token_a: str,
        token_b: str,
        amount: int
    ) -> dict[str, Any]:
        """
        Calculate potential arbitrage spread between DEXs.

        Compares direct routes from different DEXs to find spread.
        """
        routes = await self.compare_dex_routes(token_a, token_b, amount)
        dex_routes = routes.get("dex_routes", {})

        if len(dex_routes) < 2:
            return {"spread_pct": 0, "profitable": False}

        # Find best and worst routes
        outputs = []
        for dex, data in dex_routes.items():
            out = int(data.get("out_amount", 0))
            if out > 0:
                outputs.append({"dex": dex, "output": out})

        if len(outputs) < 2:
            return {"spread_pct": 0, "profitable": False}

        outputs.sort(key=lambda x: x["output"], reverse=True)
        best = outputs[0]
        worst = outputs[-1]

        spread_pct = ((best["output"] - worst["output"]) / worst["output"]) * 100

        return {
            "best_dex": best["dex"],
            "best_output": best["output"],
            "worst_dex": worst["dex"],
            "worst_output": worst["output"],
            "spread_pct": round(spread_pct, 4),
            "profitable": spread_pct > 0.5  # Profitable if > 0.5% spread
        }

