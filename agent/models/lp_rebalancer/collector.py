"""
Pool Data Collector for LP Rebalancer

Collects hourly snapshots of pool metrics for training data.
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "lp_rebalancer"


class PoolDataCollector:
    """Collects and stores hourly pool data."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_file = self.data_dir / "pool_snapshots.jsonl"
    
    async def collect_raydium_pools(self, client: httpx.AsyncClient) -> List[Dict]:
        """Collect data from Raydium pools."""
        try:
            url = "https://api.raydium.io/v2/main/pairs"
            resp = await client.get(url, timeout=30)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            pools = []
            
            for p in data[:20]:  # Top 20 by liquidity
                pools.append({
                    "dex": "raydium",
                    "pool_address": p.get("ammId", ""),
                    "pool_name": p.get("name", ""),
                    "tvl": float(p.get("liquidity", 0)),
                    "volume_24h": float(p.get("volume24h", 0)),
                    "apy": float(p.get("apr7d", 0)) if p.get("apr7d") else 0,
                    "fee_tier": 0.25,
                    "token_a_symbol": p.get("name", "").split("-")[0] if "-" in p.get("name", "") else "",
                    "token_b_symbol": p.get("name", "").split("-")[1] if "-" in p.get("name", "") else "",
                })
            
            return pools
        except Exception as e:
            print(f"Raydium error: {e}")
            return []
    
    async def collect_orca_pools(self, client: httpx.AsyncClient) -> List[Dict]:
        """Collect data from Orca pools."""
        try:
            url = "https://api.mainnet.orca.so/v1/whirlpool/list"
            resp = await client.get(url, timeout=30)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            whirlpools = sorted(
                data.get("whirlpools", []),
                key=lambda x: float(x.get("tvl", 0)),
                reverse=True
            )[:20]
            
            pools = []
            for p in whirlpools:
                token_a = p.get("tokenA", {})
                token_b = p.get("tokenB", {})
                
                pools.append({
                    "dex": "orca",
                    "pool_address": p.get("address", ""),
                    "pool_name": f"{token_a.get('symbol', '')}-{token_b.get('symbol', '')}",
                    "tvl": float(p.get("tvl", 0)),
                    "volume_24h": float(p.get("volume", {}).get("day", 0)),
                    "apy": float(p.get("feeApr", {}).get("week", 0)) * 100,
                    "fee_tier": float(p.get("tickSpacing", 64)) / 100 * 0.01,
                    "token_a_symbol": token_a.get("symbol", ""),
                    "token_b_symbol": token_b.get("symbol", ""),
                    "token_a_price": float(p.get("tokenA", {}).get("price", 0)),
                    "token_b_price": float(p.get("tokenB", {}).get("price", 0)),
                })
            
            return pools
        except Exception as e:
            print(f"Orca error: {e}")
            return []
    
    async def collect_meteora_pools(self, client: httpx.AsyncClient) -> List[Dict]:
        """Collect data from Meteora pools."""
        try:
            url = "https://dlmm-api.meteora.ag/pair/all"
            resp = await client.get(url, timeout=30)
            if resp.status_code != 200:
                return []
            
            data = sorted(resp.json(), key=lambda x: float(x.get("liquidity", 0)), reverse=True)[:20]
            
            pools = []
            for p in data:
                name = p.get("name", "")
                parts = name.split("-") if "-" in name else ["", ""]
                
                pools.append({
                    "dex": "meteora",
                    "pool_address": p.get("address", ""),
                    "pool_name": name,
                    "tvl": float(p.get("liquidity", 0)),
                    "volume_24h": float(p.get("trade_volume_24h", 0)),
                    "apy": float(p.get("apr", 0)) if p.get("apr") else 0,
                    "fee_tier": float(p.get("base_fee_percentage", 0.25)),
                    "token_a_symbol": parts[0].strip(),
                    "token_b_symbol": parts[1].strip() if len(parts) > 1 else "",
                })
            
            return pools
        except Exception as e:
            print(f"Meteora error: {e}")
            return []
    
    async def collect_market_context(self, client: httpx.AsyncClient) -> Dict:
        """Collect SOL and BTC prices for market context."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbols=[\"SOLUSDC\",\"BTCUSDC\"]"
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "sol_price": float(next((x["price"] for x in data if x["symbol"] == "SOLUSDC"), 0)),
                    "btc_price": float(next((x["price"] for x in data if x["symbol"] == "BTCUSDC"), 0)),
                }
        except:
            pass
        return {"sol_price": 0, "btc_price": 0}
    
    async def collect_snapshot(self) -> Dict:
        """Collect a complete snapshot of all pools."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        async with httpx.AsyncClient(timeout=60) as client:
            # Collect from all DEXes concurrently
            results = await asyncio.gather(
                self.collect_raydium_pools(client),
                self.collect_orca_pools(client),
                self.collect_meteora_pools(client),
                self.collect_market_context(client),
            )
            
            all_pools = results[0] + results[1] + results[2]
            market = results[3]
            
            # Add timestamp and market context
            for pool in all_pools:
                pool["timestamp"] = timestamp
                pool["sol_price"] = market["sol_price"]
                pool["btc_price"] = market["btc_price"]
            
            return {"timestamp": timestamp, "pools": all_pools, "pool_count": len(all_pools)}

