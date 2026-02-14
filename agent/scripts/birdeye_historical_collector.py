#!/usr/bin/env python3
"""
Birdeye Historical Pool Data Collector

Collects 30 days of hourly OHLCV data for top pools.
Rate limited: 2s between requests, 10s on 429, max 3 retries.
"""
import asyncio
import httpx
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
DATA_DIR = Path(__file__).parent.parent / "data" / "lp_rebalancer" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Time range
NOW = int(datetime.now(timezone.utc).timestamp())
DAYS_30_AGO = NOW - (30 * 24 * 3600)

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between requests
RETRY_DELAY = 10.0   # seconds on 429
MAX_RETRIES = 3

# Priority pools (address, name, dex)
PRIORITY_POOLS = [
    ("58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2", "SOL-USDC", "raydium"),
    ("7qbRF6YsyGuLUVs6Y1sfC93Ym6eVhfhxQzgJCfXr7HU4", "SOL-USDC", "orca"),
    ("CXoVvnGFNmySLqd8sXXo8K1xi5gcPKwS3xWs3HSj7Dtw", "WSOL-pippin", "raydium"),
    ("9tQxBEKfBfU1n7TD4DKvPMyxZ6xqLEKu7xcLwpLzUDxM", "SOL-cbBTC", "meteora"),
    ("6P6rAjsktP5TtrScgL3CJgE1aoVD9LY6HjnUzQjsyEJN", "JTO-JitoSOL", "orca"),
    ("HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ", "SOL-USDT", "raydium"),
    ("Czfq3xZZDmsdGdUyrNLtRhGc47cXcZtLG4crLCEgK8kw", "SOL-RAY", "raydium"),
    ("2QdhepnKRTLjjSqPL1PtKNwqrUkoLee5Gqs8bvZhRdMv", "BONK-SOL", "raydium"),
    ("5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1", "RAY-USDC", "raydium"),
    ("AVs9TA4nWDzfPJE9procWNHNx4baoJDmAzxKT9Cg1VLR", "JUP-SOL", "raydium"),
    ("FpCMFDFGYotvufJ7HrFHsWEiiQCGbkLCtwHiDnh7o28Q", "WIF-SOL", "raydium"),
    ("2wT8p2vmSjsR6sBHH3YwP7VL99HLhPPz2KqBEUE5YwMz", "PYTH-SOL", "orca"),
    ("7XawhbbxtsRcQA8KTkHT9f9nc6d69UwqCDh6U5EEbEmX", "mSOL-SOL", "orca"),
    ("B32UuhPSp6srSBbRTh4qZNjkegsehY9qXTwQgnPWYMZy", "stSOL-SOL", "orca"),
    ("3ne4mWqdYuNiYrYZC9TrA3FcfuFdErghH97vNPbjicr1", "ETH-SOL", "raydium"),
    ("8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj", "BTC-SOL", "raydium"),
    ("DdSvDp6AZ4RuY1Y2X39vy5Fo1c3vsKrCfRzKg3xj3Y3G", "ORCA-SOL", "orca"),
    ("HWHvQhFmJB3NUcu1aihKmrKegfVxBEHzwVX6yZCKEsi1", "MNGO-SOL", "orca"),
    ("EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U", "SRM-SOL", "raydium"),
    ("6UmmUiYoBjSrhakAobJw8BvkmJtDVxaeBtbt7rxWo1mg", "COPE-SOL", "raydium"),
]


class BirdeyeHistoricalCollector:
    """Collects historical pool data from Birdeye API."""
    
    def __init__(self):
        self.headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
        self.stats = {"success": 0, "failed": 0, "rate_limited": 0, "total_candles": 0}
        self.results: List[Dict] = []
    
    async def fetch_with_retry(self, client: httpx.AsyncClient, url: str) -> Tuple[bool, Optional[Dict]]:
        """Fetch URL with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url, headers=self.headers, timeout=30)
                
                if resp.status_code == 200:
                    return True, resp.json()
                elif resp.status_code == 429:
                    self.stats["rate_limited"] += 1
                    print(f"      ⚠️ Rate limited (429), waiting {RETRY_DELAY}s... (attempt {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"      ❌ HTTP {resp.status_code}: {resp.text[:100]}")
                    return False, None
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(REQUEST_DELAY)
        
        return False, None
    
    async def collect_pool_ohlcv(self, client: httpx.AsyncClient, pool_address: str, 
                                  pool_name: str, dex: str) -> Optional[Dict]:
        """Collect 30-day hourly OHLCV for a pool."""
        url = f"https://public-api.birdeye.so/defi/ohlcv/pair?address={pool_address}&type=1H&time_from={DAYS_30_AGO}&time_to={NOW}"
        
        success, data = await self.fetch_with_retry(client, url)
        
        if success and data:
            items = data.get("data", {}).get("items", [])
            self.stats["total_candles"] += len(items)
            
            return {
                "pool_address": pool_address,
                "pool_name": pool_name,
                "dex": dex,
                "candles": len(items),
                "data": items,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        return None
    
    async def collect_token_ohlcv(self, client: httpx.AsyncClient, token_mint: str, 
                                   token_symbol: str) -> Optional[Dict]:
        """Collect 30-day hourly OHLCV for a token."""
        url = f"https://public-api.birdeye.so/defi/ohlcv?address={token_mint}&type=1H&time_from={DAYS_30_AGO}&time_to={NOW}"
        
        success, data = await self.fetch_with_retry(client, url)
        
        if success and data:
            items = data.get("data", {}).get("items", [])
            self.stats["total_candles"] += len(items)
            
            return {
                "token_mint": token_mint,
                "token_symbol": token_symbol,
                "candles": len(items),
                "data": items,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
        return None
    
    def print_progress(self, current: int, total: int, pool_name: str, candles: int):
        """Print progress update."""
        pct = current / total * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"   [{bar}] {current}/{total} - {pool_name}: {candles} candles")
    
    def print_stats(self):
        """Print collection statistics."""
        print("\n" + "=" * 60)
        print("  📊 COLLECTION STATISTICS")
        print("=" * 60)
        print(f"   ✅ Successful: {self.stats['success']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⚠️ Rate limited events: {self.stats['rate_limited']}")
        print(f"   📈 Total candles collected: {self.stats['total_candles']:,}")
        print("=" * 60)

    async def run(self):
        """Run the full collection process."""
        print("\n" + "🔄" * 30)
        print("  BIRDEYE HISTORICAL DATA COLLECTOR")
        print("🔄" * 30)
        print(f"\nStarted: {datetime.now(timezone.utc).isoformat()}")
        print(f"Target: 30 days hourly data for {len(PRIORITY_POOLS)} pools")
        print(f"Rate limit: {REQUEST_DELAY}s between requests")

        async with httpx.AsyncClient(timeout=60) as client:
            # 1. Collect token prices first (for IL calculation)
            print("\n" + "─" * 50)
            print("  PHASE 1: Token Price History")
            print("─" * 50)

            tokens = [
                ("So11111111111111111111111111111111111111112", "SOL"),
                ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "USDC"),
                ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", "USDT"),
            ]

            token_results = []
            for i, (mint, symbol) in enumerate(tokens):
                print(f"\n   [{i+1}/{len(tokens)}] Fetching {symbol} price history...")
                result = await self.collect_token_ohlcv(client, mint, symbol)

                if result:
                    token_results.append(result)
                    self.stats["success"] += 1
                    print(f"      ✅ {result['candles']} hourly candles")
                else:
                    self.stats["failed"] += 1
                    print(f"      ❌ Failed to fetch {symbol}")

                await asyncio.sleep(REQUEST_DELAY)

            # Save token data
            token_file = DATA_DIR / "token_prices_30d.json"
            with open(token_file, "w") as f:
                json.dump(token_results, f)
            print(f"\n   💾 Saved to {token_file}")

            # 2. Collect pool OHLCV
            print("\n" + "─" * 50)
            print("  PHASE 2: Pool OHLCV History")
            print("─" * 50)

            pool_results = []
            for i, (address, name, dex) in enumerate(PRIORITY_POOLS):
                print(f"\n   [{i+1}/{len(PRIORITY_POOLS)}] {name} ({dex})...")
                result = await self.collect_pool_ohlcv(client, address, name, dex)

                if result:
                    pool_results.append(result)
                    self.stats["success"] += 1
                    self.print_progress(i+1, len(PRIORITY_POOLS), name, result["candles"])
                else:
                    self.stats["failed"] += 1
                    print(f"      ❌ Failed - trying alternative...")

                    # Try pair info endpoint as fallback
                    alt_url = f"https://public-api.birdeye.so/defi/v2/pair?address={address}"
                    alt_success, alt_data = await self.fetch_with_retry(client, alt_url)
                    if alt_success:
                        print(f"      ⚠️ Got current data only (no history)")
                        pool_results.append({
                            "pool_address": address,
                            "pool_name": name,
                            "dex": dex,
                            "candles": 0,
                            "current_data": alt_data.get("data", {}),
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        })

                # Progress report every 5 pools
                if (i + 1) % 5 == 0:
                    print(f"\n   📊 Progress: {i+1}/{len(PRIORITY_POOLS)} pools processed")
                    print(f"      Success: {self.stats['success']}, Failed: {self.stats['failed']}")

                await asyncio.sleep(REQUEST_DELAY)

            # Save pool data
            pool_file = DATA_DIR / "pool_ohlcv_30d.json"
            with open(pool_file, "w") as f:
                json.dump(pool_results, f)
            print(f"\n   💾 Saved to {pool_file}")

        # Final stats
        self.print_stats()
        self.results = {"tokens": token_results, "pools": pool_results}

        return self.results


async def main():
    """Main entry point."""
    if not BIRDEYE_API_KEY:
        print("❌ BIRDEYE_API_KEY not set in environment")
        return

    collector = BirdeyeHistoricalCollector()
    results = await collector.run()

    print("\n✅ Collection complete!")
    print(f"   Token files: {len(results.get('tokens', []))}")
    print(f"   Pool files: {len(results.get('pools', []))}")


if __name__ == "__main__":
    asyncio.run(main())

