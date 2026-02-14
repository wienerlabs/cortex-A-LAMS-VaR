#!/usr/bin/env python3
"""
Collect additional pool data from DeFiLlama and alternative sources.
"""
import asyncio
import httpx
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "lp_rebalancer" / "historical"


async def collect_defillama_protocol_tvl():
    """Collect protocol-level TVL history from DeFiLlama."""
    print("\n📊 Collecting DeFiLlama Protocol TVL History...")
    
    protocols = ["raydium", "orca", "meteora"]
    results = []
    
    async with httpx.AsyncClient(timeout=60) as client:
        for protocol in protocols:
            print(f"   Fetching {protocol}...")
            url = f"https://api.llama.fi/protocol/{protocol}"
            
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    tvl = data.get("tvl", [])
                    
                    # Get last 30 days
                    recent_tvl = tvl[-30*24:] if len(tvl) > 30*24 else tvl
                    
                    results.append({
                        "protocol": protocol,
                        "name": data.get("name"),
                        "category": data.get("category"),
                        "current_tvl": data.get("currentChainTvls", {}).get("Solana", 0),
                        "tvl_history_count": len(recent_tvl),
                        "tvl_history": recent_tvl,
                    })
                    print(f"      ✅ {len(recent_tvl)} TVL data points")
                else:
                    print(f"      ❌ HTTP {resp.status_code}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
            
            await asyncio.sleep(1)
    
    # Save
    with open(DATA_DIR / "defillama_protocol_tvl.json", "w") as f:
        json.dump(results, f)
    
    return results


async def collect_birdeye_top_pools():
    """Get current top pools from Birdeye to find more addresses."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    
    print("\n📊 Collecting Birdeye Top Pairs...")
    
    async with httpx.AsyncClient(timeout=60) as client:
        headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
        
        # Get top pairs by volume
        url = "https://public-api.birdeye.so/defi/v2/pairs?sort_by=volume24hUSD&sort_type=desc&limit=50"
        
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get("data", {}).get("items", [])
                
                print(f"   ✅ Found {len(pairs)} top pairs")
                
                # Save
                with open(DATA_DIR / "birdeye_top_pairs.json", "w") as f:
                    json.dump(pairs, f)
                
                # Print top 10
                print("\n   Top 10 by Volume:")
                for i, p in enumerate(pairs[:10]):
                    addr = p.get("address", "")[:8]
                    name = p.get("name", "Unknown")
                    vol = p.get("volume24hUSD", 0)
                    print(f"      {i+1}. {name}: ${vol/1e6:.2f}M vol (addr: {addr}...)")
                
                return pairs
            else:
                print(f"   ❌ HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return []


async def try_collect_missing_pools():
    """Try to collect OHLCV for pools that failed earlier."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    
    # Load existing pool data
    with open(DATA_DIR / "pool_ohlcv_30d.json") as f:
        existing = json.load(f)
    
    # Find pools with 0 candles
    missing = [p for p in existing if p["candles"] == 0]
    
    if not missing:
        print("\n✅ No missing pools!")
        return
    
    print(f"\n📊 Retrying {len(missing)} missing pools with alternative method...")
    
    NOW = int(datetime.now(timezone.utc).timestamp())
    DAYS_7_AGO = NOW - (7 * 24 * 3600)  # Try 7 days instead
    
    async with httpx.AsyncClient(timeout=60) as client:
        headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
        
        updated = []
        for p in missing:
            addr = p["pool_address"]
            name = p["pool_name"]
            
            print(f"   Trying {name} (7 day history)...")
            
            url = f"https://public-api.birdeye.so/defi/ohlcv/pair?address={addr}&type=1H&time_from={DAYS_7_AGO}&time_to={NOW}"
            
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", {}).get("items", [])
                    
                    if items:
                        p["candles"] = len(items)
                        p["data"] = items
                        print(f"      ✅ Got {len(items)} candles (7d)")
                    else:
                        print(f"      ⚠️ Still no data")
                else:
                    print(f"      ❌ HTTP {resp.status_code}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
            
            updated.append(p)
            await asyncio.sleep(2)
        
        # Update existing data
        for p in existing:
            if p["candles"] == 0:
                for u in updated:
                    if u["pool_address"] == p["pool_address"]:
                        p.update(u)
                        break
        
        # Save updated
        with open(DATA_DIR / "pool_ohlcv_30d.json", "w") as f:
            json.dump(existing, f)


async def main():
    print("\n" + "=" * 60)
    print("  ADDITIONAL DATA COLLECTION")
    print("=" * 60)
    
    await collect_defillama_protocol_tvl()
    await collect_birdeye_top_pools()
    await try_collect_missing_pools()
    
    print("\n✅ Additional collection complete!")


if __name__ == "__main__":
    asyncio.run(main())

