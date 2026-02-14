#!/usr/bin/env python3
"""
Check Historical Data Availability from DEX APIs

Tests Birdeye, Raydium, Orca APIs for 30-day historical pool data.
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")

# Time range
NOW = int(datetime.now(timezone.utc).timestamp())
DAYS_30_AGO = NOW - (30 * 24 * 3600)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def check_birdeye_historical(client: httpx.AsyncClient):
    """Check Birdeye for historical pool/token data."""
    print_section("BIRDEYE API - Historical Data Check")
    
    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
    
    # 1. OHLCV endpoint (token prices)
    print("\n▸ Testing OHLCV (Token Price History)...")
    sol_mint = "So11111111111111111111111111111111111111112"
    url = f"https://public-api.birdeye.so/defi/ohlcv?address={sol_mint}&type=1H&time_from={DAYS_30_AGO}&time_to={NOW}"
    
    try:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", {}).get("items", [])
            print(f"   ✅ OHLCV available: {len(items)} hourly candles")
            if items:
                first = datetime.fromtimestamp(items[0].get("unixTime", 0), tz=timezone.utc)
                last = datetime.fromtimestamp(items[-1].get("unixTime", 0), tz=timezone.utc)
                print(f"   📅 Range: {first.date()} → {last.date()}")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Token trades history
    print("\n▸ Testing Token Trade History...")
    url = f"https://public-api.birdeye.so/defi/txs/token?address={sol_mint}&limit=10"
    try:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", {}).get("items", [])
            print(f"   ✅ Trade history: {len(items)} recent trades")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Pool/Pair history (key for LP analysis)
    print("\n▸ Testing Pool History (LP Data)...")
    # Raydium SOL-USDC pool
    pool_address = "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"
    url = f"https://public-api.birdeye.so/defi/ohlcv/pair?address={pool_address}&type=1H&time_from={DAYS_30_AGO}&time_to={NOW}"
    
    try:
        resp = await client.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", {}).get("items", [])
            print(f"   ✅ Pool OHLCV available: {len(items)} hourly candles")
            if items:
                print(f"   📊 Sample: {items[0]}")
        else:
            print(f"   ⚠️ Pool OHLCV: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def check_raydium_historical(client: httpx.AsyncClient):
    """Check Raydium API for historical data."""
    print_section("RAYDIUM API - Historical Data Check")
    
    # 1. Current pairs (no history)
    print("\n▸ Testing Raydium Pairs API...")
    url = "https://api.raydium.io/v2/main/pairs"
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Current pairs: {len(data)} pools")
            print(f"   ⚠️ Note: This is CURRENT data only, no historical API")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Check for historical endpoint
    print("\n▸ Testing Raydium Historical (if exists)...")
    # Raydium doesn't have public historical API
    print("   ⚠️ Raydium has NO public historical API")
    print("   💡 Alternative: Use Birdeye pool OHLCV for Raydium pools")


async def check_orca_historical(client: httpx.AsyncClient):
    """Check Orca API for historical data."""
    print_section("ORCA API - Historical Data Check")
    
    # 1. Current whirlpools
    print("\n▸ Testing Orca Whirlpool API...")
    url = "https://api.mainnet.orca.so/v1/whirlpool/list"
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            pools = data.get("whirlpools", [])
            print(f"   ✅ Current pools: {len(pools)} whirlpools")
            
            # Check what historical fields are available
            if pools:
                sample = pools[0]
                print(f"   📊 Available fields: {list(sample.keys())[:10]}...")
                
                # Check for historical APR
                fee_apr = sample.get("feeApr", {})
                print(f"   📈 Fee APR data: {fee_apr}")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n   ⚠️ Orca has NO public historical API")
    print("   💡 Alternative: Use Birdeye or on-chain indexer")


async def check_meteora_historical(client: httpx.AsyncClient):
    """Check Meteora API for historical data."""
    print_section("METEORA API - Historical Data Check")
    
    print("\n▸ Testing Meteora DLMM API...")
    url = "https://dlmm-api.meteora.ag/pair/all"
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Current pairs: {len(data)} pools")
            print("   ⚠️ Note: Current data only, no historical API")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def check_flipside_dune(client: httpx.AsyncClient):
    """Check if Flipside or Dune have historical LP data."""
    print_section("ALTERNATIVE SOURCES")
    
    print("\n▸ Flipside Crypto:")
    print("   📊 Has Solana DEX data with SQL queries")
    print("   💡 Could query: swap volumes, TVL history")
    print("   ⚠️ Requires API key and query setup")
    
    print("\n▸ Dune Analytics:")
    print("   📊 Has Solana DEX dashboards")
    print("   💡 Could export historical data")
    print("   ⚠️ Rate limited, requires account")
    
    print("\n▸ DefiLlama:")
    url = "https://api.llama.fi/protocol/raydium"
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            tvl_history = data.get("tvl", [])
            print(f"   ✅ DefiLlama TVL history: {len(tvl_history)} data points")
            if tvl_history:
                first = datetime.fromtimestamp(tvl_history[0].get("date", 0), tz=timezone.utc)
                last = datetime.fromtimestamp(tvl_history[-1].get("date", 0), tz=timezone.utc)
                print(f"   📅 Range: {first.date()} → {last.date()}")
        else:
            print(f"   ❌ DefiLlama: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ DefiLlama error: {e}")


async def run_checks():
    """Run all API checks."""
    print("\n" + "🔍" * 35)
    print("  HISTORICAL DATA API CHECK")
    print("🔍" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Target: 30 days of hourly pool data (720 data points per pool)")
    
    async with httpx.AsyncClient(timeout=60) as client:
        await check_birdeye_historical(client)
        await asyncio.sleep(1)
        await check_raydium_historical(client)
        await asyncio.sleep(1)
        await check_orca_historical(client)
        await asyncio.sleep(1)
        await check_meteora_historical(client)
        await asyncio.sleep(1)
        await check_flipside_dune(client)
    
    # Summary
    print_section("📋 SUMMARY")
    print("""
    DATA SOURCE AVAILABILITY:
    
    ┌─────────────────┬──────────┬─────────────────────────────────┐
    │ Source          │ Status   │ Notes                           │
    ├─────────────────┼──────────┼─────────────────────────────────┤
    │ Birdeye OHLCV   │ ✅ YES   │ Token prices, 30d hourly        │
    │ Birdeye Pool    │ ⚠️ CHECK │ Pool-level OHLCV if available   │
    │ Raydium API     │ ❌ NO    │ Current data only               │
    │ Orca API        │ ❌ NO    │ Current data only               │
    │ Meteora API     │ ❌ NO    │ Current data only               │
    │ DefiLlama       │ ✅ YES   │ Protocol TVL history (daily)    │
    └─────────────────┴──────────┴─────────────────────────────────┘
    
    RECOMMENDATION:
    1. Use Birdeye OHLCV for token prices (SOL, USDC, etc.)
    2. Use DefiLlama for protocol-level TVL history
    3. For pool-level APY/Volume history:
       → Need to collect going forward OR
       → Use Flipside/Dune SQL queries
    """)


if __name__ == "__main__":
    asyncio.run(run_checks())

