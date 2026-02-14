#!/usr/bin/env python3
"""
Solana API Connection Tests.

Tests connections to:
1. Helius RPC - Solana latest block
2. Birdeye - SOL/USDC price
3. Jupiter - SOL/USDC quote
4. Solscan - Recent transaction data
"""
import asyncio
import httpx
import json
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os

# API Keys from environment
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
SOLSCAN_API_KEY = os.getenv("SOLSCAN_API_KEY", "")

# Token addresses
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(success: bool, message: str, data: dict = None):
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n{status}: {message}")
    if data:
        print(f"Response Sample: {json.dumps(data, indent=2)[:500]}...")


async def test_helius_rpc():
    """Test Helius RPC - Get latest block."""
    print_header("1. HELIUS RPC - Latest Block")
    
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSlot",
        "params": []
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)
            
            print(f"Status Code: {response.status_code}")
            print(f"Rate Limit Headers: {dict((k, v) for k, v in response.headers.items() if 'limit' in k.lower() or 'remaining' in k.lower())}")
            
            if response.status_code == 200:
                data = response.json()
                slot = data.get("result", 0)
                print_result(True, f"Current Slot: {slot:,}", data)
                
                # Get block time
                block_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "getBlockTime",
                    "params": [slot]
                }
                block_response = await client.post(url, json=block_payload)
                if block_response.status_code == 200:
                    block_data = block_response.json()
                    block_time = block_data.get("result", 0)
                    print(f"Block Time: {datetime.fromtimestamp(block_time).isoformat()}")
                return True
            else:
                print_result(False, f"HTTP {response.status_code}", response.json())
                return False
    except Exception as e:
        print_result(False, str(e))
        return False


async def test_birdeye():
    """Test Birdeye API - SOL/USDC price."""
    print_header("2. BIRDEYE API - SOL/USDC Price")
    
    url = f"https://public-api.birdeye.so/defi/price?address={SOL_MINT}"
    headers = {
        "X-API-KEY": BIRDEYE_API_KEY,
        "x-chain": "solana"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            
            print(f"Status Code: {response.status_code}")
            print(f"Rate Limit Headers: {dict((k, v) for k, v in response.headers.items() if 'limit' in k.lower() or 'remaining' in k.lower() or 'x-' in k.lower())}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    price = data.get("data", {}).get("value", 0)
                    print_result(True, f"SOL Price: ${price:.2f}", data)
                    return True
                else:
                    print_result(False, "API returned success=false", data)
                    return False
            else:
                print_result(False, f"HTTP {response.status_code}", response.json() if response.text else {})
                return False
    except Exception as e:
        print_result(False, str(e))
        return False


async def test_jupiter():
    """Test Jupiter API - SOL/USDC quote using Ultra endpoint."""
    print_header("3. JUPITER API - SOL/USDC Quote (Ultra)")

    # 1 SOL = 1,000,000,000 lamports
    amount = 1_000_000_000

    # Jupiter Ultra Beta endpoint
    url = f"https://api.jup.ag/ultra/v1/order?inputMint={SOL_MINT}&outputMint={USDC_MINT}&amount={amount}"

    headers = {
        "Content-Type": "application/json"
    }
    if JUPITER_API_KEY:
        headers["x-api-key"] = JUPITER_API_KEY
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            
            print(f"Status Code: {response.status_code}")
            print(f"Rate Limit Headers: {dict((k, v) for k, v in response.headers.items() if 'limit' in k.lower() or 'remaining' in k.lower())}")
            
            if response.status_code == 200:
                data = response.json()
                out_amount = int(data.get("outAmount", 0)) / 1_000_000  # USDC has 6 decimals
                price_impact = data.get("priceImpactPct", "0")
                route_plan = data.get("routePlan", [])
                
                print_result(True, f"1 SOL = ${out_amount:.2f} USDC", {
                    "outAmount": out_amount,
                    "priceImpactPct": price_impact,
                    "routeSteps": len(route_plan)
                })
                
                # Show route details
                if route_plan:
                    print("\nRoute Plan:")
                    for i, step in enumerate(route_plan[:3]):
                        swap = step.get("swapInfo", {})
                        print(f"  Step {i+1}: {swap.get('label', 'Unknown')} - {swap.get('ammKey', '')[:20]}...")
                return True
            else:
                print_result(False, f"HTTP {response.status_code}", response.json() if response.text else {})
                return False
    except Exception as e:
        print_result(False, str(e))
        return False


async def test_helius_das():
    """Test Helius DAS API - Get asset info."""
    print_header("4. HELIUS DAS API - Asset Info")

    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

    # Get asset info for USDC token
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAsset",
        "params": {"id": USDC_MINT}
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    asset = data["result"]
                    name = asset.get("content", {}).get("metadata", {}).get("name", "Unknown")
                    symbol = asset.get("content", {}).get("metadata", {}).get("symbol", "Unknown")
                    supply = asset.get("token_info", {}).get("supply", 0)
                    decimals = asset.get("token_info", {}).get("decimals", 0)

                    print_result(True, f"Asset: {name} ({symbol})", {
                        "name": name,
                        "symbol": symbol,
                        "supply": supply,
                        "decimals": decimals
                    })
                    return True
                elif "error" in data:
                    print_result(False, data["error"].get("message", "Unknown error"), data)
                    return False
            else:
                print_result(False, f"HTTP {response.status_code}", response.json() if response.text else {})
                return False
    except Exception as e:
        print_result(False, str(e))
        return False


async def main():
    print("\n" + "🔗" * 30)
    print("  SOLANA API CONNECTION TESTS")
    print("🔗" * 30)
    print(f"\nTimestamp: {datetime.utcnow().isoformat()}Z")
    print(f"\nAPI Keys Configured:")
    print(f"  HELIUS:   {'✅' if HELIUS_API_KEY else '❌'} {HELIUS_API_KEY[:10]}..." if HELIUS_API_KEY else "  HELIUS:   ❌ Not set")
    print(f"  BIRDEYE:  {'✅' if BIRDEYE_API_KEY else '❌'} {BIRDEYE_API_KEY[:10]}..." if BIRDEYE_API_KEY else "  BIRDEYE:  ❌ Not set")
    print(f"  JUPITER:  {'✅' if JUPITER_API_KEY else '❌'} {JUPITER_API_KEY[:10]}..." if JUPITER_API_KEY else "  JUPITER:  ❌ Not set")
    print(f"  SOLSCAN:  {'✅' if SOLSCAN_API_KEY else '❌'} {SOLSCAN_API_KEY[:20]}..." if SOLSCAN_API_KEY else "  SOLSCAN:  ❌ Not set")
    
    results = []
    
    # Run tests
    results.append(("Helius RPC", await test_helius_rpc()))
    results.append(("Birdeye", await test_birdeye()))
    results.append(("Jupiter", await test_jupiter()))
    results.append(("Helius DAS", await test_helius_das()))
    
    # Summary
    print_header("SUMMARY")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

