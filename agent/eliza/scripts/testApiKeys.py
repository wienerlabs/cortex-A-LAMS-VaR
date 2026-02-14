#!/usr/bin/env python3
"""Test all API keys from .env file"""

import os
import httpx
import asyncio
from pathlib import Path

# Load .env
env_file = Path(__file__).parent.parent / '.env'
with open(env_file) as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            key, _, value = line.strip().partition('=')
            os.environ[key] = value

async def test_apis():
    print("=" * 60)
    print("API KEY TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    async with httpx.AsyncClient(timeout=15) as client:
        
        # 1. Birdeye
        print("\n1. BIRDEYE API")
        try:
            resp = await client.get(
                "https://public-api.birdeye.so/defi/price?address=So11111111111111111111111111111111111111112",
                headers={"X-API-KEY": os.environ.get("BIRDEYE_API_KEY", ""), "x-chain": "solana"}
            )
            data = resp.json()
            if data.get("success"):
                print(f"   ✅ ÇALIŞIYOR - SOL: ${data['data']['value']:.2f}")
                results["Birdeye"] = "✅"
            else:
                print(f"   ❌ HATA: {data.get('message', 'Unknown')}")
                results["Birdeye"] = "❌"
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            results["Birdeye"] = "❌"
        
        # 2. Helius
        print("\n2. HELIUS API")
        try:
            resp = await client.post(
                f"https://mainnet.helius-rpc.com/?api-key={os.environ.get('HELIUS_API_KEY', '')}",
                json={"jsonrpc": "2.0", "id": 1, "method": "getSlot"}
            )
            data = resp.json()
            if "result" in data:
                print(f"   ✅ ÇALIŞIYOR - Slot: {data['result']}")
                results["Helius"] = "✅"
            else:
                print(f"   ❌ HATA: {data.get('error', 'Unknown')}")
                results["Helius"] = "❌"
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            results["Helius"] = "❌"
        
        # 3. Jupiter (no key required for quote)
        print("\n3. JUPITER API")
        try:
            resp = await client.get(
                "https://quote-api.jup.ag/v6/quote?inputMint=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&outputMint=So11111111111111111111111111111111111111112&amount=1000000"
            )
            data = resp.json()
            if "outAmount" in data:
                out_sol = int(data["outAmount"]) / 1e9
                print(f"   ✅ ÇALIŞIYOR - 1 USDC = {out_sol:.6f} SOL")
                results["Jupiter"] = "✅"
            else:
                print(f"   ❌ HATA: {data}")
                results["Jupiter"] = "❌"
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            results["Jupiter"] = "❌"
        
        # 4. Solscan
        print("\n4. SOLSCAN API")
        try:
            resp = await client.get(
                "https://pro-api.solscan.io/v2.0/account/So11111111111111111111111111111111111111112",
                headers={"token": os.environ.get("SOLSCAN_API_KEY", "")}
            )
            data = resp.json()
            if data.get("success"):
                print(f"   ✅ ÇALIŞIYOR")
                results["Solscan"] = "✅"
            else:
                print(f"   ❌ HATA: {str(data)[:100]}")
                results["Solscan"] = "❌"
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            results["Solscan"] = "❌"
        
        # 5. CoinGecko
        print("\n5. COINGECKO API")
        try:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
                headers={"x-cg-demo-api-key": os.environ.get("COINGECKO_API_KEY", "")}
            )
            data = resp.json()
            if "solana" in data:
                print(f"   ✅ ÇALIŞIYOR - SOL: ${data['solana']['usd']}")
                results["CoinGecko"] = "✅"
            else:
                print(f"   ❌ HATA: {data}")
                results["CoinGecko"] = "❌"
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            results["CoinGecko"] = "❌"
        
        # 6-9: CEX + DexScreener
        for name, url in [
            ("Binance", "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"),
            ("Coinbase", "https://api.coinbase.com/v2/prices/SOL-USD/spot"),
            ("Kraken", "https://api.kraken.com/0/public/Ticker?pair=SOLUSD"),
            ("DexScreener", "https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112"),
        ]:
            print(f"\n{6 + list(results.keys()).index(name) if name in results else len(results) + 6}. {name.upper()} API")
            try:
                resp = await client.get(url)
                data = resp.json()
                if name == "Binance" and "price" in data:
                    print(f"   ✅ ÇALIŞIYOR - SOL: ${float(data['price']):.2f}")
                    results[name] = "✅"
                elif name == "Coinbase" and "data" in data:
                    print(f"   ✅ ÇALIŞIYOR - SOL: ${data['data']['amount']}")
                    results[name] = "✅"
                elif name == "Kraken" and "result" in data:
                    price = list(data["result"].values())[0]["c"][0]
                    print(f"   ✅ ÇALIŞIYOR - SOL: ${float(price):.2f}")
                    results[name] = "✅"
                elif name == "DexScreener" and data.get("pairs"):
                    print(f"   ✅ ÇALIŞIYOR - SOL: ${data['pairs'][0]['priceUsd']}")
                    results[name] = "✅"
                else:
                    print(f"   ❌ HATA: Unexpected response")
                    results[name] = "❌"
            except Exception as e:
                print(f"   ❌ HATA: {e}")
                results[name] = "❌"
    
    print("\n" + "=" * 60)
    print("ÖZET RAPOR")
    print("=" * 60)
    for api, status in results.items():
        print(f"   {status} {api}")

if __name__ == "__main__":
    asyncio.run(test_apis())

