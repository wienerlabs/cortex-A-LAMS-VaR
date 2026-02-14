#!/usr/bin/env python3
"""
Comprehensive Solana Data Pipeline Test.
CRITICAL: All data is REAL-TIME from APIs. No hardcoded/synthetic values.
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# API Keys
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")

# Token addresses
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
RAYDIUM_SOL_USDC = "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"

test_results = {}
collected_data = {}


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    print(f"\n--- {title} ---")


def print_result(success: bool, msg: str, data: dict = None):
    status = "✅" if success else "❌"
    print(f"{status} {msg}")
    if data:
        for k, v in list(data.items())[:5]:
            print(f"   {k}: {v}")


async def make_request(client: httpx.AsyncClient, method: str, url: str,
                       headers: dict = None, json_data: dict = None) -> tuple:
    try:
        if method == "GET":
            resp = await client.get(url, headers=headers)
        else:
            resp = await client.post(url, headers=headers, json=json_data)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"HTTP {resp.status_code}", "body": resp.text[:200]}
    except Exception as e:
        return False, {"error": str(e)}


async def test_helius_block_and_tx(client: httpx.AsyncClient) -> dict:
    """Test Helius: Get latest block + sample transaction."""
    print_subsection("Helius RPC - Block & Transaction")
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    results = {"success": False}

    success, data = await make_request(client, "POST", url, json_data={
        "jsonrpc": "2.0", "id": 1, "method": "getSlot", "params": []
    })
    if not success:
        print_result(False, f"Failed to get slot: {data}")
        return results

    slot = data.get("result", 0)
    print_result(True, f"Latest slot: {slot:,}")
    results["slot"] = slot

    success, data = await make_request(client, "POST", url, json_data={
        "jsonrpc": "2.0", "id": 2, "method": "getBlock", "params": [
            slot - 10,
            {"encoding": "jsonParsed", "transactionDetails": "signatures", "maxSupportedTransactionVersion": 0}
        ]
    })
    if success and "result" in data:
        block = data["result"]
        tx_count = len(block.get("signatures", []))
        block_time = block.get("blockTime", 0)
        print_result(True, f"Block {slot-10}: {tx_count} transactions", {
            "blockTime": datetime.fromtimestamp(block_time, tz=timezone.utc).isoformat() if block_time else "N/A"
        })
        results["block_tx_count"] = tx_count
        if block.get("signatures"):
            results["sample_tx"] = block["signatures"][0][:50] + "..."

    results["success"] = True
    return results


async def test_birdeye_price_and_liquidity(client: httpx.AsyncClient) -> dict:
    """Test Birdeye: Get SOL/USDC price + pool liquidity."""
    print_subsection("Birdeye - Price & Liquidity")
    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}
    results = {"success": False}

    url = f"https://public-api.birdeye.so/defi/price?address={SOL_MINT}"
    success, data = await make_request(client, "GET", url, headers=headers)
    if success and data.get("success"):
        price_data = data["data"]
        results["sol_price"] = price_data["value"]
        results["price_change_24h"] = price_data.get("priceChange24h", 0)
        print_result(True, f"SOL Price: ${results['sol_price']:.4f}", {
            "24h_change": f"{results['price_change_24h']:.2f}%"
        })
    else:
        print_result(False, f"Price fetch failed: {data}")
        return results

    pool_url = f"https://public-api.birdeye.so/defi/v3/pool/info?address={RAYDIUM_SOL_USDC}"
    success, data = await make_request(client, "GET", pool_url, headers=headers)
    if success and data.get("success"):
        pool = data.get("data", {})
        results["raydium_liquidity"] = pool.get("liquidity", 0)
        print_result(True, f"Raydium Liquidity: ${results['raydium_liquidity']:,.0f}")

    results["success"] = True
    return results


async def test_jupiter_quote_and_route(client: httpx.AsyncClient) -> dict:
    """Test Jupiter: Get SOL/USDC quote + best route."""
    print_subsection("Jupiter Ultra - Quote & Route")
    results = {"success": False}
    headers = {"Content-Type": "application/json"}
    if JUPITER_API_KEY:
        headers["x-api-key"] = JUPITER_API_KEY

    amount = 1_000_000_000
    url = f"https://api.jup.ag/ultra/v1/order?inputMint={SOL_MINT}&outputMint={USDC_MINT}&amount={amount}"
    success, data = await make_request(client, "GET", url, headers=headers)

    if not success:
        print_result(False, f"Jupiter quote failed: {data}")
        return results

    out_amount = int(data.get("outAmount", 0)) / 1_000_000
    price_impact = float(data.get("priceImpactPct", 0))
    results["quote_1_sol"] = out_amount
    results["price_impact"] = price_impact
    print_result(True, f"1 SOL = ${out_amount:.4f} USDC", {"price_impact": f"{price_impact:.6f}%"})

    route_plan = data.get("routePlan", [])
    results["routes"] = [step.get("swapInfo", {}).get("label", "?") for step in route_plan]
    if route_plan:
        print(f"   Routes: {' -> '.join(results['routes'][:5])}")

    results["success"] = True
    return results


# =============================================================================
# PART 2: 2-DAY DATA COLLECTION
# =============================================================================

async def collect_birdeye_ohlcv(client: httpx.AsyncClient, hours: int = 48, retries: int = 3) -> pd.DataFrame:
    """Collect hourly OHLCV data from Birdeye with retry logic."""
    print_subsection(f"Birdeye OHLCV - Last {hours} hours")
    headers = {"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"}

    now = int(datetime.now(timezone.utc).timestamp())
    time_from = now - (hours * 3600)

    url = (f"https://public-api.birdeye.so/defi/ohlcv?"
           f"address={SOL_MINT}&type=1H&time_from={time_from}&time_to={now}")

    for attempt in range(retries):
        success, data = await make_request(client, "GET", url, headers=headers)

        if success and data.get("success"):
            break
        elif "429" in str(data):
            wait_time = (attempt + 1) * 5
            print(f"   ⏳ Rate limited, waiting {wait_time}s... (attempt {attempt+1}/{retries})")
            await asyncio.sleep(wait_time)
        else:
            print_result(False, f"OHLCV fetch failed: {data}")
            return pd.DataFrame()

    if not success or not data.get("success"):
        print_result(False, f"OHLCV fetch failed after {retries} retries")
        return pd.DataFrame()

    items = data.get("data", {}).get("items", [])
    if not items:
        print_result(False, "No OHLCV data returned")
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["datetime"] = pd.to_datetime(df["unixTime"], unit="s", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.sort_values("datetime").reset_index(drop=True)

    print_result(True, f"Collected {len(df)} hourly candles", {
        "from": df["datetime"].min().isoformat(),
        "to": df["datetime"].max().isoformat(),
        "price_range": f"${df['low'].min():.2f} - ${df['high'].max():.2f}"
    })

    return df


async def collect_jupiter_spreads(client: httpx.AsyncClient, samples: int = 10) -> list:
    """Collect spread samples from Jupiter for multiple amounts."""
    print_subsection(f"Jupiter Spread Samples - {samples} data points")
    headers = {"Content-Type": "application/json"}
    if JUPITER_API_KEY:
        headers["x-api-key"] = JUPITER_API_KEY

    amounts = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000][:samples]
    spreads = []

    for sol_amount in amounts:
        lamports = int(sol_amount * 1_000_000_000)
        url = f"https://api.jup.ag/ultra/v1/order?inputMint={SOL_MINT}&outputMint={USDC_MINT}&amount={lamports}"

        success, data = await make_request(client, "GET", url, headers=headers)

        if success:
            out_amount = int(data.get("outAmount", 0)) / 1_000_000
            price_impact = float(data.get("priceImpactPct", 0))
            route_plan = data.get("routePlan", [])
            effective_price = out_amount / sol_amount if sol_amount > 0 else 0

            spreads.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sol_amount": sol_amount,
                "usdc_out": out_amount,
                "effective_price": effective_price,
                "price_impact_pct": price_impact,
                "route_count": len(route_plan),
                "primary_dex": route_plan[0]["swapInfo"]["label"] if route_plan else "unknown"
            })

        await asyncio.sleep(0.2)

    if spreads:
        print_result(True, f"Collected {len(spreads)} spread samples", {
            "amount_range": f"{amounts[0]} - {amounts[-1]} SOL",
            "price_range": f"${min(s['effective_price'] for s in spreads):.4f} - ${max(s['effective_price'] for s in spreads):.4f}"
        })

    return spreads


async def collect_helius_fees(client: httpx.AsyncClient, samples: int = 20) -> list:
    """Collect recent transaction fee data from Helius."""
    print_subsection(f"Helius Fee Data - {samples} samples")
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    fees = []

    success, data = await make_request(client, "POST", url, json_data={
        "jsonrpc": "2.0", "id": 1, "method": "getRecentPrioritizationFees", "params": []
    })

    if success and "result" in data:
        fee_data = data["result"][-samples:]
        for item in fee_data:
            fees.append({
                "slot": item["slot"],
                "prioritization_fee": item["prioritizationFee"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        avg_fee = sum(f["prioritization_fee"] for f in fees) / len(fees) if fees else 0
        print_result(True, f"Collected {len(fees)} fee samples", {
            "avg_prioritization_fee": f"{avg_fee:.2f} micro-lamports",
            "slot_range": f"{fees[0]['slot']} - {fees[-1]['slot']}" if fees else "N/A"
        })

    return fees


# =============================================================================
# PART 3: FEATURE ENGINEERING
# =============================================================================

def generate_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical analysis features from OHLCV data."""
    print_subsection("TA-Lib Feature Engineering")

    if df.empty:
        print_result(False, "No data for feature engineering")
        return df

    try:
        import talib
        has_talib = True
    except ImportError:
        has_talib = False
        print("   ⚠️ TA-Lib not installed, using basic calculations")

    # Basic features (no TA-Lib required)
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_6h"] = df["returns"].rolling(window=6).std()
    df["volatility_24h"] = df["returns"].rolling(window=24).std()
    df["volume_ma_6h"] = df["volume"].rolling(window=6).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_6h"]
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    df["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 0.0001)

    if has_talib:
        df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"])
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(df["close"])
        df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        df["adx_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        df["ema_12"] = talib.EMA(df["close"], timeperiod=12)
        df["ema_26"] = talib.EMA(df["close"], timeperiod=26)
        df["sma_50"] = talib.SMA(df["close"], timeperiod=min(50, len(df)-1))
    else:
        # Simple RSI approximation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 0.0001)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Simple Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (std * 2)
        df["bb_lower"] = df["bb_middle"] - (std * 2)

        # EMAs
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # Count missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    features_added = len([c for c in df.columns if c not in ["datetime", "unixTime", "open", "high", "low", "close", "volume"]])

    print_result(True, f"Generated {features_added} features", {
        "rows": len(df),
        "total_missing": total_missing,
        "ta_lib": "Yes" if has_talib else "No (basic calc)"
    })

    return df


def calculate_spread_metrics(spreads: list) -> dict:
    """Calculate spread metrics from Jupiter data."""
    print_subsection("Spread Analysis")

    if not spreads:
        print_result(False, "No spread data")
        return {}

    df = pd.DataFrame(spreads)

    metrics = {
        "avg_effective_price": df["effective_price"].mean(),
        "price_std": df["effective_price"].std(),
        "max_price_impact": df["price_impact_pct"].abs().max(),
        "avg_route_count": df["route_count"].mean(),
        "primary_dex_counts": df["primary_dex"].value_counts().to_dict()
    }

    print_result(True, f"Spread metrics calculated", {
        "avg_price": f"${metrics['avg_effective_price']:.4f}",
        "price_std": f"${metrics['price_std']:.6f}",
        "max_impact": f"{metrics['max_price_impact']:.4f}%",
        "primary_dexes": list(metrics["primary_dex_counts"].keys())[:3]
    })

    return metrics


def find_opportunities(df: pd.DataFrame, spreads: list) -> list:
    """Find potential profitable opportunities."""
    print_subsection("Opportunity Detection")

    opportunities = []

    if df.empty:
        print_result(False, "No data for opportunity detection")
        return opportunities

    # RSI-based opportunities
    if "rsi_14" in df.columns:
        oversold = df[df["rsi_14"] < 30]
        overbought = df[df["rsi_14"] > 70]
        for _, row in oversold.iterrows():
            if pd.notna(row["rsi_14"]):
                opportunities.append({
                    "type": "RSI_OVERSOLD",
                    "datetime": row["datetime"].isoformat(),
                    "price": row["close"],
                    "rsi": row["rsi_14"],
                    "signal": "BUY"
                })
        for _, row in overbought.iterrows():
            if pd.notna(row["rsi_14"]):
                opportunities.append({
                    "type": "RSI_OVERBOUGHT",
                    "datetime": row["datetime"].isoformat(),
                    "price": row["close"],
                    "rsi": row["rsi_14"],
                    "signal": "SELL"
                })

    # Bollinger Band opportunities
    if "bb_lower" in df.columns:
        below_bb = df[df["close"] < df["bb_lower"]]
        above_bb = df[df["close"] > df["bb_upper"]]
        for _, row in below_bb.iterrows():
            if pd.notna(row["bb_lower"]):
                opportunities.append({
                    "type": "BB_BELOW_LOWER",
                    "datetime": row["datetime"].isoformat(),
                    "price": row["close"],
                    "bb_lower": row["bb_lower"],
                    "signal": "BUY"
                })
        for _, row in above_bb.iterrows():
            if pd.notna(row["bb_upper"]):
                opportunities.append({
                    "type": "BB_ABOVE_UPPER",
                    "datetime": row["datetime"].isoformat(),
                    "price": row["close"],
                    "bb_upper": row["bb_upper"],
                    "signal": "SELL"
                })

    # High volatility opportunities
    if "volatility_24h" in df.columns:
        vol_threshold = df["volatility_24h"].quantile(0.9)
        high_vol = df[df["volatility_24h"] > vol_threshold]
        for _, row in high_vol.iterrows():
            if pd.notna(row["volatility_24h"]):
                opportunities.append({
                    "type": "HIGH_VOLATILITY",
                    "datetime": row["datetime"].isoformat(),
                    "price": row["close"],
                    "volatility": row["volatility_24h"],
                    "signal": "ALERT"
                })

    print_result(True, f"Found {len(opportunities)} opportunities", {
        "RSI_signals": len([o for o in opportunities if "RSI" in o["type"]]),
        "BB_signals": len([o for o in opportunities if "BB" in o["type"]]),
        "volatility_alerts": len([o for o in opportunities if "VOLATILITY" in o["type"]])
    })

    return opportunities



# =============================================================================
# PART 4: MAIN EXECUTION
# =============================================================================

async def run_all_tests():
    """Run all tests and generate report."""
    print("  COMPREHENSIVE SOLANA DATA PIPELINE TEST")
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")

    async with httpx.AsyncClient(timeout=60) as client:
        # =================================================================
        # PART 1: API Connection Tests
        # =================================================================
        print_section("PART 1: API CONNECTION TESTS")

        helius_result = await test_helius_block_and_tx(client)
        birdeye_result = await test_birdeye_price_and_liquidity(client)
        jupiter_result = await test_jupiter_quote_and_route(client)

        api_success = all([
            helius_result.get("success"),
            birdeye_result.get("success"),
            jupiter_result.get("success")
        ])

        if not api_success:
            print("\n❌ API Connection Tests FAILED. Stopping.")
            return False

        print("\n✅ All API connections successful!")

        # =================================================================
        # PART 2: 2-Day Data Collection
        # =================================================================
        print_section("PART 2: 2-DAY DATA COLLECTION")

        ohlcv_df = await collect_birdeye_ohlcv(client, hours=48)
        spreads = await collect_jupiter_spreads(client, samples=10)
        fees = await collect_helius_fees(client, samples=20)

        collected_data["ohlcv"] = ohlcv_df
        collected_data["spreads"] = spreads
        collected_data["fees"] = fees

        # =================================================================
        # PART 3: Feature Engineering
        # =================================================================
        print_section("PART 3: FEATURE ENGINEERING")

        featured_df = generate_ta_features(ohlcv_df.copy())
        spread_metrics = calculate_spread_metrics(spreads)
        opportunities = find_opportunities(featured_df, spreads)

        # =================================================================
        # PART 4: Sample Output Report
        # =================================================================
        print_section("PART 4: SAMPLE OUTPUT REPORT")

        print_subsection("Data Summary")

        # OHLCV Summary
        if not featured_df.empty:
            print(f"\n📊 OHLCV Data:")
            print(f"   Rows: {len(featured_df)}")
            print(f"   Columns: {len(featured_df.columns)}")
            print(f"   Date Range: {featured_df['datetime'].min()} to {featured_df['datetime'].max()}")
            print(f"   Price Range: ${featured_df['low'].min():.2f} - ${featured_df['high'].max():.2f}")

            # Missing values
            missing = featured_df.isnull().sum()
            cols_with_missing = missing[missing > 0]
            print(f"\n   Missing Values:")
            if len(cols_with_missing) > 0:
                for col, count in cols_with_missing.items():
                    print(f"      {col}: {count} ({count/len(featured_df)*100:.1f}%)")
            else:
                print(f"      None (all values present)")

        # Spread Summary
        print(f"\n📈 Spread Data:")
        print(f"   Samples: {len(spreads)}")
        if spread_metrics:
            print(f"   Avg Price: ${spread_metrics['avg_effective_price']:.4f}")
            print(f"   Price Std: ${spread_metrics['price_std']:.6f}")
            print(f"   Max Impact: {spread_metrics['max_price_impact']:.4f}%")

        # Fee Summary
        print(f"\n💰 Fee Data:")
        print(f"   Samples: {len(fees)}")
        if fees:
            avg_fee = sum(f["prioritization_fee"] for f in fees) / len(fees)
            print(f"   Avg Prioritization Fee: {avg_fee:.2f} micro-lamports")

        # Feature Distributions
        print_subsection("Feature Distributions")
        if not featured_df.empty:
            key_features = ["returns", "volatility_24h", "rsi_14", "volume_ratio"]
            for feat in key_features:
                if feat in featured_df.columns:
                    vals = featured_df[feat].dropna()
                    if len(vals) > 0:
                        print(f"   {feat}:")
                        print(f"      min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}, std={vals.std():.6f}")

        # Opportunities
        print_subsection("Profitable Opportunities")
        print(f"   Total Found: {len(opportunities)}")
        if opportunities:
            by_type = {}
            for opp in opportunities:
                t = opp["type"]
                by_type[t] = by_type.get(t, 0) + 1
            for t, count in by_type.items():
                print(f"      {t}: {count}")

            print(f"\n   Sample Opportunities:")
            for opp in opportunities[:5]:
                print(f"      [{opp['signal']}] {opp['type']} @ {opp['datetime'][:16]} - ${opp['price']:.2f}")

        # Final Summary
        print_section("FINAL SUMMARY")
        print(f"   ✅ API Connections: 3/3 passed")
        print(f"   ✅ OHLCV Data: {len(ohlcv_df)} rows collected")
        print(f"   ✅ Spread Samples: {len(spreads)} collected")
        print(f"   ✅ Fee Samples: {len(fees)} collected")
        print(f"   ✅ Features Generated: {len(featured_df.columns) - 7 if not featured_df.empty else 0}")
        print(f"   ✅ Opportunities Found: {len(opportunities)}")

        return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)