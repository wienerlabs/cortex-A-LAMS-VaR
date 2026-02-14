#!/usr/bin/env python3
"""
Solana Liquidity Pool Analysis

Analyzes top LP pools across Raydium, Orca, and Meteora.
Calculates risk-adjusted returns and recommends best pools.
"""
import asyncio
import httpx
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")


@dataclass
class Pool:
    dex: str
    name: str
    address: str
    token_a: str
    token_b: str
    tvl: float
    volume_24h: float
    apy: float
    fee_tier: float  # percentage
    volatility: Optional[float] = None  # for IL calculation

    @property
    def volume_tvl_ratio(self) -> float:
        return self.volume_24h / self.tvl if self.tvl > 0 else 0

    @property
    def il_risk_score(self) -> float:
        # Higher volatility = higher IL risk (1-10 scale)
        if self.volatility is None:
            # Estimate based on token types
            stable = ["USDC", "USDT", "PYUSD", "UXD"]
            if self.token_a in stable and self.token_b in stable:
                return 1.0  # Stable-stable, minimal IL
            elif self.token_a in stable or self.token_b in stable:
                return 5.0  # Stable-volatile, medium IL
            else:
                return 8.0  # Volatile-volatile, high IL
        return min(10, self.volatility * 100)

    @property
    def risk_adjusted_apy(self) -> float:
        # Simple risk adjustment: APY / (1 + IL_risk/10)
        return self.apy / (1 + self.il_risk_score / 10)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def fetch_raydium_pools(client: httpx.AsyncClient) -> list[Pool]:
    """Fetch top pools from Raydium."""
    print("\n▸ Fetching Raydium pools...")

    try:
        # Raydium API v2
        url = "https://api.raydium.io/v2/main/pairs"
        resp = await client.get(url, timeout=30)

        if resp.status_code != 200:
            print(f"   ⚠️ Raydium API: {resp.status_code}")
            return []

        data = resp.json()
        pools = []

        # Sort by liquidity and take top 10
        sorted_pairs = sorted(data, key=lambda x: float(x.get("liquidity", 0)), reverse=True)[:10]

        for p in sorted_pairs:
            try:
                pool = Pool(
                    dex="Raydium",
                    name=p.get("name", "Unknown"),
                    address=p.get("ammId", ""),
                    token_a=p.get("name", "").split("-")[0] if "-" in p.get("name", "") else "?",
                    token_b=p.get("name", "").split("-")[1] if "-" in p.get("name", "") else "?",
                    tvl=float(p.get("liquidity", 0)),
                    volume_24h=float(p.get("volume24h", 0)),
                    apy=float(p.get("apr7d", 0)) if p.get("apr7d") else float(p.get("apr24h", 0)),
                    fee_tier=0.25  # Raydium standard
                )
                pools.append(pool)
            except Exception as e:
                continue

        print(f"   ✅ Found {len(pools)} Raydium pools")
        return pools

    except Exception as e:
        print(f"   ❌ Raydium error: {e}")
        return []


async def fetch_orca_pools(client: httpx.AsyncClient) -> list[Pool]:
    """Fetch top pools from Orca."""
    print("\n▸ Fetching Orca pools...")

    try:
        url = "https://api.mainnet.orca.so/v1/whirlpool/list"
        resp = await client.get(url, timeout=30)

        if resp.status_code != 200:
            print(f"   ⚠️ Orca API: {resp.status_code}")
            return []

        data = resp.json()
        whirlpools = data.get("whirlpools", [])

        # Sort by TVL
        sorted_pools = sorted(whirlpools, key=lambda x: float(x.get("tvl", 0)), reverse=True)[:10]

        pools = []
        for p in sorted_pools:
            try:
                token_a = p.get("tokenA", {})
                token_b = p.get("tokenB", {})

                pool = Pool(
                    dex="Orca",
                    name=f"{token_a.get('symbol', '?')}-{token_b.get('symbol', '?')}",
                    address=p.get("address", ""),
                    token_a=token_a.get("symbol", "?"),
                    token_b=token_b.get("symbol", "?"),
                    tvl=float(p.get("tvl", 0)),
                    volume_24h=float(p.get("volume", {}).get("day", 0)),
                    apy=float(p.get("feeApr", {}).get("week", 0)) * 100,
                    fee_tier=float(p.get("tickSpacing", 64)) / 100 * 0.01  # Approximate
                )
                pools.append(pool)
            except:
                continue

        print(f"   ✅ Found {len(pools)} Orca pools")
        return pools

    except Exception as e:
        print(f"   ❌ Orca error: {e}")
        return []


async def fetch_meteora_pools(client: httpx.AsyncClient) -> list[Pool]:
    """Fetch top pools from Meteora."""
    print("\n▸ Fetching Meteora pools...")

    try:
        url = "https://dlmm-api.meteora.ag/pair/all"
        resp = await client.get(url, timeout=30)

        if resp.status_code != 200:
            print(f"   ⚠️ Meteora API: {resp.status_code}")
            return []

        data = resp.json()
        sorted_pools = sorted(data, key=lambda x: float(x.get("liquidity", 0)), reverse=True)[:10]

        pools = []
        for p in sorted_pools:
            try:
                pool = Pool(
                    dex="Meteora",
                    name=p.get("name", "Unknown"),
                    address=p.get("address", ""),
                    token_a=p.get("mint_x", "")[:4] if p.get("mint_x") else "?",
                    token_b=p.get("mint_y", "")[:4] if p.get("mint_y") else "?",
                    tvl=float(p.get("liquidity", 0)),
                    volume_24h=float(p.get("trade_volume_24h", 0)),
                    apy=float(p.get("apr", 0)) if p.get("apr") else 0,
                    fee_tier=float(p.get("base_fee_percentage", 0.25))
                )
                # Try to get proper names
                if "-" in p.get("name", ""):
                    parts = p["name"].split("-")
                    pool.token_a = parts[0].strip()
                    pool.token_b = parts[1].strip() if len(parts) > 1 else "?"
                pools.append(pool)
            except:
                continue

        print(f"   ✅ Found {len(pools)} Meteora pools")
        return pools

    except Exception as e:
        print(f"   ❌ Meteora error: {e}")
        return []


def analyze_pools(pools: list[Pool]) -> list[Pool]:
    """Analyze and rank pools."""
    print_section("POOL ANALYSIS")

    if not pools:
        print("   ❌ No pools to analyze")
        return []

    # Filter out pools with 0 TVL or APY
    valid_pools = [p for p in pools if p.tvl > 10000 and p.apy > 0]
    print(f"\n   Valid pools (TVL > $10k, APY > 0): {len(valid_pools)}")

    # Sort by risk-adjusted APY
    ranked = sorted(valid_pools, key=lambda x: x.risk_adjusted_apy, reverse=True)

    return ranked


def print_pool_table(pools: list[Pool], title: str, limit: int = 10):
    """Print pool data in table format."""
    print(f"\n{title}")
    print("-" * 120)
    print(f"{'#':<3} {'DEX':<10} {'Pool':<20} {'TVL':>15} {'Vol 24h':>15} {'APY':>8} {'IL Risk':>8} {'Adj APY':>10} {'Vol/TVL':>8}")
    print("-" * 120)

    for i, p in enumerate(pools[:limit], 1):
        tvl_str = f"${p.tvl/1e6:.2f}M" if p.tvl >= 1e6 else f"${p.tvl/1e3:.0f}K"
        vol_str = f"${p.volume_24h/1e6:.2f}M" if p.volume_24h >= 1e6 else f"${p.volume_24h/1e3:.0f}K"

        print(f"{i:<3} {p.dex:<10} {p.name:<20} {tvl_str:>15} {vol_str:>15} {p.apy:>7.1f}% {p.il_risk_score:>7.1f} {p.risk_adjusted_apy:>9.1f}% {p.volume_tvl_ratio:>7.2f}")


def generate_recommendations(pools: list[Pool]) -> list[dict]:
    """Generate top 5 pool recommendations with allocation."""
    print_section("🎯 TOP 5 POOL RECOMMENDATIONS")

    if len(pools) < 5:
        print("   ⚠️ Not enough pools for recommendations")
        return []

    top5 = pools[:5]

    # Calculate allocations based on risk-adjusted APY
    total_score = sum(p.risk_adjusted_apy for p in top5)

    recommendations = []
    for i, p in enumerate(top5, 1):
        allocation = (p.risk_adjusted_apy / total_score * 100) if total_score > 0 else 20

        # Risk classification
        if p.il_risk_score <= 2:
            risk_label = "🟢 LOW"
        elif p.il_risk_score <= 5:
            risk_label = "🟡 MEDIUM"
        else:
            risk_label = "🔴 HIGH"

        rec = {
            "rank": i,
            "pool": p.name,
            "dex": p.dex,
            "allocation": allocation,
            "expected_apy": p.apy,
            "risk_adjusted_apy": p.risk_adjusted_apy,
            "risk_score": p.il_risk_score,
            "risk_label": risk_label,
            "tvl": p.tvl
        }
        recommendations.append(rec)

        print(f"\n   {i}. {p.name} ({p.dex})")
        print(f"      📊 Allocation: {allocation:.1f}%")
        print(f"      💰 Expected APY: {p.apy:.1f}%")
        print(f"      📈 Risk-Adjusted APY: {p.risk_adjusted_apy:.1f}%")
        print(f"      ⚠️ Risk: {risk_label} (IL Score: {p.il_risk_score:.1f})")
        print(f"      💧 TVL: ${p.tvl/1e6:.2f}M")
        print(f"      📊 Vol/TVL: {p.volume_tvl_ratio:.2f}")

    return recommendations


def generate_verdict(pools: list[Pool], recommendations: list[dict]):
    """Generate final verdict on LP strategy."""
    print_section("📋 FINAL VERDICT")

    if not recommendations:
        print("\n   ❌ LP Strategy: NOT VIABLE")
        print("   Reason: Insufficient pool data")
        return

    # Calculate portfolio metrics
    weighted_apy = sum(r["expected_apy"] * r["allocation"] / 100 for r in recommendations)
    weighted_risk_apy = sum(r["risk_adjusted_apy"] * r["allocation"] / 100 for r in recommendations)
    avg_risk = sum(r["risk_score"] * r["allocation"] / 100 for r in recommendations)

    print(f"\n   📊 PORTFOLIO METRICS (Top 5 Weighted)")
    print(f"      Expected APY: {weighted_apy:.1f}%")
    print(f"      Risk-Adjusted APY: {weighted_risk_apy:.1f}%")
    print(f"      Average Risk Score: {avg_risk:.1f}/10")

    # Decision criteria
    # - If risk-adjusted APY > 20%: EXCELLENT
    # - If risk-adjusted APY > 10%: GOOD
    # - If risk-adjusted APY > 5%: MARGINAL
    # - Below 5%: NOT WORTH IT

    print(f"\n   🎯 VERDICT")
    if weighted_risk_apy > 20:
        print(f"      ✅ EXCELLENT - Strong LP opportunity")
        print(f"      Proceed with deployment")
    elif weighted_risk_apy > 10:
        print(f"      ✅ GOOD - Viable LP strategy")
        print(f"      Consider partial deployment")
    elif weighted_risk_apy > 5:
        print(f"      ⚠️ MARGINAL - Low returns vs risk")
        print(f"      Only deploy if no better options")
    else:
        print(f"      ❌ NOT VIABLE - Returns too low")
        print(f"      Seek alternative strategies")

    # Risk warnings
    print(f"\n   ⚠️ RISK WARNINGS")
    print(f"      • Impermanent Loss can exceed gains in volatile periods")
    print(f"      • APYs fluctuate based on volume")
    print(f"      • Smart contract risk exists on all DEXes")
    print(f"      • Concentrated liquidity requires active management")


async def run_analysis():
    """Run the full LP pool analysis."""
    print("\n" + "💧" * 35)
    print("  SOLANA LIQUIDITY POOL ANALYSIS")
    print("💧" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")

    async with httpx.AsyncClient(timeout=60) as client:
        print_section("DATA COLLECTION")

        # Fetch from all DEXes concurrently
        raydium_task = fetch_raydium_pools(client)
        orca_task = fetch_orca_pools(client)
        meteora_task = fetch_meteora_pools(client)

        results = await asyncio.gather(raydium_task, orca_task, meteora_task)

        all_pools = []
        for pools in results:
            all_pools.extend(pools)

        print(f"\n   📊 Total pools collected: {len(all_pools)}")

        # Print raw data by DEX
        for dex in ["Raydium", "Orca", "Meteora"]:
            dex_pools = [p for p in all_pools if p.dex == dex]
            if dex_pools:
                print_pool_table(dex_pools, f"\n📊 {dex.upper()} TOP POOLS", 10)

        # Analyze and rank
        ranked_pools = analyze_pools(all_pools)

        if ranked_pools:
            print_pool_table(ranked_pools, "\n🏆 ALL POOLS RANKED BY RISK-ADJUSTED APY", 15)

        # Generate recommendations
        recommendations = generate_recommendations(ranked_pools)

        # Final verdict
        generate_verdict(ranked_pools, recommendations)

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70)

        return recommendations


if __name__ == "__main__":
    result = asyncio.run(run_analysis())
    sys.exit(0 if result else 1)
