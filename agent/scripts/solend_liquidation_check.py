#!/usr/bin/env python3
"""
Solend/Save Liquidation Quick Check

Analyzes liquidation opportunities on Solend (now Save) protocol.
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

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

# Solend/Save Program IDs
SOLEND_PROGRAM_ID = "So1endDq2YkqhipRh3WViPa8hdiSpxWy6z3Z6tMCpAo"
SAVE_PROGRAM_ID = "SoLendVKb49t2vHaVF9gG43M8d2oVNSHsNrz3u7nfMB"  # New Save program

# Known liquidation instruction discriminators
LIQUIDATE_DISCRIMINATOR = "12"  # LiquidateObligation instruction


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def get_recent_transactions(client: httpx.AsyncClient, program_id: str, limit: int = 100) -> list:
    """Get recent transactions for a program using Helius."""
    url = f"https://api.helius.xyz/v0/addresses/{program_id}/transactions?api-key={HELIUS_API_KEY}&limit={limit}"
    
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        print(f"   ⚠️ Helius response: {resp.status_code}")
        return []
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


async def analyze_liquidations(client: httpx.AsyncClient) -> dict:
    """Analyze recent liquidation events."""
    print_section("FETCHING RECENT SOLEND TRANSACTIONS")
    
    # Get transactions from Solend program
    print("\n▸ Fetching Solend program transactions...")
    txs = await get_recent_transactions(client, SOLEND_PROGRAM_ID, limit=100)
    print(f"   Found {len(txs)} recent transactions")
    
    if not txs:
        print("   ⚠️ No transactions found. Trying alternative approach...")
        return {"liquidations": [], "count": 0}
    
    # Analyze transaction types
    liquidations = []
    tx_types = {}
    
    for tx in txs:
        tx_type = tx.get("type", "UNKNOWN")
        tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
        
        # Check for liquidation-related transactions
        description = tx.get("description", "").lower()
        if "liquidat" in description or tx_type == "LIQUIDATE":
            liquidations.append({
                "signature": tx.get("signature", "")[:20] + "...",
                "timestamp": tx.get("timestamp"),
                "type": tx_type,
                "description": description[:100] if description else "N/A",
                "fee": tx.get("fee", 0) / 1e9,
                "source": tx.get("source", "UNKNOWN")
            })
    
    print(f"\n▸ Transaction Type Distribution:")
    for tx_type, count in sorted(tx_types.items(), key=lambda x: -x[1])[:10]:
        print(f"      {tx_type}: {count}")
    
    print(f"\n▸ Liquidation Events Found: {len(liquidations)}")
    
    return {
        "liquidations": liquidations,
        "count": len(liquidations),
        "tx_types": tx_types,
        "total_txs": len(txs)
    }


async def check_at_risk_positions(client: httpx.AsyncClient) -> dict:
    """Check for positions close to liquidation threshold."""
    print_section("CHECKING AT-RISK POSITIONS")
    
    # Solend API for at-risk positions
    # Note: Solend's API might have changed since rebranding to Save
    urls = [
        "https://api.solend.fi/v1/markets/main",
        "https://api.save.finance/v1/markets/main",  # New Save API
    ]
    
    for url in urls:
        try:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✅ Connected to {url}")
                return {"api_available": True, "data": data}
        except:
            continue
    
    print("   ⚠️ No API endpoint available")
    return {"api_available": False}


def generate_verdict(analysis: dict) -> dict:
    """Generate final verdict on Solend liquidation opportunity."""
    print_section("🎯 SOLEND LIQUIDATION VERDICT")
    
    liq_count = analysis.get("count", 0)
    total_txs = analysis.get("total_txs", 0)
    
    # Calculate liquidation rate
    liq_rate = (liq_count / total_txs * 100) if total_txs > 0 else 0
    
    print(f"\n📊 STATISTICS")
    print(f"   Total transactions analyzed: {total_txs}")
    print(f"   Liquidation events: {liq_count}")
    print(f"   Liquidation rate: {liq_rate:.2f}%")
    
    # Show recent liquidations
    if analysis.get("liquidations"):
        print(f"\n🏆 RECENT LIQUIDATIONS")
        for liq in analysis["liquidations"][:5]:
            print(f"   • {liq['signature']}")
            print(f"     Type: {liq['type']} | Fee: {liq['fee']:.6f} SOL")
    
    # Verdict
    print(f"\n💡 VERDICT")
    
    # Key metrics for decision:
    # - Liquidation bonus: 5% (per Solend docs)
    # - Bot competition: HIGH (many liquidator bots)
    # - Frequency: Need >5 events/day to be viable
    
    if liq_count >= 5:
        verdict = "PROMISING"
        reason = f"{liq_count} liquidations in recent sample - worth investigating"
        next_step = "Set up 24-hour monitoring to track daily liquidation volume"
    elif liq_count >= 1:
        verdict = "MARGINAL"
        reason = f"Only {liq_count} liquidations - low frequency"
        next_step = "Consider during high volatility periods only"
    else:
        verdict = "NOT VIABLE"
        reason = "No liquidations detected in recent transactions"
        next_step = "Look at alternative strategies (MEV, cross-chain)"
    
    print(f"\n   Decision: {'✅' if verdict == 'PROMISING' else '⚠️' if verdict == 'MARGINAL' else '❌'} {verdict}")
    print(f"   Reason: {reason}")
    print(f"   Next Step: {next_step}")
    
    # Additional context
    print(f"\n📌 CONTEXT")
    print(f"   • Liquidation bonus: 5% (Solend default)")
    print(f"   • Bot competition: HIGH (open-source liquidator available)")
    print(f"   • Required capital: Variable (need to hold repay tokens)")
    print(f"   • Latency sensitivity: VERY HIGH (first bot wins)")
    
    return {
        "verdict": verdict,
        "liq_count": liq_count,
        "liq_rate": liq_rate,
        "reason": reason
    }


async def run_check():
    """Run the Solend liquidation check."""
    print("\n" + "💧" * 35)
    print("  SOLEND/SAVE LIQUIDATION QUICK CHECK")
    print("💧" * 35)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    
    async with httpx.AsyncClient(timeout=60) as client:
        # Analyze recent transactions
        analysis = await analyze_liquidations(client)
        
        # Check at-risk positions
        await check_at_risk_positions(client)
        
        # Generate verdict
        result = generate_verdict(analysis)
        
        print("\n" + "=" * 70)
        print("  CHECK COMPLETE")
        print("=" * 70)
        
        return result


if __name__ == "__main__":
    result = asyncio.run(run_check())
    sys.exit(0)

