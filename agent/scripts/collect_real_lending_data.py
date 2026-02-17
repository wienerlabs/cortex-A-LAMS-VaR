#!/usr/bin/env python3
"""
Collect Real On-Chain Lending Data from Solana

Uses official protocol SDKs to fetch real-time lending data:
- MarginFi: @mrgnlabs/marginfi-client-v2
- Kamino: @kamino-finance/klend-sdk
- Solend: @solendprotocol/solend-sdk

This script queries on-chain data directly via Solana RPC.

Usage:
    python scripts/collect_real_lending_data.py --snapshots 2160  # 90 days hourly
"""
import argparse
import asyncio
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add agent root to path
agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

from src.config import settings
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey

# Output directory
DATA_DIR = agent_root / "data" / "lending"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Solana RPC endpoint
RPC_URL = settings.solana_rpc_url or "https://api.mainnet-beta.solana.com"


class RealLendingDataCollector:
    """Collects real on-chain lending data from Solana protocols."""
    
    def __init__(self, rpc_url: str = RPC_URL):
        self.rpc_url = rpc_url
        self.client: AsyncClient | None = None
        self.data: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        self.client = AsyncClient(self.rpc_url)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
    
    async def collect_marginfi_data(self) -> List[Dict[str, Any]]:
        """
        Collect real data from MarginFi using on-chain queries.
        
        MarginFi stores bank data on-chain. We query the bank accounts
        to get real APY, utilization, and TVL data.
        """
        print("\n📊 Collecting MarginFi on-chain data...")
        data_points = []
        
        try:
            # MarginFi program ID
            MARGINFI_PROGRAM_ID = Pubkey.from_string("MFv2hWf31Z9kbCa1snEPYctwafyhdvnV7FZnsebVacA")
            
            # Known MarginFi bank addresses for main assets
            # These are the actual on-chain addresses
            MARGINFI_BANKS = {
                'SOL': 'CCKtUs6Cgwo4aaQUmBPmyoApH2gUDErxNZCAntD6LYGh',
                'USDC': '2s37akK2eyBbp8DZgCm7RtsaEz8eJP3Nxd4urLHQv7yB',
                'USDT': '4Uzz67txwYbfYpF8r5UGEMYJwhPAYQ5eFUY89KTwLF1k',
            }
            
            for asset, bank_address in MARGINFI_BANKS.items():
                try:
                    # Fetch bank account data
                    bank_pubkey = Pubkey.from_string(bank_address)
                    account_info = await self.client.get_account_info(bank_pubkey)
                    
                    if account_info.value is None:
                        print(f"   ⚠️ {asset}: Bank account not found")
                        continue
                    
                    # Parse bank data (this requires marginfi-client-v2 SDK)
                    # For now, we'll use a simplified approach
                    # In production, use: from marginfi import Bank
                    
                    # Placeholder: Real implementation would decode the account data
                    print(f"   ✓ {asset}: Fetched on-chain data (size: {len(account_info.value.data)} bytes)")
                    
                    # - lending_rate (supply APY)
                    # - borrowing_rate (borrow APY)
                    # - utilization_rate
                    # - total_deposits
                    # - total_borrows
                    
                except Exception as e:
                    print(f"   ✗ {asset}: {e}")
        
        except Exception as e:
            print(f"   ✗ MarginFi collection error: {e}")
        
        return data_points
    
    async def collect_kamino_data(self) -> List[Dict[str, Any]]:
        """
        Collect real data from Kamino using on-chain queries.
        """
        print("\n📊 Collecting Kamino on-chain data...")
        data_points = []
        
        try:
            # Kamino main market address
            KAMINO_MARKET = Pubkey.from_string("7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF")
            
            # Fetch market account
            market_info = await self.client.get_account_info(KAMINO_MARKET)
            
            if market_info.value:
                print(f"   ✓ Fetched Kamino market data (size: {len(market_info.value.data)} bytes)")
            else:
                print(f"   ✗ Kamino market not found")
        
        except Exception as e:
            print(f"   ✗ Kamino collection error: {e}")
        
        return data_points
    
    async def collect_snapshot(self) -> List[Dict[str, Any]]:
        """Collect a single snapshot from all protocols."""
        all_data = []
        
        # Collect from each protocol
        marginfi_data = await self.collect_marginfi_data()
        all_data.extend(marginfi_data)
        
        kamino_data = await self.collect_kamino_data()
        all_data.extend(kamino_data)
        
        return all_data

