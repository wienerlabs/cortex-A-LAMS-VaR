"""
Spot Trading Data Collector
Collects historical OHLCV data and enriches with sentiment/fundamental data
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os


class SpotDataCollector:
    """Collect historical data for spot trading ML model"""
    
    def __init__(self, birdeye_api_key: Optional[str] = None):
        self.birdeye_api_key = birdeye_api_key or os.getenv('BIRDEYE_API_KEY')
        self.base_url = 'https://public-api.birdeye.so'
        
    def collect_token_data(
        self,
        token_address: str,
        days: int = 180
    ) -> pd.DataFrame:
        """
        Collect historical data for a single token
        
        Args:
            token_address: Solana token address
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV + enriched data
        """
        print(f"[DataCollector] Collecting {days} days of data for {token_address}")
        
        # 1. Get OHLCV data from Birdeye
        ohlcv = self._fetch_ohlcv(token_address, days)
        
        if ohlcv is None or len(ohlcv) == 0:
            print(f"[DataCollector] No OHLCV data for {token_address}")
            return pd.DataFrame()
        
        # 2. Get token metadata
        metadata = self._fetch_token_metadata(token_address)
        
        # 3. Get SOL price history
        sol_prices = self._fetch_sol_prices(days)
        
        # 4. Merge data
        df = ohlcv.copy()
        df['token_address'] = token_address
        df['symbol'] = metadata.get('symbol', 'UNKNOWN')
        
        # Add SOL prices
        df = df.merge(sol_prices, on='timestamp', how='left')
        df['sol_price'] = df['sol_price'].fillna(method='ffill')
        
        # Add metadata
        df['market_cap'] = metadata.get('market_cap', 0)
        df['liquidity'] = metadata.get('liquidity', 0)
        df['holders'] = metadata.get('holders', 0)
        df['token_age'] = metadata.get('token_age', 365)
        df['top_holder_share'] = metadata.get('top_holder_share', 0)
        
        # 5. Add sentiment data (placeholder - will be enriched later)
        df['sentiment_score'] = 0.0
        df['social_volume'] = 0
        df['news_sentiment'] = 0.0
        df['influencer_mentions'] = 0
        df['whale_activity'] = 0
        df['sector_performance'] = 0.0
        
        print(f"[DataCollector] Collected {len(df)} rows for {metadata.get('symbol', 'UNKNOWN')}")
        
        return df
    
    def _fetch_ohlcv(self, token_address: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Birdeye"""
        if not self.birdeye_api_key:
            print("[DataCollector] No Birdeye API key, using mock data")
            return self._generate_mock_ohlcv(days)
        
        url = f"{self.base_url}/defi/ohlcv"
        params = {
            'address': token_address,
            'type': '1D',  # Daily candles
            'time_from': int((datetime.now() - timedelta(days=days)).timestamp()),
            'time_to': int(datetime.now().timestamp())
        }
        headers = {'X-API-KEY': self.birdeye_api_key}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success') or not data.get('data', {}).get('items'):
                return None
            
            items = data['data']['items']
            df = pd.DataFrame(items)
            df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"[DataCollector] Error fetching OHLCV: {e}")
            return None
    
    def _fetch_token_metadata(self, token_address: str) -> Dict:
        """Fetch token metadata from Birdeye"""
        if not self.birdeye_api_key:
            return {
                'symbol': 'MOCK',
                'market_cap': 50_000_000,
                'liquidity': 500_000,
                'holders': 5000,
                'token_age': 180,
                'top_holder_share': 0.15
            }
        
        url = f"{self.base_url}/defi/token_overview"
        params = {'address': token_address}
        headers = {'X-API-KEY': self.birdeye_api_key}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                return {}
            
            token_data = data.get('data', {})
            
            return {
                'symbol': token_data.get('symbol', 'UNKNOWN'),
                'market_cap': token_data.get('mc', 0),
                'liquidity': token_data.get('liquidity', 0),
                'holders': token_data.get('holder', 0),
                'token_age': self._calculate_token_age(token_data.get('createdAt')),
                'top_holder_share': token_data.get('top10HolderPercent', 0) / 100 if token_data.get('top10HolderPercent') else 0
            }
            
        except Exception as e:
            print(f"[DataCollector] Error fetching metadata: {e}")
            return {}

    def _fetch_sol_prices(self, days: int) -> pd.DataFrame:
        """Fetch SOL price history"""
        # SOL address on Solana
        sol_address = 'So11111111111111111111111111111111111111112'

        if not self.birdeye_api_key:
            # Generate mock SOL prices
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            base_price = 100
            prices = base_price + np.cumsum(np.random.randn(days) * 5)
            return pd.DataFrame({'timestamp': dates, 'sol_price': prices})

        url = f"{self.base_url}/defi/ohlcv"
        params = {
            'address': sol_address,
            'type': '1D',
            'time_from': int((datetime.now() - timedelta(days=days)).timestamp()),
            'time_to': int(datetime.now().timestamp())
        }
        headers = {'X-API-KEY': self.birdeye_api_key}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get('success') or not data.get('data', {}).get('items'):
                return pd.DataFrame()

            items = data['data']['items']
            df = pd.DataFrame(items)
            df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
            df['sol_price'] = df['c']  # Close price
            df = df[['timestamp', 'sol_price']]
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"[DataCollector] Error fetching SOL prices: {e}")
            return pd.DataFrame()

    def _calculate_token_age(self, created_at: Optional[int]) -> int:
        """Calculate token age in days"""
        if not created_at:
            return 365  # Default to 1 year

        created_date = datetime.fromtimestamp(created_at)
        age_days = (datetime.now() - created_date).days
        return max(age_days, 0)

    def _generate_mock_ohlcv(self, days: int) -> pd.DataFrame:
        """Generate mock OHLCV data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate realistic price movement
        base_price = 1.0
        returns = np.random.randn(days) * 0.05  # 5% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(days) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(days)) * 0.02),
            'low': prices * (1 - np.abs(np.random.randn(days)) * 0.02),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, days)
        })

        return df

    def collect_multiple_tokens(
        self,
        token_addresses: List[str],
        days: int = 180
    ) -> pd.DataFrame:
        """
        Collect data for multiple tokens

        Args:
            token_addresses: List of token addresses
            days: Number of days of historical data

        Returns:
            Combined DataFrame with all tokens
        """
        all_data = []

        for i, address in enumerate(token_addresses):
            print(f"[DataCollector] Processing token {i+1}/{len(token_addresses)}")

            df = self.collect_token_data(address, days)

            if len(df) > 0:
                all_data.append(df)

            # Rate limiting
            if self.birdeye_api_key and i < len(token_addresses) - 1:
                time.sleep(1)  # 1 second between requests

        if len(all_data) == 0:
            print("[DataCollector] No data collected")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        print(f"[DataCollector] Total rows collected: {len(combined)}")

        return combined

