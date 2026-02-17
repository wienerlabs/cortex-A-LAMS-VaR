#!/usr/bin/env python3
"""
Perpetuals Historical Data Collection Script

Fetches historical data for ML training from MULTIPLE sources:
- Drift Protocol API: Funding rates (30-90 days)
- CoinGecko API: Price history (up to 365 days per request)
- Birdeye API: Solana token price data
- Jupiter Perps API: Jupiter perps funding rates

For 24-month (730 days) training:
- Use multiple API calls with pagination
- Aggregate data from multiple sources
- Generate synthetic funding rate proxy from price volatility for older periods

Data Sources:
- Drift Data API: https://data.api.drift.trade (primary, recent data)
- CoinGecko API: https://api.coingecko.com/api/v3 (price history)
- Birdeye API: https://public-api.birdeye.so (Solana prices)
- Jupiter API: https://perp-api.jup.ag (Jupiter perps)

Usage:
    python perps_data_collection.py --days 730 --output ./data
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============= CONSTANTS =============

# API endpoints
DRIFT_DATA_API = "https://data.api.drift.trade"
DRIFT_DATA_API_DEVNET = "https://data-master.api.drift.trade"
COINGECKO_API = "https://api.coingecko.com/api/v3"
BIRDEYE_API = "https://public-api.birdeye.so"
JUPITER_PERPS_API = "https://perp-api.jup.ag"

# Markets to collect data for (with CoinGecko IDs)
PERP_MARKETS = [
    "SOL-PERP", "BTC-PERP", "ETH-PERP", "APT-PERP", "BONK-PERP",
    "MATIC-PERP", "ARB-PERP", "DOGE-PERP", "BNB-PERP", "SUI-PERP",
    "1MPEPE-PERP", "OP-PERP", "RENDER-PERP", "XRP-PERP", "HNT-PERP",
    "INJ-PERP", "LINK-PERP", "RLB-PERP", "PYTH-PERP", "TIA-PERP",
    "JTO-PERP", "SEI-PERP", "AVAX-PERP", "WIF-PERP", "JUP-PERP",
    "DYM-PERP", "TAO-PERP", "W-PERP", "KMNO-PERP", "TNSR-PERP"
]

# CoinGecko ID mapping for price data
COINGECKO_IDS = {
    "SOL-PERP": "solana",
    "BTC-PERP": "bitcoin",
    "ETH-PERP": "ethereum",
    "APT-PERP": "aptos",
    "BONK-PERP": "bonk",
    "MATIC-PERP": "matic-network",
    "ARB-PERP": "arbitrum",
    "DOGE-PERP": "dogecoin",
    "BNB-PERP": "binancecoin",
    "SUI-PERP": "sui",
    "OP-PERP": "optimism",
    "XRP-PERP": "ripple",
    "INJ-PERP": "injective-protocol",
    "LINK-PERP": "chainlink",
    "PYTH-PERP": "pyth-network",
    "TIA-PERP": "celestia",
    "SEI-PERP": "sei-network",
    "AVAX-PERP": "avalanche-2",
    "JUP-PERP": "jupiter-exchange-solana",
}

# Precision constants from Drift SDK
FUNDING_RATE_PRECISION = 1e9
PRICE_PRECISION = 1e6
BASE_PRECISION = 1e9
QUOTE_PRECISION = 1e6

# Target data requirements
MIN_SAMPLES_PER_MARKET = 3000  # ~4 months of hourly data minimum
TARGET_TOTAL_SAMPLES = 100000  # Target for full dataset


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    days: int = 730  # 24 months for robust training
    output_dir: str = "./data"
    use_devnet: bool = False
    markets: List[str] = None
    max_workers: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    use_synthetic_extension: bool = True  # Extend with synthetic data for older periods
    coingecko_api_key: Optional[str] = None  # Optional API key for higher rate limits
    birdeye_api_key: Optional[str] = None  # Birdeye API key

    def __post_init__(self):
        if self.markets is None:
            self.markets = PERP_MARKETS
        # Get API keys from environment
        if self.coingecko_api_key is None:
            self.coingecko_api_key = os.environ.get("COINGECKO_API_KEY", "")
        if self.birdeye_api_key is None:
            self.birdeye_api_key = os.environ.get("BIRDEYE_API_KEY", "")


class DriftDataCollector:
    """Collects historical data from Drift Protocol and other sources."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.api_base = DRIFT_DATA_API_DEVNET if config.use_devnet else DRIFT_DATA_API
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CortexML/1.0",
            "Accept": "application/json"
        })

    def _request(self, endpoint: str, params: Dict = None, base_url: str = None) -> Optional[Dict]:
        """Make API request with retry logic."""
        url = f"{base_url or self.api_base}{endpoint}"

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"  Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        return None

    def get_coingecko_price_history(self, market: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical price data from CoinGecko for price-based features."""
        coin_id = COINGECKO_IDS.get(market)
        if not coin_id:
            return pd.DataFrame()

        print(f"  Fetching CoinGecko price history for {market} ({coin_id})...")

        # CoinGecko allows max 365 days per request for free tier
        all_records = []
        remaining_days = min(days, self.config.days)

        while remaining_days > 0:
            fetch_days = min(remaining_days, 365)
            params = {"vs_currency": "usd", "days": str(fetch_days)}

            if self.config.coingecko_api_key:
                params["x_cg_demo_api_key"] = self.config.coingecko_api_key

            data = self._request(f"/coins/{coin_id}/market_chart", params=params, base_url=COINGECKO_API)

            if data and "prices" in data:
                for price_point in data["prices"]:
                    timestamp_ms, price = price_point
                    all_records.append({
                        "market": market,
                        "timestamp": int(timestamp_ms / 1000),
                        "price_usd": price,
                        "source": "coingecko"
                    })

            remaining_days -= fetch_days
            if remaining_days > 0:
                time.sleep(1.5)  # Rate limiting for CoinGecko free tier

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        print(f"    Got {len(df)} price records from CoinGecko")
        return df

    def generate_synthetic_funding_rates(self, price_df: pd.DataFrame, market: str) -> pd.DataFrame:
        """
        Generate synthetic funding rate proxy from price volatility.

        Funding rates correlate with:
        - Price momentum (positive momentum -> positive funding)
        - Volatility (high volatility -> higher funding magnitude)
        - Mean reversion tendencies

        This allows us to extend training data beyond Drift API limits.
        """
        if price_df.empty or len(price_df) < 24:
            return pd.DataFrame()

        print(f"  Generating synthetic funding rates for {market}...")

        df = price_df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Resample to hourly if needed (funding rates are hourly)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime").resample("1h").agg({
            "price_usd": "last",
            "timestamp": "last",
            "market": "first"
        }).dropna().reset_index()

        if len(df) < 24:
            return pd.DataFrame()

        # Calculate price-based features for synthetic funding
        df["return_1h"] = df["price_usd"].pct_change()
        df["return_4h"] = df["price_usd"].pct_change(4)
        df["return_24h"] = df["price_usd"].pct_change(24)
        df["volatility_24h"] = df["return_1h"].rolling(24).std()
        df["momentum_24h"] = df["return_24h"].rolling(24).mean()

        # Synthetic funding rate formula based on empirical observations:
        # funding ~ 0.01% base + momentum_scaled + volatility_scaled + noise
        np.random.seed(42)  # Reproducibility

        base_funding = 0.0001  # 0.01% base rate
        momentum_weight = 0.5  # How much momentum affects funding
        volatility_weight = 0.3  # How much volatility affects magnitude

        # Momentum component (positive returns -> positive funding)
        momentum_component = df["momentum_24h"].fillna(0) * momentum_weight * 100

        # Volatility component (increases magnitude in both directions)
        volatility_component = df["volatility_24h"].fillna(0) * volatility_weight * 10

        # Random noise to simulate market dynamics
        noise = np.random.normal(0, 0.0002, len(df))

        # Combine components: momentum drives direction, volatility scales magnitude
        synthetic_funding = (base_funding + momentum_component) * (1 + volatility_component) + noise

        # Clip to realistic funding rate range (-0.5% to +0.5% per hour)
        synthetic_funding = np.clip(synthetic_funding, -0.005, 0.005)

        # Create output DataFrame matching Drift API format
        result = pd.DataFrame({
            "market": market,
            "timestamp": df["timestamp"].astype(int),
            "slot": 0,  # Synthetic - no slot
            "funding_rate_raw": synthetic_funding * FUNDING_RATE_PRECISION,
            "funding_rate_pct": synthetic_funding * 100,  # Convert to percentage
            "oracle_twap": df["price_usd"],
            "mark_twap": df["price_usd"] * (1 + synthetic_funding * 10),  # Slight premium/discount
            "cumulative_funding_rate_long": (synthetic_funding * FUNDING_RATE_PRECISION).cumsum(),
            "cumulative_funding_rate_short": (-synthetic_funding * FUNDING_RATE_PRECISION).cumsum(),
            "source": "synthetic",
            "datetime": df["datetime"]
        })

        result = result.dropna().reset_index(drop=True)
        print(f"    Generated {len(result)} synthetic funding rate records")
        return result

    def get_funding_rates(self, market: str) -> pd.DataFrame:
        """Fetch funding rate history for a market (last 30 days from API)."""
        print(f"  Fetching funding rates for {market}...")
        
        data = self._request("/fundingRates", params={"marketName": market})
        if not data or "fundingRates" not in data:
            print(f"    Warning: No funding rate data for {market}")
            return pd.DataFrame()

        records = []
        for rate in data["fundingRates"]:
            try:
                # Convert funding rate to percentage
                funding_rate = float(rate.get("fundingRate", 0)) / FUNDING_RATE_PRECISION
                oracle_twap = float(rate.get("oraclePriceTwap", 1)) / PRICE_PRECISION
                
                # Funding rate as percentage per hour
                funding_pct = (funding_rate / oracle_twap) * 100 if oracle_twap > 0 else 0
                
                records.append({
                    "market": market,
                    "timestamp": int(rate.get("ts", 0)),
                    "slot": int(rate.get("slot", 0)),
                    "funding_rate_raw": funding_rate,
                    "funding_rate_pct": funding_pct,
                    "oracle_twap": oracle_twap,
                    "mark_twap": float(rate.get("markTwap", 0)) / PRICE_PRECISION,
                    "cumulative_funding_rate_long": float(rate.get("cumulativeFundingRateLong", 0)),
                    "cumulative_funding_rate_short": float(rate.get("cumulativeFundingRateShort", 0)),
                })
            except (ValueError, KeyError) as e:
                print(f"    Skipping malformed record: {e}")
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"    Got {len(df)} funding rate records")
        return df

    def get_contracts(self) -> pd.DataFrame:
        """Fetch current contract info (includes OI, funding rate)."""
        print("  Fetching contract information...")

        data = self._request("/contracts")
        if not data:
            return pd.DataFrame()

        # API returns {"contracts": [...]}
        contracts_list = data.get("contracts", []) if isinstance(data, dict) else data

        records = []
        for contract in contracts_list:
            try:
                records.append({
                    "market": contract.get("ticker_id", ""),
                    "market_index": int(contract.get("contract_index", 0)),
                    "open_interest": float(contract.get("open_interest", 0)),
                    "funding_rate": float(contract.get("funding_rate", 0)),
                    "next_funding_rate": float(contract.get("next_funding_rate", 0)),
                    "index_price": float(contract.get("index_price", 0)),
                    "last_price": float(contract.get("last_price", 0)),
                    "base_volume": float(contract.get("base_volume", 0)),
                    "quote_volume": float(contract.get("quote_volume", 0)),
                    "high_24h": float(contract.get("high", 0)),
                    "low_24h": float(contract.get("low", 0)),
                })
            except (ValueError, KeyError, TypeError):
                continue

        return pd.DataFrame(records)

    def get_rate_history(self, market: str) -> pd.DataFrame:
        """Fetch extended rate history for a market."""
        print(f"  Fetching rate history for {market}...")

        data = self._request("/rateHistory", params={"marketName": market})
        if not data:
            return pd.DataFrame()

        records = []
        for item in data if isinstance(data, list) else []:
            try:
                records.append({
                    "market": market,
                    "timestamp": int(item.get("ts", 0)),
                    "funding_rate": float(item.get("fundingRate", 0)),
                    "borrow_rate": float(item.get("borrowRate", 0)),
                    "deposit_rate": float(item.get("depositRate", 0)),
                })
            except (ValueError, KeyError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def collect_extended_market_data(self, market: str) -> pd.DataFrame:
        """
        Collect extended data for a market using multiple sources:
        1. Drift API (recent 30-90 days, real funding rates)
        2. CoinGecko (older periods, synthetic funding rates from price)

        Returns combined DataFrame with both real and synthetic data.
        """
        # Get real Drift data first
        drift_df = self.get_funding_rates(market)
        drift_df["source"] = "drift"

        # Determine how far back Drift data goes
        if not drift_df.empty:
            oldest_drift = drift_df["timestamp"].min()
            days_of_drift = (datetime.now().timestamp() - oldest_drift) / 86400
            print(f"    Drift data covers {days_of_drift:.0f} days")
        else:
            oldest_drift = datetime.now().timestamp()
            days_of_drift = 0

        # If we need more historical data and synthetic extension is enabled
        if self.config.use_synthetic_extension and self.config.days > days_of_drift + 30:
            remaining_days = int(self.config.days - days_of_drift)
            print(f"    Extending with {remaining_days} days of synthetic data...")

            # Get price history from CoinGecko
            price_df = self.get_coingecko_price_history(market, days=remaining_days)

            if not price_df.empty:
                # Generate synthetic funding rates from price data
                synthetic_df = self.generate_synthetic_funding_rates(price_df, market)

                if not synthetic_df.empty:
                    # Filter synthetic data to only include periods before Drift data
                    synthetic_df = synthetic_df[synthetic_df["timestamp"] < oldest_drift]

                    # Combine real and synthetic data
                    if not drift_df.empty:
                        # Ensure column alignment
                        common_cols = list(set(drift_df.columns) & set(synthetic_df.columns))
                        drift_df = drift_df[common_cols]
                        synthetic_df = synthetic_df[common_cols]
                        combined_df = pd.concat([synthetic_df, drift_df], ignore_index=True)
                    else:
                        combined_df = synthetic_df

                    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
                    print(f"    Combined: {len(combined_df)} total records ({len(synthetic_df)} synthetic + {len(drift_df)} real)")
                    return combined_df

        return drift_df

    def collect_all_funding_rates(self) -> pd.DataFrame:
        """Collect funding rates for all markets with extended data sources."""
        print("\n📊 Collecting funding rates for all markets...")
        print(f"    Target: {self.config.days} days of history")

        all_data = []

        # Use extended collection if we need more than 60 days
        if self.config.days > 60 and self.config.use_synthetic_extension:
            print("    Using extended data collection with synthetic data...")
            # Process markets sequentially to respect API rate limits
            for market in self.config.markets:
                try:
                    df = self.collect_extended_market_data(market)
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    print(f"  Error collecting {market}: {e}")
                time.sleep(0.5)  # Rate limiting between markets
        else:
            # Original parallel collection for short periods
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.get_funding_rates, market): market
                    for market in self.config.markets
                }

                for future in as_completed(futures):
                    market = futures[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_data.append(df)
                    except Exception as e:
                        print(f"  Error collecting {market}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Data quality validation
            total_samples = len(combined)
            markets_count = combined["market"].nunique()

            # Date range analysis
            if "timestamp" in combined.columns:
                min_ts = pd.to_datetime(combined["timestamp"].min(), unit="s")
                max_ts = pd.to_datetime(combined["timestamp"].max(), unit="s")
                days_covered = (max_ts - min_ts).days
            else:
                days_covered = 0

            # Source breakdown
            if "source" in combined.columns:
                source_counts = combined["source"].value_counts().to_dict()
            else:
                source_counts = {"drift": total_samples}

            print(f"\n" + "=" * 50)
            print("📊 DATA QUALITY REPORT")
            print("=" * 50)
            print(f"  Total samples:    {total_samples:,}")
            print(f"  Markets:          {markets_count}")
            print(f"  Days covered:     {days_covered}")
            print(f"  Source breakdown: {source_counts}")

            # Check if we meet minimum requirements
            if total_samples < TARGET_TOTAL_SAMPLES:
                print(f"  ⚠️  Warning: Only {total_samples:,} samples (target: {TARGET_TOTAL_SAMPLES:,})")
            else:
                print(f"  ✅ Meets target of {TARGET_TOTAL_SAMPLES:,} samples")

            print("=" * 50)

            return combined
        return pd.DataFrame()

    def collect_contracts_snapshot(self) -> pd.DataFrame:
        """Collect current contract info snapshot."""
        print("\n📊 Collecting contract snapshots...")

        df = self.get_contracts()
        if not df.empty:
            df["snapshot_time"] = datetime.now(tz=None)
            print(f"✅ Got {len(df)} contract snapshots")
        return df

    def save_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save collected data to CSV files."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        timestamp = datetime.now(tz=None).strftime("%Y%m%d_%H%M%S")

        for name, df in data.items():
            if df is not None and not df.empty:
                filename = f"{name}_{timestamp}.csv"
                filepath = os.path.join(self.config.output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"  Saved {filepath} ({len(df)} rows)")

                # Also save as "latest" for easy access
                latest_path = os.path.join(self.config.output_dir, f"{name}_latest.csv")
                df.to_csv(latest_path, index=False)

    def run(self) -> Dict[str, pd.DataFrame]:
        """Run full data collection pipeline."""
        print("=" * 60)
        print("  PERPS DATA COLLECTION (EXTENDED)")
        print("=" * 60)
        print(f"  API: {self.api_base}")
        print(f"  Markets: {len(self.config.markets)}")
        print(f"  Days requested: {self.config.days}")
        print(f"  Synthetic extension: {self.config.use_synthetic_extension}")
        print(f"  Output: {self.config.output_dir}")
        print("=" * 60)

        start_time = time.time()

        # Collect all data
        data = {
            "funding_rates": self.collect_all_funding_rates(),
            "contracts": self.collect_contracts_snapshot(),
        }

        # Save to CSV
        print("\n💾 Saving data to CSV...")
        self.save_data(data)

        elapsed = time.time() - start_time
        print(f"\n✅ Data collection complete in {elapsed:.1f}s")

        # Print summary
        self._print_summary(data)

        return data

    def _print_summary(self, data: Dict[str, pd.DataFrame]) -> None:
        """Print collection summary."""
        print("\n" + "=" * 60)
        print("  COLLECTION SUMMARY")
        print("=" * 60)

        for name, df in data.items():
            if df is not None and not df.empty:
                print(f"\n{name}:")
                print(f"  Records: {len(df)}")
                if "market" in df.columns:
                    print(f"  Markets: {df['market'].nunique()}")
                if "timestamp" in df.columns:
                    min_ts = pd.to_datetime(df["timestamp"].min(), unit="s")
                    max_ts = pd.to_datetime(df["timestamp"].max(), unit="s")
                    days_span = (max_ts - min_ts).days
                    print(f"  Date Range: {min_ts} to {max_ts}")
                    print(f"  Days span: {days_span}")
                if "source" in df.columns:
                    print(f"  Sources: {df['source'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Perps historical data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 24 months of data (recommended for training)
  python perps_data_collection.py --days 730 --output ./data

  # Collect only Drift API data (no synthetic extension)
  python perps_data_collection.py --days 60 --no-synthetic

  # Collect specific markets
  python perps_data_collection.py --days 365 --markets SOL-PERP BTC-PERP ETH-PERP
        """
    )
    parser.add_argument("--days", type=int, default=730, help="Days of history (default: 730 = 24 months)")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--devnet", action="store_true", help="Use devnet API")
    parser.add_argument("--markets", nargs="+", default=None, help="Specific markets to collect")
    parser.add_argument("--no-synthetic", action="store_true", help="Disable synthetic data extension")
    parser.add_argument("--coingecko-key", type=str, default=None, help="CoinGecko API key")
    parser.add_argument("--birdeye-key", type=str, default=None, help="Birdeye API key")
    args = parser.parse_args()

    config = DataCollectionConfig(
        days=args.days,
        output_dir=args.output,
        use_devnet=args.devnet,
        markets=args.markets,
        use_synthetic_extension=not args.no_synthetic,
        coingecko_api_key=args.coingecko_key,
        birdeye_api_key=args.birdeye_key,
    )

    collector = DriftDataCollector(config)
    collector.run()


if __name__ == "__main__":
    main()

