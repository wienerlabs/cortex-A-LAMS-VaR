"""
Collect Spot Trading Training Data
Fetches historical data for approved tokens and saves to CSV
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.spot_data_collector import SpotDataCollector


# Top Solana tokens for training data
# These are established tokens with good liquidity and history
TRAINING_TOKENS = [
    # Tier 1: Major tokens
    'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',  # JUP
    'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # mSOL
    'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn', # JitoSOL
    '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs', # ORCA
    'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', # USDC
    
    # Tier 2: Mid-cap tokens
    'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263', # BONK
    '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr', # POPCAT
    'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3', # PYTH
    'DFL1zNkaGPWm1BqAVqRjCZvHmwTFrEaJtbzJWgseoNJh', # DRIFT
    
    # Tier 3: Smaller but established tokens
    'kinXdEcpDQeHPEuQnqmUgtYykqKGVFq6CeVX5iAHJq6',  # KIN
    'SHDWyBxihqiCj6YekG2GUr7wqKLeLAMK1gHZck9pL6y',  # SHDW
]


def main():
    """Collect training data for spot trading model"""
    print(f"\n{'='*60}")
    print(f"SPOT TRADING DATA COLLECTION")
    print(f"{'='*60}\n")
    
    # Initialize collector
    birdeye_api_key = os.getenv('BIRDEYE_API_KEY')
    
    if not birdeye_api_key:
        print("⚠️  WARNING: No BIRDEYE_API_KEY found in environment")
        print("   Using mock data for demonstration")
        print("   Set BIRDEYE_API_KEY for real data collection\n")
    
    collector = SpotDataCollector(birdeye_api_key)
    
    # Collect data
    print(f"Collecting data for {len(TRAINING_TOKENS)} tokens...")
    print(f"Lookback period: 180 days\n")
    
    df = collector.collect_multiple_tokens(
        token_addresses=TRAINING_TOKENS,
        days=180
    )
    
    if len(df) == 0:
        print("\n❌ ERROR: No data collected")
        return
    
    # Save to CSV
    output_dir = 'agent/data/spot'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'spot_training_data_{timestamp}.csv')
    
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"DATA COLLECTION COMPLETE!")
    print(f"{'='*60}\n")
    print(f"Total rows: {len(df)}")
    print(f"Tokens: {df['symbol'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB\n")
    
    # Show sample
    print("Sample data:")
    print(df.head())
    print()
    
    # Create symlink to latest
    latest_path = os.path.join(output_dir, 'spot_training_data.csv')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(output_path), latest_path)
    print(f"Created symlink: {latest_path} -> {os.path.basename(output_path)}\n")


if __name__ == '__main__':
    main()

