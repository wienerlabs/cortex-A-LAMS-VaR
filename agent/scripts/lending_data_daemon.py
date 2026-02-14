#!/usr/bin/env python3
"""
Lending Data Collection Daemon

Continuously collects lending data at regular intervals.
Useful for building a real-time dataset for model training.

Usage:
    python lending_data_daemon.py --interval 3600  # Collect every hour
"""
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add agent root to path
agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

from collect_lending_data import LendingDataCollector, DATA_DIR


class LendingDataDaemon:
    """Daemon for continuous lending data collection."""
    
    def __init__(self, interval_seconds: int = 3600):
        self.interval = interval_seconds
        self.running = False
        self.data_buffer = []
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n\n🛑 Shutdown signal received, saving data...")
        self.running = False
    
    async def run(self):
        """Run the daemon."""
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        iteration = 0
        
        print("=" * 60)
        print("LENDING DATA COLLECTION DAEMON")
        print("=" * 60)
        print(f"Collection interval: {self.interval}s ({self.interval/3600:.1f}h)")
        print(f"Data directory: {DATA_DIR}")
        print("\nPress Ctrl+C to stop and save data\n")
        
        async with LendingDataCollector() as collector:
            while self.running:
                iteration += 1
                timestamp = datetime.now(timezone.utc)
                
                print(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Iteration {iteration}")
                print("-" * 60)
                
                try:
                    # Collect snapshot
                    data = await collector.collect_snapshot()
                    
                    if data:
                        self.data_buffer.extend(data)
                        print(f"✅ Collected {len(data)} data points")
                        print(f"   Total in buffer: {len(self.data_buffer)}")
                        
                        # Save every 24 iterations (24 hours if hourly)
                        if iteration % 24 == 0:
                            self._save_buffer(collector)
                    else:
                        print("⚠️ No data collected this iteration")
                
                except Exception as e:
                    print(f"❌ Error during collection: {e}")
                
                # Wait for next iteration
                if self.running:
                    print(f"\n⏳ Waiting {self.interval}s until next collection...")
                    await asyncio.sleep(self.interval)
            
            # Save remaining data on shutdown
            if self.data_buffer:
                self._save_buffer(collector)
        
        print("\n✅ Daemon stopped gracefully")
    
    def _save_buffer(self, collector: LendingDataCollector):
        """Save buffered data to file."""
        if not self.data_buffer:
            return
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"lending_daemon_{timestamp}.csv"
        
        print(f"\n💾 Saving {len(self.data_buffer)} records...")
        collector.save_data(self.data_buffer, filename)
        
        # Clear buffer
        self.data_buffer = []


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Lending data collection daemon"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Collection interval in seconds (default: 3600 = 1 hour)"
    )
    
    args = parser.parse_args()
    
    daemon = LendingDataDaemon(interval_seconds=args.interval)
    await daemon.run()


if __name__ == "__main__":
    asyncio.run(main())

