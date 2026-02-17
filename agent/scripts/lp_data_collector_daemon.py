#!/usr/bin/env python3
"""
LP Data Collector Daemon

Runs hourly to collect pool snapshots for ML training.
30-day collection period = 720 snapshots.

Usage:
    # Start collection (runs in background)
    python lp_data_collector_daemon.py start
    
    # Check status
    python lp_data_collector_daemon.py status
    
    # Stop collection
    python lp_data_collector_daemon.py stop
"""
import asyncio
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import signal

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.lp_rebalancer.collector import PoolDataCollector

DATA_DIR = Path(__file__).parent.parent / "data" / "lp_rebalancer"
STATUS_FILE = DATA_DIR / "collection_status.json"
SNAPSHOTS_FILE = DATA_DIR / "pool_snapshots.jsonl"
PID_FILE = DATA_DIR / "collector.pid"

# Collection parameters
INTERVAL_SECONDS = 3600  # 1 hour
TARGET_SNAPSHOTS = 720   # 30 days * 24 hours


def print_banner():
    print("\n" + "📊" * 30)
    print("  LP POOL DATA COLLECTOR DAEMON")
    print("📊" * 30)


def get_status() -> dict:
    """Get current collection status."""
    if not STATUS_FILE.exists():
        return {
            "running": False,
            "snapshots_collected": 0,
            "target_snapshots": TARGET_SNAPSHOTS,
            "start_time": None,
            "last_collection": None,
            "estimated_completion": None,
        }
    
    with open(STATUS_FILE) as f:
        return json.load(f)


def save_status(status: dict):
    """Save collection status."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def count_snapshots() -> int:
    """Count collected snapshots."""
    if not SNAPSHOTS_FILE.exists():
        return 0
    
    count = 0
    with open(SNAPSHOTS_FILE) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


async def collect_once():
    """Collect a single snapshot."""
    collector = PoolDataCollector()
    snapshot = await collector.collect_snapshot()
    
    # Append to JSONL file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOTS_FILE, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
    
    return snapshot


async def run_daemon():
    """Run the collection daemon."""
    print_banner()
    print(f"\nStarting collection at {datetime.now(timezone.utc).isoformat()}")
    print(f"Target: {TARGET_SNAPSHOTS} snapshots (30 days)")
    print(f"Interval: {INTERVAL_SECONDS} seconds (1 hour)")
    
    # Initialize status
    status = {
        "running": True,
        "snapshots_collected": count_snapshots(),
        "target_snapshots": TARGET_SNAPSHOTS,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "last_collection": None,
        "estimated_completion": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    }
    save_status(status)
    
    # Save PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    
    try:
        while status["snapshots_collected"] < TARGET_SNAPSHOTS:
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] Collecting snapshot...")
            
            try:
                snapshot = await collect_once()
                status["snapshots_collected"] = count_snapshots()
                status["last_collection"] = datetime.now(timezone.utc).isoformat()
                status["pools_in_last_snapshot"] = snapshot.get("pool_count", 0)
                save_status(status)
                
                print(f"   ✅ Collected {snapshot.get('pool_count', 0)} pools")
                print(f"   📊 Progress: {status['snapshots_collected']}/{TARGET_SNAPSHOTS}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Wait for next interval
            remaining = TARGET_SNAPSHOTS - status["snapshots_collected"]
            if remaining > 0:
                print(f"   ⏰ Next collection in {INTERVAL_SECONDS//60} minutes...")
                await asyncio.sleep(INTERVAL_SECONDS)
        
        print(f"\n✅ Collection complete! {status['snapshots_collected']} snapshots collected.")
        status["running"] = False
        status["completed"] = True
        save_status(status)
        
    except asyncio.CancelledError:
        print("\n⚠️ Collection stopped by user")
        status["running"] = False
        save_status(status)
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


def show_status():
    """Display current collection status."""
    print_banner()
    status = get_status()
    
    print(f"\n📊 COLLECTION STATUS")
    print("-" * 40)
    print(f"   Running: {'✅ Yes' if status.get('running') else '❌ No'}")
    print(f"   Snapshots: {status.get('snapshots_collected', 0)}/{status.get('target_snapshots', TARGET_SNAPSHOTS)}")
    
    if status.get('snapshots_collected', 0) > 0:
        progress = status['snapshots_collected'] / TARGET_SNAPSHOTS * 100
        print(f"   Progress: {progress:.1f}%")
    
    if status.get('start_time'):
        print(f"   Started: {status['start_time']}")
    if status.get('last_collection'):
        print(f"   Last collection: {status['last_collection']}")
    if status.get('estimated_completion'):
        print(f"   Est. completion: {status['estimated_completion']}")
    
    # Data file info
    if SNAPSHOTS_FILE.exists():
        size_mb = SNAPSHOTS_FILE.stat().st_size / (1024 * 1024)
        print(f"\n   📁 Data file: {size_mb:.2f} MB")


def stop_daemon():
    """Stop the running daemon."""
    if not PID_FILE.exists():
        print("No daemon running")
        return
    
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped daemon (PID {pid})")
        PID_FILE.unlink()
    except ProcessLookupError:
        print("Daemon not running")
        PID_FILE.unlink()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lp_data_collector_daemon.py [start|status|stop]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        asyncio.run(run_daemon())
    elif command == "status":
        show_status()
    elif command == "stop":
        stop_daemon()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

