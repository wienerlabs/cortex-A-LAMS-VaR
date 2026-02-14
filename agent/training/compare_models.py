#!/usr/bin/env python3
"""
Model Version Comparison Tool

CLI tool for comparing ML model versions using MLflow tracking data.

Usage:
    python compare_models.py --model perps --versions v1,v2
    python compare_models.py --model spot --latest 3
    python compare_models.py --list perps
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mlflow_tracking import create_tracker, MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def list_versions(model_type: str) -> List[Dict[str, Any]]:
    """List all versions for a model type."""
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available. Reading from local metadata files...")
        return list_versions_from_files(model_type)
    
    tracker = create_tracker(model_type)
    # Get all runs for the experiment
    try:
        import mlflow
        experiment = mlflow.get_experiment_by_name(f"cortex-{model_type}")
        if not experiment:
            print(f"❌ No experiment found for {model_type}")
            return []
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        versions = []
        for _, run in runs.iterrows():
            versions.append({
                "run_id": run["run_id"],
                "run_name": run.get("tags.mlflow.runName", "unknown"),
                "start_time": run["start_time"],
                "status": run["status"],
                "metrics": {k.replace("metrics.", ""): v for k, v in run.items() if k.startswith("metrics.")},
            })
        return versions
    except Exception as e:
        print(f"❌ Failed to list versions: {e}")
        return list_versions_from_files(model_type)


def list_versions_from_files(model_type: str) -> List[Dict[str, Any]]:
    """List versions from local metadata files."""
    models_dir = Path(__file__).parent.parent / "models"
    
    # Map model type to possible metadata file patterns
    patterns = {
        "perps": ["perps_metadata.json", "perps/metadata.json"],
        "spot": ["spot/model_metadata.json", "spot_metadata.json"],
        "lending": ["lending/metadata.json", "lending_metadata.json"],
        "lp": ["metadata/model_config.json"],
    }
    
    versions = []
    for pattern in patterns.get(model_type, []):
        metadata_path = models_dir / pattern
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            versions.append({
                "path": str(metadata_path),
                "version": data.get("version", "unknown"),
                "trained_at": data.get("trained_at", data.get("timestamp", "unknown")),
                "metrics": data.get("metrics", data.get("test_metrics", {})),
            })
    
    return versions


def compare_versions(model_type: str, version_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple model versions."""
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available. Comparison limited.")
        return {}
    
    tracker = create_tracker(model_type)
    comparison = tracker.compare_runs(version_ids)
    
    return {
        "model_type": model_type,
        "versions_compared": len(comparison),
        "runs": comparison,
    }


def print_version_table(versions: List[Dict[str, Any]]) -> None:
    """Print versions in a formatted table."""
    if not versions:
        print("No versions found.")
        return
    
    print(f"\n{'='*80}")
    print(f"{'Version':<20} {'Date':<25} {'Status':<12}")
    print(f"{'='*80}")
    
    for v in versions[:10]:  # Show last 10
        version = v.get("run_name", v.get("version", "unknown"))
        date = str(v.get("start_time", v.get("trained_at", "unknown")))[:25]
        status = v.get("status", "local")
        print(f"{version:<20} {date:<25} {status:<12}")
    
    print(f"{'='*80}\n")


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print comparison results."""
    if not comparison or not comparison.get("runs"):
        print("No comparison data available.")
        return
    
    runs = comparison["runs"]
    
    print(f"\n{'='*80}")
    print(f"Model Comparison: {comparison['model_type']}")
    print(f"{'='*80}")
    
    # Collect all metrics
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.get("metrics", {}).keys())
    
    # Print header
    header = f"{'Metric':<20}"
    for run in runs:
        header += f" | {run.get('run_name', 'unknown')[:15]:<15}"
    print(header)
    print("-" * len(header))
    
    # Print metrics
    for metric in sorted(all_metrics):
        row = f"{metric:<20}"
        for run in runs:
            value = run.get("metrics", {}).get(metric, "N/A")
            if isinstance(value, float):
                row += f" | {value:>15.4f}"
            else:
                row += f" | {str(value):>15}"
        print(row)
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare ML model versions")
    parser.add_argument("--model", "-m", type=str, required=True, 
                        choices=["perps", "spot", "lending", "lp"],
                        help="Model type to compare")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all versions")
    parser.add_argument("--versions", "-v", type=str,
                        help="Comma-separated run IDs to compare")
    parser.add_argument("--latest", "-n", type=int, default=3,
                        help="Compare N latest versions")
    
    args = parser.parse_args()
    
    if args.list:
        versions = list_versions(args.model)
        print_version_table(versions)
    elif args.versions:
        version_ids = [v.strip() for v in args.versions.split(",")]
        comparison = compare_versions(args.model, version_ids)
        print_comparison(comparison)
    else:
        # Compare latest N versions
        versions = list_versions(args.model)
        if versions:
            print_version_table(versions)


if __name__ == "__main__":
    main()

