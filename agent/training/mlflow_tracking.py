#!/usr/bin/env python3
"""
MLflow Tracking Integration for Cortex ML Models

Provides experiment tracking, model versioning, and artifact logging.
All metrics and parameters are dynamically captured from training runs.

Usage:
    from mlflow_tracking import MLflowTracker
    
    tracker = MLflowTracker(experiment_name="perps-model")
    with tracker.start_run(run_name="v2.0.0"):
        tracker.log_params(hyperparameters)
        tracker.log_metrics(metrics)
        tracker.log_model(model_path, metadata_path)
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess

# MLflow import with fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not installed. Run: pip install mlflow")


class MLflowTracker:
    """MLflow experiment tracking for ML models."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Where to store artifacts (default: ./mlruns/artifacts)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            f"file://{Path(__file__).parent / 'mlruns'}"
        )
        self.artifact_location = artifact_location
        self.run_id: Optional[str] = None
        self.client: Optional[Any] = None
        
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._setup_experiment()
    
    def _setup_experiment(self) -> None:
        """Create or get existing experiment."""
        if not MLFLOW_AVAILABLE:
            return
            
        self.client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location,
            )
            print(f"✅ Created MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Using existing experiment: {self.experiment_name} (ID: {experiment.experiment_id})")
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run (e.g., "v2.0.0" or timestamp)
            tags: Additional tags for the run
        """
        if not MLFLOW_AVAILABLE:
            return self._dummy_context()
        
        run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        default_tags = {
            "model_type": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
        }
        
        if tags:
            default_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=default_tags)
    
    def _dummy_context(self):
        """Dummy context manager when MLflow is not available."""
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not MLFLOW_AVAILABLE:
            return
        for key, value in params.items():
            # MLflow params must be strings
            mlflow.log_param(key, str(value) if not isinstance(value, (int, float, str, bool)) else value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training metrics."""
        if not MLFLOW_AVAILABLE:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file as an artifact."""
        if not MLFLOW_AVAILABLE or not os.path.exists(local_path):
            return
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Log model files and register in model registry.

        Args:
            model_path: Path to ONNX or pickle model file
            metadata_path: Path to metadata JSON
            calibration_path: Path to calibration JSON
            version: Semantic version (e.g., "2.0.0")

        Returns:
            Dict with model info including version and run_id
        """
        if not MLFLOW_AVAILABLE:
            return {"version": version or "unknown", "run_id": None}

        # Log model file
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model")
            file_size = os.path.getsize(model_path)
            mlflow.log_metric("model_size_bytes", file_size)

        # Log metadata
        if metadata_path and os.path.exists(metadata_path):
            mlflow.log_artifact(metadata_path, "metadata")
            with open(metadata_path) as f:
                metadata = json.load(f)
                # Log key metrics from metadata
                if "metrics" in metadata:
                    for k, v in metadata["metrics"].items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"meta_{k}", v)

        # Log calibration
        if calibration_path and os.path.exists(calibration_path):
            mlflow.log_artifact(calibration_path, "calibration")

        # Set version tag
        version = version or datetime.now().strftime("v%Y%m%d.%H%M%S")
        mlflow.set_tag("model_version", version)
        mlflow.set_tag("model_hash", self._compute_file_hash(model_path))

        run = mlflow.active_run()
        return {
            "version": version,
            "run_id": run.info.run_id if run else None,
            "model_path": model_path,
        }

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        if not os.path.exists(file_path):
            return "unknown"
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def register_model(
        self,
        model_name: str,
        run_id: Optional[str] = None,
        stage: str = "None",
    ) -> Optional[str]:
        """
        Register model in MLflow Model Registry.

        Args:
            model_name: Name for the registered model
            run_id: Run ID to register (default: current run)
            stage: Model stage (None, Staging, Production, Archived)

        Returns:
            Model version string
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return None

        run_id = run_id or (mlflow.active_run().info.run_id if mlflow.active_run() else None)
        if not run_id:
            return None

        try:
            # Create registered model if it doesn't exist
            try:
                self.client.create_registered_model(model_name)
            except Exception:
                pass  # Model already exists

            # Create model version
            model_uri = f"runs:/{run_id}/model"
            mv = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
            )

            # Transition to stage
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=stage,
                )

            print(f"✅ Registered model: {model_name} v{mv.version} ({stage})")
            return mv.version
        except Exception as e:
            print(f"⚠️ Failed to register model: {e}")
            return None

    def get_latest_version(self, model_name: str, stage: str = "Production") -> Optional[Dict]:
        """Get latest model version from registry."""
        if not MLFLOW_AVAILABLE or not self.client:
            return None

        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                v = versions[0]
                return {
                    "version": v.version,
                    "run_id": v.run_id,
                    "stage": v.current_stage,
                    "source": v.source,
                }
        except Exception:
            pass
        return None

    def compare_runs(self, run_ids: List[str]) -> List[Dict]:
        """Compare metrics across multiple runs."""
        if not MLFLOW_AVAILABLE or not self.client:
            return []

        results = []
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                results.append({
                    "run_id": run_id,
                    "run_name": run.info.run_name,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "start_time": run.info.start_time,
                })
            except Exception:
                pass
        return results

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if MLFLOW_AVAILABLE:
            mlflow.end_run(status=status)


def create_tracker(model_type: str) -> MLflowTracker:
    """Factory function to create tracker for a model type."""
    return MLflowTracker(experiment_name=f"cortex-{model_type}")

