from __future__ import annotations
"""
ONNX Runtime Inference Engine.

Provides fast inference for production deployment.
"""
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
import structlog

logger = structlog.get_logger()


class ONNXInference:
    """
    ONNX Runtime inference engine for XGBoost models.
    
    Provides:
    - Fast CPU/GPU inference
    - Model loading and caching
    - Batch prediction support
    """
    
    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None
    ):
        """
        Initialize ONNX inference engine.
        
        Args:
            model_path: Path to ONNX model file
            providers: Execution providers (default: CPU)
        """
        self.model_path = Path(model_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self.session: ort.InferenceSession | None = None
        self.input_name: str = ""
        self.output_names: list[str] = []
        self.logger = logger.bind(component="onnx_inference")
    
    def load(self) -> None:
        """Load ONNX model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        self.logger.info(
            "ONNX model loaded",
            path=str(self.model_path),
            providers=self.providers
        )
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on features.
        
        Args:
            features: Input features (n_samples, n_features)
            
        Returns:
            Model predictions
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Ensure correct dtype
        features = features.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: features}
        )
        
        return outputs[0]
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability predictions (for classifiers).
        
        Returns:
            Probability array (n_samples, n_classes)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        features = features.astype(np.float32)
        
        outputs = self.session.run(
            self.output_names,
            {self.input_name: features}
        )
        
        # Second output is usually probabilities for classifiers
        if len(outputs) > 1:
            return outputs[1]
        return outputs[0]
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata."""
        if self.session is None:
            return {}
        
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in inputs
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in outputs
            ],
            "providers": self.session.get_providers(),
        }


class ModelConverter:
    """Convert XGBoost models to ONNX format."""
    
    @staticmethod
    def convert_xgboost(
        model_path: str | Path,
        output_path: str | Path,
        n_features: int,
        model_type: str = "classifier"
    ) -> None:
        """
        Convert XGBoost model to ONNX.
        
        Args:
            model_path: Path to XGBoost model (.json)
            output_path: Output ONNX path
            n_features: Number of input features
            model_type: "classifier" or "regressor"
        """
        import xgboost as xgb
        from onnxmltools import convert_xgboost as onnx_convert
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        # Load XGBoost model
        if model_type == "classifier":
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
        
        model.load_model(str(model_path))
        
        # Define input type
        initial_type = [("input", FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        onnx_model = onnx_convert(model, initial_types=initial_type)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        logger.info("Model converted to ONNX", output=str(output_path))
