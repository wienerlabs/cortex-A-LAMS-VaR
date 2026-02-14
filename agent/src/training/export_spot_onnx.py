"""
Export Spot Trading XGBoost Model to ONNX
Converts trained XGBoost model to ONNX format for TypeScript inference
"""

import xgboost as xgb
import onnxmltools
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import json
import os
import sys


def export_to_onnx(
    model_path: str = 'agent/eliza/src/models/spot/spot_model.json',
    output_path: str = 'agent/eliza/src/models/spot/spot_model.onnx',
    feature_names_path: str = 'agent/eliza/src/models/spot/feature_names.json'
):
    """
    Export XGBoost model to ONNX format
    
    Args:
        model_path: Path to XGBoost model JSON file
        output_path: Path to save ONNX model
        feature_names_path: Path to feature names JSON
    """
    print(f"\n{'='*60}")
    print(f"EXPORTING SPOT MODEL TO ONNX")
    print(f"{'='*60}\n")
    
    # 1. Load XGBoost model
    print(f"[1/4] Loading XGBoost model from {model_path}...")
    model = xgb.Booster()
    model.load_model(model_path)
    print("  Model loaded successfully")
    
    # 2. Load feature names
    print(f"\n[2/4] Loading feature names from {feature_names_path}...")
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    n_features = len(feature_names)
    print(f"  Loaded {n_features} feature names")
    
    # 3. Convert to ONNX
    print(f"\n[3/4] Converting to ONNX format...")

    # Use onnxmltools FloatTensorType
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
    initial_type = [('input', OnnxFloatTensorType([None, n_features]))]

    try:
        # Set feature names to f0, f1, f2, ... for ONNX conversion
        model.feature_names = [f'f{i}' for i in range(n_features)]

        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=12
        )
        print("  Conversion successful")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    
    # 4. Save ONNX model
    print(f"\n[4/4] Saving ONNX model to {output_path}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"  Saved ONNX model ({file_size:.1f} KB)")
    
    # 5. Verify ONNX model
    print(f"\n[5/5] Verifying ONNX model...")
    try:
        import onnx
        onnx_model_check = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_check)
        print("  ONNX model is valid ✓")
    except Exception as e:
        print(f"  WARNING: ONNX validation failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE!")
    print(f"{'='*60}\n")
    print(f"ONNX model ready for TypeScript inference:")
    print(f"  {output_path}")
    print(f"\nFeature names:")
    print(f"  {feature_names_path}")
    print()


if __name__ == '__main__':
    export_to_onnx()

