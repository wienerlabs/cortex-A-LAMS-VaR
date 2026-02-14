#!/usr/bin/env python3
"""
XGBoost → ONNX Export Script

Converts the LP Rebalancer XGBoost model to ONNX format for TypeScript inference.

Requirements:
    pip install xgboost==1.7.6 onnxmltools skl2onnx onnxruntime numpy

Usage:
    python scripts/export_onnx.py
"""
import json
import sys
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent / "models" / "lp_rebalancer"
METADATA_DIR = MODEL_DIR / "metadata"
OUTPUT_PATH = MODEL_DIR / "lp_rebalancer.onnx"

# Feature info
FEATURE_NAMES_PATH = METADATA_DIR / "feature_names.json"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import xgboost
        xgb_version = xgboost.__version__
        print(f"✅ xgboost: {xgb_version}")
    except ImportError:
        missing.append("xgboost==1.7.6")
    
    try:
        import onnxmltools
        print(f"✅ onnxmltools: {onnxmltools.__version__}")
    except ImportError:
        missing.append("onnxmltools")
    
    try:
        import onnxruntime
        print(f"✅ onnxruntime: {onnxruntime.__version__}")
    except ImportError:
        missing.append("onnxruntime")
    
    try:
        from skl2onnx import convert_sklearn
        print("✅ skl2onnx")
    except ImportError:
        missing.append("skl2onnx")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    return True


def load_feature_names():
    """Load feature names from metadata."""
    with open(FEATURE_NAMES_PATH) as f:
        data = json.load(f)
    return [f["name"] for f in data["features"]]


def export_xgboost_to_onnx():
    """Export XGBoost model to ONNX format."""
    import xgboost as xgb
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort
    import re

    print("\n📦 Loading XGBoost model...")

    # Load model from JSON (universal format)
    model_json_path = MODEL_DIR / "lp_rebalancer.json"

    # Get feature names
    feature_names = load_feature_names()
    n_features = len(feature_names)
    print(f"   ✅ Features: {n_features}")

    # Create feature name mapping (original → f0, f1, ...)
    feature_map = {name: f"f{i}" for i, name in enumerate(feature_names)}
    reverse_map = {f"f{i}": name for i, name in enumerate(feature_names)}

    # Load JSON and replace feature names
    with open(model_json_path) as f:
        model_json_str = f.read()

    # Replace feature names with f0, f1, f2... pattern
    for orig_name, new_name in feature_map.items():
        # Match exact feature names in JSON (quoted strings)
        model_json_str = re.sub(
            rf'"{re.escape(orig_name)}"',
            f'"{new_name}"',
            model_json_str
        )

    # Save temporary model with renamed features
    temp_model_path = MODEL_DIR / "temp_model.json"
    with open(temp_model_path, 'w') as f:
        f.write(model_json_str)

    # Load modified model
    booster = xgb.Booster()
    booster.load_model(str(temp_model_path))
    print(f"   ✅ Loaded model with normalized feature names")

    # Clean up temp file
    temp_model_path.unlink()

    # Save feature mapping for TypeScript
    mapping_path = METADATA_DIR / "feature_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({
            "feature_to_index": feature_map,
            "index_to_feature": reverse_map,
            "feature_order": feature_names
        }, f, indent=2)
    print(f"   ✅ Feature mapping saved: {mapping_path}")

    # Define input type for ONNX
    initial_type = [('input', FloatTensorType([None, n_features]))]

    print("\n🔄 Converting to ONNX...")

    # Convert to ONNX
    onnx_model = convert_xgboost(
        booster,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Save ONNX model
    with open(OUTPUT_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   ✅ Saved to: {OUTPUT_PATH}")
    
    # Verify with ONNX Runtime
    print("\n🧪 Verifying ONNX model...")
    session = ort.InferenceSession(str(OUTPUT_PATH))
    
    # Test input
    test_input = np.random.randn(1, n_features).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    result = session.run([output_name], {input_name: test_input})
    print(f"   ✅ Test inference successful")
    print(f"   ✅ Output shape: {result[0].shape}")
    print(f"   ✅ Sample output: {result[0][0]}")
    
    # Model size
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n📊 Model size: {size_mb:.2f} MB")
    
    return True


def main():
    print("=" * 60)
    print("XGBoost → ONNX Export")
    print("=" * 60)
    
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    try:
        success = export_xgboost_to_onnx()
        if success:
            print("\n" + "=" * 60)
            print("✅ ONNX export successful!")
            print(f"   Output: {OUTPUT_PATH}")
            print("=" * 60)
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

