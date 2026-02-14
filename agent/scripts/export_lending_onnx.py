#!/usr/bin/env python3
"""
Export Lending Model to ONNX

Converts trained XGBoost lending model to ONNX format for TypeScript inference.

Requirements:
    pip install xgboost onnxmltools skl2onnx onnxruntime

Usage:
    python scripts/export_lending_onnx.py --model models/lending/lending_model_20260119_120000
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import xgboost
        print(f"✅ xgboost: {xgboost.__version__}")
    except ImportError:
        missing.append("xgboost")
    
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


def export_lending_model_to_onnx(model_path: Path, output_path: Path = None):
    """
    Export lending XGBoost model to ONNX format.
    
    Args:
        model_path: Path to saved model directory
        output_path: Optional output path for ONNX file
    """
    import xgboost as xgb
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort
    
    print(f"\n📦 Loading model from: {model_path}")

    # Load the model - handle both directory and file paths
    if model_path.is_dir():
        model_file = model_path / "model.json"
        if not model_file.exists():
            model_file = model_path / "model.ubj"
        meta_file = model_path / "metadata.json"
    else:
        # Model path is a file (without extension)
        model_file = model_path.with_suffix(".json")
        meta_file = model_path.with_suffix(".meta.json")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    booster = xgb.Booster()
    booster.load_model(str(model_file))
    print(f"   ✅ Loaded model: {model_file.name}")

    # Load metadata (contains feature names)
    original_feature_names = []
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
        original_feature_names = metadata.get("feature_names", [])
        n_features = len(original_feature_names)
        print(f"   ✅ Features: {n_features}")
    else:
        # Infer from model
        n_features = booster.num_features()
        original_feature_names = [f"f{i}" for i in range(n_features)]
        print(f"   ⚠️ Feature names not found, using defaults: {n_features} features")

    # XGBoost ONNX converter requires feature names to be f0, f1, f2, etc.
    # We'll rename them for conversion and save the mapping
    numeric_feature_names = [f"f{i}" for i in range(n_features)]
    booster.feature_names = numeric_feature_names

    # Define input type for ONNX
    initial_type = [('input', FloatTensorType([None, n_features]))]

    print("\n🔄 Converting to ONNX...")

    # Convert to ONNX
    onnx_model = convert_xgboost(
        booster,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Determine output path
    if output_path is None:
        if model_path.is_dir():
            output_path = model_path / "lending_model.onnx"
        else:
            output_path = model_path.with_suffix(".onnx")

    # Save ONNX model
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   ✅ Saved to: {output_path}")
    
    # Save feature names for TypeScript (use original names, not f0, f1, etc.)
    if model_path.is_dir():
        feature_metadata_path = model_path / "feature_metadata.json"
    else:
        feature_metadata_path = model_path.with_suffix(".features.json")

    with open(feature_metadata_path, 'w') as f:
        json.dump({
            "feature_names": original_feature_names,
            "n_features": n_features,
            "model_type": "lending",
            "input_name": "input",
            "output_name": "output"
        }, f, indent=2)
    print(f"   ✅ Feature metadata saved: {feature_metadata_path}")
    
    # Verify with ONNX Runtime
    print("\n🧪 Verifying ONNX model...")
    session = ort.InferenceSession(str(output_path))
    
    # Test input
    test_input = np.random.randn(1, n_features).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Run inference
    result = session.run(output_names, {input_name: test_input})
    print(f"   ✅ Test inference successful")
    print(f"   ✅ Outputs: {len(result)}")
    for i, output in enumerate(result):
        print(f"      Output {i}: shape={output.shape}, sample={output[0] if len(output.shape) == 1 else output[0][0]}")
    
    # Model size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n📊 Model size: {size_mb:.2f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export lending XGBoost model to ONNX"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for ONNX file (default: model_dir/lending_model.onnx)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LENDING MODEL → ONNX EXPORT")
    print("=" * 60)

    # Validate model path
    model_path = Path(args.model)

    # Check if path exists (could be directory or file without extension)
    if not model_path.exists():
        # Try with .json extension
        if not model_path.with_suffix(".json").exists():
            print(f"\n❌ Model path not found: {model_path}")
            print(f"   Also tried: {model_path.with_suffix('.json')}")
            sys.exit(1)

    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    # Export
    try:
        output_path = args.output
        if output_path:
            output_path = Path(output_path)

        onnx_path = export_lending_model_to_onnx(model_path, output_path)

        print("\n" + "=" * 60)
        print("✅ ONNX EXPORT SUCCESSFUL!")
        print(f"   Model: {onnx_path}")
        print("=" * 60)

        print("\n📝 Next steps:")
        print("   1. Copy ONNX model to TypeScript project:")
        print(f"      cp {onnx_path} agent/eliza/src/models/lending/")
        print("   2. Copy feature metadata:")
        print(f"      cp {model_path}/feature_metadata.json agent/eliza/src/models/lending/")
        print("   3. Integrate with lending service")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


