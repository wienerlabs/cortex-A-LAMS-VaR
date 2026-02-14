#!/usr/bin/env python3
"""
Train Lending Strategy Model

Trains XGBoost classifier for Solana lending protocol selection.

Usage:
    # Train with default settings
    python train_lending_model.py --data data/lending/lending_historical_simulated.csv
    
    # Train with cross-validation
    python train_lending_model.py --data data/lending/lending_historical_simulated.csv --cv
    
    # Train without saving
    python train_lending_model.py --data data/lending/lending_historical_simulated.csv --no-save
"""
import argparse
import sys
from pathlib import Path

# Add agent root to path
agent_root = Path(__file__).parent.parent
sys.path.insert(0, str(agent_root))

from src.models.lending import LendingTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Solana lending strategy model"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to lending data CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/lending",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Perform cross-validation"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )
    
    args = parser.parse_args()
    
    # Validate data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("\n💡 First collect data using:")
        print("   python scripts/collect_lending_data.py --mode historical")
        sys.exit(1)
    
    print("=" * 60)
    print("SOLANA LENDING STRATEGY MODEL TRAINING")
    print("=" * 60)
    print(f"\nData: {data_path}")
    print(f"Output: {args.output}")
    print(f"Cross-validation: {args.cv}")
    print()
    
    # Create trainer
    trainer = LendingTrainer(model_dir=args.output)
    
    if args.cv:
        # Cross-validation mode
        print("\n🔄 Running cross-validation...")
        fold_metrics = trainer.cross_validate(
            data_path=data_path,
            n_splits=args.cv_splits
        )
        
        # Print results
        print("\n📊 Cross-Validation Results:")
        print("-" * 60)
        for metric, values in fold_metrics.items():
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            print(f"{metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Train final model
    print("\n🏋️ Training final model...")
    model, results = trainer.train(
        data_path=data_path,
        save=not args.no_save
    )
    
    # Print results
    print("\n📊 Training Results:")
    print("-" * 60)
    print("\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric:15s}: {value:.4f}")
    
    print("\n🎯 Top 10 Feature Importances:")
    feature_importance = results['feature_importance']
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for feature, importance in sorted_features:
        print(f"  {feature:30s}: {importance:.4f}")
    
    if results['model_path']:
        print(f"\n💾 Model saved to: {results['model_path']}")
    
    print("\n✅ Training complete!")
    print("\n📝 Next steps:")
    print("   1. Export model to ONNX:")
    print(f"      python scripts/export_onnx.py --model {results['model_path']} --type lending")
    print("   2. Integrate with TypeScript lending service")


if __name__ == "__main__":
    main()

