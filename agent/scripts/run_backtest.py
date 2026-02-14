#!/usr/bin/env python3
"""
LP Rebalancer Backtest Script
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime

# Paths
MODEL_PATH = Path("agent/models/lp_rebalancer/lp_rebalancer.json")
FEATURES_PATH = Path("agent/data/lp_rebalancer/features/pool_features.csv")
METADATA_PATH = Path("agent/models/lp_rebalancer/metadata/feature_names.json")

print("=" * 60)
print("🔄 LP REBALANCER BACKTEST")
print("=" * 60)

# Load model
print("\n📦 Loading model...")
booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))
print(f"   ✅ Model loaded: {MODEL_PATH}")

# Load features - exclude future_* columns (data leakage prevention)
with open(METADATA_PATH) as f:
    meta = json.load(f)
FEATURE_COLS = [f["name"] for f in meta["features"]]
print(f"   ✅ Features: {len(FEATURE_COLS)}")

# Load data
print("\n📊 Loading data...")
df = pd.read_csv(FEATURES_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
print(f"   ✅ Rows: {len(df):,}")
print(f"   ✅ Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"   ✅ Pools: {df['pool_name'].nunique()}")

# Fill NaN with 0 for features
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

# Split: use last 20% as test
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy()
print(f"\n🧪 Test set: {len(test_df):,} rows")

# Get predictions
X_test = test_df[FEATURE_COLS].values
y_true = test_df['label'].values
dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)
y_prob = booster.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)

# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("\n" + "=" * 60)
print("📈 CLASSIFICATION METRICS")
print("=" * 60)
print(f"   Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"   Precision: {precision_score(y_true, y_pred):.4f}")
print(f"   Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"   F1 Score:  {f1_score(y_true, y_pred):.4f}")
print(f"   ROC AUC:   {roc_auc_score(y_true, y_prob):.4f}")

# Simulate trading
print("\n" + "=" * 60)
print("💰 TRADING SIMULATION")
print("=" * 60)

INITIAL_CAPITAL = 10000
REBALANCE_COST = 0.003  # 0.3% per rebalance (gas + slippage)
HOURLY_APY = 0.50 / 365 / 24  # Assume 50% APY average

capital = INITIAL_CAPITAL
positions = {}  # pool -> entry_time
trades = []
total_rebalances = 0

test_df = test_df.reset_index(drop=True)
test_df['pred_prob'] = y_prob
test_df['pred'] = y_pred

for pool in test_df['pool_name'].unique():
    pool_data = test_df[test_df['pool_name'] == pool].copy()
    in_position = True  # Start in position
    entry_capital = INITIAL_CAPITAL / 3  # Split across pools
    
    for idx, row in pool_data.iterrows():
        if row['pred'] == 1 and in_position:  # Model says EXIT
            # Simulate exit
            total_rebalances += 1
            in_position = False
        elif row['pred'] == 0 and not in_position:  # Model says STAY
            # Re-enter
            total_rebalances += 1
            in_position = True
        
        # Accumulate fees if in position
        if in_position:
            capital += entry_capital * HOURLY_APY

# Subtract rebalance costs
rebalance_costs = total_rebalances * INITIAL_CAPITAL * REBALANCE_COST / 3
final_capital = capital - rebalance_costs

print(f"   Initial:     ${INITIAL_CAPITAL:,.2f}")
print(f"   Final:       ${final_capital:,.2f}")
print(f"   Net Profit:  ${final_capital - INITIAL_CAPITAL:,.2f}")
print(f"   Return:      {(final_capital/INITIAL_CAPITAL - 1)*100:.2f}%")
print(f"   Rebalances:  {total_rebalances}")
print(f"   Rebal Cost:  ${rebalance_costs:,.2f}")

# Compare with HODL
hodl_capital = INITIAL_CAPITAL * (1 + HOURLY_APY * len(test_df) / 3)
print(f"\n   📊 HODL Return: ${hodl_capital - INITIAL_CAPITAL:,.2f} ({(hodl_capital/INITIAL_CAPITAL-1)*100:.2f}%)")
print(f"   📊 Alpha:       {(final_capital - hodl_capital):+,.2f}")

# Label distribution in predictions
print("\n" + "=" * 60)
print("🎯 PREDICTION DISTRIBUTION")
print("=" * 60)
print(f"   Predicted STAY (0):     {(y_pred == 0).sum():,} ({(y_pred == 0).mean()*100:.1f}%)")
print(f"   Predicted REBALANCE (1): {(y_pred == 1).sum():,} ({(y_pred == 1).mean()*100:.1f}%)")
print(f"   Actual STAY (0):        {(y_true == 0).sum():,} ({(y_true == 0).mean()*100:.1f}%)")
print(f"   Actual REBALANCE (1):   {(y_true == 1).sum():,} ({(y_true == 1).mean()*100:.1f}%)")

print("\n" + "=" * 60)
print("✅ BACKTEST COMPLETE")
print("=" * 60)

