"""
Spot Trading Label Generator
Generates binary labels for XGBoost training based on future price movement
"""

import pandas as pd
import numpy as np
from typing import Tuple


class SpotLabelGenerator:
    """Generate labels for spot trading ML model"""
    
    def __init__(
        self,
        target_profit_pct: float = 0.12,  # TP1 target: +12%
        stop_loss_pct: float = 0.08,      # Stop loss: -8%
        lookforward_days: int = 7          # Look forward 7 days
    ):
        self.target_profit_pct = target_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.lookforward_days = lookforward_days
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary labels based on future price movement
        
        Label = 1 (BUY) if:
        - Price reaches +12% within 7 days AND
        - Price doesn't drop >8% before reaching target
        
        Label = 0 (NO BUY) if:
        - Price drops >8% OR
        - Price stays flat (doesn't reach +12%)
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with 'label' column added
        """
        df = df.copy()
        labels = []
        
        for i in range(len(df)):
            # Can't label last N days (no future data)
            if i >= len(df) - self.lookforward_days:
                labels.append(-1)  # Invalid label
                continue
            
            current_price = df.iloc[i]['close']
            future_prices = df.iloc[i+1:i+1+self.lookforward_days]['close'].values
            
            label = self._calculate_label(current_price, future_prices)
            labels.append(label)
        
        df['label'] = labels
        
        # Remove invalid labels
        df = df[df['label'] != -1].copy()
        
        print(f"[LabelGenerator] Generated {len(df)} labels")
        print(f"[LabelGenerator] BUY labels: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
        print(f"[LabelGenerator] NO_BUY labels: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
        
        return df
    
    def _calculate_label(self, current_price: float, future_prices: np.ndarray) -> int:
        """
        Calculate label for a single data point
        
        Returns:
            1 if BUY signal (profitable trade)
            0 if NO_BUY signal (unprofitable or risky trade)
        """
        if len(future_prices) == 0:
            return -1
        
        # Track if we hit stop loss before target
        hit_stop_loss = False
        hit_target = False
        
        for price in future_prices:
            price_change = (price - current_price) / current_price
            
            # Check if we hit stop loss
            if price_change <= -self.stop_loss_pct:
                hit_stop_loss = True
                break
            
            # Check if we hit target
            if price_change >= self.target_profit_pct:
                hit_target = True
                break
        
        # Label = 1 only if we hit target WITHOUT hitting stop loss first
        if hit_target and not hit_stop_loss:
            return 1
        else:
            return 0
    
    def generate_labels_with_tiers(
        self,
        df: pd.DataFrame,
        tier_column: str = 'tier'
    ) -> pd.DataFrame:
        """
        Generate labels with tier-specific thresholds
        
        Tier 1 (high quality): Lower target (10%), tighter stop (6%)
        Tier 2 (medium quality): Standard target (12%), standard stop (8%)
        Tier 3 (lower quality): Higher target (15%), wider stop (10%)
        """
        df = df.copy()
        
        if tier_column not in df.columns:
            # No tier info, use standard labeling
            return self.generate_labels(df)
        
        labels = []
        
        for i in range(len(df)):
            if i >= len(df) - self.lookforward_days:
                labels.append(-1)
                continue
            
            tier = df.iloc[i].get(tier_column, 2)
            current_price = df.iloc[i]['close']
            future_prices = df.iloc[i+1:i+1+self.lookforward_days]['close'].values
            
            # Tier-specific thresholds
            if tier == 1:
                target = 0.10  # 10%
                stop = 0.06    # 6%
            elif tier == 3:
                target = 0.15  # 15%
                stop = 0.10    # 10%
            else:  # tier == 2
                target = 0.12  # 12%
                stop = 0.08    # 8%
            
            label = self._calculate_label_with_thresholds(
                current_price, future_prices, target, stop
            )
            labels.append(label)
        
        df['label'] = labels
        df = df[df['label'] != -1].copy()
        
        return df
    
    def _calculate_label_with_thresholds(
        self,
        current_price: float,
        future_prices: np.ndarray,
        target_pct: float,
        stop_pct: float
    ) -> int:
        """Calculate label with custom thresholds"""
        if len(future_prices) == 0:
            return -1
        
        hit_stop_loss = False
        hit_target = False
        
        for price in future_prices:
            price_change = (price - current_price) / current_price
            
            if price_change <= -stop_pct:
                hit_stop_loss = True
                break
            
            if price_change >= target_pct:
                hit_target = True
                break
        
        return 1 if (hit_target and not hit_stop_loss) else 0

