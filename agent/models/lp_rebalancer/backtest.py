"""
Backtesting Framework for LP Rebalancer

Simulates rebalancing decisions on historical data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Position:
    """Represents a position in a liquidity pool."""
    pool_address: str
    pool_name: str
    entry_time: datetime
    entry_apy: float
    capital_usd: float
    cumulative_fees: float = 0.0
    cumulative_il: float = 0.0
    exit_time: Optional[datetime] = None
    exit_apy: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def duration_hours(self) -> float:
        end = self.exit_time or datetime.utcnow()
        return (end - self.entry_time).total_seconds() / 3600
    
    @property
    def net_profit(self) -> float:
        return self.cumulative_fees - self.cumulative_il


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_fees_earned: float
    total_il_suffered: float
    positions: List[Position] = field(default_factory=list)
    
    @property
    def net_profit(self) -> float:
        return self.final_capital - self.initial_capital
    
    @property
    def net_profit_pct(self) -> float:
        return (self.net_profit / self.initial_capital) * 100
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def sharpe_ratio(self) -> float:
        # Simplified Sharpe (would need daily returns for proper calc)
        if not self.positions:
            return 0.0
        profits = [p.net_profit for p in self.positions]
        if np.std(profits) == 0:
            return 0.0
        return np.mean(profits) / np.std(profits)
    
    def summary(self) -> Dict:
        return {
            "period": f"{self.start_date.date()} → {self.end_date.date()}",
            "initial_capital": f"${self.initial_capital:,.2f}",
            "final_capital": f"${self.final_capital:,.2f}",
            "net_profit": f"${self.net_profit:,.2f}",
            "net_profit_pct": f"{self.net_profit_pct:.2f}%",
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate*100:.1f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "total_fees": f"${self.total_fees_earned:,.2f}",
            "total_il": f"${self.total_il_suffered:,.2f}",
        }


class Backtester:
    """Backtests LP rebalancing strategy."""
    
    def __init__(self, model, initial_capital: float = 10000):
        self.model = model
        self.initial_capital = initial_capital
        self.positions: List[Position] = []
        self.capital = initial_capital
        
        # Constraints
        self.max_positions = 5
        self.min_hold_hours = 24
        self.rebalance_cooldown_hours = 6
        self.last_rebalance_time: Optional[datetime] = None
    
    def run(self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame) -> BacktestResult:
        """Run backtest on historical data."""
        
        # Sort by timestamp
        features_df = features_df.sort_values("timestamp")
        timestamps = features_df["timestamp"].unique()
        
        start_date = pd.to_datetime(timestamps[0])
        end_date = pd.to_datetime(timestamps[-1])
        
        total_fees = 0.0
        total_il = 0.0
        
        # Iterate through each timestamp
        for ts in timestamps:
            ts_dt = pd.to_datetime(ts)
            
            # Check cooldown
            if self.last_rebalance_time:
                hours_since = (ts_dt - self.last_rebalance_time).total_seconds() / 3600
                if hours_since < self.rebalance_cooldown_hours:
                    continue
            
            # Get features at this timestamp
            current_features = features_df[features_df["timestamp"] == ts]
            
            # Get predictions
            predictions = self.model.predict(current_features)
            
            # Process exit signals for open positions
            for pos in self.positions:
                if not pos.is_open:
                    continue
                
                # Check minimum hold time
                if pos.duration_hours < self.min_hold_hours:
                    continue
                
                # Find prediction for this pool
                pred = next((p for p in predictions if p.pool_address == pos.pool_address), None)
                
                if pred and pred.recommendation == "EXIT":
                    # Close position
                    pos.exit_time = ts_dt
                    current_data = raw_data_df[
                        (raw_data_df["pool_address"] == pos.pool_address) &
                        (raw_data_df["timestamp"] == ts)
                    ]
                    if not current_data.empty:
                        pos.exit_apy = current_data.iloc[0]["apy"]
                    
                    # Calculate profit (simplified)
                    hourly_fee_rate = pos.entry_apy / 100 / 8760  # APY to hourly
                    pos.cumulative_fees = pos.capital_usd * hourly_fee_rate * pos.duration_hours
                    
                    # Return capital
                    profit = pos.cumulative_fees - pos.cumulative_il
                    self.capital += pos.capital_usd + profit
                    total_fees += pos.cumulative_fees
                    total_il += pos.cumulative_il
                    
                    self.last_rebalance_time = ts_dt
            
            # Process entry signals
            open_count = sum(1 for p in self.positions if p.is_open)
            available_capital = self.capital
            
            for pred in sorted(predictions, key=lambda x: x.stay_probability, reverse=True):
                if open_count >= self.max_positions:
                    break
                if available_capital < 100:  # Minimum position size
                    break
                if pred.recommendation != "STAY":
                    continue
                
                # Check if already in this pool
                if any(p.pool_address == pred.pool_address and p.is_open for p in self.positions):
                    continue
                
                # Enter position
                position_size = min(available_capital * 0.3, available_capital)  # Max 30% per position
                
                current_data = raw_data_df[
                    (raw_data_df["pool_address"] == pred.pool_address) &
                    (raw_data_df["timestamp"] == ts)
                ]
                entry_apy = current_data.iloc[0]["apy"] if not current_data.empty else 0
                
                pos = Position(
                    pool_address=pred.pool_address,
                    pool_name=pred.pool_name,
                    entry_time=ts_dt,
                    entry_apy=entry_apy,
                    capital_usd=position_size,
                )
                self.positions.append(pos)
                
                self.capital -= position_size
                available_capital -= position_size
                open_count += 1
                self.last_rebalance_time = ts_dt
        
        # Close any remaining open positions at end
        for pos in self.positions:
            if pos.is_open:
                pos.exit_time = end_date
                hourly_fee_rate = pos.entry_apy / 100 / 8760
                pos.cumulative_fees = pos.capital_usd * hourly_fee_rate * pos.duration_hours
                profit = pos.cumulative_fees - pos.cumulative_il
                self.capital += pos.capital_usd + profit
                total_fees += pos.cumulative_fees
                total_il += pos.cumulative_il
        
        # Calculate results
        winning = sum(1 for p in self.positions if p.net_profit > 0)
        losing = sum(1 for p in self.positions if p.net_profit <= 0)
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_trades=len(self.positions),
            winning_trades=winning,
            losing_trades=losing,
            total_fees_earned=total_fees,
            total_il_suffered=total_il,
            positions=self.positions,
        )

