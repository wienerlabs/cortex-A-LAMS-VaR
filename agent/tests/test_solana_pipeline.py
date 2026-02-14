"""
Integration tests for Solana Cross-DEX Arbitrage Pipeline.

Tests the complete flow:
1. Data collection (mocked)
2. Feature engineering
3. Model training
4. Inference
5. Execution simulation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Import components
from src.config import SOLANA_CHAIN_PARAMS
from src.features import SolanaFeatureEngineer, CrossDexFeatureEngineer
from src.models.arbitrage import SolanaArbitrageModel
from src.inference import SolanaArbitrageInference
from src.execution import SolanaExecutor


class TestSolanaFeatureEngineering:
    """Test Solana feature engineering."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample Solana market data."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        base_price = 200
        
        return pd.DataFrame({
            'datetime': dates,
            'sol_price': base_price + np.cumsum(np.random.randn(n) * 0.5),
            'raydium_price': base_price + np.cumsum(np.random.randn(n) * 0.5),
            'orca_price': base_price + np.cumsum(np.random.randn(n) * 0.5) + 0.1,
            'volume_usd': np.random.uniform(100000, 1000000, n),
            'priority_fee_lamports': np.random.uniform(10000, 100000, n),
        })
    
    def test_solana_feature_engineer(self, sample_data):
        """Test SolanaFeatureEngineer creates expected features."""
        engineer = SolanaFeatureEngineer()
        result = engineer.engineer_features(sample_data)
        
        # Check technical indicators
        assert 'rsi' in result.columns
        assert 'volatility_1h' in result.columns
        assert 'volume_ma_1h' in result.columns
        
        # Check Solana-specific features
        assert 'priority_fee_normalized' in result.columns
        assert 'tx_cost_pct' in result.columns
        
        # Check values are reasonable
        assert result['rsi'].dropna().between(0, 100).all()
    
    def test_cross_dex_feature_engineer(self, sample_data):
        """Test CrossDexFeatureEngineer creates spread features."""
        engineer = CrossDexFeatureEngineer()
        result = engineer.engineer_features(sample_data)
        
        # Check spread features
        assert 'spread_abs' in result.columns
        assert 'spread_pct' in result.columns
        assert 'spread_zscore' in result.columns
        
        # Check DEX features
        assert 'raydium_premium' in result.columns
        assert 'orca_premium' in result.columns


class TestSolanaArbitrageModel:
    """Test Solana arbitrage model."""
    
    @pytest.fixture
    def training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create training data with features and labels."""
        np.random.seed(42)
        n = 500
        
        # Features
        X = pd.DataFrame({
            'spread_pct': np.random.uniform(0, 1, n),
            'rsi': np.random.uniform(20, 80, n),
            'volatility_1h': np.random.uniform(0.01, 0.05, n),
            'volume_ma_1h': np.random.uniform(100000, 1000000, n),
            'priority_fee_normalized': np.random.uniform(0, 1, n),
            'tx_cost_pct': np.random.uniform(0.01, 0.1, n),
        })
        
        # Labels: profitable if spread > costs
        y = (X['spread_pct'] > X['tx_cost_pct'] + 0.1).astype(int)
        
        return X, y
    
    def test_model_training(self, training_data):
        """Test model training completes successfully."""
        X, y = training_data
        
        model = SolanaArbitrageModel()
        model.train(X, y)
        
        assert model.model is not None
        assert len(model.feature_names) == len(X.columns)
    
    def test_model_prediction(self, training_data):
        """Test model makes predictions."""
        X, y = training_data
        
        model = SolanaArbitrageModel()
        model.train(X, y)
        
        # Predict on subset
        predictions = model.predict(X.head(10))
        
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_should_execute(self, training_data):
        """Test execution decision logic."""
        X, y = training_data
        
        model = SolanaArbitrageModel()
        model.train(X, y)
        
        # Test with high spread (should execute)
        high_spread = X.head(1).copy()
        high_spread['spread_pct'] = 0.5
        
        decision = model.should_execute(high_spread, min_confidence=0.5)
        
        assert 'execute' in decision
        assert 'probability' in decision
        assert 'reason' in decision


class TestSolanaInference:
    """Test Solana inference engine."""
    
    def test_calculate_expected_profit(self):
        """Test profit calculation."""
        inference = SolanaArbitrageInference()
        
        result = inference.calculate_expected_profit(
            spread_pct=0.5,
            trade_size_usd=10000,
            sol_price=200
        )
        
        assert 'gross_profit_pct' in result
        assert 'net_profit_pct' in result
        assert 'net_profit_usd' in result
        assert 'profitable' in result
        
        # With 0.5% spread, should be profitable
        assert result['gross_profit_pct'] == 0.5


class TestSolanaExecutor:
    """Test Solana executor."""
    
    def test_simulate_arbitrage(self):
        """Test arbitrage simulation."""
        executor = SolanaExecutor()
        
        result = executor.simulate_arbitrage(
            spread_pct=0.3,
            trade_size_usd=10000,
            sol_price=200
        )
        
        assert 'success' in result
        assert 'simulated' in result
        assert result['simulated'] is True
        assert 'costs' in result
        assert 'profit' in result
    
    def test_estimate_gas_cost(self):
        """Test gas cost estimation."""
        executor = SolanaExecutor()
        
        result = executor.estimate_gas_cost(
            priority_level="medium",
            sol_price=200
        )
        
        assert 'cost_sol' in result
        assert 'cost_usd' in result
        assert result['cost_sol'] > 0

