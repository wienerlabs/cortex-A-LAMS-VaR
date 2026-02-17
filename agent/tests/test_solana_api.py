"""
API tests for Solana endpoints.

Tests the FastAPI endpoints for Solana arbitrage.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestSolanaEndpoints:
    """Test Solana API endpoints."""
    
    def test_model_info_endpoint(self, client):
        """Test /api/v1/solana/model/info endpoint."""
        response = client.get("/api/v1/solana/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'model' in data
        assert 'chain_params' in data
        assert 'timestamp' in data
        
        # Check chain params
        params = data['chain_params']
        assert 'raydium_fee_pct' in params
        assert 'orca_fee_pct' in params
    
    @patch('src.api.routes.solana.get_birdeye')
    @patch('src.api.routes.solana.get_jupiter')
    def test_market_endpoint(self, mock_jupiter, mock_birdeye, client):
        """Test /api/v1/solana/market endpoint."""
        # Mock Birdeye response
        mock_collector = AsyncMock()
        mock_collector.fetch_latest.return_value = {
            'prices': {'SOL': {'price_usd': 200}},
            'slot': 12345678,
            'avg_priority_fee_lamports': 50000
        }
        mock_birdeye.return_value = mock_collector
        
        response = client.get("/api/v1/solana/market")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'sol_price_usd' in data
        assert 'current_slot' in data
        assert 'avg_priority_fee_lamports' in data
    
    @patch('src.api.routes.solana.get_birdeye')
    @patch('src.api.routes.solana.get_jupiter')
    def test_spreads_endpoint(self, mock_jupiter, mock_birdeye, client):
        """Test /api/v1/solana/spreads endpoint."""
        # Mock collectors
        mock_birdeye_collector = AsyncMock()
        mock_birdeye_collector.fetch_latest.return_value = {
            'prices': {'SOL': {'price_usd': 200}}
        }
        mock_birdeye.return_value = mock_birdeye_collector
        
        mock_jupiter_collector = AsyncMock()
        mock_jupiter_collector.compare_dex_routes.return_value = {
            'spread_pct': 0.15,
            'best_dex': 'raydium',
            'dex_routes': {
                'raydium': {'out_amount': 1000000},
                'orca': {'out_amount': 998500}
            }
        }
        mock_jupiter.return_value = mock_jupiter_collector
        
        response = client.get("/api/v1/solana/spreads?amount_usd=10000")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'spread_pct' in data
        assert 'best_dex' in data
        assert 'dex_routes' in data
    
    @patch('src.api.routes.solana.get_solana_model')
    @patch('src.api.routes.solana.get_birdeye')
    @patch('src.api.routes.solana.get_jupiter')
    def test_predict_endpoint(self, mock_jupiter, mock_birdeye, mock_model, client):
        """Test /api/v1/solana/predict endpoint."""
        # Mock model
        mock_inference = MagicMock()
        mock_inference.calculate_expected_profit.return_value = {
            'gross_profit_pct': 0.3,
            'dex_fees_pct': 0.05,
            'tx_fee_pct': 0.01,
            'slippage_pct': 0.1,
            'total_cost_pct': 0.16,
            'net_profit_pct': 0.14,
            'net_profit_usd': 14.0,
            'profitable': True
        }
        mock_model.return_value = mock_inference
        
        # Mock collectors
        mock_birdeye_collector = AsyncMock()
        mock_birdeye_collector.fetch_latest.return_value = {
            'prices': {'SOL': {'price_usd': 200}}
        }
        mock_birdeye.return_value = mock_birdeye_collector
        
        mock_jupiter_collector = AsyncMock()
        mock_jupiter_collector.compare_dex_routes.return_value = {
            'spread_pct': 0.3,
            'best_dex': 'raydium',
            'dex_routes': {
                'raydium': {'out_amount': 1000000},
                'orca': {'out_amount': 997000}
            }
        }
        mock_jupiter.return_value = mock_jupiter_collector
        
        response = client.post(
            "/api/v1/solana/predict",
            json={
                "trade_size_usd": 10000,
                "min_confidence": 0.65
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'execute' in data
        assert 'probability' in data
        assert 'spread' in data
        assert 'costs' in data
        assert 'net_profit_pct' in data
    
    @patch('src.api.routes.solana.get_solana_model')
    @patch('src.api.routes.solana.get_birdeye')
    def test_simulate_endpoint(self, mock_birdeye, mock_model, client):
        """Test /api/v1/solana/simulate endpoint."""
        # Mock model
        mock_inference = MagicMock()
        mock_inference.calculate_expected_profit.return_value = {
            'gross_profit_pct': 0.5,
            'net_profit_pct': 0.3,
            'net_profit_usd': 30.0,
            'profitable': True,
            'dex_fees_pct': 0.05,
            'tx_fee_pct': 0.01,
            'slippage_pct': 0.1,
            'total_cost_pct': 0.16
        }
        mock_model.return_value = mock_inference
        
        # Mock birdeye
        mock_birdeye_collector = AsyncMock()
        mock_birdeye_collector.fetch_latest.return_value = {
            'prices': {'SOL': {'price_usd': 200}}
        }
        mock_birdeye.return_value = mock_birdeye_collector
        
        response = client.post(
            "/api/v1/solana/simulate?spread_pct=0.5&trade_size_usd=10000"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'simulation' in data
        assert 'input' in data
        assert data['input']['spread_pct'] == 0.5

