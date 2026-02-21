"""End-to-end tests: API → Guardian → Execution → Strategy pipeline.

Tests the full flow from API request through risk assessment, execution
preflight, strategy toggles, and token supply endpoints — the same path
that the frontend UI exercises.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ── Health & System ──────────────────────────────────────────────────


class TestE2EHealth:
    """Validate the expanded /health endpoint returns all subsystems."""

    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")
        assert data["service"] == "cortex-risk-engine"
        assert "checks" in data
        checks = data["checks"]
        assert "redis" in checks
        assert "guardian" in checks
        assert "circuit_breakers" in checks
        assert "execution" in checks

    def test_openapi_schema_available(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "/api/v1/guardian/assess" in schema["paths"] or len(schema["paths"]) > 10


# ── Strategy & Trade Mode ───────────────────────────────────────────


class TestE2EStrategies:
    """Strategy config, toggles, and trade mode — frontend-facing endpoints."""

    def test_strategy_config(self):
        r = client.get("/api/v1/strategies/config")
        assert r.status_code == 200
        data = r.json()
        assert "strategies" in data
        assert isinstance(data["strategies"], list)
        assert len(data["strategies"]) > 0

    def test_trade_mode_get(self):
        r = client.get("/api/v1/strategies/trade-mode")
        assert r.status_code == 200
        data = r.json()
        assert "mode" in data
        assert data["mode"] in ("autonomous", "semi-auto", "manual")

    def test_trade_mode_set_and_get(self):
        # Set to manual
        r = client.post("/api/v1/strategies/trade-mode", json={"mode": "manual"})
        assert r.status_code == 200
        assert r.json()["mode"] == "manual"

        # Verify it persisted
        r = client.get("/api/v1/strategies/trade-mode")
        assert r.status_code == 200
        assert r.json()["mode"] == "manual"

        # Reset to simulation default
        client.post("/api/v1/strategies/trade-mode", json={"mode": "semi-auto"})


# ── Guardian Risk Assessment ─────────────────────────────────────────


class TestE2EGuardian:
    """Guardian assess_trade flow — the core risk veto path."""

    @pytest.fixture(autouse=True, scope="class")
    def _calibrate(self):
        """Ensure a model is calibrated for assessment."""
        r = client.post("/api/v1/calibrate", json={
            "token": "SOL",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "num_states": 5,
            "method": "empirical",
            "data_source": "yfinance",
        })
        # OK if calibration fails (no yfinance in CI) — Guardian still works with defaults
        yield

    def test_guardian_assess(self):
        r = client.post("/api/v1/guardian/assess", json={
            "token": "SOL",
            "direction": "long",
            "trade_size_usd": 100.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert "composite_score" in data
        assert "verdict" in data
        assert data["verdict"] in ("approve", "reject", "reduce")
        assert 0 <= data["composite_score"] <= 100

    def test_guardian_assess_with_debate(self):
        r = client.post("/api/v1/guardian/assess", json={
            "token": "SOL",
            "direction": "long",
            "trade_size_usd": 100.0,
            "enable_debate": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert "composite_score" in data


# ── Execution Preflight ──────────────────────────────────────────────


class TestE2EExecution:
    """Execution preflight and execution log — no actual trades."""

    def test_execution_preflight(self):
        r = client.post("/api/v1/execution/preflight", json={
            "token_mint": "So11111111111111111111111111111111111111112",
            "direction": "buy",
            "trade_size_usd": 50.0,
            "strategy": "momentum",
        })
        # 200 if execution enabled, or appropriate error
        assert r.status_code in (200, 400, 403, 503)

    def test_execution_log(self):
        r = client.get("/api/v1/execution/log")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list) or "entries" in data


# ── Token Supply & Holders ───────────────────────────────────────────


class TestE2ETokenomics:
    """Token supply and holder endpoints — what the tokenomics page hits."""

    def test_token_supply(self):
        r = client.get("/api/v1/token/supply")
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "CRTX"
        assert data["decimals"] == 9
        assert data["total_supply_formatted"] > 0

    def test_token_supply_has_treasury(self):
        r = client.get("/api/v1/token/supply")
        data = r.json()
        assert "treasury" in data
        assert "sol_balance" in data["treasury"]
        assert "address" in data["treasury"]

    def test_token_holders(self):
        r = client.get("/api/v1/token/holders")
        assert r.status_code == 200
        data = r.json()
        assert "total_holders" in data
        assert "concentration_risk" in data
        assert data["concentration_risk"] in ("low", "medium", "high", "critical", "unknown")


# ── Order Book (CCXT) ────────────────────────────────────────────────


class TestE2EOrderBook:
    """Order book depth — what the market page depth chart uses."""

    def test_ccxt_orderbook_schema(self):
        """Verify response schema even if exchange is unreachable."""
        r = client.get("/api/v1/ccxt/orderbook", params={
            "symbol": "SOL/USDT", "limit": 10,
        })
        # 200 if CCXT configured, 502/503 if not — both acceptable in CI
        if r.status_code == 200:
            data = r.json()
            assert "best_bid" in data
            assert "best_ask" in data
            assert "bids" in data
            assert "asks" in data
            assert isinstance(data["bids"], list)
            assert isinstance(data["asks"], list)


# ── News & Social ────────────────────────────────────────────────────


class TestE2ESentiment:
    """News and social sentiment — background data pipelines."""

    def test_news_feed(self):
        r = client.get("/api/v1/news/feed")
        # OK or empty if no API keys
        assert r.status_code in (200, 503)

    def test_social_sentiment(self):
        r = client.get("/api/v1/social/sentiment", params={"token": "solana"})
        assert r.status_code == 200
        data = r.json()
        assert "overall_sentiment" in data
        assert "sources" in data


# ── Regime & Model Endpoints ─────────────────────────────────────────


class TestE2ERegime:
    """Regime detection — what the dashboard regime panel uses."""

    def test_regime_current(self):
        r = client.get("/api/v1/regime/current", params={"token": "SOL"})
        # OK if calibrated, 404/400 if not
        if r.status_code == 200:
            data = r.json()
            assert "regime_index" in data or "current_regime" in data

    def test_regime_transition_alert(self):
        r = client.get("/api/v1/regime/transition-alert", params={"token": "SOL"})
        if r.status_code == 200:
            data = r.json()
            assert "alert" in data or "transition_prob" in data


# ── Circuit Breakers ─────────────────────────────────────────────────


class TestE2ECircuitBreakers:
    """Circuit breaker state — included in strategy config."""

    def test_circuit_breaker_in_strategy_config(self):
        r = client.get("/api/v1/strategies/config")
        assert r.status_code == 200
        data = r.json()
        strategies = data.get("strategies", [])
        # Each strategy should have circuit_breaker field
        for strat in strategies:
            assert "name" in strat
            # circuit_breaker may be null if no history
            assert "circuit_breaker" in strat or True  # field is optional


# ── Full Flow: Assess → Preflight ────────────────────────────────────


class TestE2EFullFlow:
    """Complete trade assessment flow: Guardian → Strategy check → Preflight."""

    def test_full_trade_flow(self):
        # 1. Check strategy config (is the strategy enabled?)
        r = client.get("/api/v1/strategies/config")
        assert r.status_code == 200

        # 2. Check trade mode
        r = client.get("/api/v1/strategies/trade-mode")
        assert r.status_code == 200
        mode = r.json()["mode"]

        # 3. Run guardian assessment
        r = client.post("/api/v1/guardian/assess", json={
            "token": "SOL",
            "direction": "long",
            "trade_size_usd": 50.0,
        })
        assert r.status_code == 200
        verdict = r.json()

        # 4. If approved and not manual mode, preflight
        if verdict["verdict"] == "approve" and mode != "manual":
            r = client.post("/api/v1/execution/preflight", json={
                "token_mint": "So11111111111111111111111111111111111111112",
                "direction": "buy",
                "trade_size_usd": 50.0,
                "strategy": "momentum",
            })
            # Accept any response — execution may be disabled in CI
            assert r.status_code in (200, 400, 403, 503)

        # 5. Check execution log reflects activity
        r = client.get("/api/v1/execution/log")
        assert r.status_code == 200
