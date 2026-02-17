"""Tests for per-user/per-wallet rate limiting middleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from api.middleware import RateLimitMiddleware


@pytest.fixture
def middleware():
    app = AsyncMock()
    mw = RateLimitMiddleware(app)
    return mw


def _make_request(
    path: str = "/api/v1/var/95",
    method: str = "GET",
    headers: dict | None = None,
    client_host: str = "127.0.0.1",
) -> MagicMock:
    req = MagicMock()
    req.url.path = path
    req.method = method
    req.headers = headers or {}
    req.client = MagicMock()
    req.client.host = client_host
    return req


class TestClientKey:
    def test_api_key_takes_priority(self, middleware):
        req = _make_request(headers={"x-api-key": "test-key-1"})
        assert middleware._client_key(req) == "key:test-key-1"

    def test_solana_pubkey_second_priority(self, middleware):
        req = _make_request(headers={"x-solana-pubkey": "WaLLeTpUbKeY123"})
        assert middleware._client_key(req) == "wallet:WaLLeTpUbKeY123"

    def test_api_key_over_solana_pubkey(self, middleware):
        req = _make_request(headers={
            "x-api-key": "my-api-key",
            "x-solana-pubkey": "WaLLeTpUbKeY123",
        })
        assert middleware._client_key(req) == "key:my-api-key"

    def test_solana_pubkey_over_ip(self, middleware):
        req = _make_request(
            headers={"x-solana-pubkey": "WaLLeTpUbKeY123"},
            client_host="10.0.0.1",
        )
        assert middleware._client_key(req) == "wallet:WaLLeTpUbKeY123"

    def test_forwarded_for_ip(self, middleware):
        req = _make_request(headers={"x-forwarded-for": "1.2.3.4, 5.6.7.8"})
        assert middleware._client_key(req) == "ip:1.2.3.4"

    def test_direct_client_ip(self, middleware):
        req = _make_request(client_host="192.168.1.1")
        assert middleware._client_key(req) == "ip:192.168.1.1"

    def test_unknown_fallback(self, middleware):
        req = _make_request()
        req.client = None
        assert middleware._client_key(req) == "ip:unknown"


class TestPerWalletIsolation:
    def test_different_wallets_have_separate_limits(self, middleware):
        """Two different wallets should each get their own rate limit bucket."""
        wallet_a = _make_request(headers={"x-solana-pubkey": "WalletA"})
        wallet_b = _make_request(headers={"x-solana-pubkey": "WalletB"})

        key_a = middleware._client_key(wallet_a)
        key_b = middleware._client_key(wallet_b)

        assert key_a != key_b
        assert key_a == "wallet:WalletA"
        assert key_b == "wallet:WalletB"

    def test_same_wallet_same_bucket(self, middleware):
        req1 = _make_request(headers={"x-solana-pubkey": "SameWallet"})
        req2 = _make_request(headers={"x-solana-pubkey": "SameWallet"})
        assert middleware._client_key(req1) == middleware._client_key(req2)
