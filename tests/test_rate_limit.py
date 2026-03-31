"""Tests for sliding-window rate limiting on MCP server endpoints.

Covers:
  - Under-limit requests pass through with rate limit headers
  - Over-limit requests get 429 with Retry-After
  - Per-IP and per-identity (IP:token) independent limits
  - Configurable thresholds via env vars
  - Sliding window expiry (requests become available again)
  - Direct client IP used (X-Forwarded-For NOT trusted)
  - Non-HTTP scopes pass through
  - Max bucket eviction prevents memory exhaustion
  - Rate limit headers report governing limit correctly
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from openspace.auth.rate_limit import (
    DEFAULT_PER_IP,
    DEFAULT_PER_TOKEN,
    DEFAULT_WINDOW,
    MAX_BUCKETS,
    RATE_LIMIT_PER_IP_ENV,
    RATE_LIMIT_PER_TOKEN_ENV,
    RATE_LIMIT_WINDOW_ENV,
    RateLimitMiddleware,
    SlidingWindowCounter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_app():
    """ASGI app that records calls and returns 200."""

    async def app(scope, receive, send):
        app.call_count += 1
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        })
        await send({"type": "http.response.body", "body": b"OK"})

    app.call_count = 0
    return app


def _http_scope(
    path="/test",
    client_ip="10.0.0.1",
    headers=None,
    token=None,
):
    """Build HTTP ASGI scope with optional bearer token."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append([k.encode(), v.encode()])
    if token:
        raw_headers.append([b"authorization", f"Bearer {token}".encode()])
    return {
        "type": "http",
        "path": path,
        "headers": raw_headers,
        "client": (client_ip, 12345),
    }


class ResponseCollector:
    def __init__(self):
        self.status = None
        self.headers = {}
        self.body = b""

    async def __call__(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
            for k, v in message.get("headers", []):
                self.headers[k.decode()] = v.decode()
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")


# ---------------------------------------------------------------------------
# SlidingWindowCounter unit tests
# ---------------------------------------------------------------------------


class TestSlidingWindowCounter:
    @pytest.mark.asyncio
    async def test_under_limit_allowed(self):
        counter = SlidingWindowCounter(limit=5, window=60.0)
        ok, remaining, retry = await counter.is_allowed("key1")
        assert ok is True
        assert remaining == 4
        assert retry == 0.0

    @pytest.mark.asyncio
    async def test_at_limit_rejected(self):
        counter = SlidingWindowCounter(limit=3, window=60.0)
        for _ in range(3):
            await counter.is_allowed("key1")
        ok, remaining, retry = await counter.is_allowed("key1")
        assert ok is False
        assert remaining == 0
        assert retry > 0

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        counter = SlidingWindowCounter(limit=2, window=60.0)
        await counter.is_allowed("a")
        await counter.is_allowed("a")
        # "a" is exhausted
        ok_a, _, _ = await counter.is_allowed("a")
        assert ok_a is False
        # "b" is fresh
        ok_b, remaining_b, _ = await counter.is_allowed("b")
        assert ok_b is True
        assert remaining_b == 1

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Requests expire after the window passes."""
        counter = SlidingWindowCounter(limit=2, window=0.1)  # 100ms window
        await counter.is_allowed("key1")
        await counter.is_allowed("key1")
        # Exhausted
        ok, _, _ = await counter.is_allowed("key1")
        assert ok is False
        # Wait for window to expire
        await asyncio.sleep(0.15)
        ok, remaining, _ = await counter.is_allowed("key1")
        assert ok is True
        assert remaining == 1


# ---------------------------------------------------------------------------
# RateLimitMiddleware integration tests
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_under_limit_passes_through(self, dummy_app, monkeypatch):
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "10")
        monkeypatch.setenv(RATE_LIMIT_PER_TOKEN_ENV, "10")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)
        scope = _http_scope(client_ip="1.2.3.4")
        collector = ResponseCollector()
        await mw(scope, AsyncMock(), collector)
        assert collector.status == 200
        assert dummy_app.call_count == 1
        assert "x-ratelimit-remaining" in collector.headers

    @pytest.mark.asyncio
    async def test_ip_limit_exceeded_returns_429(self, dummy_app, monkeypatch):
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "3")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)

        for _ in range(3):
            collector = ResponseCollector()
            await mw(_http_scope(client_ip="5.5.5.5"), AsyncMock(), collector)
            assert collector.status == 200

        # 4th request should be rate limited
        collector = ResponseCollector()
        await mw(_http_scope(client_ip="5.5.5.5"), AsyncMock(), collector)
        assert collector.status == 429
        body = json.loads(collector.body)
        assert body["error"] == "rate_limited"
        assert "retry-after" in collector.headers

    @pytest.mark.asyncio
    async def test_token_limit_uses_ip_token_composite(self, dummy_app, monkeypatch):
        """Per-identity limit uses IP:token composite key, not raw token."""
        monkeypatch.setenv(RATE_LIMIT_PER_TOKEN_ENV, "2")
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "100")  # high IP limit
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)
        token = "shared-secret-token-value"

        # Two requests from IP A with token → OK
        for _ in range(2):
            collector = ResponseCollector()
            await mw(_http_scope(client_ip="10.0.0.1", token=token), AsyncMock(), collector)
            assert collector.status == 200

        # 3rd from IP A → 429 (identity exhausted)
        collector = ResponseCollector()
        await mw(_http_scope(client_ip="10.0.0.1", token=token), AsyncMock(), collector)
        assert collector.status == 429

        # Same token from IP B → still allowed (different composite key)
        collector = ResponseCollector()
        await mw(_http_scope(client_ip="10.0.0.2", token=token), AsyncMock(), collector)
        assert collector.status == 200

    @pytest.mark.asyncio
    async def test_different_ips_independent(self, dummy_app, monkeypatch):
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "2")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)

        for _ in range(2):
            await mw(_http_scope(client_ip="1.1.1.1"), AsyncMock(), ResponseCollector())
        # IP 1.1.1.1 exhausted
        collector = ResponseCollector()
        await mw(_http_scope(client_ip="1.1.1.1"), AsyncMock(), collector)
        assert collector.status == 429

        # IP 2.2.2.2 still allowed
        collector = ResponseCollector()
        await mw(_http_scope(client_ip="2.2.2.2"), AsyncMock(), collector)
        assert collector.status == 200

    @pytest.mark.asyncio
    async def test_x_forwarded_for_ignored(self, dummy_app, monkeypatch):
        """X-Forwarded-For is NOT trusted — prevents IP spoofing bypass."""
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "1")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)

        # First request from 127.0.0.1 with spoofed XFF
        scope = _http_scope(
            client_ip="127.0.0.1",
            headers={"x-forwarded-for": "203.0.113.50"},
        )
        collector = ResponseCollector()
        await mw(scope, AsyncMock(), collector)
        assert collector.status == 200

        # Second request from same client IP but different XFF
        # Should be rate limited because we use client IP, not XFF
        scope2 = _http_scope(
            client_ip="127.0.0.1",
            headers={"x-forwarded-for": "198.51.100.99"},
        )
        collector = ResponseCollector()
        await mw(scope2, AsyncMock(), collector)
        assert collector.status == 429

    @pytest.mark.asyncio
    async def test_lifespan_passes_through(self, dummy_app, monkeypatch):
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "1")
        mw = RateLimitMiddleware(dummy_app)
        scope = {"type": "lifespan"}
        await mw(scope, AsyncMock(), AsyncMock())
        assert dummy_app.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, dummy_app, monkeypatch):
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "10")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)
        collector = ResponseCollector()
        await mw(_http_scope(), AsyncMock(), collector)
        assert "x-ratelimit-remaining" in collector.headers
        assert "x-ratelimit-limit" in collector.headers
        assert "x-ratelimit-window" in collector.headers

    @pytest.mark.asyncio
    async def test_window_recovery(self, dummy_app, monkeypatch):
        """After window expires, requests are allowed again."""
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "1")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "1")  # 1 second window
        mw = RateLimitMiddleware(dummy_app)

        collector = ResponseCollector()
        await mw(_http_scope(client_ip="9.9.9.9"), AsyncMock(), collector)
        assert collector.status == 200

        collector = ResponseCollector()
        await mw(_http_scope(client_ip="9.9.9.9"), AsyncMock(), collector)
        assert collector.status == 429

        await asyncio.sleep(1.1)

        collector = ResponseCollector()
        await mw(_http_scope(client_ip="9.9.9.9"), AsyncMock(), collector)
        assert collector.status == 200

    @pytest.mark.asyncio
    async def test_governing_headers_reflect_token_limit(self, dummy_app, monkeypatch):
        """When token limit < IP limit, headers report the token (governing) limit."""
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "100")
        monkeypatch.setenv(RATE_LIMIT_PER_TOKEN_ENV, "5")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")
        mw = RateLimitMiddleware(dummy_app)

        collector = ResponseCollector()
        await mw(
            _http_scope(client_ip="7.7.7.7", token="test-token"),
            AsyncMock(),
            collector,
        )
        assert collector.status == 200
        # Governing limit should be min(100, 5) = 5
        assert collector.headers["x-ratelimit-limit"] == "5"


class TestSlidingWindowCounterEviction:
    """Test max bucket cap prevents memory exhaustion without resetting active clients."""

    @pytest.mark.asyncio
    async def test_new_keys_rejected_at_capacity(self):
        """When at max_buckets, new keys are rejected (not existing ones evicted)."""
        counter = SlidingWindowCounter(limit=10, window=60.0, max_buckets=3)
        # Fill to capacity
        for i in range(3):
            ok, _, _ = await counter.is_allowed(f"key-{i}")
            assert ok is True
        # New key should be rejected
        ok, remaining, retry = await counter.is_allowed("key-new")
        assert ok is False
        assert remaining == 0
        assert retry > 0

    @pytest.mark.asyncio
    async def test_existing_keys_still_work_at_capacity(self):
        """Existing clients are not affected when new keys are rejected."""
        counter = SlidingWindowCounter(limit=10, window=60.0, max_buckets=3)
        for i in range(3):
            await counter.is_allowed(f"key-{i}")
        # Existing key should still work
        ok, remaining, _ = await counter.is_allowed("key-1")
        assert ok is True
        assert remaining == 8  # 10 - 2 requests

    @pytest.mark.asyncio
    async def test_stale_cleanup_frees_capacity(self):
        """After stale keys expire, new keys can be accepted again."""
        counter = SlidingWindowCounter(limit=10, window=0.1, max_buckets=2)
        counter._cleanup_interval = 0.05  # fast cleanup for test
        await counter.is_allowed("old-key-1")
        await counter.is_allowed("old-key-2")
        # At capacity — new key rejected
        ok, _, _ = await counter.is_allowed("new-key")
        assert ok is False
        # Wait for window expiry
        await asyncio.sleep(0.15)
        # Now new key should work (stale keys cleaned up)
        ok, _, _ = await counter.is_allowed("new-key")
        assert ok is True

    @pytest.mark.asyncio
    async def test_stale_cleanup_at_capacity_with_default_interval(self):
        """Stale buckets are force-cleaned at capacity even with long cleanup interval."""
        # Reproduces: default cleanup_interval >> window, but expired buckets
        # should NOT block new clients
        counter = SlidingWindowCounter(limit=10, window=0.1, max_buckets=2)
        # Deliberately keep long cleanup interval (simulates default behavior)
        counter._cleanup_interval = 999.0
        await counter.is_allowed("a")
        await counter.is_allowed("b")
        # At capacity
        ok, _, _ = await counter.is_allowed("c")
        assert ok is False
        # Wait for window expiry
        await asyncio.sleep(0.15)
        # New key should succeed: force-cleanup runs at capacity
        ok, _, _ = await counter.is_allowed("c")
        assert ok is True
