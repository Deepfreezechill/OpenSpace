"""Integration tests: full auth + rate-limit middleware chain.

Exercises BearerTokenMiddleware → RateLimitMiddleware → app
in the same order as run_mcp_server() wires them (line 911-912).

Validates:
  - Valid bearer token → 200 (request reaches app)
  - Missing/invalid token → 401 (rejected by BearerTokenMiddleware)
  - Rate limit exceeded → 429 (rejected by RateLimitMiddleware)
  - Auth rejects BEFORE rate-limit state is created (middleware order)
  - Rate limit headers present on successful requests
  - Per-IP independent rate limiting
  - Rate limit recovery after sliding window expires
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from openspace.auth.bearer import BearerTokenMiddleware
from openspace.auth.rate_limit import (
    RATE_LIMIT_PER_IP_ENV,
    RATE_LIMIT_PER_TOKEN_ENV,
    RATE_LIMIT_WINDOW_ENV,
    RateLimitMiddleware,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TOKEN = "integration-test-token-" + "x" * 32  # 54 chars, well above 32
WRONG_TOKEN = "wrong-token-value-pad-" + "y" * 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_scope(
    path: str = "/test",
    client_ip: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
) -> dict:
    """Build an HTTP ASGI scope with optional headers and client IP."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append([k.encode(), v.encode()])
    return {
        "type": "http",
        "path": path,
        "headers": raw_headers,
        "client": (client_ip, 12345),
    }


class ResponseCollector:
    """Collects ASGI send() calls into status, headers, body."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.body: bytes = b""

    async def __call__(self, message: dict) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            for k, v in message.get("headers", []):
                self.headers[k.decode()] = v.decode()
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")

    @property
    def json(self) -> dict:
        return json.loads(self.body)


async def _send(
    chain,
    *,
    client_ip: str = "127.0.0.1",
    token: str | None = None,
    path: str = "/test",
) -> ResponseCollector:
    """Send a single request through the middleware chain."""
    headers: dict[str, str] = {}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    scope = _http_scope(path=path, client_ip=client_ip, headers=headers)
    collector = ResponseCollector()
    await chain(scope, AsyncMock(), collector)
    return collector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_app():
    """ASGI app that counts calls and returns 200 JSON."""

    async def app(scope, receive, send):
        app.call_count += 1
        body = json.dumps({"status": "ok"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    app.call_count = 0
    return app


@pytest.fixture
def tight_rate_env(monkeypatch):
    """Configure tight rate limits: 3 req / 60s window."""
    monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "3")
    monkeypatch.setenv(RATE_LIMIT_PER_TOKEN_ENV, "3")
    monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "60")


@pytest.fixture
def chain(dummy_app, tight_rate_env):
    """Full middleware chain matching run_mcp_server() wiring:

    BearerTokenMiddleware(RateLimitMiddleware(app))
    """
    rate_limited = RateLimitMiddleware(dummy_app)
    return BearerTokenMiddleware(rate_limited, VALID_TOKEN)


# ---------------------------------------------------------------------------
# End-to-end chain tests
# ---------------------------------------------------------------------------


class TestFullChainEndToEnd:
    """Core integration: auth + rate-limit chain behaves correctly."""

    @pytest.mark.asyncio
    async def test_valid_token_returns_200(self, chain, dummy_app):
        """Valid bearer token → request reaches app → 200."""
        resp = await _send(chain, token=VALID_TOKEN)
        assert resp.status == 200
        assert resp.json["status"] == "ok"
        assert dummy_app.call_count == 1

    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self, chain, dummy_app):
        """No Authorization header → 401 from bearer middleware."""
        resp = await _send(chain)  # no token
        assert resp.status == 401
        assert resp.json["error"] == "unauthorized"
        assert "missing" in resp.json["detail"].lower()
        assert dummy_app.call_count == 0

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self, chain, dummy_app):
        """Wrong bearer token → 401 from bearer middleware."""
        resp = await _send(chain, token=WRONG_TOKEN)
        assert resp.status == 401
        assert resp.json["error"] == "unauthorized"
        assert "invalid" in resp.json["detail"].lower()
        assert dummy_app.call_count == 0

    @pytest.mark.asyncio
    async def test_non_bearer_scheme_returns_401(self, chain, dummy_app):
        """Authorization header with non-Bearer scheme → 401."""
        scope = _http_scope(headers={"authorization": "Basic dXNlcjpwYXNz"})
        collector = ResponseCollector()
        await chain(scope, AsyncMock(), collector)
        assert collector.status == 401
        assert dummy_app.call_count == 0

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, chain, dummy_app):
        """Valid token but rate limit (3 req) exceeded → 429."""
        for i in range(3):
            resp = await _send(chain, token=VALID_TOKEN)
            assert resp.status == 200, f"Request {i + 1}/3 should pass"

        resp = await _send(chain, token=VALID_TOKEN)
        assert resp.status == 429
        body = resp.json
        assert body["error"] == "rate_limited"
        assert "retry-after" in resp.headers
        assert dummy_app.call_count == 3  # only first 3 reached the app


# ---------------------------------------------------------------------------
# Middleware order tests
# ---------------------------------------------------------------------------


class TestMiddlewareOrder:
    """Auth rejects BEFORE rate-limit state is created.

    This is the key security property: unauthenticated floods are
    cheap hmac rejections that never pollute rate-limit buckets,
    preventing memory DoS via fake tokens.
    """

    @pytest.mark.asyncio
    async def test_invalid_token_does_not_consume_rate_limit(
        self,
        chain,
        dummy_app,
    ):
        """Flood with bad tokens → valid requests still have full quota."""
        for _ in range(20):
            resp = await _send(chain, token=WRONG_TOKEN)
            assert resp.status == 401

        # All 3 valid requests should pass (quota untouched)
        for i in range(3):
            resp = await _send(chain, token=VALID_TOKEN)
            assert resp.status == 200, f"Request {i + 1} should pass: auth rejections must not consume rate limit"
        assert dummy_app.call_count == 3

    @pytest.mark.asyncio
    async def test_missing_token_does_not_consume_rate_limit(
        self,
        chain,
        dummy_app,
    ):
        """Flood with no auth → valid requests still have full quota."""
        for _ in range(20):
            resp = await _send(chain)  # no token
            assert resp.status == 401

        for i in range(3):
            resp = await _send(chain, token=VALID_TOKEN)
            assert resp.status == 200
        assert dummy_app.call_count == 3


# ---------------------------------------------------------------------------
# Rate limit header tests
# ---------------------------------------------------------------------------


class TestRateLimitHeaders:
    """Verify rate limit headers on successful and rejected requests."""

    @pytest.mark.asyncio
    async def test_success_includes_ratelimit_headers(self, chain):
        """200 responses include x-ratelimit-* headers."""
        resp = await _send(chain, token=VALID_TOKEN)
        assert resp.status == 200
        assert "x-ratelimit-remaining" in resp.headers
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-window" in resp.headers

    @pytest.mark.asyncio
    async def test_remaining_decreases_with_requests(self, chain):
        """x-ratelimit-remaining decreases after each request."""
        r1 = await _send(chain, token=VALID_TOKEN)
        r2 = await _send(chain, token=VALID_TOKEN)
        remaining1 = int(r1.headers["x-ratelimit-remaining"])
        remaining2 = int(r2.headers["x-ratelimit-remaining"])
        assert remaining2 < remaining1

    @pytest.mark.asyncio
    async def test_401_has_no_ratelimit_headers(self, chain):
        """Auth-rejected requests don't include rate limit headers."""
        resp = await _send(chain, token=WRONG_TOKEN)
        assert resp.status == 401
        assert "x-ratelimit-remaining" not in resp.headers
        assert "x-ratelimit-limit" not in resp.headers

    @pytest.mark.asyncio
    async def test_429_has_retry_after(self, chain):
        """Rate-limited responses include retry-after header."""
        for _ in range(3):
            await _send(chain, token=VALID_TOKEN)

        resp = await _send(chain, token=VALID_TOKEN)
        assert resp.status == 429
        retry = int(resp.headers["retry-after"])
        assert retry >= 1


# ---------------------------------------------------------------------------
# Per-IP and recovery tests
# ---------------------------------------------------------------------------


class TestPerIPRateLimiting:
    """Per-IP independent rate limiting through the full chain."""

    @pytest.mark.asyncio
    async def test_different_ips_have_independent_limits(
        self,
        chain,
        dummy_app,
    ):
        """Two IPs each get their own rate limit quota."""
        # Exhaust 10.0.0.1
        for _ in range(3):
            resp = await _send(
                chain,
                client_ip="10.0.0.1",
                token=VALID_TOKEN,
            )
            assert resp.status == 200

        # 10.0.0.1 is now rate limited
        resp = await _send(chain, client_ip="10.0.0.1", token=VALID_TOKEN)
        assert resp.status == 429

        # 10.0.0.2 should still have full quota
        resp = await _send(chain, client_ip="10.0.0.2", token=VALID_TOKEN)
        assert resp.status == 200
        assert dummy_app.call_count == 4  # 3 + 1

    @pytest.mark.asyncio
    async def test_rate_limit_recovery_after_window(self, dummy_app, monkeypatch):
        """After the sliding window expires, requests are allowed again."""
        monkeypatch.setenv(RATE_LIMIT_PER_IP_ENV, "1")
        monkeypatch.setenv(RATE_LIMIT_PER_TOKEN_ENV, "1")
        monkeypatch.setenv(RATE_LIMIT_WINDOW_ENV, "1")  # 1-second window

        rate_limited = RateLimitMiddleware(dummy_app)
        short_chain = BearerTokenMiddleware(rate_limited, VALID_TOKEN)

        resp = await _send(short_chain, token=VALID_TOKEN)
        assert resp.status == 200

        resp = await _send(short_chain, token=VALID_TOKEN)
        assert resp.status == 429

        await asyncio.sleep(1.1)

        resp = await _send(short_chain, token=VALID_TOKEN)
        assert resp.status == 200
        assert dummy_app.call_count == 2
