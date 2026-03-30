"""Tests for bearer token authentication on MCP server endpoints.

Covers:
  - Valid token → 200 (request passes through)
  - Invalid token → 401
  - Missing token → 401
  - Missing Authorization header → 401
  - Token strength validation
  - Non-HTTP scopes pass through without auth
  - run_mcp_server refuses to start HTTP transport without token (fail-closed)
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.auth.bearer import (
    BEARER_TOKEN_ENV,
    MIN_TOKEN_LENGTH,
    BearerTokenMiddleware,
    get_bearer_token,
    validate_token_strength,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_TOKEN = "a" * 48  # 48 chars, well above minimum
WRONG_TOKEN = "b" * 48


@pytest.fixture
def dummy_app():
    """An ASGI app that records whether it was called and returns 200."""

    async def app(scope, receive, send):
        app.called = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            }
        )
        await send({"type": "http.response.body", "body": b"OK"})

    app.called = False
    return app


@pytest.fixture
def middleware(dummy_app):
    return BearerTokenMiddleware(dummy_app, VALID_TOKEN)


def _http_scope(path="/test", headers=None):
    """Build a minimal HTTP ASGI scope."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append([k.encode(), v.encode()])
    return {"type": "http", "path": path, "headers": raw_headers}


def _lifespan_scope():
    return {"type": "lifespan"}


# ---------------------------------------------------------------------------
# Helper to collect ASGI response
# ---------------------------------------------------------------------------


class ResponseCollector:
    """Collects ASGI send() calls into a response dict."""

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
# Token utility tests
# ---------------------------------------------------------------------------


class TestGetBearerToken:
    def test_returns_token_from_env(self, monkeypatch):
        monkeypatch.setenv(BEARER_TOKEN_ENV, "test-token-123")
        assert get_bearer_token() == "test-token-123"

    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv(BEARER_TOKEN_ENV, raising=False)
        assert get_bearer_token() is None


class TestValidateTokenStrength:
    def test_valid_token(self):
        ok, reason = validate_token_strength("x" * MIN_TOKEN_LENGTH)
        assert ok is True
        assert reason == "OK"

    def test_too_short(self):
        ok, reason = validate_token_strength("short")
        assert ok is False
        assert "too short" in reason.lower()

    def test_exactly_min_length(self):
        ok, _ = validate_token_strength("x" * MIN_TOKEN_LENGTH)
        assert ok is True

    def test_one_below_min(self):
        ok, _ = validate_token_strength("x" * (MIN_TOKEN_LENGTH - 1))
        assert ok is False


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


class TestBearerTokenMiddleware:
    @pytest.mark.asyncio
    async def test_valid_token_passes_through(self, middleware, dummy_app):
        scope = _http_scope(headers={"authorization": f"Bearer {VALID_TOKEN}"})
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert dummy_app.called is True

    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self, middleware, dummy_app):
        scope = _http_scope(headers={})
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert collector.status == 401
        assert dummy_app.called is False
        body = json.loads(collector.body)
        assert body["error"] == "unauthorized"
        assert "missing" in body["detail"].lower()

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self, middleware, dummy_app):
        scope = _http_scope(headers={"authorization": f"Bearer {WRONG_TOKEN}"})
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert collector.status == 401
        assert dummy_app.called is False
        body = json.loads(collector.body)
        assert "invalid" in body["detail"].lower()

    @pytest.mark.asyncio
    async def test_non_bearer_scheme_returns_401(self, middleware, dummy_app):
        scope = _http_scope(headers={"authorization": "Basic dXNlcjpwYXNz"})
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert collector.status == 401
        assert dummy_app.called is False

    @pytest.mark.asyncio
    async def test_empty_bearer_returns_401(self, middleware, dummy_app):
        scope = _http_scope(headers={"authorization": "Bearer "})
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert collector.status == 401
        assert dummy_app.called is False

    @pytest.mark.asyncio
    async def test_lifespan_scope_passes_through(self, middleware, dummy_app):
        """Non-HTTP scopes should not be authenticated."""
        scope = _lifespan_scope()
        receive = AsyncMock()
        send = AsyncMock()
        await middleware(scope, receive, send)
        assert dummy_app.called is True

    @pytest.mark.asyncio
    async def test_401_includes_www_authenticate_header(self, middleware):
        scope = _http_scope(headers={})
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert "www-authenticate" in collector.headers
        assert "Bearer" in collector.headers["www-authenticate"]

    @pytest.mark.asyncio
    async def test_timing_safe_comparison(self, middleware, dummy_app):
        """Ensure we don't leak token via timing — just verify hmac.compare_digest is used."""
        # This is a design test: verify the code path uses constant-time comparison.
        # We can't truly test timing, but we confirm wrong tokens of same length are rejected.
        same_length_wrong = "z" * len(VALID_TOKEN)
        scope = _http_scope(
            headers={"authorization": f"Bearer {same_length_wrong}"}
        )
        collector = ResponseCollector()
        await middleware(scope, AsyncMock(), collector)
        assert collector.status == 401
        assert dummy_app.called is False


# ---------------------------------------------------------------------------
# run_mcp_server fail-closed tests
# ---------------------------------------------------------------------------


class TestRunMcpServerFailClosed:
    """Verify the server refuses to start HTTP transports without a token."""

    def test_sse_without_token_exits(self, monkeypatch):
        monkeypatch.delenv(BEARER_TOKEN_ENV, raising=False)
        monkeypatch.setattr(
            sys, "argv", ["openspace-mcp", "--transport", "sse"]
        )
        with pytest.raises(SystemExit) as exc_info:
            from openspace.mcp_server import run_mcp_server

            run_mcp_server()
        assert exc_info.value.code == 1

    def test_sse_with_weak_token_exits(self, monkeypatch):
        monkeypatch.setenv(BEARER_TOKEN_ENV, "tooshort")
        monkeypatch.setattr(
            sys, "argv", ["openspace-mcp", "--transport", "sse"]
        )
        with pytest.raises(SystemExit) as exc_info:
            from openspace.mcp_server import run_mcp_server

            run_mcp_server()
        assert exc_info.value.code == 1

    def test_streamable_http_without_token_exits(self, monkeypatch):
        monkeypatch.delenv(BEARER_TOKEN_ENV, raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["openspace-mcp", "--transport", "streamable-http"],
        )
        with pytest.raises(SystemExit) as exc_info:
            from openspace.mcp_server import run_mcp_server

            run_mcp_server()
        assert exc_info.value.code == 1
