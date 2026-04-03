"""Tests for EPIC 0.7: Integration Tests.

Covers:
- #26: MCP end-to-end test (HTTP stack with auth + rate limiting)
- #27: Skill execution integration test (execute_task path)
- #28: Error path integration tests (auth, rate limit, malformed)
- #29: Coverage gate (≥20% enforced in CI and pyproject.toml)
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Valid 32-char token for tests
TEST_TOKEN = "a" * 32


# ---------------------------------------------------------------------------
# Shared ASGI test utilities
# ---------------------------------------------------------------------------


async def asgi_request(app, method="GET", path="/", headers=None, body=None):
    """Send an ASGI HTTP request and collect the response."""
    if headers is None:
        headers = []

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 12345),
    }

    body_bytes = body.encode() if isinstance(body, str) else (body or b"")

    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        # After body is sent, wait indefinitely (simulate client waiting)
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    response_started = False
    status_code = None
    response_headers = {}
    response_body = b""

    async def send(message):
        nonlocal response_started, status_code, response_headers, response_body
        if message["type"] == "http.response.start":
            response_started = True
            status_code = message["status"]
            response_headers = {k.decode(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            response_body += message.get("body", b"")

    await app(scope, receive, send)

    return {
        "status": status_code,
        "headers": response_headers,
        "body": response_body,
        "json": json.loads(response_body) if response_body else None,
    }


def make_protected_app(inner_app=None, token=TEST_TOKEN):
    """Build the full middleware stack matching production wiring."""
    from scion.auth.bearer import BearerTokenMiddleware
    from scion.auth.rate_limit import RateLimitMiddleware

    if inner_app is None:
        # Simple echo app
        async def echo_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps({"ok": True}).encode(),
                }
            )

        inner_app = echo_app

    rate_limited = RateLimitMiddleware(inner_app)
    protected = BearerTokenMiddleware(rate_limited, token)
    return protected


# ---------------------------------------------------------------------------
# Issue #26: MCP end-to-end test
# ---------------------------------------------------------------------------


class TestMCPEndToEnd:
    """Full HTTP stack tests: request → auth → rate limit → app."""

    @pytest.mark.asyncio
    async def test_authenticated_request_reaches_app(self):
        """Valid bearer token passes through middleware to inner app."""
        app = make_protected_app()
        resp = await asgi_request(
            app,
            "GET",
            "/sse",
            headers=[("Authorization", f"Bearer {TEST_TOKEN}")],
        )
        assert resp["status"] == 200
        assert resp["json"]["ok"] is True

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self):
        """Successful requests include rate limit headers."""
        app = make_protected_app()
        resp = await asgi_request(
            app,
            "GET",
            "/test",
            headers=[("Authorization", f"Bearer {TEST_TOKEN}")],
        )
        assert resp["status"] == 200
        assert "x-ratelimit-remaining" in resp["headers"]
        assert "x-ratelimit-limit" in resp["headers"]

    @pytest.mark.asyncio
    async def test_middleware_ordering_auth_before_rate_limit(self):
        """Auth middleware is outermost — unauthenticated requests
        don't create rate-limit state."""
        call_count = 0

        async def counting_app(scope, receive, send):
            nonlocal call_count
            call_count += 1
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        app = make_protected_app(counting_app)

        # Unauthenticated request — should be rejected by auth, never reach app
        await asgi_request(app, "GET", "/test")
        assert call_count == 0, "Unauthenticated request should not reach inner app"

    @pytest.mark.asyncio
    async def test_lifespan_passes_through(self):
        """Non-HTTP scopes (lifespan) pass through without auth."""
        lifespan_called = False

        async def lifespan_app(scope, receive, send):
            nonlocal lifespan_called
            if scope["type"] == "lifespan":
                lifespan_called = True

        app = make_protected_app(lifespan_app)

        scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
        await app(scope, lambda: None, lambda m: None)
        assert lifespan_called


# ---------------------------------------------------------------------------
# Issue #27: Skill execution integration test
# ---------------------------------------------------------------------------


class TestSkillExecutionIntegration:
    """Integration tests for execute_task flow."""

    @pytest.mark.asyncio
    async def test_execute_task_returns_formatted_result(self):
        """execute_task produces MCP-formatted result when engine succeeds."""
        from scion.mcp_server import execute_task

        mock_os = MagicMock()
        mock_os.is_initialized.return_value = True
        mock_os.execute = AsyncMock(
            return_value={
                "status": "success",
                "output": "Hello, world!",
                "error": None,
            }
        )

        with (
            patch("scion.mcp_server._get_openspace", new_callable=AsyncMock, return_value=mock_os),
            patch("scion.mcp_server._auto_register_skill_dirs", new_callable=AsyncMock),
            patch("scion.mcp_server._cloud_search_and_import", new_callable=AsyncMock, return_value=[]),
            patch("scion.mcp_server._format_task_result", return_value={"result": "formatted"}),
            patch("scion.mcp_server._write_upload_meta"),
        ):
            result = await execute_task(task="test task")
            mock_os.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_auto_registers_skill_dirs(self):
        """execute_task calls _auto_register_skill_dirs when skill_dirs provided."""
        from scion.mcp_server import execute_task

        mock_os = MagicMock()
        mock_os.is_initialized.return_value = True
        mock_os.execute = AsyncMock(
            return_value={
                "status": "success",
                "output": "ok",
                "error": None,
            }
        )

        with (
            patch("scion.mcp_server._get_openspace", new_callable=AsyncMock, return_value=mock_os),
            patch("scion.mcp_server._auto_register_skill_dirs", new_callable=AsyncMock) as mock_register,
            patch("scion.mcp_server._cloud_search_and_import", new_callable=AsyncMock, return_value=[]),
            patch("scion.mcp_server._format_task_result", return_value={"result": "ok"}),
            patch("scion.mcp_server._write_upload_meta"),
        ):
            await execute_task(task="test", skill_dirs=["/tmp/skills"])
            mock_register.assert_called()

    @pytest.mark.asyncio
    async def test_execute_task_respects_search_scope_local(self):
        """execute_task skips cloud import when search_scope='local'."""
        from scion.mcp_server import execute_task

        mock_os = MagicMock()
        mock_os.is_initialized.return_value = True
        mock_os.execute = AsyncMock(
            return_value={
                "status": "success",
                "output": "ok",
                "error": None,
            }
        )

        with (
            patch("scion.mcp_server._get_openspace", new_callable=AsyncMock, return_value=mock_os),
            patch("scion.mcp_server._auto_register_skill_dirs", new_callable=AsyncMock),
            patch("scion.mcp_server._cloud_search_and_import", new_callable=AsyncMock) as mock_import,
            patch("scion.mcp_server._format_task_result", return_value={"result": "ok"}),
            patch("scion.mcp_server._write_upload_meta"),
        ):
            await execute_task(task="test", search_scope="local")
            mock_import.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #28: Error path integration tests
# ---------------------------------------------------------------------------


class TestErrorPathIntegration:
    """Tests for auth failure, rate limit, and error formatting."""

    @pytest.mark.asyncio
    async def test_no_auth_header_returns_401(self):
        """Missing Authorization header → 401."""
        app = make_protected_app()
        resp = await asgi_request(app, "GET", "/test")
        assert resp["status"] == 401
        assert resp["json"]["error"] == "unauthorized"

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self):
        """Wrong bearer token → 401."""
        app = make_protected_app()
        resp = await asgi_request(
            app,
            "GET",
            "/test",
            headers=[("Authorization", "Bearer wrong-token-that-is-long-enough-32chars!")],
        )
        assert resp["status"] == 401

    @pytest.mark.asyncio
    async def test_non_bearer_scheme_returns_401(self):
        """Non-Bearer auth scheme → 401."""
        app = make_protected_app()
        resp = await asgi_request(
            app,
            "GET",
            "/test",
            headers=[("Authorization", f"Basic {TEST_TOKEN}")],
        )
        assert resp["status"] == 401

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self):
        """Exceeding rate limit → 429 with Retry-After."""
        app = make_protected_app()

        # Set very low limits for testing
        with patch.dict(
            os.environ,
            {
                "OPENSPACE_RATE_LIMIT_PER_IP": "2",
                "OPENSPACE_RATE_LIMIT_PER_TOKEN": "2",
                "OPENSPACE_RATE_LIMIT_WINDOW": "60",
            },
        ):
            # Re-create app with fresh rate limiter to pick up env vars
            app = make_protected_app()

            auth_headers = [("Authorization", f"Bearer {TEST_TOKEN}")]

            # First 2 requests should succeed
            for _ in range(2):
                resp = await asgi_request(app, "GET", "/test", headers=auth_headers)
                assert resp["status"] == 200, f"Expected 200, got {resp['status']}"

            # Third request should be rate-limited
            resp = await asgi_request(app, "GET", "/test", headers=auth_headers)
            assert resp["status"] == 429
            assert resp["json"]["error"] == "rate_limited"
            assert "retry-after" in resp["headers"]

    @pytest.mark.asyncio
    async def test_error_responses_contain_no_tracebacks(self):
        """Error responses must not leak stack traces."""
        app = make_protected_app()

        # 401 response
        resp = await asgi_request(app, "GET", "/test")
        body_text = resp["body"].decode()
        assert "Traceback" not in body_text
        assert not ("File " in body_text and "line " in body_text)
        # Must have structured error JSON
        assert resp["json"]["error"] == "unauthorized"
        assert "detail" in resp["json"]

    @pytest.mark.asyncio
    async def test_execute_task_exception_returns_safe_error(self):
        """Internal exception in execute_task returns safe error, no traceback."""
        from scion.mcp_server import execute_task

        mock_os = MagicMock()
        mock_os.is_initialized.return_value = True
        mock_os.execute = AsyncMock(side_effect=RuntimeError("internal boom"))

        with (
            patch("scion.mcp_server._get_openspace", new_callable=AsyncMock, return_value=mock_os),
            patch("scion.mcp_server._auto_register_skill_dirs", new_callable=AsyncMock),
            patch("scion.mcp_server._cloud_search_and_import", new_callable=AsyncMock),
        ):
            result = await execute_task(task="crash test")
            result_str = str(result)
            assert "Traceback" not in result_str
            # Safe error must contain structured error info
            assert "error" in result_str.lower() or "failed" in result_str.lower()


# ---------------------------------------------------------------------------
# Issue #29: Coverage gate (≥20% in CI)
# ---------------------------------------------------------------------------


class TestCoverageGate:
    """Coverage threshold must be enforced."""

    def test_pyproject_coverage_fail_under_at_least_20(self):
        """pyproject.toml coverage.report.fail_under >= 20."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject_path = ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text())
        fail_under = data["tool"]["coverage"]["report"]["fail_under"]
        assert fail_under >= 20, f"Coverage fail_under is {fail_under}, must be >= 20"

    def test_ci_yaml_has_cov_fail_under(self):
        """CI workflow enforces --cov-fail-under."""
        ci_path = ROOT / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert "--cov-fail-under" in content, "CI must include --cov-fail-under flag in pytest command"

    def test_ci_yaml_cov_threshold_at_least_20(self):
        """CI --cov-fail-under value is >= 20."""
        import re

        ci_path = ROOT / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        match = re.search(r"--cov-fail-under[=\s]+(\d+)", content)
        assert match, "Could not find --cov-fail-under value in CI"
        threshold = int(match.group(1))
        assert threshold >= 20, f"CI coverage threshold is {threshold}, must be >= 20"
