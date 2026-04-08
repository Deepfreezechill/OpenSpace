"""Shared-secret bearer token authentication for Scion servers.

Provides ASGI middleware that validates HTTP requests against a
shared-secret bearer token read from the environment.

Design decisions:
  - Fail-closed: missing or invalid tokens → 401, never fallback.
  - Constant-time comparison via hmac.compare_digest (timing-safe).
  - Minimum token length enforced (32 chars) to prevent weak secrets.
  - Only applies to HTTP/WebSocket scopes; ASGI lifespan passes through.

Usage:
    Set SCION_MCP_BEARER_TOKEN in your environment before starting
    any HTTP transport (SSE, streamable-http).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger("scion.auth")

BEARER_TOKEN_ENV = "SCION_MCP_BEARER_TOKEN"
MIN_TOKEN_LENGTH = 32


def get_bearer_token() -> str | None:
    """Read the shared-secret bearer token from environment."""
    return os.environ.get(BEARER_TOKEN_ENV)


def validate_token_strength(token: str) -> tuple[bool, str]:
    """Check that a token meets minimum security requirements.

    Returns (is_valid, reason).
    """
    if len(token) < MIN_TOKEN_LENGTH:
        return False, (f"Token too short ({len(token)} chars). Minimum is {MIN_TOKEN_LENGTH} characters.")
    return True, "OK"


class BearerTokenMiddleware:
    """ASGI middleware: reject HTTP requests without a valid bearer token.

    Wraps any ASGI application. Non-HTTP scopes (e.g. lifespan) are
    passed through without authentication.
    """

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self._token = token

    async def __call__(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
    ) -> None:
        # Only authenticate HTTP and WebSocket requests
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode("latin-1")

        if not auth_value.startswith("Bearer "):
            logger.warning(
                "Rejected request: missing bearer token (path=%s)",
                scope.get("path", "?"),
            )
            await self._send_401(send, "Missing bearer token")
            return

        provided = auth_value[7:]  # strip "Bearer "
        if not hmac.compare_digest(provided, self._token):
            logger.warning(
                "Rejected request: invalid bearer token (path=%s)",
                scope.get("path", "?"),
            )
            await self._send_401(send, "Invalid bearer token")
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Callable, detail: str) -> None:
        """Send a 401 Unauthorized JSON response."""
        body = json.dumps(
            {"error": "unauthorized", "detail": detail},
            ensure_ascii=False,
        ).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"www-authenticate", b'Bearer realm="scion-mcp"'],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
