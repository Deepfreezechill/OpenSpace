"""Sliding-window rate limiter for OpenSpace servers.

Implements per-IP and per-token rate limiting as ASGI middleware.
Uses an in-memory sliding-window log algorithm — no external dependencies.

Design decisions:
  - Sliding window (not fixed window) for smoother rate enforcement.
  - Per-IP buckets keyed on direct client address (scope["client"]).
    X-Forwarded-For is NOT trusted by default — only explicit trusted
    proxies are honored (prevents IP spoofing bypass).
  - Per-identity buckets keyed on IP:token composite (not raw token)
    so that shared-secret auth doesn't create a single global bucket.
  - MUST be placed AFTER BearerTokenMiddleware so only authenticated
    requests create rate-limit state (prevents memory DoS via fake tokens).
  - Max bucket count enforced to prevent memory exhaustion from unique keys.
  - Configurable via environment variables with sensible defaults.
  - Returns 429 with Retry-After header on limit breach.
  - Thread-safe via asyncio.Lock (single-process model).

Environment variables:
  OPENSPACE_RATE_LIMIT_PER_TOKEN  — requests per window per identity (default: 60)
  OPENSPACE_RATE_LIMIT_PER_IP     — requests per window per IP (default: 30)
  OPENSPACE_RATE_LIMIT_WINDOW     — window size in seconds (default: 60)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger("openspace.auth")

# Environment variable names
RATE_LIMIT_PER_TOKEN_ENV = "OPENSPACE_RATE_LIMIT_PER_TOKEN"
RATE_LIMIT_PER_IP_ENV = "OPENSPACE_RATE_LIMIT_PER_IP"
RATE_LIMIT_WINDOW_ENV = "OPENSPACE_RATE_LIMIT_WINDOW"

# Defaults
DEFAULT_PER_TOKEN = 60
DEFAULT_PER_IP = 30
DEFAULT_WINDOW = 60  # seconds
MAX_BUCKETS = 10_000  # hard cap to prevent memory exhaustion


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


class SlidingWindowCounter:
    """In-memory sliding-window rate counter.

    Each key (IP or identity) gets a list of timestamps. On each check,
    expired entries are pruned and the count is compared to the limit.
    Hard-capped at MAX_BUCKETS to prevent memory exhaustion.
    """

    def __init__(self, limit: int, window: float, max_buckets: int = MAX_BUCKETS) -> None:
        self.limit = limit
        self.window = window
        self._max_buckets = max_buckets
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = max(window * 2, 120.0)

    async def is_allowed(self, key: str) -> tuple[bool, int, float]:
        """Check if a request is allowed for the given key.

        Returns (allowed, remaining, retry_after).
          - allowed: True if under limit
          - remaining: requests left in window
          - retry_after: seconds until oldest entry expires (0 if allowed)
        """
        now = time.monotonic()
        cutoff = now - self.window

        async with self._lock:
            # Periodic cleanup of stale keys
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup(cutoff)
                self._last_cleanup = now

            # If this is a NEW key and we're at capacity, try harder:
            # force-clean stale buckets before rejecting. This prevents
            # expired entries from blocking legitimate new clients.
            if key not in self._buckets and len(self._buckets) >= self._max_buckets:
                self._cleanup(cutoff)
                self._last_cleanup = now

            # Still at capacity after cleanup? Reject the new key.
            # Never evict active buckets — that would reset their quota
            # and let attackers bypass rate limiting via key churn.
            if key not in self._buckets and len(self._buckets) >= self._max_buckets:
                logger.warning(
                    "Rate limiter at max capacity (%d buckets), "
                    "rejecting new key",
                    self._max_buckets,
                )
                return False, 0, float(self.window)

            bucket = self._buckets[key]

            # Prune expired timestamps
            while bucket and bucket[0] <= cutoff:
                bucket.pop(0)

            if len(bucket) >= self.limit:
                retry_after = bucket[0] + self.window - now
                return False, 0, max(retry_after, 0.1)

            bucket.append(now)
            remaining = self.limit - len(bucket)

            return True, remaining, 0.0

    def _cleanup(self, cutoff: float) -> None:
        """Remove keys with no recent activity."""
        stale = [k for k, v in self._buckets.items() if not v or v[-1] <= cutoff]
        for k in stale:
            del self._buckets[k]


class RateLimitMiddleware:
    """ASGI middleware: per-IP and per-identity sliding-window rate limiting.

    MUST be placed AFTER BearerTokenMiddleware in the middleware chain.
    This ensures only authenticated requests create rate-limit state,
    preventing memory DoS via fake tokens from unauthenticated floods.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._per_identity = _env_int(RATE_LIMIT_PER_TOKEN_ENV, DEFAULT_PER_TOKEN)
        per_ip = _env_int(RATE_LIMIT_PER_IP_ENV, DEFAULT_PER_IP)
        window = _env_int(RATE_LIMIT_WINDOW_ENV, DEFAULT_WINDOW)

        self._identity_limiter = SlidingWindowCounter(self._per_identity, window)
        self._ip_limiter = SlidingWindowCounter(per_ip, window)
        self._window = window

        logger.info(
            "Rate limiter: %d req/identity, %d req/IP, %ds window",
            self._per_identity, per_ip, window,
        )

    async def __call__(
        self, scope: dict, receive: Callable, send: Callable,
    ) -> None:
        if scope["type"] not in ("http",):
            await self.app(scope, receive, send)
            return

        client_ip = self._extract_ip(scope)
        token = self._extract_token(scope)

        # Check IP limit first
        ip_ok, ip_remaining, ip_retry = await self._ip_limiter.is_allowed(client_ip)
        if not ip_ok:
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            await self._send_429(send, ip_retry, "IP rate limit exceeded")
            return

        # Check identity limit (IP:token composite key)
        # Using composite key ensures shared-secret auth doesn't create
        # a single global bucket — each IP gets its own token quota.
        id_remaining = self._per_identity
        if token:
            identity_key = f"{client_ip}:{token[:8]}"
            id_ok, id_remaining, id_retry = await self._identity_limiter.is_allowed(
                identity_key
            )
            if not id_ok:
                logger.warning("Rate limit exceeded for identity (IP=%s)", client_ip)
                await self._send_429(send, id_retry, "Rate limit exceeded")
                return

        # Determine governing limit for response headers
        if token:
            governing_remaining = min(ip_remaining, id_remaining)
            governing_limit = min(self._ip_limiter.limit, self._identity_limiter.limit)
        else:
            governing_remaining = ip_remaining
            governing_limit = self._ip_limiter.limit

        async def rate_limit_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    [b"x-ratelimit-remaining", str(governing_remaining).encode()],
                    [b"x-ratelimit-limit", str(governing_limit).encode()],
                    [b"x-ratelimit-window", str(self._window).encode()],
                ])
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, rate_limit_send)

    @staticmethod
    def _extract_ip(scope: dict) -> str:
        """Extract client IP from ASGI scope.

        Only uses the direct peer address (scope["client"]).
        X-Forwarded-For is NOT trusted — an attacker can trivially
        spoof it to bypass IP-based rate limiting. If deployed behind
        a reverse proxy, configure the proxy to set the real client IP
        in scope["client"] (e.g., uvicorn --proxy-headers with
        --forwarded-allow-ips).
        """
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    @staticmethod
    def _extract_token(scope: dict) -> str | None:
        """Extract bearer token from Authorization header (if present)."""
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode("latin-1")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None

    @staticmethod
    async def _send_429(send: Callable, retry_after: float, detail: str) -> None:
        retry_int = max(1, int(retry_after + 0.5))
        body = json.dumps(
            {"error": "rate_limited", "detail": detail, "retry_after": retry_int},
            ensure_ascii=False,
        ).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                [b"content-type", b"application/json"],
                [b"retry-after", str(retry_int).encode()],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})
