"""Structured logging with context propagation.

Wraps :mod:`structlog` over the existing stdlib ``logging`` infrastructure
so that every log event carries structured key-value pairs (``task_id``,
``correlation_id``, ``session_id``, etc.) without changing existing call
sites.

Usage — new code::

    from openspace.domain.logging import get_logger, bind_context

    log = get_logger(__name__)

    # Bind context for the current async task / request
    bind_context(task_id="t-42", correlation_id="abc123")
    log.info("task_started", workspace="/tmp")  # structured event

Usage — existing code (zero changes needed)::

    # stdlib loggers continue to work; structlog processors
    # will format their output through the shared formatter.
    import logging
    logger = logging.getLogger("openspace.mcp_server")
    logger.info("old-style message")  # still works, gets structured formatting

Context propagation uses :mod:`contextvars` so it is safe across
``asyncio`` task boundaries.
"""

from __future__ import annotations

import contextvars
import logging
import re
import sys
from typing import Any, Dict, Optional

import structlog

# ── Context variables (async-safe) ────────────────────────────────────

_task_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="")
_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

_ALL_CONTEXT_VARS: Dict[str, contextvars.ContextVar[str]] = {
    "task_id": _task_id_var,
    "correlation_id": _correlation_id_var,
    "session_id": _session_id_var,
    "request_id": _request_id_var,
}


# ── Public context helpers ────────────────────────────────────────────


def bind_context(**kwargs: str) -> None:
    """Bind context variables for the current async task / request.

    Example::

        bind_context(task_id="t-42", correlation_id="abc123")
    """
    for key, value in kwargs.items():
        var = _ALL_CONTEXT_VARS.get(key)
        if var is not None:
            var.set(value)


def clear_context() -> None:
    """Reset all context variables to their defaults."""
    for var in _ALL_CONTEXT_VARS.values():
        var.set("")


def get_context() -> Dict[str, str]:
    """Return a snapshot of all non-empty context variables."""
    return {key: var.get() for key, var in _ALL_CONTEXT_VARS.items() if var.get()}


# ── Sensitive data redaction ──────────────────────────────────────────

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "token",
        "bearer_token",
        "password",
        "secret",
        "authorization",
        "credentials",
        "private_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "secret_key",
        "auth_token",
    }
)

# Suffixes / prefixes that indicate a key holds sensitive data.
# More precise than substring matching to avoid false positives like
# "token_count" or "basket_id".
_SENSITIVE_SUFFIXES = ("_key", "_token", "_secret", "_password", "_credential")
_SENSITIVE_PREFIXES = ("secret_", "password_", "auth_")

_MAX_VALUE_LENGTH = 1000

# camelCase → snake_case normalizer so "apiKey" matches "api_key"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _normalize_key(key: str) -> str:
    """Normalize a key to lower snake_case for sensitive matching."""
    return _CAMEL_BOUNDARY.sub(r"_\1", key).lower()


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name indicates sensitive data.

    Handles snake_case, camelCase, and PascalCase variants by normalizing
    to lower snake_case before matching (e.g. ``apiKey`` → ``api_key``).
    """
    normalized = _normalize_key(key)
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith(_SENSITIVE_SUFFIXES)
        or normalized.startswith(_SENSITIVE_PREFIXES)
    )


_MAX_REDACT_DEPTH = 10


def _redact_value(value: Any, parent_sensitive: bool = False, _depth: int = 0) -> Any:
    """Recursively redact sensitive values in nested structures.

    Stops at ``_MAX_REDACT_DEPTH`` to prevent ``RecursionError`` on
    deeply nested or self-referential payloads.
    """
    if parent_sensitive:
        return "***REDACTED***"
    if _depth >= _MAX_REDACT_DEPTH:
        return value
    if isinstance(value, dict):
        return {k: _redact_value(v, _is_sensitive_key(k), _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v, _depth=_depth + 1) for v in value)
    return value


def _redact_sensitive(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Structlog processor: redact sensitive keys and truncate long values.

    Handles nested dicts/lists recursively to prevent leakage via
    structured payloads in JSON logging mode.
    """
    for key in list(event_dict.keys()):
        if _is_sensitive_key(key):
            event_dict[key] = "***REDACTED***"
        elif isinstance(event_dict[key], (dict, list, tuple)):
            event_dict[key] = _redact_value(event_dict[key])
        elif isinstance(event_dict[key], str) and len(event_dict[key]) > _MAX_VALUE_LENGTH:
            event_dict[key] = event_dict[key][:_MAX_VALUE_LENGTH] + "...[truncated]"
    return event_dict


# ── Inject contextvars into every log event ───────────────────────────


def _inject_context_vars(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Structlog processor: merge contextvars into the event dict."""
    ctx = get_context()
    for key, value in ctx.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


# ── Configuration ─────────────────────────────────────────────────────

_configured = False


def configure_logging(
    *,
    level: int = logging.INFO,
    json_output: bool = False,
    colors: bool = True,
) -> None:
    """Configure structlog + stdlib logging.

    Call once at application startup.  Safe to call multiple times
    (subsequent calls are no-ops unless the module is reset).

    Args:
        level: stdlib log level (default INFO).
        json_output: If True, emit JSON lines to stdout (for production).
                     If False, emit human-readable colored output.
        colors: Enable colored console output (ignored if json_output=True).
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Shared structlog processors (run for both structlog and stdlib loggers)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context_vars,
        _redact_sensitive,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON lines
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: human-readable
        renderer = structlog.dev.ConsoleRenderer(colors=colors)

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root logger with structlog formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        # Apply shared processors (redaction, context injection) to stdlib
        # log records that bypass structlog's pipeline.
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "litellm", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def reset_logging() -> None:
    """Reset logging configuration (for testing)."""
    global _configured
    _configured = False
    structlog.reset_defaults()


# ── Public logger factory ─────────────────────────────────────────────


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger.

    Auto-configures on first call if not yet configured.
    Compatible with stdlib logging — all events flow through
    the same formatter pipeline.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name or "openspace")


__all__ = [
    "bind_context",
    "clear_context",
    "configure_logging",
    "get_context",
    "get_logger",
    "reset_logging",
]
