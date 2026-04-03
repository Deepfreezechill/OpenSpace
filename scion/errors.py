"""Structured error handling for MCP responses.

All MCP tool responses MUST use these helpers so that internal details
(tracebacks, file paths, line numbers, module names) are NEVER leaked
to the client.  Full diagnostics are logged server-side with a
correlation ID that operators can use to match client errors to logs.

Error codes
-----------
EXECUTION_ERROR   — Task execution failed (execute_task runtime errors)
VALIDATION_ERROR  — Bad input from the caller (missing / invalid args)
SKILL_NOT_FOUND   — Requested skill directory or record doesn't exist
PERMISSION_DENIED — Auth / authz failure
INTERNAL_ERROR    — Catch-all for unexpected server errors
TIMEOUT_ERROR     — Operation exceeded time limit
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger("scion.mcp_server")

# ── Error codes ──────────────────────────────────────────────────────
EXECUTION_ERROR = "EXECUTION_ERROR"
VALIDATION_ERROR = "VALIDATION_ERROR"
SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
PERMISSION_DENIED = "PERMISSION_DENIED"
INTERNAL_ERROR = "INTERNAL_ERROR"
TIMEOUT_ERROR = "TIMEOUT_ERROR"

# Patterns that must never appear in client-facing messages
_TRACEBACK_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)"
    r"|File \".+\", line \d+"
    r"|^\s+raise\s"
    r"|^\s+at\s+[\w.]+\("
    r"|openspace[/\\.]"
    r"|\.py:\d+"
    r"|\.py\b",
    re.MULTILINE,
)

# Generic fallback — never expose internal exception class names
_GENERIC_ERROR = "An internal error occurred"


def _generate_correlation_id() -> str:
    """Short correlation ID for matching client errors to server logs."""
    return uuid.uuid4().hex[:12]


def sanitize_error(exc: BaseException) -> str:
    """Extract a safe, human-readable message from an exception.

    Strips file paths, line numbers, module names, and stack traces.
    Returns a generic message if the raw string contains internal details.
    Never returns exception class names (type(exc).__name__).
    """
    raw = str(exc)
    if not raw or _TRACEBACK_PATTERNS.search(raw):
        return _GENERIC_ERROR
    # Windows paths — including spaces and quoted paths
    sanitized = re.sub(r"[A-Za-z]:\\[^\s\"']*(?:\s[^\s\\\"']+)*\.?\w*", "<path>", raw)
    # UNC paths (\\server\share\...)
    sanitized = re.sub(r"\\\\[^\s\"']+", "<path>", sanitized)
    # Unix-style absolute paths
    sanitized = re.sub(r"/(?:[\w.-]+/)+[\w.-]*", "<path>", sanitized)
    # Dotted module names (e.g. scion.cloud.auth.TokenResolver)
    sanitized = re.sub(r"\b\w+(?:\.\w+){2,}\b", "<module>", sanitized)
    # Standalone line-number references
    sanitized = re.sub(r"\bline \d+\b", "<location>", sanitized)
    # If everything got redacted to placeholders, return generic
    cleaned = re.sub(r"<(?:path|module|location)>", "", sanitized).strip()
    if not cleaned:
        return _GENERIC_ERROR
    # Truncate to a reasonable length
    if len(sanitized) > 300:
        sanitized = sanitized[:297] + "..."
    return sanitized


def safe_error_response(
    error_code: str,
    message: str,
    *,
    correlation_id: str | None = None,
) -> str:
    """Build a structured JSON error response for MCP tool results.

    Returns a JSON string with:
      - isError: true
      - error_code: one of the module-level constants
      - message: human-readable description (no internals)
      - correlation_id: opaque ID to match server-side logs
    """
    cid = correlation_id or _generate_correlation_id()
    payload: dict[str, Any] = {
        "isError": True,
        "error_code": error_code,
        "message": message,
        "correlation_id": cid,
    }
    return json.dumps(payload, ensure_ascii=False)


def handle_mcp_exception(
    exc: BaseException,
    *,
    tool_name: str,
    error_code: str = INTERNAL_ERROR,
) -> str:
    """One-liner for MCP except blocks: log full traceback, return safe JSON.

    For :class:`~scion.domain.exceptions.OpenSpaceError` instances,
    uses ``client_message`` (never the raw message) and ``error_code``
    from the exception.  For all other exceptions, sanitizes via
    :func:`sanitize_error`.

    Usage::

        except Exception as e:
            return handle_mcp_exception(e, tool_name="execute_task",
                                         error_code=EXECUTION_ERROR)
    """
    cid = _generate_correlation_id()
    logger.error(
        "%s failed [%s]: %s",
        tool_name,
        cid,
        exc,
        exc_info=True,
    )

    # Prefer domain exception's safe client_message when available
    try:
        from scion.domain.exceptions import OpenSpaceError as _OSE
        from scion.domain.exceptions import map_to_mcp_error_code

        if isinstance(exc, _OSE):
            safe_msg = exc.client_message
            error_code = exc.error_code
            return safe_error_response(error_code, safe_msg, correlation_id=cid)

        # Use centralized mapping for builtin exceptions too
        error_code = map_to_mcp_error_code(exc)
    except ImportError:
        pass

    safe_msg = sanitize_error(exc)
    return safe_error_response(error_code, safe_msg, correlation_id=cid)
