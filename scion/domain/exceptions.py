"""Domain exception hierarchy.

All domain-specific exceptions inherit from :class:`ScionError`.
Each subclass maps to a stable ``error_code`` that the MCP layer can
surface to callers without leaking internals.

Usage::

    from scion.domain.exceptions import ValidationError, NotFoundError

    raise ValidationError("skill_dirs must be a list")
    raise NotFoundError("skill", skill_id="abc-123")

The centralized :func:`map_to_mcp_error_code` converts any exception
into the appropriate MCP error code string for ``safe_error_response()``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Generic fallback for client_message when no safe_message is provided.
_GENERIC_CLIENT_MESSAGE = "An internal error occurred"


# ═══════════════════════════════════════════════════════════════════════
#  Base exception
# ═══════════════════════════════════════════════════════════════════════


class ScionError(Exception):
    """Root exception for all domain errors.

    Attributes:
        message: Human-readable description (server-side only, may contain internals).
        error_code: Stable string code for MCP/API responses.
        retryable: Whether the caller may retry the operation.
        context: Extra key-value pairs for **server-side** logging / diagnostics.
        safe_message: Optional sanitized message for client-facing use.
    """

    error_code: str = "INTERNAL_ERROR"
    retryable: bool = False

    def __init__(
        self,
        message: str = "",
        *,
        retryable: Optional[bool] = None,
        safe_message: Optional[str] = None,
        **context: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable
        self.safe_message = safe_message
        self.context: Dict[str, Any] = context

    @property
    def client_message(self) -> str:
        """Message safe for client-facing responses.

        Returns ``safe_message`` if explicitly set, otherwise a generic
        fallback.  **Never** returns the raw ``message`` to prevent
        accidental disclosure of internal details.
        """
        return self.safe_message or _GENERIC_CLIENT_MESSAGE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for **server-side** structured logging.

        .. warning::
           This output may contain sensitive data from ``message`` and
           ``context``.  Do NOT send it to clients.  Use
           :meth:`to_safe_dict` for client-facing serialization.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context,
        }

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize for client-facing responses (redacted)."""
        return {
            "error_code": self.error_code,
            "message": self.client_message,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        ctx = f", context={self.context}" if self.context else ""
        return f"{type(self).__name__}({self.message!r}{ctx})"


# ═══════════════════════════════════════════════════════════════════════
#  Concrete domain exceptions
# ═══════════════════════════════════════════════════════════════════════


class ValidationError(ScionError):
    """Bad input from the caller (missing / invalid args)."""

    error_code = "VALIDATION_ERROR"


class NotFoundError(ScionError):
    """Requested resource does not exist.

    Usage::

        raise NotFoundError("skill", skill_id="abc-123")
        raise NotFoundError("session", session_name="default")
    """

    error_code = "SKILL_NOT_FOUND"

    def __init__(self, resource_type: str = "resource", **context: Any) -> None:
        resource_id = context.get("skill_id") or context.get("session_name") or ""
        msg = f"{resource_type} not found"
        if resource_id:
            msg += f": {resource_id}"
        super().__init__(msg, **context)
        self.resource_type = resource_type


class PermissionDeniedError(ScionError):
    """Authentication or authorization failure."""

    error_code = "PERMISSION_DENIED"


class OperationTimeoutError(ScionError):
    """Operation exceeded its time limit.

    Named ``OperationTimeoutError`` to avoid shadowing the builtin
    ``TimeoutError``.
    """

    error_code = "TIMEOUT_ERROR"
    retryable = True


class ExecutionError(ScionError):
    """Task execution failed at runtime."""

    error_code = "EXECUTION_ERROR"


class DependencyError(ScionError):
    """Required external dependency is missing or broken."""

    error_code = "EXECUTION_ERROR"

    def __init__(self, message: str = "", *, dependency: str = "", **context: Any) -> None:
        super().__init__(message, dependency=dependency, **context)
        self.dependency = dependency


class ConfigurationError(ScionError):
    """Invalid or missing configuration."""

    error_code = "VALIDATION_ERROR"


class ExternalServiceError(ScionError):
    """Upstream / third-party service failure.

    ``retryable`` is derived from ``status_code`` when present:
    5xx and 429 are retryable; 4xx (except 429) are not.
    Can be overridden explicitly.
    """

    error_code = "EXECUTION_ERROR"

    def __init__(
        self,
        message: str = "",
        *,
        service: str = "",
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
        **context: Any,
    ) -> None:
        # Derive retryable from status_code if not explicitly set
        if retryable is None and status_code is not None:
            if status_code == 429 or 500 <= status_code < 600:
                retryable = True
            elif 400 <= status_code < 500:
                retryable = False
            else:
                retryable = True  # Unknown codes: optimistic retry
        elif retryable is None:
            retryable = True  # Default for unknown upstream failures
        super().__init__(message, retryable=retryable, service=service, **context)
        self.service = service
        self.status_code = status_code


class SandboxError(ScionError):
    """Sandbox creation, execution, or teardown failure."""

    error_code = "EXECUTION_ERROR"


class EvolutionError(ScionError):
    """Skill evolution failed."""

    error_code = "EXECUTION_ERROR"


class InternalError(ScionError):
    """Catch-all for unexpected server errors."""

    error_code = "INTERNAL_ERROR"


# ═══════════════════════════════════════════════════════════════════════
#  Centralized error code mapping
# ═══════════════════════════════════════════════════════════════════════

# Built-in exceptions that map to specific MCP codes.
_BUILTIN_TO_CODE: list[tuple[type, str]] = [
    (TimeoutError, "TIMEOUT_ERROR"),  # builtin TimeoutError
    (PermissionError, "PERMISSION_DENIED"),  # builtin PermissionError
    (FileNotFoundError, "SKILL_NOT_FOUND"),  # builtin FileNotFoundError
    (ValueError, "VALIDATION_ERROR"),  # builtin ValueError
]


def map_to_mcp_error_code(exc: BaseException) -> str:
    """Convert any exception to the appropriate MCP error code.

    Handles:
    1. :class:`ScionError` subclasses — returns their ``error_code``
    2. Built-in exceptions (TimeoutError, PermissionError, etc.)
    3. Unknown exceptions — returns ``INTERNAL_ERROR``
    """
    if isinstance(exc, ScionError):
        return exc.error_code
    for exc_type, code in _BUILTIN_TO_CODE:
        if isinstance(exc, exc_type):
            return code
    return "INTERNAL_ERROR"


__all__ = [
    "ConfigurationError",
    "DependencyError",
    "EvolutionError",
    "ExecutionError",
    "ExternalServiceError",
    "InternalError",
    "NotFoundError",
    "ScionError",
    "OperationTimeoutError",
    "PermissionDeniedError",
    "SandboxError",
    "ValidationError",
    "map_to_mcp_error_code",
]
