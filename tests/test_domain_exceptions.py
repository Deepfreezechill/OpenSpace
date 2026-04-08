"""Tests for EPIC 1.5 — Domain Exception Hierarchy (Issues #72-75).

Validates:
- Exception hierarchy structure and inheritance
- Error code mapping for all exception types
- Serialization (to_dict)
- Context propagation
- Centralized map_to_mcp_error_code()
- Retryable flag behavior
- Client-safe message handling
- NotFoundError resource type formatting
- Integration with existing errors.py helpers
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════════
#  Hierarchy & Inheritance Tests
# ═══════════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """All domain exceptions inherit from ScionError."""

    def test_all_exceptions_importable(self):
        from scion.domain.exceptions import (
            ConfigurationError,
            DependencyError,
            EvolutionError,
            ExecutionError,
            ExternalServiceError,
            InternalError,
            NotFoundError,
            ScionError,
            OperationTimeoutError,
            PermissionDeniedError,
            SandboxError,
            ValidationError,
        )

        all_exc = [
            ConfigurationError,
            DependencyError,
            EvolutionError,
            ExecutionError,
            ExternalServiceError,
            InternalError,
            NotFoundError,
            ScionError,
            OperationTimeoutError,
            PermissionDeniedError,
            SandboxError,
            ValidationError,
        ]
        assert len(all_exc) == 12

    def test_all_subclass_scion_error(self):
        from scion.domain.exceptions import (
            ConfigurationError,
            DependencyError,
            EvolutionError,
            ExecutionError,
            ExternalServiceError,
            InternalError,
            NotFoundError,
            ScionError,
            OperationTimeoutError,
            PermissionDeniedError,
            SandboxError,
            ValidationError,
        )

        for exc_cls in [
            ConfigurationError,
            DependencyError,
            EvolutionError,
            ExecutionError,
            ExternalServiceError,
            InternalError,
            NotFoundError,
            OperationTimeoutError,
            PermissionDeniedError,
            SandboxError,
            ValidationError,
        ]:
            assert issubclass(exc_cls, ScionError), f"{exc_cls.__name__} does not inherit ScionError"
            assert issubclass(exc_cls, Exception)

    def test_scion_error_is_exception(self):
        from scion.domain.exceptions import ScionError

        assert issubclass(ScionError, Exception)

    def test_can_catch_all_with_scion_error(self):
        from scion.domain.exceptions import (
            ExecutionError,
            NotFoundError,
            ScionError,
            ValidationError,
        )

        for exc_cls in [ValidationError, NotFoundError, ExecutionError]:
            try:
                raise exc_cls("test")
            except ScionError:
                pass  # Expected
            else:
                pytest.fail(f"{exc_cls.__name__} not caught by ScionError")


# ═══════════════════════════════════════════════════════════════════════
#  Error Code Tests
# ═══════════════════════════════════════════════════════════════════════


class TestErrorCodes:
    """Each exception type has the correct error_code."""

    @pytest.mark.parametrize(
        "exc_cls,expected_code",
        [
            ("ValidationError", "VALIDATION_ERROR"),
            ("ConfigurationError", "VALIDATION_ERROR"),
            ("NotFoundError", "SKILL_NOT_FOUND"),
            ("PermissionDeniedError", "PERMISSION_DENIED"),
            ("OperationTimeoutError", "TIMEOUT_ERROR"),
            ("ExecutionError", "EXECUTION_ERROR"),
            ("DependencyError", "EXECUTION_ERROR"),
            ("ExternalServiceError", "EXECUTION_ERROR"),
            ("SandboxError", "EXECUTION_ERROR"),
            ("EvolutionError", "EXECUTION_ERROR"),
            ("InternalError", "INTERNAL_ERROR"),
            ("ScionError", "INTERNAL_ERROR"),
        ],
    )
    def test_error_code_mapping(self, exc_cls: str, expected_code: str):
        import scion.domain.exceptions as mod

        cls = getattr(mod, exc_cls)
        exc = cls("test message")
        assert exc.error_code == expected_code


# ═══════════════════════════════════════════════════════════════════════
#  Serialization Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSerialization:
    """to_dict() and __str__/__repr__ work correctly."""

    def test_to_dict_basic(self):
        from scion.domain.exceptions import ValidationError

        exc = ValidationError("bad input", field="name")
        d = exc.to_dict()
        assert d["error_code"] == "VALIDATION_ERROR"
        assert d["message"] == "bad input"
        assert d["retryable"] is False
        assert d["context"]["field"] == "name"

    def test_to_dict_with_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        exc = ExternalServiceError("API down", service="cloud", status_code=503)
        d = exc.to_dict()
        assert d["retryable"] is True
        assert d["context"]["service"] == "cloud"

    def test_str_includes_error_code(self):
        from scion.domain.exceptions import NotFoundError

        exc = NotFoundError("skill", skill_id="abc-123")
        s = str(exc)
        assert "[SKILL_NOT_FOUND]" in s
        assert "skill not found: abc-123" in s

    def test_repr_includes_class_name(self):
        from scion.domain.exceptions import ExecutionError

        exc = ExecutionError("task failed", task_id="t42")
        r = repr(exc)
        assert "ExecutionError" in r
        assert "task failed" in r


# ═══════════════════════════════════════════════════════════════════════
#  Context Propagation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestContextPropagation:
    """Context kwargs are preserved in .context dict."""

    def test_context_stored(self):
        from scion.domain.exceptions import ExecutionError

        exc = ExecutionError("boom", task_id="t1", tool_name="bash", iteration=3)
        assert exc.context["task_id"] == "t1"
        assert exc.context["tool_name"] == "bash"
        assert exc.context["iteration"] == 3

    def test_empty_context(self):
        from scion.domain.exceptions import ValidationError

        exc = ValidationError("no context")
        assert exc.context == {}

    def test_not_found_resource_type(self):
        from scion.domain.exceptions import NotFoundError

        exc = NotFoundError("session", session_name="default")
        assert exc.resource_type == "session"
        assert "session not found: default" in exc.message

    def test_not_found_without_id(self):
        from scion.domain.exceptions import NotFoundError

        exc = NotFoundError("skill")
        assert "skill not found" in exc.message
        assert ":" not in exc.message  # No ID appended

    def test_dependency_error_stores_dependency(self):
        from scion.domain.exceptions import DependencyError

        exc = DependencyError("npm not found", dependency="npm")
        assert exc.dependency == "npm"

    def test_external_service_error_stores_service(self):
        from scion.domain.exceptions import ExternalServiceError

        exc = ExternalServiceError("timeout", service="cloud-api", status_code=504)
        assert exc.service == "cloud-api"
        assert exc.status_code == 504


# ═══════════════════════════════════════════════════════════════════════
#  Retryable Flag Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRetryable:
    """Retryable defaults are correct and overridable."""

    def test_default_not_retryable(self):
        from scion.domain.exceptions import ValidationError

        assert ValidationError("x").retryable is False

    def test_timeout_default_retryable(self):
        from scion.domain.exceptions import OperationTimeoutError

        assert OperationTimeoutError("x").retryable is True

    def test_external_service_default_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        # No status_code → default retryable
        assert ExternalServiceError("x").retryable is True

    def test_external_service_4xx_not_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        assert ExternalServiceError("x", status_code=401).retryable is False

    def test_external_service_any_4xx_not_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        # All 4xx (except 429) should be non-retryable
        for code in [
            400,
            402,
            403,
            404,
            405,
            406,
            407,
            408,
            410,
            411,
            412,
            413,
            414,
            415,
            416,
            417,
            418,
            421,
            422,
            423,
            424,
            425,
            426,
            428,
            431,
            451,
        ]:
            exc = ExternalServiceError("x", status_code=code)
            assert exc.retryable is False, f"status_code={code} should NOT be retryable"

    def test_external_service_429_is_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        assert ExternalServiceError("x", status_code=429).retryable is True
        assert ExternalServiceError("x", status_code=403).retryable is False
        assert ExternalServiceError("x", status_code=404).retryable is False

    def test_external_service_5xx_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        assert ExternalServiceError("x", status_code=500).retryable is True
        assert ExternalServiceError("x", status_code=503).retryable is True
        assert ExternalServiceError("x", status_code=429).retryable is True

    def test_external_service_edge_case_status_codes(self):
        from scion.domain.exceptions import ExternalServiceError

        # Boundary: 399 is outside 4xx range → optimistic retry
        assert ExternalServiceError("x", status_code=399).retryable is True
        # Boundary: 500 is start of 5xx → retryable
        assert ExternalServiceError("x", status_code=500).retryable is True
        # Boundary: 599 is end of 5xx → retryable
        assert ExternalServiceError("x", status_code=599).retryable is True
        # Boundary: 600+ is unknown → optimistic retry
        assert ExternalServiceError("x", status_code=600).retryable is True
        # Boundary: 200 is success range → optimistic retry (shouldn't happen but safe)
        assert ExternalServiceError("x", status_code=200).retryable is True

    def test_override_retryable(self):
        from scion.domain.exceptions import ValidationError

        exc = ValidationError("retry me", retryable=True)
        assert exc.retryable is True

    def test_override_non_retryable(self):
        from scion.domain.exceptions import OperationTimeoutError

        exc = OperationTimeoutError("no retry", retryable=False)
        assert exc.retryable is False

    def test_external_service_override_retryable(self):
        from scion.domain.exceptions import ExternalServiceError

        # Override: force retryable even on 401
        exc = ExternalServiceError("x", status_code=401, retryable=True)
        assert exc.retryable is True


# ═══════════════════════════════════════════════════════════════════════
#  Client-Safe Message Tests
# ═══════════════════════════════════════════════════════════════════════


class TestClientMessage:
    """safe_message / client_message handling."""

    def test_client_message_defaults_to_generic(self):
        from scion.domain.exceptions import ExecutionError

        exc = ExecutionError("internal details here")
        # Without safe_message, client_message returns generic (never raw)
        assert exc.client_message == "An internal error occurred"

    def test_client_message_uses_safe_message(self):
        from scion.domain.exceptions import ExecutionError

        exc = ExecutionError(
            "NullPointerException at line 42",
            safe_message="Task execution failed",
        )
        assert exc.client_message == "Task execution failed"
        assert exc.message == "NullPointerException at line 42"

    def test_to_safe_dict_redacts(self):
        from scion.domain.exceptions import ExecutionError

        exc = ExecutionError(
            "secret path /opt/secrets/key.pem",
            safe_message="Task failed",
            task_id="t-42",
        )
        safe = exc.to_safe_dict()
        assert safe["message"] == "Task failed"
        assert "context" not in safe  # No context in safe dict
        assert safe["error_code"] == "EXECUTION_ERROR"

    def test_to_safe_dict_without_safe_message(self):
        from scion.domain.exceptions import ExecutionError

        exc = ExecutionError("internal details")
        safe = exc.to_safe_dict()
        assert safe["message"] == "An internal error occurred"


# ═══════════════════════════════════════════════════════════════════════
#  Centralized Mapping Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMapToMCPErrorCode:
    """map_to_mcp_error_code() handles all exception types."""

    def test_domain_exceptions_map_correctly(self):
        from scion.domain.exceptions import (
            ConfigurationError,
            DependencyError,
            ExecutionError,
            ExternalServiceError,
            InternalError,
            NotFoundError,
            OperationTimeoutError,
            PermissionDeniedError,
            SandboxError,
            ValidationError,
            map_to_mcp_error_code,
        )

        assert map_to_mcp_error_code(ValidationError("x")) == "VALIDATION_ERROR"
        assert map_to_mcp_error_code(ConfigurationError("x")) == "VALIDATION_ERROR"
        assert map_to_mcp_error_code(NotFoundError("x")) == "SKILL_NOT_FOUND"
        assert map_to_mcp_error_code(PermissionDeniedError("x")) == "PERMISSION_DENIED"
        assert map_to_mcp_error_code(OperationTimeoutError("x")) == "TIMEOUT_ERROR"
        assert map_to_mcp_error_code(ExecutionError("x")) == "EXECUTION_ERROR"
        assert map_to_mcp_error_code(DependencyError("x")) == "EXECUTION_ERROR"
        assert map_to_mcp_error_code(ExternalServiceError("x")) == "EXECUTION_ERROR"
        assert map_to_mcp_error_code(SandboxError("x")) == "EXECUTION_ERROR"
        assert map_to_mcp_error_code(InternalError("x")) == "INTERNAL_ERROR"

    def test_unknown_exception_maps_to_internal(self):
        from scion.domain.exceptions import map_to_mcp_error_code

        assert map_to_mcp_error_code(RuntimeError("oops")) == "INTERNAL_ERROR"
        assert map_to_mcp_error_code(Exception("generic")) == "INTERNAL_ERROR"

    def test_builtin_timeout_maps_to_timeout(self):
        from scion.domain.exceptions import map_to_mcp_error_code

        assert map_to_mcp_error_code(TimeoutError("t")) == "TIMEOUT_ERROR"

    def test_builtin_permission_maps_to_denied(self):
        from scion.domain.exceptions import map_to_mcp_error_code

        assert map_to_mcp_error_code(PermissionError("p")) == "PERMISSION_DENIED"

    def test_builtin_file_not_found_maps_to_not_found(self):
        from scion.domain.exceptions import map_to_mcp_error_code

        assert map_to_mcp_error_code(FileNotFoundError("f")) == "SKILL_NOT_FOUND"

    def test_builtin_value_error_maps_to_validation(self):
        from scion.domain.exceptions import map_to_mcp_error_code

        assert map_to_mcp_error_code(ValueError("bad")) == "VALIDATION_ERROR"

    def test_base_scion_error_maps_to_internal(self):
        from scion.domain.exceptions import ScionError, map_to_mcp_error_code

        assert map_to_mcp_error_code(ScionError("x")) == "INTERNAL_ERROR"


# ═══════════════════════════════════════════════════════════════════════
#  Integration with existing errors.py
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationWithExistingErrors:
    """Domain exceptions work with existing error helpers."""

    def test_sanitize_error_handles_domain_exception(self):
        from scion.domain.exceptions import ExecutionError
        from scion.errors import sanitize_error

        exc = ExecutionError("simple error message")
        safe = sanitize_error(exc)
        assert "simple error message" in safe

    def test_handle_mcp_exception_with_domain_exception(self):
        import json

        from scion.domain.exceptions import ValidationError
        from scion.errors import handle_mcp_exception

        result = handle_mcp_exception(
            ValidationError("bad input", safe_message="Invalid request"),
            tool_name="execute_task",
            error_code="VALIDATION_ERROR",
        )
        parsed = json.loads(result)
        assert parsed["isError"] is True
        assert parsed["error_code"] == "VALIDATION_ERROR"
        assert parsed["message"] == "Invalid request"  # Uses client_message
        assert "correlation_id" in parsed

    def test_handle_mcp_exception_prefers_domain_error_code(self):
        import json

        from scion.domain.exceptions import NotFoundError
        from scion.errors import handle_mcp_exception

        # Even though we pass error_code=EXECUTION_ERROR, the domain
        # exception's error_code should win
        result = handle_mcp_exception(
            NotFoundError("skill", skill_id="abc"),
            tool_name="fix_skill",
            error_code="EXECUTION_ERROR",
        )
        parsed = json.loads(result)
        assert parsed["error_code"] == "SKILL_NOT_FOUND"

    def test_handle_mcp_exception_generic_fallback_without_safe_message(self):
        import json

        from scion.domain.exceptions import ExecutionError
        from scion.errors import handle_mcp_exception

        # Without safe_message, client_message returns generic fallback
        result = handle_mcp_exception(
            ExecutionError("secret internal stack trace here"),
            tool_name="execute_task",
            error_code="EXECUTION_ERROR",
        )
        parsed = json.loads(result)
        assert parsed["isError"] is True
        assert parsed["message"] == "An internal error occurred"
        assert "secret" not in parsed["message"]

    def test_handle_mcp_exception_maps_builtin_timeout(self):
        import json

        from scion.errors import handle_mcp_exception

        result = handle_mcp_exception(
            TimeoutError("connection timed out"),
            tool_name="call_api",
            error_code="EXECUTION_ERROR",  # caller passes generic
        )
        parsed = json.loads(result)
        assert parsed["error_code"] == "TIMEOUT_ERROR"  # centralized mapping wins

    def test_handle_mcp_exception_maps_builtin_permission(self):
        import json

        from scion.errors import handle_mcp_exception

        result = handle_mcp_exception(
            PermissionError("access denied"),
            tool_name="read_file",
            error_code="EXECUTION_ERROR",
        )
        parsed = json.loads(result)
        assert parsed["error_code"] == "PERMISSION_DENIED"

    def test_handle_mcp_exception_maps_builtin_value_error(self):
        import json

        from scion.errors import handle_mcp_exception

        result = handle_mcp_exception(
            ValueError("invalid argument"),
            tool_name="parse_input",
            error_code="EXECUTION_ERROR",
        )
        parsed = json.loads(result)
        assert parsed["error_code"] == "VALIDATION_ERROR"
