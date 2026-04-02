"""Tests for EPIC 1.6 — Structured Logging (Issues #76-79).

Validates:
- Context variable binding and propagation
- Sensitive data redaction
- structlog configuration and logger creation
- Integration with stdlib logging
- Context isolation across async tasks
- Reset/reconfigure behavior
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest


# ═══════════════════════════════════════════════════════════════════════
#  Context Variable Tests
# ═══════════════════════════════════════════════════════════════════════


class TestContextVariables:
    """bind_context / clear_context / get_context work correctly."""

    def setup_method(self):
        from openspace.domain.logging import clear_context

        clear_context()

    def teardown_method(self):
        from openspace.domain.logging import clear_context

        clear_context()

    def test_bind_and_get_context(self):
        from openspace.domain.logging import bind_context, get_context

        bind_context(task_id="t-42", correlation_id="abc123")
        ctx = get_context()
        assert ctx["task_id"] == "t-42"
        assert ctx["correlation_id"] == "abc123"
        assert "session_id" not in ctx  # Empty values excluded

    def test_clear_context(self):
        from openspace.domain.logging import bind_context, clear_context, get_context

        bind_context(task_id="t-1", session_id="s-1")
        clear_context()
        assert get_context() == {}

    def test_bind_unknown_key_ignored(self):
        from openspace.domain.logging import bind_context, get_context

        bind_context(task_id="t-1", unknown_field="ignored")
        ctx = get_context()
        assert ctx == {"task_id": "t-1"}

    def test_bind_overwrites_previous(self):
        from openspace.domain.logging import bind_context, get_context

        bind_context(task_id="t-1")
        bind_context(task_id="t-2")
        assert get_context()["task_id"] == "t-2"

    def test_all_four_context_vars(self):
        from openspace.domain.logging import bind_context, get_context

        bind_context(
            task_id="t-1",
            correlation_id="c-1",
            session_id="s-1",
            request_id="r-1",
        )
        ctx = get_context()
        assert len(ctx) == 4
        assert ctx["task_id"] == "t-1"
        assert ctx["correlation_id"] == "c-1"
        assert ctx["session_id"] == "s-1"
        assert ctx["request_id"] == "r-1"


class TestContextIsolation:
    """Context vars are isolated across async tasks."""

    def setup_method(self):
        from openspace.domain.logging import clear_context

        clear_context()

    def teardown_method(self):
        from openspace.domain.logging import clear_context

        clear_context()

    @pytest.mark.asyncio
    async def test_async_task_isolation(self):
        from openspace.domain.logging import bind_context, get_context

        results = {}

        async def task_a():
            bind_context(task_id="task-a")
            await asyncio.sleep(0.01)
            results["a"] = get_context().get("task_id", "")

        async def task_b():
            bind_context(task_id="task-b")
            await asyncio.sleep(0.01)
            results["b"] = get_context().get("task_id", "")

        await asyncio.gather(task_a(), task_b())
        assert results["a"] == "task-a"
        assert results["b"] == "task-b"


# ═══════════════════════════════════════════════════════════════════════
#  Redaction Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRedaction:
    """Sensitive data is redacted in log events."""

    def test_redacts_api_key(self):
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "api_key": "sk-secret-123"}
        result = _redact_sensitive(None, "info", event)
        assert result["api_key"] == "***REDACTED***"

    def test_redacts_token_field(self):
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "bearer_token": "eyJ..."}
        result = _redact_sensitive(None, "info", event)
        assert result["bearer_token"] == "***REDACTED***"

    def test_redacts_password(self):
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "password": "hunter2"}
        result = _redact_sensitive(None, "info", event)
        assert result["password"] == "***REDACTED***"

    def test_redacts_key_suffix_match(self):
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "my_api_key": "secret", "client_secret": "s3cret"}
        result = _redact_sensitive(None, "info", event)
        assert result["my_api_key"] == "***REDACTED***"
        assert result["client_secret"] == "***REDACTED***"

    def test_no_false_positive_on_non_sensitive(self):
        from openspace.domain.logging import _redact_sensitive

        # These should NOT be redacted despite containing "key"/"token" substrings
        event = {
            "event": "test",
            "token_count": 42,
            "keyboard_layout": "qwerty",
            "monkey_patch": True,
        }
        result = _redact_sensitive(None, "info", event)
        assert result["token_count"] == 42
        assert result["keyboard_layout"] == "qwerty"
        assert result["monkey_patch"] is True

    def test_redacts_nested_dict_sensitive_keys(self):
        from openspace.domain.logging import _redact_sensitive

        event = {
            "event": "api_call",
            "payload": {"api_key": "sk-secret", "user_id": "u-42"},
        }
        result = _redact_sensitive(None, "info", event)
        assert result["payload"]["api_key"] == "***REDACTED***"
        assert result["payload"]["user_id"] == "u-42"

    def test_redacts_deeply_nested(self):
        from openspace.domain.logging import _redact_sensitive

        event = {
            "event": "test",
            "response": {"headers": {"authorization": "Bearer xyz"}},
        }
        result = _redact_sensitive(None, "info", event)
        assert result["response"]["headers"]["authorization"] == "***REDACTED***"

    def test_redacts_top_level_list_with_sensitive_dicts(self):
        """Top-level list/tuple fields containing dicts with sensitive keys."""
        from openspace.domain.logging import _redact_sensitive

        event = {
            "event": "test",
            "payload": [{"authorization": "Bearer xyz"}, {"safe": "ok"}],
        }
        result = _redact_sensitive(None, "info", event)
        assert result["payload"][0]["authorization"] == "***REDACTED***"
        assert result["payload"][1]["safe"] == "ok"

    def test_redacts_top_level_tuple_with_sensitive_dicts(self):
        """Tuple variant of top-level list redaction."""
        from openspace.domain.logging import _redact_sensitive

        event = {
            "event": "test",
            "items": ({"api_key": "secret123"},),
        }
        result = _redact_sensitive(None, "info", event)
        assert result["items"][0]["api_key"] == "***REDACTED***"

    def test_redacts_camel_case_keys(self):
        """camelCase keys like apiKey, accessToken are normalized and redacted."""
        from openspace.domain.logging import _redact_sensitive

        event = {
            "event": "test",
            "apiKey": "sk-abc123",
            "accessToken": "tok-xyz",
            "clientSecret": "s3cr3t",
            "refreshToken": "ref-999",
        }
        result = _redact_sensitive(None, "info", event)
        assert result["apiKey"] == "***REDACTED***"
        assert result["accessToken"] == "***REDACTED***"
        assert result["clientSecret"] == "***REDACTED***"
        assert result["refreshToken"] == "***REDACTED***"

    def test_redacts_pascal_case_keys(self):
        """PascalCase keys are also normalized and redacted."""
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "ApiKey": "key1", "AuthToken": "tok1"}
        result = _redact_sensitive(None, "info", event)
        assert result["ApiKey"] == "***REDACTED***"
        assert result["AuthToken"] == "***REDACTED***"

    def test_camel_case_not_false_positive(self):
        """camelCase keys that aren't sensitive should NOT be redacted."""
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "tokenCount": 42, "keyboardLayout": "us"}
        result = _redact_sensitive(None, "info", event)
        assert result["tokenCount"] == 42
        assert result["keyboardLayout"] == "us"

    def test_recursion_depth_limit(self):
        """Deeply nested payloads stop redacting at _MAX_REDACT_DEPTH, no RecursionError."""
        from openspace.domain.logging import _MAX_REDACT_DEPTH, _redact_value

        # Build a structure deeper than the limit
        nested: dict = {"api_key": "leak-at-depth"}
        for _ in range(_MAX_REDACT_DEPTH + 5):
            nested = {"inner": nested}

        result = _redact_value(nested)
        # Walk down to the depth limit — should still be a dict
        node = result
        for _ in range(_MAX_REDACT_DEPTH + 5):
            node = node["inner"]
        # Beyond depth limit, the sensitive key is NOT redacted (safe bail-out)
        assert node["api_key"] == "leak-at-depth"

        # But within-limit nesting IS redacted
        shallow = {"wrapper": {"api_key": "should-redact"}}
        result2 = _redact_value(shallow)
        assert result2["wrapper"]["api_key"] == "***REDACTED***"

    def test_preserves_non_sensitive(self):
        from openspace.domain.logging import _redact_sensitive

        event = {"event": "test", "task_id": "t-42", "status": "ok"}
        result = _redact_sensitive(None, "info", event)
        assert result["task_id"] == "t-42"
        assert result["status"] == "ok"

    def test_truncates_long_values(self):
        from openspace.domain.logging import _MAX_VALUE_LENGTH, _redact_sensitive

        long_value = "x" * 2000
        event = {"event": "test", "output": long_value}
        result = _redact_sensitive(None, "info", event)
        assert len(result["output"]) < 2000
        assert result["output"].endswith("...[truncated]")


# ═══════════════════════════════════════════════════════════════════════
#  Logger Creation & Configuration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLoggerCreation:
    """get_logger and configure_logging work correctly."""

    def setup_method(self):
        from openspace.domain.logging import reset_logging

        reset_logging()

    def teardown_method(self):
        from openspace.domain.logging import reset_logging

        reset_logging()

    def test_get_logger_returns_bound_logger(self):
        from openspace.domain.logging import get_logger

        log = get_logger("test.module")
        assert log is not None
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "debug")

    def test_get_logger_default_name(self):
        from openspace.domain.logging import get_logger

        log = get_logger()
        assert log is not None

    def test_configure_idempotent(self):
        from openspace.domain.logging import configure_logging

        configure_logging(level=logging.DEBUG)
        configure_logging(level=logging.WARNING)  # Should be no-op
        # No exception = pass

    def test_configure_json_output(self):
        from openspace.domain.logging import configure_logging, reset_logging

        reset_logging()
        configure_logging(json_output=True)
        # No exception = pass

    def test_configure_no_colors(self):
        from openspace.domain.logging import configure_logging, reset_logging

        reset_logging()
        configure_logging(colors=False)
        # No exception = pass


# ═══════════════════════════════════════════════════════════════════════
#  Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Structured logging integrates with existing code patterns."""

    def setup_method(self):
        from openspace.domain.logging import clear_context, reset_logging

        reset_logging()
        clear_context()

    def teardown_method(self):
        from openspace.domain.logging import clear_context, reset_logging

        reset_logging()
        clear_context()

    def test_structlog_event_includes_context(self, capsys):
        from openspace.domain.logging import (
            bind_context,
            configure_logging,
            get_logger,
        )

        configure_logging(level=logging.DEBUG, colors=False)
        bind_context(task_id="t-99")
        log = get_logger("test.integration")
        log.info("task_started", workspace="/tmp")

        captured = capsys.readouterr()
        # The output goes to stderr via structlog
        assert "task_started" in captured.err or "task_started" in captured.out

    def test_stdlib_logger_still_works(self):
        """stdlib loggers produce output and go through shared processors."""
        from openspace.domain.logging import configure_logging, bind_context

        configure_logging(level=logging.DEBUG, colors=False)
        bind_context(task_id="stdlib-t1")

        stdlib_logger = logging.getLogger("openspace.test_stdlib_bridge")
        # Capture output from the root handler
        import io
        capture = io.StringIO()
        handler = logging.StreamHandler(capture)
        handler.setLevel(logging.DEBUG)
        # Use the same formatter as configure_logging installs
        root = logging.getLogger()
        if root.handlers:
            handler.setFormatter(root.handlers[0].formatter)
        root.addHandler(handler)
        try:
            stdlib_logger.warning("bridge test", extra={"api_key": "secret99"})
            output = capture.getvalue()
            # Should produce output (not silently swallowed)
            assert len(output) > 0, "stdlib logger produced no output"
        finally:
            root.removeHandler(handler)

    def test_context_vars_processor_injects(self):
        from openspace.domain.logging import _inject_context_vars, bind_context

        bind_context(task_id="t-1", correlation_id="c-1")
        event: dict = {"event": "test"}
        result = _inject_context_vars(None, "info", event)
        assert result["task_id"] == "t-1"
        assert result["correlation_id"] == "c-1"

    def test_context_vars_dont_overwrite_explicit(self):
        from openspace.domain.logging import _inject_context_vars, bind_context

        bind_context(task_id="t-1")
        event: dict = {"event": "test", "task_id": "explicit-id"}
        result = _inject_context_vars(None, "info", event)
        assert result["task_id"] == "explicit-id"  # Explicit wins

    def test_reset_allows_reconfigure(self):
        from openspace.domain.logging import (
            configure_logging,
            reset_logging,
        )

        configure_logging(level=logging.INFO)
        reset_logging()
        configure_logging(level=logging.DEBUG)  # Should work, not no-op
        # No exception = pass
