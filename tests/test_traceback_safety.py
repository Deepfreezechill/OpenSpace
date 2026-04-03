"""Contract tests: zero traceback leakage from MCP tool responses.

Covers Issues #9-#12 (EPIC 0.3a):
  - No MCP response ever contains traceback strings, file paths,
    line numbers, module paths, or "Traceback (most recent call last)".
  - Error responses use the structured format
    {isError, error_code, message, correlation_id}.
  - Full tracebacks ARE logged server-side (logger.error with exc_info).
"""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────

# Patterns that MUST NOT appear in any client-facing MCP response
_FORBIDDEN_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'File ".*", line \d+'),
    re.compile(r"line \d+, in "),
    re.compile(r"openspace[/\\]\w+\.py"),
    re.compile(r"\.py:\d+"),
    re.compile(r"raise \w+"),
]

VALID_ERROR_CODES = {
    "EXECUTION_ERROR",
    "VALIDATION_ERROR",
    "SKILL_NOT_FOUND",
    "PERMISSION_DENIED",
    "INTERNAL_ERROR",
    "TIMEOUT_ERROR",
}


def _assert_no_traceback_leak(response_str: str, tool_name: str) -> None:
    """Assert that *response_str* contains no forbidden patterns."""
    for pat in _FORBIDDEN_PATTERNS:
        assert not pat.search(response_str), (
            f"[{tool_name}] Traceback leak detected — pattern {pat.pattern!r} found in response:\n{response_str[:500]}"
        )


def _assert_structured_error(response_str: str, tool_name: str) -> dict:
    """Parse response as JSON and validate the structured error schema."""
    data = json.loads(response_str)
    assert data.get("isError") is True, f"[{tool_name}] Expected isError=true, got {data}"
    assert data.get("error_code") in VALID_ERROR_CODES, f"[{tool_name}] Invalid error_code: {data.get('error_code')}"
    assert isinstance(data.get("message"), str) and data["message"], f"[{tool_name}] Missing or empty 'message'"
    assert isinstance(data.get("correlation_id"), str) and data["correlation_id"], (
        f"[{tool_name}] Missing or empty 'correlation_id'"
    )
    # Double-check: message itself must be leak-free
    _assert_no_traceback_leak(data["message"], tool_name)
    return data


# ── Unit tests for openspace.errors ──────────────────────────────────


class TestSanitizeError:
    """Test that sanitize_error strips dangerous content."""

    def test_plain_message_preserved(self):
        from openspace.errors import sanitize_error

        exc = ValueError("missing required field")
        assert sanitize_error(exc) == "missing required field"

    def test_traceback_string_stripped(self):
        from openspace.errors import sanitize_error

        exc = RuntimeError('Traceback (most recent call last):\n  File "foo.py", line 42\nKeyError')
        result = sanitize_error(exc)
        assert "Traceback" not in result
        assert "foo.py" not in result

    def test_file_path_stripped(self):
        from openspace.errors import sanitize_error

        exc = OSError("Cannot open C:\\Users\\dev\\project\\secret.py")
        result = sanitize_error(exc)
        assert "C:\\Users" not in result
        assert "secret.py" not in result

    def test_unix_path_stripped(self):
        from openspace.errors import sanitize_error

        exc = OSError("Failed at /home/user/openspace/mcp_server.py")
        result = sanitize_error(exc)
        assert "/home/user" not in result
        assert "mcp_server.py" not in result

    def test_empty_message_returns_generic(self):
        from openspace.errors import sanitize_error

        exc = RuntimeError()
        result = sanitize_error(exc)
        assert result == "An internal error occurred"
        assert "RuntimeError" not in result

    def test_long_message_truncated(self):
        from openspace.errors import sanitize_error

        exc = ValueError("x" * 500)
        result = sanitize_error(exc)
        assert len(result) <= 300

    # ── Regression tests from /collab + /8eyes review ────────────

    def test_windows_path_with_spaces_stripped(self):
        from openspace.errors import sanitize_error

        exc = OSError(r"Cannot open C:\Program Files\OpenSpace\secret.py")
        result = sanitize_error(exc)
        assert "Program Files" not in result
        assert "OpenSpace" not in result
        assert "secret.py" not in result

    def test_unc_path_stripped(self):
        from openspace.errors import sanitize_error

        exc = OSError(r"Failed reading \\server\share\secret.py")
        result = sanitize_error(exc)
        assert r"\\server" not in result
        assert "secret.py" not in result

    def test_dotted_module_name_stripped(self):
        from openspace.errors import sanitize_error

        exc = ImportError("No module named openspace.cloud.auth.TokenResolver")
        result = sanitize_error(exc)
        assert "openspace.cloud.auth" not in result
        assert "TokenResolver" not in result

    def test_standalone_line_number_stripped(self):
        from openspace.errors import sanitize_error

        exc = RuntimeError("failed at line 99 in processing")
        result = sanitize_error(exc)
        assert "line 99" not in result

    def test_never_returns_exception_class_name(self):
        """Regression: fallback must not leak type(exc).__name__."""
        from openspace.errors import sanitize_error

        class InternalSecretError(Exception):
            pass

        exc = InternalSecretError()
        result = sanitize_error(exc)
        assert "InternalSecretError" not in result
        assert result == "An internal error occurred"


class TestSafeErrorResponse:
    """Test the structured JSON builder."""

    def test_schema(self):
        from openspace.errors import EXECUTION_ERROR, safe_error_response

        raw = safe_error_response(EXECUTION_ERROR, "Something went wrong")
        data = json.loads(raw)
        assert data["isError"] is True
        assert data["error_code"] == "EXECUTION_ERROR"
        assert data["message"] == "Something went wrong"
        assert isinstance(data["correlation_id"], str)
        assert len(data["correlation_id"]) == 12

    def test_custom_correlation_id(self):
        from openspace.errors import VALIDATION_ERROR, safe_error_response

        raw = safe_error_response(VALIDATION_ERROR, "bad input", correlation_id="abc123")
        data = json.loads(raw)
        assert data["correlation_id"] == "abc123"


class TestHandleMcpException:
    """Test the one-liner exception handler."""

    def test_logs_and_returns_safe_json(self):
        from openspace.errors import EXECUTION_ERROR, handle_mcp_exception

        with patch("openspace.errors.logger") as mock_logger:
            exc = RuntimeError("boom at /secret/path.py:42")
            result = handle_mcp_exception(exc, tool_name="test_tool", error_code=EXECUTION_ERROR)

            # Logger was called with exc_info=True
            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args
            assert call_kwargs[1].get("exc_info") is True

        # Response is structured and leak-free
        _assert_structured_error(result, "test_tool")
        _assert_no_traceback_leak(result, "test_tool")

    def test_correlation_id_in_log_and_response(self):
        from openspace.errors import INTERNAL_ERROR, handle_mcp_exception

        with patch("openspace.errors.logger") as mock_logger:
            exc = ValueError("oops")
            result = handle_mcp_exception(exc, tool_name="x", error_code=INTERNAL_ERROR)

            data = json.loads(result)
            cid = data["correlation_id"]
            # The correlation ID must appear in the log call args
            log_args = mock_logger.error.call_args[0]
            assert cid in str(log_args), f"Correlation ID {cid} not found in log call: {log_args}"


# ── Integration tests: MCP tool error paths ─────────────────────────


def _make_openspace_mock():
    """Create a mock OpenSpace instance sufficient for mcp_server imports."""
    mock = MagicMock()
    mock.is_initialized.return_value = True
    mock._skill_registry = MagicMock()
    mock._skill_evolver = MagicMock()
    return mock


@pytest.fixture(autouse=True)
def _patch_openspace_init(monkeypatch):
    """Prevent real OpenSpace initialization in every test."""
    import openspace.mcp_server as srv

    mock = _make_openspace_mock()
    monkeypatch.setattr(srv, "_openspace_instance", mock)


# ---- execute_task ----


@pytest.mark.asyncio
async def test_execute_task_error_no_traceback():
    """execute_task: exception → structured error, no traceback leak."""
    import openspace.mcp_server as srv

    with patch.object(
        srv,
        "_get_openspace",
        new_callable=AsyncMock,
        side_effect=RuntimeError("DB connection to /var/lib/pg failed at line 99"),
    ):
        result = await srv.execute_task(task="hello")

    _assert_no_traceback_leak(result, "execute_task")
    data = _assert_structured_error(result, "execute_task")
    assert "var/lib" not in data["message"]


@pytest.mark.asyncio
async def test_execute_task_deep_traceback_not_leaked():
    """execute_task: real traceback chain → nothing leaks."""
    import openspace.mcp_server as srv

    def _blow_up():
        raise KeyError("secret_key")

    async def _explode():
        try:
            _blow_up()
        except KeyError:
            raise RuntimeError("nested failure") from None

    with patch.object(srv, "_get_openspace", new_callable=AsyncMock, side_effect=_explode):
        result = await srv.execute_task(task="hello")

    _assert_no_traceback_leak(result, "execute_task")
    _assert_structured_error(result, "execute_task")


# ---- search_skills ----


@pytest.mark.asyncio
async def test_search_skills_error_no_traceback():
    """search_skills: exception → structured error, no traceback leak."""
    import openspace.mcp_server as srv

    with patch(
        "openspace.mcp_server.hybrid_search_skills",
        create=True,
        side_effect=ImportError("No module named 'openspace.cloud.search'"),
    ):
        # The import happens inside the try block, so we need to make it raise
        with patch.dict("sys.modules", {"openspace.cloud.search": None}):
            result = await srv.search_skills(query="test query")

    _assert_no_traceback_leak(result, "search_skills")
    _assert_structured_error(result, "search_skills")


# ---- fix_skill ----


@pytest.mark.asyncio
async def test_fix_skill_missing_skill_md():
    """fix_skill: missing SKILL.md → SKILL_NOT_FOUND, no path leak."""
    import openspace.mcp_server as srv

    result = await srv.fix_skill(
        skill_dir="/secret/internal/path/my-skill",
        direction="fix the API endpoint",
    )

    _assert_no_traceback_leak(result, "fix_skill")
    data = json.loads(result)
    assert data.get("isError") is True
    assert "SKILL_NOT_FOUND" == data.get("error_code")
    assert "/secret" not in data["message"]
    assert "internal" not in data["message"]


@pytest.mark.asyncio
async def test_fix_skill_exception_no_traceback():
    """fix_skill: runtime exception → structured error."""
    import os

    import openspace.mcp_server as srv

    # Create a temporary directory with SKILL.md so we pass validation
    tmpdir = os.path.join(os.path.dirname(__file__), "_test_skill_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    skill_md = os.path.join(tmpdir, "SKILL.md")
    try:
        with open(skill_md, "w") as f:
            f.write("# Test Skill\n")

        mock_os = _make_openspace_mock()
        mock_os._skill_registry.register_skill_dir.side_effect = RuntimeError(
            "segfault at openspace/skill_engine/registry.py:123"
        )

        with patch.object(srv, "_get_openspace", new_callable=AsyncMock, return_value=mock_os):
            result = await srv.fix_skill(skill_dir=tmpdir, direction="fix it")

        _assert_no_traceback_leak(result, "fix_skill")
        _assert_structured_error(result, "fix_skill")
    finally:
        if os.path.exists(skill_md):
            os.remove(skill_md)
        if os.path.exists(tmpdir):
            os.rmdir(tmpdir)


# ---- upload_skill ----


@pytest.mark.asyncio
async def test_upload_skill_missing_skill_md():
    """upload_skill: missing SKILL.md → SKILL_NOT_FOUND, no path leak."""
    import openspace.mcp_server as srv

    result = await srv.upload_skill(skill_dir="/opt/secret/skills/broken")

    _assert_no_traceback_leak(result, "upload_skill")
    data = json.loads(result)
    assert data.get("isError") is True
    assert data["error_code"] == "SKILL_NOT_FOUND"
    assert "/opt/secret" not in data["message"]


@pytest.mark.asyncio
async def test_upload_skill_exception_no_traceback():
    """upload_skill: cloud auth failure → structured error, no traceback."""
    import os

    import openspace.mcp_server as srv

    tmpdir = os.path.join(os.path.dirname(__file__), "_test_upload_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    skill_md = os.path.join(tmpdir, "SKILL.md")
    try:
        with open(skill_md, "w") as f:
            f.write("# Test Skill\n")

        with patch.object(
            srv,
            "_get_cloud_client",
            side_effect=PermissionError("Invalid API key at openspace/cloud/auth.py:55"),
        ):
            result = await srv.upload_skill(skill_dir=tmpdir)

        _assert_no_traceback_leak(result, "upload_skill")
        _assert_structured_error(result, "upload_skill")
    finally:
        if os.path.exists(skill_md):
            os.remove(skill_md)
        if os.path.exists(tmpdir):
            os.rmdir(tmpdir)


# ---- Server-side logging verification ----


@pytest.mark.asyncio
async def test_server_logs_full_exception():
    """Verify that full exception details are logged server-side."""
    import openspace.mcp_server as srv

    with patch("openspace.errors.logger") as mock_logger:
        with patch.object(
            srv,
            "_get_openspace",
            new_callable=AsyncMock,
            side_effect=RuntimeError("detailed internal error info"),
        ):
            result = await srv.execute_task(task="test")

        # Logger MUST have been called with exc_info=True
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert kwargs.get("exc_info") is True, "Server must log with exc_info=True"

        # The log message should contain the tool name
        log_message = args[0] % args[1:] if len(args) > 1 else str(args)
        assert "execute_task" in str(log_message)


# ---- Parametrized: all four tools produce structured errors ----


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,call",
    [
        ("execute_task", lambda srv: srv.execute_task(task="test")),
        ("search_skills", lambda srv: srv.search_skills(query="test")),
        ("fix_skill", lambda srv: srv.fix_skill(skill_dir="/nonexistent", direction="fix")),
        ("upload_skill", lambda srv: srv.upload_skill(skill_dir="/nonexistent")),
    ],
)
async def test_all_tools_structured_error_on_failure(tool_name, call):
    """Every MCP tool returns structured error JSON on failure — never raw tracebacks."""
    import openspace.mcp_server as srv

    # Sabotage everything to force errors
    with patch.object(
        srv,
        "_get_openspace",
        new_callable=AsyncMock,
        side_effect=Exception(f"Synthetic failure in {tool_name}"),
    ):
        with patch.dict("sys.modules", {"openspace.cloud.search": None}):
            result = await call(srv)

    _assert_no_traceback_leak(result, tool_name)
    data = json.loads(result)
    assert data.get("isError") is True, f"[{tool_name}] Expected isError=true"
    assert data.get("error_code") in VALID_ERROR_CODES, f"[{tool_name}] Bad error_code: {data.get('error_code')}"
