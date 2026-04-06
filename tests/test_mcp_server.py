"""Tests for openspace.mcp.server — Epic 4.9 extraction."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from openspace.mcp.server import (
        _MCPSafeStdout,
        _cleanup_file_handles,
        create_mcp_app,
    )

    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

pytestmark = pytest.mark.skipif(not _HAS_MODULE, reason="openspace.mcp.server not importable")


# ---------------------------------------------------------------------------
# _MCPSafeStdout
# ---------------------------------------------------------------------------
class TestMCPSafeStdout:
    """Verify stdout wrapper routes text→stderr, binary→real stdout."""

    def test_write_goes_to_stderr(self):
        real = MagicMock()
        err = MagicMock()
        wrapper = _MCPSafeStdout(real, err)
        wrapper.write("hello")
        err.write.assert_called_once_with("hello")
        real.write.assert_not_called()

    def test_buffer_returns_real_stdout_buffer(self):
        real = MagicMock()
        err = MagicMock()
        wrapper = _MCPSafeStdout(real, err)
        assert wrapper.buffer is real.buffer

    def test_fileno_returns_real_fileno(self):
        real = MagicMock()
        real.fileno.return_value = 1
        err = MagicMock()
        wrapper = _MCPSafeStdout(real, err)
        assert wrapper.fileno() == 1

    def test_flush_flushes_both(self):
        real = MagicMock()
        err = MagicMock()
        wrapper = _MCPSafeStdout(real, err)
        wrapper.flush()
        err.flush.assert_called_once()
        real.flush.assert_called_once()

    def test_writelines_goes_to_stderr(self):
        real = MagicMock()
        err = MagicMock()
        wrapper = _MCPSafeStdout(real, err)
        wrapper.writelines(["a", "b"])
        err.writelines.assert_called_once_with(["a", "b"])

    def test_not_readable(self):
        wrapper = _MCPSafeStdout(MagicMock(), MagicMock())
        assert wrapper.readable() is False

    def test_is_writable(self):
        wrapper = _MCPSafeStdout(MagicMock(), MagicMock())
        assert wrapper.writable() is True

    def test_not_seekable(self):
        wrapper = _MCPSafeStdout(MagicMock(), MagicMock())
        assert wrapper.seekable() is False

    def test_encoding_from_stderr(self):
        err = MagicMock()
        err.encoding = "utf-8"
        wrapper = _MCPSafeStdout(MagicMock(), err)
        assert wrapper.encoding == "utf-8"

    def test_getattr_delegates_to_stderr(self):
        err = MagicMock()
        err.some_method.return_value = 42
        wrapper = _MCPSafeStdout(MagicMock(), err)
        assert wrapper.some_method() == 42


# ---------------------------------------------------------------------------
# create_mcp_app
# ---------------------------------------------------------------------------
class TestCreateMcpApp:
    """Verify factory returns a FastMCP instance with handlers wired."""

    def test_returns_fastmcp_instance(self):
        app = create_mcp_app()
        # FastMCP should have a name attribute
        assert hasattr(app, "name") or hasattr(app, "_tool_manager")

    def test_tools_registered(self):
        app = create_mcp_app()
        # Verify all 4 MCP tools are actually wired
        if hasattr(app, "_tool_manager") and hasattr(app._tool_manager, "_tools"):
            tool_names = set(app._tool_manager._tools.keys())
            assert tool_names == {
                "execute_task", "search_skills", "fix_skill", "upload_skill",
                "health_check", "get_metrics", "get_execution_traces",
            }, (
                f"Expected 7 tools, got: {tool_names}"
            )
        else:
            # Fallback: at least verify create_mcp_app didn't silently fail
            assert app is not None

    def test_idempotent_calls(self):
        """Multiple calls should each return independent instances."""
        app1 = create_mcp_app()
        app2 = create_mcp_app()
        assert app1 is not app2


# ---------------------------------------------------------------------------
# _cleanup_file_handles
# ---------------------------------------------------------------------------
class TestCleanupFileHandles:
    """Verify cleanup properly closes handles."""

    def test_cleanup_closes_stderr_file(self):
        import openspace.mcp.server as srv
        mock_file = MagicMock()
        original = srv._stderr_file
        try:
            srv._stderr_file = mock_file
            _cleanup_file_handles()
            mock_file.close.assert_called_once()
            assert srv._stderr_file is None
        finally:
            srv._stderr_file = original

    def test_cleanup_noop_when_none(self):
        """Should not raise when handles are already None."""
        import openspace.mcp.server as srv
        original_stderr = srv._stderr_file
        original_handler = srv._log_file_handler
        try:
            srv._stderr_file = None
            srv._log_file_handler = None
            _cleanup_file_handles()  # should not raise
        finally:
            srv._stderr_file = original_stderr
            srv._log_file_handler = original_handler
