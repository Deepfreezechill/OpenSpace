"""Tests for openspace.mcp.tool_handlers — Epic 4.7 extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Guard: skip all tests if the tool_handlers module can't be imported
try:
    from openspace.mcp.tool_handlers import (
        _format_task_result,
        _is_auto_import_enabled,
        _json_error,
        _json_ok,
        _read_upload_meta,
        _write_upload_meta,
        register_handlers,
    )

    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

pytestmark = pytest.mark.skipif(not _HAS_MODULE, reason="openspace.mcp.tool_handlers not importable")


# ---------------------------------------------------------------------------
# register_handlers
# ---------------------------------------------------------------------------
class TestRegisterHandlers:
    """Verify register_handlers wires all 8 tools to the FastMCP instance."""

    def test_registers_seven_tools(self):
        mock_mcp = MagicMock()
        mock_decorator = MagicMock(side_effect=lambda fn: fn)
        mock_mcp.tool.return_value = mock_decorator

        register_handlers(mock_mcp)

        assert mock_mcp.tool.call_count == 8
        registered_names = {call.args[0].__name__ for call in mock_decorator.call_args_list}
        assert registered_names == {
            "execute_task", "search_skills", "fix_skill", "upload_skill",
            "health_check", "get_metrics", "get_execution_traces",
            "check_slos",
        }

    def test_handlers_preserve_docstrings(self):
        mock_mcp = MagicMock()
        registered_fns = []
        mock_mcp.tool.return_value = lambda fn: (registered_fns.append(fn), fn)[1]

        register_handlers(mock_mcp)

        for fn in registered_fns:
            assert fn.__doc__ is not None, f"{fn.__name__} has no docstring"
            assert len(fn.__doc__) > 20, f"{fn.__name__} docstring too short"

    def test_idempotent_double_register(self):
        """Calling register_handlers twice should not raise."""
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: fn

        register_handlers(mock_mcp)
        register_handlers(mock_mcp)
        assert mock_mcp.tool.call_count == 16  # 8 + 8


# ---------------------------------------------------------------------------
# _format_task_result
# ---------------------------------------------------------------------------
class TestFormatTaskResult:
    """Verify result formatting for MCP transport."""

    def test_minimal_result(self):
        result = _format_task_result({})
        assert result["status"] == "unknown"
        assert result["response"] == ""
        assert result["execution_time"] == 0
        assert result["iterations"] == 0
        assert result["skills_used"] == []
        assert result["tool_call_count"] == 0
        assert result["tool_summary"] == []

    def test_full_result(self):
        result = _format_task_result({
            "status": "success",
            "response": "Done!",
            "execution_time": 3.456789,
            "iterations": 5,
            "skills_used": ["skill_a"],
            "task_id": "t1",
            "tool_executions": [
                {"tool_name": "bash", "status": "ok", "error": None},
                {"tool_name": "python", "status": "error", "error": "x" * 300},
            ],
        })
        assert result["status"] == "success"
        assert result["execution_time"] == 3.46
        assert result["tool_call_count"] == 2
        assert len(result["tool_summary"][1]["error"]) <= 200

    def test_evolved_skills_formatting(self):
        result = _format_task_result({
            "evolved_skills": [
                {"path": "/skills/foo/SKILL.md", "name": "foo", "origin": "derived", "change_summary": "test"},
            ],
        })
        assert result["evolved_skills"][0]["skill_dir"] == str(Path("/skills/foo"))
        assert result["evolved_skills"][0]["upload_ready"] is True
        assert "action_required" in result

    def test_warning_passthrough(self):
        result = _format_task_result({"warning": "timeout approaching"})
        assert result["warning"] == "timeout approaching"

    def test_no_warning_when_absent(self):
        result = _format_task_result({})
        assert "warning" not in result

    def test_tool_summary_caps_at_20(self):
        execs = [{"tool_name": f"t{i}", "status": "ok"} for i in range(30)]
        result = _format_task_result({"tool_executions": execs})
        assert len(result["tool_summary"]) == 20
        assert result["tool_call_count"] == 30


# ---------------------------------------------------------------------------
# _json_ok / _json_error
# ---------------------------------------------------------------------------
class TestJsonHelpers:
    def test_json_ok_roundtrip(self):
        data = {"key": "value", "num": 42}
        raw = _json_ok(data)
        assert json.loads(raw) == data

    def test_json_ok_unicode(self):
        raw = _json_ok({"emoji": "🚀"})
        assert "🚀" in raw  # ensure_ascii=False

    def test_json_error_structured(self):
        raw = _json_error("something broke")
        parsed = json.loads(raw)
        assert "error" in parsed or "message" in parsed or "error_code" in parsed


# ---------------------------------------------------------------------------
# _write_upload_meta / _read_upload_meta
# ---------------------------------------------------------------------------
class TestUploadMeta:
    def test_write_then_read_roundtrip(self, tmp_path):
        info = {
            "origin": "derived",
            "parent_skill_ids": ["s1"],
            "change_summary": "Fixed it",
            "created_by": "bot",
            "tags": ["test"],
        }
        _write_upload_meta(tmp_path, info)
        result = _read_upload_meta(tmp_path)
        assert result["origin"] == "derived"
        assert result["parent_skill_ids"] == ["s1"]
        assert result["tags"] == ["test"]

    def test_read_empty_dir_returns_empty(self, tmp_path):
        """No sidecar, no DB → empty dict."""
        with patch("openspace.mcp.tool_handlers._get_store") as mock_store:
            mock_store.return_value.load_record_by_path.return_value = None
            result = _read_upload_meta(tmp_path)
        assert result == {}

    def test_write_missing_fields_defaults(self, tmp_path):
        _write_upload_meta(tmp_path, {})
        result = _read_upload_meta(tmp_path)
        assert result["origin"] == "imported"
        assert result["parent_skill_ids"] == []

    def test_corrupt_sidecar_falls_through(self, tmp_path):
        """Corrupt JSON → falls to DB tier → empty dict."""
        (tmp_path / ".upload_meta.json").write_text("{bad json", encoding="utf-8")
        with patch("openspace.mcp.tool_handlers._get_store") as mock_store:
            mock_store.return_value.load_record_by_path.return_value = None
            result = _read_upload_meta(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# _is_auto_import_enabled
# ---------------------------------------------------------------------------
class TestAutoImportEnabled:
    def test_returns_false_when_no_instance(self):
        with patch("openspace.mcp.tool_handlers._openspace_instance", None):
            assert _is_auto_import_enabled() is False

    def test_returns_false_when_not_initialized(self):
        mock_os = MagicMock()
        mock_os.is_initialized.return_value = False
        with patch("openspace.mcp.tool_handlers._openspace_instance", mock_os):
            assert _is_auto_import_enabled() is False

    def test_returns_true_when_enabled(self):
        mock_os = MagicMock()
        mock_os.is_initialized.return_value = True
        mock_os._grounding_config.skills.auto_import_enabled = True
        with patch("openspace.mcp.tool_handlers._openspace_instance", mock_os):
            assert _is_auto_import_enabled() is True

    def test_returns_false_when_no_skills_config(self):
        mock_os = MagicMock()
        mock_os.is_initialized.return_value = True
        mock_os._grounding_config = None
        with patch("openspace.mcp.tool_handlers._openspace_instance", mock_os):
            assert _is_auto_import_enabled() is False
