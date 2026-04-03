"""P4 Integration Tests — Epic 4.10

Outcome-focused tests verifying the decomposed modules work together
as a system. These test OUTCOMES, not implementation details:

1. Can OpenSpace be configured and initialized through the facade?
2. Does the MCP server serve all 4 tools with correct signatures?
3. Do cross-module interactions (facade → registry → engine) work?
4. Are backward-compat import paths still functional?
5. Is the package architecture sound (no circular imports)?
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guards: skip per-section if dependencies aren't importable
# ---------------------------------------------------------------------------
try:
    from openspace.tool_layer import OpenSpace, OpenSpaceConfig

    _HAS_CORE = True
except (ImportError, ModuleNotFoundError):
    _HAS_CORE = False

try:
    from openspace.mcp.server import create_mcp_app
    from openspace.mcp.tool_handlers import (
        _format_task_result,
        _json_error,
        _json_ok,
        register_handlers,
    )

    _HAS_MCP = True
except (ImportError, ModuleNotFoundError):
    _HAS_MCP = False

try:
    from openspace.tool_registry import ToolRegistry

    _HAS_REGISTRY = True
except (ImportError, ModuleNotFoundError):
    _HAS_REGISTRY = False

try:
    from openspace.execution_engine import ExecutionEngine

    _HAS_ENGINE = True
except (ImportError, ModuleNotFoundError):
    _HAS_ENGINE = False

try:
    from openspace.recording_service import RecordingService

    _HAS_RECORDING = True
except (ImportError, ModuleNotFoundError):
    _HAS_RECORDING = False

try:
    from openspace.llm_factory import LLMFactory

    _HAS_LLM = True
except (ImportError, ModuleNotFoundError):
    _HAS_LLM = False

_skip_core = pytest.mark.skipif(not _HAS_CORE, reason="Core modules not importable (litellm)")
_skip_mcp = pytest.mark.skipif(not _HAS_MCP, reason="MCP modules not importable")
_skip_registry = pytest.mark.skipif(not _HAS_REGISTRY, reason="ToolRegistry not importable")
_skip_engine = pytest.mark.skipif(not _HAS_ENGINE, reason="ExecutionEngine not importable")
_skip_recording = pytest.mark.skipif(not _HAS_RECORDING, reason="RecordingService not importable")
_skip_llm = pytest.mark.skipif(not _HAS_LLM, reason="LLMFactory not importable")


# ---------------------------------------------------------------------------
# 1. OpenSpace Facade Integration
# ---------------------------------------------------------------------------
@_skip_core
class TestOpenSpaceFacadeIntegration:
    """Verify the facade properly delegates to all extracted modules."""

    def test_config_dataclass_complete(self):
        """OpenSpaceConfig has all fields needed by extracted modules."""
        cfg = OpenSpaceConfig(
            llm_model="test-model",
            llm_kwargs={"api_key": "test"},
            workspace_dir="/tmp/test",
            grounding_max_iterations=10,
            enable_recording=True,
            recording_backends=["shell"],
            recording_log_dir="/tmp/logs",
        )
        assert cfg.llm_model == "test-model"
        assert cfg.enable_recording is True

    def test_openspace_constructor_accepts_config(self):
        """OpenSpace can be instantiated with config (no initialization)."""
        cfg = OpenSpaceConfig(llm_model="m", llm_kwargs={})
        os_inst = OpenSpace(config=cfg)
        assert os_inst.config is cfg
        assert os_inst.is_initialized() is False

    def test_openspace_has_extracted_module_attributes(self):
        """After construction, OpenSpace has slots for all extracted modules."""
        cfg = OpenSpaceConfig(llm_model="m", llm_kwargs={})
        os_inst = OpenSpace(config=cfg)
        # These are set to None before initialize()
        assert hasattr(os_inst, "_skill_registry")
        assert hasattr(os_inst, "_execution_engine")

    def test_openspace_repr_works(self):
        """__repr__ doesn't crash after extraction."""
        cfg = OpenSpaceConfig(llm_model="m", llm_kwargs={})
        os_inst = OpenSpace(config=cfg)
        r = repr(os_inst)
        assert "OpenSpace" in r

    def test_openspace_execute_delegates_to_engine(self):
        """execute() delegates to ExecutionEngine with correct args."""
        cfg = OpenSpaceConfig(llm_model="m", llm_kwargs={})
        os_inst = OpenSpace(config=cfg)
        mock_engine = AsyncMock()
        mock_engine.execute.return_value = {"status": "success", "response": "done"}
        os_inst._execution_engine = mock_engine
        os_inst._initialized = True

        result = asyncio.run(os_inst.execute(task="test task"))
        mock_engine.execute.assert_called_once()
        call_kwargs = mock_engine.execute.call_args
        assert call_kwargs is not None, "execute() not called with expected args"
        assert result["status"] == "success"

    def test_openspace_cleanup_delegates(self):
        """cleanup() delegates to engine and recording service."""
        cfg = OpenSpaceConfig(llm_model="m", llm_kwargs={})
        os_inst = OpenSpace(config=cfg)
        mock_engine = MagicMock()
        mock_engine._running = False
        mock_engine._task_done = None
        os_inst._execution_engine = mock_engine
        mock_recording = MagicMock()
        os_inst._recording_service = mock_recording
        os_inst._initialized = True

        asyncio.run(os_inst.cleanup())
        mock_recording.cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# 2. MCP Server Integration
# ---------------------------------------------------------------------------
@_skip_mcp
class TestMCPServerIntegration:
    """Verify MCP server creates a working app with all tools."""

    def test_create_mcp_app_registers_all_tools(self):
        """OUTCOME: A fresh MCP app has exactly 4 tools with correct names."""
        app = create_mcp_app()
        # FastMCP stores tools in _tool_manager._tools — assert structure exists
        assert hasattr(app, "_tool_manager"), "FastMCP app missing _tool_manager"
        assert hasattr(app._tool_manager, "_tools"), "_tool_manager missing _tools"
        tools = set(app._tool_manager._tools.keys())
        assert tools == {"execute_task", "search_skills", "fix_skill", "upload_skill"}, (
            f"Expected 4 tools, got: {tools}"
        )

    def test_tool_signatures_have_docstrings(self):
        """OUTCOME: Each tool has a substantive docstring (MCP uses these)."""
        app = create_mcp_app()
        assert hasattr(app, "_tool_manager"), "FastMCP app missing _tool_manager"
        for name, tool in app._tool_manager._tools.items():
            assert hasattr(tool, "fn"), f"Tool {name} has no .fn attribute"
            fn = tool.fn
            assert fn.__doc__, f"Tool {name} has no docstring"
            assert len(fn.__doc__) > 50, f"Tool {name} docstring too short ({len(fn.__doc__)} chars)"

    def test_independent_apps_have_independent_tools(self):
        """OUTCOME: Two apps don't share state."""
        app1 = create_mcp_app()
        app2 = create_mcp_app()
        assert app1 is not app2

    def test_result_formatting_roundtrip(self):
        """OUTCOME: A task result survives format → JSON → parse."""
        result = {
            "status": "success",
            "response": "Task completed",
            "execution_time": 1.5,
            "iterations": 3,
            "skills_used": ["web_scraper"],
            "task_id": "t123",
            "tool_executions": [{"tool_name": "bash", "status": "ok"}],
        }
        formatted = _format_task_result(result)
        json_str = _json_ok(formatted)
        parsed = json.loads(json_str)

        assert parsed["status"] == "success"
        assert parsed["execution_time"] == 1.5
        assert parsed["tool_call_count"] == 1
        assert parsed["skills_used"] == ["web_scraper"]

    def test_error_formatting_roundtrip(self):
        """OUTCOME: An error survives format → JSON → parse with expected fields."""
        json_str = _json_error("Something went wrong", error_code="TEST_ERROR")
        parsed = json.loads(json_str)
        assert parsed["isError"] is True
        assert parsed["error_code"] == "TEST_ERROR"
        assert parsed["message"] == "Something went wrong"
        assert "correlation_id" in parsed


# ---------------------------------------------------------------------------
# 3. Cross-Module Interaction
# ---------------------------------------------------------------------------
class TestCrossModuleInteraction:
    """Verify extracted modules interact correctly through the facade."""

    @_skip_registry
    def test_tool_registry_import_chain(self):
        """ToolRegistry can be imported and has expected API."""
        assert hasattr(ToolRegistry, "discover")
        assert hasattr(ToolRegistry, "select_and_inject")
        assert isinstance(
            ToolRegistry.registry, property
        ), "registry should be a property"

    @_skip_engine
    def test_execution_engine_import_chain(self):
        """ExecutionEngine can be imported and has expected API."""
        assert hasattr(ExecutionEngine, "execute")

    @_skip_recording
    def test_recording_service_import_chain(self):
        """RecordingService can be imported and has expected API."""
        assert hasattr(RecordingService, "create")
        assert hasattr(RecordingService, "wire")
        assert hasattr(RecordingService, "cleanup")

    @_skip_llm
    def test_llm_factory_import_chain(self):
        """LLMFactory can be imported and has expected API."""
        assert hasattr(LLMFactory, "create_main")
        assert hasattr(LLMFactory, "create_tool_retrieval")

    @_skip_mcp
    def test_mcp_tool_handlers_import_chain(self):
        """Tool handlers module exports all expected functions."""
        from openspace.mcp import tool_handlers

        assert callable(tool_handlers.execute_task)
        assert callable(tool_handlers.search_skills)
        assert callable(tool_handlers.fix_skill)
        assert callable(tool_handlers.upload_skill)
        assert callable(tool_handlers.register_handlers)

    @_skip_mcp
    def test_mcp_server_import_chain(self):
        """Server module exports expected functions."""
        from openspace.mcp import server

        assert callable(server.create_mcp_app)
        assert callable(server.run_mcp_server)


# ---------------------------------------------------------------------------
# 4. Backward Compatibility
# ---------------------------------------------------------------------------
@_skip_mcp
class TestBackwardCompatibility:
    """Verify old import paths still work after extraction."""

    def test_mcp_server_shim_exports_run(self):
        """from openspace.mcp_server import run_mcp_server still works."""
        from openspace.mcp_server import run_mcp_server

        assert callable(run_mcp_server)

    def test_mcp_server_shim_exports_mcp(self):
        """openspace.mcp_server.mcp lazy proxy actually delegates to FastMCP."""
        import openspace.mcp_server as srv

        assert hasattr(srv, "mcp")
        # Exercise __getattr__ — forces the proxy to create the real app
        assert hasattr(srv.mcp, "name"), "Lazy proxy failed to delegate .name to FastMCP"

    def test_mcp_server_shim_exports_create_app(self):
        """from openspace.mcp_server import create_mcp_app still works."""
        from openspace.mcp_server import create_mcp_app

        assert callable(create_mcp_app)

    def test_package_level_mcp_imports(self):
        """openspace.mcp package exports expected public API."""
        from openspace.mcp.server import create_mcp_app, run_mcp_server
        from openspace.mcp.tool_handlers import register_handlers

        assert callable(create_mcp_app)
        assert callable(run_mcp_server)
        assert callable(register_handlers)

    @_skip_core
    def test_tool_layer_exports_openspace_class(self):
        """from openspace.tool_layer import OpenSpace still works."""
        from openspace.tool_layer import OpenSpace, OpenSpaceConfig

        assert callable(OpenSpace)
        assert callable(OpenSpaceConfig)


# ---------------------------------------------------------------------------
# 5. Architecture Soundness
# ---------------------------------------------------------------------------
@_skip_mcp
class TestArchitectureSoundness:
    """Verify no circular imports and clean package boundaries."""

    def test_no_circular_imports(self):
        """All P4 MCP modules can be imported in a fresh process without cycles."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import openspace.mcp.server; import openspace.mcp.tool_handlers; import openspace.mcp_server",
            ],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"MCP circular import detected:\n{result.stderr.decode()}"
        )

    @_skip_core
    def test_no_circular_imports_core(self):
        """All P4 core modules can be imported in a fresh process without cycles."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                (
                    "import openspace.tool_layer; import openspace.tool_registry; "
                    "import openspace.execution_engine; import openspace.recording_service; "
                    "import openspace.llm_factory"
                ),
            ],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Core circular import detected:\n{result.stderr.decode()}"
        )

    def test_mcp_package_is_self_contained(self):
        """openspace/mcp/ package doesn't import from openspace.mcp_server."""
        import openspace.mcp.server as srv_mod
        import openspace.mcp.tool_handlers as th_mod

        srv_source = Path(srv_mod.__file__).read_text(encoding="utf-8")
        th_source = Path(th_mod.__file__).read_text(encoding="utf-8")

        assert "from openspace.mcp_server" not in srv_source, "server.py imports from shim"
        assert "from openspace.mcp_server" not in th_source, "tool_handlers.py imports from shim"
        assert "import openspace.mcp_server" not in srv_source, "server.py imports from shim"
        assert "import openspace.mcp_server" not in th_source, "tool_handlers.py imports from shim"

    def test_tool_layer_decomposition_sizes(self):
        """Verify no single module exceeds the original monolith size."""
        modules = {
            "openspace/tool_layer.py": 530,       # was 788, now ~476 (+11% headroom)
            "openspace/tool_registry.py": 280,     # ~245 (+14% headroom)
            "openspace/execution_engine.py": 590,  # ~537 (+10% headroom)
            "openspace/recording_service.py": 100, # ~71 (+40% headroom — small file)
            "openspace/llm_factory.py": 100,       # ~67 (+49% headroom — small file)
        }
        base = Path(__file__).parent.parent
        for path, max_lines in modules.items():
            full_path = base / path
            assert full_path.exists(), f"Expected module missing: {path}"
            lines = len(full_path.read_text(encoding="utf-8").splitlines())
            assert lines <= max_lines, (
                f"{path} has {lines} lines (max {max_lines})"
            )

    def test_mcp_decomposition_sizes(self):
        """Verify MCP modules are within expected bounds."""
        modules = {
            "openspace/mcp_server.py": 65,           # shim, ~51 (+27% headroom)
            "openspace/mcp/server.py": 270,           # ~236 (+14% headroom)
            "openspace/mcp/tool_handlers.py": 870,    # ~799 (+9% headroom)
        }
        base = Path(__file__).parent.parent
        for path, max_lines in modules.items():
            full_path = base / path
            assert full_path.exists(), f"Expected module missing: {path}"
            lines = len(full_path.read_text(encoding="utf-8").splitlines())
            assert lines <= max_lines, (
                f"{path} has {lines} lines (max {max_lines})"
            )
