"""Tests for ExecutionEngine — extracted from OpenSpace.tool_layer.

Verifies:
  - Task execution lifecycle (init check, busy-wait, dispatch)
  - Skill-first → fallback two-phase orchestration
  - Post-execution analysis and quality evolution
  - OpenSpace backward compatibility delegation
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# tool_layer imports litellm which may not be available
try:
    from openspace.tool_layer import OpenSpace, OpenSpaceConfig

    _HAS_TOOL_LAYER = True
except (ImportError, ModuleNotFoundError):
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(
    not _HAS_TOOL_LAYER,
    reason="openspace.tool_layer requires litellm (not installed or broken)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return OpenSpaceConfig(
        grounding_max_iterations=10,
        llm_kwargs={},
    )


@pytest.fixture
def mock_grounding_agent():
    agent = MagicMock()
    agent.backend_scope = ["shell", "mcp"]
    agent.clear_skill_context = MagicMock()
    agent.set_skill_context = MagicMock()
    agent._active_skill_ids = []
    agent._last_tools = []
    agent.process = AsyncMock(return_value={
        "status": "success",
        "iterations": 3,
        "tool_executions": [],
        "response": "done",
    })
    return agent


@pytest.fixture
def mock_grounding_client():
    client = MagicMock()
    client.quality_manager = None
    return client


@pytest.fixture
def mock_recording_manager():
    rm = MagicMock()
    rm.recording_status = False
    rm.trajectory_dir = "/tmp/traj"
    rm.task_id = ""
    rm.start = AsyncMock()
    rm.stop = AsyncMock()
    rm.add_metadata = AsyncMock()
    rm.save_execution_outcome = AsyncMock()
    return rm


@pytest.fixture
def engine(mock_config, mock_grounding_agent, mock_grounding_client):
    from openspace.execution_engine import ExecutionEngine
    return ExecutionEngine(
        config=mock_config,
        grounding_agent=mock_grounding_agent,
        grounding_client=mock_grounding_client,
    )


# ---------------------------------------------------------------------------
# Constructor & Properties
# ---------------------------------------------------------------------------

class TestExecutionEngineInit:

    def test_initial_state(self, engine):
        """Engine starts not running, with zero execution count."""
        assert engine.is_running is False
        assert engine.execution_count == 0
        assert engine.last_evolved_skills == []

    def test_stores_dependencies(self, mock_config, mock_grounding_agent, mock_grounding_client):
        from openspace.execution_engine import ExecutionEngine
        e = ExecutionEngine(
            config=mock_config,
            grounding_agent=mock_grounding_agent,
            grounding_client=mock_grounding_client,
        )
        assert e._config is mock_config
        assert e._grounding_agent is mock_grounding_agent

    def test_optional_deps_default_none(self, mock_config, mock_grounding_agent, mock_grounding_client):
        from openspace.execution_engine import ExecutionEngine
        e = ExecutionEngine(
            config=mock_config,
            grounding_agent=mock_grounding_agent,
            grounding_client=mock_grounding_client,
        )
        assert e._tool_registry is None
        assert e._skill_registry is None
        assert e._recording_manager is None
        assert e._execution_analyzer is None
        assert e._skill_evolver is None


# ---------------------------------------------------------------------------
# execute() — lifecycle
# ---------------------------------------------------------------------------

class TestExecuteLifecycle:

    @pytest.mark.asyncio
    async def test_raises_without_agent(self, mock_config, mock_grounding_client):
        """execute() raises if grounding_agent is None."""
        from openspace.execution_engine import ExecutionEngine
        e = ExecutionEngine(
            config=mock_config,
            grounding_agent=None,
            grounding_client=mock_grounding_client,
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            await e.execute("test task")

    @pytest.mark.asyncio
    async def test_generates_task_id(self, engine, mock_grounding_agent):
        """Auto-generates task_id when not provided."""
        result = await engine.execute("test task")
        assert "task_id" in result
        assert result["task_id"].startswith("task_")

    @pytest.mark.asyncio
    async def test_uses_provided_task_id(self, engine, mock_grounding_agent):
        """Uses caller-supplied task_id."""
        result = await engine.execute("test task", task_id="custom-123")
        assert result["task_id"] == "custom-123"

    @pytest.mark.asyncio
    async def test_sets_running_flag(self, engine, mock_grounding_agent):
        """is_running is True during execution."""
        was_running = False

        async def capture_running(ctx):
            nonlocal was_running
            was_running = engine.is_running
            return {"status": "success", "iterations": 1, "tool_executions": []}

        mock_grounding_agent.process = capture_running
        await engine.execute("test task")
        assert was_running is True
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_returns_execution_time(self, engine):
        """Result includes execution_time."""
        result = await engine.execute("test task")
        assert "execution_time" in result
        assert result["execution_time"] >= 0

    @pytest.mark.asyncio
    async def test_increments_execution_count(self, engine):
        """Execution count increments after each call."""
        assert engine.execution_count == 0
        await engine.execute("task 1")
        # count increments in _maybe_evolve_quality, called in finally
        assert engine.execution_count >= 1


# ---------------------------------------------------------------------------
# execute() — two-phase orchestration
# ---------------------------------------------------------------------------

class TestTwoPhaseExecution:

    @pytest.mark.asyncio
    async def test_no_skills_direct_execution(self, engine, mock_grounding_agent):
        """Without skills, goes directly to tool-only execution."""
        result = await engine.execute("test task")
        mock_grounding_agent.process.assert_called_once()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_skill_first_success_skips_fallback(self, engine, mock_grounding_agent):
        """When skill-guided phase succeeds, no fallback is triggered."""
        from openspace.execution_engine import ExecutionEngine
        from openspace.tool_registry import ToolRegistry

        mock_tr = MagicMock(spec=ToolRegistry)
        mock_tr.select_and_inject = AsyncMock(return_value=True)

        mock_registry = MagicMock()
        engine._tool_registry = mock_tr
        engine._skill_registry = mock_registry

        mock_grounding_agent._active_skill_ids = ["my-skill"]
        mock_grounding_agent.process = AsyncMock(return_value={
            "status": "success", "iterations": 2, "tool_executions": [],
        })

        result = await engine.execute("test task")
        assert result["status"] == "success"
        assert result["skills_used"] == ["my-skill"]
        # Only one process call (skill phase succeeded)
        assert mock_grounding_agent.process.call_count == 1

    @pytest.mark.asyncio
    async def test_skill_failure_triggers_fallback(self, engine, mock_grounding_agent):
        """When skill phase fails, fallback to tool-only execution."""
        from openspace.tool_registry import ToolRegistry

        mock_tr = MagicMock(spec=ToolRegistry)
        mock_tr.select_and_inject = AsyncMock(return_value=True)

        engine._tool_registry = mock_tr
        engine._skill_registry = MagicMock()

        mock_grounding_agent._active_skill_ids = ["failing-skill"]

        # First call: skill phase fails. Second call: fallback succeeds.
        mock_grounding_agent.process = AsyncMock(side_effect=[
            {"status": "error", "iterations": 5, "tool_executions": []},
            {"status": "success", "iterations": 3, "tool_executions": [], "response": "fallback ok"},
        ])

        result = await engine.execute("test task")
        assert mock_grounding_agent.process.call_count == 2
        mock_grounding_agent.clear_skill_context.assert_called_once()


# ---------------------------------------------------------------------------
# execute() — error handling
# ---------------------------------------------------------------------------

class TestExecuteErrorHandling:

    @pytest.mark.asyncio
    async def test_agent_exception_caught(self, engine, mock_grounding_agent):
        """Exceptions from grounding agent are caught and returned as error result."""
        mock_grounding_agent.process = AsyncMock(side_effect=RuntimeError("agent boom"))
        result = await engine.execute("test task")
        assert result["status"] == "error"
        assert "agent boom" in result["error"]
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_recording_start_stop(self, engine, mock_grounding_agent, mock_recording_manager):
        """Recording is started and stopped around execution."""
        engine._recording_manager = mock_recording_manager
        await engine.execute("test task")
        mock_recording_manager.start.assert_called_once()
        mock_recording_manager.stop.assert_called_once()


# ---------------------------------------------------------------------------
# _maybe_analyze_execution
# ---------------------------------------------------------------------------

class TestMaybeAnalyzeExecution:

    @pytest.mark.asyncio
    async def test_no_analyzer_is_noop(self, engine):
        """No-op when execution_analyzer is None."""
        # Should not raise
        await engine._maybe_analyze_execution("t1", "/dir", {"status": "success"})

    @pytest.mark.asyncio
    async def test_no_recording_dir_is_noop(self, engine):
        """No-op when recording_dir is None."""
        engine._execution_analyzer = MagicMock()
        await engine._maybe_analyze_execution("t1", None, {"status": "success"})
        engine._execution_analyzer.analyze_execution.assert_not_called()

    @pytest.mark.asyncio
    async def test_analysis_exception_swallowed(self, engine):
        """Analyzer exceptions are caught, not propagated."""
        analyzer = MagicMock()
        analyzer.analyze_execution = AsyncMock(side_effect=RuntimeError("analysis failed"))
        engine._execution_analyzer = analyzer
        # Should not raise
        await engine._maybe_analyze_execution("t1", "/dir", {"status": "success"})


# ---------------------------------------------------------------------------
# _maybe_evolve_quality
# ---------------------------------------------------------------------------

class TestMaybeEvolveQuality:

    @pytest.mark.asyncio
    async def test_increments_count(self, engine):
        """Execution count increments each call."""
        assert engine._execution_count == 0
        await engine._maybe_evolve_quality()
        assert engine._execution_count == 1
        await engine._maybe_evolve_quality()
        assert engine._execution_count == 2

    @pytest.mark.asyncio
    async def test_no_quality_manager_is_noop(self, engine):
        """No-op when quality_manager is None."""
        await engine._maybe_evolve_quality()  # Should not raise


# ---------------------------------------------------------------------------
# OpenSpace backward compatibility
# ---------------------------------------------------------------------------

class TestOpenSpaceDelegation:

    def test_openspace_has_execution_engine_attr(self):
        """OpenSpace instance has _execution_engine attribute."""
        os_instance = OpenSpace()
        assert hasattr(os_instance, "_execution_engine")
        assert os_instance._execution_engine is None
