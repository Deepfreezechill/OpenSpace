"""Tests for ExecutionEngine — extracted from Scion.tool_layer.

Verifies:
  - Task execution lifecycle (init check, busy-wait, dispatch)
  - Skill-first → fallback two-phase orchestration
  - Post-execution analysis and quality evolution
  - Scion backward compatibility delegation
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# tool_layer imports litellm which may not be available
try:
    from scion.tool_layer import Scion, ScionConfig

    _HAS_TOOL_LAYER = True
except (ImportError, ModuleNotFoundError):
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(
    not _HAS_TOOL_LAYER,
    reason="scion.tool_layer requires litellm (not installed or broken)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return ScionConfig(
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

    async def _start():
        rm.recording_status = True

    rm.start = AsyncMock(side_effect=_start)
    rm.stop = AsyncMock()
    rm.add_metadata = AsyncMock()
    rm.save_execution_outcome = AsyncMock()
    return rm


@pytest.fixture
def engine(mock_config, mock_grounding_agent, mock_grounding_client):
    from scion.execution_engine import ExecutionEngine
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
        from scion.execution_engine import ExecutionEngine
        e = ExecutionEngine(
            config=mock_config,
            grounding_agent=mock_grounding_agent,
            grounding_client=mock_grounding_client,
        )
        assert e._config is mock_config
        assert e._grounding_agent is mock_grounding_agent

    def test_optional_deps_default_none(self, mock_config, mock_grounding_agent, mock_grounding_client):
        from scion.execution_engine import ExecutionEngine
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
        from scion.execution_engine import ExecutionEngine
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
        from scion.execution_engine import ExecutionEngine
        from scion.tool_registry import ToolRegistry

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
        from scion.tool_registry import ToolRegistry

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
        # Backward compat: skills_used reflects attempted skills even on fallback
        assert result["skills_used"] == ["failing-skill"]


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

    @pytest.mark.asyncio
    async def test_metric_check_on_every_5(self, engine):
        """Trigger 3 fires every 5 executions."""
        evolver = MagicMock()
        evolver.schedule_background = MagicMock()
        evolver.process_metric_check = MagicMock(return_value=AsyncMock())
        evolver.set_available_tools = MagicMock()
        engine._skill_evolver = evolver
        engine._execution_count = 4  # next call → 5

        await engine._maybe_evolve_quality()
        evolver.schedule_background.assert_called_once()

    @pytest.mark.asyncio
    async def test_quality_evolution_trigger(self, engine, mock_grounding_client):
        """Trigger 2 fires when quality manager detects problems."""
        qm = MagicMock()
        qm.should_evolve = MagicMock(return_value=True)
        qm.get_problematic_tools = MagicMock(return_value=["tool_a"])
        mock_grounding_client.quality_manager = qm
        mock_grounding_client.evolve_quality = AsyncMock(return_value={"recommendations": ["r1"]})

        evolver = MagicMock()
        evolver.set_available_tools = MagicMock()
        evolver.schedule_background = MagicMock()
        evolver.process_tool_degradation = MagicMock(return_value=AsyncMock())
        engine._skill_evolver = evolver

        await engine._maybe_evolve_quality()
        mock_grounding_client.evolve_quality.assert_awaited_once()
        evolver.schedule_background.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_workspace
# ---------------------------------------------------------------------------

class TestResolveWorkspace:

    def test_explicit_workspace_dir(self, engine):
        """Explicit workspace_dir argument wins."""
        ctx = {}
        engine._resolve_workspace(ctx, "/explicit/dir", "t1")
        assert ctx["workspace_dir"] == "/explicit/dir"

    def test_config_workspace_dir(self, engine, mock_config):
        """Falls back to config.workspace_dir."""
        mock_config.workspace_dir = "/config/ws"
        ctx = {}
        engine._resolve_workspace(ctx, None, "t1")
        assert ctx["workspace_dir"] == "/config/ws"

    def test_recording_manager_trajectory_dir(self, engine, mock_recording_manager, mock_config):
        """Falls back to recording_manager.trajectory_dir."""
        mock_config.workspace_dir = None
        engine._recording_manager = mock_recording_manager
        mock_recording_manager.trajectory_dir = "/traj/dir"
        ctx = {}
        engine._resolve_workspace(ctx, None, "t1")
        assert ctx["workspace_dir"] == "/traj/dir"

    def test_tempdir_fallback(self, engine, mock_config):
        """Creates temp directory when nothing else available."""
        mock_config.workspace_dir = None
        ctx = {}
        engine._resolve_workspace(ctx, None, "task_abc")
        assert "scion_workspace" in ctx["workspace_dir"]
        assert "task_abc" in ctx["workspace_dir"]


# ---------------------------------------------------------------------------
# _cleanup_workspace
# ---------------------------------------------------------------------------

class TestCleanupWorkspace:

    def test_removes_new_files_preserves_old(self, tmp_path):
        """Removes files not in pre_skill_files set, preserves originals."""
        from scion.execution_engine import ExecutionEngine

        (tmp_path / "existing.txt").write_text("keep me")
        (tmp_path / "new_file.txt").write_text("remove me")
        new_dir = tmp_path / "new_dir"
        new_dir.mkdir()

        ExecutionEngine._cleanup_workspace(str(tmp_path), {"existing.txt"})
        assert (tmp_path / "existing.txt").exists()
        assert not (tmp_path / "new_file.txt").exists()
        assert not new_dir.exists()

    def test_empty_path_is_noop(self):
        """Empty workspace path does nothing."""
        from scion.execution_engine import ExecutionEngine
        ExecutionEngine._cleanup_workspace("", set())  # no raise


# ---------------------------------------------------------------------------
# Concurrency guards
# ---------------------------------------------------------------------------

class TestConcurrencyGuards:

    @pytest.mark.asyncio
    async def test_exception_resets_task_done(self, engine, mock_grounding_agent):
        """_task_done event is re-set after exception (prevents deadlock)."""
        mock_grounding_agent.process = AsyncMock(side_effect=RuntimeError("boom"))
        await engine.execute("task")
        assert engine._task_done.is_set()
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_concurrent_execute_raises_on_timeout(self, engine):
        """Raises RuntimeError if busy-wait exceeds timeout."""
        engine._running = True
        engine._task_done.clear()
        with patch("scion.execution_engine.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(RuntimeError, match="still running"):
                await engine.execute("task")


# ---------------------------------------------------------------------------
# _maybe_analyze_execution — evolution path
# ---------------------------------------------------------------------------

class TestAnalyzeExecutionEvolution:

    @pytest.mark.asyncio
    async def test_triggers_evolution(self, engine):
        """Skills are evolved when analysis has candidate_for_evolution=True."""
        analyzer = MagicMock()
        analysis = MagicMock()
        analysis.candidate_for_evolution = True
        analysis.evolution_suggestions = [MagicMock()]
        analyzer.analyze_execution = AsyncMock(return_value=analysis)

        evolved_rec = MagicMock(
            skill_id="s1", name="Skill1", description="desc", path=None,
            lineage=MagicMock(
                origin=MagicMock(value="synthesized"),
                generation=1, parent_skill_ids=[], change_summary="init"
            ),
        )
        evolver = MagicMock()
        evolver.process_analysis = AsyncMock(return_value=[evolved_rec])
        evolver.set_available_tools = MagicMock()

        engine._execution_analyzer = analyzer
        engine._skill_evolver = evolver

        await engine._maybe_analyze_execution("t1", "/dir", {"status": "ok"})
        assert len(engine._last_evolved_skills) == 1
        assert engine._last_evolved_skills[0]["skill_id"] == "s1"


# ---------------------------------------------------------------------------
# Scion backward compatibility
# ---------------------------------------------------------------------------

class TestScionDelegation:

    def test_scion_has_execution_engine_attr(self):
        """Scion instance has _execution_engine attribute."""
        os_instance = Scion()
        assert hasattr(os_instance, "_execution_engine")
        assert os_instance._execution_engine is None
