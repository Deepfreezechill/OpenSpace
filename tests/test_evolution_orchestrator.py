"""Tests for scion.skill_engine.evolution.orchestrator (Epic 5.2).

Verifies:
  1. dispatch_evolution routes FIX/DERIVED/CAPTURED correctly
  2. dispatch_evolution handles unknown types and exceptions gracefully
  3. execute_contexts throttles via semaphore and collects results
  4. execute_contexts tolerates exceptions in individual contexts
  5. schedule_background creates and tracks tasks
  6. schedule_background handles no-event-loop case
  7. log_background_result logs errors/cancellations
  8. Backward compat: SkillEvolver.evolve delegates to orchestrator
  9. Backward compat: SkillEvolver.schedule_background delegates
  10. Identity: orchestrator functions are the canonical implementations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we don't import the whole skill engine
# ---------------------------------------------------------------------------

class _FakeEvolutionType(Enum):
    FIX = "fix"
    DERIVED = "derived"
    CAPTURED = "captured"
    UNKNOWN = "unknown"


@dataclass
class _FakeSuggestion:
    evolution_type: _FakeEvolutionType = _FakeEvolutionType.FIX
    target_skill_ids: List[str] = field(default_factory=lambda: ["skill-1"])


@dataclass
class _FakeContext:
    suggestion: _FakeSuggestion = field(default_factory=_FakeSuggestion)


@dataclass
class _FakeSkillRecord:
    name: str = "test-skill"


class _FakeEvolver:
    """Minimal duck-typed evolver for testing orchestrator functions."""

    def __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._background_tasks: Set[asyncio.Task] = set()
        self._evolve_fix = AsyncMock(return_value=_FakeSkillRecord(name="fixed"))
        self._evolve_derived = AsyncMock(return_value=_FakeSkillRecord(name="derived"))
        self._evolve_captured = AsyncMock(return_value=_FakeSkillRecord(name="captured"))

    async def evolve(self, ctx):
        """Delegate to dispatch_evolution, mirroring real SkillEvolver."""
        return await dispatch_evolution(self, ctx)


# ---------------------------------------------------------------------------
# Import orchestrator — patch EvolutionType so dispatch_evolution resolves
# ---------------------------------------------------------------------------

from scion.skill_engine.evolution.orchestrator import (
    dispatch_evolution,
    execute_contexts,
    log_background_result,
    schedule_background,
)


# ---------------------------------------------------------------------------
# dispatch_evolution tests
# ---------------------------------------------------------------------------

class TestDispatchEvolution:
    @pytest.mark.asyncio
    async def test_routes_fix(self):
        evolver = _FakeEvolver()
        ctx = _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX))

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            result = await dispatch_evolution(evolver, ctx)

        assert result is not None
        assert result.name == "fixed"
        evolver._evolve_fix.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_routes_derived(self):
        evolver = _FakeEvolver()
        ctx = _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.DERIVED))

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            result = await dispatch_evolution(evolver, ctx)

        assert result is not None
        assert result.name == "derived"
        evolver._evolve_derived.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_routes_captured(self):
        evolver = _FakeEvolver()
        ctx = _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.CAPTURED))

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            result = await dispatch_evolution(evolver, ctx)

        assert result is not None
        assert result.name == "captured"
        evolver._evolve_captured.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_unknown_type_returns_none(self):
        evolver = _FakeEvolver()
        ctx = _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.UNKNOWN))

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            result = await dispatch_evolution(evolver, ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none_and_logs(self, caplog):
        evolver = _FakeEvolver()
        evolver._evolve_fix = AsyncMock(side_effect=RuntimeError("boom"))
        ctx = _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX))

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            with caplog.at_level(logging.ERROR):
                result = await dispatch_evolution(evolver, ctx)

        assert result is None
        assert "boom" in caplog.text


# ---------------------------------------------------------------------------
# execute_contexts tests
# ---------------------------------------------------------------------------

class TestExecuteContexts:
    @pytest.mark.asyncio
    async def test_runs_all_contexts_and_collects_results(self):
        evolver = _FakeEvolver()
        ctxs = [
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX)),
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.DERIVED)),
        ]

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            results = await execute_contexts(evolver, ctxs, "test-trigger")

        assert len(results) == 2
        names = {r.name for r in results}
        assert "fixed" in names
        assert "derived" in names

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_empty(self):
        evolver = _FakeEvolver()

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            results = await execute_contexts(evolver, [], "empty")

        assert results == []

    @pytest.mark.asyncio
    async def test_exception_in_one_context_does_not_block_others(self, caplog):
        evolver = _FakeEvolver()
        evolver._evolve_fix = AsyncMock(side_effect=RuntimeError("fail-fix"))
        ctxs = [
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX)),
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.DERIVED)),
        ]

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            with caplog.at_level(logging.ERROR):
                results = await execute_contexts(evolver, ctxs, "mixed")

        # Derived should still succeed
        assert len(results) == 1
        assert results[0].name == "derived"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Verify semaphore actually gates concurrent executions."""
        evolver = _FakeEvolver(max_concurrent=1)
        order = []

        async def slow_fix(ctx):
            order.append("fix-start")
            await asyncio.sleep(0.05)
            order.append("fix-end")
            return _FakeSkillRecord(name="fixed")

        async def slow_derived(ctx):
            order.append("derived-start")
            await asyncio.sleep(0.05)
            order.append("derived-end")
            return _FakeSkillRecord(name="derived")

        evolver._evolve_fix = slow_fix
        evolver._evolve_derived = slow_derived

        ctxs = [
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX)),
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.DERIVED)),
        ]

        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            results = await execute_contexts(evolver, ctxs, "serial")

        assert len(results) == 2
        # With semaphore=1, tasks run serially: first ends before second starts
        fix_end = order.index("fix-end")
        derived_start = order.index("derived-start")
        assert fix_end < derived_start, f"Expected serial execution, got {order}"


# ---------------------------------------------------------------------------
# schedule_background tests
# ---------------------------------------------------------------------------

class TestScheduleBackground:
    @pytest.mark.asyncio
    async def test_creates_and_tracks_task(self):
        tasks: Set[asyncio.Task] = set()

        async def noop():
            return 42

        task = schedule_background(tasks, noop(), label="test-bg")
        assert task is not None
        assert task in tasks
        assert task.get_name() == "test-bg"

        result = await task
        assert result == 42
        # done callback should have removed it
        assert task not in tasks

    @pytest.mark.asyncio
    async def test_failed_task_removed_and_logged(self, caplog):
        tasks: Set[asyncio.Task] = set()

        async def fail():
            raise ValueError("bg-boom")

        task = schedule_background(tasks, fail(), label="fail-bg")
        assert task is not None

        with caplog.at_level(logging.ERROR):
            # Wait for task to complete (it will raise)
            await asyncio.sleep(0.1)

        assert task not in tasks
        assert "bg-boom" in caplog.text

    def test_no_event_loop_returns_none(self, caplog):
        """When called outside an event loop, returns None gracefully."""
        tasks: set = set()

        async def noop():
            pass

        coro = noop()
        # We need to be outside a running loop for this test
        # Since pytest-asyncio provides a loop, we test via a sync function
        # that patches get_running_loop to raise
        with patch("scion.skill_engine.evolution.orchestrator.asyncio") as mock_asyncio:
            mock_asyncio.get_running_loop.side_effect = RuntimeError("no loop")
            with caplog.at_level(logging.WARNING):
                result = schedule_background(tasks, coro, label="no-loop")

        assert result is None
        assert len(tasks) == 0
        # Clean up the coroutine
        coro.close()


# ---------------------------------------------------------------------------
# log_background_result tests
# ---------------------------------------------------------------------------

class TestLogBackgroundResult:
    def test_cancelled_task_logs_debug(self, caplog):
        task = MagicMock()
        task.cancelled.return_value = True
        task.get_name.return_value = "cancelled-task"

        with caplog.at_level(logging.DEBUG):
            log_background_result(task)

        assert "cancelled" in caplog.text

    def test_failed_task_logs_error(self, caplog):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("task-error")
        task.get_name.return_value = "error-task"

        with caplog.at_level(logging.ERROR):
            log_background_result(task)

        assert "task-error" in caplog.text

    def test_successful_task_logs_nothing(self, caplog):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        task.get_name.return_value = "ok-task"

        with caplog.at_level(logging.DEBUG):
            log_background_result(task)

        assert "ok-task" not in caplog.text


# ---------------------------------------------------------------------------
# Backward compatibility — SkillEvolver delegates to orchestrator
# ---------------------------------------------------------------------------

try:
    from scion.skill_engine.evolution.orchestrator import (
        dispatch_evolution as _canonical_dispatch,
        execute_contexts as _canonical_execute,
        log_background_result as _canonical_log,
        schedule_background as _canonical_schedule,
    )
    _HAS_COMPAT = True
except ImportError:
    _HAS_COMPAT = False


@pytest.mark.skipif(not _HAS_COMPAT, reason="orchestrator not importable")
class TestBackwardCompat:
    def test_orchestrator_importable_from_evolution_package(self):
        """Canonical import path works."""
        from scion.skill_engine.evolution import orchestrator
        assert hasattr(orchestrator, "dispatch_evolution")
        assert hasattr(orchestrator, "execute_contexts")
        assert hasattr(orchestrator, "schedule_background")
        assert hasattr(orchestrator, "log_background_result")

    def test_functions_are_canonical(self):
        """The orchestrator module functions are the real implementations."""
        import types
        assert isinstance(_canonical_dispatch, types.FunctionType)
        assert isinstance(_canonical_execute, types.FunctionType)
        assert isinstance(_canonical_schedule, types.FunctionType)
        assert isinstance(_canonical_log, types.FunctionType)


# ---------------------------------------------------------------------------
# Delegation integration — SkillEvolver → orchestrator seam
# ---------------------------------------------------------------------------

try:
    from scion.skill_engine.evolver import SkillEvolver
    _HAS_EVOLVER = True
except ImportError:
    _HAS_EVOLVER = False


@pytest.mark.skipif(not _HAS_EVOLVER, reason="SkillEvolver not importable (litellm)")
class TestDelegationSeam:
    """Verify SkillEvolver methods actually delegate to orchestrator functions."""

    @pytest.mark.asyncio
    async def test_evolve_reaches_dispatch(self):
        """SkillEvolver.evolve() → dispatch_evolution → _evolve_fix."""
        evolver = object.__new__(SkillEvolver)  # skip __init__
        called_with = []

        async def fake_fix(ctx):
            called_with.append(ctx)
            return _FakeSkillRecord(name="via-evolve")

        evolver._evolve_fix = fake_fix

        ctx = _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX))
        with patch("scion.skill_engine.evolution.orchestrator.EvolutionType", _FakeEvolutionType):
            result = await evolver.evolve(ctx)

        assert result is not None
        assert result.name == "via-evolve"
        assert called_with == [ctx], "evolve() should delegate through to _evolve_fix"

    @pytest.mark.asyncio
    async def test_execute_contexts_goes_through_evolve(self):
        """_execute_contexts calls self.evolve(), not dispatch_evolution directly."""
        evolver = object.__new__(SkillEvolver)
        evolver._semaphore = asyncio.Semaphore(2)
        evolve_calls = []

        async def tracking_evolve(ctx):
            evolve_calls.append(ctx)
            return _FakeSkillRecord(name="tracked")

        evolver.evolve = tracking_evolve

        ctxs = [
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.FIX)),
            _FakeContext(_FakeSuggestion(evolution_type=_FakeEvolutionType.DERIVED)),
        ]
        results = await evolver._execute_contexts(ctxs, "seam-test")

        assert len(results) == 2
        assert len(evolve_calls) == 2, "Batch path must go through evolve(), not bypass it"

    @pytest.mark.asyncio
    async def test_schedule_background_delegates(self):
        """SkillEvolver.schedule_background() tracks task in _background_tasks."""
        evolver = object.__new__(SkillEvolver)
        evolver._background_tasks = set()

        async def noop():
            return "bg-done"

        task = evolver.schedule_background(noop(), label="seam-bg")
        assert task is not None
        assert task in evolver._background_tasks

        result = await task
        assert result == "bg-done"


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

class TestSizeGuard:
    def test_orchestrator_module_size(self):
        """orchestrator.py should stay focused (< 160 lines)."""
        from pathlib import Path
        mod_path = Path(__file__).resolve().parent.parent / "scion" / "skill_engine" / "evolution" / "orchestrator.py"
        assert mod_path.exists(), f"orchestrator.py not found at {mod_path}"
        lines = mod_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) < 160, f"orchestrator.py has {len(lines)} lines (limit 160)"
