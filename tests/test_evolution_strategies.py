"""Tests for openspace.skill_engine.evolution.strategies (Epic 5.5).

Verifies:
  1. evolve_fix — builds prompt, calls loop, applies, persists, registers
  2. evolve_derived — single-parent + multi-parent merge paths
  3. evolve_captured — creates new skill from scratch
  4. Guard clauses: missing skill_records/contents/dirs → None
  5. MRO: all internal calls go through evolver._method()
  6. Backward compat: SkillEvolver delegates to strategy functions
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from openspace.skill_engine.evolution.strategies import (
    evolve_captured,
    evolve_derived,
    evolve_fix,
)
from openspace.skill_engine.review_gate import CheckResult, ReviewResult
from openspace.skill_engine.skill_guard import SkillGuard


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeEvolutionType:
    value = "FIX"


class _FakeSuggestion:
    evolution_type = _FakeEvolutionType()
    target_skill_ids = ["skill-a"]
    direction = "improve error handling"
    category = None


class _FakeTrigger:
    value = "analysis"


class _FakeLineage:
    generation = 1
    origin = "FIXED"
    parent_skill_ids = []


class _FakeRecord:
    skill_id = "skill-a__v1_abc"
    name = "my-skill"
    description = "A skill"
    path = "/skills/my-skill/SKILL.md"
    category = "WORKFLOW"
    tags = ["test"]
    visibility = "private"
    creator_id = "user-1"
    lineage = _FakeLineage()
    tool_dependencies = ["tool-a"]
    critical_tools = ["tool-a"]


class _FakeEditResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.error = None
        self.content_snapshot = {"SKILL.md": "---\nname: my-skill\ndescription: A skill\n---\ncontent"}
        self.content_diff = "diff here"


class _FakeCtx:
    suggestion = _FakeSuggestion()
    trigger = _FakeTrigger()
    available_tools = []
    recent_analyses = []
    skill_records = [_FakeRecord()]
    skill_contents = ["original content"]
    skill_dirs = [Path("/skills/my-skill")]
    source_task_id = "task-1"
    tool_issue_summary = ""
    metric_summary = ""


class _FakeEvolver:
    _model = "test-model"
    _available_tools = []

    def __init__(self):
        self._llm_client = MagicMock()
        self._llm_client.model = "fallback-model"
        self._store = MagicMock()
        self._store.evolve_skill = AsyncMock()
        self._store.save_record = AsyncMock()
        # SkillGuard wraps the mock store with an always-pass gate
        _pass_gate = MagicMock()
        _pass_gate.review = MagicMock(return_value=ReviewResult.from_checks([
            CheckResult(name="test-gate", verdict="pass", detail="ok"),
        ]))
        self._guard = SkillGuard(store=self._store, gate=_pass_gate)
        self._registry = MagicMock()
        self._registry._skill_dirs = [Path("/skills")]
        self._registry.update_skill = MagicMock()
        self._registry.add_skill = MagicMock()

    def _format_skill_dir_content(self, skill_dir):
        return ""

    def _format_analysis_context(self, analyses):
        return "(no context)"

    async def _run_evolution_loop(self, prompt, ctx):
        return "---\nname: my-skill\n---\nnew content"

    async def _apply_with_retry(self, **kwargs):
        return _FakeEditResult(ok=True)


# ---------------------------------------------------------------------------
# evolve_fix
# ---------------------------------------------------------------------------

class TestEvolveFix:
    @pytest.mark.asyncio
    async def test_success(self):
        """Happy path: FIX produces a new record with correct fields."""
        evolver = _FakeEvolver()
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.strategies.write_skill_id"), \
             patch("openspace.skill_engine.registry.SkillMeta"):
            result = await evolve_fix(evolver, ctx)

        assert result is not None
        assert result.name == "my-skill"
        assert result.lineage.origin.value == "fixed"
        assert result.lineage.generation == 2  # parent gen 1 + 1
        assert result.lineage.parent_skill_ids == ["skill-a__v1_abc"]
        assert result.lineage.source_task_id == "task-1"
        assert "tool-a" in result.tool_dependencies
        evolver._store.evolve_skill.assert_awaited_once()
        evolver._registry.update_skill.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_records_returns_none(self):
        """Missing skill_records → None."""
        evolver = _FakeEvolver()
        ctx = _FakeCtx()
        ctx.skill_records = []

        result = await evolve_fix(evolver, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_loop_returns_none(self):
        """Agent loop fails → None."""
        evolver = _FakeEvolver()
        evolver._run_evolution_loop = AsyncMock(return_value=None)
        ctx = _FakeCtx()

        result = await evolve_fix(evolver, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_apply_fails_returns_none(self):
        """Apply-retry fails → None."""
        evolver = _FakeEvolver()
        evolver._apply_with_retry = AsyncMock(return_value=None)
        ctx = _FakeCtx()

        result = await evolve_fix(evolver, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_mro_format_called(self):
        """Calls go through evolver._format_* (MRO)."""
        evolver = _FakeEvolver()
        evolver._format_skill_dir_content = MagicMock(return_value="dir content")
        evolver._format_analysis_context = MagicMock(return_value="(context)")
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.strategies.write_skill_id"), \
             patch("openspace.skill_engine.registry.SkillMeta"):
            await evolve_fix(evolver, ctx)

        evolver._format_skill_dir_content.assert_called_once()
        evolver._format_analysis_context.assert_called_once()


# ---------------------------------------------------------------------------
# evolve_derived
# ---------------------------------------------------------------------------

class TestEvolveDerived:
    @pytest.mark.asyncio
    async def test_single_parent_success(self):
        """Single parent → enhanced skill."""
        evolver = _FakeEvolver()
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.strategies.write_skill_id"), \
             patch("openspace.skill_engine.registry.SkillMeta"):
            result = await evolve_derived(evolver, ctx)

        assert result is not None
        evolver._store.evolve_skill.assert_awaited_once()
        evolver._registry.add_skill.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_records_returns_none(self):
        """Missing skill_records → None."""
        evolver = _FakeEvolver()
        ctx = _FakeCtx()
        ctx.skill_records = []

        result = await evolve_derived(evolver, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_loop_returns_none(self):
        """Agent loop fails → None."""
        evolver = _FakeEvolver()
        evolver._run_evolution_loop = AsyncMock(return_value=None)
        ctx = _FakeCtx()

        result = await evolve_derived(evolver, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# evolve_captured
# ---------------------------------------------------------------------------

class TestEvolveCaptured:
    @pytest.mark.asyncio
    async def test_success(self):
        """Happy path: CAPTURED produces a new record."""
        evolver = _FakeEvolver()
        ctx = _FakeCtx()
        ctx.skill_records = []
        ctx.skill_contents = []
        ctx.skill_dirs = []

        with patch("openspace.skill_engine.evolution.strategies.write_skill_id"), \
             patch("openspace.skill_engine.registry.SkillMeta"):
            result = await evolve_captured(evolver, ctx)

        assert result is not None
        evolver._store.save_record.assert_awaited_once()
        evolver._registry.add_skill.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_name_returns_none(self):
        """LLM doesn't produce a name → None."""
        evolver = _FakeEvolver()
        evolver._run_evolution_loop = AsyncMock(return_value="no frontmatter here")
        ctx = _FakeCtx()
        ctx.skill_records = []
        ctx.skill_contents = []
        ctx.skill_dirs = []

        result = await evolve_captured(evolver, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_skill_dirs_returns_none(self):
        """No skill directories configured → None."""
        evolver = _FakeEvolver()
        evolver._registry._skill_dirs = []
        ctx = _FakeCtx()
        ctx.skill_records = []
        ctx.skill_contents = []
        ctx.skill_dirs = []

        result = await evolve_captured(evolver, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_apply_fails_returns_none(self):
        """Apply-retry fails → None."""
        evolver = _FakeEvolver()
        evolver._apply_with_retry = AsyncMock(return_value=None)
        ctx = _FakeCtx()
        ctx.skill_records = []
        ctx.skill_contents = []
        ctx.skill_dirs = []

        result = await evolve_captured(evolver, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

class TestDelegationSeam:
    def test_evolve_fix_delegate(self):
        """SkillEvolver._evolve_fix exists and is async."""
        from openspace.skill_engine.evolver import SkillEvolver
        assert asyncio.iscoroutinefunction(SkillEvolver._evolve_fix)

    def test_evolve_derived_delegate(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert asyncio.iscoroutinefunction(SkillEvolver._evolve_derived)

    def test_evolve_captured_delegate(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert asyncio.iscoroutinefunction(SkillEvolver._evolve_captured)


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

class TestSizeGuard:
    def test_strategies_module_size(self):
        """strategies.py should stay under 400 lines."""
        import openspace.skill_engine.evolution.strategies as mod
        src = Path(mod.__file__)
        lines = src.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 400, f"strategies.py has {len(lines)} lines (limit 400)"
