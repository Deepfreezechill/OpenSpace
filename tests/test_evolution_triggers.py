"""Tests for openspace.skill_engine.evolution.triggers (Epic 5.3).

Verifies:
  1. process_analysis — dispatches suggestions → contexts → execution
  2. process_tool_degradation — screens, confirms, de-dups, anti-loop
  3. process_metric_check — health classification → LLM confirm → execute
  4. build_context_from_analysis — FIX/DERIVED/CAPTURED context construction
  5. load_skill_content — registry → disk fallback
  6. diagnose_skill_health — pure metric classifier thresholds
  7. Constants moved correctly from evolver.py
  8. Backward compat: SkillEvolver delegates to trigger functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs
# ---------------------------------------------------------------------------

class _FakeEvolutionType(Enum):
    FIX = "fix"
    DERIVED = "derived"
    CAPTURED = "captured"


class _FakeTrigger(Enum):
    ANALYSIS = "analysis"
    TOOL_DEGRADATION = "tool_degradation"
    METRIC_MONITOR = "metric_monitor"


@dataclass
class _FakeSuggestion:
    evolution_type: _FakeEvolutionType = _FakeEvolutionType.FIX
    target_skill_ids: List[str] = field(default_factory=lambda: ["skill-1"])
    direction: str = "fix it"


@dataclass
class _FakeAnalysis:
    task_id: str = "task-001"
    candidate_for_evolution: bool = True
    evolution_suggestions: List[_FakeSuggestion] = field(default_factory=list)


@dataclass
class _FakeSkillRecord:
    skill_id: str = "skill-1"
    name: str = "test-skill"
    is_active: bool = True
    path: str = ""
    total_selections: int = 10
    applied_rate: float = 0.6
    completion_rate: float = 0.5
    effective_rate: float = 0.4
    fallback_rate: float = 0.3


@dataclass
class _FakeToolQualityRecord:
    tool_key: str = "shell_exec"
    recent_success_rate: float = 0.3
    total_calls: int = 50
    llm_flagged_count: int = 3


# ---------------------------------------------------------------------------
# Import trigger functions
# ---------------------------------------------------------------------------

from openspace.skill_engine.evolution.triggers import (
    _ANALYSIS_CONTEXT_MAX,
    _ANALYSIS_NOTE_MAX_CHARS,
    _FALLBACK_THRESHOLD,
    _HIGH_APPLIED_FOR_FIX,
    _LOW_COMPLETION_THRESHOLD,
    _MIN_APPLIED_FOR_DERIVED,
    _MODERATE_EFFECTIVE_THRESHOLD,
    build_context_from_analysis,
    diagnose_skill_health,
    load_skill_content,
    process_analysis,
    process_metric_check,
    process_tool_degradation,
)


# ---------------------------------------------------------------------------
# diagnose_skill_health tests (pure — no mocks needed)
# ---------------------------------------------------------------------------

class TestDiagnoseSkillHealth:
    def test_healthy_skill_returns_none(self):
        record = _FakeSkillRecord(
            fallback_rate=0.1, applied_rate=0.2,
            completion_rate=0.8, effective_rate=0.9,
        )
        evo_type, direction = diagnose_skill_health(record)
        assert evo_type is None
        assert direction == ""

    def test_high_fallback_returns_fix(self):
        record = _FakeSkillRecord(fallback_rate=0.5)
        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType):
            evo_type, direction = diagnose_skill_health(record)
        assert evo_type == _FakeEvolutionType.FIX
        assert "fallback" in direction.lower()

    def test_low_completion_high_applied_returns_fix(self):
        record = _FakeSkillRecord(
            applied_rate=0.5, completion_rate=0.2, fallback_rate=0.1,
        )
        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType):
            evo_type, direction = diagnose_skill_health(record)
        assert evo_type == _FakeEvolutionType.FIX
        assert "completion" in direction.lower()

    def test_moderate_effectiveness_returns_derived(self):
        record = _FakeSkillRecord(
            effective_rate=0.4, applied_rate=0.3,
            fallback_rate=0.1, completion_rate=0.8,
        )
        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType):
            evo_type, direction = diagnose_skill_health(record)
        assert evo_type == _FakeEvolutionType.DERIVED
        assert "effectiveness" in direction.lower()


# ---------------------------------------------------------------------------
# load_skill_content tests
# ---------------------------------------------------------------------------

class TestLoadSkillContent:
    def test_registry_hit(self):
        evolver = MagicMock()
        evolver._registry.load_skill_content.return_value = "# Skill Content"
        record = _FakeSkillRecord(skill_id="s1")

        result = load_skill_content(evolver, record)
        assert result == "# Skill Content"
        evolver._registry.load_skill_content.assert_called_once_with("s1")

    def test_disk_fallback(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# Disk Content", encoding="utf-8")

        evolver = MagicMock()
        evolver._registry.load_skill_content.return_value = ""
        record = _FakeSkillRecord(path=str(skill_file))

        result = load_skill_content(evolver, record)
        assert result == "# Disk Content"

    def test_missing_returns_empty(self):
        evolver = MagicMock()
        evolver._registry.load_skill_content.return_value = ""
        record = _FakeSkillRecord(path="/nonexistent/SKILL.md")

        result = load_skill_content(evolver, record)
        assert result == ""


# ---------------------------------------------------------------------------
# build_context_from_analysis tests
# ---------------------------------------------------------------------------

class TestBuildContextFromAnalysis:
    def test_captured_type_returns_empty_parents(self):
        evolver = MagicMock()
        evolver._available_tools = []
        analysis = _FakeAnalysis()
        suggestion = _FakeSuggestion(
            evolution_type=_FakeEvolutionType.CAPTURED,
            target_skill_ids=[],
        )

        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType), \
             patch("openspace.skill_engine.evolution.triggers.EvolutionTrigger", _FakeTrigger):
            ctx = build_context_from_analysis(evolver, analysis, suggestion)

        assert ctx is not None
        assert ctx.skill_records == []
        assert ctx.skill_contents == []

    def test_fix_missing_targets_returns_none(self):
        evolver = MagicMock()
        analysis = _FakeAnalysis()
        suggestion = _FakeSuggestion(
            evolution_type=_FakeEvolutionType.FIX,
            target_skill_ids=[],
        )

        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType):
            ctx = build_context_from_analysis(evolver, analysis, suggestion)

        assert ctx is None

    def test_fix_loads_single_target(self):
        evolver = MagicMock()
        evolver._available_tools = []
        evolver._store.load_record.return_value = _FakeSkillRecord(path="/skills/s1/SKILL.md")
        evolver._registry.load_skill_content.return_value = "# Content"

        analysis = _FakeAnalysis()
        suggestion = _FakeSuggestion(
            evolution_type=_FakeEvolutionType.FIX,
            target_skill_ids=["skill-1"],
        )

        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType), \
             patch("openspace.skill_engine.evolution.triggers.EvolutionTrigger", _FakeTrigger):
            ctx = build_context_from_analysis(evolver, analysis, suggestion)

        assert ctx is not None
        assert len(ctx.skill_records) == 1


# ---------------------------------------------------------------------------
# process_analysis tests
# ---------------------------------------------------------------------------

class TestProcessAnalysis:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self):
        evolver = MagicMock()
        analysis = _FakeAnalysis(candidate_for_evolution=False)

        result = await process_analysis(evolver, analysis)
        assert result == []

    @pytest.mark.asyncio
    async def test_suggestions_dispatched(self):
        evolver = MagicMock()
        evolver._available_tools = []
        evolver._store.load_record.return_value = _FakeSkillRecord(path="/s/SKILL.md")
        evolver._registry.load_skill_content.return_value = "# Content"
        evolver._execute_contexts = AsyncMock(return_value=[_FakeSkillRecord(name="evolved")])

        analysis = _FakeAnalysis(
            evolution_suggestions=[
                _FakeSuggestion(evolution_type=_FakeEvolutionType.FIX),
            ],
        )

        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType), \
             patch("openspace.skill_engine.evolution.triggers.EvolutionTrigger", _FakeTrigger):
            result = await process_analysis(evolver, analysis)

        assert len(result) == 1
        evolver._execute_contexts.assert_awaited_once()


# ---------------------------------------------------------------------------
# process_tool_degradation tests
# ---------------------------------------------------------------------------

class TestProcessToolDegradation:
    @pytest.mark.asyncio
    async def test_empty_tools_returns_empty(self):
        evolver = MagicMock()
        evolver._addressed_degradations = {}

        result = await process_tool_degradation(evolver, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_anti_loop_skips_addressed_skills(self):
        evolver = MagicMock()
        evolver._addressed_degradations = {"shell_exec": {"skill-1"}}
        evolver._store.find_skills_by_tool.return_value = ["skill-1"]
        evolver._store.load_record.return_value = _FakeSkillRecord()
        evolver._available_tools = []
        evolver._execute_contexts = AsyncMock(return_value=[])

        tool = _FakeToolQualityRecord(tool_key="shell_exec")
        result = await process_tool_degradation(evolver, [tool])
        assert result == []

    @pytest.mark.asyncio
    async def test_prunes_recovered_tools(self):
        evolver = MagicMock()
        evolver._addressed_degradations = {"old_tool": {"skill-x"}, "shell_exec": set()}
        evolver._store.find_skills_by_tool.return_value = []
        evolver._available_tools = []
        evolver._execute_contexts = AsyncMock(return_value=[])

        tool = _FakeToolQualityRecord(tool_key="shell_exec")
        await process_tool_degradation(evolver, [tool])

        # old_tool should have been pruned
        assert "old_tool" not in evolver._addressed_degradations


# ---------------------------------------------------------------------------
# process_metric_check tests
# ---------------------------------------------------------------------------

class TestProcessMetricCheck:
    @pytest.mark.asyncio
    async def test_skips_low_selection_skills(self):
        evolver = MagicMock()
        evolver._store.load_active.return_value = {
            "s1": _FakeSkillRecord(total_selections=2),
        }
        evolver._available_tools = []

        result = await process_metric_check(evolver, min_selections=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_healthy_skills_skipped(self):
        evolver = MagicMock()
        evolver._store.load_active.return_value = {
            "s1": _FakeSkillRecord(
                total_selections=10,
                fallback_rate=0.1, applied_rate=0.2,
                completion_rate=0.8, effective_rate=0.9,
            ),
        }
        evolver._available_tools = []

        with patch("openspace.skill_engine.evolution.triggers.EvolutionType", _FakeEvolutionType):
            result = await process_metric_check(evolver, min_selections=5)

        assert result == []


# ---------------------------------------------------------------------------
# Constants integrity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_thresholds_are_numbers(self):
        assert isinstance(_FALLBACK_THRESHOLD, float)
        assert isinstance(_LOW_COMPLETION_THRESHOLD, float)
        assert isinstance(_HIGH_APPLIED_FOR_FIX, float)
        assert isinstance(_MODERATE_EFFECTIVE_THRESHOLD, float)
        assert isinstance(_MIN_APPLIED_FOR_DERIVED, float)

    def test_analysis_constants(self):
        assert isinstance(_ANALYSIS_CONTEXT_MAX, int)
        assert isinstance(_ANALYSIS_NOTE_MAX_CHARS, int)

    def test_constants_no_longer_in_evolver(self):
        """Constants should be imported, not defined, in evolver.py."""
        import inspect
        from openspace.skill_engine import evolver as mod
        source = inspect.getsource(mod)
        # Should not have the original assignment (but import is fine)
        assert "_FALLBACK_THRESHOLD = 0.4" not in source


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

try:
    from openspace.skill_engine.evolver import SkillEvolver
    _HAS_EVOLVER = True
except ImportError:
    _HAS_EVOLVER = False


@pytest.mark.skipif(not _HAS_EVOLVER, reason="SkillEvolver not importable")
class TestBackwardCompat:
    def test_trigger_methods_exist(self):
        assert hasattr(SkillEvolver, "process_analysis")
        assert hasattr(SkillEvolver, "process_tool_degradation")
        assert hasattr(SkillEvolver, "process_metric_check")
        assert hasattr(SkillEvolver, "_build_context_from_analysis")
        assert hasattr(SkillEvolver, "_load_skill_content")
        assert hasattr(SkillEvolver, "_diagnose_skill_health")


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

class TestSizeGuard:
    def test_triggers_module_size(self):
        mod_path = (
            Path(__file__).resolve().parent.parent
            / "openspace" / "skill_engine" / "evolution" / "triggers.py"
        )
        assert mod_path.exists(), f"triggers.py not found at {mod_path}"
        lines = mod_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) < 400, f"triggers.py has {len(lines)} lines (limit 400)"
