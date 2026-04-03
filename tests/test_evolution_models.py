"""Tests for openspace.skill_engine.evolution.models — Epic 5.1."""
from __future__ import annotations

import pytest

from openspace.skill_engine.evolution.models import (
    _MAX_SKILL_NAME_LENGTH,
    EvolutionContext,
    EvolutionTrigger,
    _sanitize_skill_name,
)


class TestEvolutionTrigger:
    """Verify EvolutionTrigger enum."""

    def test_has_three_values(self):
        assert len(EvolutionTrigger) == 3

    def test_values(self):
        assert EvolutionTrigger.ANALYSIS == "analysis"
        assert EvolutionTrigger.TOOL_DEGRADATION == "tool_degradation"
        assert EvolutionTrigger.METRIC_MONITOR == "metric_monitor"

    def test_is_string_enum(self):
        assert isinstance(EvolutionTrigger.ANALYSIS, str)


class TestEvolutionContext:
    """Verify EvolutionContext dataclass."""

    def test_minimal_construction(self):
        from openspace.skill_engine.types import EvolutionSuggestion, EvolutionType

        suggestion = EvolutionSuggestion(
            evolution_type=EvolutionType.FIX,
            target_skill_ids=["test-skill__imp_abc123"],
            direction="Fix broken skill",
        )
        ctx = EvolutionContext(
            trigger=EvolutionTrigger.ANALYSIS,
            suggestion=suggestion,
        )
        assert ctx.trigger == EvolutionTrigger.ANALYSIS
        assert ctx.suggestion is suggestion
        assert ctx.skill_records == []
        assert ctx.source_task_id is None
        assert ctx.tool_issue_summary == ""
        assert ctx.metric_summary == ""
        assert ctx.available_tools == []

    def test_trigger_specific_fields(self):
        from openspace.skill_engine.types import EvolutionSuggestion, EvolutionType

        suggestion = EvolutionSuggestion(
            evolution_type=EvolutionType.FIX,
            target_skill_ids=["test__imp_abc123"],
            direction="fix it",
        )
        ctx = EvolutionContext(
            trigger=EvolutionTrigger.TOOL_DEGRADATION,
            suggestion=suggestion,
            tool_issue_summary="bash tool failing 50% of calls",
        )
        assert ctx.tool_issue_summary == "bash tool failing 50% of calls"


class TestSanitizeSkillName:
    """Verify _sanitize_skill_name enforces naming rules."""

    def test_lowercase(self):
        assert _sanitize_skill_name("MySkill") == "myskill"

    def test_underscores_to_hyphens(self):
        assert _sanitize_skill_name("my_cool_skill") == "my-cool-skill"

    def test_spaces_to_hyphens(self):
        assert _sanitize_skill_name("my cool skill") == "my-cool-skill"

    def test_special_chars_removed(self):
        assert _sanitize_skill_name("skill@v2.0!") == "skill-v2-0"

    def test_collapses_multiple_hyphens(self):
        assert _sanitize_skill_name("a---b---c") == "a-b-c"

    def test_strips_leading_trailing_hyphens(self):
        assert _sanitize_skill_name("--skill--") == "skill"

    def test_truncation_at_word_boundary(self):
        long_name = "this-is-a-very-long-skill-name-that-exceeds-fifty-characters-limit"
        result = _sanitize_skill_name(long_name)
        assert len(result) <= _MAX_SKILL_NAME_LENGTH
        assert not result.endswith("-")

    def test_short_name_unchanged(self):
        assert _sanitize_skill_name("web-scraper") == "web-scraper"

    def test_empty_string(self):
        assert _sanitize_skill_name("") == ""

    def test_max_length_constant(self):
        assert _MAX_SKILL_NAME_LENGTH == 50


class TestBackwardCompatImports:
    """Verify old import paths still resolve to the new location."""

    def test_import_from_evolver(self):
        """evolver.py re-exports EvolutionTrigger and EvolutionContext."""
        from openspace.skill_engine.evolver import EvolutionContext as EC1
        from openspace.skill_engine.evolver import EvolutionTrigger as ET1
        from openspace.skill_engine.evolution.models import EvolutionContext as EC2
        from openspace.skill_engine.evolution.models import EvolutionTrigger as ET2

        assert ET1 is ET2
        assert EC1 is EC2

    def test_import_from_package(self):
        """skill_engine.__init__ re-exports from new location."""
        from openspace.skill_engine import EvolutionContext, EvolutionTrigger
        from openspace.skill_engine.evolution.models import (
            EvolutionContext as EC2,
            EvolutionTrigger as ET2,
        )

        assert EvolutionTrigger is ET2
        assert EvolutionContext is EC2
