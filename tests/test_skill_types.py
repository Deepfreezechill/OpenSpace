"""Tests for skill_engine/types.py — EPIC 3.1: SkillSchema + SkillVersion.

Covers:
  - SkillVersion parsing, comparison, incrementing, serialization
  - SkillRecord creation, defaults, serialization round-trip, validation
  - SkillLineage parent-child tracking, lineage-rule validation
  - SkillJudgment & ExecutionAnalysis round-trips, validation
  - EvolutionSuggestion target-rule validation
  - Enum completeness & EvolutionType↔SkillOrigin mapping
  - Edge cases: empty strings, None, boundary dates, negative counters
"""

from __future__ import annotations

from datetime import datetime

import pytest

from scion.skill_engine.types import (
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
    SkillCategory,
    SkillJudgment,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillVersion,
    SkillVisibility,
    ValidationError,
)

# ═══════════════════════════════════════════════════════════════════════
#  SkillVersion
# ═══════════════════════════════════════════════════════════════════════


class TestSkillVersion:
    """Semantic versioning for skills: parse, compare, bump, serialize."""

    # --- Construction & parsing ---

    def test_default_version(self):
        v = SkillVersion()
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0

    def test_explicit_construction(self):
        v = SkillVersion(2, 3, 7)
        assert (v.major, v.minor, v.patch) == (2, 3, 7)

    def test_parse_valid_string(self):
        v = SkillVersion.parse("1.2.3")
        assert (v.major, v.minor, v.patch) == (1, 2, 3)

    def test_parse_two_part(self):
        v = SkillVersion.parse("3.5")
        assert (v.major, v.minor, v.patch) == (3, 5, 0)

    def test_parse_single_part(self):
        v = SkillVersion.parse("7")
        assert (v.major, v.minor, v.patch) == (7, 0, 0)

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            SkillVersion.parse("not.a.version")

    def test_parse_negative_raises(self):
        with pytest.raises(ValueError):
            SkillVersion.parse("-1.0.0")

    def test_parse_too_many_parts_raises(self):
        with pytest.raises(ValueError):
            SkillVersion.parse("1.2.3.4")

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError):
            SkillVersion.parse("")

    # --- String representation ---

    def test_str(self):
        assert str(SkillVersion(1, 2, 3)) == "1.2.3"

    def test_repr(self):
        r = repr(SkillVersion(1, 2, 3))
        assert "SkillVersion" in r
        assert "1" in r and "2" in r and "3" in r

    # --- Comparison operators ---

    def test_eq(self):
        assert SkillVersion(1, 0, 0) == SkillVersion(1, 0, 0)

    def test_neq(self):
        assert SkillVersion(1, 0, 0) != SkillVersion(1, 0, 1)

    def test_lt_patch(self):
        assert SkillVersion(1, 0, 0) < SkillVersion(1, 0, 1)

    def test_lt_minor(self):
        assert SkillVersion(1, 0, 9) < SkillVersion(1, 1, 0)

    def test_lt_major(self):
        assert SkillVersion(1, 9, 9) < SkillVersion(2, 0, 0)

    def test_le(self):
        assert SkillVersion(1, 0, 0) <= SkillVersion(1, 0, 0)
        assert SkillVersion(1, 0, 0) <= SkillVersion(1, 0, 1)

    def test_gt(self):
        assert SkillVersion(2, 0, 0) > SkillVersion(1, 9, 9)

    def test_ge(self):
        assert SkillVersion(2, 0, 0) >= SkillVersion(2, 0, 0)
        assert SkillVersion(2, 0, 1) >= SkillVersion(2, 0, 0)

    def test_sorting(self):
        versions = [
            SkillVersion(2, 0, 0),
            SkillVersion(1, 0, 0),
            SkillVersion(1, 1, 0),
            SkillVersion(1, 0, 1),
        ]
        assert sorted(versions) == [
            SkillVersion(1, 0, 0),
            SkillVersion(1, 0, 1),
            SkillVersion(1, 1, 0),
            SkillVersion(2, 0, 0),
        ]

    # --- Bumping ---

    def test_bump_patch(self):
        assert SkillVersion(1, 2, 3).bump_patch() == SkillVersion(1, 2, 4)

    def test_bump_minor(self):
        assert SkillVersion(1, 2, 3).bump_minor() == SkillVersion(1, 3, 0)

    def test_bump_major(self):
        assert SkillVersion(1, 2, 3).bump_major() == SkillVersion(2, 0, 0)

    # --- Serialization ---

    def test_to_dict(self):
        d = SkillVersion(1, 2, 3).to_dict()
        assert d == {"major": 1, "minor": 2, "patch": 3}

    def test_from_dict(self):
        v = SkillVersion.from_dict({"major": 1, "minor": 2, "patch": 3})
        assert v == SkillVersion(1, 2, 3)

    def test_round_trip(self):
        v = SkillVersion(5, 12, 99)
        assert SkillVersion.from_dict(v.to_dict()) == v

    # --- Validation ---

    def test_negative_major_raises(self):
        with pytest.raises(ValueError):
            SkillVersion(-1, 0, 0)

    def test_negative_minor_raises(self):
        with pytest.raises(ValueError):
            SkillVersion(0, -1, 0)

    def test_negative_patch_raises(self):
        with pytest.raises(ValueError):
            SkillVersion(0, 0, -1)


# ═══════════════════════════════════════════════════════════════════════
#  Enum completeness & mapping
# ═══════════════════════════════════════════════════════════════════════


class TestEnums:
    """Verify enum members and cross-enum mapping tables."""

    def test_skill_category_members(self):
        assert set(SkillCategory) == {
            SkillCategory.TOOL_GUIDE,
            SkillCategory.WORKFLOW,
            SkillCategory.REFERENCE,
        }

    def test_skill_visibility_members(self):
        assert set(SkillVisibility) == {
            SkillVisibility.PRIVATE,
            SkillVisibility.PUBLIC,
        }

    def test_evolution_type_members(self):
        assert set(EvolutionType) == {
            EvolutionType.FIX,
            EvolutionType.DERIVED,
            EvolutionType.CAPTURED,
        }

    def test_skill_origin_members(self):
        assert set(SkillOrigin) == {
            SkillOrigin.IMPORTED,
            SkillOrigin.CAPTURED,
            SkillOrigin.DERIVED,
            SkillOrigin.FIXED,
        }

    def test_evolution_type_to_origin_mapping(self):
        assert EvolutionType.FIX.to_origin() == SkillOrigin.FIXED
        assert EvolutionType.DERIVED.to_origin() == SkillOrigin.DERIVED
        assert EvolutionType.CAPTURED.to_origin() == SkillOrigin.CAPTURED

    def test_all_evolution_types_have_origin(self):
        for et in EvolutionType:
            origin = et.to_origin()
            assert isinstance(origin, SkillOrigin)

    def test_str_enum_values(self):
        """str(Enum) enums have string values usable in serialization."""
        assert SkillCategory.WORKFLOW.value == "workflow"
        assert SkillOrigin.FIXED.value == "fixed"
        assert EvolutionType.FIX.value == "fix"
        assert SkillVisibility.PRIVATE.value == "private"


# ═══════════════════════════════════════════════════════════════════════
#  SkillLineage
# ═══════════════════════════════════════════════════════════════════════


class TestSkillLineage:
    """Lineage creation, serialization, and validation rules."""

    def test_default_lineage(self):
        lin = SkillLineage(origin=SkillOrigin.IMPORTED)
        assert lin.generation == 0
        assert lin.parent_skill_ids == []
        assert lin.change_summary == ""

    def test_round_trip(self):
        lin = SkillLineage(
            origin=SkillOrigin.FIXED,
            generation=3,
            parent_skill_ids=["parent_abc"],
            source_task_id="task_42",
            change_summary="Fixed curl params",
            content_diff="--- a\n+++ b\n",
            content_snapshot={"SKILL.md": "# Fixed"},
            created_by="gpt-4",
        )
        restored = SkillLineage.from_dict(lin.to_dict())
        assert restored.origin == lin.origin
        assert restored.generation == lin.generation
        assert restored.parent_skill_ids == lin.parent_skill_ids
        assert restored.source_task_id == lin.source_task_id
        assert restored.change_summary == lin.change_summary
        assert restored.content_diff == lin.content_diff
        assert restored.content_snapshot == lin.content_snapshot
        assert restored.created_by == lin.created_by

    def test_from_dict_missing_optional_fields(self):
        """Minimal dict should produce valid defaults."""
        lin = SkillLineage.from_dict({"origin": "imported"})
        assert lin.origin == SkillOrigin.IMPORTED
        assert lin.generation == 0
        assert lin.parent_skill_ids == []

    # --- Validation ---

    def test_validate_imported_no_parents(self):
        """IMPORTED must have no parents."""
        lin = SkillLineage(origin=SkillOrigin.IMPORTED, parent_skill_ids=[])
        lin.validate()  # should not raise

    def test_validate_imported_with_parents_raises(self):
        lin = SkillLineage(origin=SkillOrigin.IMPORTED, parent_skill_ids=["x"])
        with pytest.raises(ValidationError):
            lin.validate()

    def test_validate_captured_no_parents(self):
        lin = SkillLineage(origin=SkillOrigin.CAPTURED, parent_skill_ids=[])
        lin.validate()

    def test_validate_captured_with_parents_raises(self):
        lin = SkillLineage(origin=SkillOrigin.CAPTURED, parent_skill_ids=["x"])
        with pytest.raises(ValidationError):
            lin.validate()

    def test_validate_fixed_one_parent(self):
        lin = SkillLineage(origin=SkillOrigin.FIXED, parent_skill_ids=["prev_v"])
        lin.validate()

    def test_validate_fixed_no_parent_raises(self):
        lin = SkillLineage(origin=SkillOrigin.FIXED, parent_skill_ids=[])
        with pytest.raises(ValidationError):
            lin.validate()

    def test_validate_fixed_multiple_parents_raises(self):
        lin = SkillLineage(origin=SkillOrigin.FIXED, parent_skill_ids=["a", "b"])
        with pytest.raises(ValidationError):
            lin.validate()

    def test_validate_derived_with_parents(self):
        lin = SkillLineage(origin=SkillOrigin.DERIVED, parent_skill_ids=["a", "b"])
        lin.validate()

    def test_validate_derived_no_parents_raises(self):
        lin = SkillLineage(origin=SkillOrigin.DERIVED, parent_skill_ids=[])
        with pytest.raises(ValidationError):
            lin.validate()

    def test_validate_negative_generation_raises(self):
        lin = SkillLineage(origin=SkillOrigin.IMPORTED, generation=-1)
        with pytest.raises(ValidationError):
            lin.validate()


# ═══════════════════════════════════════════════════════════════════════
#  SkillJudgment
# ═══════════════════════════════════════════════════════════════════════


class TestSkillJudgment:
    """SkillJudgment creation, round-trip, validation."""

    def test_defaults(self):
        j = SkillJudgment(skill_id="s1")
        assert j.skill_applied is False
        assert j.note == ""

    def test_round_trip(self):
        j = SkillJudgment(skill_id="s1", skill_applied=True, note="worked great")
        restored = SkillJudgment.from_dict(j.to_dict())
        assert restored.skill_id == j.skill_id
        assert restored.skill_applied == j.skill_applied
        assert restored.note == j.note

    def test_validate_empty_skill_id_raises(self):
        j = SkillJudgment(skill_id="")
        with pytest.raises(ValidationError):
            j.validate()

    def test_validate_ok(self):
        j = SkillJudgment(skill_id="skill_abc")
        j.validate()


# ═══════════════════════════════════════════════════════════════════════
#  EvolutionSuggestion
# ═══════════════════════════════════════════════════════════════════════


class TestEvolutionSuggestion:
    """EvolutionSuggestion round-trip, properties, and validation."""

    def test_round_trip(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.FIX,
            target_skill_ids=["skill_a"],
            category=SkillCategory.TOOL_GUIDE,
            direction="Fix curl params",
        )
        restored = EvolutionSuggestion.from_dict(s.to_dict())
        assert restored.evolution_type == s.evolution_type
        assert restored.target_skill_ids == s.target_skill_ids
        assert restored.category == s.category
        assert restored.direction == s.direction

    def test_target_skill_id_property(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.DERIVED,
            target_skill_ids=["a", "b"],
        )
        assert s.target_skill_id == "a"

    def test_target_skill_id_empty(self):
        s = EvolutionSuggestion(evolution_type=EvolutionType.CAPTURED)
        assert s.target_skill_id == ""

    def test_from_dict_legacy_single_target(self):
        """Legacy format has 'target_skill' instead of 'target_skills'."""
        d = {"type": "fix", "target_skill": "sk1", "direction": "fix it"}
        s = EvolutionSuggestion.from_dict(d)
        assert s.target_skill_ids == ["sk1"]

    def test_validate_fix_one_target(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.FIX,
            target_skill_ids=["skill_a"],
        )
        s.validate()

    def test_validate_fix_no_target_raises(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.FIX,
            target_skill_ids=[],
        )
        with pytest.raises(ValidationError):
            s.validate()

    def test_validate_fix_multiple_targets_raises(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.FIX,
            target_skill_ids=["a", "b"],
        )
        with pytest.raises(ValidationError):
            s.validate()

    def test_validate_derived_with_targets(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.DERIVED,
            target_skill_ids=["a"],
        )
        s.validate()

    def test_validate_derived_no_targets_raises(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.DERIVED,
            target_skill_ids=[],
        )
        with pytest.raises(ValidationError):
            s.validate()

    def test_validate_captured_no_targets(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.CAPTURED,
            target_skill_ids=[],
        )
        s.validate()

    def test_validate_captured_with_targets_raises(self):
        s = EvolutionSuggestion(
            evolution_type=EvolutionType.CAPTURED,
            target_skill_ids=["a"],
        )
        with pytest.raises(ValidationError):
            s.validate()


# ═══════════════════════════════════════════════════════════════════════
#  ExecutionAnalysis
# ═══════════════════════════════════════════════════════════════════════


class TestExecutionAnalysis:
    """ExecutionAnalysis round-trip, computed properties, validation."""

    @pytest.fixture
    def sample_analysis(self) -> ExecutionAnalysis:
        return ExecutionAnalysis(
            task_id="task_99",
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            task_completed=True,
            execution_note="All good",
            tool_issues=["curl"],
            skill_judgments=[
                SkillJudgment(skill_id="s1", skill_applied=True, note="ok"),
                SkillJudgment(skill_id="s2", skill_applied=False, note="skipped"),
            ],
            evolution_suggestions=[
                EvolutionSuggestion(
                    evolution_type=EvolutionType.FIX,
                    target_skill_ids=["s2"],
                    direction="Repair step 3",
                ),
            ],
            analyzed_by="gpt-4",
        )

    def test_round_trip(self, sample_analysis: ExecutionAnalysis):
        restored = ExecutionAnalysis.from_dict(sample_analysis.to_dict())
        assert restored.task_id == sample_analysis.task_id
        assert restored.task_completed == sample_analysis.task_completed
        assert restored.execution_note == sample_analysis.execution_note
        assert restored.tool_issues == sample_analysis.tool_issues
        assert len(restored.skill_judgments) == 2
        assert len(restored.evolution_suggestions) == 1

    def test_get_judgment_found(self, sample_analysis: ExecutionAnalysis):
        j = sample_analysis.get_judgment("s1")
        assert j is not None
        assert j.skill_applied is True

    def test_get_judgment_not_found(self, sample_analysis: ExecutionAnalysis):
        assert sample_analysis.get_judgment("nonexistent") is None

    def test_skill_ids(self, sample_analysis: ExecutionAnalysis):
        assert sample_analysis.skill_ids == ["s1", "s2"]

    def test_candidate_for_evolution(self, sample_analysis: ExecutionAnalysis):
        assert sample_analysis.candidate_for_evolution is True

    def test_not_candidate_for_evolution(self):
        a = ExecutionAnalysis(
            task_id="t1",
            timestamp=datetime.now(),
        )
        assert a.candidate_for_evolution is False

    def test_suggestions_by_type(self, sample_analysis: ExecutionAnalysis):
        fixes = sample_analysis.suggestions_by_type(EvolutionType.FIX)
        assert len(fixes) == 1
        assert fixes[0].direction == "Repair step 3"
        assert sample_analysis.suggestions_by_type(EvolutionType.DERIVED) == []

    def test_validate_empty_task_id_raises(self):
        a = ExecutionAnalysis(task_id="", timestamp=datetime.now())
        with pytest.raises(ValidationError):
            a.validate()

    def test_validate_ok(self):
        a = ExecutionAnalysis(task_id="t1", timestamp=datetime.now())
        a.validate()


# ═══════════════════════════════════════════════════════════════════════
#  SkillRecord
# ═══════════════════════════════════════════════════════════════════════


class TestSkillRecord:
    """SkillRecord creation, defaults, serialization, computed props, validation."""

    @pytest.fixture
    def sample_record(self) -> SkillRecord:
        return SkillRecord(
            skill_id="weather__imp_a1b2c3d4",
            name="weather_lookup",
            description="Look up weather for a city",
            path="/skills/weather_lookup/SKILL.md",
            category=SkillCategory.TOOL_GUIDE,
            tags=["weather", "api"],
            lineage=SkillLineage(origin=SkillOrigin.IMPORTED),
            tool_dependencies=["curl", "jq"],
            critical_tools=["curl"],
            total_selections=10,
            total_applied=8,
            total_completions=7,
            total_fallbacks=1,
        )

    # --- Defaults ---

    def test_defaults(self):
        r = SkillRecord(skill_id="s1", name="test", description="desc")
        assert r.is_active is True
        assert r.category == SkillCategory.WORKFLOW
        assert r.visibility == SkillVisibility.PRIVATE
        assert r.tags == []
        assert r.total_selections == 0
        assert r.recent_analyses == []

    # --- Computed properties ---

    def test_applied_rate(self, sample_record: SkillRecord):
        assert sample_record.applied_rate == pytest.approx(0.8)

    def test_completion_rate(self, sample_record: SkillRecord):
        assert sample_record.completion_rate == pytest.approx(7 / 8)

    def test_effective_rate(self, sample_record: SkillRecord):
        assert sample_record.effective_rate == pytest.approx(0.7)

    def test_fallback_rate(self, sample_record: SkillRecord):
        assert sample_record.fallback_rate == pytest.approx(0.1)

    def test_rates_zero_division(self):
        r = SkillRecord(skill_id="s1", name="n", description="d")
        assert r.applied_rate == 0.0
        assert r.completion_rate == 0.0
        assert r.effective_rate == 0.0
        assert r.fallback_rate == 0.0

    # --- Serialization ---

    def test_round_trip(self, sample_record: SkillRecord):
        d = sample_record.to_dict()
        restored = SkillRecord.from_dict(d)
        assert restored.skill_id == sample_record.skill_id
        assert restored.name == sample_record.name
        assert restored.description == sample_record.description
        assert restored.path == sample_record.path
        assert restored.category == sample_record.category
        assert restored.tags == sample_record.tags
        assert restored.visibility == sample_record.visibility
        assert restored.lineage.origin == sample_record.lineage.origin
        assert restored.tool_dependencies == sample_record.tool_dependencies
        assert restored.critical_tools == sample_record.critical_tools
        assert restored.total_selections == sample_record.total_selections
        assert restored.total_applied == sample_record.total_applied
        assert restored.total_completions == sample_record.total_completions
        assert restored.total_fallbacks == sample_record.total_fallbacks

    def test_from_dict_minimal(self):
        """Minimal dict should populate defaults."""
        d = {"skill_id": "s1", "name": "test"}
        r = SkillRecord.from_dict(d)
        assert r.skill_id == "s1"
        assert r.name == "test"
        assert r.description == ""
        assert r.is_active is True

    def test_to_dict_has_all_keys(self, sample_record: SkillRecord):
        d = sample_record.to_dict()
        expected_keys = {
            "skill_id", "name", "description", "path", "is_active",
            "category", "tags", "visibility", "creator_id", "lineage",
            "tool_dependencies", "critical_tools",
            "total_selections", "total_applied",
            "total_completions", "total_fallbacks",
            "recent_analyses", "first_seen", "last_updated",
        }
        assert set(d.keys()) == expected_keys

    # --- Validation ---

    def test_validate_ok(self, sample_record: SkillRecord):
        sample_record.validate()

    def test_validate_empty_skill_id_raises(self):
        r = SkillRecord(skill_id="", name="n", description="d")
        with pytest.raises(ValidationError, match="skill_id"):
            r.validate()

    def test_validate_empty_name_raises(self):
        r = SkillRecord(skill_id="s1", name="", description="d")
        with pytest.raises(ValidationError, match="name"):
            r.validate()

    def test_validate_negative_selections_raises(self):
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            total_selections=-1,
        )
        with pytest.raises(ValidationError, match="total_selections"):
            r.validate()

    def test_validate_negative_applied_raises(self):
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            total_applied=-1,
        )
        with pytest.raises(ValidationError, match="total_applied"):
            r.validate()

    def test_validate_negative_completions_raises(self):
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            total_completions=-1,
        )
        with pytest.raises(ValidationError, match="total_completions"):
            r.validate()

    def test_validate_negative_fallbacks_raises(self):
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            total_fallbacks=-1,
        )
        with pytest.raises(ValidationError, match="total_fallbacks"):
            r.validate()

    def test_validate_applied_exceeds_selections_raises(self):
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            total_selections=5, total_applied=10,
        )
        with pytest.raises(ValidationError, match="total_applied"):
            r.validate()

    def test_validate_completions_exceeds_applied_raises(self):
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            total_selections=10, total_applied=5, total_completions=8,
        )
        with pytest.raises(ValidationError, match="total_completions"):
            r.validate()

    def test_validate_cascades_to_lineage(self):
        """Validation should also validate nested lineage."""
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            lineage=SkillLineage(
                origin=SkillOrigin.FIXED,
                parent_skill_ids=[],  # Invalid: FIXED needs 1 parent
            ),
        )
        with pytest.raises(ValidationError):
            r.validate()


# ═══════════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_skill_record_with_analyses(self):
        """SkillRecord round-trip preserves nested analyses."""
        analysis = ExecutionAnalysis(
            task_id="t1",
            timestamp=datetime(2025, 6, 1),
            task_completed=True,
        )
        r = SkillRecord(
            skill_id="s1", name="n", description="d",
            recent_analyses=[analysis],
        )
        d = r.to_dict()
        restored = SkillRecord.from_dict(d)
        assert len(restored.recent_analyses) == 1
        assert restored.recent_analyses[0].task_id == "t1"

    def test_lineage_empty_content_snapshot(self):
        lin = SkillLineage(origin=SkillOrigin.IMPORTED, content_snapshot={})
        d = lin.to_dict()
        restored = SkillLineage.from_dict(d)
        assert restored.content_snapshot == {}

    def test_lineage_preserves_multiline_diff(self):
        diff = "--- a/SKILL.md\n+++ b/SKILL.md\n@@ -1 +1 @@\n-old\n+new\n"
        lin = SkillLineage(origin=SkillOrigin.FIXED, parent_skill_ids=["p1"], content_diff=diff)
        restored = SkillLineage.from_dict(lin.to_dict())
        assert restored.content_diff == diff

    def test_skill_version_zero(self):
        v = SkillVersion(0, 0, 0)
        assert str(v) == "0.0.0"

    def test_skill_version_large_numbers(self):
        v = SkillVersion(999, 999, 999)
        assert SkillVersion.parse(str(v)) == v

    def test_evolution_suggestion_from_dict_unknown_category(self):
        """Unknown category should not crash — just set None."""
        d = {"type": "fix", "target_skills": ["s1"], "category": "unknown_cat"}
        s = EvolutionSuggestion.from_dict(d)
        assert s.category is None

    def test_datetime_round_trip_precision(self):
        """ISO format preserves at least second precision."""
        dt = datetime(2025, 7, 10, 14, 30, 59)
        lin = SkillLineage(origin=SkillOrigin.IMPORTED, created_at=dt)
        restored = SkillLineage.from_dict(lin.to_dict())
        assert restored.created_at.year == 2025
        assert restored.created_at.second == 59


# ═══════════════════════════════════════════════════════════════════════
#  Hostile / adversarial input tests (F5)
# ═══════════════════════════════════════════════════════════════════════


class TestFromDictHostileInput:
    """from_dict() with wrong types, partial payloads, and adversarial values."""

    # --- Wrong types: int where string expected ---

    def test_skill_record_numeric_skill_id_coerced(self):
        """Numeric skill_id should be coerced to str."""
        d = {"skill_id": 12345, "name": "test"}
        r = SkillRecord.from_dict(d)
        assert r.skill_id == "12345"
        assert isinstance(r.skill_id, str)

    def test_skill_record_numeric_name_coerced(self):
        """Numeric name should be coerced to str."""
        d = {"skill_id": "s1", "name": 42}
        r = SkillRecord.from_dict(d)
        assert r.name == "42"
        assert isinstance(r.name, str)

    def test_execution_analysis_numeric_task_id_coerced(self):
        """Numeric task_id should be coerced to str."""
        d = {"task_id": 999, "timestamp": "2025-01-01T00:00:00"}
        a = ExecutionAnalysis.from_dict(d)
        assert a.task_id == "999"
        assert isinstance(a.task_id, str)

    def test_skill_version_string_components_coerced(self):
        """String version components should be coerced to int."""
        d = {"major": "2", "minor": "3", "patch": "4"}
        v = SkillVersion.from_dict(d)
        assert v == SkillVersion(2, 3, 4)

    def test_skill_record_string_counters_coerced(self):
        """String counter values should be coerced to int."""
        d = {
            "skill_id": "s1",
            "name": "test",
            "total_selections": "10",
            "total_applied": "8",
            "total_completions": "7",
            "total_fallbacks": "1",
        }
        r = SkillRecord.from_dict(d)
        assert r.total_selections == 10
        assert isinstance(r.total_selections, int)
        assert r.total_applied == 8
        assert r.total_completions == 7
        assert r.total_fallbacks == 1

    # --- Partial payloads: missing keys ---

    def test_skill_version_from_dict_empty(self):
        """Empty dict should yield 0.0.0."""
        v = SkillVersion.from_dict({})
        assert v == SkillVersion(0, 0, 0)

    def test_skill_version_from_dict_major_only(self):
        """Only major → minor and patch default to 0."""
        v = SkillVersion.from_dict({"major": 2})
        assert v == SkillVersion(2, 0, 0)

    def test_skill_record_from_dict_minimal(self):
        """Only required keys; all optional fields get safe defaults."""
        d = {"skill_id": "s1", "name": "n"}
        r = SkillRecord.from_dict(d)
        assert r.total_selections == 0
        assert r.total_applied == 0
        assert r.total_completions == 0
        assert r.total_fallbacks == 0
        assert r.description == ""
        assert r.is_active is True

    def test_execution_analysis_from_dict_minimal(self):
        """Only required keys for ExecutionAnalysis."""
        d = {"task_id": "t1", "timestamp": "2025-01-01T00:00:00"}
        a = ExecutionAnalysis.from_dict(d)
        assert a.task_id == "t1"
        assert a.skill_judgments == []
        assert a.evolution_suggestions == []

    # --- Adversarial values ---

    def test_skill_record_empty_string_skill_id(self):
        """Empty string skill_id should be coerced to str, validation catches it."""
        d = {"skill_id": "", "name": "test"}
        r = SkillRecord.from_dict(d)
        assert r.skill_id == ""
        with pytest.raises(ValidationError, match="skill_id"):
            r.validate()

    def test_skill_record_very_long_name(self):
        """Very long name should still round-trip without crash."""
        long_name = "x" * 10_000
        d = {"skill_id": "s1", "name": long_name}
        r = SkillRecord.from_dict(d)
        assert r.name == long_name
        assert len(r.name) == 10_000

    def test_skill_version_non_numeric_raises(self):
        """Non-numeric version components should raise ValidationError."""
        with pytest.raises(ValidationError):
            SkillVersion.from_dict({"major": "abc"})

    def test_skill_record_non_numeric_counter_raises(self):
        """Non-numeric counter should raise ValidationError."""
        with pytest.raises(ValidationError):
            SkillRecord.from_dict({
                "skill_id": "s1",
                "name": "test",
                "total_selections": "not_a_number",
            })

    # --- Type coercion correctness ---

    def test_skill_version_float_coerced(self):
        """Float version components should be truncated to int."""
        d = {"major": 1.9, "minor": 2.1, "patch": 0.0}
        v = SkillVersion.from_dict(d)
        assert v == SkillVersion(1, 2, 0)

    def test_skill_record_float_counters_coerced(self):
        """Float counter values should be truncated to int."""
        d = {
            "skill_id": "s1",
            "name": "test",
            "total_selections": 5.7,
            "total_applied": 3.2,
        }
        r = SkillRecord.from_dict(d)
        assert r.total_selections == 5
        assert r.total_applied == 3
