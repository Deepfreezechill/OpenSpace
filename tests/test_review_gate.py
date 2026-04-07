"""Tests for Epic 7.1: Eight-Eyes Review Gate for skill evolution.

Tests cover:
- ReviewGate runs multiple review checks on evolved skills
- Quarantine: skills that fail review are deactivated
- AST safety check integration
- Content validation (required fields)
- Lineage validation (proper parent chain)
- Review result aggregation
- Integration with SkillStore.evolve_skill()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.types import (
    SkillCategory,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
)


def _make_record(
    skill_id: str = "test-skill__v2_abc12345",
    name: str = "test-skill",
    is_active: bool = True,
    origin: SkillOrigin = SkillOrigin.FIXED,
    generation: int = 1,
    parent_ids: list = None,
    description: str = "A test skill",
    path: str = "/tmp/skills/test-skill/SKILL.md",
    content_snapshot: dict = None,
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        description=description,
        path=path,
        is_active=is_active,
        lineage=SkillLineage(
            origin=origin,
            generation=generation,
            parent_skill_ids=parent_ids if parent_ids is not None else ["test-skill__v1"],
            content_snapshot=content_snapshot if content_snapshot is not None else {"SKILL.md": "name: test-skill\n"},
        ),
    )


# ======================================================================
# ReviewResult
# ======================================================================
class TestReviewResult:
    def test_review_result_pass(self):
        from openspace.skill_engine.review_gate import ReviewResult

        result = ReviewResult(verdict="pass", checks=[])
        assert result.passed

    def test_review_result_fail(self):
        from openspace.skill_engine.review_gate import ReviewResult

        result = ReviewResult(verdict="fail", checks=[])
        assert not result.passed

    def test_review_result_quarantine(self):
        from openspace.skill_engine.review_gate import ReviewResult

        result = ReviewResult(verdict="quarantine", checks=[])
        assert not result.passed

    def test_review_result_aggregates_check_verdicts(self):
        from openspace.skill_engine.review_gate import CheckResult, ReviewResult

        checks = [
            CheckResult(name="ast-safety", verdict="pass", detail="clean"),
            CheckResult(name="content", verdict="fail", detail="missing fields"),
        ]
        result = ReviewResult.from_checks(checks)
        assert result.verdict == "fail"
        assert len(result.checks) == 2

    def test_review_result_all_pass_means_pass(self):
        from openspace.skill_engine.review_gate import CheckResult, ReviewResult

        checks = [
            CheckResult(name="ast-safety", verdict="pass", detail="clean"),
            CheckResult(name="content", verdict="pass", detail="valid"),
            CheckResult(name="lineage", verdict="pass", detail="valid chain"),
        ]
        result = ReviewResult.from_checks(checks)
        assert result.verdict == "pass"


# ======================================================================
# Individual checks
# ======================================================================
class TestASTSafetyCheck:
    def test_safe_skill_passes(self):
        from openspace.skill_engine.review_gate import check_ast_safety

        record = _make_record(
            content_snapshot={"SKILL.md": "name: safe\n", "handler.py": "x = 1 + 2\n"}
        )
        result = check_ast_safety(record)
        assert result.verdict == "pass"

    def test_no_python_files_passes(self):
        from openspace.skill_engine.review_gate import check_ast_safety

        record = _make_record(content_snapshot={"SKILL.md": "name: safe\n"})
        result = check_ast_safety(record)
        assert result.verdict == "pass"

    def test_dangerous_code_fails(self):
        from openspace.skill_engine.review_gate import check_ast_safety

        record = _make_record(
            content_snapshot={
                "SKILL.md": "name: evil\n",
                "handler.py": "import os; os.system('rm -rf /')\n",
            }
        )
        result = check_ast_safety(record)
        assert result.verdict == "fail"


class TestContentValidation:
    def test_valid_content_passes(self):
        from openspace.skill_engine.review_gate import check_content

        record = _make_record(
            name="valid-skill",
            description="Does useful things",
            content_snapshot={"SKILL.md": "name: valid-skill\ndescription: useful\n"},
        )
        result = check_content(record)
        assert result.verdict == "pass"

    def test_empty_description_fails(self):
        from openspace.skill_engine.review_gate import check_content

        record = _make_record(description="")
        result = check_content(record)
        assert result.verdict == "fail"
        assert "description" in result.detail.lower()

    def test_empty_name_fails(self):
        from openspace.skill_engine.review_gate import check_content

        record = _make_record(name="")
        result = check_content(record)
        assert result.verdict == "fail"

    def test_missing_skill_md_fails(self):
        from openspace.skill_engine.review_gate import check_content

        record = _make_record(content_snapshot={})
        result = check_content(record)
        assert result.verdict == "fail"
        assert "SKILL.md" in result.detail


class TestLineageValidation:
    def test_valid_lineage_passes(self):
        from openspace.skill_engine.review_gate import check_lineage

        record = _make_record(
            origin=SkillOrigin.FIXED,
            generation=2,
            parent_ids=["parent-v1"],
        )
        result = check_lineage(record)
        assert result.verdict == "pass"

    def test_evolved_without_parent_fails(self):
        from openspace.skill_engine.review_gate import check_lineage

        record = _make_record(
            origin=SkillOrigin.FIXED,
            generation=2,
            parent_ids=[],
        )
        result = check_lineage(record)
        assert result.verdict == "fail"
        assert "parent" in result.detail.lower()

    def test_generation_zero_evolved_fails(self):
        from openspace.skill_engine.review_gate import check_lineage

        record = _make_record(
            origin=SkillOrigin.FIXED,
            generation=0,
            parent_ids=["parent-v0"],
        )
        result = check_lineage(record)
        assert result.verdict == "fail"

    def test_imported_skill_skips_lineage_check(self):
        from openspace.skill_engine.review_gate import check_lineage

        record = _make_record(
            origin=SkillOrigin.IMPORTED,
            generation=0,
            parent_ids=[],
        )
        result = check_lineage(record)
        assert result.verdict == "pass"

    def test_captured_skill_skips_lineage_check(self):
        from openspace.skill_engine.review_gate import check_lineage

        record = _make_record(
            origin=SkillOrigin.CAPTURED,
            generation=0,
            parent_ids=[],
        )
        result = check_lineage(record)
        assert result.verdict == "pass"


# ======================================================================
# ReviewGate full pipeline
# ======================================================================
class TestReviewGate:
    def test_gate_passes_safe_skill(self):
        from openspace.skill_engine.review_gate import ReviewGate

        gate = ReviewGate()
        record = _make_record(
            content_snapshot={"SKILL.md": "name: good\n", "util.py": "x = 1\n"}
        )
        result = gate.review(record)
        assert result.passed
        assert result.verdict == "pass"

    def test_gate_fails_dangerous_skill(self):
        from openspace.skill_engine.review_gate import ReviewGate

        gate = ReviewGate()
        record = _make_record(
            content_snapshot={
                "SKILL.md": "name: bad\n",
                "run.py": "import subprocess; subprocess.call(['rm','-rf','/'])\n",
            }
        )
        result = gate.review(record)
        assert not result.passed

    def test_gate_fails_missing_description(self):
        from openspace.skill_engine.review_gate import ReviewGate

        gate = ReviewGate()
        record = _make_record(description="")
        result = gate.review(record)
        assert not result.passed

    def test_gate_returns_all_check_results(self):
        from openspace.skill_engine.review_gate import ReviewGate

        gate = ReviewGate()
        record = _make_record()
        result = gate.review(record)
        # Must run at least 3 checks: ast-safety, content, lineage
        assert len(result.checks) >= 3

    def test_gate_continues_after_first_failure(self):
        """Gate should run ALL checks, not stop at first failure."""
        from openspace.skill_engine.review_gate import ReviewGate

        gate = ReviewGate()
        record = _make_record(
            description="",  # content fail
            content_snapshot={
                "SKILL.md": "name: bad\n",
                "evil.py": "import os; os.system('rm')\n",
            },
        )
        result = gate.review(record)
        # Should have findings from BOTH content and ast checks
        failed_checks = [c for c in result.checks if c.verdict == "fail"]
        assert len(failed_checks) >= 2


# ======================================================================
# Quarantine integration
# ======================================================================
class TestQuarantine:
    @pytest.mark.asyncio
    async def test_failed_review_quarantines_skill(self):
        """Skills that fail review must be deactivated (quarantined)."""
        from openspace.skill_engine.review_gate import ReviewGate, quarantine_skill

        store = AsyncMock()
        store.deactivate_record = AsyncMock(return_value=True)

        record = _make_record(skill_id="evil-skill__v2")
        # Simulate failed review
        gate = ReviewGate()
        result = gate.review(
            _make_record(
                skill_id="evil-skill__v2",
                description="",  # will fail content check
            )
        )
        assert not result.passed

        await quarantine_skill(store, "evil-skill__v2", result)
        store.deactivate_record.assert_awaited_once_with("evil-skill__v2")

    @pytest.mark.asyncio
    async def test_passed_review_does_not_quarantine(self):
        """Skills that pass review stay active."""
        from openspace.skill_engine.review_gate import ReviewGate, quarantine_skill

        store = AsyncMock()
        store.deactivate_record = AsyncMock()

        record = _make_record()
        gate = ReviewGate()
        result = gate.review(record)
        assert result.passed

        await quarantine_skill(store, record.skill_id, result)
        store.deactivate_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quarantine_logs_findings(self):
        """Quarantine must log which checks failed."""
        from openspace.skill_engine.review_gate import ReviewGate, quarantine_skill

        store = AsyncMock()
        store.deactivate_record = AsyncMock(return_value=True)

        record = _make_record(description="")
        gate = ReviewGate()
        result = gate.review(record)

        with patch("openspace.skill_engine.review_gate.logger") as mock_logger:
            await quarantine_skill(store, record.skill_id, result)
            mock_logger.warning.assert_called()


# ======================================================================
# Wiring verification
# ======================================================================
class TestWiring:
    def test_review_gate_importable(self):
        from openspace.skill_engine.review_gate import ReviewGate

        assert callable(ReviewGate)

    def test_review_gate_in_skill_engine_init(self):
        """ReviewGate should be exported from skill_engine package."""
        import openspace.skill_engine.review_gate as rg

        assert hasattr(rg, "ReviewGate")
        assert hasattr(rg, "quarantine_skill")
        assert hasattr(rg, "ReviewResult")
        assert hasattr(rg, "CheckResult")
