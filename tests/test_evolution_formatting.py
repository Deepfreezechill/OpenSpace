"""Tests for openspace.skill_engine.evolution.formatting (Epic 5.6).

Verifies:
  1. format_skill_dir_content — single-file, multi-file, empty
  2. format_analysis_context — with analyses, empty, truncation
  3. Backward compat: SkillEvolver delegates to formatting functions
  4. Logger namespace
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openspace.skill_engine.evolution.formatting import (
    format_analysis_context,
    format_skill_dir_content,
)


# ---------------------------------------------------------------------------
# format_skill_dir_content
# ---------------------------------------------------------------------------

class TestFormatSkillDirContent:
    def test_empty_dir(self):
        """Empty directory → empty string."""
        with patch("openspace.skill_engine.evolution.formatting.collect_skill_snapshot", return_value={}):
            result = format_skill_dir_content(Path("/fake"))
        assert result == ""

    def test_single_file_skill(self):
        """Single SKILL.md → return content directly (no formatting)."""
        with patch(
            "openspace.skill_engine.evolution.formatting.collect_skill_snapshot",
            return_value={"SKILL.md": "---\nname: test\n---\ncontent"},
        ):
            result = format_skill_dir_content(Path("/fake"))
        assert result == "---\nname: test\n---\ncontent"

    def test_multi_file_skill(self):
        """Multiple files → formatted listing with SKILL.md first."""
        files = {
            "SKILL.md": "skill content",
            "helper.py": "def foo(): pass",
            "README.md": "readme text",
        }
        with patch(
            "openspace.skill_engine.evolution.formatting.collect_skill_snapshot",
            return_value=files,
        ):
            result = format_skill_dir_content(Path("/fake"))

        # SKILL.md appears first
        assert result.index("### File: SKILL.md") < result.index("### File: README.md")
        assert result.index("### File: SKILL.md") < result.index("### File: helper.py")
        # All files present
        assert "skill content" in result
        assert "def foo(): pass" in result
        assert "readme text" in result

    def test_multi_file_without_skill_md(self):
        """Multi-file with no SKILL.md → still formats all files."""
        files = {"helper.py": "code", "config.yaml": "key: val"}
        with patch(
            "openspace.skill_engine.evolution.formatting.collect_skill_snapshot",
            return_value=files,
        ):
            result = format_skill_dir_content(Path("/fake"))
        assert "### File: config.yaml" in result
        assert "### File: helper.py" in result


# ---------------------------------------------------------------------------
# format_analysis_context
# ---------------------------------------------------------------------------

class _FakeSkillJudgment:
    def __init__(self, skill_id, applied=True, note=""):
        self.skill_id = skill_id
        self.skill_applied = applied
        self.note = note


class _FakeAnalysis:
    def __init__(self, task_id, completed=True, note="", judgments=None, tool_issues=None):
        self.task_id = task_id
        self.task_completed = completed
        self.execution_note = note
        self.skill_judgments = judgments or []
        self.tool_issues = tool_issues or []


class TestFormatAnalysisContext:
    def test_empty_analyses(self):
        """No analyses → fallback message."""
        result = format_analysis_context([])
        assert "no execution history" in result.lower()

    def test_single_analysis(self):
        """Single analysis → formatted block."""
        analysis = _FakeAnalysis(
            task_id="task-1",
            completed=True,
            note="Skill worked well",
            judgments=[_FakeSkillJudgment("skill-a", applied=True, note="Good result")],
        )
        result = format_analysis_context([analysis])
        assert "task-1" in result
        assert "completed" in result
        assert "skill-a" in result
        assert "applied" in result
        assert "Good result" in result

    def test_failed_task(self):
        """Failed task shows 'failed' label."""
        analysis = _FakeAnalysis(task_id="task-2", completed=False)
        result = format_analysis_context([analysis])
        assert "failed" in result

    def test_tool_issues_included(self):
        """Tool issues appear in output."""
        analysis = _FakeAnalysis(
            task_id="task-3",
            tool_issues=["Tool X timed out", "Tool Y returned error"],
        )
        result = format_analysis_context([analysis])
        assert "Tool X timed out" in result
        assert "Tool Y returned error" in result

    def test_note_truncation(self):
        """Execution notes are truncated to 500 chars."""
        long_note = "x" * 1000
        analysis = _FakeAnalysis(task_id="task-4", note=long_note)
        result = format_analysis_context([analysis])
        # The note should be truncated — full 1000 chars should NOT appear
        assert "x" * 501 not in result

    def test_not_applied_skill(self):
        """Skill not applied shows 'NOT applied'."""
        analysis = _FakeAnalysis(
            task_id="task-5",
            judgments=[_FakeSkillJudgment("skill-b", applied=False, note="Didn't match")],
        )
        result = format_analysis_context([analysis])
        assert "NOT applied" in result


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

class TestDelegationSeam:
    def test_format_skill_dir_is_staticmethod(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert isinstance(
            SkillEvolver.__dict__["_format_skill_dir_content"],
            staticmethod,
        )

    def test_format_analysis_is_staticmethod(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert isinstance(
            SkillEvolver.__dict__["_format_analysis_context"],
            staticmethod,
        )

    def test_format_skill_dir_callable(self):
        from openspace.skill_engine.evolver import SkillEvolver
        with patch(
            "openspace.skill_engine.evolution.formatting.collect_skill_snapshot",
            return_value={"SKILL.md": "test"},
        ):
            result = SkillEvolver._format_skill_dir_content(Path("/fake"))
        assert result == "test"


# ---------------------------------------------------------------------------
# Constants / logger
# ---------------------------------------------------------------------------

class TestMeta:
    def test_module_size(self):
        """formatting.py should stay under 120 lines."""
        import openspace.skill_engine.evolution.formatting as mod
        src = Path(mod.__file__)
        lines = src.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 120, f"formatting.py has {len(lines)} lines (limit 120)"
