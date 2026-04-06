"""Formatting helpers for evolution prompts.

Pure functions that transform skill snapshots and execution analyses
into text blocks suitable for LLM prompt inclusion.
"""

from __future__ import annotations

from typing import List

from openspace.utils.logging import Logger

from ..patch import SKILL_FILENAME, collect_skill_snapshot
from ..types import ExecutionAnalysis
from .triggers import _ANALYSIS_CONTEXT_MAX

logger = Logger.get_logger("openspace.skill_engine.evolver")


def format_skill_dir_content(skill_dir) -> str:
    """Format all text files in a skill directory for prompt inclusion.

    Returns a multi-file listing if there are auxiliary files beyond
    SKILL.md, or just the SKILL.md content for single-file skills.
    """
    files = collect_skill_snapshot(skill_dir)
    if not files:
        return ""

    # Single-file skill: return just the content
    if len(files) == 1 and SKILL_FILENAME in files:
        return files[SKILL_FILENAME]

    # Multi-file: format as directory listing
    parts: list[str] = []
    # SKILL.md first
    if SKILL_FILENAME in files:
        parts.append(f"### File: {SKILL_FILENAME}\n```markdown\n{files[SKILL_FILENAME]}\n```")
    for name, content in sorted(files.items()):
        if name == SKILL_FILENAME:
            continue
        parts.append(f"### File: {name}\n```\n{content}\n```")

    return "\n\n".join(parts)


def format_analysis_context(analyses: List[ExecutionAnalysis]) -> str:
    """Format recent analyses into a concise context block for prompts."""
    _ANALYSIS_NOTE_MAX_CHARS = 500  # Per-analysis note truncation

    if not analyses:
        return "(no execution history available)"

    parts: List[str] = []
    for a in analyses[:_ANALYSIS_CONTEXT_MAX]:
        completed = "completed" if a.task_completed else "failed"

        # Per-skill notes
        skill_notes = []
        for j in a.skill_judgments:
            applied = "applied" if j.skill_applied else "NOT applied"
            note = f"  - {j.skill_id}: {applied}"
            if j.note:
                note += f" — {j.note[:_ANALYSIS_NOTE_MAX_CHARS]}"
            skill_notes.append(note)

        # Tool issues
        tool_lines = []
        for issue in a.tool_issues[:3]:
            tool_lines.append(f"  - {issue[:200]}")

        block = f"### Task: {a.task_id} ({completed})\n"
        if a.execution_note:
            block += f"{a.execution_note[:_ANALYSIS_NOTE_MAX_CHARS]}\n"
        if skill_notes:
            block += "Skills:\n" + "\n".join(skill_notes) + "\n"
        if tool_lines:
            block += "Tool issues:\n" + "\n".join(tool_lines) + "\n"
        parts.append(block)

    return "\n".join(parts)
