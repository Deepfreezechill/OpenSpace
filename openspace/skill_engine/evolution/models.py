"""Evolution domain models — pure data types and utilities.

Contains the core evolution types used across the evolution subsystem:
  - EvolutionTrigger: enum of what initiated evolution
  - EvolutionContext: unified context dataclass for all trigger sources
  - _sanitize_skill_name: naming-rule enforcer for skill directory names
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from openspace.skill_engine.types import (
    EvolutionSuggestion,
    ExecutionAnalysis,
    SkillRecord,
)

if TYPE_CHECKING:
    from openspace.grounding.core.tool import BaseTool

_MAX_SKILL_NAME_LENGTH = 50  # Max chars for a skill name (directory name)


def _sanitize_skill_name(name: str) -> str:
    """Enforce naming rules for skill names (used as directory names).

    - Lowercase, hyphens only (no underscores or special chars)
    - Truncate to ``_MAX_SKILL_NAME_LENGTH`` at a word boundary
    - Remove trailing hyphens
    """
    # Normalize: lowercase, replace underscores and spaces with hyphens
    clean = re.sub(r"[^a-z0-9\-]", "-", name.lower().strip())
    # Collapse multiple hyphens
    clean = re.sub(r"-{2,}", "-", clean).strip("-")

    if len(clean) <= _MAX_SKILL_NAME_LENGTH:
        return clean

    # Truncate at a hyphen boundary to avoid cutting words
    truncated = clean[:_MAX_SKILL_NAME_LENGTH]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > _MAX_SKILL_NAME_LENGTH // 2:
        truncated = truncated[:last_hyphen]
    return truncated.strip("-")


class EvolutionTrigger(str, Enum):
    """What initiated this evolution."""

    ANALYSIS = "analysis"  # Post-execution analysis suggestion
    TOOL_DEGRADATION = "tool_degradation"  # Tool quality degradation detected
    METRIC_MONITOR = "metric_monitor"  # Periodic skill health check


@dataclass
class EvolutionContext:
    """Unified context for all evolution triggers.

    For trigger 1 (ANALYSIS): source_task_id is set, recent_analyses may be
    just the single triggering analysis.
    For triggers 2/3: source_task_id is None, recent_analyses are loaded
    from the skill's historical records.
    """

    trigger: EvolutionTrigger
    suggestion: EvolutionSuggestion

    # Parent skill context
    skill_records: List[SkillRecord] = field(default_factory=list)
    skill_contents: List[str] = field(default_factory=list)
    skill_dirs: List[Path] = field(default_factory=list)

    # Task context
    source_task_id: Optional[str] = None
    recent_analyses: List[ExecutionAnalysis] = field(default_factory=list)

    # Trigger-specific context
    tool_issue_summary: str = ""  # For TOOL_DEGRADATION
    metric_summary: str = ""  # For METRIC_MONITOR

    # Available tools for agent loop (read_file, web_search, shell, MCP, etc.)
    available_tools: List["BaseTool"] = field(default_factory=list)
