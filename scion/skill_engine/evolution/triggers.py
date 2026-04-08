"""Evolution trigger functions — analysis, tool degradation, metric monitor.

Functions extracted from ``SkillEvolver`` (Epic 5.3):
- ``process_analysis``           — Trigger 1: post-analysis evolution
- ``process_tool_degradation``   — Trigger 2: fix skills for degraded tools
- ``process_metric_check``       — Trigger 3: periodic health-based evolution
- ``build_context_from_analysis`` — Build EvolutionContext from analyzer output
- ``load_skill_content``         — Load SKILL.md content (registry or disk)
- ``diagnose_skill_health``      — Pure metric classifier for Trigger 3
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from scion.utils.logging import Logger

from .models import EvolutionContext, EvolutionTrigger

from scion.skill_engine.types import EvolutionSuggestion, EvolutionType

if TYPE_CHECKING:
    from scion.grounding.core.quality.types import ToolQualityRecord
    from scion.skill_engine.types import (
        ExecutionAnalysis,
        SkillRecord,
    )

logger = Logger.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (moved from evolver.py — only used by trigger functions)
# ---------------------------------------------------------------------------

_ANALYSIS_CONTEXT_MAX = 5  # Max recent analyses to include in prompt

# Rule-based thresholds for candidate screening (relaxed — LLM confirms)
_FALLBACK_THRESHOLD = 0.4
_LOW_COMPLETION_THRESHOLD = 0.35
_HIGH_APPLIED_FOR_FIX = 0.4
_MODERATE_EFFECTIVE_THRESHOLD = 0.55
_MIN_APPLIED_FOR_DERIVED = 0.25


# ---------------------------------------------------------------------------
# Trigger 1: post-analysis
# ---------------------------------------------------------------------------

async def process_analysis(
    evolver,
    analysis: "ExecutionAnalysis",
) -> List["SkillRecord"]:
    """Process all evolution suggestions from a completed analysis.

    Called immediately after ``ExecutionAnalyzer.analyze_execution()``.
    Each suggestion becomes one evolution action, executed in parallel
    (throttled by semaphore).
    """
    if not analysis.candidate_for_evolution:
        return []

    contexts: List[EvolutionContext] = []
    for suggestion in analysis.evolution_suggestions:
        ctx = evolver._build_context_from_analysis(analysis, suggestion)
        if ctx is not None:
            contexts.append(ctx)

    if not contexts:
        return []

    results = await evolver._execute_contexts(contexts, "analysis")

    if results:
        names = [r.name for r in results]
        logger.info(
            "[Trigger:analysis] Evolved %d skill(s): %s from task %s",
            len(results), names, analysis.task_id,
        )
    return results


# ---------------------------------------------------------------------------
# Trigger 2: tool degradation
# ---------------------------------------------------------------------------

async def process_tool_degradation(
    evolver,
    problematic_tools: List["ToolQualityRecord"],
) -> List["SkillRecord"]:
    """Fix skills that depend on degraded tools.

    Two-phase: rule-based candidate screening → LLM confirmation.

    Anti-loop (state-driven):
      ``evolver._addressed_degradations[tool_key]`` records skill names
      already evolved for that tool's degradation.  Recovered tools are
      pruned so future re-degradation gets a fresh pass.
    """
    if not problematic_tools:
        return []

    # Prune recovered tools
    current_tool_keys = {t.tool_key for t in problematic_tools}
    recovered = [k for k in evolver._addressed_degradations if k not in current_tool_keys]
    for k in recovered:
        logger.debug("[Trigger:tool_degradation] Tool '%s' recovered, clearing addressed set", k)
        del evolver._addressed_degradations[k]

    # Phase 1: screen & confirm candidates
    confirmed_contexts: List[EvolutionContext] = []
    seen_skills: set = set()

    for tool_rec in problematic_tools:
        addressed = evolver._addressed_degradations.get(tool_rec.tool_key, set())

        skill_ids = evolver._store.find_skills_by_tool(tool_rec.tool_key)
        for skill_id in skill_ids:
            skill_record = evolver._store.load_record(skill_id)
            if not skill_record or not skill_record.is_active:
                continue

            if skill_record.skill_id in seen_skills:
                continue
            seen_skills.add(skill_record.skill_id)

            if skill_record.skill_id in addressed:
                logger.debug(
                    "[Trigger:tool_degradation] Skipping '%s' "
                    "(already addressed for tool '%s')",
                    skill_record.skill_id, tool_rec.tool_key,
                )
                continue

            recent = evolver._store.load_analyses(
                skill_id=skill_record.skill_id, limit=_ANALYSIS_CONTEXT_MAX,
            )
            content = evolver._load_skill_content(skill_record)
            if not content:
                continue

            issue_summary = (
                f"Tool `{tool_rec.tool_key}` degraded — "
                f"recent success rate: {tool_rec.recent_success_rate:.0%}, "
                f"total calls: {tool_rec.total_calls}, "
                f"LLM flagged: {tool_rec.llm_flagged_count} time(s)."
            )

            direction = (
                f"Tool `{tool_rec.tool_key}` has degraded "
                f"(success_rate={tool_rec.recent_success_rate:.0%}). "
                f"Update skill instructions to handle this tool's "
                f"failures gracefully or suggest alternatives."
            )

            confirmed = await evolver._llm_confirm_evolution(
                skill_record=skill_record,
                skill_content=content,
                proposed_type=EvolutionType.FIX,
                proposed_direction=direction,
                trigger_context=f"Tool degradation: {issue_summary}",
                recent_analyses=recent,
            )
            if not confirmed:
                logger.debug(
                    "[Trigger:tool_degradation] LLM rejected evolution "
                    "for skill '%s' (tool=%s)",
                    skill_record.skill_id, tool_rec.tool_key,
                )
                evolver._addressed_degradations.setdefault(
                    tool_rec.tool_key, set(),
                ).add(skill_record.skill_id)
                continue

            skill_dir = Path(skill_record.path).parent if skill_record.path else None
            confirmed_contexts.append(
                EvolutionContext(
                    trigger=EvolutionTrigger.TOOL_DEGRADATION,
                    suggestion=EvolutionSuggestion(
                        evolution_type=EvolutionType.FIX,
                        target_skill_ids=[skill_record.skill_id],
                        direction=direction,
                    ),
                    skill_records=[skill_record],
                    skill_contents=[content],
                    skill_dirs=[skill_dir] if skill_dir else [],
                    recent_analyses=recent,
                    tool_issue_summary=issue_summary,
                    available_tools=evolver._available_tools,
                )
            )

            evolver._addressed_degradations.setdefault(
                tool_rec.tool_key, set(),
            ).add(skill_record.skill_id)

    if not confirmed_contexts:
        return []

    return await evolver._execute_contexts(confirmed_contexts, "tool_degradation")


# ---------------------------------------------------------------------------
# Trigger 3: metric monitor
# ---------------------------------------------------------------------------

async def process_metric_check(
    evolver,
    min_selections: int = 5,
) -> List["SkillRecord"]:
    """Scan active skills and evolve those with poor health metrics.

    Two-phase: rule-based candidate screening (relaxed thresholds) →
    LLM confirmation.  Only considers skills with enough data.

    Anti-loop (data-driven): newly-evolved skills start with
    ``total_selections=0``, needing ``min_selections`` fresh executions.
    """
    confirmed_contexts: List[EvolutionContext] = []
    all_active = evolver._store.load_active()

    for skill_id, record in all_active.items():
        if record.total_selections < min_selections:
            continue

        evo_type, direction = evolver._diagnose_skill_health(record)
        if evo_type is None:
            continue

        content = evolver._load_skill_content(record)
        if not content:
            continue

        recent = evolver._store.load_analyses(
            skill_id=record.skill_id, limit=_ANALYSIS_CONTEXT_MAX,
        )
        metric_summary = (
            f"selections={record.total_selections}, "
            f"applied_rate={record.applied_rate:.0%}, "
            f"completion_rate={record.completion_rate:.0%}, "
            f"effective_rate={record.effective_rate:.0%}, "
            f"fallback_rate={record.fallback_rate:.0%}"
        )

        confirmed = await evolver._llm_confirm_evolution(
            skill_record=record,
            skill_content=content,
            proposed_type=evo_type,
            proposed_direction=direction,
            trigger_context=f"Metric check: {metric_summary}",
            recent_analyses=recent,
        )
        if not confirmed:
            logger.debug(
                "[Trigger:metric_monitor] LLM rejected evolution for skill '%s' (%s)",
                record.name, evo_type.value,
            )
            continue

        skill_dir = Path(record.path).parent if record.path else None
        confirmed_contexts.append(
            EvolutionContext(
                trigger=EvolutionTrigger.METRIC_MONITOR,
                suggestion=EvolutionSuggestion(
                    evolution_type=evo_type,
                    target_skill_ids=[record.skill_id],
                    direction=direction,
                ),
                skill_records=[record],
                skill_contents=[content],
                skill_dirs=[skill_dir] if skill_dir else [],
                recent_analyses=recent,
                metric_summary=metric_summary,
                available_tools=evolver._available_tools,
            )
        )

    if not confirmed_contexts:
        return []

    return await evolver._execute_contexts(confirmed_contexts, "metric_monitor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_context_from_analysis(
    evolver,
    analysis: "ExecutionAnalysis",
    suggestion: "EvolutionSuggestion",
) -> Optional[EvolutionContext]:
    """Build EvolutionContext from a single analysis suggestion.

    Loads all target skills referenced by ``suggestion.target_skill_ids``.
    For FIX: exactly 1 parent required.
    For DERIVED: 1+ parents (multi-parent = merge).
    For CAPTURED: parents list is empty.
    """
    records: List["SkillRecord"] = []
    contents: List[str] = []
    dirs: List[Path] = []

    if suggestion.evolution_type in (EvolutionType.FIX, EvolutionType.DERIVED):
        if not suggestion.target_skill_ids:
            logger.warning("FIX/DERIVED suggestion missing target_skill_ids")
            return None

        for target_id in suggestion.target_skill_ids:
            rec = evolver._store.load_record(target_id)
            if not rec:
                logger.warning("Target skill not found: %s", target_id)
                return None
            content = evolver._load_skill_content(rec)
            if not content:
                logger.warning("Cannot load content for skill: %s", target_id)
                return None
            skill_dir = Path(rec.path).parent if rec.path else None

            records.append(rec)
            contents.append(content)
            if skill_dir:
                dirs.append(skill_dir)

        if suggestion.evolution_type == EvolutionType.FIX and len(records) != 1:
            logger.warning(
                "FIX requires exactly 1 target, got %d: %s",
                len(records), suggestion.target_skill_ids,
            )
            return None

    return EvolutionContext(
        trigger=EvolutionTrigger.ANALYSIS,
        suggestion=suggestion,
        skill_records=records,
        skill_contents=contents,
        skill_dirs=dirs,
        source_task_id=analysis.task_id,
        recent_analyses=[analysis],
        available_tools=evolver._available_tools,
    )


def load_skill_content(evolver, record: "SkillRecord") -> str:
    """Load SKILL.md content from disk via registry or direct read."""
    content = evolver._registry.load_skill_content(record.skill_id)
    if content:
        return content
    if record.path:
        p = Path(record.path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return ""


def diagnose_skill_health(
    record: "SkillRecord",
) -> tuple[Optional[EvolutionType], str]:
    """Diagnose what type of evolution a skill needs based on metrics.

    Returns ``(EvolutionType, direction_str)`` or ``(None, "")`` if healthy.
    Thresholds are intentionally relaxed — LLM confirmation filters
    false positives.
    """
    if record.fallback_rate > _FALLBACK_THRESHOLD:
        return EvolutionType.FIX, (
            f"High fallback rate ({record.fallback_rate:.0%}): "
            f"skill is frequently selected but not applied, "
            f"suggesting instructions are unclear or outdated."
        )

    if record.applied_rate > _HIGH_APPLIED_FOR_FIX and record.completion_rate < _LOW_COMPLETION_THRESHOLD:
        return EvolutionType.FIX, (
            f"Low completion rate ({record.completion_rate:.0%}) despite "
            f"high applied rate ({record.applied_rate:.0%}): "
            f"skill instructions may be incorrect or incomplete."
        )

    if record.effective_rate < _MODERATE_EFFECTIVE_THRESHOLD and record.applied_rate > _MIN_APPLIED_FOR_DERIVED:
        return EvolutionType.DERIVED, (
            f"Moderate effectiveness ({record.effective_rate:.0%}): "
            f"skill works sometimes but could be enhanced with "
            f"better error handling or alternative approaches."
        )

    return None, ""
