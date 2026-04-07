"""SkillEvolver — execute skill evolution actions.

Three evolution types:
  FIX      — repair broken/outdated instructions (in-place, same name)
  DERIVED  — create enhanced version from existing skill (new directory)
  CAPTURED — capture novel reusable pattern from execution (brand new skill)

Three trigger sources:
  1. Post-analysis — analyzer found evolution suggestions for a specific task
  2. Tool degradation — ToolQualityManager detected problematic tools
  3. Metric monitor — periodic scan of skill health indicators

All triggers produce an EvolutionContext → evolve() → LLM agent loop →
apply-retry cycle → validation → store persistence.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from openspace.utils.logging import Logger

from .evolution.models import (
    EvolutionContext,
    EvolutionTrigger,
)
from .evolution.orchestrator import (
    dispatch_evolution as _dispatch_evolution,
    execute_contexts as _execute_contexts_impl,
    log_background_result as _log_background_result_impl,
    schedule_background as _schedule_background_impl,
)
from .evolution.confirmation import (
    llm_confirm_evolution as _llm_confirm_evolution_impl,
    parse_confirmation as _parse_confirmation_impl,
)
from .evolution.formatting import (
    format_analysis_context as _format_analysis_context_impl,
    format_skill_dir_content as _format_skill_dir_content_impl,
)
from .evolution.loop import (
    EVOLUTION_COMPLETE,
    EVOLUTION_FAILED,
    _MAX_EVOLUTION_ATTEMPTS,
    _MAX_EVOLUTION_ITERATIONS,
    apply_with_retry as _apply_with_retry_impl,
    parse_evolution_output as _parse_evolution_output_impl,
    run_evolution_loop as _run_evolution_loop_impl,
)
from .evolution.strategies import (
    evolve_captured as _evolve_captured_impl,
    evolve_derived as _evolve_derived_impl,
    evolve_fix as _evolve_fix_impl,
)
from .evolution.triggers import (
    build_context_from_analysis as _build_context_from_analysis_impl,
    diagnose_skill_health as _diagnose_skill_health_impl,
    load_skill_content as _load_skill_content_impl,
    process_analysis as _process_analysis_impl,
    process_metric_check as _process_metric_check_impl,
    process_tool_degradation as _process_tool_degradation_impl,
)
from .patch import SkillEditResult
from .skill_guard import SkillGuard
from .store import SkillStore
from .types import (
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
    SkillRecord,
)

if TYPE_CHECKING:
    from openspace.grounding.core.quality.types import ToolQualityRecord
    from openspace.grounding.core.tool import BaseTool
    from openspace.llm import LLMClient

    from .registry import SkillRegistry

logger = Logger.get_logger(__name__)


class SkillEvolver:
    """Execute skill evolution actions.

    Single entry point: ``evolve()`` takes an EvolutionContext, runs an
    LLM agent loop (with optional tool use), applies the edit with retry,
    validates the result, and persists the new SkillRecord via ``SkillStore``.

    Concurrency:
        ``max_concurrent`` controls the semaphore that throttles parallel
        evolutions across all trigger types.  File I/O is synchronous and
        naturally serialized by the event loop; only LLM calls run in
        parallel.

    Anti-loop (Trigger 2 — tool degradation):
        ``_addressed_degradations`` is a ``Dict[str, Set[str]]`` mapping
        ``tool_key → {skill_id, …}`` for skills that have already been
        evolved to handle a specific tool's degradation.  At the start of
        each ``process_tool_degradation`` call, tools that are no longer
        in the problematic list are pruned — so if a tool **recovers and
        then degrades again**, all its dependent skills are re-evaluated.

    Anti-loop (Trigger 3 — metric check):
        Newly-evolved skills have ``total_selections=0``, requiring
        ``min_selections`` (default 5) fresh data points before being
        re-evaluated.  This is data-driven and needs no time-based guard.

    Background:
        Trigger 2 and 3 are always launched as ``asyncio.Task``s via
        ``schedule_background()`` so they never block the main flow.
    """

    def __init__(
        self,
        store: SkillStore,
        registry: "SkillRegistry",
        llm_client: "LLMClient",
        model: Optional[str] = None,
        available_tools: Optional[List["BaseTool"]] = None,
        *,
        max_concurrent: int = 3,
    ) -> None:
        self._store = store
        self._guard = SkillGuard(store=store)
        self._registry = registry
        self._llm_client = llm_client
        self._model = model
        self._available_tools: List["BaseTool"] = available_tools or []

        # Concurrency: semaphore limits parallel LLM sessions
        self._max_concurrent = max(1, max_concurrent)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Anti-loop for Trigger 2: tracks which skills have already been
        # evolved for each degraded tool.  Keyed by tool_key.
        # Pruned when a tool leaves the problematic list (= recovered).
        self._addressed_degradations: Dict[str, Set[str]] = {}

        # Track background tasks so they can be awaited on shutdown.
        self._background_tasks: Set[asyncio.Task] = set()

    def set_available_tools(self, tools: List["BaseTool"]) -> None:
        """Update the tools available for evolution agent loops."""
        self._available_tools = list(tools)

    async def wait_background(self) -> None:
        """Await all outstanding background evolution tasks.

        Call this during shutdown / cleanup to ensure nothing is lost.
        """
        if self._background_tasks:
            logger.info(f"Waiting for {len(self._background_tasks)} background evolution task(s) to finish...")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    async def evolve(self, ctx: EvolutionContext) -> Optional[SkillRecord]:
        """Execute one evolution action. Returns new SkillRecord or None.

        The global semaphore is NOT acquired here — it is managed at the
        trigger-method level so the concurrency limit covers the whole batch.
        """
        return await _dispatch_evolution(self, ctx)

    # Trigger 1: post-analysis
    async def process_analysis(
        self,
        analysis: ExecutionAnalysis,
    ) -> List[SkillRecord]:
        """Process all evolution suggestions from a completed analysis."""
        return await _process_analysis_impl(self, analysis)

    # Trigger 2: tool quality degradation
    async def process_tool_degradation(
        self,
        problematic_tools: List["ToolQualityRecord"],
    ) -> List[SkillRecord]:
        """Fix skills that depend on degraded tools."""
        return await _process_tool_degradation_impl(self, problematic_tools)

    # Trigger 3: periodic metric check
    async def process_metric_check(
        self,
        min_selections: int = 5,
    ) -> List[SkillRecord]:
        """Scan active skills and evolve those with poor health metrics."""
        return await _process_metric_check_impl(self, min_selections)

    async def _execute_contexts(
        self,
        contexts: List[EvolutionContext],
        trigger_label: str,
    ) -> List[SkillRecord]:
        """Execute a list of evolution contexts in parallel (throttled).

        Used by all three triggers after building/confirming contexts.
        """
        return await _execute_contexts_impl(self, contexts, trigger_label)

    def schedule_background(
        self,
        coro,
        *,
        label: str = "background_evolution",
    ) -> Optional[asyncio.Task]:
        """Launch a coroutine as a background ``asyncio.Task``.

        Used by the caller (``OpenSpace._maybe_evolve_quality``) when
        ``background_triggers`` is True.  The task is tracked so it can
        be awaited on shutdown via ``wait_background()``.
        """
        return _schedule_background_impl(
            self._background_tasks, coro, label=label,
        )

    _log_background_result = staticmethod(_log_background_result_impl)

    # LLM confirmation for Trigger 2/3
    async def _llm_confirm_evolution(
        self,
        *,
        skill_record: SkillRecord,
        skill_content: str,
        proposed_type: EvolutionType,
        proposed_direction: str,
        trigger_context: str,
        recent_analyses: List[ExecutionAnalysis],
    ) -> bool:
        """Ask LLM to confirm whether a rule-based evolution candidate
        truly needs evolution."""
        return await _llm_confirm_evolution_impl(
            self,
            skill_record=skill_record,
            skill_content=skill_content,
            proposed_type=proposed_type,
            proposed_direction=proposed_direction,
            trigger_context=trigger_context,
            recent_analyses=recent_analyses,
        )

    _parse_confirmation = staticmethod(_parse_confirmation_impl)

    async def _evolve_fix(self, ctx: EvolutionContext) -> Optional[SkillRecord]:
        """In-place fix — delegates to ``evolution.strategies.evolve_fix``."""
        return await _evolve_fix_impl(self, ctx)

    async def _evolve_derived(self, ctx: EvolutionContext) -> Optional[SkillRecord]:
        """Create enhanced version — delegates to ``evolution.strategies.evolve_derived``."""
        return await _evolve_derived_impl(self, ctx)

    async def _evolve_captured(self, ctx: EvolutionContext) -> Optional[SkillRecord]:
        """Capture novel pattern — delegates to ``evolution.strategies.evolve_captured``."""
        return await _evolve_captured_impl(self, ctx)

    async def _run_evolution_loop(
        self,
        prompt: str,
        ctx: EvolutionContext,
    ) -> Optional[str]:
        """Token-driven LLM agent loop — delegates to ``evolution.loop.run_evolution_loop``."""
        return await _run_evolution_loop_impl(self, prompt, ctx)

    _parse_evolution_output = staticmethod(_parse_evolution_output_impl)

    async def _apply_with_retry(
        self,
        *,
        apply_fn,
        initial_content: str,
        skill_dir: Path,
        ctx: EvolutionContext,
        prompt: str,
        cleanup_on_retry: Optional[Path] = None,
    ) -> Optional[SkillEditResult]:
        """Apply edit with retry — delegates to ``evolution.loop.apply_with_retry``."""
        return await _apply_with_retry_impl(
            self,
            apply_fn=apply_fn,
            initial_content=initial_content,
            skill_dir=skill_dir,
            ctx=ctx,
            prompt=prompt,
            cleanup_on_retry=cleanup_on_retry,
        )


    def _build_context_from_analysis(
        self,
        analysis: ExecutionAnalysis,
        suggestion: EvolutionSuggestion,
    ) -> Optional[EvolutionContext]:
        """Build EvolutionContext from a single analysis suggestion."""
        return _build_context_from_analysis_impl(self, analysis, suggestion)

    def _load_skill_content(self, record: SkillRecord) -> str:
        """Load SKILL.md content from disk via registry or direct read."""
        return _load_skill_content_impl(self, record)

    _format_skill_dir_content = staticmethod(_format_skill_dir_content_impl)

    _format_analysis_context = staticmethod(_format_analysis_context_impl)

    _diagnose_skill_health = staticmethod(_diagnose_skill_health_impl)
