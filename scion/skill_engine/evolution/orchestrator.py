"""Evolution orchestration — dispatch, throttle, and background scheduling.

Functions extracted from ``SkillEvolver`` (Epic 5.2):
- ``dispatch_evolution``   — route a single context to the right engine
- ``execute_contexts``     — run a batch of contexts in parallel (throttled)
- ``schedule_background``  — fire-and-forget background evolution task
- ``log_background_result`` — callback that logs task outcome
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from scion.utils.logging import Logger

from .models import EvolutionContext

from scion.skill_engine.types import EvolutionType

if TYPE_CHECKING:
    from scion.skill_engine.types import SkillRecord

logger = Logger.get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-context dispatch
# ---------------------------------------------------------------------------

async def dispatch_evolution(
    evolver,
    ctx: EvolutionContext,
) -> Optional["SkillRecord"]:
    """Execute one evolution action.  Returns new ``SkillRecord`` or *None*.

    Delegates to the appropriate engine method on *evolver*:
    ``_evolve_fix``, ``_evolve_derived``, or ``_evolve_captured``.

    The global semaphore is **not** acquired here — it is managed by
    ``execute_contexts`` so the concurrency limit covers whole batches.
    """
    evo_type: EvolutionType = ctx.suggestion.evolution_type
    try:
        if evo_type == EvolutionType.FIX:
            return await evolver._evolve_fix(ctx)
        elif evo_type == EvolutionType.DERIVED:
            return await evolver._evolve_derived(ctx)
        elif evo_type == EvolutionType.CAPTURED:
            return await evolver._evolve_captured(ctx)
        else:
            logger.warning("Unknown evolution type: %s", evo_type)
            return None
    except Exception as e:
        targets = "+".join(ctx.suggestion.target_skill_ids) or "(new)"
        logger.error(
            "Evolution failed [%s] target=%s: %s",
            evo_type.value,
            targets,
            e,
        )
        return None


# ---------------------------------------------------------------------------
# Batch execution with throttling
# ---------------------------------------------------------------------------

async def execute_contexts(
    evolver,
    contexts: List[EvolutionContext],
    trigger_label: str,
) -> List["SkillRecord"]:
    """Execute a list of evolution contexts in parallel (throttled).

    Uses *evolver._semaphore* for concurrency control and routes each
    context through ``dispatch_evolution``.
    """

    async def _throttled(c: EvolutionContext) -> Optional["SkillRecord"]:
        async with evolver._semaphore:
            return await evolver.evolve(c)

    raw = await asyncio.gather(
        *[_throttled(c) for c in contexts],
        return_exceptions=True,
    )
    results: List["SkillRecord"] = []
    for r in raw:
        if isinstance(r, BaseException):
            logger.error("[Trigger:%s] Evolution task raised: %s", trigger_label, r)
        elif r is not None:
            results.append(r)

    if results:
        names = [r.name for r in results]
        logger.info(
            "[Trigger:%s] Evolved %d skill(s): %s",
            trigger_label,
            len(results),
            names,
        )
    return results


# ---------------------------------------------------------------------------
# Background task scheduling
# ---------------------------------------------------------------------------

def schedule_background(
    background_tasks: set,
    coro,
    *,
    label: str = "background_evolution",
) -> Optional[asyncio.Task]:
    """Launch *coro* as a background ``asyncio.Task``.

    *background_tasks* is the ``set`` that tracks outstanding tasks
    (typically ``SkillEvolver._background_tasks``).  The task is removed
    from the set on completion via ``discard`` callback.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop — cannot schedule %s", label)
        return None

    task = loop.create_task(coro, name=label)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    task.add_done_callback(log_background_result)
    return task


def log_background_result(task: asyncio.Task) -> None:
    """Log the outcome of a background evolution task."""
    if task.cancelled():
        logger.debug("Background task '%s' was cancelled", task.get_name())
        return
    exc = task.exception()
    if exc:
        logger.error(
            "Background task '%s' failed: %s",
            task.get_name(),
            exc,
            exc_info=exc,
        )
