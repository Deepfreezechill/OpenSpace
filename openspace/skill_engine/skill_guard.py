"""SkillGuard — quality gates for skill evolution.

Wraps ReviewGate + SkillStore into a single guarded API that ensures
every skill mutation is reviewed BEFORE persistence. Unsafe skills are
never written to the store.

Three guarded entry points:
  - guarded_evolve()    — for FIXED/DERIVED evolutions (store.evolve_skill)
  - guarded_save()      — for CAPTURED skills (store.save_record)
  - guarded_reactivate() — re-review before reactivating quarantined skills

Usage::

    guard = SkillGuard(store=store)
    result = await guard.guarded_evolve(new_record, parent_ids)
    if not result.passed:
        logger.error("Skill blocked: %s", result)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from openspace.skill_engine.review_gate import ReviewGate, ReviewResult, CheckResult

logger = logging.getLogger(__name__)


class SkillGuard:
    """Pre-persist quality gate for all skill mutations.

    Every mutation path (evolve, save, reactivate) runs through
    ReviewGate BEFORE the store write. If review fails, the write
    never happens — the skill is never activated.
    """

    def __init__(self, store, gate: Optional[ReviewGate] = None) -> None:
        self._store = store
        self._gate = gate or ReviewGate()

    async def guarded_evolve(
        self,
        new_record,
        parent_skill_ids: List[str],
    ) -> ReviewResult:
        """Review, then persist an evolved skill (FIXED or DERIVED).

        Returns the ReviewResult. If passed, the skill is persisted and
        activated. If failed, the skill is NOT persisted at all.
        """
        result = self._gate.review(new_record)

        if not result.passed:
            failed = [c for c in result.checks if c.verdict == "fail"]
            logger.warning(
                "SkillGuard BLOCKED evolve of %s — failed: %s",
                new_record.skill_id,
                ", ".join(c.name for c in failed),
            )
            for c in failed:
                logger.warning("  [%s] %s", c.name, c.detail)
            return result

        await self._store.evolve_skill(new_record, parent_skill_ids)
        logger.info(
            "SkillGuard APPROVED evolve of %s (gen=%d)",
            new_record.skill_id,
            new_record.lineage.generation,
        )
        return result

    async def guarded_save(self, record) -> ReviewResult:
        """Review, then save a new captured skill.

        Returns the ReviewResult. If passed, the skill is saved.
        If failed, the skill is NOT saved.
        """
        result = self._gate.review(record)

        if not result.passed:
            failed = [c for c in result.checks if c.verdict == "fail"]
            logger.warning(
                "SkillGuard BLOCKED save of %s — failed: %s",
                record.skill_id,
                ", ".join(c.name for c in failed),
            )
            return result

        await self._store.save_record(record)
        logger.info("SkillGuard APPROVED save of %s", record.skill_id)
        return result

    async def guarded_reactivate(self, skill_id: str) -> ReviewResult:
        """Re-review a quarantined skill before reactivation.

        Fetches the record, runs review, and only reactivates if it passes.
        """
        record = await self._store.get_record(skill_id)
        if record is None:
            return ReviewResult.from_checks([
                CheckResult(
                    name="lookup",
                    verdict="fail",
                    detail=f"Skill {skill_id} not found in store",
                ),
            ])

        result = self._gate.review(record)

        if not result.passed:
            logger.warning(
                "SkillGuard BLOCKED reactivation of %s — still fails review",
                skill_id,
            )
            return result

        await self._store.reactivate_record(skill_id)
        logger.info("SkillGuard APPROVED reactivation of %s", skill_id)
        return result
