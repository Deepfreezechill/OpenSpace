"""Skill-context and skill-registry helpers for GroundingAgent.

Pure side-effect functions that operate on agent instance state.
Extracted from grounding_agent.py (Epic 5.7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from scion.utils.logging import Logger

if TYPE_CHECKING:
    from scion.skill_engine import SkillRegistry

logger = Logger.get_logger("scion.agents.grounding_agent")


def set_skill_context(
    agent,
    context: str,
    skill_ids: Optional[List[str]] = None,
) -> None:
    """Inject skill guidance into the agent's system prompt.

    Called by ``Scion.execute()`` before ``process()`` when skills
    are matched.  The context is a formatted string built by
    ``SkillRegistry.build_context_injection()``.

    Args:
        agent: GroundingAgent instance.
        context: Formatted skill content for system prompt injection.
        skill_ids: skill_id values of injected skills.
    """
    agent._skill_context = context if context else None
    agent._active_skill_ids = skill_ids or []
    if agent._skill_context:
        logger.info(
            f"Skill context set: {', '.join(agent._active_skill_ids) or '(unnamed)'}"
        )


def clear_skill_context(agent) -> None:
    """Remove skill guidance (used before fallback execution)."""
    if agent._skill_context:
        logger.info(
            f"Skill context cleared (was: {', '.join(agent._active_skill_ids)})"
        )
    agent._skill_context = None
    agent._active_skill_ids = []


def has_skill_context(agent) -> bool:
    """Return True if skill context is currently set."""
    return agent._skill_context is not None


def set_skill_registry(agent, registry: Optional["SkillRegistry"]) -> None:
    """Attach a SkillRegistry so the agent can offer ``retrieve_skill`` as a tool."""
    agent._skill_registry = registry
    if registry:
        count = len(registry.list_skills())
        logger.info(
            f"Skill registry attached ({count} skill(s) available for mid-iteration retrieval)"
        )
