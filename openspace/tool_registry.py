"""ToolRegistry — skill discovery, selection, and context injection.

Extracted from ``OpenSpace`` (tool_layer.py) in Epic 4.1.  Owns:

- Skill directory discovery (env, config, builtin)
- LLM-based skill selection for tasks
- Skill context injection into grounding agents
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openspace.llm import LLMClient
from openspace.recording import RecordingManager
from openspace.skill_engine import SkillRegistry
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.agents import GroundingAgent
    from openspace.tool_layer import OpenSpaceConfig

logger = Logger.get_logger(__name__)


class ToolRegistry:
    """Manages skill discovery, LLM-based selection, and context injection.

    Parameters
    ----------
    config:
        The ``OpenSpaceConfig`` (for ``skill_registry_model``, ``llm_kwargs``).
    grounding_config:
        The grounding configuration object (``skills.enabled``, ``skills.skill_dirs``,
        ``skills.max_select``).
    llm_client:
        Fallback LLM client for skill selection when no dedicated model is
        configured.
    """

    def __init__(
        self,
        *,
        config: OpenSpaceConfig,
        grounding_config: Any,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._config = config
        self._grounding_config = grounding_config
        self._llm_client = llm_client
        self._registry: Optional[SkillRegistry] = None

    # ── Public properties ─────────────────────────────────────────────

    @property
    def registry(self) -> Optional[SkillRegistry]:
        """The underlying ``SkillRegistry``, or ``None`` if not yet discovered."""
        return self._registry

    # ── Discovery ─────────────────────────────────────────────────────

    def discover(self) -> bool:
        """Build and populate the SkillRegistry from configured directories.

        Discovery order (earlier wins on name collision):
          1. ``OPENSPACE_HOST_SKILL_DIRS`` env — host agent skill directories
          2. ``config_grounding.json → skills.skill_dirs`` — user-specified
          3. ``openspace/skills/``       — built-in skills (always present)

        Returns ``True`` if at least one skill directory was resolved and the
        registry was created; ``False`` otherwise.
        """
        skill_paths: List[Path] = []
        skill_cfg = (
            self._grounding_config.skills
            if self._grounding_config
            else None
        )

        # 1. Host agent skill directories from env (standalone mode support)
        host_dirs_raw = os.environ.get("OPENSPACE_HOST_SKILL_DIRS", "")
        if host_dirs_raw:
            for d in host_dirs_raw.split(","):
                d = d.strip()
                if not d:
                    continue
                p = Path(d)
                if p.exists():
                    skill_paths.append(p)
                    logger.info(f"Host skill dir (from env): {p}")
                else:
                    logger.warning(f"Host skill dir does not exist: {d}")

        # 2. User-specified skill directories from config_grounding.json
        if skill_cfg and skill_cfg.skill_dirs:
            for d in skill_cfg.skill_dirs:
                p = Path(d)
                if p in skill_paths:
                    continue  # Already added via OPENSPACE_HOST_SKILL_DIRS
                if p.exists():
                    skill_paths.append(p)
                else:
                    logger.warning(f"Configured skill dir does not exist: {d}")

        # 3. Built-in skills (openspace/skills/)
        builtin_skills = Path(__file__).resolve().parent / "skills"
        if builtin_skills.exists():
            skill_paths.append(builtin_skills)

        if not skill_paths:
            logger.debug("No skill directories found, skills disabled")
            self._registry = None
            return False

        self._registry = SkillRegistry(skill_dirs=skill_paths)
        self._registry.discover()
        return True

    # ── Selection & injection ─────────────────────────────────────────

    async def select_and_inject(
        self,
        task: str,
        *,
        agent: Optional[GroundingAgent],
        store: Optional[Any] = None,
        recording_mgr: Optional[RecordingManager] = None,
    ) -> bool:
        """Select skills for *task* via LLM, inject into *agent*.

        When the registry has many skills, a BM25 + embedding pre-filter
        narrows the candidate set before LLM selection (see
        ``SkillRegistry.select_skills_with_llm``).

        Only selected skills are injected (full SKILL.md content).
        Returns ``True`` if at least one active skill was injected.
        """
        if not self._registry or not agent:
            return False

        selection_record = None
        skill_cfg = (
            self._grounding_config.skills
            if self._grounding_config
            else None
        )
        max_select = skill_cfg.max_select if skill_cfg else 2
        skill_llm = self._get_selection_llm(agent=agent)

        # Fetch quality metrics so the selector can filter/annotate
        skill_quality: Optional[Dict[str, Dict[str, Any]]] = None
        if store:
            try:
                rows = store.get_summary(active_only=True)
                skill_quality = {
                    r["skill_id"]: {
                        "total_selections": r.get("total_selections", 0),
                        "total_applied": r.get("total_applied", 0),
                        "total_completions": r.get("total_completions", 0),
                        "total_fallbacks": r.get("total_fallbacks", 0),
                    }
                    for r in rows
                }
            except Exception as e:
                logger.debug(f"Could not load skill quality metrics: {e}")

        if skill_llm:
            selected, selection_record = (
                await self._registry.select_skills_with_llm(
                    task,
                    llm_client=skill_llm,
                    max_skills=max_select,
                    skill_quality=skill_quality,
                )
            )
        else:
            logger.info(
                "No LLM client available for skill selection — proceeding without skills"
            )
            selected = []
            selection_record = {
                "method": "no_llm",
                "task": task[:500],
                "available_skills": [
                    s.skill_id for s in self._registry.list_skills()
                ],
                "selected": [],
            }

        # Record skill selection to metadata.json
        if recording_mgr and selection_record:
            selection_record["model"] = (
                skill_llm.model if skill_llm else "keyword_only"
            )
            await RecordingManager.record_skill_selection(selection_record)

        if not selected:
            agent.clear_skill_context()
            return False

        # Inject active skills (full SKILL.md content, backend-aware)
        agent_backends = agent.backend_scope if agent else None
        context_text = self._registry.build_context_injection(
            selected, backends=agent_backends
        )
        skill_ids = [s.skill_id for s in selected]
        agent.set_skill_context(context_text, skill_ids)
        logger.info(f"Injected {len(selected)} active skill(s): {skill_ids}")

        return True

    # ── Internal ──────────────────────────────────────────────────────

    def _get_selection_llm(
        self, *, agent: Optional[GroundingAgent] = None
    ) -> Optional[LLMClient]:
        """Get the LLM client to use for skill selection.

        Priority: config.skill_registry_model > tool_retrieval_model > llm_model.
        """
        # 1. Dedicated skill selection model
        if self._config.skill_registry_model:
            return LLMClient(
                model=self._config.skill_registry_model,
                timeout=30.0,
                max_retries=2,
                **self._config.llm_kwargs,
            )

        # 2. Tool retrieval model from grounding agent
        if (
            agent
            and hasattr(agent, "_tool_retrieval_llm")
            and agent._tool_retrieval_llm
        ):
            return agent._tool_retrieval_llm

        # 3. Main LLM client
        return self._llm_client
