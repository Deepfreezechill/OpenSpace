"""Tool retrieval helpers for GroundingAgent.

Handles tool discovery, filtering, and fallback loading from backends.
Extracted from grounding_agent.py (Epic 5.9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from openspace.grounding.core.types import BackendType
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.grounding.core.grounding_client import GroundingClient

logger = Logger.get_logger("openspace.agents.grounding_agent")


async def _get_available_tools(agent, task_description: Optional[str]) -> List:
    """Retrieve tools for the current execution phase.

    Both skill-augmented and normal modes use the same
    ``get_tools_with_auto_search`` pipeline:
    - Non-MCP tools (shell, gui, web, system) are always included.
    - MCP tools are filtered by relevance only when their count
      exceeds ``max_tools``.

    When skills are active, the shell backend is guaranteed to be in
    scope (skills commonly reference ``shell_agent``).

    Falls back to returning all tools if anything fails.
    """
    grounding_client = agent.grounding_client
    if not grounding_client:
        return []

    backends = [BackendType(name) for name in agent._backend_scope]

    # Ensure shell backend is available when skills are active
    # (skills commonly reference shell_agent, read_file, etc.)
    if agent.has_skill_context:
        shell_bt = BackendType.SHELL
        if shell_bt not in backends:
            backends = list(backends) + [shell_bt]
            logger.info("Added Shell backend to scope for skill file I/O")

    try:
        retrieval_llm = agent._tool_retrieval_llm or agent._llm_client
        tools = await grounding_client.get_tools_with_auto_search(
            task_description=task_description,
            backend=backends,
            use_cache=True,
            llm_callable=retrieval_llm,
        )
        logger.info(
            f"GroundingAgent selected {len(tools)} tools (auto-search) "
            f"from {len(backends)} backends"
            + (" [skill-augmented]" if agent.has_skill_context else "")
        )
    except Exception as e:
        logger.warning(f"Auto-search tools failed, falling back to full list: {e}")
        tools = await _load_all_tools(agent, grounding_client)

    # Append retrieve_skill tool when skill registry is available
    if agent._skill_registry and agent._skill_registry.list_skills():
        from openspace.skill_engine.retrieve_tool import RetrieveSkillTool

        retrieve_llm = agent._tool_retrieval_llm or agent._llm_client
        retrieve_tool = RetrieveSkillTool(
            agent._skill_registry,
            backends=[b.value for b in backends],
            llm_client=retrieve_llm,
            skill_store=getattr(agent, "_skill_store", None),
        )
        retrieve_tool.bind_runtime_info(
            backend=BackendType.SYSTEM,
            session_name="internal",
        )
        tools.append(retrieve_tool)
        logger.info("Added retrieve_skill tool for mid-iteration skill retrieval")

    return tools


async def _load_all_tools(agent, grounding_client: "GroundingClient") -> List:
    """Fallback: load all tools from all backends without search."""
    all_tools = []
    for backend_name in agent._backend_scope:
        try:
            backend_type = BackendType(backend_name)
            tools = await grounding_client.list_tools(backend=backend_type)
            all_tools.extend(tools)
            logger.debug(f"Retrieved {len(tools)} tools from backend: {backend_name}")
        except Exception as e:
            logger.debug(f"Could not get tools from {backend_name}: {e}")

    logger.info(
        f"GroundingAgent fallback retrieved {len(all_tools)} tools from {len(agent._backend_scope)} backends"
    )
    return all_tools
