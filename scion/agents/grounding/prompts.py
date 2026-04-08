"""Prompt construction helpers for GroundingAgent.

Builds the system prompt and initial message list from execution context.
Extracted from grounding_agent.py (Epic 5.8).
"""

from __future__ import annotations

from typing import Any, Dict, List

from scion.prompts import GroundingAgentPrompts
from scion.utils.logging import Logger

logger = Logger.get_logger("scion.agents.grounding_agent")


def default_system_prompt(agent) -> str:
    """Default system prompt tailored to the agent's actual backend scope."""
    return GroundingAgentPrompts.build_system_prompt(agent._backend_scope)


def construct_messages(agent, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the initial message list for the LLM from *context*.

    Assembles: system prompt → workspace dir → workspace artifacts →
    skill context → user instruction.

    Args:
        agent: GroundingAgent instance.
        context: Execution context dict (must contain ``instruction``).

    Returns:
        List of message dicts ready for the LLM.

    Raises:
        ValueError: If *context* has no ``instruction``.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": agent._system_prompt}
    ]

    # Get instruction from context
    instruction = context.get("instruction", "")
    if not instruction:
        raise ValueError("context must contain 'instruction' field")

    # Add workspace directory
    workspace_dir = context.get("workspace_dir")
    if workspace_dir:
        messages.append(
            {"role": "system", "content": GroundingAgentPrompts.workspace_directory(workspace_dir)}
        )

    # Add workspace artifacts information
    workspace_artifacts = context.get("workspace_artifacts")
    if workspace_artifacts and workspace_artifacts.get("has_files"):
        files = workspace_artifacts.get("files", [])
        matching_files = workspace_artifacts.get("matching_files", [])
        recent_files = workspace_artifacts.get("recent_files", [])

        if matching_files:
            artifact_msg = GroundingAgentPrompts.workspace_matching_files(matching_files)
        elif len(recent_files) >= 2:
            artifact_msg = GroundingAgentPrompts.workspace_recent_files(
                total_files=len(files), recent_files=recent_files
            )
        else:
            artifact_msg = GroundingAgentPrompts.workspace_file_list(files)

        messages.append({"role": "system", "content": artifact_msg})

    # Skill injection — only active (selected) skills, full content
    if agent._skill_context:
        messages.append({"role": "system", "content": agent._skill_context})
        logger.info(
            f"Injected active skill context ({len(agent._active_skill_ids)} skill(s))"
        )

    # User instruction
    messages.append({"role": "user", "content": instruction})

    return messages
