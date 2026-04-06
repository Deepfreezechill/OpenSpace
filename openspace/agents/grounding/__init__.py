"""openspace.agents.grounding — GroundingAgent subsystem package.

Extracted from the original monolithic ``grounding_agent.py`` (1 181 lines) across
Epics 5.7–5.10 of Phase 5b.  Each submodule owns a single responsibility:

* **context** — Skill-context and skill-registry injection
* **execution** — Core multi-iteration execution loop
* **messages** — Message safety: truncation and content capping
* **prompts** — System-prompt construction and message assembly
* **results** — Result building, telemetry recording, iteration feedback
* **tools** — Tool retrieval and fallback loading
* **visual** — Visual-analysis callback and screenshot selection
* **workspace** — Workspace scanning and artifact detection

The class itself remains in ``openspace.agents.grounding_agent`` as a thin
facade (< 200 lines) that delegates every method to this package.
"""

from openspace.agents.grounding.context import (
    clear_skill_context,
    has_skill_context,
    set_skill_context,
    set_skill_registry,
)
from openspace.agents.grounding.execution import process
from openspace.agents.grounding.messages import (
    _MAX_SINGLE_CONTENT_CHARS,
    cap_message_content,
    truncate_messages,
)
from openspace.agents.grounding.prompts import construct_messages, default_system_prompt
from openspace.agents.grounding.results import (
    build_final_result,
    build_iteration_feedback,
    check_task_completion,
    extract_last_assistant_message,
    format_tool_executions,
    generate_final_summary,
    record_agent_execution,
    remove_previous_guidance,
)
from openspace.agents.grounding.tools import _get_available_tools, _load_all_tools
from openspace.agents.grounding.visual import (
    _enhance_result_with_visual_context,
    _select_key_screenshots,
    _visual_analysis_callback,
)
from openspace.agents.grounding.workspace import (
    _check_workspace_artifacts,
    _get_workspace_path,
    _scan_workspace_files,
)

__all__ = [
    # context
    "set_skill_context",
    "clear_skill_context",
    "has_skill_context",
    "set_skill_registry",
    # execution
    "process",
    # messages
    "cap_message_content",
    "truncate_messages",
    # prompts
    "default_system_prompt",
    "construct_messages",
    # results
    "build_final_result",
    "build_iteration_feedback",
    "check_task_completion",
    "extract_last_assistant_message",
    "format_tool_executions",
    "generate_final_summary",
    "record_agent_execution",
    "remove_previous_guidance",
]
