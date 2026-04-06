from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openspace.agents.base import BaseAgent
from openspace.agents.grounding.context import (
    clear_skill_context as _clear_skill_context,
    has_skill_context as _has_skill_context,
    set_skill_context as _set_skill_context,
    set_skill_registry as _set_skill_registry,
)
from openspace.agents.grounding.execution import (
    process as _process_impl,
)
from openspace.agents.grounding.messages import (
    _MAX_SINGLE_CONTENT_CHARS,
    cap_message_content as _cap_message_content,
    truncate_messages as _truncate_messages_impl,
)
from openspace.agents.grounding.prompts import (
    construct_messages as _construct_messages,
    default_system_prompt as _default_system_prompt,
)
from openspace.agents.grounding.results import (
    build_final_result as _build_final_result_impl,
    build_iteration_feedback as _build_iteration_feedback_impl,
    check_task_completion as _check_task_completion_impl,
    extract_last_assistant_message as _extract_last_assistant_message_impl,
    format_tool_executions as _format_tool_executions_impl,
    generate_final_summary as _generate_final_summary_impl,
    record_agent_execution as _record_agent_execution_impl,
    remove_previous_guidance as _remove_previous_guidance_impl,
)
from openspace.agents.grounding.tools import (
    _get_available_tools as _get_available_tools_impl,
    _load_all_tools as _load_all_tools_impl,
)
from openspace.agents.grounding.visual import (
    _enhance_result_with_visual_context as _enhance_result_with_visual_context_impl,
    _select_key_screenshots as _select_key_screenshots_impl,
    _visual_analysis_callback as _visual_analysis_callback_impl,
)
from openspace.agents.grounding.workspace import (
    _check_workspace_artifacts as _check_workspace_artifacts_impl,
    _get_workspace_path as _get_workspace_path_impl,
    _scan_workspace_files as _scan_workspace_files_impl,
)
from openspace.grounding.core.types import ToolResult
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.grounding.core.grounding_client import GroundingClient
    from openspace.llm import LLMClient
    from openspace.recording import RecordingManager
    from openspace.skill_engine import SkillRegistry

logger = Logger.get_logger(__name__)


class GroundingAgent(BaseAgent):
    def __init__(
        self,
        name: str = "GroundingAgent",
        backend_scope: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
        grounding_client: Optional[GroundingClient] = None,
        recording_manager: Optional[RecordingManager] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 15,
        visual_analysis_timeout: float = 30.0,
        tool_retrieval_llm: Optional[LLMClient] = None,
        visual_analysis_model: Optional[str] = None,
    ) -> None:
        """
        Initialize the Grounding Agent.

        Args:
            name: Agent name
            backend_scope: List of backends this agent can access (None = all available)
            llm_client: LLM client for reasoning
            grounding_client: GroundingClient for tool execution
            recording_manager: RecordingManager for recording execution
            system_prompt: Custom system prompt
            max_iterations: Maximum LLM reasoning iterations for self-correction
            visual_analysis_timeout: Timeout for visual analysis LLM calls in seconds
            tool_retrieval_llm: LLM client for tool retrieval filter (None = use llm_client)
            visual_analysis_model: Model name for visual analysis (None = use llm_client.model)
        """
        super().__init__(
            name=name,
            backend_scope=backend_scope or ["gui", "shell", "mcp", "web", "system"],
            llm_client=llm_client,
            grounding_client=grounding_client,
            recording_manager=recording_manager,
        )

        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_iterations = max_iterations
        self._visual_analysis_timeout = visual_analysis_timeout
        self._tool_retrieval_llm = tool_retrieval_llm
        self._visual_analysis_model = visual_analysis_model

        # Skill context injection (set externally before process())
        self._skill_context: Optional[str] = None
        self._active_skill_ids: List[str] = []

        # Skill registry for mid-iteration retrieve_skill tool
        self._skill_registry: Optional["SkillRegistry"] = None

        # Tools from the last execution (available for post-execution analysis)
        self._last_tools: List = []

        logger.info(f"Grounding Agent initialized: {name}")
        logger.info(f"Backend scope: {self._backend_scope}")
        logger.info(f"Max iterations: {self._max_iterations}")
        logger.info(f"Visual analysis timeout: {self._visual_analysis_timeout}s")
        if tool_retrieval_llm:
            logger.info(f"Tool retrieval model: {tool_retrieval_llm.model}")
        if visual_analysis_model:
            logger.info(f"Visual analysis model: {visual_analysis_model}")

    def set_skill_context(
        self,
        context: str,
        skill_ids: Optional[List[str]] = None,
    ) -> None:
        """Inject skill guidance into the agent's system prompt."""
        return _set_skill_context(self, context, skill_ids)

    def clear_skill_context(self) -> None:
        """Remove skill guidance (used before fallback execution)."""
        return _clear_skill_context(self)

    @property
    def has_skill_context(self) -> bool:
        return _has_skill_context(self)

    def set_skill_registry(self, registry: Optional["SkillRegistry"]) -> None:
        """Attach a SkillRegistry so the agent can offer ``retrieve_skill`` as a tool."""
        return _set_skill_registry(self, registry)

    _MAX_SINGLE_CONTENT_CHARS = _MAX_SINGLE_CONTENT_CHARS

    @classmethod
    def _cap_message_content(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate oversized individual message contents in-place."""
        return _cap_message_content(messages, cls._MAX_SINGLE_CONTENT_CHARS)

    def _truncate_messages(
        self, messages: List[Dict[str, Any]], keep_recent: int = 8, max_tokens_estimate: int = 120000
    ) -> List[Dict[str, Any]]:
        return _truncate_messages_impl(
            messages, keep_recent, max_tokens_estimate, cap=self._MAX_SINGLE_CONTENT_CHARS
        )

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task execution request with multi-round iteration control."""
        return await _process_impl(self, context)

    def _default_system_prompt(self) -> str:
        """Default system prompt tailored to the agent's actual backend scope."""
        return _default_system_prompt(self)

    def construct_messages(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        return _construct_messages(self, context)

    async def _get_available_tools(self, task_description: Optional[str]) -> List:
        """Retrieve tools for the current execution phase."""
        return await _get_available_tools_impl(self, task_description)

    async def _load_all_tools(self, grounding_client: "GroundingClient") -> List:
        """Fallback: load all tools from all backends without search."""
        return await _load_all_tools_impl(self, grounding_client)

    async def _visual_analysis_callback(
        self, result: ToolResult, tool_name: str, tool_call: Dict, backend: str
    ) -> ToolResult:
        """Callback for LLMClient to handle visual analysis after tool execution."""
        return await _visual_analysis_callback_impl(self, result, tool_name, tool_call, backend)

    async def _enhance_result_with_visual_context(self, result: ToolResult, tool_name: str) -> ToolResult:
        """Enhance tool result with visual analysis for grounding agent workflows."""
        return await _enhance_result_with_visual_context_impl(self, result, tool_name)

    _select_key_screenshots = staticmethod(_select_key_screenshots_impl)

    _get_workspace_path = staticmethod(_get_workspace_path_impl)

    _scan_workspace_files = staticmethod(_scan_workspace_files_impl)

    async def _check_workspace_artifacts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check workspace for existing artifacts relevant to the task."""
        return await _check_workspace_artifacts_impl(self, context)

    # ── Results / telemetry delegates (Epic 5.10) ────────────────────

    _build_iteration_feedback = staticmethod(_build_iteration_feedback_impl)

    _remove_previous_guidance = staticmethod(_remove_previous_guidance_impl)

    _format_tool_executions = staticmethod(_format_tool_executions_impl)

    _check_task_completion = staticmethod(_check_task_completion_impl)

    _extract_last_assistant_message = staticmethod(_extract_last_assistant_message_impl)

    async def _generate_final_summary(
        self, instruction: str, messages: List[Dict], iterations: int
    ) -> tuple[str, bool, List[Dict]]:
        return await _generate_final_summary_impl(self, instruction, messages, iterations)

    async def _build_final_result(
        self,
        instruction: str,
        messages: List[Dict],
        all_tool_results: List[Dict],
        iterations: int,
        max_iterations: int,
        iteration_contexts: List[Dict] = None,
        retrieved_tools_list: List[Dict] = None,
        search_debug_info: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        return await _build_final_result_impl(
            self,
            instruction,
            messages,
            all_tool_results,
            iterations,
            max_iterations,
            iteration_contexts,
            retrieved_tools_list,
            search_debug_info,
        )

    async def _record_agent_execution(self, result: Dict[str, Any], instruction: str) -> None:
        return await _record_agent_execution_impl(self, result, instruction)

