from __future__ import annotations

import copy
import json
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
from openspace.prompts import GroundingAgentPrompts
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

    def _build_iteration_feedback(
        self, iteration: int, llm_summary: Optional[str] = None, add_guidance: bool = True
    ) -> Optional[Dict[str, str]]:
        """
        Build feedback message to add to next iteration.
        """
        if not llm_summary:
            return None

        feedback_content = GroundingAgentPrompts.iteration_feedback(
            iteration=iteration, llm_summary=llm_summary, add_guidance=add_guidance
        )

        return {"role": "system", "content": feedback_content}

    def _remove_previous_guidance(self, messages: List[Dict[str, Any]]) -> None:
        """
        Remove guidance section from previous iteration feedback messages.
        """
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Check if this is an iteration feedback message with guidance
                if "## Iteration" in content and "Summary" in content and "---" in content:
                    # Remove everything from "---" onwards (the guidance part)
                    summary_only = content.split("---")[0].strip()
                    msg["content"] = summary_only

    async def _generate_final_summary(
        self, instruction: str, messages: List[Dict], iterations: int
    ) -> tuple[str, bool, List[Dict]]:
        """
        Generate final summary across all iterations for reporting to upper layer.

        Returns:
            tuple[str, bool, List[Dict]]: (summary_text, success_flag, context_used)
                - summary_text: The generated summary or error message
                - success_flag: True if summary was generated successfully, False otherwise
                - context_used: The cleaned messages used for generating summary
        """
        final_summary_prompt = {
            "role": "user",
            "content": GroundingAgentPrompts.final_summary(instruction=instruction, iterations=iterations),
        }

        clean_messages = []
        for msg in messages:
            # Skip tool result messages
            if msg.get("role") == "tool":
                continue
            # Copy message and remove tool_calls if present
            clean_msg = msg.copy()
            if "tool_calls" in clean_msg:
                del clean_msg["tool_calls"]
            clean_messages.append(clean_msg)

        clean_messages.append(final_summary_prompt)

        # Save context for return
        context_for_return = copy.deepcopy(clean_messages)

        try:
            # Call LLMClient to generate final summary (without tools)
            summary_response = await self._llm_client.complete(messages=clean_messages, tools=None, execute_tools=False)

            final_summary = summary_response.get("message", {}).get("content", "")

            if final_summary:
                logger.info(f"Generated final summary: {final_summary[:200]}...")
                return final_summary, True, context_for_return
            else:
                logger.warning("LLM returned empty final summary")
                return (
                    f"Task completed after {iterations} iteration(s). Check execution history for details.",
                    True,
                    context_for_return,
                )

        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            return (
                f"Task completed after {iterations} iteration(s), but failed to generate summary: {str(e)}",
                False,
                context_for_return,
            )

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
        """
        Build final execution result.

        Args:
            instruction: Original instruction
            messages: Complete conversation history (including all iteration summaries)
            all_tool_results: All tool execution results
            iterations: Number of iterations performed
            max_iterations: Maximum allowed iterations
            iteration_contexts: Context snapshots for each iteration
            retrieved_tools_list: List of tools retrieved for this task
            search_debug_info: Debug info from tool search (similarity scores, LLM selections)
        """
        is_complete = self._check_task_completion(messages)

        tool_executions = self._format_tool_executions(all_tool_results)

        result = {
            "instruction": instruction,
            "step": self.step,
            "iterations": iterations,
            "tool_executions": tool_executions,
            "messages": messages,
            "iteration_contexts": iteration_contexts or [],
            "retrieved_tools_list": retrieved_tools_list or [],
            "search_debug_info": search_debug_info,
            "active_skills": list(self._active_skill_ids),
            "keep_session": True,
        }

        if is_complete:
            logger.info("Task completed with <COMPLETE> marker")
            # Use LLM's own completion response directly (no extra LLM call needed)
            # LLM already generates a summary before outputting <COMPLETE>
            last_response = self._extract_last_assistant_message(messages)
            # Remove the <COMPLETE> token from response for cleaner output
            result["response"] = last_response.replace(GroundingAgentPrompts.TASK_COMPLETE, "").strip()
            result["status"] = "success"

            # [DISABLED] Extra LLM call to generate final summary
            # final_summary, summary_success, final_summary_context = await self._generate_final_summary(
            #     instruction=instruction,
            #     messages=messages,
            #     iterations=iterations
            # )
            # result["response"] = final_summary
            # result["final_summary_context"] = final_summary_context
        else:
            result["response"] = self._extract_last_assistant_message(messages)
            result["status"] = "incomplete"
            result["warning"] = (
                f"Task reached max iterations ({max_iterations}) without completion. "
                f"This may indicate the task needs more steps or clarification."
            )

        return result

    def _format_tool_executions(self, all_tool_results: List[Dict]) -> List[Dict]:
        executions = []
        for tr in all_tool_results:
            tool_result_obj = tr.get("result")
            tool_call = tr.get("tool_call")

            status = "unknown"
            if hasattr(tool_result_obj, "status"):
                status_obj = tool_result_obj.status
                status = getattr(status_obj, "value", status_obj)

            # Extract tool_name and arguments from tool_call object (litellm format)
            tool_name = "unknown"
            arguments = {}
            if tool_call is not None:
                if hasattr(tool_call, "function"):
                    # tool_call is an object with .function attribute
                    tool_name = getattr(tool_call.function, "name", "unknown")
                    args_raw = getattr(tool_call.function, "arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            arguments = json.loads(args_raw) if args_raw.strip() else {}
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = args_raw if isinstance(args_raw, dict) else {}
                elif isinstance(tool_call, dict):
                    # Fallback: tool_call is a dict
                    func = tool_call.get("function", {})
                    tool_name = func.get("name", "unknown")
                    args_raw = func.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            arguments = json.loads(args_raw) if args_raw.strip() else {}
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = args_raw if isinstance(args_raw, dict) else {}

            executions.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "backend": tr.get("backend"),
                    "server_name": tr.get("server_name"),
                    "status": status,
                    "content": tool_result_obj.content if hasattr(tool_result_obj, "content") else None,
                    "error": tool_result_obj.error if hasattr(tool_result_obj, "error") else None,
                    "execution_time": tool_result_obj.execution_time
                    if hasattr(tool_result_obj, "execution_time")
                    else None,
                    "metadata": tool_result_obj.metadata if hasattr(tool_result_obj, "metadata") else {},
                }
            )
        return executions

    def _check_task_completion(self, messages: List[Dict]) -> bool:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return GroundingAgentPrompts.TASK_COMPLETE in content
        return False

    def _extract_last_assistant_message(self, messages: List[Dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    async def _record_agent_execution(self, result: Dict[str, Any], instruction: str) -> None:
        """
        Record agent execution to recording manager.

        Args:
            result: Execution result
            instruction: Original instruction
        """
        if not self._recording_manager:
            return

        # Extract tool execution summary
        tool_summary = []
        if result.get("tool_executions"):
            for exec_info in result["tool_executions"]:
                tool_summary.append(
                    {
                        "tool": exec_info.get("tool_name", "unknown"),
                        "backend": exec_info.get("backend", "unknown"),
                        "status": exec_info.get("status", "unknown"),
                    }
                )

        await self._recording_manager.record_agent_action(
            agent_name=self.name,
            action_type="execute",
            input_data={"instruction": instruction},
            reasoning={
                "response": result.get("response", ""),
                "tools_selected": tool_summary,
            },
            output_data={
                "status": result.get("status", "unknown"),
                "iterations": result.get("iterations", 0),
                "num_tool_executions": len(result.get("tool_executions", [])),
            },
            metadata={
                "step": self.step,
                "instruction": instruction,
            },
        )
