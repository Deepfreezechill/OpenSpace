"""Results, telemetry, and iteration helpers for GroundingAgent.

Pure functions and agent-delegating helpers for building iteration
feedback, final summaries, task-completion checks, and recording.
Extracted from grounding_agent.py (Epic 5.10).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from scion.prompts import GroundingAgentPrompts
from scion.utils.logging import Logger

logger = Logger.get_logger("scion.agents.grounding_agent")


# ── Pure functions (no agent parameter) ────────────────────────────


def build_iteration_feedback(
    iteration: int,
    llm_summary: Optional[str] = None,
    add_guidance: bool = True,
) -> Optional[Dict[str, str]]:
    """Build feedback message to add to next iteration.

    Returns ``None`` when *llm_summary* is falsy.
    """
    if not llm_summary:
        return None

    feedback_content = GroundingAgentPrompts.iteration_feedback(
        iteration=iteration, llm_summary=llm_summary, add_guidance=add_guidance
    )

    return {"role": "system", "content": feedback_content}


def remove_previous_guidance(messages: List[Dict[str, Any]]) -> None:
    """Remove guidance section from previous iteration feedback messages.

    Mutates *messages* in-place.
    """
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            # Check if this is an iteration feedback message with guidance
            if "## Iteration" in content and "Summary" in content and "---" in content:
                # Remove everything from "---" onwards (the guidance part)
                summary_only = content.split("---")[0].strip()
                msg["content"] = summary_only


def format_tool_executions(all_tool_results: List[Dict]) -> List[Dict]:
    """Format raw tool-result dicts into a serialisable execution list."""
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
        arguments: dict = {}
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


def check_task_completion(messages: List[Dict]) -> bool:
    """Return True if the last assistant message contains the COMPLETE token."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return GroundingAgentPrompts.TASK_COMPLETE in content
    return False


def extract_last_assistant_message(messages: List[Dict]) -> str:
    """Return content of the last assistant message, or ``""``."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


# ── Agent-dependent functions ──────────────────────────────────────


async def generate_final_summary(
    agent,
    instruction: str,
    messages: List[Dict],
    iterations: int,
) -> tuple[str, bool, List[Dict]]:
    """Generate final summary across all iterations for reporting to upper layer.

    Returns:
        tuple[str, bool, List[Dict]]: (summary_text, success_flag, context_used)
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
        summary_response = await agent._llm_client.complete(
            messages=clean_messages, tools=None, execute_tools=False
        )

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
            f"Task completed after {iterations} iteration(s), but failed to generate summary.",
            False,
            context_for_return,
        )


async def build_final_result(
    agent,
    instruction: str,
    messages: List[Dict],
    all_tool_results: List[Dict],
    iterations: int,
    max_iterations: int,
    iteration_contexts: List[Dict] = None,
    retrieved_tools_list: List[Dict] = None,
    search_debug_info: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build the final execution result dict.

    Routes ``_check_task_completion``, ``_format_tool_executions``, and
    ``_extract_last_assistant_message`` through *agent* to preserve MRO.
    """
    is_complete = agent._check_task_completion(messages)

    tool_executions = agent._format_tool_executions(all_tool_results)

    result = {
        "instruction": instruction,
        "step": agent.step,
        "iterations": iterations,
        "tool_executions": tool_executions,
        "messages": messages,
        "iteration_contexts": iteration_contexts or [],
        "retrieved_tools_list": retrieved_tools_list or [],
        "search_debug_info": search_debug_info,
        "active_skills": list(agent._active_skill_ids),
        "keep_session": True,
    }

    if is_complete:
        logger.info("Task completed with <COMPLETE> marker")
        last_response = agent._extract_last_assistant_message(messages)
        result["response"] = last_response.replace(GroundingAgentPrompts.TASK_COMPLETE, "").strip()
        result["status"] = "success"
    else:
        result["response"] = agent._extract_last_assistant_message(messages)
        result["status"] = "incomplete"
        result["warning"] = (
            f"Task reached max iterations ({max_iterations}) without completion. "
            f"This may indicate the task needs more steps or clarification."
        )

    return result


async def record_agent_execution(
    agent,
    result: Dict[str, Any],
    instruction: str,
) -> None:
    """Record agent execution to recording manager.

    No-op when ``agent._recording_manager`` is falsy.
    """
    if not agent._recording_manager:
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

    await agent._recording_manager.record_agent_action(
        agent_name=agent.name,
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
            "step": agent.step,
            "instruction": instruction,
        },
    )
