"""Core execution loop for GroundingAgent.

Implements the multi-round LLM iteration with tool calling,
skill-context stripping, message truncation, and result building.
Extracted from grounding_agent.py (Epic 5.8).
Instrumented with observability (Epic 6.1).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from openspace.observability.metrics import metrics as _metrics
from openspace.observability.tracing import tracer as _tracer
from openspace.prompts import GroundingAgentPrompts
from openspace.utils.logging import Logger

logger = Logger.get_logger("openspace.agents.grounding_agent")

# Exit after this many consecutive empty LLM responses.
_MAX_CONSECUTIVE_EMPTY = 5


async def process(agent, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a task with multi-round LLM iteration control.

    This is the main entry-point for task processing.  It:
    1. Validates the instruction.
    2. Checks for existing workspace artifacts.
    3. Retrieves available tools.
    4. Constructs initial messages (via ``agent.construct_messages``).
    5. Enters the LLM loop (up to *max_iterations* rounds).
    6. Builds and returns the final result dict.

    Args:
        agent: GroundingAgent instance.
        context: Execution context (must contain ``instruction``).

    Returns:
        Result dict with status, output, tool results, etc.
    """
    instruction = context.get("instruction", "")
    if not instruction:
        logger.error("Grounding Agent: No instruction provided")
        return {"error": "No instruction provided", "status": "error"}

    # Store current instruction for visual analysis context
    agent._current_instruction = instruction

    logger.info(f"Grounding Agent: Processing instruction at step {agent.step}")

    # ── Observability: start trace + metrics ─────────────────────────
    trace = _tracer.start_trace("grounding.process", instruction=instruction[:200])
    _metrics.execution_in_flight.labels(agent=agent._name).inc()

    # Existing workspace files check
    workspace_info = await agent._check_workspace_artifacts(context)
    if workspace_info["has_files"]:
        context["workspace_artifacts"] = workspace_info
        logger.info(
            f"Workspace has {len(workspace_info['files'])} existing files: "
            f"{workspace_info['files']}"
        )

    # Get available tools (auto-search with cap)
    tools = await agent._get_available_tools(instruction)
    agent._last_tools = tools  # expose for post-execution analysis

    # Get search debug info (similarity scores, LLM selections)
    search_debug_info = None
    if agent.grounding_client:
        search_debug_info = agent.grounding_client.get_last_search_debug_info()

    # Build retrieved tools list for return value
    retrieved_tools_list = _build_retrieved_tools_list(tools, search_debug_info)

    # Record retrieved tools
    if agent._recording_manager:
        from openspace.recording import RecordingManager

        await RecordingManager.record_retrieved_tools(
            task_instruction=instruction,
            tools=tools,
            search_debug_info=search_debug_info,
        )

    # Initialize iteration state
    max_iterations = context.get("max_iterations", agent._max_iterations)
    current_iteration = 0
    all_tool_results: List[Dict] = []
    iteration_contexts: List[Dict] = []
    consecutive_empty_responses = 0

    # Build initial messages
    messages = agent.construct_messages(context)

    # Record initial conversation setup once
    from openspace.recording import RecordingManager

    await RecordingManager.record_conversation_setup(
        setup_messages=copy.deepcopy(messages),
        tools=tools,
    )

    try:
        while current_iteration < max_iterations:
            current_iteration += 1
            logger.info(
                f"Grounding Agent: Iteration {current_iteration}/{max_iterations}"
            )

            # Strip skill context after the first iteration to save prompt tokens.
            if current_iteration == 2 and agent._skill_context:
                skill_ctx = agent._skill_context
                messages = [
                    m
                    for m in messages
                    if not (
                        m.get("role") == "system"
                        and m.get("content") == skill_ctx
                    )
                ]
                logger.info(
                    "Skill context removed from messages after first iteration"
                )

            # Cap oversized individual messages every iteration
            if current_iteration >= 2:
                messages = agent._cap_message_content(messages)

            # Truncate message history after 5 iterations
            if current_iteration >= 5:
                messages = agent._truncate_messages(
                    messages, keep_recent=8, max_tokens_estimate=120_000
                )

            messages_input_snapshot = copy.deepcopy(messages)

            # Call LLMClient for single round
            llm_response = await agent._llm_client.complete(
                messages=messages,
                tools=tools if context.get("auto_execute", True) else None,
                execute_tools=context.get("auto_execute", True),
                summary_prompt=None,
                tool_result_callback=agent._visual_analysis_callback,
            )

            # Update messages with LLM response
            messages = llm_response["messages"]

            # Collect tool results
            tool_results_this_iteration = llm_response.get("tool_results", [])
            if tool_results_this_iteration:
                all_tool_results.extend(tool_results_this_iteration)

            assistant_message = llm_response.get("message", {})
            assistant_content = assistant_message.get("content", "")

            has_tool_calls = llm_response.get("has_tool_calls", False)
            logger.info(
                f"Iteration {current_iteration} - Has tool calls: {has_tool_calls}, "
                f"Tool results: {len(tool_results_this_iteration)}, "
                f"Content length: {len(assistant_content)} chars"
            )

            if len(assistant_content) > 0:
                logger.info(
                    f"Iteration {current_iteration} - Assistant content preview: "
                    f"{repr(assistant_content[:300])}"
                )
                consecutive_empty_responses = 0
            else:
                if not has_tool_calls:
                    consecutive_empty_responses += 1
                    logger.warning(
                        f"Iteration {current_iteration} - NO tool calls and NO content "
                        f"(empty response {consecutive_empty_responses}/"
                        f"{_MAX_CONSECUTIVE_EMPTY})"
                    )

                    if consecutive_empty_responses >= _MAX_CONSECUTIVE_EMPTY:
                        logger.error(
                            f"Exiting due to {_MAX_CONSECUTIVE_EMPTY} consecutive "
                            "empty LLM responses. This may indicate API issues, "
                            "rate limiting, or context too long."
                        )
                        break
                else:
                    consecutive_empty_responses = 0

            # Snapshot messages after LLM call
            messages_output_snapshot = copy.deepcopy(messages)

            # Delta messages: only produced in this iteration
            delta_messages = messages[len(messages_input_snapshot):]

            # Response metadata
            response_metadata = {
                "has_tool_calls": has_tool_calls,
                "tool_calls_count": len(tool_results_this_iteration),
            }
            iteration_context = {
                "iteration": current_iteration,
                "messages_input": messages_input_snapshot,
                "messages_output": messages_output_snapshot,
                "response_metadata": response_metadata,
            }
            iteration_contexts.append(iteration_context)

            # Real-time save to conversations.jsonl (delta only)
            await RecordingManager.record_iteration_context(
                iteration=current_iteration,
                delta_messages=copy.deepcopy(delta_messages),
                response_metadata=response_metadata,
            )

            # Check for completion token
            is_complete = GroundingAgentPrompts.TASK_COMPLETE in assistant_content

            if is_complete:
                logger.info(
                    f"Task completed at iteration {current_iteration} "
                    f"(found {GroundingAgentPrompts.TASK_COMPLETE})"
                )
                break
            else:
                if tool_results_this_iteration:
                    logger.debug(
                        f"Task in progress, LLM called "
                        f"{len(tool_results_this_iteration)} tools"
                    )
                else:
                    logger.debug(
                        "Task in progress, LLM did not generate <COMPLETE>"
                    )

                # Remove previous iteration guidance to avoid accumulation
                messages = [
                    msg
                    for msg in messages
                    if not (
                        msg.get("role") == "system"
                        and "Iteration" in msg.get("content", "")
                        and "complete" in msg.get("content", "")
                    )
                ]

                guidance_msg = {
                    "role": "system",
                    "content": (
                        f"Iteration {current_iteration} complete. "
                        f"Check if task is finished - if yes, output "
                        f"{GroundingAgentPrompts.TASK_COMPLETE}. "
                        f"If not, continue with next action."
                    ),
                }
                messages.append(guidance_msg)
                continue

        # Build final result
        result = await agent._build_final_result(
            instruction=instruction,
            messages=messages,
            all_tool_results=all_tool_results,
            iterations=current_iteration,
            max_iterations=max_iterations,
            iteration_contexts=iteration_contexts,
            retrieved_tools_list=retrieved_tools_list,
            search_debug_info=search_debug_info,
        )

        # Record agent action
        if agent._recording_manager:
            await agent._record_agent_execution(result, instruction)

        # Increment step
        agent.increment_step()

        logger.info(
            f"Grounding Agent: Execution completed with status: "
            f"{result.get('status')}"
        )

        # ── Observability: record success metrics ────────────────────
        _metrics.execution_iterations.labels(agent=agent._name).observe(current_iteration)
        _metrics.execution_total.labels(agent=agent._name, status="success").inc()
        _metrics.execution_in_flight.labels(agent=agent._name).dec()
        root_span = _tracer.current_span()
        if root_span:
            root_span.attributes["iterations"] = current_iteration
            root_span.attributes["status"] = result.get("status", "unknown")
        _tracer.finish_trace()

        return result

    except Exception as e:
        logger.error(f"Grounding Agent: Execution failed: {e}")

        # ── Observability: record error metrics ──────────────────────
        _metrics.execution_total.labels(agent=agent._name, status="error").inc()
        _metrics.execution_in_flight.labels(agent=agent._name).dec()
        root_span = _tracer.current_span()
        if root_span:
            root_span.add_event("error", error_type=type(e).__name__)
        _tracer.finish_trace()

        result = {
            "error": str(e),
            "status": "error",
            "instruction": instruction,
            "iteration": current_iteration,
        }
        agent.increment_step()
        return result


def _build_retrieved_tools_list(
    tools: List, search_debug_info: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build the retrieved-tools metadata list for the result dict."""
    retrieved_tools_list: List[Dict[str, Any]] = []
    for tool in tools:
        tool_info: Dict[str, Any] = {
            "name": getattr(tool, "name", str(tool)),
            "description": getattr(tool, "description", ""),
        }
        # Prefer runtime_info.backend over backend_type
        runtime_info = getattr(tool, "_runtime_info", None)
        if runtime_info and hasattr(runtime_info, "backend"):
            tool_info["backend"] = (
                runtime_info.backend.value
                if hasattr(runtime_info.backend, "value")
                else str(runtime_info.backend)
            )
            tool_info["server_name"] = runtime_info.server_name
        elif hasattr(tool, "backend_type"):
            tool_info["backend"] = (
                tool.backend_type.value
                if hasattr(tool.backend_type, "value")
                else str(tool.backend_type)
            )

        # Add similarity score if available
        if search_debug_info and search_debug_info.get("tool_scores"):
            for score_info in search_debug_info["tool_scores"]:
                if score_info["name"] == tool_info["name"]:
                    tool_info["similarity_score"] = score_info["score"]
                    break

        retrieved_tools_list.append(tool_info)
    return retrieved_tools_list
