"""Evolution execution loop and apply-retry cycle.

Provides the core execution engine used by all three evolution strategies:

- ``run_evolution_loop`` — token-driven LLM agent loop with tool support
- ``parse_evolution_output`` — extract edit content or failure reason
- ``apply_with_retry`` — apply edit with retry on validation failure

All functions accept an ``evolver`` parameter (the ``SkillEvolver`` instance)
and delegate back through ``evolver._method()`` to preserve method resolution
order for subclass / hook / telemetry compatibility.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openspace.prompts import SkillEnginePrompts
from openspace.utils.logging import Logger

from ..skill_utils import (
    extract_change_summary as _extract_change_summary,
    strip_markdown_fences as _strip_markdown_fences,
    truncate as _truncate,
    validate_skill_dir as _validate_skill_dir,
)
from .confirmation import _RECORDING_MAX_CHARS, _SKILL_CONTENT_MAX_CHARS
from .models import EvolutionContext

if TYPE_CHECKING:
    from ..patch import SkillEditResult

logger = Logger.get_logger("openspace.skill_engine.evolver")

EVOLUTION_COMPLETE = SkillEnginePrompts.EVOLUTION_COMPLETE
EVOLUTION_FAILED = SkillEnginePrompts.EVOLUTION_FAILED

# Agent loop / retry constants
_MAX_EVOLUTION_ITERATIONS = 5  # Max tool-calling rounds for evolution agent
_MAX_EVOLUTION_ATTEMPTS = 3  # Max apply-retry attempts per evolution


async def run_evolution_loop(
    evolver,
    prompt: str,
    ctx: EvolutionContext,
) -> Optional[str]:
    """Run evolution as a token-driven agent loop.

    Modeled after ``GroundingAgent.process()`` — the loop continues
    until the LLM outputs an explicit completion/failure token, NOT
    based on whether tools were called.

    Termination signals (checked every iteration, regardless of tool use):
      - ``EVOLUTION_COMPLETE`` in assistant content → success, return edit.
      - ``EVOLUTION_FAILED``   in assistant content → failure, return None.

    Tool availability:
      - Iterations 1 … N-1: tools enabled (LLM may gather information).
      - Iteration N (final): tools disabled, LLM must output a decision.

    Each non-final iteration without a token gets a nudge message
    telling the LLM which iteration it is on and how many remain.

    Conversations are recorded to ``conversations.jsonl`` via
    ``RecordingManager`` (agent_name="SkillEvolver") so the full
    evolution dialogue is preserved for debugging and replay.
    """
    from openspace.recording import RecordingManager

    model = evolver._model or evolver._llm_client.model

    # Merge tools from context and instance-level
    evolution_tools: List = list(ctx.available_tools or [])
    if not evolution_tools:
        evolution_tools = list(evolver._available_tools)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": prompt},
    ]

    # Record initial conversation setup (truncated for data minimization)
    recorded_setup = [
        {"role": m["role"], "content": _truncate(m["content"], _RECORDING_MAX_CHARS)}
        for m in messages
    ]
    await RecordingManager.record_conversation_setup(
        setup_messages=recorded_setup,
        tools=evolution_tools if evolution_tools else None,
        agent_name="SkillEvolver",
        extra={
            "evolution_type": ctx.suggestion.evolution_type.value,
            "trigger": ctx.trigger.value,
            "target_skills": ctx.suggestion.target_skill_ids,
        },
    )

    for iteration in range(_MAX_EVOLUTION_ITERATIONS):
        is_last = iteration == _MAX_EVOLUTION_ITERATIONS - 1

        # Snapshot message count before any additions + LLM call
        msg_count_before = len(messages)

        # Final round: disable tools and force a decision
        if is_last:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"This is your FINAL round (iteration "
                        f"{iteration + 1}/{_MAX_EVOLUTION_ITERATIONS}) — "
                        f"no more tool calls allowed. "
                        f"You MUST output the skill edit content now based on "
                        f"all information gathered so far. Follow the output "
                        f"format specified in the original instructions. "
                        f"End with {EVOLUTION_COMPLETE} if the edit is satisfactory, "
                        f"or {EVOLUTION_FAILED} with a reason if you cannot produce one."
                    ),
                }
            )

        try:
            result = await evolver._llm_client.complete(
                messages=messages,
                tools=evolution_tools if (evolution_tools and not is_last) else None,
                execute_tools=True,
                model=model,
            )
        except Exception as e:
            logger.error(f"Evolution LLM call failed (iter {iteration + 1}): {e}")
            return None

        content = result["message"].get("content", "")
        updated_messages = result["messages"]
        has_tool_calls = result.get("has_tool_calls", False)

        # Record iteration delta (truncated for data minimization)
        delta = updated_messages[msg_count_before:]
        recorded_delta = [
            {"role": m["role"], "content": _truncate(m.get("content", ""), _RECORDING_MAX_CHARS)}
            for m in delta
        ]
        await RecordingManager.record_iteration_context(
            iteration=iteration + 1,
            delta_messages=recorded_delta,
            response_metadata={
                "has_tool_calls": has_tool_calls,
                "tool_calls_count": len(result.get("tool_results", [])),
                "has_completion_token": bool(
                    content and (EVOLUTION_COMPLETE in content or EVOLUTION_FAILED in content)
                ),
            },
            agent_name="SkillEvolver",
        )

        messages = updated_messages

        # ── Token check (every iteration, regardless of tool calls) ──
        if content and (EVOLUTION_COMPLETE in content or EVOLUTION_FAILED in content):
            edit_content, failure_reason = evolver._parse_evolution_output(content)
            if failure_reason is not None:
                targets = "+".join(ctx.suggestion.target_skill_ids) or "(new)"
                logger.warning(
                    f"Evolution LLM signalled failure "
                    f"[{ctx.suggestion.evolution_type.value}] "
                    f"target={targets}: {failure_reason}"
                )
                return None
            return edit_content

        # No token found
        if is_last:
            # Final round exhausted without a decision
            logger.warning(
                f"Evolution agent finished {_MAX_EVOLUTION_ITERATIONS} iterations "
                f"without signalling {EVOLUTION_COMPLETE} or {EVOLUTION_FAILED}"
            )
            return None

        if has_tool_calls:
            logger.debug(f"Evolution agent used tools (iter {iteration + 1}/{_MAX_EVOLUTION_ITERATIONS})")
        else:
            # No tools, no token — nudge the LLM
            logger.debug(
                f"Evolution agent produced content without token or tools "
                f"(iter {iteration + 1}/{_MAX_EVOLUTION_ITERATIONS})"
            )

        # Iteration guidance
        remaining = _MAX_EVOLUTION_ITERATIONS - iteration - 1
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Iteration {iteration + 1}/{_MAX_EVOLUTION_ITERATIONS} complete "
                    f"({remaining} remaining). "
                    f"If your edit is ready, output it and include {EVOLUTION_COMPLETE} "
                    f"at the end. "
                    f"If you cannot complete this evolution, output {EVOLUTION_FAILED} "
                    f"with a reason. "
                    f"Otherwise, continue gathering information with tools."
                ),
            }
        )

    # Should never reach here (is_last handles the final iteration)
    return None


def parse_evolution_output(content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract edit content or failure reason from LLM output.

    MUST only be called when ``EVOLUTION_COMPLETE`` or
    ``EVOLUTION_FAILED`` is present in *content*.

    Returns ``(clean_content, failure_reason)``:
      - ``(content, None)`` — ``EVOLUTION_COMPLETE`` found.
      - ``(None, reason)``  — ``EVOLUTION_FAILED`` found.
    """
    stripped = content.strip()

    # Failure takes priority (if both tokens appear, treat as failure)
    if EVOLUTION_FAILED in stripped:
        idx = stripped.index(EVOLUTION_FAILED)
        reason_part = stripped[idx + len(EVOLUTION_FAILED) :].strip()
        if reason_part.lower().startswith("reason:"):
            reason_part = reason_part[len("reason:") :].strip()
        reason = reason_part[:500] if reason_part else "LLM declined to produce edit (no reason given)"
        return None, reason

    if EVOLUTION_COMPLETE in stripped:
        clean = stripped.replace(EVOLUTION_COMPLETE, "").strip()
        clean = _strip_markdown_fences(clean)
        return clean, None

    # Caller guarantees a token is present; defensive fallback
    return None, "No completion token found (unexpected)"


async def apply_with_retry(
    evolver,
    *,
    apply_fn,
    initial_content: str,
    skill_dir: Path,
    ctx: EvolutionContext,
    prompt: str,
    cleanup_on_retry: Optional[Path] = None,
) -> "Optional[SkillEditResult]":
    """Apply an edit with retry on failure.

    If the first attempt fails (patch parse error, path mismatch, etc.),
    feeds the error back to the LLM and asks for a corrected version.

    After successful application, runs structural validation.

    Retry conversations are recorded to ``conversations.jsonl`` under
    agent_name="SkillEvolver.retry" so failed apply attempts and LLM
    corrections are preserved for debugging.

    Args:
        evolver: The SkillEvolver instance.
        apply_fn: Callable that takes content str and returns SkillEditResult.
        initial_content: First LLM-generated content to try.
        skill_dir: Skill directory for validation.
        ctx: Evolution context (for retry LLM calls).
        prompt: Original prompt (for retry context).
        cleanup_on_retry: Directory to remove before retrying (for derive/create).
    """
    from openspace.recording import RecordingManager

    current_content = initial_content
    msg_history: List[Dict[str, Any]] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": initial_content},
    ]

    # Track whether we've recorded the retry setup (only on first retry)
    retry_setup_recorded = False

    for attempt in range(_MAX_EVOLUTION_ATTEMPTS):
        # Clean up previous failed attempt (for derive/create)
        if attempt > 0 and cleanup_on_retry and cleanup_on_retry.exists():
            shutil.rmtree(cleanup_on_retry, ignore_errors=True)

        # Apply the edit
        edit_result = apply_fn(current_content)

        if edit_result.ok:
            # Validate the result
            validation_error = _validate_skill_dir(skill_dir)
            if validation_error is None:
                if attempt > 0:
                    logger.info(f"Apply-retry succeeded on attempt {attempt + 1}/{_MAX_EVOLUTION_ATTEMPTS}")
                return edit_result
            else:
                # Validation failed — treat as error for retry
                error_msg = f"Validation failed: {validation_error}"
                logger.warning(
                    f"Apply succeeded but validation failed "
                    f"(attempt {attempt + 1}/{_MAX_EVOLUTION_ATTEMPTS}): "
                    f"{validation_error}"
                )
        else:
            error_msg = edit_result.error or "Unknown apply error"
            logger.warning(f"Apply failed (attempt {attempt + 1}/{_MAX_EVOLUTION_ATTEMPTS}): {error_msg}")

        # Last attempt? Give up.
        if attempt >= _MAX_EVOLUTION_ATTEMPTS - 1:
            logger.error(f"Apply-retry exhausted after {_MAX_EVOLUTION_ATTEMPTS} attempts. Last error: {error_msg}")
            # Clean up any partially created directory
            if cleanup_on_retry and cleanup_on_retry.exists():
                shutil.rmtree(cleanup_on_retry, ignore_errors=True)
            return None

        # Record retry setup on first retry attempt
        if not retry_setup_recorded:
            recorded_retry = [
                {"role": m["role"], "content": _truncate(m.get("content", ""), _RECORDING_MAX_CHARS)}
                for m in msg_history
            ]
            await RecordingManager.record_conversation_setup(
                setup_messages=recorded_retry,
                agent_name="SkillEvolver.retry",
                extra={
                    "evolution_type": ctx.suggestion.evolution_type.value,
                    "target_skills": ctx.suggestion.target_skill_ids,
                    "first_error": error_msg[:300],
                },
            )
            retry_setup_recorded = True

        # Feed error back to LLM for retry, including current file
        # content so the LLM doesn't hallucinate what's on disk.
        current_on_disk = evolver._format_skill_dir_content(skill_dir) if skill_dir.is_dir() else ""
        retry_prompt = f"The previous edit was not successful. This was the error:\n\n{error_msg}\n\n"
        if current_on_disk:
            retry_prompt += (
                f"Here is the CURRENT content of the skill files on disk "
                f"(use this as the ground truth for any SEARCH/REPLACE or "
                f"context anchors):\n\n{_truncate(current_on_disk, _SKILL_CONTENT_MAX_CHARS)}\n\n"
            )
        retry_prompt += "Please fix the issue and generate the edit again. Follow the same output format as before."
        msg_history.append({"role": "user", "content": retry_prompt})

        # Call LLM for corrected version (no tools — just fix the edit)
        model = evolver._model or evolver._llm_client.model
        try:
            result = await evolver._llm_client.complete(
                messages=msg_history,
                model=model,
            )
            new_content = result["message"].get("content", "")
            if not new_content:
                logger.warning("Retry LLM returned empty content")
                continue

            new_content = _strip_markdown_fences(new_content)
            # Strip evolution tokens that the LLM may include in retry responses
            new_content = new_content.replace(EVOLUTION_COMPLETE, "").replace(EVOLUTION_FAILED, "").strip()
            new_content, _ = _extract_change_summary(new_content)
            msg_history.append({"role": "assistant", "content": new_content})
            current_content = new_content

            # Record retry iteration
            await RecordingManager.record_iteration_context(
                iteration=attempt + 1,
                delta_messages=[
                    {"role": "user", "content": _truncate(retry_prompt, _RECORDING_MAX_CHARS)},
                    {"role": "assistant", "content": _truncate(new_content, _RECORDING_MAX_CHARS)},
                ],
                response_metadata={
                    "has_tool_calls": False,
                    "attempt": attempt + 1,
                    "error": error_msg[:300],
                },
                agent_name="SkillEvolver.retry",
            )

        except Exception as e:
            logger.error(f"Retry LLM call failed: {e}")
            continue

    return None
