"""Evolution confirmation — LLM-based candidate validation.

Functions extracted from ``SkillEvolver`` (Epic 5.4):
- ``llm_confirm_evolution`` — Ask LLM to confirm a rule-based candidate
- ``parse_confirmation``    — Parse LLM yes/no response (JSON or keyword)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, List

from openspace.prompts import SkillEnginePrompts
from openspace.skill_engine.skill_utils import truncate
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.skill_engine.types import (
        EvolutionType,
        ExecutionAnalysis,
        SkillRecord,
    )

# Preserve evolver's logger namespace so existing log filters/alerts
# continue to capture confirmation warnings without reconfiguration.
logger = Logger.get_logger("openspace.skill_engine.evolver")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKILL_CONTENT_MAX_CHARS = 12_000  # Max chars of SKILL.md in evolution prompt
_RECORDING_MAX_CHARS = 2_000  # Cap recorded prompt content for data minimization


# ---------------------------------------------------------------------------
# LLM confirmation
# ---------------------------------------------------------------------------

async def llm_confirm_evolution(
    evolver,
    *,
    skill_record: "SkillRecord",
    skill_content: str,
    proposed_type: "EvolutionType",
    proposed_direction: str,
    trigger_context: str,
    recent_analyses: List["ExecutionAnalysis"],
) -> bool:
    """Ask LLM to confirm whether a rule-based evolution candidate
    truly needs evolution.

    Returns True if LLM agrees, False otherwise.
    This prevents false positives from rigid threshold-based rules.

    The confirmation prompt and response are recorded (truncated) to
    ``conversations.jsonl`` under agent_name="SkillEvolver.confirm".
    """
    from openspace.recording import RecordingManager

    analysis_ctx = evolver._format_analysis_context(recent_analyses)

    prompt = SkillEnginePrompts.evolution_confirm(
        skill_id=skill_record.skill_id,
        skill_content=truncate(skill_content, _SKILL_CONTENT_MAX_CHARS // 2),
        proposed_type=proposed_type.value,
        proposed_direction=proposed_direction,
        trigger_context=trigger_context,
        recent_analyses=analysis_ctx,
    )

    confirm_messages = [{"role": "user", "content": prompt}]

    # Record truncated confirmation setup — full prompt goes to LLM only
    recorded_messages = [
        {"role": m["role"], "content": truncate(m["content"], _RECORDING_MAX_CHARS)}
        for m in confirm_messages
    ]
    await RecordingManager.record_conversation_setup(
        setup_messages=recorded_messages,
        agent_name="SkillEvolver.confirm",
        extra={
            "skill_id": skill_record.skill_id,
            "proposed_type": proposed_type.value,
            "trigger_context": trigger_context[:200],
        },
    )

    model = evolver._model or evolver._llm_client.model
    try:
        result = await evolver._llm_client.complete(
            messages=confirm_messages,
            model=model,
        )
        content = result["message"].get("content", "").strip().lower()
        confirmed = evolver._parse_confirmation(content)

        # Record truncated confirmation response
        await RecordingManager.record_iteration_context(
            iteration=1,
            delta_messages=[
                {"role": "assistant", "content": truncate(content, _RECORDING_MAX_CHARS)},
            ],
            response_metadata={
                "has_tool_calls": False,
                "confirmed": confirmed,
            },
            agent_name="SkillEvolver.confirm",
        )

        return confirmed
    except Exception as e:
        logger.warning("LLM confirmation failed, defaulting to skip: %s", e)
        return False


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_confirmation(response: str) -> bool:
    """Parse LLM confirmation response (expects JSON with 'proceed' field).

    Parsing order:
    1. JSON ``{"proceed": true/false}`` (with markdown-fence stripping)
    2. Keyword matching (yes/confirm → True, no/reject/skip → False)
    3. Default: False (ambiguous → skip costly evolution)
    """
    # Try JSON parse first
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return bool(data.get("proceed", False))
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: keyword matching.
    # - yes/no use strict word boundaries to avoid false positives
    #   (e.g. "know" matching "no").
    # - confirm/reject/skip use stem-style matching so that common
    #   LLM variants like "confirmed", "rejected", "skipping" still
    #   parse correctly.
    _wb = re.search  # shorthand
    if (
        any(w in response for w in ('"proceed": true', "proceed: true"))
        or _wb(r"\byes\b", response)
        or _wb(r"\bconfirm\w*\b", response)
    ):
        return True
    if (
        any(w in response for w in ('"proceed": false', "proceed: false"))
        or _wb(r"\bno\b", response)
        or _wb(r"\breject\w*\b", response)
        or _wb(r"\bskip\w*\b", response)
    ):
        return False

    # Default: skip — ambiguous response should not trigger costly evolution
    logger.debug("LLM confirmation response was ambiguous, defaulting to skip")
    return False
