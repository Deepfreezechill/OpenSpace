"""Message safety helpers for GroundingAgent.

Functions to cap oversized message content and truncate long
conversation histories before LLM calls.
Extracted from grounding_agent.py (Epic 5.7).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openspace.utils.logging import Logger

logger = Logger.get_logger("openspace.agents.grounding_agent")

# Maximum characters allowed in a single message content field.
_MAX_SINGLE_CONTENT_CHARS = 30_000


def cap_message_content(
    messages: List[Dict[str, Any]],
    cap: int = _MAX_SINGLE_CONTENT_CHARS,
) -> List[Dict[str, Any]]:
    """Truncate oversized individual message contents in-place.

    Targets tool-result messages and assistant messages that can
    carry enormous file contents (read_file on large CSVs/scripts).
    System messages and the first user instruction are never touched.

    Args:
        messages: The message list (mutated in-place).
        cap: Maximum character count per message.

    Returns:
        The same *messages* list (for chaining).
    """
    trimmed = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= cap:
            continue
        if msg.get("role") == "system":
            continue
        original_len = len(content)
        msg["content"] = (
            content[: cap // 2]
            + f"\n\n... [truncated {original_len - cap:,} chars] ...\n\n"
            + content[-(cap // 2) :]
        )
        trimmed += 1
    if trimmed:
        logger.info(f"Capped {trimmed} oversized message(s) to {cap:,} chars each")
    return messages


def truncate_messages(
    messages: List[Dict[str, Any]],
    keep_recent: int = 8,
    max_tokens_estimate: int = 120_000,
) -> List[Dict[str, Any]]:
    """Trim conversation history to fit within token budget.

    Steps:
    1. Cap any single oversized message (via :func:`cap_message_content`).
    2. If total estimated tokens exceed *max_tokens_estimate*, keep only
       the system messages, the first user instruction, and the most
       recent *keep_recent* conversation rounds.

    Args:
        messages: Full message list.
        keep_recent: Number of recent conversation rounds to preserve.
        max_tokens_estimate: Approximate token budget.

    Returns:
        Possibly shortened message list.
    """
    messages = cap_message_content(messages)

    if len(messages) <= keep_recent + 2:  # +2 for system and initial user
        return messages

    total_text = json.dumps(messages, ensure_ascii=False)
    estimated_tokens = len(total_text) // 4

    if estimated_tokens < max_tokens_estimate:
        return messages

    logger.info(
        f"Truncating message history: {len(messages)} messages, "
        f"~{estimated_tokens:,} tokens -> keeping recent {keep_recent} rounds"
    )

    system_messages: List[Dict[str, Any]] = []
    user_instruction = None
    conversation_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_messages.append(msg)
        elif role == "user" and user_instruction is None:
            user_instruction = msg
        else:
            conversation_messages.append(msg)

    recent_messages = (
        conversation_messages[-(keep_recent * 2) :] if conversation_messages else []
    )

    truncated = system_messages.copy()
    if user_instruction:
        truncated.append(user_instruction)
    truncated.extend(recent_messages)

    logger.info(
        f"After truncation: {len(truncated)} messages, "
        f"~{len(json.dumps(truncated, ensure_ascii=False)) // 4:,} tokens (estimated)"
    )

    return truncated
