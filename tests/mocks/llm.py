"""Deterministic LLM mock for testing.

Provides ``MockLLMClient`` — a drop-in replacement for
``openspace.llm.LLMClient`` that returns pre-configured responses
from a response pool instead of calling a real LLM API.

Usage in tests::

    client = MockLLMClient(responses=["Hello!", "Goodbye!"])
    result = await client.complete("Say hi")
    assert result["message"]["content"] == "Hello!"

Responses cycle: after the pool is exhausted it wraps around.
"""

from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_RESPONSES_DIR = Path(__file__).parent / "responses"

# Pre-built response templates
CHAT_RESPONSE = _RESPONSES_DIR / "chat_completion.json"
TOOL_CALL_RESPONSE = _RESPONSES_DIR / "tool_call.json"
SKILL_ANALYSIS_RESPONSE = _RESPONSES_DIR / "skill_analysis.json"


def _make_completion(
    content: str,
    *,
    model: str = "mock-model",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a response dict matching the production LLMClient.complete() shape.

    Production returns::

        {
            "message": {"role": "assistant", "content": "..."},
            "tool_results": [...],
            "messages": [...],
            "has_tool_calls": bool,
            "iteration_summary": ""
        }
    """
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    tool_results: List[Dict[str, Any]] = []
    has_tool_calls = False

    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = ""  # Production uses empty string, not None
        has_tool_calls = True

    return {
        "message": message,
        "tool_results": tool_results,
        "messages": [message],
        "has_tool_calls": has_tool_calls,
        "iteration_summary": None,  # Production defaults to None, not ""
    }


def load_response_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON response template from disk."""
    with open(path) as f:
        return json.load(f)


class MockLLMClient:
    """Deterministic mock that replaces ``LLMClient`` for testing.

    Parameters
    ----------
    responses : sequence of str | dict
        If str — each string becomes the assistant ``content`` of a
        chat-completion response.
        If dict — used as-is (must look like a litellm response).
    model : str
        Model name echoed in responses.
    record_calls : bool
        When ``True`` (default), every ``complete()`` invocation is
        appended to ``self.calls`` for later assertion.
    """

    def __init__(
        self,
        responses: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        *,
        model: str = "mock-model",
        record_calls: bool = True,
    ) -> None:
        if responses is None:
            responses = ["Mock LLM response."]

        built: List[Dict[str, Any]] = []
        for r in responses:
            if isinstance(r, str):
                built.append(_make_completion(r, model=model))
            else:
                built.append(r)

        self.model = model
        self._pool = cycle(built)
        self._responses_list = built
        self.record_calls = record_calls
        self.calls: List[Dict[str, Any]] = []
        self.call_count = 0

        # Mirror real LLMClient attributes for compatibility
        self.enable_thinking = False
        self.rate_limit_delay = 0.0
        self.max_retries = 1
        self.retry_delay = 0.0
        self.timeout = 30.0
        self.summarize_threshold_chars = 200000
        self.enable_tool_result_summarization = False
        self.litellm_kwargs: Dict[str, Any] = {}

    async def complete(
        self,
        messages: Union[List[Dict], str],
        tools: Optional[List] = None,
        execute_tools: bool = False,
        summary_prompt: Optional[str] = None,
        tool_result_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return the next response from the pool.

        Matches the signature of ``LLMClient.complete()`` so it can be
        used as a drop-in replacement.  Tool execution is always
        skipped — tests should assert on tool_calls in the response
        if needed.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        call_record = {
            "messages": messages,
            "tools": tools,
            "execute_tools": execute_tools,
            "kwargs": kwargs,
        }

        if self.record_calls:
            self.calls.append(call_record)

        self.call_count += 1
        return next(self._pool)

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Return the most recent call record, or None."""
        return self.calls[-1] if self.calls else None

    def reset(self) -> None:
        """Reset call history and re-cycle the response pool."""
        self.calls.clear()
        self.call_count = 0
        self._pool = cycle(self._responses_list)
