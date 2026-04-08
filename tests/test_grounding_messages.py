"""Tests for scion.agents.grounding.messages — message safety helpers.

Epic 5.7 extraction: cap_message_content, truncate_messages.
"""

from __future__ import annotations

import pytest

from scion.agents.grounding.messages import (
    _MAX_SINGLE_CONTENT_CHARS,
    cap_message_content,
    truncate_messages,
)


# ── cap_message_content ──────────────────────────────────────────


class TestCapMessageContent:
    def test_short_messages_untouched(self):
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = cap_message_content(msgs)
        assert result[1]["content"] == "hello"
        assert result[2]["content"] == "world"

    def test_system_message_never_truncated(self):
        long = "x" * (_MAX_SINGLE_CONTENT_CHARS + 1000)
        msgs = [{"role": "system", "content": long}]
        cap_message_content(msgs)
        assert len(msgs[0]["content"]) == len(long)

    def test_oversized_user_message_truncated(self):
        long = "A" * 50_000
        msgs = [{"role": "user", "content": long}]
        cap_message_content(msgs)
        assert len(msgs[0]["content"]) < 50_000
        assert "truncated" in msgs[0]["content"]

    def test_oversized_assistant_message_truncated(self):
        long = "B" * 50_000
        msgs = [{"role": "assistant", "content": long}]
        cap_message_content(msgs)
        assert "truncated" in msgs[0]["content"]

    def test_preserves_start_and_end(self):
        half = _MAX_SINGLE_CONTENT_CHARS // 2
        content = "START" + "x" * 50_000 + "END"
        msgs = [{"role": "tool", "content": content}]
        cap_message_content(msgs)
        result = msgs[0]["content"]
        assert result.startswith("START")
        assert result.endswith("END")

    def test_non_string_content_ignored(self):
        msgs = [{"role": "assistant", "content": None}]
        cap_message_content(msgs)
        assert msgs[0]["content"] is None

    def test_missing_content_key_ignored(self):
        msgs = [{"role": "assistant"}]
        cap_message_content(msgs)
        assert "content" not in msgs[0]

    def test_custom_cap(self):
        msgs = [{"role": "user", "content": "x" * 200}]
        cap_message_content(msgs, cap=100)
        assert len(msgs[0]["content"]) < 200
        assert "truncated" in msgs[0]["content"]

    def test_returns_same_list(self):
        msgs = [{"role": "user", "content": "short"}]
        assert cap_message_content(msgs) is msgs

    def test_multiple_oversized_messages(self):
        msgs = [
            {"role": "user", "content": "x" * 50_000},
            {"role": "assistant", "content": "y" * 50_000},
            {"role": "tool", "content": "z" * 50_000},
        ]
        cap_message_content(msgs)
        for m in msgs:
            assert "truncated" in m["content"]


# ── truncate_messages ────────────────────────────────────────────


class TestTruncateMessages:
    def _make_messages(self, count: int) -> list:
        """Build a message list: system + user + N conversation messages."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial instruction"},
        ]
        for i in range(count):
            role = "assistant" if i % 2 == 0 else "user"
            msgs.append({"role": role, "content": f"msg-{i}"})
        return msgs

    def test_short_history_untouched(self):
        msgs = self._make_messages(5)
        result = truncate_messages(msgs)
        assert len(result) == len(msgs)

    def test_keeps_system_and_initial_user(self):
        msgs = self._make_messages(100)
        # Force low token budget
        result = truncate_messages(msgs, keep_recent=4, max_tokens_estimate=1)
        roles = [m["role"] for m in result]
        assert roles[0] == "system"
        assert result[1]["content"] == "initial instruction"

    def test_keeps_recent_messages(self):
        msgs = self._make_messages(100)
        result = truncate_messages(msgs, keep_recent=4, max_tokens_estimate=1)
        # Should have: system(1) + user(1) + recent(8) = 10
        assert len(result) == 10

    def test_under_budget_no_truncation(self):
        msgs = self._make_messages(20)
        result = truncate_messages(msgs, max_tokens_estimate=1_000_000)
        assert len(result) == len(msgs)

    def test_preserves_message_content(self):
        msgs = self._make_messages(50)
        result = truncate_messages(msgs, keep_recent=2, max_tokens_estimate=1)
        # Last messages should be from the original list
        last_content = result[-1]["content"]
        assert last_content.startswith("msg-")

    def test_default_keep_recent_is_8(self):
        msgs = self._make_messages(200)
        result = truncate_messages(msgs, max_tokens_estimate=1)
        # system(1) + user(1) + recent(16) = 18
        assert len(result) == 18

    def test_custom_cap_forwarded(self):
        """Verify the cap parameter flows through to cap_message_content."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        # Add enough oversized messages to trigger truncation
        for i in range(20):
            msgs.append({"role": "assistant", "content": "x" * 500})
        result = truncate_messages(msgs, keep_recent=2, max_tokens_estimate=1, cap=200)
        # Messages should have been capped at 200 chars
        for m in result:
            if m["role"] != "system" and m["role"] != "user":
                assert len(m["content"]) <= 250  # some overhead from truncation marker


# ── Delegation seam tests ──────────────────────────────────────────


class TestDelegationSeams:
    """Verify grounding_agent.py properly delegates to messages module."""

    def test_cap_message_content_is_classmethod(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert isinstance(
            GroundingAgent.__dict__.get("_cap_message_content"), classmethod
        )

    def test_max_single_content_chars_matches(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert GroundingAgent._MAX_SINGLE_CONTENT_CHARS == _MAX_SINGLE_CONTENT_CHARS

    def test_truncate_messages_exists(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert callable(getattr(GroundingAgent, "_truncate_messages", None))
