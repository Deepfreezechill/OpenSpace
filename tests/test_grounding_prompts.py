"""Tests for scion.agents.grounding.prompts — prompt construction helpers.

Epic 5.8 extraction: default_system_prompt, construct_messages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scion.agents.grounding.prompts import construct_messages, default_system_prompt


class _FakeAgent:
    """Minimal stand-in for GroundingAgent instance state."""

    def __init__(self):
        self._backend_scope = ["gui", "shell"]
        self._system_prompt = "You are a test agent."
        self._skill_context = None
        self._active_skill_ids = []


# ── default_system_prompt ──────────────────────────────────────────


class TestDefaultSystemPrompt:
    @patch("scion.agents.grounding.prompts.GroundingAgentPrompts")
    def test_calls_build_system_prompt(self, mock_prompts):
        mock_prompts.build_system_prompt.return_value = "sys prompt"
        agent = _FakeAgent()
        result = default_system_prompt(agent)
        mock_prompts.build_system_prompt.assert_called_once_with(["gui", "shell"])
        assert result == "sys prompt"


# ── construct_messages ─────────────────────────────────────────────


class TestConstructMessages:
    def test_basic_messages(self):
        agent = _FakeAgent()
        ctx = {"instruction": "do something"}
        msgs = construct_messages(agent, ctx)
        assert msgs[0] == {"role": "system", "content": "You are a test agent."}
        assert msgs[-1] == {"role": "user", "content": "do something"}

    def test_raises_on_missing_instruction(self):
        agent = _FakeAgent()
        with pytest.raises(ValueError, match="instruction"):
            construct_messages(agent, {})

    def test_raises_on_empty_instruction(self):
        agent = _FakeAgent()
        with pytest.raises(ValueError, match="instruction"):
            construct_messages(agent, {"instruction": ""})

    @patch("scion.agents.grounding.prompts.GroundingAgentPrompts")
    def test_includes_workspace_dir(self, mock_prompts):
        mock_prompts.workspace_directory.return_value = "workspace info"
        agent = _FakeAgent()
        ctx = {"instruction": "go", "workspace_dir": "/tmp/work"}
        msgs = construct_messages(agent, ctx)
        # system + workspace_dir system msg + user
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert "workspace info" in system_contents
        mock_prompts.workspace_directory.assert_called_once_with("/tmp/work")

    @patch("scion.agents.grounding.prompts.GroundingAgentPrompts")
    def test_includes_matching_files(self, mock_prompts):
        mock_prompts.workspace_matching_files.return_value = "matched!"
        agent = _FakeAgent()
        ctx = {
            "instruction": "go",
            "workspace_artifacts": {
                "has_files": True,
                "files": ["a.py"],
                "matching_files": ["a.py"],
                "recent_files": [],
            },
        }
        msgs = construct_messages(agent, ctx)
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert "matched!" in system_contents

    @patch("scion.agents.grounding.prompts.GroundingAgentPrompts")
    def test_includes_recent_files(self, mock_prompts):
        mock_prompts.workspace_recent_files.return_value = "recent!"
        agent = _FakeAgent()
        ctx = {
            "instruction": "go",
            "workspace_artifacts": {
                "has_files": True,
                "files": ["a.py", "b.py", "c.py"],
                "matching_files": [],
                "recent_files": ["a.py", "b.py"],
            },
        }
        msgs = construct_messages(agent, ctx)
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert "recent!" in system_contents

    @patch("scion.agents.grounding.prompts.GroundingAgentPrompts")
    def test_includes_file_list_fallback(self, mock_prompts):
        mock_prompts.workspace_file_list.return_value = "file list!"
        agent = _FakeAgent()
        ctx = {
            "instruction": "go",
            "workspace_artifacts": {
                "has_files": True,
                "files": ["a.py"],
                "matching_files": [],
                "recent_files": [],
            },
        }
        msgs = construct_messages(agent, ctx)
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert "file list!" in system_contents

    def test_includes_skill_context(self):
        agent = _FakeAgent()
        agent._skill_context = "Use the search tool."
        agent._active_skill_ids = ["search-skill"]
        ctx = {"instruction": "find files"}
        msgs = construct_messages(agent, ctx)
        system_contents = [m["content"] for m in msgs if m["role"] == "system"]
        assert "Use the search tool." in system_contents

    def test_no_skill_context_when_none(self):
        agent = _FakeAgent()
        ctx = {"instruction": "go"}
        msgs = construct_messages(agent, ctx)
        assert len(msgs) == 2  # system + user only

    def test_no_workspace_artifacts_when_no_files(self):
        agent = _FakeAgent()
        ctx = {
            "instruction": "go",
            "workspace_artifacts": {"has_files": False},
        }
        msgs = construct_messages(agent, ctx)
        assert len(msgs) == 2  # system + user only


# ── Delegation seam tests ──────────────────────────────────────────


class TestDelegationSeams:
    """Verify grounding_agent.py properly delegates to prompts module."""

    def test_construct_messages_exists(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert callable(getattr(GroundingAgent, "construct_messages", None))

    def test_default_system_prompt_exists(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert callable(getattr(GroundingAgent, "_default_system_prompt", None))

    def test_process_exists(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert callable(getattr(GroundingAgent, "process", None))
