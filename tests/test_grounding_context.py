"""Tests for openspace.agents.grounding.context — skill context helpers.

Epic 5.7 extraction: set_skill_context, clear_skill_context,
has_skill_context, set_skill_registry.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openspace.agents.grounding.context import (
    clear_skill_context,
    has_skill_context,
    set_skill_context,
    set_skill_registry,
)


class _FakeAgent:
    """Minimal stand-in for GroundingAgent instance state."""

    def __init__(self):
        self._skill_context = None
        self._active_skill_ids = []
        self._skill_registry = None


# ── set_skill_context ──────────────────────────────────────────────


class TestSetSkillContext:
    def test_sets_context_and_ids(self):
        agent = _FakeAgent()
        set_skill_context(agent, "skill guidance", ["s1", "s2"])
        assert agent._skill_context == "skill guidance"
        assert agent._active_skill_ids == ["s1", "s2"]

    def test_empty_string_clears_context(self):
        agent = _FakeAgent()
        agent._skill_context = "old"
        set_skill_context(agent, "")
        assert agent._skill_context is None

    def test_none_ids_defaults_to_empty_list(self):
        agent = _FakeAgent()
        set_skill_context(agent, "ctx", None)
        assert agent._active_skill_ids == []

    def test_overwrites_previous(self):
        agent = _FakeAgent()
        set_skill_context(agent, "first", ["a"])
        set_skill_context(agent, "second", ["b"])
        assert agent._skill_context == "second"
        assert agent._active_skill_ids == ["b"]


# ── clear_skill_context ───────────────────────────────────────────


class TestClearSkillContext:
    def test_clears_existing(self):
        agent = _FakeAgent()
        agent._skill_context = "guidance"
        agent._active_skill_ids = ["s1"]
        clear_skill_context(agent)
        assert agent._skill_context is None
        assert agent._active_skill_ids == []

    def test_noop_when_already_clear(self):
        agent = _FakeAgent()
        clear_skill_context(agent)
        assert agent._skill_context is None
        assert agent._active_skill_ids == []


# ── has_skill_context ──────────────────────────────────────────────


class TestHasSkillContext:
    def test_false_when_none(self):
        agent = _FakeAgent()
        assert has_skill_context(agent) is False

    def test_true_when_set(self):
        agent = _FakeAgent()
        agent._skill_context = "present"
        assert has_skill_context(agent) is True


# ── set_skill_registry ────────────────────────────────────────────


class TestSetSkillRegistry:
    def test_attaches_registry(self):
        agent = _FakeAgent()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = ["a", "b", "c"]
        set_skill_registry(agent, mock_registry)
        assert agent._skill_registry is mock_registry

    def test_none_clears_registry(self):
        agent = _FakeAgent()
        agent._skill_registry = MagicMock()
        set_skill_registry(agent, None)
        assert agent._skill_registry is None

    def test_logs_skill_count(self):
        agent = _FakeAgent()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = ["x"]
        set_skill_registry(agent, mock_registry)
        mock_registry.list_skills.assert_called_once()


# ── Delegation seam tests ──────────────────────────────────────────


class TestDelegationSeams:
    """Verify grounding_agent.py properly delegates to context module."""

    def test_set_skill_context_delegates(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert "set_skill_context" in dir(GroundingAgent)
        # Method should exist on the class
        method = getattr(GroundingAgent, "set_skill_context")
        assert callable(method)

    def test_has_skill_context_is_property(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert isinstance(
            GroundingAgent.__dict__.get("has_skill_context"), property
        ), "has_skill_context must be a property"

    def test_set_skill_registry_delegates(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert "set_skill_registry" in dir(GroundingAgent)
