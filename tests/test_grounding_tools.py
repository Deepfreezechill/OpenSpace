"""Tests for scion.agents.grounding.tools — tool retrieval helpers.

Epic 5.9 extraction: _get_available_tools, _load_all_tools.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scion.agents.grounding.tools import _get_available_tools, _load_all_tools


class _FakeAgent:
    """Minimal stand-in for GroundingAgent instance state."""

    def __init__(self):
        self.grounding_client = None
        self._backend_scope = ["gui", "shell", "mcp"]
        self._skill_context = None
        self._active_skill_ids = []
        self._tool_retrieval_llm = None
        self._llm_client = MagicMock()
        self._skill_registry = None
        self._skill_store = None

    @property
    def has_skill_context(self) -> bool:
        return self._skill_context is not None

    async def _load_all_tools(self, grounding_client):
        """Delegate for MRO — calls module function."""
        from scion.agents.grounding.tools import _load_all_tools
        return await _load_all_tools(self, grounding_client)


# ── _get_available_tools ───────────────────────────────────────────


class TestGetAvailableTools:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_grounding_client(self):
        agent = _FakeAgent()
        agent.grounding_client = None
        result = await _get_available_tools(agent, "do something")
        assert result == []

    @pytest.mark.asyncio
    async def test_calls_auto_search(self):
        agent = _FakeAgent()
        mock_gc = AsyncMock()
        mock_gc.get_tools_with_auto_search = AsyncMock(return_value=["tool1", "tool2"])
        agent.grounding_client = mock_gc

        result = await _get_available_tools(agent, "open browser")
        assert result == ["tool1", "tool2"]
        mock_gc.get_tools_with_auto_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_adds_shell_backend_when_skill_context(self):
        agent = _FakeAgent()
        agent._backend_scope = ["gui", "mcp"]  # no shell
        agent._skill_context = "some skill guidance"
        mock_gc = AsyncMock()
        mock_gc.get_tools_with_auto_search = AsyncMock(return_value=[])
        agent.grounding_client = mock_gc

        await _get_available_tools(agent, "task")

        call_kwargs = mock_gc.get_tools_with_auto_search.call_args
        backends = call_kwargs.kwargs.get("backend") or call_kwargs[1].get("backend")
        from scion.grounding.core.types import BackendType

        assert BackendType.SHELL in backends

    @pytest.mark.asyncio
    async def test_fallback_on_auto_search_failure(self):
        agent = _FakeAgent()
        mock_gc = AsyncMock()
        mock_gc.get_tools_with_auto_search = AsyncMock(side_effect=RuntimeError("boom"))
        mock_gc.list_tools = AsyncMock(return_value=["fallback_tool"])
        agent.grounding_client = mock_gc

        result = await _get_available_tools(agent, "task")
        assert "fallback_tool" in result

    @pytest.mark.asyncio
    async def test_appends_retrieve_skill_tool(self):
        agent = _FakeAgent()
        mock_gc = AsyncMock()
        mock_gc.get_tools_with_auto_search = AsyncMock(return_value=[])
        agent.grounding_client = mock_gc

        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = ["skill_a"]
        agent._skill_registry = mock_registry

        with patch("scion.agents.grounding.tools.RetrieveSkillTool", create=True) as MockRST:
            # Patch the lazy import inside the function
            mock_tool = MagicMock()
            MockRST.return_value = mock_tool

            # Need to patch the import inside the function
            import scion.agents.grounding.tools as tools_mod

            with patch.dict("sys.modules", {"scion.skill_engine.retrieve_tool": MagicMock(RetrieveSkillTool=MockRST)}):
                result = await _get_available_tools(agent, "task")
                assert mock_tool in result

    @pytest.mark.asyncio
    async def test_uses_tool_retrieval_llm_when_set(self):
        agent = _FakeAgent()
        retrieval_llm = MagicMock()
        agent._tool_retrieval_llm = retrieval_llm
        mock_gc = AsyncMock()
        mock_gc.get_tools_with_auto_search = AsyncMock(return_value=[])
        agent.grounding_client = mock_gc

        await _get_available_tools(agent, "task")
        call_kwargs = mock_gc.get_tools_with_auto_search.call_args
        assert call_kwargs.kwargs.get("llm_callable") is retrieval_llm


# ── _load_all_tools ────────────────────────────────────────────────


class TestLoadAllTools:
    @pytest.mark.asyncio
    async def test_loads_from_all_backends(self):
        agent = _FakeAgent()
        agent._backend_scope = ["gui", "shell"]
        mock_gc = AsyncMock()
        mock_gc.list_tools = AsyncMock(side_effect=[["t1"], ["t2", "t3"]])

        result = await _load_all_tools(agent, mock_gc)
        assert result == ["t1", "t2", "t3"]
        assert mock_gc.list_tools.await_count == 2

    @pytest.mark.asyncio
    async def test_skips_failed_backends(self):
        agent = _FakeAgent()
        agent._backend_scope = ["gui", "shell"]
        mock_gc = AsyncMock()
        mock_gc.list_tools = AsyncMock(side_effect=[RuntimeError("fail"), ["ok_tool"]])

        result = await _load_all_tools(agent, mock_gc)
        assert result == ["ok_tool"]

    @pytest.mark.asyncio
    async def test_empty_when_all_backends_fail(self):
        agent = _FakeAgent()
        agent._backend_scope = ["gui"]
        mock_gc = AsyncMock()
        mock_gc.list_tools = AsyncMock(side_effect=RuntimeError("fail"))

        result = await _load_all_tools(agent, mock_gc)
        assert result == []


# ── Delegation seam tests ──────────────────────────────────────────


class TestToolsDelegationSeams:
    """Verify grounding_agent.py properly delegates to tools module."""

    def test_get_available_tools_delegates(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert "_get_available_tools" in dir(GroundingAgent)
        method = getattr(GroundingAgent, "_get_available_tools")
        assert callable(method)

    def test_load_all_tools_delegates(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert "_load_all_tools" in dir(GroundingAgent)
        method = getattr(GroundingAgent, "_load_all_tools")
        assert callable(method)
