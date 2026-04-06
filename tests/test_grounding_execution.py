"""Tests for openspace.agents.grounding.execution — core execution loop.

Epic 5.8 extraction: process, _build_retrieved_tools_list.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.agents.grounding.execution import (
    _MAX_CONSECUTIVE_EMPTY,
    _build_retrieved_tools_list,
    process,
)


class _FakeAgent:
    """Minimal stand-in for GroundingAgent instance state."""

    def __init__(self):
        self.step = 0
        self._current_instruction = None
        self._skill_context = None
        self._active_skill_ids = []
        self._max_iterations = 15
        self._recording_manager = None
        self._last_tools = []
        self._llm_client = AsyncMock()
        self.grounding_client = None

        # Mock methods that process() calls on the agent
        self._check_workspace_artifacts = AsyncMock(
            return_value={"has_files": False, "files": []}
        )
        self._get_available_tools = AsyncMock(return_value=[])
        self.construct_messages = MagicMock(
            return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
            ]
        )
        self._cap_message_content = MagicMock(side_effect=lambda m: m)
        self._truncate_messages = MagicMock(side_effect=lambda m, **kw: m)
        self._visual_analysis_callback = AsyncMock()
        self._build_final_result = AsyncMock(
            return_value={"status": "completed", "response": "done"}
        )
        self._record_agent_execution = AsyncMock()
        self.increment_step = MagicMock()


# ── _build_retrieved_tools_list ────────────────────────────────────


class TestBuildRetrievedToolsList:
    def test_empty_tools(self):
        assert _build_retrieved_tools_list([], None) == []

    def test_basic_tool_info(self):
        tool = MagicMock()
        tool.name = "read_file"
        tool.description = "Read a file"
        tool._runtime_info = None
        del tool.backend_type  # ensure hasattr fails
        result = _build_retrieved_tools_list([tool], None)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read a file"

    def test_runtime_info_preferred(self):
        tool = MagicMock()
        tool.name = "shell"
        tool.description = "Run shell"
        tool._runtime_info = MagicMock()
        tool._runtime_info.backend = MagicMock()
        tool._runtime_info.backend.value = "shell"
        tool._runtime_info.server_name = "local"
        result = _build_retrieved_tools_list([tool], None)
        assert result[0]["backend"] == "shell"
        assert result[0]["server_name"] == "local"

    def test_similarity_score_attached(self):
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search"
        tool._runtime_info = None
        del tool.backend_type
        debug_info = {"tool_scores": [{"name": "search", "score": 0.95}]}
        result = _build_retrieved_tools_list([tool], debug_info)
        assert result[0]["similarity_score"] == 0.95

    def test_no_score_when_name_mismatch(self):
        tool = MagicMock()
        tool.name = "read_file"
        tool.description = ""
        tool._runtime_info = None
        del tool.backend_type
        debug_info = {"tool_scores": [{"name": "other", "score": 0.5}]}
        result = _build_retrieved_tools_list([tool], debug_info)
        assert "similarity_score" not in result[0]


# ── process ────────────────────────────────────────────────────────


class TestProcess:
    @pytest.mark.asyncio
    async def test_returns_error_on_empty_instruction(self):
        agent = _FakeAgent()
        result = await process(agent, {"instruction": ""})
        assert result["status"] == "error"
        assert "No instruction" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_instruction(self):
        agent = _FakeAgent()
        result = await process(agent, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_single_iteration_complete(self):
        """Agent completes in 1 iteration when LLM returns COMPLETE token."""
        agent = _FakeAgent()
        agent._llm_client.complete = AsyncMock(
            return_value={
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": "Done! <COMPLETE>"},
                ],
                "message": {"content": "Done! <COMPLETE>"},
                "tool_results": [],
                "has_tool_calls": False,
            }
        )
        with patch("openspace.recording.RecordingManager") as mock_rm:
            mock_rm.record_retrieved_tools = AsyncMock()
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()
            result = await process(agent, {"instruction": "test task"})
        assert result["status"] == "completed"
        agent.increment_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_stores_current_instruction(self):
        agent = _FakeAgent()
        agent._llm_client.complete = AsyncMock(
            return_value={
                "messages": [{"role": "assistant", "content": "<COMPLETE>"}],
                "message": {"content": "<COMPLETE>"},
                "tool_results": [],
                "has_tool_calls": False,
            }
        )
        with patch("openspace.recording.RecordingManager") as mock_rm:
            mock_rm.record_retrieved_tools = AsyncMock()
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()
            await process(agent, {"instruction": "my task"})
        assert agent._current_instruction == "my task"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        agent = _FakeAgent()
        agent._llm_client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("openspace.recording.RecordingManager") as mock_rm:
            mock_rm.record_retrieved_tools = AsyncMock()
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()
            result = await process(agent, {"instruction": "crash"})
        assert result["status"] == "error"
        assert "boom" in result["error"]
        agent.increment_step.assert_called_once()

    def test_max_consecutive_empty_constant(self):
        assert _MAX_CONSECUTIVE_EMPTY == 5


# ── Delegation seam tests ──────────────────────────────────────────


class TestDelegationSeams:
    def test_process_is_coroutine(self):
        import asyncio
        import inspect

        from openspace.agents.grounding_agent import GroundingAgent

        method = getattr(GroundingAgent, "process")
        assert inspect.iscoroutinefunction(method)
