"""Tests for scion.agents.grounding.results — results & telemetry helpers.

Epic 5.10 extraction: build_iteration_feedback, remove_previous_guidance,
format_tool_executions, check_task_completion, extract_last_assistant_message,
generate_final_summary, build_final_result, record_agent_execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scion.agents.grounding.results import (
    build_final_result,
    build_iteration_feedback,
    check_task_completion,
    extract_last_assistant_message,
    format_tool_executions,
    generate_final_summary,
    record_agent_execution,
    remove_previous_guidance,
)
from scion.prompts import GroundingAgentPrompts


# ── helpers ────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal stand-in for GroundingAgent instance state."""

    def __init__(self):
        self.step = 3
        self.name = "TestAgent"
        self._llm_client = AsyncMock()
        self._recording_manager = None
        self._active_skill_ids = ["skill-a"]

        # These are the staticmethod bindings on real GroundingAgent;
        # for _build_final_result tests we attach module-level functions.
        self._check_task_completion = check_task_completion
        self._format_tool_executions = format_tool_executions
        self._extract_last_assistant_message = extract_last_assistant_message


# ══════════════════════════════════════════════════════════════════════
# build_iteration_feedback
# ══════════════════════════════════════════════════════════════════════


class TestBuildIterationFeedback:
    def test_returns_none_when_no_summary(self):
        assert build_iteration_feedback(1, llm_summary=None) is None

    def test_returns_none_for_empty_string(self):
        assert build_iteration_feedback(1, llm_summary="") is None

    def test_returns_system_message_with_summary(self):
        result = build_iteration_feedback(2, llm_summary="Did stuff")
        assert result is not None
        assert result["role"] == "system"
        assert "## Iteration 2" in result["content"]
        assert "Did stuff" in result["content"]

    def test_guidance_included_by_default(self):
        result = build_iteration_feedback(1, llm_summary="progress")
        assert "---" in result["content"]

    def test_guidance_excluded_when_disabled(self):
        result = build_iteration_feedback(1, llm_summary="progress", add_guidance=False)
        assert "---" not in result["content"]


# ══════════════════════════════════════════════════════════════════════
# remove_previous_guidance
# ══════════════════════════════════════════════════════════════════════


class TestRemovePreviousGuidance:
    def test_strips_guidance_from_iteration_feedback(self):
        msgs = [
            {
                "role": "system",
                "content": "## Iteration 1 Summary\nDid things\n---\nContinue with iteration 2.",
            }
        ]
        remove_previous_guidance(msgs)
        assert "---" not in msgs[0]["content"]
        assert "Did things" in msgs[0]["content"]

    def test_leaves_non_feedback_messages_alone(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are an assistant"},
        ]
        remove_previous_guidance(msgs)
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["content"] == "You are an assistant"

    def test_handles_empty_list(self):
        msgs = []
        remove_previous_guidance(msgs)
        assert msgs == []


# ══════════════════════════════════════════════════════════════════════
# format_tool_executions
# ══════════════════════════════════════════════════════════════════════


class TestFormatToolExecutions:
    def test_empty_list(self):
        assert format_tool_executions([]) == []

    def test_litellm_object_format(self):
        """tool_call is an object with .function attribute (litellm style)."""
        func = SimpleNamespace(name="click", arguments='{"x": 10}')
        tool_call = SimpleNamespace(function=func)
        result_obj = SimpleNamespace(
            status=SimpleNamespace(value="success"),
            content="clicked",
            error=None,
            execution_time=0.5,
            metadata={"k": "v"},
        )
        raw = [{"result": result_obj, "tool_call": tool_call, "backend": "gui", "server_name": "s1"}]

        out = format_tool_executions(raw)
        assert len(out) == 1
        assert out[0]["tool_name"] == "click"
        assert out[0]["arguments"] == {"x": 10}
        assert out[0]["status"] == "success"
        assert out[0]["backend"] == "gui"

    def test_dict_format(self):
        """tool_call is a dict (fallback path)."""
        tool_call = {"function": {"name": "type_text", "arguments": '{"text": "hi"}'}}
        result_obj = SimpleNamespace(
            status="ok",
            content="typed",
            error=None,
            execution_time=0.1,
            metadata={},
        )
        raw = [{"result": result_obj, "tool_call": tool_call}]

        out = format_tool_executions(raw)
        assert out[0]["tool_name"] == "type_text"
        assert out[0]["arguments"] == {"text": "hi"}
        assert out[0]["status"] == "ok"

    def test_none_tool_call(self):
        """tool_call is None — should produce 'unknown' tool_name."""
        result_obj = SimpleNamespace()
        raw = [{"result": result_obj, "tool_call": None}]
        out = format_tool_executions(raw)
        assert out[0]["tool_name"] == "unknown"

    def test_invalid_json_arguments(self):
        """Malformed JSON arguments should produce empty dict."""
        func = SimpleNamespace(name="bad", arguments="not-json")
        tool_call = SimpleNamespace(function=func)
        result_obj = SimpleNamespace()
        raw = [{"result": result_obj, "tool_call": tool_call}]
        out = format_tool_executions(raw)
        assert out[0]["arguments"] == {}


# ══════════════════════════════════════════════════════════════════════
# check_task_completion
# ══════════════════════════════════════════════════════════════════════


class TestCheckTaskCompletion:
    def test_true_when_complete_in_last_assistant(self):
        msgs = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": f"Done {GroundingAgentPrompts.TASK_COMPLETE}"},
        ]
        assert check_task_completion(msgs) is True

    def test_false_when_not_present(self):
        msgs = [
            {"role": "assistant", "content": "Still working..."},
        ]
        assert check_task_completion(msgs) is False

    def test_false_when_no_assistant_messages(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert check_task_completion(msgs) is False

    def test_false_on_empty_list(self):
        assert check_task_completion([]) is False


# ══════════════════════════════════════════════════════════════════════
# extract_last_assistant_message
# ══════════════════════════════════════════════════════════════════════


class TestExtractLastAssistantMessage:
    def test_extracts_last_assistant(self):
        msgs = [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "second"},
        ]
        assert extract_last_assistant_message(msgs) == "second"

    def test_returns_empty_when_none(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert extract_last_assistant_message(msgs) == ""

    def test_returns_empty_for_empty_list(self):
        assert extract_last_assistant_message([]) == ""


# ══════════════════════════════════════════════════════════════════════
# generate_final_summary
# ══════════════════════════════════════════════════════════════════════


class TestGenerateFinalSummary:
    @pytest.mark.asyncio
    async def test_returns_summary_text(self):
        agent = _FakeAgent()
        agent._llm_client.complete.return_value = {
            "message": {"content": "Summary of execution"}
        }
        msgs = [{"role": "assistant", "content": "did things"}]

        text, success, ctx = await generate_final_summary(agent, "do stuff", msgs, 2)

        assert text == "Summary of execution"
        assert success is True
        assert len(ctx) > 0
        agent._llm_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        agent = _FakeAgent()
        agent._llm_client.complete.return_value = {"message": {"content": ""}}
        msgs = [{"role": "assistant", "content": "work"}]

        text, success, _ = await generate_final_summary(agent, "inst", msgs, 3)

        assert "3 iteration(s)" in text
        assert success is True

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        agent = _FakeAgent()
        agent._llm_client.complete.side_effect = RuntimeError("LLM down")
        msgs = [{"role": "assistant", "content": "work"}]

        text, success, _ = await generate_final_summary(agent, "inst", msgs, 1)

        assert "failed to generate summary" in text
        assert "LLM down" not in text  # str(e) must NOT leak to caller (security)
        assert success is False

    @pytest.mark.asyncio
    async def test_strips_tool_messages_and_tool_calls(self):
        agent = _FakeAgent()
        agent._llm_client.complete.return_value = {"message": {"content": "ok"}}
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "resp", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "content": "result"},
        ]

        _, _, ctx = await generate_final_summary(agent, "inst", msgs, 1)

        # tool messages stripped, tool_calls key removed
        roles = [m["role"] for m in ctx]
        assert "tool" not in roles
        assert not any("tool_calls" in m for m in ctx)


# ══════════════════════════════════════════════════════════════════════
# build_final_result
# ══════════════════════════════════════════════════════════════════════


class TestBuildFinalResult:
    @pytest.mark.asyncio
    async def test_complete_case(self):
        agent = _FakeAgent()
        msgs = [
            {"role": "assistant", "content": f"All done. {GroundingAgentPrompts.TASK_COMPLETE}"},
        ]

        result = await build_final_result(
            agent,
            instruction="do it",
            messages=msgs,
            all_tool_results=[],
            iterations=2,
            max_iterations=10,
        )

        assert result["status"] == "success"
        assert GroundingAgentPrompts.TASK_COMPLETE not in result["response"]
        assert "All done." in result["response"]
        assert result["step"] == 3
        assert result["active_skills"] == ["skill-a"]

    @pytest.mark.asyncio
    async def test_incomplete_case(self):
        agent = _FakeAgent()
        msgs = [
            {"role": "assistant", "content": "Still going..."},
        ]

        result = await build_final_result(
            agent,
            instruction="do it",
            messages=msgs,
            all_tool_results=[],
            iterations=10,
            max_iterations=10,
        )

        assert result["status"] == "incomplete"
        assert "max iterations" in result["warning"]
        assert result["response"] == "Still going..."

    @pytest.mark.asyncio
    async def test_passes_optional_fields(self):
        agent = _FakeAgent()
        msgs = [{"role": "assistant", "content": "x"}]

        result = await build_final_result(
            agent,
            instruction="i",
            messages=msgs,
            all_tool_results=[],
            iterations=1,
            max_iterations=5,
            iteration_contexts=[{"iter": 1}],
            retrieved_tools_list=[{"name": "t"}],
            search_debug_info={"scores": []},
        )

        assert result["iteration_contexts"] == [{"iter": 1}]
        assert result["retrieved_tools_list"] == [{"name": "t"}]
        assert result["search_debug_info"] == {"scores": []}


# ══════════════════════════════════════════════════════════════════════
# record_agent_execution
# ══════════════════════════════════════════════════════════════════════


class TestRecordAgentExecution:
    @pytest.mark.asyncio
    async def test_calls_recording_manager(self):
        agent = _FakeAgent()
        agent._recording_manager = AsyncMock()

        result = {
            "response": "done",
            "status": "success",
            "iterations": 2,
            "tool_executions": [
                {"tool_name": "click", "backend": "gui", "status": "success"},
            ],
        }

        await record_agent_execution(agent, result, "do it")

        agent._recording_manager.record_agent_action.assert_awaited_once()
        call_kwargs = agent._recording_manager.record_agent_action.call_args.kwargs
        assert call_kwargs["agent_name"] == "TestAgent"
        assert call_kwargs["action_type"] == "execute"
        assert call_kwargs["metadata"]["step"] == 3

    @pytest.mark.asyncio
    async def test_skips_when_no_recording_manager(self):
        agent = _FakeAgent()
        agent._recording_manager = None

        # Should not raise
        await record_agent_execution(agent, {"tool_executions": []}, "do it")

    @pytest.mark.asyncio
    async def test_handles_empty_tool_executions(self):
        agent = _FakeAgent()
        agent._recording_manager = AsyncMock()

        result = {"response": "x", "status": "ok", "iterations": 1, "tool_executions": []}
        await record_agent_execution(agent, result, "inst")

        call_kwargs = agent._recording_manager.record_agent_action.call_args.kwargs
        assert call_kwargs["reasoning"]["tools_selected"] == []


# ══════════════════════════════════════════════════════════════════════
# Delegation seam tests (GroundingAgent → results module)
# ══════════════════════════════════════════════════════════════════════


class TestGroundingAgentDelegation:
    """Verify GroundingAgent thin delegates route to the results module."""

    def _make_agent(self):
        """Build a GroundingAgent with mocked dependencies."""
        from scion.agents.grounding_agent import GroundingAgent

        agent = GroundingAgent.__new__(GroundingAgent)
        # Minimal state to avoid __init__ side-effects
        agent._backend_scope = ["shell"]
        agent._llm_client = AsyncMock()
        agent._grounding_client = None
        agent._recording_manager = None
        agent._system_prompt = "sys"
        agent._max_iterations = 5
        agent._visual_analysis_timeout = 10.0
        agent._tool_retrieval_llm = None
        agent._visual_analysis_model = None
        agent._skill_context = None
        agent._active_skill_ids = []
        agent._skill_registry = None
        agent._last_tools = []
        agent._step = 0
        agent._name = "test"
        return agent

    # Pure staticmethod bindings

    def test_build_iteration_feedback_delegates(self):
        agent = self._make_agent()
        result = agent._build_iteration_feedback(1, llm_summary="ok")
        assert result["role"] == "system"
        assert "## Iteration 1" in result["content"]

    def test_remove_previous_guidance_delegates(self):
        agent = self._make_agent()
        msgs = [
            {"role": "system", "content": "## Iteration 1 Summary\nX\n---\nGuidance"},
        ]
        agent._remove_previous_guidance(msgs)
        assert "---" not in msgs[0]["content"]

    def test_format_tool_executions_delegates(self):
        agent = self._make_agent()
        assert agent._format_tool_executions([]) == []

    def test_check_task_completion_delegates(self):
        agent = self._make_agent()
        msgs = [{"role": "assistant", "content": f"x {GroundingAgentPrompts.TASK_COMPLETE}"}]
        assert agent._check_task_completion(msgs) is True

    def test_extract_last_assistant_message_delegates(self):
        agent = self._make_agent()
        msgs = [{"role": "assistant", "content": "hi"}]
        assert agent._extract_last_assistant_message(msgs) == "hi"

    # Async thin delegates

    @pytest.mark.asyncio
    async def test_generate_final_summary_delegates(self):
        agent = self._make_agent()
        agent._llm_client.complete.return_value = {"message": {"content": "summary"}}
        msgs = [{"role": "assistant", "content": "work"}]

        text, ok, _ = await agent._generate_final_summary("inst", msgs, 1)
        assert text == "summary"
        assert ok is True

    @pytest.mark.asyncio
    async def test_build_final_result_delegates(self):
        agent = self._make_agent()
        msgs = [{"role": "assistant", "content": f"Done {GroundingAgentPrompts.TASK_COMPLETE}"}]

        result = await agent._build_final_result(
            instruction="i",
            messages=msgs,
            all_tool_results=[],
            iterations=1,
            max_iterations=5,
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_mro_override_respected_in_build_final_result(self):
        """Adversarial MRO test: subclass overrides _check_task_completion
        and _build_final_result MUST call the override, not the module function."""
        from scion.agents.grounding_agent import GroundingAgent

        SENTINEL = "SUBCLASS_OVERRIDE_SENTINEL"

        class SubGroundingAgent(GroundingAgent):
            @staticmethod
            def _check_task_completion(messages):
                # Always returns True regardless of content
                return True

            @staticmethod
            def _extract_last_assistant_message(messages):
                return SENTINEL

        agent = SubGroundingAgent.__new__(SubGroundingAgent)
        agent._backend_scope = ["shell"]
        agent._llm_client = AsyncMock()
        agent._recording_manager = None
        agent._active_skill_ids = []
        agent._step = 0
        agent._name = "sub-test"
        # Give it a staticmethod binding for format (not overridden)
        agent._format_tool_executions = format_tool_executions

        # Messages WITHOUT the TASK_COMPLETE token — base would return False
        msgs = [{"role": "assistant", "content": "no complete token here"}]
        result = await agent._build_final_result(
            instruction="test",
            messages=msgs,
            all_tool_results=[],
            iterations=1,
            max_iterations=5,
        )
        # Subclass override makes _check_task_completion return True
        assert result["status"] == "success"
        # Subclass override makes _extract_last_assistant_message return sentinel
        assert result["response"] == SENTINEL

    @pytest.mark.asyncio
    async def test_record_agent_execution_delegates(self):
        agent = self._make_agent()
        agent._recording_manager = AsyncMock()

        await agent._record_agent_execution(
            {"response": "x", "status": "ok", "iterations": 1, "tool_executions": []},
            "inst",
        )
        agent._recording_manager.record_agent_action.assert_awaited_once()
