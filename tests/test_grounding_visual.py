"""Tests for openspace.agents.grounding.visual — visual analysis helpers.

Epic 5.9 extraction: _visual_analysis_callback, _enhance_result_with_visual_context,
_select_key_screenshots.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.agents.grounding.visual import (
    _enhance_result_with_visual_context,
    _select_key_screenshots,
    _visual_analysis_callback,
)
from openspace.grounding.core.types import ToolResult, ToolStatus


def _make_result(**kwargs):
    defaults = {"status": ToolStatus.SUCCESS, "content": "ok", "metadata": {}}
    defaults.update(kwargs)
    return ToolResult(**defaults)


def _make_tool_call(args_dict=None, args_str=None):
    """Build a mock tool_call with .function.arguments."""
    tc = MagicMock()
    if args_str is not None:
        tc.function.arguments = args_str
    elif args_dict is not None:
        tc.function.arguments = args_dict
    else:
        tc.function.arguments = "{}"
    return tc


class _FakeAgent:
    """Minimal stand-in for GroundingAgent instance state."""

    def __init__(self):
        self._visual_analysis_model = None
        self._llm_client = MagicMock(model="test-model")
        self._visual_analysis_timeout = 5.0
        self._current_instruction = "test instruction"


# ── _select_key_screenshots (pure function) ────────────────────────


class TestSelectKeyScreenshots:
    def test_returns_all_when_under_max(self):
        shots = [b"a", b"b"]
        assert _select_key_screenshots(shots, max_count=3) == [b"a", b"b"]

    def test_returns_all_when_equal_max(self):
        shots = [b"a", b"b", b"c"]
        assert _select_key_screenshots(shots, max_count=3) == shots

    def test_selects_first_and_last(self):
        shots = [b"0", b"1", b"2", b"3", b"4"]
        result = _select_key_screenshots(shots, max_count=2)
        assert result[0] == b"0"
        assert result[-1] == b"4"
        assert len(result) == 2

    def test_selects_three_from_many(self):
        shots = [bytes([i]) for i in range(10)]
        result = _select_key_screenshots(shots, max_count=3)
        assert len(result) == 3
        # First and last are always included
        assert result[0] == shots[0]
        assert result[-1] == shots[-1]

    def test_preserves_order(self):
        shots = [bytes([i]) for i in range(7)]
        result = _select_key_screenshots(shots, max_count=3)
        indices = [shots.index(r) for r in result]
        assert indices == sorted(indices)

    def test_max_count_one_returns_last(self):
        shots = [b"a", b"b", b"c"]
        result = _select_key_screenshots(shots, max_count=1)
        assert len(result) == 1
        assert result[0] == b"c"


# ── _visual_analysis_callback ──────────────────────────────────────


class TestVisualAnalysisCallback:
    @pytest.mark.asyncio
    async def test_skip_when_meta_parameter_set(self):
        agent = _FakeAgent()
        result = _make_result()
        tc = _make_tool_call(args_dict={"skip_visual_analysis": True})
        out = await _visual_analysis_callback(agent, result, "click", tc, "gui")
        assert out is result  # unchanged

    @pytest.mark.asyncio
    async def test_skip_for_non_gui_backend(self):
        agent = _FakeAgent()
        result = _make_result()
        tc = _make_tool_call()
        out = await _visual_analysis_callback(agent, result, "run_shell", tc, "shell")
        assert out is result

    @pytest.mark.asyncio
    async def test_returns_original_when_no_visual_data_and_capture_fails(self):
        agent = _FakeAgent()
        result = _make_result(metadata={})
        tc = _make_tool_call()

        with patch("openspace.agents.grounding.visual.ScreenshotClient") as MockSC:
            mock_client = AsyncMock()
            mock_client.capture.return_value = None
            MockSC.return_value = mock_client
            out = await _visual_analysis_callback(agent, result, "click", tc, "gui")
            assert out is result

    @pytest.mark.asyncio
    async def test_captures_and_enhances_when_no_initial_visual(self):
        agent = _FakeAgent()
        result = _make_result(metadata={})
        tc = _make_tool_call()

        with patch("openspace.agents.grounding.visual.ScreenshotClient") as MockSC:
            mock_client = AsyncMock()
            mock_client.capture.return_value = b"screenshot_bytes"
            MockSC.return_value = mock_client

            with patch(
                "openspace.agents.grounding.visual._enhance_result_with_visual_context",
                new_callable=AsyncMock,
            ) as mock_enhance:
                enhanced = _make_result(content="enhanced!")
                mock_enhance.return_value = enhanced
                out = await _visual_analysis_callback(agent, result, "click", tc, "gui")
                assert out is enhanced

    @pytest.mark.asyncio
    async def test_skip_visual_analysis_from_string_args(self):
        agent = _FakeAgent()
        result = _make_result()
        tc = _make_tool_call(args_str='{"skip_visual_analysis": true}')
        out = await _visual_analysis_callback(agent, result, "click", tc, "gui")
        assert out is result

    @pytest.mark.asyncio
    async def test_tolerates_unparseable_arguments(self):
        """Should not crash when tool_call arguments are invalid JSON."""
        agent = _FakeAgent()
        result = _make_result(metadata={})
        tc = _make_tool_call(args_str="not-json{{{")

        with patch("openspace.agents.grounding.visual.ScreenshotClient") as MockSC:
            mock_client = AsyncMock()
            mock_client.capture.return_value = None
            MockSC.return_value = mock_client
            out = await _visual_analysis_callback(agent, result, "click", tc, "gui")
            assert out is result


# ── _enhance_result_with_visual_context ────────────────────────────


class TestEnhanceResultWithVisualContext:
    @pytest.mark.asyncio
    async def test_returns_original_when_empty_metadata(self):
        agent = _FakeAgent()
        result = _make_result(metadata={})
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            out = await _enhance_result_with_visual_context(agent, result, "click")
            # No screenshots → returns original
            assert out is result

    @pytest.mark.asyncio
    async def test_returns_original_when_no_screenshots(self):
        agent = _FakeAgent()
        result = _make_result(metadata={"some_key": "val"})
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            out = await _enhance_result_with_visual_context(agent, result, "click")
            assert out is result


# ── Delegation seam tests ──────────────────────────────────────────


class TestVisualDelegationSeams:
    """Verify grounding_agent.py properly delegates to visual module."""

    def test_visual_analysis_callback_delegates(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert "_visual_analysis_callback" in dir(GroundingAgent)

    def test_enhance_result_delegates(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert "_enhance_result_with_visual_context" in dir(GroundingAgent)

    def test_select_key_screenshots_is_static(self):
        from openspace.agents.grounding_agent import GroundingAgent

        # staticmethod binding — callable on both class and instance
        assert callable(GroundingAgent._select_key_screenshots)
        # Verify it's the same function
        assert GroundingAgent._select_key_screenshots is _select_key_screenshots
