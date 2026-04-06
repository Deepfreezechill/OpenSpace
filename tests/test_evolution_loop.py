"""Tests for openspace.skill_engine.evolution.loop (Epic 5.5).

Verifies:
  1. parse_evolution_output — complete/failed token extraction
  2. run_evolution_loop — agent loop flow, recording, termination
  3. apply_with_retry — success, validation failure, retry, cleanup
  4. Constants: _MAX_EVOLUTION_ITERATIONS, _MAX_EVOLUTION_ATTEMPTS
  5. Backward compat: SkillEvolver delegates to loop functions
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import module-level functions
from openspace.skill_engine.evolution.loop import (
    EVOLUTION_COMPLETE,
    EVOLUTION_FAILED,
    _MAX_EVOLUTION_ATTEMPTS,
    _MAX_EVOLUTION_ITERATIONS,
    apply_with_retry,
    parse_evolution_output,
    run_evolution_loop,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeEvolutionType:
    value = "FIX"


class _FakeSuggestion:
    evolution_type = _FakeEvolutionType()
    target_skill_ids = ["skill-a"]
    direction = "fix it"
    category = None


class _FakeTrigger:
    value = "analysis"


class _FakeCtx:
    suggestion = _FakeSuggestion()
    trigger = _FakeTrigger()
    available_tools = []
    recent_analyses = []
    skill_records = []
    skill_contents = []
    skill_dirs = []
    source_task_id = "task-1"
    tool_issue_summary = ""
    metric_summary = ""


class _FakeEvolver:
    _model = "test-model"
    _available_tools = []

    def __init__(self):
        self._llm_client = MagicMock()
        self._llm_client.model = "fallback-model"
        self._llm_client.complete = AsyncMock()

    def _parse_evolution_output(self, content):
        return parse_evolution_output(content)

    def _format_skill_dir_content(self, skill_dir):
        return ""


# ---------------------------------------------------------------------------
# parse_evolution_output
# ---------------------------------------------------------------------------

class TestParseEvolutionOutput:
    def test_complete_token_returns_content(self):
        content = f"Here is the fix\n{EVOLUTION_COMPLETE}"
        result, failure = parse_evolution_output(content)
        assert result is not None
        assert failure is None
        assert "Here is the fix" in result

    def test_failed_token_returns_reason(self):
        content = f"{EVOLUTION_FAILED} reason: cannot fix this skill"
        result, failure = parse_evolution_output(content)
        assert result is None
        assert "cannot fix this skill" in failure

    def test_failed_no_reason(self):
        content = f"{EVOLUTION_FAILED}"
        result, failure = parse_evolution_output(content)
        assert result is None
        assert "no reason given" in failure.lower()

    def test_both_tokens_failure_wins(self):
        """If both COMPLETE and FAILED appear, treat as failure (conservative)."""
        content = f"some content {EVOLUTION_COMPLETE} but also {EVOLUTION_FAILED} oops"
        result, failure = parse_evolution_output(content)
        assert result is None
        assert failure is not None

    def test_no_token_returns_defensive_failure(self):
        result, failure = parse_evolution_output("just some text")
        assert result is None
        assert "No completion token" in failure

    def test_strips_markdown_fences(self):
        content = f"```\nfixed content\n```\n{EVOLUTION_COMPLETE}"
        result, failure = parse_evolution_output(content)
        assert failure is None
        assert "```" not in result
        assert "fixed content" in result

    def test_reason_truncated_to_500(self):
        long_reason = "x" * 1000
        content = f"{EVOLUTION_FAILED} {long_reason}"
        _, failure = parse_evolution_output(content)
        assert len(failure) <= 500


# ---------------------------------------------------------------------------
# run_evolution_loop
# ---------------------------------------------------------------------------

class TestRunEvolutionLoop:
    @pytest.mark.asyncio
    async def test_immediate_complete(self):
        """LLM returns COMPLETE on first iteration → returns content."""
        evolver = _FakeEvolver()
        evolver._llm_client.complete.return_value = {
            "message": {"content": f"fixed\n{EVOLUTION_COMPLETE}"},
            "messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": f"fixed\n{EVOLUTION_COMPLETE}"}],
            "has_tool_calls": False,
            "tool_results": [],
        }

        with patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            result = await run_evolution_loop(evolver, "prompt", _FakeCtx())

        assert result is not None
        assert "fixed" in result

    @pytest.mark.asyncio
    async def test_immediate_failed(self):
        """LLM returns FAILED on first iteration → returns None."""
        evolver = _FakeEvolver()
        evolver._llm_client.complete.return_value = {
            "message": {"content": f"{EVOLUTION_FAILED} cannot do it"},
            "messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": f"{EVOLUTION_FAILED} cannot do it"}],
            "has_tool_calls": False,
            "tool_results": [],
        }

        with patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            result = await run_evolution_loop(evolver, "prompt", _FakeCtx())

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self):
        """LLM call raises → returns None (graceful)."""
        evolver = _FakeEvolver()
        evolver._llm_client.complete.side_effect = RuntimeError("API down")

        with patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            result = await run_evolution_loop(evolver, "prompt", _FakeCtx())

        assert result is None

    @pytest.mark.asyncio
    async def test_recording_called(self):
        """Recording setup and iteration are called."""
        evolver = _FakeEvolver()
        evolver._llm_client.complete.return_value = {
            "message": {"content": f"ok\n{EVOLUTION_COMPLETE}"},
            "messages": [{"role": "assistant", "content": f"ok\n{EVOLUTION_COMPLETE}"}],
            "has_tool_calls": False,
            "tool_results": [],
        }

        with patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            await run_evolution_loop(evolver, "prompt", _FakeCtx())

        mock_rec.record_conversation_setup.assert_awaited_once()
        mock_rec.record_iteration_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mro_parse_called(self):
        """Loop calls evolver._parse_evolution_output (MRO)."""
        evolver = _FakeEvolver()
        evolver._parse_evolution_output = MagicMock(return_value=("content", None))
        evolver._llm_client.complete.return_value = {
            "message": {"content": f"x {EVOLUTION_COMPLETE}"},
            "messages": [{"role": "assistant", "content": f"x {EVOLUTION_COMPLETE}"}],
            "has_tool_calls": False,
            "tool_results": [],
        }

        with patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            await run_evolution_loop(evolver, "prompt", _FakeCtx())

        evolver._parse_evolution_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_fallback(self):
        """When evolver._model is None, uses llm_client.model."""
        evolver = _FakeEvolver()
        evolver._model = None
        evolver._llm_client.complete.return_value = {
            "message": {"content": f"ok\n{EVOLUTION_COMPLETE}"},
            "messages": [{"role": "assistant", "content": f"ok\n{EVOLUTION_COMPLETE}"}],
            "has_tool_calls": False,
            "tool_results": [],
        }

        with patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            await run_evolution_loop(evolver, "prompt", _FakeCtx())

        call_kwargs = evolver._llm_client.complete.call_args[1]
        assert call_kwargs["model"] == "fallback-model"


# ---------------------------------------------------------------------------
# apply_with_retry
# ---------------------------------------------------------------------------

class _FakeEditResult:
    def __init__(self, ok=True, error=None, content_snapshot=None, content_diff=None):
        self.ok = ok
        self.error = error
        self.content_snapshot = content_snapshot or {}
        self.content_diff = content_diff or ""


class TestApplyWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Apply succeeds + validation passes → returns result."""
        evolver = _FakeEvolver()
        apply_fn = MagicMock(return_value=_FakeEditResult(ok=True))
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.loop._validate_skill_dir", return_value=None), \
             patch("openspace.recording.RecordingManager"):
            result = await apply_with_retry(
                evolver, apply_fn=apply_fn, initial_content="content",
                skill_dir=Path("/fake"), ctx=ctx, prompt="prompt",
            )

        assert result is not None
        assert result.ok
        apply_fn.assert_called_once_with("content")

    @pytest.mark.asyncio
    async def test_apply_fails_then_retry_succeeds(self):
        """First apply fails, retry LLM produces fixed content → success."""
        evolver = _FakeEvolver()
        apply_fn = MagicMock(side_effect=[
            _FakeEditResult(ok=False, error="parse error"),
            _FakeEditResult(ok=True),
        ])
        evolver._llm_client.complete.return_value = {
            "message": {"content": "fixed content"},
            "messages": [],
        }
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.loop._validate_skill_dir", return_value=None), \
             patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            result = await apply_with_retry(
                evolver, apply_fn=apply_fn, initial_content="bad content",
                skill_dir=Path("/fake"), ctx=ctx, prompt="prompt",
            )

        assert result is not None
        assert result.ok
        assert apply_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_none(self):
        """All attempts fail → returns None."""
        evolver = _FakeEvolver()
        apply_fn = MagicMock(return_value=_FakeEditResult(ok=False, error="bad"))
        evolver._llm_client.complete.return_value = {
            "message": {"content": "still bad"},
            "messages": [],
        }
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.loop._validate_skill_dir"), \
             patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            result = await apply_with_retry(
                evolver, apply_fn=apply_fn, initial_content="bad",
                skill_dir=Path("/fake"), ctx=ctx, prompt="prompt",
            )

        assert result is None
        assert apply_fn.call_count == _MAX_EVOLUTION_ATTEMPTS

    @pytest.mark.asyncio
    async def test_validation_failure_triggers_retry(self):
        """Apply OK but validation fails → treated as error, retries."""
        evolver = _FakeEvolver()
        apply_fn = MagicMock(return_value=_FakeEditResult(ok=True))
        evolver._llm_client.complete.return_value = {
            "message": {"content": "fixed"},
            "messages": [],
        }
        ctx = _FakeCtx()

        validate_results = iter(["missing frontmatter", None])
        with patch("openspace.skill_engine.evolution.loop._validate_skill_dir", side_effect=validate_results), \
             patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            result = await apply_with_retry(
                evolver, apply_fn=apply_fn, initial_content="incomplete",
                skill_dir=Path("/fake"), ctx=ctx, prompt="prompt",
            )

        assert result is not None
        assert apply_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_on_retry(self, tmp_path):
        """cleanup_on_retry dir is removed before each retry."""
        target = tmp_path / "new-skill"
        target.mkdir()
        (target / "SKILL.md").write_text("old")

        evolver = _FakeEvolver()
        apply_fn = MagicMock(side_effect=[
            _FakeEditResult(ok=False, error="bad"),
            _FakeEditResult(ok=True),
        ])
        evolver._llm_client.complete.return_value = {
            "message": {"content": "fixed"},
            "messages": [],
        }
        ctx = _FakeCtx()

        with patch("openspace.skill_engine.evolution.loop._validate_skill_dir", return_value=None), \
             patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            result = await apply_with_retry(
                evolver, apply_fn=apply_fn, initial_content="bad",
                skill_dir=target, ctx=ctx, prompt="prompt",
                cleanup_on_retry=target,
            )

        assert result is not None
        # The dir was cleaned up before the retry attempt

    @pytest.mark.asyncio
    async def test_mro_format_skill_dir_called(self):
        """Retry path calls evolver._format_skill_dir_content (MRO)."""
        evolver = _FakeEvolver()
        evolver._format_skill_dir_content = MagicMock(return_value="on disk content")
        apply_fn = MagicMock(return_value=_FakeEditResult(ok=False, error="bad"))
        evolver._llm_client.complete.return_value = {
            "message": {"content": "fixed"},
            "messages": [],
        }
        ctx = _FakeCtx()
        fake_dir = MagicMock()
        fake_dir.is_dir.return_value = True
        fake_dir.exists.return_value = False

        with patch("openspace.skill_engine.evolution.loop._validate_skill_dir"), \
             patch("openspace.recording.RecordingManager") as mock_rec:
            mock_rec.record_conversation_setup = AsyncMock()
            mock_rec.record_iteration_context = AsyncMock()
            await apply_with_retry(
                evolver, apply_fn=apply_fn, initial_content="bad",
                skill_dir=fake_dir, ctx=ctx, prompt="prompt",
            )

        evolver._format_skill_dir_content.assert_called()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_iterations(self):
        assert _MAX_EVOLUTION_ITERATIONS == 5

    def test_max_attempts(self):
        assert _MAX_EVOLUTION_ATTEMPTS == 3

    def test_logger_uses_evolver_namespace(self):
        from openspace.skill_engine.evolution.loop import logger
        assert "evolver" in logger.name


# ---------------------------------------------------------------------------
# Backward compat: SkillEvolver delegates
# ---------------------------------------------------------------------------

class TestDelegationSeam:
    def test_parse_evolution_output_is_staticmethod(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert isinstance(
            SkillEvolver.__dict__["_parse_evolution_output"],
            staticmethod,
        )

    def test_evolve_fix_method_exists(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert hasattr(SkillEvolver, "_evolve_fix")
        assert asyncio.iscoroutinefunction(SkillEvolver._evolve_fix)

    def test_evolve_derived_method_exists(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert hasattr(SkillEvolver, "_evolve_derived")
        assert asyncio.iscoroutinefunction(SkillEvolver._evolve_derived)

    def test_evolve_captured_method_exists(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert hasattr(SkillEvolver, "_evolve_captured")
        assert asyncio.iscoroutinefunction(SkillEvolver._evolve_captured)

    def test_run_evolution_loop_method_exists(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert hasattr(SkillEvolver, "_run_evolution_loop")
        assert asyncio.iscoroutinefunction(SkillEvolver._run_evolution_loop)

    def test_apply_with_retry_method_exists(self):
        from openspace.skill_engine.evolver import SkillEvolver
        assert hasattr(SkillEvolver, "_apply_with_retry")
        assert asyncio.iscoroutinefunction(SkillEvolver._apply_with_retry)


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

class TestSizeGuard:
    def test_loop_module_size(self):
        """loop.py should stay under 400 lines."""
        import openspace.skill_engine.evolution.loop as mod
        src = Path(mod.__file__)
        lines = src.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 400, f"loop.py has {len(lines)} lines (limit 400)"
