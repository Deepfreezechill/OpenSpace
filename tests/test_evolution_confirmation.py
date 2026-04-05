"""Tests for openspace.skill_engine.evolution.confirmation (Epic 5.4).

Verifies:
  1. parse_confirmation — JSON, keywords, ambiguous, edge cases
  2. llm_confirm_evolution — full flow (prompt build → LLM call → parse → record)
  3. llm_confirm_evolution — failure path (LLM exception → False)
  4. _SKILL_CONTENT_MAX_CHARS constant
  5. Backward compat: SkillEvolver delegates to confirmation functions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.evolution.confirmation import (
    _SKILL_CONTENT_MAX_CHARS,
    llm_confirm_evolution,
    parse_confirmation,
)


# ---------------------------------------------------------------------------
# parse_confirmation — pure function, no mocks needed
# ---------------------------------------------------------------------------

class TestParseConfirmation:
    """Test the LLM response parser with all known formats."""

    # --- JSON responses ---

    def test_json_proceed_true(self):
        assert parse_confirmation('{"proceed": true}') is True

    def test_json_proceed_false(self):
        assert parse_confirmation('{"proceed": false}') is False

    def test_json_with_markdown_fences(self):
        response = '```json\n{"proceed": true}\n```'
        assert parse_confirmation(response) is True

    def test_json_missing_proceed_key(self):
        assert parse_confirmation('{"analysis": "looks good"}') is False

    def test_json_proceed_zero_is_false(self):
        assert parse_confirmation('{"proceed": 0}') is False

    def test_json_proceed_one_is_true(self):
        assert parse_confirmation('{"proceed": 1}') is True

    # --- Keyword responses ---

    def test_keyword_yes(self):
        assert parse_confirmation("yes, this skill needs evolution") is True

    def test_keyword_no(self):
        assert parse_confirmation("no, the skill is fine") is False

    def test_keyword_confirmed(self):
        assert parse_confirmation("I've confirmed the change is needed") is True

    def test_keyword_rejected(self):
        assert parse_confirmation("rejected — not enough evidence") is False

    def test_keyword_skip(self):
        assert parse_confirmation("I'd recommend skipping this one") is False

    def test_keyword_proceed_true_in_text(self):
        assert parse_confirmation('Based on analysis: proceed: true') is True

    def test_keyword_proceed_false_in_text(self):
        assert parse_confirmation('My assessment: "proceed": false') is False

    # --- Edge cases ---

    def test_empty_string_returns_false(self):
        assert parse_confirmation("") is False

    def test_ambiguous_returns_false(self):
        assert parse_confirmation("I think maybe it could be improved") is False

    def test_know_does_not_match_no(self):
        """'know' should NOT match 'no' — word boundary check."""
        # "know" contains "no" but is a different word
        assert parse_confirmation("I know this is good, yes proceed") is True

    def test_case_sensitivity(self):
        """parse_confirmation receives lowered text from llm_confirm_evolution,
        but raw uppercase YES also works via keyword match."""
        # Lowercase (normal path via llm_confirm_evolution)
        assert parse_confirmation("yes") is True
        # Uppercase does NOT match \byes\b — by design, caller lowercases
        assert parse_confirmation("YES") is False

    def test_conflicting_keywords_first_wins(self):
        """When both yes and no keywords appear, positive match wins
        because the positive branch is checked first."""
        assert parse_confirmation("yes but also no") is True

    def test_json_array_falls_through_to_keywords(self):
        """JSON array is not a dict — falls through to keyword matching.
        The string '"proceed": true' is found as a substring, so returns True."""
        assert parse_confirmation('[{"proceed": true}]') is True

    def test_json_with_extra_fields(self):
        """Extra JSON fields are ignored; only 'proceed' matters."""
        assert parse_confirmation('{"proceed": true, "reason": "looks good"}') is True


# ---------------------------------------------------------------------------
# llm_confirm_evolution — async integration
# ---------------------------------------------------------------------------

class TestLlmConfirmEvolution:
    """Test the LLM confirmation flow with mocked evolver."""

    def _make_evolver(self, *, llm_response: str = '{"proceed": true}'):
        evolver = MagicMock()
        evolver._format_analysis_context.return_value = "(no history)"
        evolver._model = "test-model"
        evolver._llm_client.complete = AsyncMock(
            return_value={"message": {"content": llm_response}}
        )
        evolver._parse_confirmation.side_effect = parse_confirmation
        return evolver

    _RM_PATCH = "openspace.recording.RecordingManager"

    @pytest.mark.asyncio
    async def test_confirmed_returns_true(self):
        evolver = self._make_evolver(llm_response='{"proceed": true}')

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            result = await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# My Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix the bug",
                trigger_context="Tool degradation",
                recent_analyses=[],
            )

        assert result is True
        evolver._llm_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejected_returns_false(self):
        evolver = self._make_evolver(llm_response='{"proceed": false}')

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            result = await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# My Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix the bug",
                trigger_context="Metric check",
                recent_analyses=[],
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_llm_exception_returns_false(self):
        evolver = self._make_evolver()
        evolver._llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()

            result = await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# My Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix",
                trigger_context="test",
                recent_analyses=[],
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_recording_called(self):
        """Verify both setup and response are recorded."""
        evolver = self._make_evolver(llm_response='{"proceed": true}')

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix",
                trigger_context="test",
                recent_analyses=[],
            )

        mock_rm.record_conversation_setup.assert_awaited_once()
        mock_rm.record_iteration_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mro_format_analysis_called(self):
        """Confirm we call evolver._format_analysis_context (MRO preserved)."""
        evolver = self._make_evolver()

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix",
                trigger_context="test",
                recent_analyses=["a1"],
            )

        evolver._format_analysis_context.assert_called_once_with(["a1"])

    @pytest.mark.asyncio
    async def test_mro_parse_confirmation_called(self):
        """Confirm we call evolver._parse_confirmation (MRO preserved)."""
        evolver = self._make_evolver(llm_response='{"proceed": true}')

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix",
                trigger_context="test",
                recent_analyses=[],
            )

        evolver._parse_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_fallback_to_client_model(self):
        """When evolver._model is None, falls back to evolver._llm_client.model."""
        evolver = self._make_evolver(llm_response='{"proceed": true}')
        evolver._model = None
        evolver._llm_client.model = "fallback-model"

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            result = await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix",
                trigger_context="test",
                recent_analyses=[],
            )

        assert result is True
        # Verify fallback model was used
        call_kwargs = evolver._llm_client.complete.call_args
        assert call_kwargs[1]["model"] == "fallback-model"

    @pytest.mark.asyncio
    async def test_recording_truncates_content(self):
        """Recorded messages are truncated to _RECORDING_MAX_CHARS."""
        from openspace.skill_engine.evolution.confirmation import _RECORDING_MAX_CHARS

        # Create skill content larger than recording limit
        big_content = "x" * (_RECORDING_MAX_CHARS + 5000)
        evolver = self._make_evolver(llm_response='{"proceed": true}')

        with patch(self._RM_PATCH) as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            await llm_confirm_evolution(
                evolver,
                skill_record=MagicMock(skill_id="s1"),
                skill_content=big_content,
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix",
                trigger_context="test",
                recent_analyses=[],
            )

        # Verify recorded setup messages are truncated
        setup_call = mock_rm.record_conversation_setup.call_args
        recorded_msgs = setup_call[1]["setup_messages"]
        for msg in recorded_msgs:
            assert len(msg["content"]) <= _RECORDING_MAX_CHARS + 50  # small margin for truncation suffix


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_skill_content_max_chars(self):
        assert isinstance(_SKILL_CONTENT_MAX_CHARS, int)
        assert _SKILL_CONTENT_MAX_CHARS == 12_000

    def test_logger_uses_evolver_namespace(self):
        """Logger preserves evolver namespace for log filter compatibility."""
        from openspace.skill_engine.evolution import confirmation
        assert confirmation.logger.name == "openspace.skill_engine.evolver"


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

try:
    from openspace.skill_engine.evolver import SkillEvolver
    _HAS_EVOLVER = True
except ImportError:
    _HAS_EVOLVER = False


@pytest.mark.skipif(not _HAS_EVOLVER, reason="SkillEvolver not importable")
class TestBackwardCompat:
    def test_confirm_method_exists(self):
        assert hasattr(SkillEvolver, "_llm_confirm_evolution")

    def test_parse_method_exists(self):
        assert hasattr(SkillEvolver, "_parse_confirmation")

    def test_parse_is_staticmethod(self):
        # _parse_confirmation should be callable without self
        result = SkillEvolver._parse_confirmation('{"proceed": true}')
        assert result is True


# ---------------------------------------------------------------------------
# Delegation seam tests (real SkillEvolver → confirmation module)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_EVOLVER, reason="SkillEvolver not importable")
class TestDelegationSeam:
    """Verify real SkillEvolver delegates to confirmation.py functions."""

    def test_parse_confirmation_delegates(self):
        """Real SkillEvolver._parse_confirmation routes to confirmation module."""
        evolver = object.__new__(SkillEvolver)
        assert evolver._parse_confirmation('{"proceed": true}') is True
        assert evolver._parse_confirmation('{"proceed": false}') is False
        assert evolver._parse_confirmation("yes do it") is True
        assert evolver._parse_confirmation("no skip") is False

    @pytest.mark.asyncio
    async def test_llm_confirm_delegates_to_module(self):
        """Real SkillEvolver._llm_confirm_evolution calls confirmation module."""
        evolver = object.__new__(SkillEvolver)
        # Wire up required attributes
        evolver._model = "test-model"
        evolver._llm_client = MagicMock()
        evolver._llm_client.complete = AsyncMock(
            return_value={"message": {"content": '{"proceed": true}'}}
        )

        with patch("openspace.recording.RecordingManager") as mock_rm:
            mock_rm.record_conversation_setup = AsyncMock()
            mock_rm.record_iteration_context = AsyncMock()

            result = await evolver._llm_confirm_evolution(
                skill_record=MagicMock(skill_id="s1"),
                skill_content="# Skill",
                proposed_type=MagicMock(value="fix"),
                proposed_direction="fix it",
                trigger_context="test",
                recent_analyses=[],
            )

        assert result is True
        evolver._llm_client.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

class TestSizeGuard:
    def test_confirmation_module_size(self):
        from pathlib import Path
        mod_path = (
            Path(__file__).resolve().parent.parent
            / "openspace" / "skill_engine" / "evolution" / "confirmation.py"
        )
        assert mod_path.exists(), f"confirmation.py not found at {mod_path}"
        lines = mod_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) < 200, f"confirmation.py has {len(lines)} lines (limit 200)"
