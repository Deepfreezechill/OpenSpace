"""Tests for ToolRegistry — extracted from Scion.tool_layer.

Verifies:
  - Skill directory discovery (env, config, builtin)
  - LLM-based skill selection and context injection
  - Selection LLM priority chain
  - Backward compatibility (Scion delegates to ToolRegistry)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# tool_layer / tool_registry import litellm which may not be available
try:
    from scion.tool_layer import Scion, ScionConfig
    from scion.tool_registry import ToolRegistry

    _HAS_TOOL_LAYER = True
except (ImportError, ModuleNotFoundError):
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(
    not _HAS_TOOL_LAYER,
    reason="scion.tool_layer requires litellm (not installed or broken)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_grounding_config():
    """Minimal grounding config with skills enabled."""
    cfg = MagicMock()
    cfg.skills.enabled = True
    cfg.skills.skill_dirs = []
    cfg.skills.max_select = 2
    return cfg


@pytest.fixture
def mock_config():
    """Minimal ScionConfig."""
    return ScionConfig(
        skill_registry_model=None,
        llm_kwargs={},
    )


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def mock_grounding_agent():
    agent = MagicMock()
    agent.backend_scope = ["shell", "mcp"]
    agent.clear_skill_context = MagicMock()
    agent.set_skill_context = MagicMock()
    agent._tool_retrieval_llm = None
    return agent


# ---------------------------------------------------------------------------
# ToolRegistry.discover()
# ---------------------------------------------------------------------------

class TestToolRegistryDiscover:
    """Tests for skill directory discovery."""

    def test_discover_returns_true_when_skills_found(self, mock_config, mock_grounding_config, mock_llm_client, tmp_path):
        """discover() returns True when at least one skill dir has skills."""

        # Create a fake skill dir with a SKILL.md
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill\nDoes stuff.")

        mock_grounding_config.skills.skill_dirs = [str(tmp_path)]

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        result = tr.discover()
        assert result is True
        assert tr.registry is not None

    def test_discover_returns_false_when_disabled(self, mock_config, mock_llm_client):
        """discover() returns False when skills.enabled is False."""

        cfg = MagicMock()
        cfg.skills.enabled = False
        cfg.skills.skill_dirs = []

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=cfg,
            llm_client=mock_llm_client,
        )
        result = tr.discover()
        assert result is False
        assert tr.registry is None

    def test_discover_with_no_grounding_config(self, mock_config, mock_llm_client):
        """discover() handles grounding_config=None without AttributeError."""

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=None,
            llm_client=mock_llm_client,
        )
        # Should not raise — skill_cfg will be None, falls through to builtin check
        result = tr.discover()
        assert isinstance(result, bool)

    def test_discover_includes_env_skill_dirs(self, mock_config, mock_grounding_config, mock_llm_client, tmp_path):
        """SCION_HOST_SKILL_DIRS env var adds skill directories."""

        skill_dir = tmp_path / "env-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Env Skill")

        with patch.dict(os.environ, {"SCION_HOST_SKILL_DIRS": str(tmp_path)}):
            tr = ToolRegistry(
                config=mock_config,
                grounding_config=mock_grounding_config,
                llm_client=mock_llm_client,
            )
            result = tr.discover()
        assert result is True
        assert tr.registry is not None

    def test_discover_skips_nonexistent_dirs(self, mock_config, mock_llm_client):
        """Non-existent skill dirs are logged and skipped, not fatal."""

        cfg = MagicMock()
        cfg.skills.enabled = True
        cfg.skills.skill_dirs = ["/no/such/path", "/also/missing"]

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=cfg,
            llm_client=mock_llm_client,
        )
        # Should not raise
        tr.discover()

    def test_registry_is_none_before_discover(self, mock_config, mock_grounding_config, mock_llm_client):
        """registry property is None until discover() is called."""

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        assert tr.registry is None


# ---------------------------------------------------------------------------
# ToolRegistry.select_and_inject()
# ---------------------------------------------------------------------------

class TestToolRegistrySelectAndInject:
    """Tests for LLM-based skill selection and injection."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_registry(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """select_and_inject returns False if no registry (skills not discovered)."""

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        result = await tr.select_and_inject(
            task="test task",
            agent=mock_grounding_agent,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_agent(self, mock_config, mock_grounding_config, mock_llm_client):
        """select_and_inject returns False if agent is None."""

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = MagicMock()  # Fake having a registry
        result = await tr.select_and_inject(
            task="test task",
            agent=None,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_clears_skill_context_when_no_skills_selected(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """When LLM selects no skills, agent's skill context is cleared."""

        mock_registry = MagicMock()
        mock_registry.select_skills_with_llm = AsyncMock(return_value=([], {"method": "llm", "selected": []}))
        mock_registry.list_skills.return_value = []

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = mock_registry

        result = await tr.select_and_inject(
            task="test task",
            agent=mock_grounding_agent,
        )
        assert result is False
        mock_grounding_agent.clear_skill_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_injects_selected_skills(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """When skills are selected, they are injected into the agent."""

        mock_skill = MagicMock()
        mock_skill.skill_id = "test-skill"

        mock_registry = MagicMock()
        mock_registry.select_skills_with_llm = AsyncMock(
            return_value=([mock_skill], {"method": "llm", "selected": ["test-skill"]})
        )
        mock_registry.build_context_injection.return_value = "## Skill: test-skill\nDoes stuff."

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = mock_registry

        result = await tr.select_and_inject(
            task="test task",
            agent=mock_grounding_agent,
        )
        assert result is True
        mock_grounding_agent.set_skill_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_skill_selection(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """Skill selection is recorded via RecordingManager when provided."""

        mock_skill = MagicMock()
        mock_skill.skill_id = "rec-skill"

        mock_registry = MagicMock()
        mock_registry.select_skills_with_llm = AsyncMock(
            return_value=([mock_skill], {"method": "llm", "selected": ["rec-skill"]})
        )
        mock_registry.build_context_injection.return_value = "context"

        mock_recording = MagicMock()
        mock_recording.record_skill_selection = AsyncMock()

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = mock_registry

        with patch("scion.tool_registry.RecordingManager") as MockRM:
            MockRM.record_skill_selection = AsyncMock()
            await tr.select_and_inject(
                task="test task",
                agent=mock_grounding_agent,
                recording_mgr=mock_recording,
            )
            MockRM.record_skill_selection.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_quality_metrics_forwarded(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """Quality metrics from store are passed to select_skills_with_llm."""

        mock_store = MagicMock()
        mock_store.get_summary.return_value = [
            {"skill_id": "s1", "total_selections": 5, "total_applied": 3,
             "total_completions": 2, "total_fallbacks": 1},
        ]

        mock_registry = MagicMock()
        mock_registry.select_skills_with_llm = AsyncMock(return_value=([], {"method": "llm", "selected": []}))

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = mock_registry

        await tr.select_and_inject(
            task="test task",
            agent=mock_grounding_agent,
            store=mock_store,
        )
        _, kwargs = mock_registry.select_skills_with_llm.call_args
        assert kwargs["skill_quality"]["s1"]["total_selections"] == 5

    @pytest.mark.asyncio
    async def test_store_exception_swallowed(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """Exception from store.get_summary is caught, not propagated."""

        mock_store = MagicMock()
        mock_store.get_summary.side_effect = RuntimeError("db locked")

        mock_registry = MagicMock()
        mock_registry.select_skills_with_llm = AsyncMock(return_value=([], {"method": "llm", "selected": []}))

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = mock_registry

        # Should not raise
        await tr.select_and_inject(
            task="test task",
            agent=mock_grounding_agent,
            store=mock_store,
        )
        # Verify LLM selection still happened (with skill_quality=None)
        _, kwargs = mock_registry.select_skills_with_llm.call_args
        assert kwargs["skill_quality"] is None

    @pytest.mark.asyncio
    async def test_no_llm_client_skips_selection(self, mock_grounding_config, mock_grounding_agent):
        """When no LLM is available, selection is skipped and context cleared."""

        config = ScionConfig(skill_registry_model=None, llm_kwargs={})
        mock_grounding_agent._tool_retrieval_llm = None

        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = [MagicMock(skill_id="s1")]

        tr = ToolRegistry(
            config=config,
            grounding_config=mock_grounding_config,
            llm_client=None,
        )
        tr._registry = mock_registry

        result = await tr.select_and_inject(
            task="test task",
            agent=mock_grounding_agent,
        )
        assert result is False
        mock_grounding_agent.clear_skill_context.assert_called_once()
        mock_registry.select_skills_with_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_scope_forwarded(self, mock_config, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """Agent's backend_scope is forwarded to build_context_injection."""

        mock_skill = MagicMock()
        mock_skill.skill_id = "scope-skill"

        mock_registry = MagicMock()
        mock_registry.select_skills_with_llm = AsyncMock(
            return_value=([mock_skill], {"method": "llm", "selected": ["scope-skill"]})
        )
        mock_registry.build_context_injection.return_value = "ctx"

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        tr._registry = mock_registry

        await tr.select_and_inject(task="test", agent=mock_grounding_agent)
        mock_registry.build_context_injection.assert_called_once_with(
            [mock_skill], backends=["shell", "mcp"]
        )


# ---------------------------------------------------------------------------
# ToolRegistry._get_selection_llm()
# ---------------------------------------------------------------------------

class TestGetSelectionLLM:
    """Tests for LLM client selection priority chain."""

    def test_dedicated_model_takes_priority(self, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """config.skill_registry_model overrides everything."""

        config = ScionConfig(skill_registry_model="gpt-4o-mini")
        tr = ToolRegistry(
            config=config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        with patch("scion.tool_registry.LLMClient") as MockLLM:
            result = tr._get_selection_llm(agent=mock_grounding_agent)
            MockLLM.assert_called_once()
            call_kwargs = MockLLM.call_args
            assert call_kwargs[1]["model"] == "gpt-4o-mini"

    def test_tool_retrieval_llm_fallback(self, mock_config, mock_grounding_config, mock_grounding_agent):
        """Falls back to agent's tool_retrieval_llm if no dedicated model."""

        agent_llm = MagicMock()
        mock_grounding_agent._tool_retrieval_llm = agent_llm

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=MagicMock(),
        )
        result = tr._get_selection_llm(agent=mock_grounding_agent)
        assert result is agent_llm

    def test_main_llm_client_final_fallback(self, mock_config, mock_grounding_config, mock_grounding_agent):
        """Falls back to main llm_client if nothing else available."""

        main_llm = MagicMock()
        mock_grounding_agent._tool_retrieval_llm = None

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=main_llm,
        )
        result = tr._get_selection_llm(agent=mock_grounding_agent)
        assert result is main_llm

    def test_returns_none_when_nothing_available(self, mock_grounding_config, mock_grounding_agent):
        """Returns None when all three LLM options are unavailable."""

        config = ScionConfig(skill_registry_model=None, llm_kwargs={})
        mock_grounding_agent._tool_retrieval_llm = None

        tr = ToolRegistry(
            config=config,
            grounding_config=mock_grounding_config,
            llm_client=None,
        )
        assert tr._get_selection_llm(agent=mock_grounding_agent) is None

    def test_agent_none_falls_back_to_llm_client(self, mock_config, mock_grounding_config):
        """With agent=None, falls back to main llm_client."""

        main_llm = MagicMock()
        tr = ToolRegistry(
            config=mock_config,
            grounding_config=mock_grounding_config,
            llm_client=main_llm,
        )
        result = tr._get_selection_llm(agent=None)
        assert result is main_llm

    def test_llm_kwargs_forwarded(self, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """llm_kwargs are passed through to the dedicated LLM client."""

        config = ScionConfig(
            skill_registry_model="gpt-4o",
            llm_kwargs={"api_base": "http://local"},
        )
        tr = ToolRegistry(
            config=config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        with patch("scion.tool_registry.LLMClient") as MockLLM:
            tr._get_selection_llm(agent=mock_grounding_agent)
            call_kwargs = MockLLM.call_args.kwargs
            assert call_kwargs["api_base"] == "http://local"
            # Explicit params take precedence over llm_kwargs
            assert call_kwargs["timeout"] == 30.0
            assert call_kwargs["max_retries"] == 2


# ---------------------------------------------------------------------------
# Scion backward compatibility
# ---------------------------------------------------------------------------

class TestScionDelegation:
    """Verify Scion still exposes the same public API via delegation."""

    def test_skill_registry_property_returns_inner_registry(self):
        """Scion.skill_registry returns the ToolRegistry's inner SkillRegistry."""

        os_instance = Scion()
        # Before init, should be None
        assert os_instance.skill_registry is None

    def test_scion_has_tool_registry_attr(self):
        """Scion instance has a _tool_registry attribute initialized to None."""

        os_instance = Scion()
        assert hasattr(os_instance, "_tool_registry")
        assert os_instance._tool_registry is None
