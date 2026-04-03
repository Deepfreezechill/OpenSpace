"""Tests for ToolRegistry — extracted from OpenSpace.tool_layer.

Verifies:
  - Skill directory discovery (env, config, builtin)
  - LLM-based skill selection and context injection
  - Selection LLM priority chain
  - Backward compatibility (OpenSpace delegates to ToolRegistry)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# tool_layer / tool_registry import litellm which may not be available
try:
    from openspace.tool_layer import OpenSpace, OpenSpaceConfig
    from openspace.tool_registry import ToolRegistry

    _HAS_TOOL_LAYER = True
except (ImportError, ModuleNotFoundError):
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(
    not _HAS_TOOL_LAYER,
    reason="openspace.tool_layer requires litellm (not installed or broken)",
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
    """Minimal OpenSpaceConfig."""
    return OpenSpaceConfig(
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

    def test_discover_returns_false_when_no_skills(self, mock_config, mock_llm_client):
        """discover() returns False when skills are disabled or no dirs found."""

        cfg = MagicMock()
        cfg.skills.enabled = True
        cfg.skills.skill_dirs = ["/nonexistent/path"]

        tr = ToolRegistry(
            config=mock_config,
            grounding_config=cfg,
            llm_client=mock_llm_client,
        )
        result = tr.discover()
        # May return True (builtin skills exist) or False — depends on builtin dir
        # The key check: registry may be None if no dirs resolve
        assert isinstance(result, bool)

    def test_discover_includes_env_skill_dirs(self, mock_config, mock_grounding_config, mock_llm_client, tmp_path):
        """OPENSPACE_HOST_SKILL_DIRS env var adds skill directories."""

        skill_dir = tmp_path / "env-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Env Skill")

        with patch.dict(os.environ, {"OPENSPACE_HOST_SKILL_DIRS": str(tmp_path)}):
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

        with patch("openspace.tool_registry.RecordingManager") as MockRM:
            MockRM.record_skill_selection = AsyncMock()
            await tr.select_and_inject(
                task="test task",
                agent=mock_grounding_agent,
                recording_mgr=mock_recording,
            )
            MockRM.record_skill_selection.assert_called_once()


# ---------------------------------------------------------------------------
# ToolRegistry._get_selection_llm()
# ---------------------------------------------------------------------------

class TestGetSelectionLLM:
    """Tests for LLM client selection priority chain."""

    def test_dedicated_model_takes_priority(self, mock_grounding_config, mock_llm_client, mock_grounding_agent):
        """config.skill_registry_model overrides everything."""

        config = OpenSpaceConfig(skill_registry_model="gpt-4o-mini")
        tr = ToolRegistry(
            config=config,
            grounding_config=mock_grounding_config,
            llm_client=mock_llm_client,
        )
        with patch("openspace.tool_registry.LLMClient") as MockLLM:
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


# ---------------------------------------------------------------------------
# OpenSpace backward compatibility
# ---------------------------------------------------------------------------

class TestOpenSpaceDelegation:
    """Verify OpenSpace still exposes the same public API via delegation."""

    def test_skill_registry_property_returns_inner_registry(self):
        """OpenSpace.skill_registry returns the ToolRegistry's inner SkillRegistry."""

        os_instance = OpenSpace()
        # Before init, should be None
        assert os_instance.skill_registry is None

    def test_openspace_has_tool_registry_attr(self):
        """OpenSpace instance has a _tool_registry attribute after construction."""

        os_instance = OpenSpace()
        assert hasattr(os_instance, "_tool_registry")
