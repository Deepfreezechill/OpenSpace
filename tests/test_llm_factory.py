"""Tests for LLMFactory — extracted LLM client creation from OpenSpace."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

try:
    from openspace.tool_layer import OpenSpace, OpenSpaceConfig
    from openspace.llm_factory import LLMFactory

    _HAS_TOOL_LAYER = True
except Exception:
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(not _HAS_TOOL_LAYER, reason="tool_layer deps unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return OpenSpaceConfig(
        llm_model="openrouter/anthropic/claude-sonnet-4.5",
        llm_enable_thinking=True,
        llm_timeout=60.0,
        llm_max_retries=5,
        llm_rate_limit_delay=0.5,
        llm_kwargs={"api_key": "sk-test"},
        tool_retrieval_model="openrouter/openai/gpt-4o",
    )


@pytest.fixture
def minimal_config():
    return OpenSpaceConfig()


# ---------------------------------------------------------------------------
# LLMFactory.__init__
# ---------------------------------------------------------------------------

class TestLLMFactoryInit:

    def test_initial_state(self, config):
        factory = LLMFactory(config=config)
        assert factory.llm_client is None
        assert factory.tool_retrieval_llm is None

    def test_stores_config(self, config):
        factory = LLMFactory(config=config)
        assert factory._config is config


# ---------------------------------------------------------------------------
# LLMFactory.create_main()
# ---------------------------------------------------------------------------

class TestCreateMain:

    def test_creates_llm_client(self, config):
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            mock_client = MagicMock()
            MockLLM.return_value = mock_client

            factory = LLMFactory(config=config)
            result = factory.create_main()

            assert result is mock_client
            assert factory.llm_client is mock_client

    def test_passes_all_config_fields(self, config):
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            MockLLM.return_value = MagicMock()

            factory = LLMFactory(config=config)
            factory.create_main()

            MockLLM.assert_called_once_with(
                model="openrouter/anthropic/claude-sonnet-4.5",
                enable_thinking=True,
                rate_limit_delay=0.5,
                max_retries=5,
                timeout=60.0,
                api_key="sk-test",
            )

    def test_uses_defaults_when_minimal_config(self, minimal_config):
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            MockLLM.return_value = MagicMock()

            factory = LLMFactory(config=minimal_config)
            factory.create_main()

            MockLLM.assert_called_once_with(
                model="openrouter/anthropic/claude-sonnet-4.5",
                enable_thinking=False,
                rate_limit_delay=0.0,
                max_retries=3,
                timeout=120.0,
            )

    def test_create_main_twice_replaces_client(self, config):
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            first = MagicMock()
            second = MagicMock()
            MockLLM.side_effect = [first, second]

            factory = LLMFactory(config=config)
            factory.create_main()
            assert factory.llm_client is first
            factory.create_main()
            assert factory.llm_client is second


# ---------------------------------------------------------------------------
# LLMFactory.create_tool_retrieval()
# ---------------------------------------------------------------------------

class TestCreateToolRetrieval:

    def test_creates_when_model_configured(self, config):
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            mock_client = MagicMock()
            MockLLM.return_value = mock_client

            factory = LLMFactory(config=config)
            result = factory.create_tool_retrieval()

            assert result is mock_client
            assert factory.tool_retrieval_llm is mock_client

    def test_returns_none_when_no_model(self, minimal_config):
        factory = LLMFactory(config=minimal_config)
        result = factory.create_tool_retrieval()
        assert result is None
        assert factory.tool_retrieval_llm is None

    def test_passes_correct_config(self, config):
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            MockLLM.return_value = MagicMock()

            factory = LLMFactory(config=config)
            factory.create_tool_retrieval()

            MockLLM.assert_called_once_with(
                model="openrouter/openai/gpt-4o",
                timeout=60.0,
                max_retries=5,
                api_key="sk-test",
            )

    def test_inherits_llm_kwargs(self, config):
        """Tool retrieval LLM inherits credentials from llm_kwargs."""
        config.llm_kwargs = {"api_key": "sk-shared", "api_base": "https://custom"}
        with patch("openspace.llm_factory.LLMClient") as MockLLM:
            MockLLM.return_value = MagicMock()

            factory = LLMFactory(config=config)
            factory.create_tool_retrieval()

            _, kwargs = MockLLM.call_args
            assert kwargs["api_key"] == "sk-shared"
            assert kwargs["api_base"] == "https://custom"


# ---------------------------------------------------------------------------
# OpenSpace backward compatibility
# ---------------------------------------------------------------------------

class TestOpenSpaceDelegation:

    def test_openspace_has_llm_factory_attr(self):
        os_instance = OpenSpace()
        assert hasattr(os_instance, "_llm_factory")

    def test_llm_client_still_accessible(self):
        os_instance = OpenSpace()
        assert hasattr(os_instance, "_llm_client")
        assert os_instance._llm_client is None
