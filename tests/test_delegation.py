"""Tests for EPIC 1.4 — Scion Delegation.

Issues #68-71:
- #68: Scion.__init__ accepts AppContainer
- #69: Public property accessors expose container services (Phase 1 seam)
- #70: Backward-compatible factory for existing callers
- #71: Regression tests for identical behavior

Validates:
- Legacy creation path (ScionConfig only) still works
- Container-based creation path (from_container) works
- Public property accessors replace private field access pattern
- Backward compatibility — no behavior change for existing callers
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scion.app.container import AppContainer
from scion.app.factory import _StubLLM, _StubTelemetry

# tool_layer imports litellm which may not be available in all test envs
try:
    from scion.tool_layer import Scion, ScionConfig

    _HAS_TOOL_LAYER = True
except (ImportError, ModuleNotFoundError):
    _HAS_TOOL_LAYER = False

pytestmark = pytest.mark.skipif(
    not _HAS_TOOL_LAYER,
    reason="scion.tool_layer requires litellm (not installed or broken)",
)


# ══════════════════════════════════════════════════════════════════════
# Legacy creation path (backward compatibility)
# ══════════════════════════════════════════════════════════════════════


class TestLegacyCreation:
    """#70 — backward-compatible factory for existing callers."""

    def test_default_config(self):
        """Scion() with no args still works."""
        cs = Scion()
        assert cs.config is not None
        assert cs.is_initialized() is False
        assert cs.is_running() is False

    def test_explicit_config(self):
        """Scion(config=...) still works."""
        config = ScionConfig(llm_model="test/model")
        cs = Scion(config=config)
        assert cs.get_config() is config
        assert cs.config.llm_model == "test/model"

    def test_legacy_has_empty_container(self):
        """Legacy path gets an empty AppContainer."""
        cs = Scion()
        assert cs.container is not None
        assert cs.container.llm is None

    def test_private_fields_still_exist(self):
        """Private fields remain for internal use during transition."""
        cs = Scion()
        assert cs._llm_client is None
        assert cs._grounding_client is None
        assert cs._skill_registry is None


# ══════════════════════════════════════════════════════════════════════
# Container-based creation path
# ══════════════════════════════════════════════════════════════════════


class TestContainerCreation:
    """#68 — Scion.__init__ accepts AppContainer."""

    def test_from_container(self):
        """from_container() classmethod creates Scion with container."""
        container = AppContainer(llm=_StubLLM())
        cs = Scion.from_container(container)
        assert cs.container is container
        assert cs.container.llm is not None

    def test_from_container_with_config(self):
        """from_container() accepts optional config."""
        config = ScionConfig(llm_model="test/model")
        container = AppContainer(llm=_StubLLM())
        cs = Scion.from_container(container, config=config)
        assert cs.get_config() is config
        assert cs.container is container

    def test_init_with_container_kwarg(self):
        """Direct __init__ with container= keyword arg."""
        container = AppContainer(telemetry=_StubTelemetry())
        cs = Scion(container=container)
        assert cs.container is container
        assert cs.container.telemetry is not None

    def test_container_default_config(self):
        """from_container() uses default config if none provided."""
        cs = Scion.from_container(AppContainer())
        assert cs.config is not None
        assert cs.config.llm_model is not None


# ══════════════════════════════════════════════════════════════════════
# Public property accessors
# ══════════════════════════════════════════════════════════════════════


class TestPropertyAccessors:
    """#69 — Public properties replace private field access."""

    def test_llm_client_property(self):
        cs = Scion()
        assert cs.llm_client is None
        # Property reflects internal state
        cs._llm_client = MagicMock()
        assert cs.llm_client is cs._llm_client

    def test_grounding_client_property(self):
        cs = Scion()
        assert cs.grounding_client is None
        cs._grounding_client = MagicMock()
        assert cs.grounding_client is cs._grounding_client

    def test_grounding_config_property(self):
        cs = Scion()
        assert cs.grounding_config is None
        cs._grounding_config = {"test": True}
        assert cs.grounding_config == {"test": True}

    def test_skill_registry_property(self):
        cs = Scion()
        assert cs.skill_registry is None
        cs._skill_registry = MagicMock()
        assert cs.skill_registry is cs._skill_registry

    def test_skill_store_property(self):
        cs = Scion()
        assert cs.skill_store is None
        cs._skill_store = MagicMock()
        assert cs.skill_store is cs._skill_store

    def test_skill_evolver_property(self):
        cs = Scion()
        assert cs.skill_evolver is None
        cs._skill_evolver = MagicMock()
        assert cs.skill_evolver is cs._skill_evolver

    def test_container_property(self):
        container = AppContainer(llm=_StubLLM())
        cs = Scion(container=container)
        assert cs.container is container


# ══════════════════════════════════════════════════════════════════════
# Regression — identical behavior before/after
# ══════════════════════════════════════════════════════════════════════


class TestRegression:
    """#71 — identical behavior before and after delegation refactor."""

    def test_not_initialized_by_default(self):
        """Both paths start un-initialized."""
        legacy = Scion()
        container_based = Scion.from_container(AppContainer())
        assert legacy.is_initialized() is False
        assert container_based.is_initialized() is False

    def test_not_running_by_default(self):
        """Both paths start not-running."""
        legacy = Scion()
        container_based = Scion.from_container(AppContainer())
        assert legacy.is_running() is False
        assert container_based.is_running() is False

    def test_config_accessible(self):
        """get_config() works for both paths."""
        config = ScionConfig(llm_model="test/model")
        legacy = Scion(config=config)
        container_based = Scion.from_container(AppContainer(), config=config)
        assert legacy.get_config() is config
        assert container_based.get_config() is config

    def test_context_manager_protocol(self):
        """Both paths support async context manager protocol."""
        cs = Scion()
        assert hasattr(cs, "__aenter__")
        assert hasattr(cs, "__aexit__")

    def test_all_public_methods_exist(self):
        """Public API surface unchanged."""
        cs = Scion()
        for method in [
            "initialize",
            "execute",
            "cleanup",
            "is_initialized",
            "is_running",
            "get_config",
            "list_backends",
            "list_sessions",
        ]:
            assert hasattr(cs, method), f"Missing public method: {method}"

    def test_new_properties_exist(self):
        """New public property accessors are available."""
        cs = Scion()
        for prop in [
            "container",
            "llm_client",
            "grounding_client",
            "grounding_config",
            "skill_registry",
            "skill_store",
            "skill_evolver",
        ]:
            assert hasattr(cs, prop), f"Missing property: {prop}"
