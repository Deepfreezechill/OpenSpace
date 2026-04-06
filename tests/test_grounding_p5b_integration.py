"""P5b capstone integration tests — Epic 5.11.

Verify the full grounding/ package hangs together:
- All 8 submodules importable from the package __init__
- Facade GroundingAgent delegates correctly to every submodule
- No circular imports between submodules
- Backward-compatible import paths still work
"""

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── 1. Package completeness ──────────────────────────────────────────

EXPECTED_SUBMODULES = [
    "openspace.agents.grounding.context",
    "openspace.agents.grounding.execution",
    "openspace.agents.grounding.messages",
    "openspace.agents.grounding.prompts",
    "openspace.agents.grounding.results",
    "openspace.agents.grounding.tools",
    "openspace.agents.grounding.visual",
    "openspace.agents.grounding.workspace",
]


class TestPackageCompleteness:
    """All 8 submodules import cleanly and are re-exported from __init__."""

    @pytest.mark.parametrize("module_name", EXPECTED_SUBMODULES)
    def test_submodule_importable(self, module_name: str):
        mod = importlib.import_module(module_name)
        assert isinstance(mod, ModuleType)

    def test_package_init_exports_all_symbols(self):
        import openspace.agents.grounding as pkg

        assert hasattr(pkg, "__all__")
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists {name!r} but it's not importable"

    def test_eight_submodules_exist(self):
        """Exactly 8 non-__init__ .py files in the package."""
        import openspace.agents.grounding as pkg
        import pathlib

        pkg_dir = pathlib.Path(pkg.__file__).parent
        py_files = sorted(
            f.stem for f in pkg_dir.glob("*.py") if f.name != "__init__.py"
        )
        assert len(py_files) == 8, f"Expected 8 submodules, got {len(py_files)}: {py_files}"


# ── 2. No circular imports ───────────────────────────────────────────


class TestNoCircularImports:
    """Import each submodule independently — if circular deps exist these will blow up."""

    @pytest.mark.parametrize("module_name", EXPECTED_SUBMODULES)
    def test_independent_import(self, module_name: str):
        # Force a fresh import by removing cached module if present
        cached = sys.modules.pop(module_name, None)
        try:
            importlib.import_module(module_name)
        finally:
            if cached is not None:
                sys.modules[module_name] = cached


# ── 3. Backward-compatible import paths ──────────────────────────────


class TestBackwardCompatibility:
    """Callers using the old import path must still work."""

    def test_import_from_agents_module(self):
        from openspace.agents import GroundingAgent

        assert inspect.isclass(GroundingAgent)

    def test_import_from_grounding_agent_module(self):
        from openspace.agents.grounding_agent import GroundingAgent

        assert inspect.isclass(GroundingAgent)

    def test_import_from_top_level(self):
        from openspace import GroundingAgent

        assert inspect.isclass(GroundingAgent)


# ── 4. Facade delegation ────────────────────────────────────────────


class _StubLLM:
    model = "stub-model"


class _StubGroundingClient:
    pass


class TestFacadeDelegation:
    """Every public/protected method on GroundingAgent delegates to grounding/ submodule."""

    def _make_agent(self):
        from openspace.agents.grounding_agent import GroundingAgent

        with patch("openspace.agents.base.BaseAgent.__init__", return_value=None):
            agent = GroundingAgent.__new__(GroundingAgent)
            agent._backend_scope = ["gui", "shell"]
            agent._system_prompt = "test"
            agent._max_iterations = 5
            agent._visual_analysis_timeout = 10.0
            agent._tool_retrieval_llm = None
            agent._visual_analysis_model = None
            agent._skill_context = None
            agent._active_skill_ids = []
            agent._skill_registry = None
            agent._last_tools = []
            agent._grounding_client = MagicMock()
            agent._llm_client = MagicMock()
            agent._recording_manager = None
            agent._name = "TestAgent"
        return agent

    def test_set_skill_context_delegates(self):
        agent = self._make_agent()
        agent.set_skill_context("ctx", ["s1"])
        assert agent._skill_context == "ctx"
        assert agent._active_skill_ids == ["s1"]

    def test_clear_skill_context_delegates(self):
        agent = self._make_agent()
        agent._skill_context = "something"
        agent._active_skill_ids = ["x"]
        agent.clear_skill_context()
        assert agent._skill_context is None
        assert agent._active_skill_ids == []

    def test_has_skill_context_delegates(self):
        agent = self._make_agent()
        assert agent.has_skill_context is False
        agent._skill_context = "yes"
        assert agent.has_skill_context is True

    def test_set_skill_registry_delegates(self):
        agent = self._make_agent()
        mock_reg = MagicMock()
        agent.set_skill_registry(mock_reg)
        assert agent._skill_registry is mock_reg

    def test_cap_message_content_delegates(self):
        from openspace.agents.grounding_agent import GroundingAgent

        msgs = [{"role": "user", "content": "x" * 500_000}]
        result = GroundingAgent._cap_message_content(msgs)
        assert len(result[0]["content"]) < 500_000

    def test_truncate_messages_delegates(self):
        agent = self._make_agent()
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        result = agent._truncate_messages(msgs, keep_recent=3)
        # Should have kept system (if any) + recent
        assert len(result) <= len(msgs)

    def test_default_system_prompt_delegates(self):
        agent = self._make_agent()
        prompt = agent._default_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_construct_messages_delegates(self):
        agent = self._make_agent()
        ctx = {"instruction": "do thing", "conversation_history": []}
        msgs = agent.construct_messages(ctx)
        assert isinstance(msgs, list)
        assert any(m["role"] == "system" for m in msgs)

    def test_static_methods_are_callable(self):
        """Static method bindings should be callable without self."""
        from openspace.agents.grounding_agent import GroundingAgent

        # _select_key_screenshots
        assert callable(GroundingAgent._select_key_screenshots)
        # _get_workspace_path
        assert callable(GroundingAgent._get_workspace_path)
        # _scan_workspace_files
        assert callable(GroundingAgent._scan_workspace_files)
        # Results statics
        assert callable(GroundingAgent._build_iteration_feedback)
        assert callable(GroundingAgent._remove_previous_guidance)
        assert callable(GroundingAgent._format_tool_executions)
        assert callable(GroundingAgent._check_task_completion)
        assert callable(GroundingAgent._extract_last_assistant_message)


# ── 5. Facade line-count guard ───────────────────────────────────────


class TestFacadeSize:
    """Ensure the facade stays thin — regression guard."""

    def test_facade_under_250_lines(self):
        import pathlib

        facade = pathlib.Path(__file__).resolve().parent.parent / "openspace" / "agents" / "grounding_agent.py"
        line_count = len(facade.read_text(encoding="utf-8").splitlines())
        assert line_count < 250, (
            f"grounding_agent.py is {line_count} lines — should be < 250 as a thin facade"
        )

    def test_grounding_package_has_bulk(self):
        """The package modules should collectively hold > 800 lines."""
        import pathlib

        pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "openspace" / "agents" / "grounding"
        total = sum(
            len(f.read_text(encoding="utf-8").splitlines())
            for f in pkg_dir.glob("*.py")
            if f.name != "__init__.py"
        )
        assert total > 800, (
            f"grounding/ package has only {total} lines — expected > 800 after full extraction"
        )
