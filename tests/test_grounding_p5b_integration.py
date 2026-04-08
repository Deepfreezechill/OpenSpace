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
import re
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── 1. Package completeness ──────────────────────────────────────────

EXPECTED_SUBMODULES = [
    "scion.agents.grounding.context",
    "scion.agents.grounding.execution",
    "scion.agents.grounding.messages",
    "scion.agents.grounding.prompts",
    "scion.agents.grounding.results",
    "scion.agents.grounding.tools",
    "scion.agents.grounding.visual",
    "scion.agents.grounding.workspace",
]


class TestPackageCompleteness:
    """All 8 submodules import cleanly and are re-exported from __init__."""

    @pytest.mark.parametrize("module_name", EXPECTED_SUBMODULES)
    def test_submodule_importable(self, module_name: str):
        mod = importlib.import_module(module_name)
        assert isinstance(mod, ModuleType)

    def test_package_init_exports_all_symbols(self):
        import scion.agents.grounding as pkg

        assert hasattr(pkg, "__all__")
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists {name!r} but it's not importable"

    def test_eight_submodules_exist(self):
        """Exactly 8 non-__init__ .py files in the package."""
        import scion.agents.grounding as pkg
        import pathlib

        pkg_dir = pathlib.Path(pkg.__file__).parent
        py_files = sorted(
            f.stem for f in pkg_dir.glob("*.py") if f.name != "__init__.py"
        )
        assert len(py_files) == 8, f"Expected 8 submodules, got {len(py_files)}: {py_files}"


# ── 2. No circular imports ───────────────────────────────────────────


class TestNoCircularImports:
    """Import each submodule with the full grounding subtree evicted from sys.modules.

    Popping only the target module is insufficient — circular dependencies only
    manifest when *both* sides of the cycle are absent from the module cache.
    """

    @pytest.mark.parametrize("module_name", EXPECTED_SUBMODULES)
    def test_independent_import(self, module_name: str):
        prefix = "scion.agents.grounding"
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith(prefix)}
        try:
            importlib.import_module(module_name)
        finally:
            sys.modules.update(saved)


# ── 3. Backward-compatible import paths ──────────────────────────────


class TestBackwardCompatibility:
    """Callers using the old import path must still work and resolve to the same class."""

    def test_import_from_agents_module(self):
        from scion.agents import GroundingAgent

        assert inspect.isclass(GroundingAgent)

    def test_import_from_grounding_agent_module(self):
        from scion.agents.grounding_agent import GroundingAgent

        assert inspect.isclass(GroundingAgent)

    def test_import_from_top_level(self):
        from scion import GroundingAgent

        assert inspect.isclass(GroundingAgent)

    def test_all_import_paths_resolve_to_same_class(self):
        """Identity check — a broken re-export that duplicates the class would fail here."""
        from scion import GroundingAgent as GA_top
        from scion.agents import GroundingAgent as GA_agents
        from scion.agents.grounding_agent import GroundingAgent as GA_module

        assert GA_top is GA_agents is GA_module


# ── 4. Facade delegation ────────────────────────────────────────────


class _StubLLM:
    model = "stub-model"


class _StubGroundingClient:
    pass


class TestFacadeDelegation:
    """Every public/protected method on GroundingAgent delegates to grounding/ submodule."""

    def _make_agent(self):
        from scion.agents.grounding_agent import GroundingAgent

        with patch("scion.agents.base.BaseAgent.__init__", return_value=None):
            agent = GroundingAgent.__new__(GroundingAgent)
            # BaseAgent attrs
            agent._name = "TestAgent"
            agent._backend_scope = ["gui", "shell"]
            agent._grounding_client = MagicMock()
            agent._llm_client = MagicMock()
            agent._recording_manager = None
            agent._step = 0
            agent._status = "active"
            # GroundingAgent attrs
            agent._system_prompt = "test"
            agent._max_iterations = 5
            agent._visual_analysis_timeout = 10.0
            agent._tool_retrieval_llm = None
            agent._visual_analysis_model = None
            agent._skill_context = None
            agent._active_skill_ids = []
            agent._skill_registry = None
            agent._last_tools = []
        return agent

    def test_make_agent_covers_all_init_attrs(self):
        from scion.agents.grounding_agent import GroundingAgent

        source = inspect.getsource(GroundingAgent.__init__)
        init_attrs = {m.group(1) for m in re.finditer(r"self\.(_\w+)\s*=", source)}
        agent = self._make_agent()
        instance_attrs = set(vars(agent))
        missing = init_attrs - instance_attrs
        assert not missing, f"_make_agent() missing attrs from __init__: {missing}"

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
        from scion.agents.grounding_agent import GroundingAgent

        msgs = [{"role": "user", "content": "x" * 500_000}]
        result = GroundingAgent._cap_message_content(msgs)
        assert len(result[0]["content"]) < 500_000

    def test_truncate_messages_delegates(self):
        agent = self._make_agent()
        # Generate enough messages that truncation kicks in (each ~big content)
        msgs = [{"role": "user", "content": "x" * 10_000} for _ in range(20)]
        result = agent._truncate_messages(msgs, keep_recent=3, max_tokens_estimate=1000)
        # Must actually truncate — result should be shorter than input
        assert len(result) < len(msgs), "truncate_messages should reduce message count"
        assert len(result) >= 3, "should keep at least keep_recent messages"

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
        from scion.agents.grounding_agent import GroundingAgent

        assert callable(GroundingAgent._select_key_screenshots)
        assert callable(GroundingAgent._get_workspace_path)
        assert callable(GroundingAgent._scan_workspace_files)
        assert callable(GroundingAgent._build_iteration_feedback)
        assert callable(GroundingAgent._remove_previous_guidance)
        assert callable(GroundingAgent._format_tool_executions)
        assert callable(GroundingAgent._check_task_completion)
        assert callable(GroundingAgent._extract_last_assistant_message)


# ── 4b. Async delegation tests ──────────────────────────────────────


class TestAsyncDelegation:
    """Async facade methods must delegate to the correct submodule function."""

    def _make_agent(self):
        from scion.agents.grounding_agent import GroundingAgent

        with patch("scion.agents.base.BaseAgent.__init__", return_value=None):
            agent = GroundingAgent.__new__(GroundingAgent)
            # BaseAgent attrs
            agent._name = "TestAgent"
            agent._backend_scope = ["gui", "shell"]
            agent._grounding_client = MagicMock()
            agent._llm_client = MagicMock()
            agent._recording_manager = None
            agent._step = 0
            agent._status = "active"
            # GroundingAgent attrs
            agent._system_prompt = "test"
            agent._max_iterations = 5
            agent._visual_analysis_timeout = 10.0
            agent._tool_retrieval_llm = None
            agent._visual_analysis_model = None
            agent._skill_context = None
            agent._active_skill_ids = []
            agent._skill_registry = None
            agent._last_tools = []
        return agent

    @pytest.mark.asyncio
    async def test_process_delegates_to_execution(self):
        agent = self._make_agent()
        # Patch the module-level import reference in the facade module
        with patch(
            "scion.agents.grounding_agent._process_impl",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_proc.return_value = {"status": "success", "response": "done"}
            result = await agent.process({"instruction": "test task"})
            mock_proc.assert_called_once_with(agent, {"instruction": "test task"})
            assert result == {"status": "success", "response": "done"}

    @pytest.mark.asyncio
    async def test_get_available_tools_delegates(self):
        agent = self._make_agent()
        with patch(
            "scion.agents.grounding_agent._get_available_tools_impl",
            new_callable=AsyncMock,
        ) as mock_tools:
            mock_tools.return_value = [{"name": "tool1"}]
            result = await agent._get_available_tools("describe task")
            mock_tools.assert_called_once_with(agent, "describe task")
            assert result == [{"name": "tool1"}]

    @pytest.mark.asyncio
    async def test_load_all_tools_delegates(self):
        agent = self._make_agent()
        mock_gc = MagicMock()
        with patch(
            "scion.agents.grounding_agent._load_all_tools_impl",
            new_callable=AsyncMock,
        ) as mock_load:
            mock_load.return_value = [{"name": "fallback_tool"}]
            result = await agent._load_all_tools(mock_gc)
            mock_load.assert_called_once_with(agent, mock_gc)
            assert result == [{"name": "fallback_tool"}]

    @pytest.mark.asyncio
    async def test_visual_analysis_callback_delegates(self):
        agent = self._make_agent()
        fake_result = MagicMock()
        with patch(
            "scion.agents.grounding_agent._visual_analysis_callback_impl",
            new_callable=AsyncMock,
        ) as mock_cb:
            mock_cb.return_value = fake_result
            result = await agent._visual_analysis_callback(fake_result, "click", {}, "gui")
            mock_cb.assert_called_once_with(agent, fake_result, "click", {}, "gui")

    @pytest.mark.asyncio
    async def test_enhance_result_with_visual_context_delegates(self):
        agent = self._make_agent()
        fake_result = MagicMock()
        with patch(
            "scion.agents.grounding_agent._enhance_result_with_visual_context_impl",
            new_callable=AsyncMock,
        ) as mock_enh:
            mock_enh.return_value = fake_result
            result = await agent._enhance_result_with_visual_context(fake_result, "screenshot")
            mock_enh.assert_called_once_with(agent, fake_result, "screenshot")

    @pytest.mark.asyncio
    async def test_check_workspace_artifacts_delegates(self):
        agent = self._make_agent()
        with patch(
            "scion.agents.grounding_agent._check_workspace_artifacts_impl",
            new_callable=AsyncMock,
        ) as mock_ws:
            mock_ws.return_value = {"artifacts": []}
            result = await agent._check_workspace_artifacts({"instruction": "test"})
            mock_ws.assert_called_once_with(agent, {"instruction": "test"})
            assert result == {"artifacts": []}

    @pytest.mark.asyncio
    async def test_generate_final_summary_delegates(self):
        agent = self._make_agent()
        with patch(
            "scion.agents.grounding_agent._generate_final_summary_impl",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = ("summary text", True, [])
            result = await agent._generate_final_summary("do thing", [], 3)
            mock_gen.assert_called_once_with(agent, "do thing", [], 3)
            assert result == ("summary text", True, [])

    @pytest.mark.asyncio
    async def test_build_final_result_delegates(self):
        agent = self._make_agent()
        with patch(
            "scion.agents.grounding_agent._build_final_result_impl",
            new_callable=AsyncMock,
        ) as mock_build:
            mock_build.return_value = {"status": "complete"}
            result = await agent._build_final_result("inst", [], [], 2, 5)
            mock_build.assert_called_once_with(
                agent, "inst", [], [], 2, 5, None, None, None
            )
            assert result == {"status": "complete"}

    @pytest.mark.asyncio
    async def test_record_agent_execution_delegates(self):
        agent = self._make_agent()
        with patch(
            "scion.agents.grounding_agent._record_agent_execution_impl",
            new_callable=AsyncMock,
        ) as mock_rec:
            await agent._record_agent_execution({"status": "ok"}, "test task")
            mock_rec.assert_called_once_with(agent, {"status": "ok"}, "test task")


# ── 5. Facade line-count guard ───────────────────────────────────────


class TestFacadeSize:
    """Ensure the facade stays thin — regression guard."""

    def test_facade_under_250_lines(self):
        import pathlib

        facade = pathlib.Path(__file__).resolve().parent.parent / "scion" / "agents" / "grounding_agent.py"
        line_count = len(facade.read_text(encoding="utf-8").splitlines())
        assert line_count < 250, (
            f"grounding_agent.py is {line_count} lines — should be < 250 as a thin facade"
        )

    def test_grounding_package_has_bulk(self):
        """The package modules should collectively hold > 800 lines."""
        import pathlib

        pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "scion" / "agents" / "grounding"
        total = sum(
            len(f.read_text(encoding="utf-8").splitlines())
            for f in pkg_dir.glob("*.py")
            if f.name != "__init__.py"
        )
        assert total > 800, (
            f"grounding/ package has only {total} lines — expected > 800 after full extraction"
        )
