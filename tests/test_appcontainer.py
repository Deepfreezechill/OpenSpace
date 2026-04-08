"""Tests for EPIC 1.3 — AppContainer (Composition Root).

Issues #64-67:
- #64: AppContainer dataclass
- #65: build_container factory
- #66: build_test_container factory with mocks
- #67: startup() / shutdown() lifecycle hooks

Validates:
- Container construction with all ports Optional
- Lifecycle hooks (startup/shutdown) ordering and error handling
- require() accessor raises on missing services
- build_container wires services and registers shutdown hooks
- build_test_container provides working stubs
- Protocol compliance of test stubs
"""

from __future__ import annotations

from typing import Any

import pytest

from scion.app.container import AppContainer
from scion.app.factory import (
    _StubLLM,
    _StubSkillStore,
    _StubTelemetry,
    build_container,
    build_test_container,
)
from scion.domain.ports import LLMClientPort, SkillStorePort, TelemetryPort

# ══════════════════════════════════════════════════════════════════════
# Container construction
# ══════════════════════════════════════════════════════════════════════


class TestContainerConstruction:
    """#64 — AppContainer dataclass creation."""

    def test_empty_container(self):
        """All services default to None."""
        c = AppContainer()
        assert c.llm is None
        assert c.agent_executor is None
        assert c.skill_store is None
        assert c.sandbox is None
        assert c.telemetry is None
        assert c.auth is None
        assert c.secret_broker is None
        assert c.capability_lease_resolver is None
        assert c.policy_engine is None
        assert c.tool_backend is None
        assert c.skill_evolution is None
        assert c.analysis is None
        assert c.cloud_skill is None
        assert c.is_started is False

    def test_partial_wiring(self):
        """Container can be partially wired."""
        stub_llm = _StubLLM()
        c = AppContainer(llm=stub_llm)
        assert c.llm is stub_llm
        assert c.skill_store is None

    def test_full_wiring(self):
        """Container accepts all 13 service slots."""
        stub = _StubLLM()
        c = AppContainer(
            llm=stub,
            agent_executor=None,
            telemetry=None,
            skill_store=None,
            skill_evolution=None,
            analysis=None,
            cloud_skill=None,
            sandbox=None,
            policy_engine=None,
            auth=None,
            secret_broker=None,
            capability_lease_resolver=None,
            tool_backend=None,
        )
        assert c.llm is stub


# ══════════════════════════════════════════════════════════════════════
# Lifecycle hooks
# ══════════════════════════════════════════════════════════════════════


class TestLifecycle:
    """#67 — startup() / shutdown() lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_startup_runs_hooks_in_order(self):
        order = []
        c = AppContainer()
        c.register_startup_hook(lambda: _async_record(order, "a"))
        c.register_startup_hook(lambda: _async_record(order, "b"))
        c.register_startup_hook(lambda: _async_record(order, "c"))
        await c.startup()
        assert order == ["a", "b", "c"]
        assert c.is_started is True

    @pytest.mark.asyncio
    async def test_shutdown_runs_hooks_in_reverse(self):
        order = []
        c = AppContainer()
        c.register_shutdown_hook(lambda: _async_record(order, "x"))
        c.register_shutdown_hook(lambda: _async_record(order, "y"))
        c.register_shutdown_hook(lambda: _async_record(order, "z"))
        c._started = True
        await c.shutdown()
        assert order == ["z", "y", "x"]
        assert c.is_started is False

    @pytest.mark.asyncio
    async def test_startup_raises_if_already_started(self):
        c = AppContainer()
        await c.startup()
        with pytest.raises(RuntimeError, match="already started"):
            await c.startup()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent_when_not_started(self):
        """Shutdown on un-started container is a safe no-op."""
        c = AppContainer()
        await c.shutdown()  # should not raise
        assert c.is_started is False

    @pytest.mark.asyncio
    async def test_shutdown_collects_errors(self):
        """All hooks run even if some fail; first error is raised."""
        order = []
        c = AppContainer()

        async def failing_hook():
            order.append("fail")
            raise ValueError("hook failed")

        c.register_shutdown_hook(lambda: _async_record(order, "first"))
        c.register_shutdown_hook(failing_hook)
        c.register_shutdown_hook(lambda: _async_record(order, "last"))
        c._started = True

        with pytest.raises(ValueError, match="hook failed"):
            await c.shutdown()

        # All hooks ran (reverse order: last, failing, first)
        assert order == ["last", "fail", "first"]
        assert c.is_started is False

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """startup → use → shutdown cycle."""
        log = []
        c = AppContainer(llm=_StubLLM())
        c.register_startup_hook(lambda: _async_record(log, "started"))
        c.register_shutdown_hook(lambda: _async_record(log, "stopped"))

        await c.startup()
        assert c.is_started
        # Use a service
        result = await c.require("llm").complete("hello")
        assert result["content"] == "stub response"

        await c.shutdown()
        assert not c.is_started
        assert log == ["started", "stopped"]

    @pytest.mark.asyncio
    async def test_partial_startup_allows_shutdown_cleanup(self):
        """If startup fails mid-way, shutdown() still runs to clean up."""
        log = []

        async def hook_a():
            log.append("a_started")

        async def hook_b():
            log.append("b_failed")
            raise RuntimeError("hook B exploded")

        c = AppContainer()
        c.register_startup_hook(hook_a)
        c.register_startup_hook(hook_b)
        c.register_shutdown_hook(lambda: _async_record(log, "cleanup"))

        with pytest.raises(RuntimeError, match="hook B exploded"):
            await c.startup()

        # Container is marked started so shutdown can clean up
        assert c.is_started is True
        await c.shutdown()
        assert c.is_started is False
        assert log == ["a_started", "b_failed", "cleanup"]


# ══════════════════════════════════════════════════════════════════════
# require() accessor
# ══════════════════════════════════════════════════════════════════════


class TestRequire:
    def test_require_returns_service(self):
        stub = _StubLLM()
        c = AppContainer(llm=stub)
        assert c.require("llm") is stub

    def test_require_raises_on_missing(self):
        c = AppContainer()
        with pytest.raises(RuntimeError, match="not wired"):
            c.require("llm")

    def test_require_raises_on_unknown(self):
        c = AppContainer()
        with pytest.raises(AttributeError, match="Unknown service"):
            c.require("nonexistent_service")


# ══════════════════════════════════════════════════════════════════════
# build_container factory
# ══════════════════════════════════════════════════════════════════════


class TestBuildContainer:
    """#65 — build_container factory."""

    @pytest.mark.asyncio
    async def test_build_empty(self):
        c = await build_container()
        assert c.llm is None
        assert c.is_started is False

    @pytest.mark.asyncio
    async def test_build_with_services(self):
        stub_llm = _StubLLM()
        stub_store = _StubSkillStore()
        c = await build_container(llm=stub_llm, skill_store=stub_store)
        assert c.llm is stub_llm
        assert c.skill_store is stub_store

    @pytest.mark.asyncio
    async def test_sandbox_shutdown_hook_registered(self):
        """build_container auto-registers sandbox.stop as shutdown hook."""

        class FakeSandbox:
            stopped = False

            async def start(self) -> bool:
                return True

            async def stop(self) -> None:
                self.stopped = True

            async def execute_safe(self, command: str, **kw: Any) -> Any:
                return None

            @property
            def is_active(self) -> bool:
                return True

        sb = FakeSandbox()
        c = await build_container(sandbox=sb)
        c._started = True
        await c.shutdown()
        assert sb.stopped is True

    @pytest.mark.asyncio
    async def test_telemetry_shutdown_hook_registered(self):
        """build_container auto-registers telemetry.shutdown as shutdown hook."""
        telem = _StubTelemetry()
        c = await build_container(telemetry=telem)
        c._started = True
        # Shutdown should call telemetry.shutdown() — no error means success
        await c.shutdown()


# ══════════════════════════════════════════════════════════════════════
# build_test_container factory
# ══════════════════════════════════════════════════════════════════════


class TestBuildTestContainer:
    """#66 — build_test_container with stubs."""

    def test_default_stubs(self):
        c = build_test_container()
        assert c.llm is not None
        assert c.skill_store is not None
        assert c.telemetry is not None

    def test_override_stubs(self):
        custom_llm = _StubLLM()
        c = build_test_container(llm=custom_llm)
        assert c.llm is custom_llm

    @pytest.mark.asyncio
    async def test_stub_llm_works(self):
        c = build_test_container()
        result = await c.llm.complete("hello")
        assert result["content"] == "stub response"
        assert c.llm.estimate_tokens("hello world") >= 1

    @pytest.mark.asyncio
    async def test_stub_skill_store_crud(self):
        """StubSkillStore supports full CRUD cycle."""
        from dataclasses import dataclass

        @dataclass
        class FakeManifest:
            skill_id: str = "test-skill"

        c = build_test_container()
        store = c.skill_store

        # Save
        manifest = FakeManifest()
        await store.save_record(manifest)
        assert store.count() == 1

        # Load
        loaded = store.load_record("test-skill")
        assert loaded is manifest

        # Load all
        all_records = store.load_all()
        assert "test-skill" in all_records

        # Load active
        active = store.load_active()
        assert "test-skill" in active

        # Delete
        deleted = await store.delete_record("test-skill")
        assert deleted is True
        assert store.count() == 0

        # Delete non-existent
        assert await store.delete_record("nope") is False

    def test_stub_telemetry_captures(self):
        c = build_test_container()
        telem = c.telemetry
        telem.capture("event1", {"key": "val"})
        telem.capture("event2")
        assert len(telem.events) == 2
        assert telem.events[0] == ("event1", {"key": "val"})
        telem.flush()  # no-op, should not raise
        telem.shutdown()  # no-op, should not raise


# ══════════════════════════════════════════════════════════════════════
# Protocol compliance
# ══════════════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """Stubs satisfy their Protocol interfaces at runtime."""

    def test_stub_llm_is_llm_port(self):
        assert isinstance(_StubLLM(), LLMClientPort)

    def test_stub_store_is_store_port(self):
        assert isinstance(_StubSkillStore(), SkillStorePort)

    def test_stub_telemetry_is_telemetry_port(self):
        assert isinstance(_StubTelemetry(), TelemetryPort)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


async def _async_record(log: list, value: str) -> None:
    log.append(value)
