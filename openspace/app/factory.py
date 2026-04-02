"""Container factory — wires real or mock implementations.

Production::

    from openspace.app.factory import build_container

    container = await build_container(config)
    await container.startup()

Testing::

    from openspace.app.factory import build_test_container

    container = build_test_container()
    assert container.llm is not None  # pre-wired mock
"""

from __future__ import annotations

from typing import Any, Optional

from openspace.app.container import AppContainer
from openspace.domain.ports import (
    AgentExecutorPort,
    AnalysisPort,
    AuthPort,
    CapabilityLeaseResolverPort,
    CloudSkillPort,
    LLMClientPort,
    PolicyEnginePort,
    SandboxPort,
    SecretBrokerPort,
    SkillEvolutionPort,
    SkillStorePort,
    TelemetryPort,
    ToolBackendPort,
)

# ══════════════════════════════════════════════════════════════════════
# Production factory
# ══════════════════════════════════════════════════════════════════════


async def build_container(
    config: Any = None,
    *,
    llm: Optional[LLMClientPort] = None,
    skill_store: Optional[SkillStorePort] = None,
    telemetry: Optional[TelemetryPort] = None,
    agent_executor: Optional[AgentExecutorPort] = None,
    sandbox: Optional[SandboxPort] = None,
    policy_engine: Optional[PolicyEnginePort] = None,
    auth: Optional[AuthPort] = None,
    secret_broker: Optional[SecretBrokerPort] = None,
    capability_lease_resolver: Optional[CapabilityLeaseResolverPort] = None,
    tool_backend: Optional[ToolBackendPort] = None,
    skill_evolution: Optional[SkillEvolutionPort] = None,
    analysis: Optional[AnalysisPort] = None,
    cloud_skill: Optional[CloudSkillPort] = None,
) -> AppContainer:
    """Build an :class:`AppContainer` with real implementations.

    Currently accepts explicit service overrides.  In Phase 4, this
    will read ``config`` to auto-construct concrete implementations
    from :mod:`openspace.tool_layer.OpenSpaceConfig`.

    Parameters
    ----------
    config:
        Application configuration (reserved for Phase 4 auto-wiring).
    **kwargs:
        Explicit service instances to wire.  Any service not provided
        will remain ``None`` on the container.

    Returns
    -------
    AppContainer
        A wired (but not yet started) container.  Call
        :meth:`~AppContainer.startup` to run lifecycle hooks.
    """
    container = AppContainer(
        llm=llm,
        agent_executor=agent_executor,
        telemetry=telemetry,
        skill_store=skill_store,
        skill_evolution=skill_evolution,
        analysis=analysis,
        cloud_skill=cloud_skill,
        sandbox=sandbox,
        policy_engine=policy_engine,
        auth=auth,
        secret_broker=secret_broker,
        capability_lease_resolver=capability_lease_resolver,
        tool_backend=tool_backend,
    )

    # Register shutdown hooks for services that need cleanup
    if sandbox is not None:
        container.register_shutdown_hook(sandbox.stop)
    if telemetry is not None:
        container.register_shutdown_hook(lambda: _wrap_sync(telemetry.shutdown))

    return container


# ══════════════════════════════════════════════════════════════════════
# Test factory
# ══════════════════════════════════════════════════════════════════════


class _StubLLM:
    """Minimal LLM stub for testing — satisfies LLMClientPort."""

    async def complete(self, messages: Any, *, tools: Any = None, execute_tools: bool = True, **kw: Any) -> dict:
        return {"role": "assistant", "content": "stub response"}

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class _StubSkillStore:
    """Minimal skill store stub for testing — satisfies SkillStorePort."""

    def __init__(self) -> None:
        self._records: dict = {}

    async def save_record(self, record: Any) -> None:
        self._records[getattr(record, "skill_id", "unknown")] = record

    def load_record(self, skill_id: str) -> Any:
        return self._records.get(skill_id)

    def load_all(self, *, active_only: bool = False) -> dict:
        return dict(self._records)

    def load_active(self) -> dict:
        return dict(self._records)

    async def delete_record(self, skill_id: str) -> bool:
        return self._records.pop(skill_id, None) is not None

    def count(self, *, active_only: bool = False) -> int:
        return len(self._records)


class _StubTelemetry:
    """Minimal telemetry stub for testing — satisfies TelemetryPort."""

    def __init__(self) -> None:
        self.events: list = []

    def capture(self, event_name: str, properties: Any = None) -> None:
        self.events.append((event_name, properties))

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def build_test_container(
    *,
    llm: Optional[LLMClientPort] = None,
    skill_store: Optional[SkillStorePort] = None,
    telemetry: Optional[TelemetryPort] = None,
    **overrides: Any,
) -> AppContainer:
    """Build an :class:`AppContainer` pre-wired with stubs for testing.

    Provides sensible defaults for ``llm``, ``skill_store``, and
    ``telemetry``.  Pass explicit mocks to override any service.

    Returns
    -------
    AppContainer
        A container ready for unit/integration tests (not started).
    """
    return AppContainer(
        llm=llm or _StubLLM(),  # type: ignore[arg-type]
        skill_store=skill_store or _StubSkillStore(),  # type: ignore[arg-type]
        telemetry=telemetry or _StubTelemetry(),  # type: ignore[arg-type]
        **overrides,
    )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


async def _wrap_sync(fn: Any) -> None:
    """Wrap a synchronous callable as an async no-arg coroutine."""
    fn()
