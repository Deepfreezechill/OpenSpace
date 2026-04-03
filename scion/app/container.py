"""AppContainer — composition root for SkillGuard services.

The container holds **Optional** references to all domain services,
typed against Protocol interfaces.  Services are wired by
:func:`build_container` (production) or :func:`build_test_container`
(testing with mocks).

Lifecycle::

    container = await build_container(config)
    await container.startup()
    ...
    await container.shutdown()

Direct construction is also supported for testing::

    container = AppContainer(llm=mock_llm, skill_store=mock_store)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from scion.domain.ports import (
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

# ── Lifecycle callback type ───────────────────────────────────────────

LifecycleHook = Any  # Callable[[], Awaitable[None]] — avoid complex generic


@dataclass
class AppContainer:
    """Composition root — holds all domain service references.

    Every field is ``Optional`` so the container can be partially wired
    (e.g. CLI mode doesn't need ``sandbox``).  Services are typed
    against Protocol interfaces, never concrete classes.

    Attributes are grouped by domain concern:

    **Core services** — always expected in production:
        llm, agent_executor, telemetry

    **Skill engine** — enabled when skill features are active:
        skill_store, skill_evolution, analysis, cloud_skill

    **Security / sandbox** — Phase 2+ features:
        sandbox, policy_engine, auth, secret_broker, capability_lease_resolver

    **Tool layer**:
        tool_backend
    """

    # ── Core services ─────────────────────────────────────────────────
    llm: Optional[LLMClientPort] = None
    agent_executor: Optional[AgentExecutorPort] = None
    telemetry: Optional[TelemetryPort] = None

    # ── Skill engine ──────────────────────────────────────────────────
    skill_store: Optional[SkillStorePort] = None
    skill_evolution: Optional[SkillEvolutionPort] = None
    analysis: Optional[AnalysisPort] = None
    cloud_skill: Optional[CloudSkillPort] = None

    # ── Security / sandbox ────────────────────────────────────────────
    sandbox: Optional[SandboxPort] = None
    policy_engine: Optional[PolicyEnginePort] = None
    auth: Optional[AuthPort] = None
    secret_broker: Optional[SecretBrokerPort] = None
    capability_lease_resolver: Optional[CapabilityLeaseResolverPort] = None

    # ── Tool layer ────────────────────────────────────────────────────
    tool_backend: Optional[ToolBackendPort] = None

    # ── Lifecycle hooks ───────────────────────────────────────────────
    _startup_hooks: List[LifecycleHook] = field(default_factory=list, repr=False)
    _shutdown_hooks: List[LifecycleHook] = field(default_factory=list, repr=False)
    _started: bool = field(default=False, repr=False)

    # ── Lifecycle management ──────────────────────────────────────────

    def register_startup_hook(self, hook: LifecycleHook) -> None:
        """Register an async callable to run during :meth:`startup`."""
        self._startup_hooks.append(hook)

    def register_shutdown_hook(self, hook: LifecycleHook) -> None:
        """Register an async callable to run during :meth:`shutdown`."""
        self._shutdown_hooks.append(hook)

    async def startup(self) -> None:
        """Run all registered startup hooks in order.

        On partial failure, marks the container as started so that
        :meth:`shutdown` can clean up resources from hooks that
        succeeded.  Re-raises the original exception after marking.

        Raises :class:`RuntimeError` if called when already started.
        """
        if self._started:
            raise RuntimeError("AppContainer already started")
        try:
            for hook in self._startup_hooks:
                await hook()
        except Exception:
            # Mark as started so shutdown() can clean up partial resources
            self._started = True
            raise
        self._started = True

    async def shutdown(self) -> None:
        """Run all registered shutdown hooks in reverse order.

        Collects exceptions from hooks and raises the first one after
        all hooks have been attempted (best-effort cleanup).
        Safe to call on un-started containers (no-op).
        """
        if not self._started:
            return  # idempotent — safe to call on un-started container
        errors: List[Exception] = []
        for hook in reversed(self._shutdown_hooks):
            try:
                await hook()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                errors.append(exc)
        self._started = False
        if errors:
            raise errors[0]

    @property
    def is_started(self) -> bool:
        """Whether :meth:`startup` has been called."""
        return self._started

    # ── Convenience accessors ─────────────────────────────────────────

    # Service slot names (for validation in require())
    _SERVICE_SLOTS = frozenset(
        {
            "llm",
            "agent_executor",
            "telemetry",
            "skill_store",
            "skill_evolution",
            "analysis",
            "cloud_skill",
            "sandbox",
            "policy_engine",
            "auth",
            "secret_broker",
            "capability_lease_resolver",
            "tool_backend",
        }
    )

    def require(self, service_name: str) -> Any:
        """Get a service by name, raising if it is ``None``.

        Example::

            llm = container.require("llm")
            # equivalent to: assert container.llm is not None

        Raises:
            AttributeError: If *service_name* is not a known service slot.
            RuntimeError: If the service is known but not wired (``None``).
        """
        if service_name not in self._SERVICE_SLOTS:
            raise AttributeError(f"Unknown service '{service_name}'. Known services: {sorted(self._SERVICE_SLOTS)}")
        value = getattr(self, service_name)
        if value is None:
            raise RuntimeError(f"Required service '{service_name}' is not wired in AppContainer")
        return value
