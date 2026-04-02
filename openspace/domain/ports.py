"""Port protocols — structural typing contracts for all domain boundaries.

Each ``Protocol`` here defines the *minimal* interface that the domain
layer requires from an adapter.  Concrete implementations live in
infrastructure packages (``llm``, ``cloud``, ``grounding``, etc.).

Usage::

    from openspace.domain.ports import SkillStorePort

    def some_service(store: SkillStorePort) -> None:
        record = store.load_record("skill-42")
        ...

All methods use domain types (``openspace.domain.types``) at the
boundary, **not** adapter-specific types.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from openspace.domain.types import (
    CapabilityLease,
    EvolutionRequest,
    EvolutionResult,
    ExecutionAnalysisSnapshot,
    SandboxPolicy,
    SkillManifest,
    SkillSearchResult,
    TaskRequest,
    TaskResult,
    ToolCallResult,
    ToolDescriptor,
)

# ═══════════════════════════════════════════════════════════════════════
# 1. SkillStorePort — persistence for skill records & analyses
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class SkillStorePort(Protocol):
    """Port for skill record persistence (Issue #42).

    Note: The concrete ``SkillStore`` uses ``SkillRecord`` internally.
    An adapter (Phase 1.3) will map between ``SkillManifest`` and
    ``SkillRecord`` at the boundary.
    """

    async def save_record(self, record: SkillManifest) -> None: ...

    def load_record(self, skill_id: str) -> Optional[SkillManifest]: ...

    def load_all(self, *, active_only: bool = False) -> Dict[str, SkillManifest]: ...

    def load_active(self) -> Dict[str, SkillManifest]: ...

    async def delete_record(self, skill_id: str) -> bool: ...

    def count(self, *, active_only: bool = False) -> int: ...


# ═══════════════════════════════════════════════════════════════════════
# 2. LLMClientPort — language model completion
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class LLMClientPort(Protocol):
    """Port for LLM completions (Issue #43)."""

    async def complete(
        self,
        messages: List[Dict[str, Any]] | str,
        *,
        tools: Optional[List[Any]] = None,
        execute_tools: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Default implementation uses a simple heuristic.
        """
        return max(1, len(text) // 4)


# ═══════════════════════════════════════════════════════════════════════
# 3. CloudSkillPort — remote skill marketplace
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class CloudSkillPort(Protocol):
    """Port for cloud skill operations (Issue #44)."""

    async def search_skills(self, query: str, *, limit: int = 20) -> List[SkillSearchResult]: ...

    async def import_skill(self, skill_id: str, target_dir: str) -> Dict[str, Any]: ...

    async def publish_skill(self, skill_dir: str, *, visibility: str = "private") -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════════════
# 4. SandboxPort — isolated code execution
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class SandboxPort(Protocol):
    """Port for sandboxed execution environments (Issue #45)."""

    async def start(self) -> bool: ...

    async def stop(self) -> None: ...

    async def execute_safe(self, command: str, **kwargs: Any) -> Any: ...

    @property
    def is_active(self) -> bool: ...


# ═══════════════════════════════════════════════════════════════════════
# 5. SkillEvolutionPort — skill evolution engine
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class SkillEvolutionPort(Protocol):
    """Port for skill evolution (Issue #46)."""

    async def evolve(self, request: EvolutionRequest) -> Optional[EvolutionResult]: ...

    async def process_analysis(self, analysis: ExecutionAnalysisSnapshot) -> List[EvolutionResult]: ...

    async def wait_background(self) -> None: ...


# ═══════════════════════════════════════════════════════════════════════
# 6. AgentExecutorPort — task execution orchestration
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class AgentExecutorPort(Protocol):
    """Port for high-level task execution (Issue #47)."""

    async def execute(self, request: TaskRequest) -> TaskResult: ...


# ═══════════════════════════════════════════════════════════════════════
# 7. AnalysisPort — post-execution analysis
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class AnalysisPort(Protocol):
    """Port for execution analysis (Issue #48)."""

    async def analyze_execution(
        self,
        task_id: str,
        recording_dir: str,
        execution_result: Dict[str, Any],
        *,
        available_tools: Optional[List[Any]] = None,
    ) -> Optional[ExecutionAnalysisSnapshot]: ...


# ═══════════════════════════════════════════════════════════════════════
# 8. ToolBackendPort — tool discovery and invocation
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class ToolBackendPort(Protocol):
    """Port for tool backends (Issue #49).

    Note: Concrete ``Provider`` subclasses use a different call signature.
    An adapter (Phase 1.3) will normalize the interface at the boundary.
    """

    async def list_tools(self, *, session_name: Optional[str] = None) -> List[ToolDescriptor]: ...

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        session_name: Optional[str] = None,
    ) -> ToolCallResult: ...


# ═══════════════════════════════════════════════════════════════════════
# 9. PolicyEnginePort — security policy evaluation
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class PolicyEnginePort(Protocol):
    """Port for security policy decisions (Issue #50).

    Uses ``str`` for ``backend_type`` to avoid coupling the domain layer
    to infrastructure enums.  Adapters convert to ``BackendType`` enum.
    """

    async def check_command_allowed(self, backend_type: str, command: str) -> bool: ...

    async def check_domain_allowed(self, backend_type: str, domain: str) -> bool: ...

    def get_policy(self, backend_type: str) -> SandboxPolicy: ...


# ═══════════════════════════════════════════════════════════════════════
# 10. AuthPort — authentication
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class AuthPort(Protocol):
    """Port for authentication (Issue #51)."""

    async def authenticate(self, token: str) -> bool: ...

    async def validate_token(self, token: str) -> tuple[bool, str]: ...


# ═══════════════════════════════════════════════════════════════════════
# 11. SecretBrokerPort — secure secret management (Phase 2)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class SecretBrokerPort(Protocol):
    """Port for secret brokering (Issue #52).

    Implementation deferred to Phase 2.  Interface defined here so
    downstream code can type-hint against it now.
    """

    async def get_secret(self, key: str, *, scope: str = "task") -> Optional[str]: ...

    async def revoke(self, key: str) -> bool: ...

    def list_available(self, *, scope: str = "task") -> List[str]: ...


# ═══════════════════════════════════════════════════════════════════════
# 12. TelemetryPort — event capture & metrics
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class TelemetryPort(Protocol):
    """Port for telemetry (Issue #53).

    Note: The concrete ``Telemetry`` class accepts ``BaseTelemetryEvent``
    objects.  An adapter (Phase 1.3) will map the ``(event_name, properties)``
    signature to the event-object API.
    """

    def capture(self, event_name: str, properties: Dict[str, Any]) -> None: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


# ═══════════════════════════════════════════════════════════════════════
# 13. CapabilityLeaseResolverPort — lease management (Phase 2)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class CapabilityLeaseResolverPort(Protocol):
    """Port for capability lease resolution (Issue #54).

    Implementation deferred to Phase 2.  Interface defined here so
    sandbox and security code can type-hint against it now.
    """

    async def acquire(
        self,
        capability: str,
        *,
        trust_tier: str = "T1",
        ttl_seconds: int = 300,
    ) -> Optional[CapabilityLease]: ...

    async def release(self, lease_id: str) -> bool: ...

    async def validate(self, lease_id: str) -> bool: ...

    async def list_active(self, *, granted_to: Optional[str] = None) -> List[CapabilityLease]: ...


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Filesystem Broker (EPIC 2.2)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class FilesystemBrokerPort(Protocol):
    """Port for policy-enforced filesystem access (EPIC 2.2).

    Provides jailed path resolution, read/write enforcement,
    deny-list checking, and TOCTOU-safe file operations.
    """

    def resolve(self, path: str) -> "Path": ...

    def check_read(self, path: str) -> "Path": ...

    def check_write(self, path: str, size_bytes: int = 0) -> "Path": ...

    def open_read(self, path: str) -> int: ...

    def open_write(self, path: str, size_bytes: int = 0) -> int: ...


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Network Proxy (EPIC 2.3)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class NetworkProxyPort(Protocol):
    """Port for policy-enforced outbound network access (EPIC 2.3).

    Provides domain allow/deny enforcement, port filtering,
    concurrent connection tracking, and proxy lifecycle management.
    """

    def check_request(self, domain: str, port: int) -> None: ...

    async def connect(self, domain: str, port: int) -> str: ...

    async def disconnect(self, connection_id: str) -> None: ...

    async def list_connections(self) -> list: ...

    async def shutdown(self) -> int: ...


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Process Broker (EPIC 2.4)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class ProcessBrokerPort(Protocol):
    """Port for policy-enforced process execution (EPIC 2.4).

    Provides command allow/deny enforcement, shell control,
    process tracking with concurrency limits, execution time bounds,
    and dangerous syscall (link/symlink) restriction.
    """

    def check_command(self, command: str, args: list[str]) -> None: ...

    def check_shell(self, shell_command: str) -> None: ...

    def track_process(self, pid: int, command: str) -> None: ...

    def release_process(self, pid: int) -> None: ...

    @property
    def active_count(self) -> int: ...

    def check_syscall(self, syscall: str, *args: str) -> None: ...


__all__ = [
    "AgentExecutorPort",
    "AnalysisPort",
    "AuthPort",
    "CapabilityLeaseResolverPort",
    "CloudSkillPort",
    "FilesystemBrokerPort",
    "LLMClientPort",
    "NetworkProxyPort",
    "PolicyEnginePort",
    "ProcessBrokerPort",
    "SandboxPort",
    "SecretBrokerPort",
    "SkillEvolutionPort",
    "SkillStorePort",
    "TelemetryPort",
    "ToolBackendPort",
]
