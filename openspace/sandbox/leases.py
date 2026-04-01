"""Capability lease schema, validation, and resolution.

EPIC 2.1 — Capability Lease System

Issues:
- #84: YAML schema definition (LeaseSchema Pydantic model)
- #85: Parser + validator (parse_lease, validate_lease)
- #86: Default tier templates (TIER_DEFAULTS)
- #87: Lease resolver implementing CapabilityLeaseResolverPort
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from openspace.domain.types import CapabilityLease, SandboxPolicy

# ---------------------------------------------------------------------------
# #84 — Capability Lease YAML Schema
# ---------------------------------------------------------------------------


class TrustTier(str, Enum):
    """Trust tiers from most restrictive to most permissive."""

    T0_UNTRUSTED = "T0"
    T1_BASIC = "T1"
    T2_STANDARD = "T2"
    T3_ELEVATED = "T3"
    T4_FULL = "T4"


REQUIRED_DENIED_PATHS = frozenset({"/etc/shadow", "/etc/passwd", "~/.ssh/*", "**/.env"})
REQUIRED_BLOCKED_DOMAINS = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.internal",
    "100.100.100.200",
    "fd00:ec2::254",
})
REQUIRED_BLOCKED_COMMANDS = frozenset({"rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "kill", "pkill"})


class FilesystemCapability(BaseModel):
    """Filesystem access capabilities."""

    read_paths: list[str] = Field(default_factory=list, description="Glob patterns for readable paths")
    write_paths: list[str] = Field(default_factory=list, description="Glob patterns for writable paths")
    denied_paths: list[str] = Field(
        default_factory=lambda: ["/etc/shadow", "/etc/passwd", "~/.ssh/*", "**/.env"],
        description="Glob patterns always denied",
    )
    max_file_size_mb: int = Field(default=10, ge=1, le=1024)
    temp_dir_only: bool = Field(default=True, description="Restrict writes to temp directories")

    @field_validator("denied_paths")
    @classmethod
    def _deny_list_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("denied_paths cannot be empty — security invariant")
        # Ensure required entries are always present
        merged = list(v)
        for required in REQUIRED_DENIED_PATHS:
            if required not in merged:
                merged.append(required)
        return merged


class NetworkCapability(BaseModel):
    """Network access capabilities."""

    allowed_domains: list[str] = Field(default_factory=list, description="Allowed outbound domains")
    blocked_domains: list[str] = Field(
        default_factory=lambda: [
            "169.254.169.254",
            "metadata.google.internal",
            "metadata.internal",
            "100.100.100.200",
            "fd00:ec2::254",
        ],
        description="Always-blocked domains (cloud metadata, etc.)",
    )
    max_connections: int = Field(default=5, ge=0, le=100)
    allowed_ports: list[int] = Field(default_factory=lambda: [80, 443], description="Allowed outbound ports")
    outbound_enabled: bool = Field(default=False)

    @field_validator("blocked_domains")
    @classmethod
    def _block_list_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("blocked_domains cannot be empty — cloud metadata must always be blocked")
        # Ensure required entries are always present
        merged = list(v)
        for required in REQUIRED_BLOCKED_DOMAINS:
            if required not in merged:
                merged.append(required)
        return merged


class ProcessCapability(BaseModel):
    """Process execution capabilities."""

    allowed_commands: list[str] = Field(default_factory=list, description="Allowed command basenames")
    blocked_commands: list[str] = Field(
        default_factory=lambda: ["rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "kill", "pkill"],
        description="Always-blocked commands",
    )
    max_processes: int = Field(default=3, ge=0, le=50)
    max_execution_time_s: int = Field(default=300, ge=1, le=3600)
    allow_shell: bool = Field(default=False, description="Allow shell invocation (bash/sh/cmd)")

    @field_validator("blocked_commands")
    @classmethod
    def _blocked_cmds_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("blocked_commands cannot be empty — safety invariant")
        merged = list(v)
        for required in REQUIRED_BLOCKED_COMMANDS:
            if required not in merged:
                merged.append(required)
        return merged


class ResourceCapability(BaseModel):
    """Resource usage limits."""

    max_memory_mb: int = Field(default=512, ge=64, le=8192)
    max_cpu_percent: int = Field(default=50, ge=1, le=100)
    max_disk_mb: int = Field(default=100, ge=1, le=10240)
    max_output_size_mb: int = Field(default=5, ge=1, le=100)


class SecretCapability(BaseModel):
    """Secret access capabilities."""

    allowed_scopes: list[str] = Field(
        default_factory=lambda: ["task"],
        description="Scopes this lease can access (task, session, global)",
    )
    allowed_keys: list[str] = Field(default_factory=list, description="Specific secret keys allowed (empty = none)")
    max_secrets: int = Field(default=0, ge=0, le=50, description="Max secrets accessible (0 = none)")


class LeaseSchema(BaseModel):
    """Complete capability lease definition.

    This is the Pydantic model that maps to a YAML lease file.
    Skills declare their required capabilities via this schema.
    """

    name: str = Field(description="Human-readable lease name")
    trust_tier: TrustTier = Field(default=TrustTier.T1_BASIC)
    ttl_seconds: int = Field(default=300, ge=10, le=3600, description="Lease duration")
    filesystem: FilesystemCapability = Field(default_factory=FilesystemCapability)
    network: NetworkCapability = Field(default_factory=NetworkCapability)
    process: ProcessCapability = Field(default_factory=ProcessCapability)
    resources: ResourceCapability = Field(default_factory=ResourceCapability)
    secrets: SecretCapability = Field(default_factory=SecretCapability)

    @model_validator(mode="after")
    def _validate_tier_consistency(self) -> "LeaseSchema":
        """Enforce per-tier upper bounds on capability domains."""
        tier = self.trust_tier

        if tier == TrustTier.T0_UNTRUSTED:
            # T0: strictest — sandbox-only isolation
            if self.network.outbound_enabled:
                raise ValueError("T0 (untrusted) cannot have outbound network enabled")
            if self.network.allowed_domains:
                raise ValueError(
                    "T0 (untrusted) cannot have allowed_domains when outbound is disabled"
                )
            if self.process.allow_shell:
                raise ValueError("T0 (untrusted) cannot allow shell execution")
            if self.process.max_processes > 1:
                raise ValueError("T0 (untrusted) cannot spawn more than 1 process")
            if self.secrets.max_secrets > 0:
                raise ValueError("T0 (untrusted) cannot access secrets")
            if not self.filesystem.temp_dir_only:
                raise ValueError("T0 (untrusted) must restrict writes to temp directories")
            if self.filesystem.write_paths:
                raise ValueError("T0 (untrusted) cannot have explicit write paths")
            if self.resources.max_memory_mb > 256:
                raise ValueError("T0 (untrusted) cannot exceed 256MB memory")

        elif tier == TrustTier.T1_BASIC:
            # T1: read-only + limited tools — no network, no shell
            if self.network.outbound_enabled:
                raise ValueError("T1 (basic) cannot have outbound network enabled")
            if self.network.allowed_domains:
                raise ValueError(
                    "T1 (basic) cannot have allowed_domains when outbound is disabled"
                )
            if self.process.allow_shell:
                raise ValueError("T1 (basic) cannot allow shell execution")
            if self.secrets.max_secrets > 0:
                raise ValueError("T1 (basic) cannot access secrets")

        elif tier == TrustTier.T2_STANDARD:
            # T2: read-write + network — no shell
            if self.process.allow_shell:
                raise ValueError("T2 (standard) cannot allow shell execution")

        return self


# ---------------------------------------------------------------------------
# #85 — Parser + Validator
# ---------------------------------------------------------------------------


def parse_lease(data: Dict[str, Any]) -> LeaseSchema:
    """Parse and validate a lease definition from a dict (e.g., YAML-loaded).

    Raises ``pydantic.ValidationError`` on invalid input.
    """
    return LeaseSchema.model_validate(data)


def validate_lease(data: Dict[str, Any]) -> list[str]:
    """Validate a lease definition and return a list of error messages.

    Returns an empty list if the lease is valid.
    """
    try:
        parse_lease(data)
        return []
    except Exception as exc:
        # Extract individual error messages from Pydantic ValidationError
        if hasattr(exc, "errors"):
            return [f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return [str(exc)]


# ---------------------------------------------------------------------------
# #86 — Default Tier Templates (T0–T4)
# ---------------------------------------------------------------------------

TIER_DEFAULTS: Dict[TrustTier, LeaseSchema] = {
    TrustTier.T0_UNTRUSTED: LeaseSchema(
        name="T0 — Untrusted (sandbox-only)",
        trust_tier=TrustTier.T0_UNTRUSTED,
        ttl_seconds=60,
        filesystem=FilesystemCapability(read_paths=[], write_paths=[], temp_dir_only=True, max_file_size_mb=1),
        network=NetworkCapability(outbound_enabled=False, max_connections=0, allowed_domains=[]),
        process=ProcessCapability(
            allowed_commands=["echo", "cat", "head", "tail"],
            max_processes=1,
            max_execution_time_s=30,
            allow_shell=False,
        ),
        resources=ResourceCapability(max_memory_mb=128, max_cpu_percent=10, max_disk_mb=10, max_output_size_mb=1),
        secrets=SecretCapability(max_secrets=0, allowed_scopes=[], allowed_keys=[]),
    ),
    TrustTier.T1_BASIC: LeaseSchema(
        name="T1 — Basic (read-only + limited tools)",
        trust_tier=TrustTier.T1_BASIC,
        ttl_seconds=300,
        filesystem=FilesystemCapability(read_paths=["workspace/**"], write_paths=[], temp_dir_only=True),
        network=NetworkCapability(outbound_enabled=False, max_connections=0),
        process=ProcessCapability(
            allowed_commands=["echo", "cat", "head", "tail", "grep", "find", "ls", "wc"],
            max_processes=2,
            max_execution_time_s=120,
            allow_shell=False,
        ),
        resources=ResourceCapability(max_memory_mb=256, max_cpu_percent=25, max_disk_mb=50),
        secrets=SecretCapability(max_secrets=0, allowed_scopes=["task"]),
    ),
    TrustTier.T2_STANDARD: LeaseSchema(
        name="T2 — Standard (read-write + network)",
        trust_tier=TrustTier.T2_STANDARD,
        ttl_seconds=600,
        filesystem=FilesystemCapability(
            read_paths=["workspace/**", "/tmp/**"],
            write_paths=["workspace/**"],
            temp_dir_only=False,
            max_file_size_mb=10,
        ),
        network=NetworkCapability(
            outbound_enabled=True,
            max_connections=5,
            allowed_domains=["*.githubusercontent.com", "pypi.org", "registry.npmjs.org"],
        ),
        process=ProcessCapability(
            allowed_commands=["echo", "cat", "head", "tail", "grep", "find", "ls", "wc", "python", "pip", "node", "npm"],
            max_processes=5,
            max_execution_time_s=300,
            allow_shell=False,
        ),
        resources=ResourceCapability(max_memory_mb=512, max_cpu_percent=50, max_disk_mb=100),
        secrets=SecretCapability(max_secrets=3, allowed_scopes=["task"]),
    ),
    TrustTier.T3_ELEVATED: LeaseSchema(
        name="T3 — Elevated (shell + broad access)",
        trust_tier=TrustTier.T3_ELEVATED,
        ttl_seconds=1200,
        filesystem=FilesystemCapability(
            read_paths=["**"],
            write_paths=["workspace/**", "/tmp/**"],
            temp_dir_only=False,
            max_file_size_mb=50,
        ),
        network=NetworkCapability(outbound_enabled=True, max_connections=20, allowed_domains=["*"]),
        process=ProcessCapability(
            allowed_commands=["*"],
            max_processes=10,
            max_execution_time_s=600,
            allow_shell=True,
        ),
        resources=ResourceCapability(max_memory_mb=2048, max_cpu_percent=75, max_disk_mb=500),
        secrets=SecretCapability(max_secrets=10, allowed_scopes=["task", "session"]),
    ),
    TrustTier.T4_FULL: LeaseSchema(
        name="T4 — Full trust (unrestricted)",
        trust_tier=TrustTier.T4_FULL,
        ttl_seconds=3600,
        filesystem=FilesystemCapability(
            read_paths=["**"],
            write_paths=["**"],
            temp_dir_only=False,
            max_file_size_mb=1024,
        ),
        network=NetworkCapability(outbound_enabled=True, max_connections=100, allowed_domains=["*"]),
        process=ProcessCapability(
            allowed_commands=["*"],
            max_processes=50,
            max_execution_time_s=3600,
            allow_shell=True,
        ),
        resources=ResourceCapability(max_memory_mb=8192, max_cpu_percent=100, max_disk_mb=10240, max_output_size_mb=100),
        secrets=SecretCapability(max_secrets=50, allowed_scopes=["task", "session", "global"]),
    ),
}


def get_tier_default(tier: TrustTier) -> LeaseSchema:
    """Return a deep copy of the default lease template for a trust tier."""
    return TIER_DEFAULTS[tier].model_copy(deep=True)


# ---------------------------------------------------------------------------
# #87 — Lease Resolver (implements CapabilityLeaseResolverPort)
# ---------------------------------------------------------------------------


class InMemoryLeaseResolver:
    """In-memory implementation of CapabilityLeaseResolverPort.

    Uses asyncio.Lock for safe concurrent access.  Production deployments
    may swap this for a database-backed resolver.
    """

    def __init__(self) -> None:
        self._leases: Dict[str, CapabilityLease] = {}
        self._lock = asyncio.Lock()

    _MIN_TTL = 10
    _MAX_TTL = 3600

    async def acquire(
        self,
        capability: str,
        *,
        trust_tier: str = "T1",
        ttl_seconds: int = 300,
    ) -> Optional[CapabilityLease]:
        """Create and store a new capability lease.

        Raises ``ValueError`` if *ttl_seconds* is outside 10–3600
        or *trust_tier* is not a recognised ``TrustTier`` value.

        **Authorization note:** This resolver is a bookkeeping layer —
        it records that a lease was granted but does not decide *who*
        may request *which* tier.  Tier authorization is enforced by the
        pre-execution pipeline (EPIC 2.10) and MCP auth layer (EPIC 2.5),
        which determine the maximum tier a caller may request before
        invoking ``acquire()``.
        """
        # Validate inputs at the resolver boundary
        valid_tiers = {t.value for t in TrustTier}
        if trust_tier not in valid_tiers:
            raise ValueError(f"Invalid trust_tier '{trust_tier}'; must be one of {sorted(valid_tiers)}")
        if not (self._MIN_TTL <= ttl_seconds <= self._MAX_TTL):
            raise ValueError(f"ttl_seconds must be {self._MIN_TTL}-{self._MAX_TTL}, got {ttl_seconds}")
        async with self._lock:
            lease_id = f"lease-{uuid.uuid4().hex[:12]}"
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            lease = CapabilityLease(
                lease_id=lease_id,
                capability=capability,
                granted_to="current_task",
                trust_tier=trust_tier,
                expires_at=expires_at,
                revoked=False,
            )
            self._leases[lease_id] = lease
            return lease

    async def release(self, lease_id: str) -> bool:
        """Revoke a lease by ID."""
        async with self._lock:
            if lease_id not in self._leases:
                return False
            old = self._leases[lease_id]
            # CapabilityLease is frozen — replace with revoked copy
            self._leases[lease_id] = CapabilityLease(
                lease_id=old.lease_id,
                capability=old.capability,
                granted_to=old.granted_to,
                trust_tier=old.trust_tier,
                expires_at=old.expires_at,
                revoked=True,
            )
            return True

    async def validate(self, lease_id: str) -> bool:
        """Check if a lease is active (not revoked, not expired)."""
        async with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.revoked:
                return False
            if lease.expires_at and lease.expires_at < datetime.now(timezone.utc):
                return False
            return True

    async def list_active(self, *, granted_to: Optional[str] = None) -> List[CapabilityLease]:
        """Return all non-revoked, non-expired leases."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            active: List[CapabilityLease] = []
            for lease in self._leases.values():
                if lease.revoked:
                    continue
                if lease.expires_at and lease.expires_at < now:
                    continue
                if granted_to and lease.granted_to != granted_to:
                    continue
                active.append(lease)
            return active


# ---------------------------------------------------------------------------
# Lease → SandboxPolicy conversion
# ---------------------------------------------------------------------------


def lease_to_sandbox_policy(lease_schema: LeaseSchema) -> SandboxPolicy:
    """Convert a LeaseSchema into a SandboxPolicy for runtime enforcement.

    Maps: trust_tier, allowed/blocked commands, allowed/blocked domains,
    max_execution_time_s, max_memory_mb.

    Deferred to Phase 2 broker EPICs (2.2–2.6): filesystem paths,
    temp_dir_only, max_file_size, outbound_enabled, allowed_ports,
    max_connections, allow_shell, max_processes, secret scopes.
    Those fields will be enforced by the respective broker layers.
    """
    return SandboxPolicy(
        sandbox_enabled=True,
        trust_tier=lease_schema.trust_tier.value,
        allowed_commands=frozenset(lease_schema.process.allowed_commands),
        blocked_commands=frozenset(lease_schema.process.blocked_commands),
        allowed_domains=frozenset(lease_schema.network.allowed_domains),
        blocked_domains=frozenset(lease_schema.network.blocked_domains),
        max_execution_time_s=lease_schema.process.max_execution_time_s,
        max_memory_mb=lease_schema.resources.max_memory_mb,
    )
