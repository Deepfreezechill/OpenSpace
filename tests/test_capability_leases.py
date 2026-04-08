"""Tests for EPIC 2.1 — Capability Lease System.

Issues:
- #84: YAML schema (LeaseSchema Pydantic model)
- #85: Parser + validator
- #86: Default tier templates (T0–T4)
- #87: Lease resolver (InMemoryLeaseResolver)
- #88: Comprehensive test coverage

Validates:
- Schema validation and rejection of invalid inputs
- Tier consistency rules (T0 restrictions)
- Default templates for all 5 tiers
- Lease lifecycle (acquire, validate, release, expiry)
- Lease → SandboxPolicy conversion
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from scion.domain.ports import CapabilityLeaseResolverPort
from scion.domain.types import CapabilityLease, SandboxPolicy
from scion.sandbox.leases import (
    REQUIRED_BLOCKED_COMMANDS,
    REQUIRED_BLOCKED_DOMAINS,
    REQUIRED_DENIED_PATHS,
    TIER_DEFAULTS,
    FilesystemCapability,
    InMemoryLeaseResolver,
    LeaseSchema,
    NetworkCapability,
    ProcessCapability,
    ResourceCapability,
    SecretCapability,
    TrustTier,
    get_tier_default,
    lease_to_sandbox_policy,
    parse_lease,
    validate_lease,
)

# ---------------------------------------------------------------------------
# Schema Tests (#84)
# ---------------------------------------------------------------------------


class TestLeaseSchema:
    """LeaseSchema Pydantic model validates correctly."""

    def test_minimal_valid_schema(self) -> None:
        schema = LeaseSchema(name="test")
        assert schema.name == "test"
        assert schema.trust_tier == TrustTier.T1_BASIC
        assert schema.ttl_seconds == 300

    def test_full_schema(self) -> None:
        schema = LeaseSchema(
            name="full-test",
            trust_tier=TrustTier.T3_ELEVATED,
            ttl_seconds=1200,
            filesystem=FilesystemCapability(read_paths=["**"], write_paths=["workspace/**"]),
            network=NetworkCapability(outbound_enabled=True, max_connections=10),
            process=ProcessCapability(allow_shell=True, max_processes=5),
            resources=ResourceCapability(max_memory_mb=2048),
            secrets=SecretCapability(max_secrets=5, allowed_scopes=["task", "session"]),
        )
        assert schema.trust_tier == TrustTier.T3_ELEVATED
        assert schema.network.outbound_enabled is True
        assert schema.process.allow_shell is True

    def test_ttl_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ttl_seconds"):
            LeaseSchema(name="bad", ttl_seconds=5)

    def test_ttl_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ttl_seconds"):
            LeaseSchema(name="bad", ttl_seconds=7200)

    def test_memory_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_memory_mb"):
            LeaseSchema(name="bad", resources=ResourceCapability(max_memory_mb=32))

    def test_empty_deny_list_rejected(self) -> None:
        with pytest.raises(ValidationError, match="denied_paths"):
            FilesystemCapability(denied_paths=[])

    def test_empty_blocked_domains_rejected(self) -> None:
        with pytest.raises(ValidationError, match="blocked_domains"):
            NetworkCapability(blocked_domains=[])

    def test_default_denied_paths_include_sensitive(self) -> None:
        fs = FilesystemCapability()
        assert "/etc/shadow" in fs.denied_paths
        assert "~/.ssh/*" in fs.denied_paths
        assert "**/.env" in fs.denied_paths

    def test_default_blocked_domains_include_metadata(self) -> None:
        net = NetworkCapability()
        assert "169.254.169.254" in net.blocked_domains
        assert "metadata.google.internal" in net.blocked_domains


class TestTierConsistency:
    """T0 tier must enforce maximum restrictions."""

    def test_t0_cannot_enable_network(self) -> None:
        with pytest.raises(ValidationError, match="T0.*outbound network"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                network=NetworkCapability(outbound_enabled=True),
            )

    def test_t0_cannot_allow_shell(self) -> None:
        with pytest.raises(ValidationError, match="T0.*shell"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                process=ProcessCapability(allow_shell=True),
            )

    def test_t0_cannot_access_secrets(self) -> None:
        with pytest.raises(ValidationError, match="T0.*secrets"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                secrets=SecretCapability(max_secrets=1),
            )

    def test_t1_default_no_network(self) -> None:
        t1 = get_tier_default(TrustTier.T1_BASIC)
        assert t1.network.outbound_enabled is False

    def test_t2_allows_network(self) -> None:
        t2 = get_tier_default(TrustTier.T2_STANDARD)
        assert t2.network.outbound_enabled is True

    def test_t4_allows_everything(self) -> None:
        t4 = get_tier_default(TrustTier.T4_FULL)
        assert t4.process.allow_shell is True
        assert t4.network.outbound_enabled is True
        assert t4.secrets.max_secrets == 50
        assert t4.resources.max_memory_mb == 8192


# ---------------------------------------------------------------------------
# Parser Tests (#85)
# ---------------------------------------------------------------------------


class TestLeaseParser:
    """parse_lease and validate_lease handle dict input correctly."""

    def test_parse_valid_dict(self) -> None:
        data = {"name": "test-skill", "trust_tier": "T2", "ttl_seconds": 600}
        schema = parse_lease(data)
        assert schema.name == "test-skill"
        assert schema.trust_tier == TrustTier.T2_STANDARD

    def test_parse_with_nested_capabilities(self) -> None:
        data = {
            "name": "net-skill",
            "trust_tier": "T2",
            "network": {"outbound_enabled": True, "allowed_domains": ["api.example.com"]},
        }
        schema = parse_lease(data)
        assert schema.network.outbound_enabled is True
        assert "api.example.com" in schema.network.allowed_domains

    def test_parse_invalid_tier_raises(self) -> None:
        with pytest.raises(ValidationError):
            parse_lease({"name": "bad", "trust_tier": "T99"})

    def test_validate_returns_empty_on_valid(self) -> None:
        errors = validate_lease({"name": "valid"})
        assert errors == []

    def test_validate_returns_errors_on_invalid(self) -> None:
        errors = validate_lease({"name": "bad", "ttl_seconds": 1})
        assert len(errors) > 0
        assert any("ttl_seconds" in e for e in errors)

    def test_validate_missing_name(self) -> None:
        errors = validate_lease({})
        assert len(errors) > 0
        assert any("name" in e for e in errors)


# ---------------------------------------------------------------------------
# Tier Defaults Tests (#86)
# ---------------------------------------------------------------------------


class TestTierDefaults:
    """Default templates exist for all 5 tiers."""

    def test_all_tiers_have_defaults(self) -> None:
        for tier in TrustTier:
            default = get_tier_default(tier)
            assert default.trust_tier == tier
            assert default.name != ""

    def test_tier_count(self) -> None:
        assert len(TIER_DEFAULTS) == 5

    def test_tiers_ordered_by_permissiveness(self) -> None:
        """Each tier should allow ≥ the resources of the previous tier."""
        tiers = [
            TrustTier.T0_UNTRUSTED,
            TrustTier.T1_BASIC,
            TrustTier.T2_STANDARD,
            TrustTier.T3_ELEVATED,
            TrustTier.T4_FULL,
        ]
        for i in range(1, len(tiers)):
            prev = get_tier_default(tiers[i - 1])
            curr = get_tier_default(tiers[i])
            assert curr.resources.max_memory_mb >= prev.resources.max_memory_mb
            assert curr.process.max_processes >= prev.process.max_processes
            assert curr.ttl_seconds >= prev.ttl_seconds

    def test_t0_is_most_restrictive(self) -> None:
        t0 = get_tier_default(TrustTier.T0_UNTRUSTED)
        assert t0.network.outbound_enabled is False
        assert t0.process.allow_shell is False
        assert t0.secrets.max_secrets == 0
        assert t0.filesystem.read_paths == []
        assert t0.filesystem.write_paths == []

    def test_defaults_are_valid_schemas(self) -> None:
        """All defaults must pass their own validation."""
        for tier, schema in TIER_DEFAULTS.items():
            errors = validate_lease(schema.model_dump())
            assert errors == [], f"Tier {tier.value} default has errors: {errors}"


# ---------------------------------------------------------------------------
# Lease Resolver Tests (#87)
# ---------------------------------------------------------------------------


class TestInMemoryLeaseResolver:
    """InMemoryLeaseResolver implements the port correctly."""

    def test_implements_port(self) -> None:
        resolver = InMemoryLeaseResolver()
        assert isinstance(resolver, CapabilityLeaseResolverPort)

    @pytest.mark.asyncio
    async def test_acquire_returns_lease(self) -> None:
        resolver = InMemoryLeaseResolver()
        lease = await resolver.acquire("filesystem.read", trust_tier="T1", ttl_seconds=60)
        assert lease is not None
        assert isinstance(lease, CapabilityLease)
        assert lease.capability == "filesystem.read"
        assert lease.trust_tier == "T1"
        assert lease.revoked is False

    @pytest.mark.asyncio
    async def test_validate_active_lease(self) -> None:
        resolver = InMemoryLeaseResolver()
        lease = await resolver.acquire("network.outbound", ttl_seconds=60)
        assert lease is not None
        assert await resolver.validate(lease.lease_id) is True

    @pytest.mark.asyncio
    async def test_validate_nonexistent_lease(self) -> None:
        resolver = InMemoryLeaseResolver()
        assert await resolver.validate("nonexistent") is False

    @pytest.mark.asyncio
    async def test_release_revokes_lease(self) -> None:
        resolver = InMemoryLeaseResolver()
        lease = await resolver.acquire("process.shell", ttl_seconds=60)
        assert lease is not None

        result = await resolver.release(lease.lease_id)
        assert result is True
        assert await resolver.validate(lease.lease_id) is False

    @pytest.mark.asyncio
    async def test_release_nonexistent_returns_false(self) -> None:
        resolver = InMemoryLeaseResolver()
        assert await resolver.release("nonexistent") is False

    @pytest.mark.asyncio
    async def test_expired_lease_invalid(self) -> None:
        resolver = InMemoryLeaseResolver()
        lease = await resolver.acquire("test", ttl_seconds=10)
        assert lease is not None

        # Manually expire the lease
        expired = CapabilityLease(
            lease_id=lease.lease_id,
            capability=lease.capability,
            granted_to=lease.granted_to,
            trust_tier=lease.trust_tier,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            revoked=False,
        )
        resolver._leases[lease.lease_id] = expired
        assert await resolver.validate(lease.lease_id) is False

    @pytest.mark.asyncio
    async def test_list_active_filters_revoked(self) -> None:
        resolver = InMemoryLeaseResolver()
        l1 = await resolver.acquire("cap-1", ttl_seconds=60)
        l2 = await resolver.acquire("cap-2", ttl_seconds=60)
        assert l1 is not None and l2 is not None

        await resolver.release(l1.lease_id)
        active = await resolver.list_active()
        assert len(active) == 1
        assert active[0].lease_id == l2.lease_id

    async def test_list_active_filters_by_grantee(self) -> None:
        resolver = InMemoryLeaseResolver()
        await resolver.acquire("cap-1", ttl_seconds=60)
        active = await resolver.list_active(granted_to="current_task")
        assert len(active) == 1
        active_other = await resolver.list_active(granted_to="other_task")
        assert len(active_other) == 0

    @pytest.mark.asyncio
    async def test_multiple_leases_independent(self) -> None:
        resolver = InMemoryLeaseResolver()
        l1 = await resolver.acquire("fs.read", ttl_seconds=60)
        l2 = await resolver.acquire("net.out", ttl_seconds=60)
        assert l1 is not None and l2 is not None
        assert l1.lease_id != l2.lease_id

        await resolver.release(l1.lease_id)
        assert await resolver.validate(l1.lease_id) is False
        assert await resolver.validate(l2.lease_id) is True


# ---------------------------------------------------------------------------
# SandboxPolicy Conversion Tests
# ---------------------------------------------------------------------------


class TestLeaseSandboxConversion:
    """lease_to_sandbox_policy correctly maps lease → policy."""

    def test_t0_produces_restrictive_policy(self) -> None:
        t0 = get_tier_default(TrustTier.T0_UNTRUSTED)
        policy = lease_to_sandbox_policy(t0)
        assert isinstance(policy, SandboxPolicy)
        assert policy.sandbox_enabled is True
        assert policy.trust_tier == "T0"
        assert policy.max_memory_mb == 128
        assert policy.max_execution_time_s == 30

    def test_t4_produces_permissive_policy(self) -> None:
        t4 = get_tier_default(TrustTier.T4_FULL)
        policy = lease_to_sandbox_policy(t4)
        assert policy.trust_tier == "T4"
        assert policy.max_memory_mb == 8192
        assert policy.max_execution_time_s == 3600

    def test_policy_always_sandbox_enabled(self) -> None:
        """Lease-derived policies always have sandbox enabled."""
        for tier in TrustTier:
            policy = lease_to_sandbox_policy(get_tier_default(tier))
            assert policy.sandbox_enabled is True

    def test_blocked_commands_preserved(self) -> None:
        t1 = get_tier_default(TrustTier.T1_BASIC)
        policy = lease_to_sandbox_policy(t1)
        assert "rm" in policy.blocked_commands
        assert "shutdown" in policy.blocked_commands

    def test_allowed_domains_mapped(self) -> None:
        t2 = get_tier_default(TrustTier.T2_STANDARD)
        policy = lease_to_sandbox_policy(t2)
        assert "pypi.org" in policy.allowed_domains

    def test_custom_lease_to_policy(self) -> None:
        custom = LeaseSchema(
            name="custom",
            trust_tier=TrustTier.T2_STANDARD,
            process=ProcessCapability(
                allowed_commands=["python", "pip"],
                max_execution_time_s=120,
            ),
            resources=ResourceCapability(max_memory_mb=1024),
        )
        policy = lease_to_sandbox_policy(custom)
        assert "python" in policy.allowed_commands
        assert "pip" in policy.allowed_commands
        assert policy.max_execution_time_s == 120
        assert policy.max_memory_mb == 1024


# ---------------------------------------------------------------------------
# Security Regression Tests (R1 review fixes)
# ---------------------------------------------------------------------------


class TestSecurityRegressions:
    """Regression tests for review findings (R1 + R2)."""

    def test_custom_denied_paths_still_include_required(self) -> None:
        """Caller cannot strip required denied_paths by supplying custom list."""
        fs = FilesystemCapability(denied_paths=["/my/custom/path"])
        for required in REQUIRED_DENIED_PATHS:
            assert required in fs.denied_paths, f"{required} must always be in denied_paths"

    def test_custom_blocked_domains_still_include_required(self) -> None:
        """Caller cannot strip required blocked_domains — ALL metadata endpoints enforced."""
        net = NetworkCapability(blocked_domains=["evil.example.com"])
        for required in REQUIRED_BLOCKED_DOMAINS:
            assert required in net.blocked_domains, f"{required} must always be in blocked_domains"

    def test_get_tier_default_returns_independent_copy(self) -> None:
        """Mutating a returned default must not affect future calls."""
        d1 = get_tier_default(TrustTier.T0_UNTRUSTED)
        d1.filesystem.read_paths.append("/hacked")
        d2 = get_tier_default(TrustTier.T0_UNTRUSTED)
        assert "/hacked" not in d2.filesystem.read_paths

    def test_blocked_domains_include_additional_metadata(self) -> None:
        """Default blocked domains include AWS/Alibaba/IPv6 metadata."""
        net = NetworkCapability()
        assert "metadata.internal" in net.blocked_domains
        assert "100.100.100.200" in net.blocked_domains
        assert "fd00:ec2::254" in net.blocked_domains

    @pytest.mark.asyncio
    async def test_concurrent_lease_operations(self) -> None:
        """Resolver handles concurrent acquire/release safely."""
        resolver = InMemoryLeaseResolver()
        tasks = [resolver.acquire(f"cap-{i}", ttl_seconds=60) for i in range(10)]
        leases = await asyncio.gather(*tasks)
        lease_ids = {l.lease_id for l in leases if l}
        assert len(lease_ids) == 10

    # --- R2 regressions ---

    @pytest.mark.asyncio
    async def test_acquire_rejects_invalid_trust_tier(self) -> None:
        """Resolver rejects unknown trust_tier values."""
        resolver = InMemoryLeaseResolver()
        with pytest.raises(ValueError, match="Invalid trust_tier"):
            await resolver.acquire("test", trust_tier="BOGUS")

    @pytest.mark.asyncio
    async def test_acquire_rejects_out_of_range_ttl(self) -> None:
        """Resolver rejects TTL outside 10–3600."""
        resolver = InMemoryLeaseResolver()
        with pytest.raises(ValueError, match="ttl_seconds"):
            await resolver.acquire("test", ttl_seconds=5)
        with pytest.raises(ValueError, match="ttl_seconds"):
            await resolver.acquire("test", ttl_seconds=99999)

    def test_t0_cannot_have_write_paths(self) -> None:
        """T0 cannot declare explicit write paths."""
        with pytest.raises(ValidationError, match="T0.*write paths"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                filesystem=FilesystemCapability(write_paths=["workspace/**"]),
                process=ProcessCapability(max_processes=1, allow_shell=False),
                secrets=SecretCapability(max_secrets=0),
            )

    def test_t0_must_be_temp_dir_only(self) -> None:
        """T0 must restrict writes to temp directories."""
        with pytest.raises(ValidationError, match="T0.*temp dir"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                filesystem=FilesystemCapability(temp_dir_only=False),
                process=ProcessCapability(max_processes=1, allow_shell=False),
                secrets=SecretCapability(max_secrets=0),
            )

    def test_t0_cannot_exceed_memory_cap(self) -> None:
        """T0 cannot exceed 256MB memory."""
        with pytest.raises(ValidationError, match="T0.*256MB"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                resources=ResourceCapability(max_memory_mb=512),
                process=ProcessCapability(max_processes=1, allow_shell=False),
                secrets=SecretCapability(max_secrets=0),
            )

    def test_t0_max_one_process(self) -> None:
        """T0 cannot spawn more than 1 process."""
        with pytest.raises(ValidationError, match="T0.*1 process"):
            LeaseSchema(
                name="bad-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                process=ProcessCapability(max_processes=5),
                secrets=SecretCapability(max_secrets=0),
            )

    def test_custom_denied_paths_preserve_ssh_and_env(self) -> None:
        """Custom denied_paths cannot strip ~/.ssh/* or **/.env."""
        fs = FilesystemCapability(denied_paths=["/my/only/path"])
        assert "~/.ssh/*" in fs.denied_paths
        assert "**/.env" in fs.denied_paths

    def test_empty_blocked_commands_rejected(self) -> None:
        """blocked_commands cannot be empty."""
        with pytest.raises(ValidationError, match="blocked_commands"):
            ProcessCapability(blocked_commands=[])

    def test_custom_blocked_commands_still_include_required(self) -> None:
        """Custom blocked_commands cannot strip required safety commands."""
        proc = ProcessCapability(blocked_commands=["my-custom-cmd"])
        for required in REQUIRED_BLOCKED_COMMANDS:
            assert required in proc.blocked_commands, f"{required} must always be in blocked_commands"

    # --- R4 regressions ---

    def test_blocked_commands_preserve_full_baseline(self) -> None:
        """Custom blocked_commands preserves rmdir, kill, pkill too."""
        proc = ProcessCapability(blocked_commands=["custom"])
        for cmd in ("rmdir", "kill", "pkill", "rm", "mkfs", "dd", "shutdown", "reboot"):
            assert cmd in proc.blocked_commands, f"{cmd} must always be in blocked_commands"

    def test_t1_cannot_enable_network(self) -> None:
        """T1 (basic) cannot have outbound network."""
        with pytest.raises(ValidationError, match="T1.*network"):
            LeaseSchema(
                name="bad-t1",
                trust_tier=TrustTier.T1_BASIC,
                network=NetworkCapability(outbound_enabled=True),
            )

    def test_t1_cannot_allow_shell(self) -> None:
        """T1 (basic) cannot allow shell."""
        with pytest.raises(ValidationError, match="T1.*shell"):
            LeaseSchema(
                name="bad-t1",
                trust_tier=TrustTier.T1_BASIC,
                process=ProcessCapability(allow_shell=True),
            )

    def test_t2_cannot_allow_shell(self) -> None:
        """T2 (standard) cannot allow shell."""
        with pytest.raises(ValidationError, match="T2.*shell"):
            LeaseSchema(
                name="bad-t2",
                trust_tier=TrustTier.T2_STANDARD,
                process=ProcessCapability(allow_shell=True),
            )
