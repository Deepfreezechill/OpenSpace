"""Tests for EPIC 2.3 — Network Proxy.

Issues:
- #95: Domain-based allow/deny enforcement
- #96: Concurrent connection tracking and limits
- #97: Port-based filtering
- #98: Outbound enable/disable enforcement + proxy lifecycle
"""

from __future__ import annotations

import asyncio

import pytest

from openspace.sandbox.leases import NetworkCapability
from openspace.sandbox.net_proxy import (
    ConnectionLimitError,
    ConnectionNotFoundError,
    ConnectionTracker,
    DomainDeniedError,
    DomainNotAllowedError,
    NetworkPolicyError,
    NetworkProxy,
    NetworkProxyConfig,
    OutboundDisabledError,
    PortNotAllowedError,
    check_domain_allowed,
    check_domain_blocked,
    check_port_allowed,
)

# ---------------------------------------------------------------------------
# #95 — Domain Matching Tests
# ---------------------------------------------------------------------------


class TestDomainBlocking:
    """Tests for deny-before-allow domain enforcement."""

    def test_exact_domain_blocked(self) -> None:
        with pytest.raises(DomainDeniedError, match="169.254.169.254"):
            check_domain_blocked("169.254.169.254", ["169.254.169.254"])

    def test_wildcard_subdomain_blocked(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("sub.evil.com", ["*.evil.com"])

    def test_case_insensitive_blocking(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("Metadata.Google.Internal", ["metadata.google.internal"])

    def test_unblocked_domain_passes(self) -> None:
        check_domain_blocked("pypi.org", ["169.254.169.254", "metadata.google.internal"])

    def test_cloud_metadata_ipv6_blocked(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("fd00:ec2::254", ["fd00:ec2::254"])

    def test_empty_block_list_allows_all(self) -> None:
        check_domain_blocked("anything.com", [])

    def test_trailing_dot_normalized(self) -> None:
        """DNS trailing dots should be stripped before matching."""
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("evil.com.", ["evil.com"])


class TestDomainAllowing:
    """Tests for domain allowlist enforcement."""

    def test_exact_domain_allowed(self) -> None:
        check_domain_allowed("pypi.org", ["pypi.org"])

    def test_wildcard_subdomain_allowed(self) -> None:
        check_domain_allowed("files.githubusercontent.com", ["*.githubusercontent.com"])

    def test_universal_wildcard_allows_all(self) -> None:
        check_domain_allowed("anything.example.com", ["*"])

    def test_empty_allowlist_denies_all(self) -> None:
        with pytest.raises(DomainNotAllowedError, match="allow list is empty"):
            check_domain_allowed("pypi.org", [])

    def test_domain_not_in_allowlist(self) -> None:
        with pytest.raises(DomainNotAllowedError):
            check_domain_allowed("evil.com", ["pypi.org", "*.github.com"])

    def test_case_insensitive_allowing(self) -> None:
        check_domain_allowed("PyPI.Org", ["pypi.org"])

    def test_multiple_patterns_any_match(self) -> None:
        check_domain_allowed("registry.npmjs.org", ["pypi.org", "registry.npmjs.org"])


# ---------------------------------------------------------------------------
# #97 — Port Filtering Tests
# ---------------------------------------------------------------------------


class TestPortFiltering:
    """Tests for port-based access control."""

    def test_allowed_port_passes(self) -> None:
        check_port_allowed(443, [80, 443])

    def test_disallowed_port_raises(self) -> None:
        with pytest.raises(PortNotAllowedError, match="8080"):
            check_port_allowed(8080, [80, 443])

    def test_empty_allowed_ports_denies_all(self) -> None:
        with pytest.raises(PortNotAllowedError, match="empty"):
            check_port_allowed(80, [])

    def test_port_zero_rejected(self) -> None:
        with pytest.raises(PortNotAllowedError, match="Invalid"):
            check_port_allowed(0, [80, 443])

    def test_negative_port_rejected(self) -> None:
        with pytest.raises(PortNotAllowedError, match="Invalid"):
            check_port_allowed(-1, [80, 443])

    def test_port_above_65535_rejected(self) -> None:
        with pytest.raises(PortNotAllowedError, match="Invalid"):
            check_port_allowed(70000, [80, 443])

    def test_port_1_allowed(self) -> None:
        check_port_allowed(1, [1, 80, 443])

    def test_port_65535_allowed(self) -> None:
        check_port_allowed(65535, [65535])


# ---------------------------------------------------------------------------
# #96 — Connection Tracker Tests
# ---------------------------------------------------------------------------


class TestConnectionTracker:
    """Tests for concurrent connection tracking."""

    @pytest.fixture
    def tracker(self) -> ConnectionTracker:
        return ConnectionTracker(max_connections=3)

    @pytest.mark.asyncio
    async def test_acquire_returns_unique_ids(self, tracker: ConnectionTracker) -> None:
        id1 = await tracker.acquire("a.com", 443)
        id2 = await tracker.acquire("b.com", 443)
        assert id1 != id2
        assert tracker.active_count == 2

    @pytest.mark.asyncio
    async def test_release_decrements_count(self, tracker: ConnectionTracker) -> None:
        conn_id = await tracker.acquire("a.com", 443)
        assert tracker.active_count == 1
        await tracker.release(conn_id)
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_connection_limit_enforced(self, tracker: ConnectionTracker) -> None:
        await tracker.acquire("a.com", 443)
        await tracker.acquire("b.com", 443)
        await tracker.acquire("c.com", 443)
        with pytest.raises(ConnectionLimitError, match="3"):
            await tracker.acquire("d.com", 443)

    @pytest.mark.asyncio
    async def test_release_unknown_raises(self, tracker: ConnectionTracker) -> None:
        with pytest.raises(ConnectionNotFoundError, match="conn-999"):
            await tracker.release("conn-999")

    @pytest.mark.asyncio
    async def test_list_active_returns_snapshot(self, tracker: ConnectionTracker) -> None:
        await tracker.acquire("a.com", 80)
        await tracker.acquire("b.com", 443)
        active = await tracker.list_active()
        assert len(active) == 2
        domains = {c.domain for c in active}
        assert domains == {"a.com", "b.com"}

    @pytest.mark.asyncio
    async def test_release_all_clears_connections(self, tracker: ConnectionTracker) -> None:
        await tracker.acquire("a.com", 80)
        await tracker.acquire("b.com", 443)
        count = await tracker.release_all()
        assert count == 2
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_zero_max_connections_blocks_all(self) -> None:
        tracker = ConnectionTracker(max_connections=0)
        with pytest.raises(ConnectionLimitError):
            await tracker.acquire("a.com", 443)

    def test_negative_max_connections_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            ConnectionTracker(max_connections=-1)

    @pytest.mark.asyncio
    async def test_concurrent_acquire_respects_limit(self) -> None:
        """Concurrent acquires must not exceed the limit."""
        tracker = ConnectionTracker(max_connections=5)
        results = await asyncio.gather(
            *[tracker.acquire(f"host-{i}.com", 443) for i in range(10)],
            return_exceptions=True,
        )
        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, ConnectionLimitError)]
        assert len(successes) == 5
        assert len(failures) == 5


# ---------------------------------------------------------------------------
# #98 — NetworkProxyConfig Tests
# ---------------------------------------------------------------------------


class TestNetworkProxyConfig:
    """Tests for config construction from NetworkCapability."""

    def test_from_capability_outbound_enabled(self) -> None:
        cap = NetworkCapability(
            outbound_enabled=True,
            allowed_domains=["pypi.org"],
            allowed_ports=[443],
            max_connections=5,
        )
        config = NetworkProxyConfig.from_capability(cap)
        assert config.outbound_enabled is True
        assert config.allowed_domains == ("pypi.org",)
        assert 443 in config.allowed_ports
        assert config.max_connections == 5

    def test_from_capability_outbound_disabled_clears_domains(self) -> None:
        """EPIC 2.1 R5 deferred fix: outbound=False must clear allowed_domains."""
        cap = NetworkCapability(
            outbound_enabled=False,
            allowed_domains=["should-be-cleared.com"],
            max_connections=0,
        )
        config = NetworkProxyConfig.from_capability(cap)
        assert config.outbound_enabled is False
        assert config.allowed_domains == ()

    def test_blocked_domains_preserved(self) -> None:
        cap = NetworkCapability(outbound_enabled=True, allowed_domains=["*"])
        config = NetworkProxyConfig.from_capability(cap)
        assert "169.254.169.254" in config.blocked_domains


# ---------------------------------------------------------------------------
# #98 — NetworkProxy Integration Tests
# ---------------------------------------------------------------------------


class TestNetworkProxy:
    """Integration tests for the full NetworkProxy lifecycle."""

    @pytest.fixture
    def t2_proxy(self) -> NetworkProxy:
        """T2-equivalent proxy: limited outbound."""
        config = NetworkProxyConfig(
            outbound_enabled=True,
            allowed_domains=("*.githubusercontent.com", "pypi.org", "registry.npmjs.org"),
            blocked_domains=("169.254.169.254", "metadata.google.internal"),
            allowed_ports=(80, 443),
            max_connections=5,
        )
        return NetworkProxy(config)

    @pytest.fixture
    def t0_proxy(self) -> NetworkProxy:
        """T0-equivalent proxy: no outbound."""
        config = NetworkProxyConfig(
            outbound_enabled=False,
            allowed_domains=(),
            blocked_domains=("169.254.169.254",),
            allowed_ports=(),
            max_connections=0,
        )
        return NetworkProxy(config)

    @pytest.fixture
    def t3_proxy(self) -> NetworkProxy:
        """T3-equivalent proxy: broad outbound."""
        config = NetworkProxyConfig(
            outbound_enabled=True,
            allowed_domains=("*",),
            blocked_domains=("169.254.169.254", "metadata.google.internal"),
            allowed_ports=(80, 443, 8080, 8443),
            max_connections=20,
        )
        return NetworkProxy(config)

    # --- Outbound gating ---

    def test_t0_blocks_all_outbound(self, t0_proxy: NetworkProxy) -> None:
        with pytest.raises(OutboundDisabledError):
            t0_proxy.check_request("pypi.org", 443)

    def test_t2_allows_permitted_domain(self, t2_proxy: NetworkProxy) -> None:
        t2_proxy.check_request("pypi.org", 443)

    def test_t2_blocks_unpermitted_domain(self, t2_proxy: NetworkProxy) -> None:
        with pytest.raises(DomainNotAllowedError):
            t2_proxy.check_request("evil.com", 443)

    # --- Deny-before-allow ---

    def test_blocked_domain_rejected_even_if_wildcard_allowed(self, t3_proxy: NetworkProxy) -> None:
        """Cloud metadata must be blocked even with allowed_domains=['*']."""
        with pytest.raises(DomainDeniedError):
            t3_proxy.check_request("169.254.169.254", 80)

    # --- Port filtering ---

    def test_t2_blocks_non_standard_port(self, t2_proxy: NetworkProxy) -> None:
        with pytest.raises(PortNotAllowedError):
            t2_proxy.check_request("pypi.org", 8080)

    def test_t3_allows_custom_port(self, t3_proxy: NetworkProxy) -> None:
        t3_proxy.check_request("example.com", 8080)

    # --- Connection lifecycle ---

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, t2_proxy: NetworkProxy) -> None:
        conn_id = await t2_proxy.connect("pypi.org", 443)
        assert t2_proxy.active_connections == 1
        await t2_proxy.disconnect(conn_id)
        assert t2_proxy.active_connections == 0

    @pytest.mark.asyncio
    async def test_connect_respects_connection_limit(self, t2_proxy: NetworkProxy) -> None:
        ids = []
        for i in range(5):
            ids.append(await t2_proxy.connect("pypi.org", 443))
        with pytest.raises(ConnectionLimitError):
            await t2_proxy.connect("pypi.org", 443)
        # Release one and try again
        await t2_proxy.disconnect(ids[0])
        new_id = await t2_proxy.connect("pypi.org", 443)
        assert new_id not in ids

    @pytest.mark.asyncio
    async def test_connect_validates_policy_before_slot(self, t2_proxy: NetworkProxy) -> None:
        """Policy violations should not consume a connection slot."""
        with pytest.raises(DomainNotAllowedError):
            await t2_proxy.connect("evil.com", 443)
        assert t2_proxy.active_connections == 0

    # --- Shutdown ---

    @pytest.mark.asyncio
    async def test_shutdown_releases_all(self, t2_proxy: NetworkProxy) -> None:
        await t2_proxy.connect("pypi.org", 443)
        await t2_proxy.connect("pypi.org", 443)
        count = await t2_proxy.shutdown()
        assert count == 2
        assert t2_proxy.active_connections == 0

    @pytest.mark.asyncio
    async def test_shutdown_rejects_new_requests(self, t2_proxy: NetworkProxy) -> None:
        await t2_proxy.shutdown()
        with pytest.raises(NetworkPolicyError, match="shut down"):
            t2_proxy.check_request("pypi.org", 443)

    @pytest.mark.asyncio
    async def test_shutdown_rejects_new_connections(self, t2_proxy: NetworkProxy) -> None:
        await t2_proxy.shutdown()
        with pytest.raises(NetworkPolicyError, match="shut down"):
            await t2_proxy.connect("pypi.org", 443)

    # --- List connections ---

    @pytest.mark.asyncio
    async def test_list_connections(self, t2_proxy: NetworkProxy) -> None:
        await t2_proxy.connect("pypi.org", 443)
        await t2_proxy.connect("registry.npmjs.org", 443)
        conns = await t2_proxy.list_connections()
        assert len(conns) == 2
        domains = {c.domain for c in conns}
        assert "pypi.org" in domains
        assert "registry.npmjs.org" in domains


# ---------------------------------------------------------------------------
# EPIC 2.1 R5 Deferred Fix — T0/T1 Validation
# ---------------------------------------------------------------------------


class TestT0T1ValidationFix:
    """Verify that T0/T1 with outbound_enabled=False and non-empty
    allowed_domains is properly handled by NetworkProxyConfig."""

    def test_t0_domains_cleared_in_config(self) -> None:
        """Even if a T0 capability somehow has allowed_domains, the proxy clears them."""
        cap = NetworkCapability(
            outbound_enabled=False,
            allowed_domains=["sneaky.com"],
            max_connections=0,
        )
        config = NetworkProxyConfig.from_capability(cap)
        assert config.allowed_domains == ()
        proxy = NetworkProxy(config)
        with pytest.raises(OutboundDisabledError):
            proxy.check_request("sneaky.com", 443)

    def test_t1_domains_cleared_in_config(self) -> None:
        cap = NetworkCapability(
            outbound_enabled=False,
            allowed_domains=["also-sneaky.com"],
            max_connections=0,
        )
        config = NetworkProxyConfig.from_capability(cap)
        assert config.allowed_domains == ()


# ---------------------------------------------------------------------------
# R1 Security Regression Tests
# ---------------------------------------------------------------------------


class TestSecurityRegressions:
    """Regression tests for R1 review findings."""

    # --- DNS rebinding bypass ---

    def test_nip_io_blocked(self) -> None:
        """169.254.169.254.nip.io must be blocked (DNS rebinding)."""
        with pytest.raises(DomainDeniedError, match="nip.io"):
            check_domain_blocked("169.254.169.254.nip.io", ["169.254.169.254"])

    def test_sslip_io_blocked(self) -> None:
        with pytest.raises(DomainDeniedError, match="sslip.io"):
            check_domain_blocked("169-254-169-254.sslip.io", ["169.254.169.254"])

    def test_xip_io_blocked(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("anything.xip.io", ["169.254.169.254"])

    # --- IPv6 metadata alias bypass ---

    def test_ipv6_mapped_metadata_blocked(self) -> None:
        """::ffff:169.254.169.254 must be blocked (IPv4-mapped IPv6)."""
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("::ffff:169.254.169.254", ["169.254.169.254"])

    def test_ipv6_hex_metadata_blocked(self) -> None:
        """::ffff:a9fe:a9fe (hex form of 169.254.169.254) must be blocked."""
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("::ffff:a9fe:a9fe", ["169.254.169.254"])

    def test_bracketed_ipv6_metadata_blocked(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("[::ffff:169.254.169.254]", ["169.254.169.254"])

    def test_ipv6_mapped_gcp_blocked(self) -> None:
        """IPv4-mapped IPv6 for 100.100.100.200 (Alibaba metadata)."""
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("::ffff:100.100.100.200", ["100.100.100.200"])

    # --- Shutdown race ---

    @pytest.mark.asyncio
    async def test_shutdown_race_no_post_shutdown_connections(self) -> None:
        """No connections must be acquired after shutdown() completes."""
        config = NetworkProxyConfig(
            outbound_enabled=True,
            allowed_domains=("*",),
            blocked_domains=(),
            allowed_ports=(443,),
            max_connections=100,
        )
        proxy = NetworkProxy(config)

        # Pre-populate some connections
        for i in range(5):
            await proxy.connect(f"host-{i}.com", 443)

        # Shutdown and attempt concurrent connects
        shutdown_task = asyncio.create_task(proxy.shutdown())
        connect_tasks = [asyncio.create_task(proxy.connect(f"late-{i}.com", 443)) for i in range(10)]
        await shutdown_task

        results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        # All post-shutdown connects must fail
        for r in results:
            assert isinstance(r, (NetworkPolicyError, ConnectionLimitError)), f"Post-shutdown connect succeeded: {r}"
        assert proxy.active_connections == 0

    # --- T0/T1 LeaseSchema validation ---

    def test_t0_schema_rejects_nonempty_allowed_domains(self) -> None:
        """LeaseSchema must reject T0 with non-empty allowed_domains."""
        from pydantic import ValidationError

        from openspace.sandbox.leases import LeaseSchema, TrustTier

        with pytest.raises(ValidationError, match="allowed_domains"):
            LeaseSchema(
                name="test-t0",
                trust_tier=TrustTier.T0_UNTRUSTED,
                network=NetworkCapability(
                    outbound_enabled=False,
                    allowed_domains=["sneaky.com"],
                    max_connections=0,
                ),
            )

    def test_t1_schema_rejects_nonempty_allowed_domains(self) -> None:
        """LeaseSchema must reject T1 with non-empty allowed_domains."""
        from pydantic import ValidationError

        from openspace.sandbox.leases import LeaseSchema, TrustTier

        with pytest.raises(ValidationError, match="allowed_domains"):
            LeaseSchema(
                name="test-t1",
                trust_tier=TrustTier.T1_BASIC,
                network=NetworkCapability(
                    outbound_enabled=False,
                    allowed_domains=["also-sneaky.com"],
                    max_connections=0,
                ),
            )


class TestSecurityRegressionsR2:
    """Regression tests for R2 /8eyes findings (apex rebinding + loopback SSRF)."""

    # --- Apex DNS rebinding ---

    def test_bare_nip_io_blocked(self) -> None:
        """Bare 'nip.io' (no subdomain) must be blocked."""
        with pytest.raises(DomainDeniedError, match="nip.io"):
            check_domain_blocked("nip.io", [])

    def test_bare_localtest_me_blocked(self) -> None:
        """Bare 'localtest.me' resolves to 127.0.0.1 — must be blocked."""
        with pytest.raises(DomainDeniedError, match="localtest.me"):
            check_domain_blocked("localtest.me", [])

    def test_bare_sslip_io_blocked(self) -> None:
        with pytest.raises(DomainDeniedError, match="sslip.io"):
            check_domain_blocked("sslip.io", [])

    def test_bare_xip_io_blocked(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("xip.io", [])

    def test_bare_traefik_me_blocked(self) -> None:
        with pytest.raises(DomainDeniedError):
            check_domain_blocked("traefik.me", [])

    # --- Loopback / link-local SSRF ---

    def test_ipv4_loopback_blocked(self) -> None:
        """127.0.0.1 must be blocked even with allowed_domains=('*',)."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("127.0.0.1", [])

    def test_ipv4_loopback_any_octet_blocked(self) -> None:
        """Any 127.x.x.x address must be blocked."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("127.0.0.2", [])

    def test_ipv6_loopback_blocked(self) -> None:
        """::1 must be blocked."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("::1", [])

    def test_bracketed_ipv6_loopback_blocked(self) -> None:
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("[::1]", [])

    def test_ipv4_mapped_loopback_blocked(self) -> None:
        """::ffff:127.0.0.1 must be blocked as loopback."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("::ffff:127.0.0.1", [])

    def test_zero_address_blocked(self) -> None:
        """0.0.0.0 must be blocked."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("0.0.0.0", [])

    def test_ipv6_link_local_blocked(self) -> None:
        """fe80::1 (link-local) must be blocked."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("fe80::1", [])

    def test_ipv6_ula_blocked(self) -> None:
        """fd00::1 (unique local address) must be blocked."""
        with pytest.raises(DomainDeniedError, match="loopback"):
            check_domain_blocked("fd00::1", [])

    def test_legitimate_ip_not_blocked(self) -> None:
        """Normal public IPs must NOT be blocked by the IP check."""
        # Should not raise — no domain-level block either
        check_domain_blocked("8.8.8.8", [])

    def test_legitimate_domain_not_blocked(self) -> None:
        """Normal domains must NOT be blocked."""
        check_domain_blocked("github.com", [])
