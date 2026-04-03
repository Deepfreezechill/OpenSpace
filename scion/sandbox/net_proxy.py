"""Network proxy — policy-enforced outbound network access control.

EPIC 2.3 — Network Proxy

Issues:
- #95: Domain-based allow/deny enforcement with glob matching
- #96: Concurrent connection tracking and limits
- #97: Port-based filtering and validation
- #98: Outbound enable/disable enforcement + proxy lifecycle
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import time
from dataclasses import dataclass, field
from typing import Optional

from scion.sandbox.leases import NetworkCapability

# ---------------------------------------------------------------------------
# Known DNS rebinding services that resolve to arbitrary IPs.
# These must be blocked to prevent metadata endpoint access via aliases
# like 169.254.169.254.nip.io → 169.254.169.254.
# ---------------------------------------------------------------------------

_DNS_REBINDING_PATTERNS: tuple[str, ...] = (
    "*.nip.io",
    "nip.io",
    "*.sslip.io",
    "sslip.io",
    "*.xip.io",
    "xip.io",
    "*.traefik.me",
    "traefik.me",
    "*.localtest.me",
    "localtest.me",
)

# IPv4-mapped IPv6 equivalents of common metadata endpoints
_IPV6_METADATA_ALIASES: tuple[str, ...] = (
    "::ffff:169.254.169.254",
    "::ffff:a9fe:a9fe",
    "::ffff:100.100.100.200",
    "::ffff:6464:64c8",
    "[::ffff:169.254.169.254]",
    "[::ffff:a9fe:a9fe]",
)

# IP networks that must be blocked to prevent SSRF to local services.
# Loopback, link-local, and unspecified addresses allow reaching localhost,
# cloud-internal endpoints, and adjacent machines on the same network segment.
_BLOCKED_IP_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.IPv4Network("127.0.0.0/8"),  # IPv4 loopback
    ipaddress.IPv6Network("::1/128"),  # IPv6 loopback
    ipaddress.IPv4Network("0.0.0.0/8"),  # "this host" — often aliases localhost
    ipaddress.IPv6Network("::/128"),  # IPv6 unspecified
    ipaddress.IPv4Network("169.254.0.0/16"),  # IPv4 link-local (already caught by metadata)
    ipaddress.IPv6Network("fe80::/10"),  # IPv6 link-local
    ipaddress.IPv6Network("fc00::/7"),  # IPv6 unique-local (ULA, private)
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NetworkPolicyError(Exception):
    """Base for all network proxy policy violations."""


class OutboundDisabledError(NetworkPolicyError):
    """Raised when outbound networking is disabled for this lease."""


class DomainDeniedError(NetworkPolicyError):
    """Raised when the target domain is on the block list."""


class DomainNotAllowedError(NetworkPolicyError):
    """Raised when the target domain is not on the allow list."""


class PortNotAllowedError(NetworkPolicyError):
    """Raised when the target port is not in the allowed set."""


class ConnectionLimitError(NetworkPolicyError):
    """Raised when the concurrent connection limit is reached."""


class ConnectionNotFoundError(NetworkPolicyError):
    """Raised when attempting to release a non-existent connection."""


# ---------------------------------------------------------------------------
# #95 — Domain-Based Allow/Deny Enforcement
# ---------------------------------------------------------------------------


def _is_blocked_ip(host: str) -> bool:
    """Return True if *host* is an IP in a blocked network (loopback, link-local, etc.)."""
    cleaned = host.strip("[]")
    try:
        addr = ipaddress.ip_address(cleaned)
    except ValueError:
        return False

    # Collapse IPv4-mapped IPv6 to plain IPv4 for network checks
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped

    return any(addr in net for net in _BLOCKED_IP_NETWORKS)


def _normalize_ip(host: str) -> Optional[str]:
    """Normalize an IP address to its canonical IPv4 form if possible.

    Handles IPv4-mapped IPv6 (``::ffff:A.B.C.D``), bracket-wrapped
    literals (``[::1]``), and plain IPv4/IPv6 strings.
    Returns the canonical string form, or None if *host* is not an IP.
    """
    cleaned = host.strip("[]")
    try:
        addr = ipaddress.ip_address(cleaned)
    except ValueError:
        return None

    # Collapse IPv4-mapped IPv6 to plain IPv4
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        return str(addr.ipv4_mapped)
    return str(addr)


def _matches_domain(domain: str, pattern: str) -> bool:
    """Check if *domain* matches a domain *pattern*.

    Supports:
    - Exact match: ``example.com`` matches ``example.com``
    - Wildcard subdomain: ``*.example.com`` matches ``sub.example.com``
    - IP addresses: normalized before comparison (IPv4-mapped IPv6 collapsed)
    - Universal wildcard: ``*`` matches everything

    Case-insensitive matching (domains are case-insensitive per RFC 4343).
    """
    domain_lower = domain.lower().strip(".")
    pattern_lower = pattern.lower().strip(".")

    # Normalize IPs: ::ffff:169.254.169.254 → 169.254.169.254
    domain_ip = _normalize_ip(domain_lower)
    pattern_ip = _normalize_ip(pattern_lower)

    if domain_ip and pattern_ip:
        return domain_ip == pattern_ip
    if domain_ip:
        return fnmatch.fnmatch(domain_ip, pattern_lower)
    if pattern_ip:
        return fnmatch.fnmatch(domain_lower, pattern_ip)

    return fnmatch.fnmatch(domain_lower, pattern_lower)


def check_domain_blocked(domain: str, blocked_domains: list[str]) -> None:
    """Raise ``DomainDeniedError`` if *domain* matches any blocked pattern.

    Block list is checked **first** (deny-before-allow), and includes
    cloud metadata endpoints by default.  Also checks against known DNS
    rebinding services (nip.io, sslip.io, etc.), IPv6 metadata aliases,
    and loopback/link-local/ULA IP ranges.
    """
    # Block loopback, link-local, ULA IPs before pattern matching
    if _is_blocked_ip(domain):
        raise DomainDeniedError(f"Domain '{domain}' is a blocked IP address (loopback/link-local/ULA)")

    # Build extended block list: explicit + rebinding + IPv6 aliases
    extended = list(blocked_domains) + list(_DNS_REBINDING_PATTERNS) + list(_IPV6_METADATA_ALIASES)

    for pattern in extended:
        if _matches_domain(domain, pattern):
            raise DomainDeniedError(f"Domain '{domain}' is blocked by pattern '{pattern}'")


def check_domain_allowed(domain: str, allowed_domains: list[str]) -> None:
    """Raise ``DomainNotAllowedError`` if *domain* is not in the allow list.

    An empty allow list means **no domains are permitted** (deny-by-default).
    A list containing ``"*"`` permits all domains.
    """
    if not allowed_domains:
        raise DomainNotAllowedError(f"Domain '{domain}' not allowed: allow list is empty")

    for pattern in allowed_domains:
        if _matches_domain(domain, pattern):
            return  # Allowed

    raise DomainNotAllowedError(f"Domain '{domain}' does not match any allowed pattern")


# ---------------------------------------------------------------------------
# #97 — Port-Based Filtering
# ---------------------------------------------------------------------------


def check_port_allowed(port: int, allowed_ports: list[int]) -> None:
    """Raise ``PortNotAllowedError`` if *port* is not in the allowed set.

    An empty allowed_ports list means **no ports are permitted**.
    Port 0 is never allowed (reserved).
    """
    if port <= 0 or port > 65535:
        raise PortNotAllowedError(f"Invalid port number: {port}")
    if not allowed_ports:
        raise PortNotAllowedError(f"Port {port} not allowed: allowed ports list is empty")
    if port not in allowed_ports:
        raise PortNotAllowedError(f"Port {port} not in allowed ports: {allowed_ports}")


# ---------------------------------------------------------------------------
# #96 — Connection Tracking
# ---------------------------------------------------------------------------


@dataclass
class ConnectionRecord:
    """Metadata for an active outbound connection."""

    connection_id: str
    domain: str
    port: int
    opened_at: float = field(default_factory=time.monotonic)


class ConnectionTracker:
    """Thread-safe concurrent connection tracker with configurable limit.

    Uses ``asyncio.Lock`` for coroutine safety. All public methods are async.
    """

    def __init__(self, max_connections: int) -> None:
        if max_connections < 0:
            raise ValueError(f"max_connections must be >= 0, got {max_connections}")
        self._max = max_connections
        self._active: dict[str, ConnectionRecord] = {}
        self._lock = asyncio.Lock()
        self._counter = 0

    @property
    def max_connections(self) -> int:
        return self._max

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def acquire(self, domain: str, port: int) -> str:
        """Register a new connection. Returns a unique connection ID.

        Raises ``ConnectionLimitError`` if the limit would be exceeded.
        """
        async with self._lock:
            if len(self._active) >= self._max:
                raise ConnectionLimitError(f"Connection limit reached ({self._max}). Active: {len(self._active)}")
            self._counter += 1
            conn_id = f"conn-{self._counter}"
            self._active[conn_id] = ConnectionRecord(
                connection_id=conn_id,
                domain=domain,
                port=port,
            )
            return conn_id

    async def release(self, connection_id: str) -> None:
        """Release a connection by ID.

        Raises ``ConnectionNotFoundError`` if the ID is not active.
        """
        async with self._lock:
            if connection_id not in self._active:
                raise ConnectionNotFoundError(f"Connection '{connection_id}' not found in active set")
            del self._active[connection_id]

    async def list_active(self) -> list[ConnectionRecord]:
        """Return a snapshot of all active connections."""
        async with self._lock:
            return list(self._active.values())

    async def release_all(self) -> int:
        """Release all active connections. Returns the count released."""
        async with self._lock:
            count = len(self._active)
            self._active.clear()
            return count


# ---------------------------------------------------------------------------
# #98 — Network Proxy Configuration + Lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkProxyConfig:
    """Immutable network proxy configuration derived from a NetworkCapability.

    Constructed from a :class:`NetworkCapability` lease grant. The proxy
    enforces all policy constraints: outbound enable/disable, domain
    allow/deny, port filtering, and connection limits.
    """

    outbound_enabled: bool
    allowed_domains: tuple[str, ...]
    blocked_domains: tuple[str, ...]
    allowed_ports: tuple[int, ...]
    max_connections: int

    @classmethod
    def from_capability(cls, cap: NetworkCapability) -> "NetworkProxyConfig":
        """Create a proxy config from a NetworkCapability.

        If outbound is disabled, allowed_domains is forced empty to prevent
        the deferred EPIC 2.1 R5 finding (non-empty allowed_domains with
        outbound_enabled=False leaking into policy).
        """
        allowed = cap.allowed_domains if cap.outbound_enabled else []
        return cls(
            outbound_enabled=cap.outbound_enabled,
            allowed_domains=tuple(allowed),
            blocked_domains=tuple(cap.blocked_domains),
            allowed_ports=tuple(cap.allowed_ports),
            max_connections=cap.max_connections,
        )


class NetworkProxy:
    """Policy-enforced network proxy for sandboxed skill execution.

    Combines outbound gating, domain allow/deny, port filtering, and
    concurrent connection tracking into a single entry point.

    Usage::

        proxy = NetworkProxy(config)
        conn_id = await proxy.connect("pypi.org", 443)
        # ... use connection ...
        await proxy.disconnect(conn_id)
        await proxy.shutdown()
    """

    def __init__(self, config: NetworkProxyConfig) -> None:
        self._config = config
        self._tracker = ConnectionTracker(config.max_connections)
        self._shutdown = False
        self._lifecycle_lock = asyncio.Lock()

    @property
    def config(self) -> NetworkProxyConfig:
        return self._config

    @property
    def active_connections(self) -> int:
        return self._tracker.active_count

    def check_request(self, domain: str, port: int) -> None:
        """Validate an outbound request against the full policy stack.

        Checks are applied in order (fail-fast):
        1. Outbound enabled?
        2. Domain not blocked? (deny-before-allow)
        3. Domain allowed?
        4. Port allowed?

        Does NOT acquire a connection slot — use :meth:`connect` for that.
        """
        if self._shutdown:
            raise NetworkPolicyError("Proxy has been shut down")

        if not self._config.outbound_enabled:
            raise OutboundDisabledError("Outbound networking is disabled for this sandbox")

        check_domain_blocked(domain, list(self._config.blocked_domains))
        check_domain_allowed(domain, list(self._config.allowed_domains))
        check_port_allowed(port, list(self._config.allowed_ports))

    async def connect(self, domain: str, port: int) -> str:
        """Validate policy and acquire a connection slot atomically.

        The policy check and slot acquisition are performed under a shared
        lifecycle lock to prevent races with :meth:`shutdown`.

        Returns a connection ID for later release via :meth:`disconnect`.
        Raises on policy violation, connection limit, or post-shutdown.
        """
        async with self._lifecycle_lock:
            self.check_request(domain, port)
            return await self._tracker.acquire(domain, port)

    async def disconnect(self, connection_id: str) -> None:
        """Release a connection slot."""
        await self._tracker.release(connection_id)

    async def list_connections(self) -> list[ConnectionRecord]:
        """Return a snapshot of all active connections."""
        return await self._tracker.list_active()

    async def shutdown(self) -> int:
        """Shut down the proxy, releasing all connections.

        Acquires the lifecycle lock to prevent races with in-flight
        :meth:`connect` calls.  After shutdown, all further requests
        are rejected.

        Returns the number of connections released.
        """
        async with self._lifecycle_lock:
            self._shutdown = True
            return await self._tracker.release_all()
