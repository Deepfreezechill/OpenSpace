"""Advanced MCP authentication and authorization — EPIC 2.5.

Provides:
- HMAC-signed token creation and validation with claims (scopes, trust tier,
  subject, expiry, audience) — no external JWT dependency required.
- Per-tool authorization enforcing scope and trust-tier requirements.
- Token lifecycle: creation, validation, expiry, revocation.
- AuthContext: request-scoped identity that flows through the MCP pipeline.

Design decisions:
- Fail-closed at every layer: invalid/missing → deny.
- HMAC-SHA256 signatures (timing-safe via hmac.compare_digest).
- Deny-before-allow: tool authorization checks blocklist before allowlist.
- Trust tier ceiling: a token's tier caps the tools it can invoke.
- Audience binding: tokens are bound to a specific service to prevent
  cross-service replay attacks.
- No external crypto dependencies: uses stdlib hashlib + hmac + secrets.

Security requirements:
- **Each service MUST use a unique signing secret.**  HMAC is symmetric —
  any party that knows the secret can mint arbitrary tokens.  Sharing a
  secret between services allows cross-service token forgery.  If
  multi-service deployment is needed with a shared trust root, migrate
  to asymmetric signing (RSA/Ed25519) with per-service key pairs.

Issues:
- #51: AuthPort concrete implementation
- #104: Token scoping and claims model
- #105: Per-tool authorization
- #106: Trust-tier gating
- #107: Token lifecycle (revoke, expire, rotate)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

from openspace.sandbox.leases import TrustTier

# ---------------------------------------------------------------------------
# #104 — Token Claims Model
# ---------------------------------------------------------------------------


class TokenScope(str, Enum):
    """Scopes that can be granted to a token."""

    TOOL_EXECUTE = "tool:execute"
    TOOL_SEARCH = "tool:search"
    TOOL_ADMIN = "tool:admin"
    SECRET_READ = "secret:read"
    SECRET_WRITE = "secret:write"
    LEASE_ACQUIRE = "lease:acquire"
    LEASE_ADMIN = "lease:admin"


# Default scopes per trust tier (deny-before-allow escalation)
TIER_DEFAULT_SCOPES: dict[TrustTier, frozenset[TokenScope]] = {
    TrustTier.T0_UNTRUSTED: frozenset(),
    TrustTier.T1_BASIC: frozenset({TokenScope.TOOL_SEARCH}),
    TrustTier.T2_STANDARD: frozenset(
        {
            TokenScope.TOOL_SEARCH,
            TokenScope.TOOL_EXECUTE,
            TokenScope.LEASE_ACQUIRE,
        }
    ),
    TrustTier.T3_ELEVATED: frozenset(
        {
            TokenScope.TOOL_SEARCH,
            TokenScope.TOOL_EXECUTE,
            TokenScope.LEASE_ACQUIRE,
            TokenScope.SECRET_READ,
        }
    ),
    TrustTier.T4_FULL: frozenset(
        {
            TokenScope.TOOL_SEARCH,
            TokenScope.TOOL_EXECUTE,
            TokenScope.TOOL_ADMIN,
            TokenScope.LEASE_ACQUIRE,
            TokenScope.LEASE_ADMIN,
            TokenScope.SECRET_READ,
            TokenScope.SECRET_WRITE,
        }
    ),
}


@dataclass(frozen=True)
class AuthClaims:
    """Immutable claims extracted from a validated token.

    These flow through the request pipeline as the caller's identity.
    """

    subject: str
    trust_tier: TrustTier
    scopes: frozenset[TokenScope]
    issued_at: float
    expires_at: float
    token_id: str
    audience: str = ""  # service binding — prevents cross-service replay

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def has_scope(self, scope: TokenScope) -> bool:
        return scope in self.scopes

    def has_any_scope(self, *scopes: TokenScope) -> bool:
        return bool(self.scopes & frozenset(scopes))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Base for all authentication/authorization errors."""


class TokenInvalidError(AuthError):
    """Token is malformed, tampered, or has an invalid signature."""


class TokenExpiredError(AuthError):
    """Token has passed its expiry time."""


class TokenRevokedError(AuthError):
    """Token has been explicitly revoked."""


class InsufficientScopeError(AuthError):
    """Caller lacks the required scope for this operation."""


class InsufficientTierError(AuthError):
    """Caller's trust tier is below the required level."""


class ToolNotAuthorizedError(AuthError):
    """Caller is not authorized to invoke this specific tool."""


# ---------------------------------------------------------------------------
# #104 — Token Creation and Validation (HMAC-SHA256)
# ---------------------------------------------------------------------------

# Token format: base64(json_payload).base64(hmac_signature)
_TOKEN_SEPARATOR = "."
_MIN_SECRET_LENGTH = 32
_MAX_TTL_SECONDS = 86_400  # 24 hours — prevents long-lived token abuse


def _compute_signature(payload_b64: str, secret: bytes) -> str:
    """Compute HMAC-SHA256 signature over the base64-encoded payload."""
    sig = hmac.new(secret, payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")


def create_token(
    *,
    secret: str,
    subject: str,
    trust_tier: TrustTier = TrustTier.T1_BASIC,
    scopes: frozenset[TokenScope] | None = None,
    ttl_seconds: int = 3600,
    token_id: str | None = None,
    audience: str = "",
) -> str:
    """Create an HMAC-signed token with claims.

    Args:
        secret: Server signing secret (min 32 chars).
        subject: Identity of the token holder (e.g., service name, user ID).
        trust_tier: Maximum trust tier this token can operate at.
        scopes: Explicit scopes; defaults to tier's default scopes.
        ttl_seconds: Token lifetime in seconds (default 1 hour).
        token_id: Optional unique token ID; auto-generated if not provided.
        audience: Service identifier for token binding. When set, the
            validator must pass the same audience to reject cross-service
            replay attacks.

    Returns:
        Signed token string (base64-payload.base64-signature).

    Raises:
        ValueError: If secret is too short or ttl_seconds is invalid.
    """
    if len(secret) < _MIN_SECRET_LENGTH:
        raise ValueError(f"Signing secret must be at least {_MIN_SECRET_LENGTH} characters")
    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be positive")
    if ttl_seconds > _MAX_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must not exceed {_MAX_TTL_SECONDS} ({_MAX_TTL_SECONDS // 3600}h)")

    now = time.time()
    # Validate scopes against tier ceiling — prevent privilege inversion
    tier_allowed = TIER_DEFAULT_SCOPES.get(trust_tier, frozenset())
    if scopes is not None:
        excess = scopes - tier_allowed
        if excess:
            excess_names = ", ".join(s.value for s in sorted(excess, key=lambda s: s.value))
            raise ValueError(f"Scopes [{excess_names}] exceed tier {trust_tier.value} ceiling")
        effective_scopes = scopes
    else:
        effective_scopes = tier_allowed

    payload = {
        "sub": subject,
        "tier": trust_tier.value,
        "scopes": sorted(s.value for s in effective_scopes),
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": token_id or secrets.token_urlsafe(16),
        "aud": audience,
    }

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).rstrip(b"=").decode("ascii")

    signature = _compute_signature(payload_b64, secret.encode("utf-8"))
    return f"{payload_b64}{_TOKEN_SEPARATOR}{signature}"


def validate_token(token: str, *, secret: str, expected_audience: str = "") -> AuthClaims:
    """Validate an HMAC-signed token and extract claims.

    Args:
        token: The token string to validate.
        secret: The server signing secret.
        expected_audience: If non-empty, the token's ``aud`` claim must
            match exactly.  This prevents cross-service token replay.

    Returns:
        AuthClaims with validated identity information.

    Raises:
        TokenInvalidError: If token is malformed, signature is invalid,
            or audience does not match.
        TokenExpiredError: If token has expired.
    """
    if _TOKEN_SEPARATOR not in token:
        raise TokenInvalidError("Malformed token: missing separator")

    parts = token.split(_TOKEN_SEPARATOR)
    if len(parts) != 2:
        raise TokenInvalidError("Malformed token: expected 2 parts")

    payload_b64, provided_sig = parts

    # Verify signature (timing-safe)
    expected_sig = _compute_signature(payload_b64, secret.encode("utf-8"))
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise TokenInvalidError("Invalid token signature")

    # Decode payload
    try:
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
    except Exception as exc:
        raise TokenInvalidError("Failed to decode token payload") from exc

    # Extract and validate fields
    try:
        subject = str(payload["sub"])
        tier = TrustTier(payload["tier"])
        scope_values = payload.get("scopes", [])
        scopes = frozenset(TokenScope(s) for s in scope_values)
        issued_at = float(payload["iat"])
        expires_at = float(payload["exp"])
        token_id = str(payload["jti"])
        audience = str(payload.get("aud", ""))
    except (KeyError, ValueError) as exc:
        raise TokenInvalidError("Invalid token claims") from exc

    # Reject non-finite timestamps (NaN/Infinity bypass expiry checks)
    if not math.isfinite(issued_at) or not math.isfinite(expires_at):
        raise TokenInvalidError("Token timestamps must be finite")
    if expires_at <= issued_at:
        raise TokenInvalidError("Token expiry must be after issuance")

    # Audience enforcement — prevents cross-service token replay
    if expected_audience and audience != expected_audience:
        raise TokenInvalidError("Token audience mismatch")

    # Check expiry
    if time.time() > expires_at:
        raise TokenExpiredError("Token has expired")

    return AuthClaims(
        subject=subject,
        trust_tier=tier,
        scopes=scopes,
        issued_at=issued_at,
        expires_at=expires_at,
        token_id=token_id,
        audience=audience,
    )


# ---------------------------------------------------------------------------
# #105 — Per-Tool Authorization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolPolicy:
    """Authorization policy for a single MCP tool.

    Defines what scopes and minimum trust tier are required to invoke a tool.
    """

    tool_name: str
    required_scopes: frozenset[TokenScope] = frozenset({TokenScope.TOOL_EXECUTE})
    min_trust_tier: TrustTier = TrustTier.T1_BASIC
    blocked_subjects: frozenset[str] = frozenset()
    allowed_subjects: frozenset[str] = frozenset()


# Canonical tier ordering — single source of truth
_TIER_ORDER: list[TrustTier] = [
    TrustTier.T0_UNTRUSTED,
    TrustTier.T1_BASIC,
    TrustTier.T2_STANDARD,
    TrustTier.T3_ELEVATED,
    TrustTier.T4_FULL,
]


# Default policies for built-in MCP tools
DEFAULT_TOOL_POLICIES: dict[str, ToolPolicy] = {
    "execute_task": ToolPolicy(
        tool_name="execute_task",
        required_scopes=frozenset({TokenScope.TOOL_EXECUTE}),
        min_trust_tier=TrustTier.T2_STANDARD,
    ),
    "search_skills": ToolPolicy(
        tool_name="search_skills",
        required_scopes=frozenset({TokenScope.TOOL_SEARCH}),
        min_trust_tier=TrustTier.T1_BASIC,
    ),
    "fix_skill": ToolPolicy(
        tool_name="fix_skill",
        required_scopes=frozenset({TokenScope.TOOL_EXECUTE, TokenScope.TOOL_ADMIN}),
        min_trust_tier=TrustTier.T3_ELEVATED,
    ),
    "upload_skill": ToolPolicy(
        tool_name="upload_skill",
        required_scopes=frozenset({TokenScope.TOOL_ADMIN}),
        min_trust_tier=TrustTier.T3_ELEVATED,
    ),
}


def authorize_tool(claims: AuthClaims, policy: ToolPolicy) -> None:
    """Check whether the caller is authorized to invoke a tool.

    Enforcement order: blocked → tier → scopes → allowed.

    Args:
        claims: The caller's validated auth claims.
        policy: The authorization policy for the target tool.

    Raises:
        ToolNotAuthorizedError: If the caller is blocked by subject.
        InsufficientTierError: If the caller's trust tier is too low.
        InsufficientScopeError: If the caller lacks required scopes.
    """
    # 1. Deny-before-allow: blocked subjects
    if policy.blocked_subjects and claims.subject in policy.blocked_subjects:
        raise ToolNotAuthorizedError(f"Subject '{claims.subject}' is blocked from tool '{policy.tool_name}'")

    # 2. Trust tier check
    caller_level = _TIER_ORDER.index(claims.trust_tier)
    required_level = _TIER_ORDER.index(policy.min_trust_tier)

    if caller_level < required_level:
        raise InsufficientTierError(
            f"Tool '{policy.tool_name}' requires tier {policy.min_trust_tier.value}, "
            f"caller has {claims.trust_tier.value}"
        )

    # 3. Scope check
    missing_scopes = policy.required_scopes - claims.scopes
    if missing_scopes:
        missing_names = ", ".join(s.value for s in sorted(missing_scopes, key=lambda s: s.value))
        raise InsufficientScopeError(
            f"Tool '{policy.tool_name}' requires scopes [{missing_names}], caller is missing them"
        )

    # 4. Allowed subjects (if set, only listed subjects may proceed)
    if policy.allowed_subjects and claims.subject not in policy.allowed_subjects:
        raise ToolNotAuthorizedError(
            f"Subject '{claims.subject}' is not in the allowed list for tool '{policy.tool_name}'"
        )


# ---------------------------------------------------------------------------
# #106 — Trust-Tier Gating
# ---------------------------------------------------------------------------


def check_tier_ceiling(
    claims: AuthClaims,
    requested_tier: TrustTier,
) -> None:
    """Enforce that a caller cannot request a tier above their own.

    This is a **tier-only** check. For full lease authorization
    (tier + scope), use :func:`authorize_lease` instead.

    Raises:
        InsufficientTierError: If requested tier exceeds the caller's tier.
    """
    _tier_order = _TIER_ORDER
    caller_level = _tier_order.index(claims.trust_tier)
    requested_level = _tier_order.index(requested_tier)

    if requested_level > caller_level:
        raise InsufficientTierError(
            f"Cannot request tier {requested_tier.value}: caller's ceiling is {claims.trust_tier.value}"
        )


def authorize_lease(
    claims: AuthClaims,
    requested_tier: TrustTier,
    *,
    admin: bool = False,
) -> None:
    """Full lease authorization: tier ceiling + scope enforcement.

    Called before capability lease acquisition. Checks both that the
    caller's tier is sufficient AND that the caller has the appropriate
    lease scope.

    Args:
        claims: The caller's validated auth claims.
        requested_tier: The trust tier being requested for the lease.
        admin: If True, requires LEASE_ADMIN scope; otherwise LEASE_ACQUIRE.

    Raises:
        InsufficientTierError: If requested tier exceeds the caller's tier.
        InsufficientScopeError: If the caller lacks the required lease scope.
    """
    check_tier_ceiling(claims, requested_tier)

    required_scope = TokenScope.LEASE_ADMIN if admin else TokenScope.LEASE_ACQUIRE
    if required_scope not in claims.scopes:
        raise InsufficientScopeError(
            f"Lease {'admin' if admin else 'acquisition'} requires scope {required_scope.value}"
        )


# ---------------------------------------------------------------------------
# #107 — Token Lifecycle (Revocation Registry)
# ---------------------------------------------------------------------------


import logging as _logging

_registry_logger = _logging.getLogger("openspace.auth.registry")


class RegistryFullError(AuthError):
    """Revocation registry has hit its hard ceiling."""


class TokenRegistry:
    """Thread-safe token lifecycle management.

    Tracks issued and revoked tokens with automatic expiry cleanup.
    Only evicts **expired** entries — unexpired revoked tokens are
    never discarded, preventing revocation-bypass via FIFO flooding.

    A hard ceiling (HARD_MAX) prevents unbounded memory growth.
    If the registry is full of unexpired entries and cannot GC enough
    room, ``revoke()`` raises ``RegistryFullError`` (fail-closed).
    """

    MAX_REVOKED = 10_000
    HARD_MAX = 100_000  # absolute ceiling — prevents OOM
    _GC_INTERVAL = 100  # run lazy GC on reads every N calls

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revoked: set[str] = set()
        # (token_id, expires_at_epoch)  — expires_at lets us GC safely
        self._revoked_order: list[tuple[str, float]] = []
        self._ops_since_gc: int = 0

    def revoke(self, token_id: str, expires_at: float | None = None) -> None:
        """Revoke a token by its ID.

        Args:
            token_id: Unique token identifier.
            expires_at: Epoch when the token expires.  Revocation entries
                for expired tokens are eligible for garbage-collection.
                If *None*, the entry is kept indefinitely (safe default).

        Raises:
            RegistryFullError: If the registry has hit HARD_MAX and
                garbage-collection cannot reclaim space.
        """
        with self._lock:
            if token_id in self._revoked:
                return

            # Always GC expired entries before adding
            self._gc_expired_locked()
            self._ops_since_gc = 0

            # Hard ceiling: fail-closed rather than silently dropping entries
            if len(self._revoked_order) >= self.HARD_MAX:
                _registry_logger.critical(
                    "Revocation registry at hard ceiling (%d entries). "
                    "New revocation rejected — investigate token lifecycle.",
                    self.HARD_MAX,
                )
                raise RegistryFullError(f"Revocation registry full ({self.HARD_MAX} entries)")

            # Warn at 2× soft limit
            if len(self._revoked_order) > 2 * self.MAX_REVOKED:
                _registry_logger.warning(
                    "Revocation registry at %d entries (soft limit: %d). Consider reviewing token TTLs.",
                    len(self._revoked_order),
                    self.MAX_REVOKED,
                )

            self._revoked.add(token_id)
            exp = expires_at if expires_at is not None else float("inf")
            self._revoked_order.append((token_id, exp))

    def is_revoked(self, token_id: str) -> bool:
        """Check if a token ID has been revoked."""
        with self._lock:
            self._ops_since_gc += 1
            # Lazy GC on reads to clean up expired entries below the cap
            if self._ops_since_gc >= self._GC_INTERVAL:
                self._gc_expired_locked()
                self._ops_since_gc = 0
            return token_id in self._revoked

    def revoked_count(self) -> int:
        """Number of currently tracked revoked tokens."""
        with self._lock:
            return len(self._revoked)

    def _gc_expired_locked(self) -> None:
        """Remove entries whose tokens have expired. Must hold _lock.

        Only removes entries where ``expires_at < now``.  Unexpired
        revoked tokens are **never** evicted — this prevents the
        FIFO-flooding attack where an adversary revokes 10K dummy
        tokens to resurrect a stolen, still-valid revoked token.
        """
        now = time.time()
        surviving: list[tuple[str, float]] = []
        for tid, exp in self._revoked_order:
            if exp < now:
                self._revoked.discard(tid)
            else:
                surviving.append((tid, exp))
        self._revoked_order = surviving


# ---------------------------------------------------------------------------
# #51 — AuthProvider (concrete AuthPort implementation)
# ---------------------------------------------------------------------------


@dataclass
class AuthProvider:
    """Concrete implementation of AuthPort with HMAC-signed tokens.

    Integrates token validation, revocation checking, and per-tool
    authorization into a single service.

    Args:
        signing_secret: Server secret for HMAC-SHA256 token signing.
        tool_policies: Per-tool authorization policies.
        registry: Token lifecycle registry (optional, created if not provided).
    """

    signing_secret: str = field(repr=False)
    audience: str = ""  # service identifier for token binding
    tool_policies: dict[str, ToolPolicy] = field(default_factory=lambda: dict(DEFAULT_TOOL_POLICIES))
    registry: TokenRegistry = field(default_factory=TokenRegistry)
    _initialized: bool = field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        if len(self.signing_secret) < _MIN_SECRET_LENGTH:
            raise ValueError(f"signing_secret must be at least {_MIN_SECRET_LENGTH} characters")
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "signing_secret" and getattr(self, "_initialized", False):
            raise AttributeError("signing_secret is immutable after init")
        super().__setattr__(name, value)

    def create_token(
        self,
        *,
        subject: str,
        trust_tier: TrustTier = TrustTier.T1_BASIC,
        scopes: frozenset[TokenScope] | None = None,
        ttl_seconds: int = 3600,
        token_id: str | None = None,
    ) -> str:
        """Create a signed token for the given subject."""
        return create_token(
            secret=self.signing_secret,
            subject=subject,
            trust_tier=trust_tier,
            scopes=scopes,
            ttl_seconds=ttl_seconds,
            token_id=token_id,
            audience=self.audience,
        )

    async def authenticate(self, token: str) -> bool:
        """AuthPort.authenticate — validate token and check revocation."""
        try:
            claims = self.validate_and_check(token)
            return claims is not None
        except AuthError:
            return False

    async def validate_token(self, token: str) -> tuple[bool, str]:
        """AuthPort.validate_token — returns (valid, subject_or_error)."""
        try:
            claims = self.validate_and_check(token)
            return True, claims.subject
        except AuthError as exc:
            return False, str(exc)

    def validate_and_check(self, token: str) -> AuthClaims:
        """Full validation: signature → expiry → audience → revocation.

        Returns validated AuthClaims or raises AuthError subclass.
        """
        claims = validate_token(
            token,
            secret=self.signing_secret,
            expected_audience=self.audience,
        )

        if self.registry.is_revoked(claims.token_id):
            raise TokenRevokedError(f"Token '{claims.token_id}' has been revoked")

        return claims

    def authorize(self, token: str, tool_name: str) -> AuthClaims:
        """Validate token AND authorize for a specific tool.

        This is the primary entry point for MCP tool dispatch.

        Returns:
            AuthClaims if authorized.

        Raises:
            AuthError subclass if authentication or authorization fails.
        """
        claims = self.validate_and_check(token)

        policy = self.tool_policies.get(tool_name)
        if policy is None:
            # Unknown tools require admin scope by default
            policy = ToolPolicy(
                tool_name=tool_name,
                required_scopes=frozenset({TokenScope.TOOL_ADMIN}),
                min_trust_tier=TrustTier.T3_ELEVATED,
            )

        authorize_tool(claims, policy)
        return claims

    def revoke_token(self, token_id: str, *, expires_at: float) -> None:
        """Revoke a token by its ID.

        Args:
            token_id: Unique token identifier to revoke.
            expires_at: Epoch when the token expires. **Required** so the
                registry can garbage-collect the entry after expiry.
                Obtain from ``AuthClaims.expires_at`` or the token payload.
        """
        self.registry.revoke(token_id, expires_at=expires_at)

    def revoke(self, token: str) -> None:
        """Revoke a token by validating it and extracting expiry.

        This is the preferred high-level API: pass the full token string,
        and the provider extracts token_id + expires_at automatically.
        Audience is enforced — a provider can only revoke tokens minted
        for its own service.  Use ``revoke_token()`` for cross-service
        admin revocation.

        Args:
            token: The full signed token string to revoke.

        Raises:
            AuthError: If the token cannot be validated (signature/format/audience).
        """
        claims = validate_token(
            token,
            secret=self.signing_secret,
            expected_audience=self.audience,
        )
        self.registry.revoke(claims.token_id, expires_at=claims.expires_at)

    def check_tier_ceiling(self, claims: AuthClaims, requested_tier: TrustTier) -> None:
        """Enforce tier ceiling for capability lease requests."""
        check_tier_ceiling(claims, requested_tier)
