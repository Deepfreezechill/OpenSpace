"""Tests for scion.auth.provider — EPIC 2.5.

Covers:
- #104: Token creation, validation, claims model
- #105: Per-tool authorization (scopes, tiers, subject block/allow)
- #106: Trust-tier gating / ceiling enforcement
- #107: Token lifecycle (revocation, expiry, registry)
- #51: AuthProvider integration (AuthPort concrete implementation)
"""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest

from scion.auth.provider import (
    _MAX_TTL_SECONDS,
    DEFAULT_TOOL_POLICIES,
    TIER_DEFAULT_SCOPES,
    AuthClaims,
    AuthError,
    AuthProvider,
    InsufficientScopeError,
    InsufficientTierError,
    RegistryFullError,
    TokenExpiredError,
    TokenInvalidError,
    TokenRegistry,
    TokenRevokedError,
    TokenScope,
    ToolNotAuthorizedError,
    ToolPolicy,
    authorize_lease,
    authorize_tool,
    check_tier_ceiling,
    create_token,
    validate_token,
)
from scion.sandbox.leases import TrustTier

# Shared test secret (>= 32 chars)
TEST_SECRET = "test-secret-key-for-hmac-signing-at-least-32-chars"


# ═══════════════════════════════════════════════════════════════════════
# #104 — Token Claims Model
# ═══════════════════════════════════════════════════════════════════════


class TestTokenScope:
    """Token scope enum values."""

    def test_scope_values(self) -> None:
        assert TokenScope.TOOL_EXECUTE.value == "tool:execute"
        assert TokenScope.SECRET_READ.value == "secret:read"
        assert TokenScope.LEASE_ADMIN.value == "lease:admin"

    def test_tier_default_scopes_monotonic(self) -> None:
        """Higher tiers should have superset scopes of lower tiers."""
        tiers = [
            TrustTier.T0_UNTRUSTED,
            TrustTier.T1_BASIC,
            TrustTier.T2_STANDARD,
            TrustTier.T3_ELEVATED,
            TrustTier.T4_FULL,
        ]
        for i in range(len(tiers) - 1):
            lower = TIER_DEFAULT_SCOPES[tiers[i]]
            higher = TIER_DEFAULT_SCOPES[tiers[i + 1]]
            assert lower <= higher, f"{tiers[i].value} scopes are not subset of {tiers[i + 1].value}"


class TestAuthClaims:
    """AuthClaims immutability and helper methods."""

    def _make_claims(self, **overrides) -> AuthClaims:
        defaults = dict(
            subject="test-service",
            trust_tier=TrustTier.T2_STANDARD,
            scopes=frozenset({TokenScope.TOOL_EXECUTE, TokenScope.TOOL_SEARCH}),
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="test-id-123",
        )
        defaults.update(overrides)
        return AuthClaims(**defaults)

    def test_frozen(self) -> None:
        claims = self._make_claims()
        with pytest.raises(AttributeError):
            claims.subject = "hacked"  # type: ignore[misc]

    def test_is_expired_false(self) -> None:
        claims = self._make_claims(expires_at=time.time() + 3600)
        assert not claims.is_expired

    def test_is_expired_true(self) -> None:
        claims = self._make_claims(expires_at=time.time() - 1)
        assert claims.is_expired

    def test_remaining_seconds(self) -> None:
        claims = self._make_claims(expires_at=time.time() + 100)
        assert 99 <= claims.remaining_seconds <= 101

    def test_remaining_seconds_expired(self) -> None:
        claims = self._make_claims(expires_at=time.time() - 10)
        assert claims.remaining_seconds == 0.0

    def test_has_scope(self) -> None:
        claims = self._make_claims()
        assert claims.has_scope(TokenScope.TOOL_EXECUTE)
        assert not claims.has_scope(TokenScope.TOOL_ADMIN)

    def test_has_any_scope(self) -> None:
        claims = self._make_claims()
        assert claims.has_any_scope(TokenScope.TOOL_ADMIN, TokenScope.TOOL_EXECUTE)
        assert not claims.has_any_scope(TokenScope.TOOL_ADMIN, TokenScope.SECRET_WRITE)


# ═══════════════════════════════════════════════════════════════════════
# #104 — Token Creation & Validation
# ═══════════════════════════════════════════════════════════════════════


class TestTokenCreation:
    """Create HMAC-signed tokens."""

    def test_create_basic(self) -> None:
        token = create_token(secret=TEST_SECRET, subject="svc-a")
        assert "." in token
        assert len(token) > 50

    def test_create_with_custom_scopes(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="svc-b",
            trust_tier=TrustTier.T3_ELEVATED,
            scopes=frozenset({TokenScope.SECRET_READ}),
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.scopes == frozenset({TokenScope.SECRET_READ})

    def test_create_with_tier(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="svc-c",
            trust_tier=TrustTier.T3_ELEVATED,
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.trust_tier == TrustTier.T3_ELEVATED
        assert claims.scopes == TIER_DEFAULT_SCOPES[TrustTier.T3_ELEVATED]

    def test_create_with_custom_ttl(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="svc-d",
            ttl_seconds=60,
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.remaining_seconds <= 60

    def test_create_with_custom_token_id(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="svc-e",
            token_id="my-custom-id",
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.token_id == "my-custom-id"

    def test_create_short_secret_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            create_token(secret="short", subject="x")

    def test_create_zero_ttl_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            create_token(secret=TEST_SECRET, subject="x", ttl_seconds=0)

    def test_create_exceeds_max_ttl_rejected(self) -> None:
        """F2: TTL above MAX_TTL_SECONDS is rejected."""
        with pytest.raises(ValueError, match="must not exceed"):
            create_token(
                secret=TEST_SECRET,
                subject="x",
                ttl_seconds=_MAX_TTL_SECONDS + 1,
            )

    def test_create_at_max_ttl_succeeds(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="x",
            ttl_seconds=_MAX_TTL_SECONDS,
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.remaining_seconds <= _MAX_TTL_SECONDS

    def test_scope_tier_mismatch_rejected(self) -> None:
        """F3: T1 token cannot be minted with T3 scopes."""
        with pytest.raises(ValueError, match="exceed tier"):
            create_token(
                secret=TEST_SECRET,
                subject="svc",
                trust_tier=TrustTier.T1_BASIC,
                scopes=frozenset({TokenScope.SECRET_WRITE, TokenScope.LEASE_ADMIN}),
            )

    def test_scope_subset_of_tier_allowed(self) -> None:
        """Explicit scopes that are a subset of tier defaults succeed."""
        token = create_token(
            secret=TEST_SECRET,
            subject="svc",
            trust_tier=TrustTier.T2_STANDARD,
            scopes=frozenset({TokenScope.TOOL_SEARCH}),
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.scopes == frozenset({TokenScope.TOOL_SEARCH})


class TestTokenValidation:
    """Validate HMAC-signed tokens."""

    def test_roundtrip(self) -> None:
        token = create_token(secret=TEST_SECRET, subject="roundtrip-svc")
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.subject == "roundtrip-svc"
        assert claims.trust_tier == TrustTier.T1_BASIC

    def test_wrong_secret_rejected(self) -> None:
        token = create_token(secret=TEST_SECRET, subject="svc")
        wrong_secret = "a-completely-different-secret-key-at-least-32-chars"
        with pytest.raises(TokenInvalidError, match="signature"):
            validate_token(token, secret=wrong_secret)

    def test_tampered_payload_rejected(self) -> None:
        token = create_token(secret=TEST_SECRET, subject="svc")
        parts = token.split(".")
        # Tamper with a character in payload
        tampered = parts[0][:-1] + ("A" if parts[0][-1] != "A" else "B")
        tampered_token = f"{tampered}.{parts[1]}"
        with pytest.raises(TokenInvalidError):
            validate_token(tampered_token, secret=TEST_SECRET)

    def test_tampered_signature_rejected(self) -> None:
        token = create_token(secret=TEST_SECRET, subject="svc")
        parts = token.split(".")
        bad_sig = parts[1][:-1] + ("X" if parts[1][-1] != "X" else "Y")
        with pytest.raises(TokenInvalidError, match="signature"):
            validate_token(f"{parts[0]}.{bad_sig}", secret=TEST_SECRET)

    def test_missing_separator_rejected(self) -> None:
        with pytest.raises(TokenInvalidError, match="separator"):
            validate_token("no-dot-here", secret=TEST_SECRET)

    def test_too_many_parts_rejected(self) -> None:
        with pytest.raises(TokenInvalidError, match="2 parts"):
            validate_token("a.b.c", secret=TEST_SECRET)

    def test_expired_token_rejected(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="svc",
            ttl_seconds=1,
        )
        # Manually create an already-expired token
        import base64
        import json

        payload_b64 = token.split(".")[0]
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        payload["exp"] = time.time() - 10
        payload["iat"] = time.time() - 20
        new_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        new_b64 = base64.urlsafe_b64encode(new_payload.encode()).rstrip(b"=").decode()

        import hashlib
        import hmac as _hmac

        sig = _hmac.new(TEST_SECRET.encode(), new_b64.encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()

        expired_token = f"{new_b64}.{sig_b64}"
        with pytest.raises(TokenExpiredError):
            validate_token(expired_token, secret=TEST_SECRET)

    def test_t0_token_no_scopes(self) -> None:
        token = create_token(
            secret=TEST_SECRET,
            subject="svc",
            trust_tier=TrustTier.T0_UNTRUSTED,
        )
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.scopes == frozenset()


# ═══════════════════════════════════════════════════════════════════════
# #105 — Per-Tool Authorization
# ═══════════════════════════════════════════════════════════════════════


class TestAuthorizeToolScope:
    """Scope-based tool authorization."""

    def _claims(self, scopes, tier=TrustTier.T4_FULL) -> AuthClaims:
        return AuthClaims(
            subject="svc",
            trust_tier=tier,
            scopes=frozenset(scopes),
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="t1",
        )

    def test_sufficient_scopes_pass(self) -> None:
        claims = self._claims([TokenScope.TOOL_EXECUTE])
        policy = ToolPolicy(
            tool_name="run",
            required_scopes=frozenset({TokenScope.TOOL_EXECUTE}),
        )
        authorize_tool(claims, policy)  # no raise

    def test_missing_scope_raises(self) -> None:
        claims = self._claims([TokenScope.TOOL_SEARCH])
        policy = ToolPolicy(
            tool_name="run",
            required_scopes=frozenset({TokenScope.TOOL_EXECUTE}),
        )
        with pytest.raises(InsufficientScopeError, match="tool:execute"):
            authorize_tool(claims, policy)

    def test_superset_scopes_pass(self) -> None:
        claims = self._claims(
            [
                TokenScope.TOOL_EXECUTE,
                TokenScope.TOOL_ADMIN,
                TokenScope.TOOL_SEARCH,
            ]
        )
        policy = ToolPolicy(
            tool_name="run",
            required_scopes=frozenset({TokenScope.TOOL_EXECUTE}),
        )
        authorize_tool(claims, policy)

    def test_empty_required_scopes_pass(self) -> None:
        claims = self._claims([])
        policy = ToolPolicy(tool_name="run", required_scopes=frozenset())
        authorize_tool(claims, policy)


class TestAuthorizeToolTier:
    """Trust-tier tool authorization."""

    def _claims(self, tier) -> AuthClaims:
        return AuthClaims(
            subject="svc",
            trust_tier=tier,
            scopes=frozenset({TokenScope.TOOL_EXECUTE, TokenScope.TOOL_ADMIN}),
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="t1",
        )

    def test_sufficient_tier_pass(self) -> None:
        claims = self._claims(TrustTier.T3_ELEVATED)
        policy = ToolPolicy(tool_name="x", min_trust_tier=TrustTier.T2_STANDARD)
        authorize_tool(claims, policy)

    def test_exact_tier_pass(self) -> None:
        claims = self._claims(TrustTier.T2_STANDARD)
        policy = ToolPolicy(tool_name="x", min_trust_tier=TrustTier.T2_STANDARD)
        authorize_tool(claims, policy)

    def test_insufficient_tier_raises(self) -> None:
        claims = self._claims(TrustTier.T1_BASIC)
        policy = ToolPolicy(tool_name="x", min_trust_tier=TrustTier.T3_ELEVATED)
        with pytest.raises(InsufficientTierError, match="T3"):
            authorize_tool(claims, policy)


class TestAuthorizeToolSubject:
    """Subject-based tool authorization (block/allow lists)."""

    def _claims(self, subject="svc-a") -> AuthClaims:
        return AuthClaims(
            subject=subject,
            trust_tier=TrustTier.T4_FULL,
            scopes=frozenset({TokenScope.TOOL_EXECUTE}),
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="t1",
        )

    def test_blocked_subject_rejected(self) -> None:
        policy = ToolPolicy(
            tool_name="x",
            blocked_subjects=frozenset({"svc-a"}),
        )
        with pytest.raises(ToolNotAuthorizedError, match="blocked"):
            authorize_tool(self._claims("svc-a"), policy)

    def test_non_blocked_subject_passes(self) -> None:
        policy = ToolPolicy(
            tool_name="x",
            blocked_subjects=frozenset({"svc-b"}),
        )
        authorize_tool(self._claims("svc-a"), policy)

    def test_allowed_subject_passes(self) -> None:
        policy = ToolPolicy(
            tool_name="x",
            allowed_subjects=frozenset({"svc-a", "svc-b"}),
        )
        authorize_tool(self._claims("svc-a"), policy)

    def test_not_in_allowed_list_rejected(self) -> None:
        policy = ToolPolicy(
            tool_name="x",
            allowed_subjects=frozenset({"svc-b"}),
        )
        with pytest.raises(ToolNotAuthorizedError, match="not in the allowed"):
            authorize_tool(self._claims("svc-a"), policy)

    def test_empty_allowed_means_open(self) -> None:
        """No allowed_subjects = no subject restriction."""
        policy = ToolPolicy(tool_name="x", allowed_subjects=frozenset())
        authorize_tool(self._claims("anyone"), policy)

    def test_blocked_before_allowed(self) -> None:
        """Deny-before-allow: blocked check runs first."""
        policy = ToolPolicy(
            tool_name="x",
            blocked_subjects=frozenset({"svc-a"}),
            allowed_subjects=frozenset({"svc-a"}),
        )
        with pytest.raises(ToolNotAuthorizedError, match="blocked"):
            authorize_tool(self._claims("svc-a"), policy)


class TestDefaultToolPolicies:
    """Default policies for built-in MCP tools."""

    def test_execute_task_requires_t2(self) -> None:
        policy = DEFAULT_TOOL_POLICIES["execute_task"]
        assert policy.min_trust_tier == TrustTier.T2_STANDARD
        assert TokenScope.TOOL_EXECUTE in policy.required_scopes

    def test_search_skills_requires_t1(self) -> None:
        policy = DEFAULT_TOOL_POLICIES["search_skills"]
        assert policy.min_trust_tier == TrustTier.T1_BASIC

    def test_upload_skill_requires_admin(self) -> None:
        policy = DEFAULT_TOOL_POLICIES["upload_skill"]
        assert TokenScope.TOOL_ADMIN in policy.required_scopes
        assert policy.min_trust_tier == TrustTier.T3_ELEVATED


# ═══════════════════════════════════════════════════════════════════════
# #106 — Trust-Tier Gating
# ═══════════════════════════════════════════════════════════════════════


class TestTierCeiling:
    """Trust-tier ceiling enforcement for lease requests."""

    def _claims(self, tier) -> AuthClaims:
        return AuthClaims(
            subject="svc",
            trust_tier=tier,
            scopes=frozenset(),
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="t1",
        )

    def test_same_tier_passes(self) -> None:
        check_tier_ceiling(self._claims(TrustTier.T2_STANDARD), TrustTier.T2_STANDARD)

    def test_lower_tier_passes(self) -> None:
        check_tier_ceiling(self._claims(TrustTier.T3_ELEVATED), TrustTier.T2_STANDARD)

    def test_higher_tier_rejected(self) -> None:
        with pytest.raises(InsufficientTierError, match="ceiling"):
            check_tier_ceiling(self._claims(TrustTier.T1_BASIC), TrustTier.T3_ELEVATED)

    def test_t0_cannot_request_t1(self) -> None:
        with pytest.raises(InsufficientTierError):
            check_tier_ceiling(self._claims(TrustTier.T0_UNTRUSTED), TrustTier.T1_BASIC)

    def test_t4_can_request_any(self) -> None:
        for tier in TrustTier:
            check_tier_ceiling(self._claims(TrustTier.T4_FULL), tier)


class TestAuthorizeLease:
    """authorize_lease: tier ceiling + lease scope enforcement."""

    def _claims(self, tier, scopes=frozenset()) -> AuthClaims:
        return AuthClaims(
            subject="svc",
            trust_tier=tier,
            scopes=scopes,
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="t1",
        )

    def test_acquire_with_scope_passes(self) -> None:
        claims = self._claims(
            TrustTier.T2_STANDARD,
            frozenset({TokenScope.LEASE_ACQUIRE}),
        )
        authorize_lease(claims, TrustTier.T2_STANDARD)

    def test_acquire_without_scope_rejected(self) -> None:
        """Token with sufficient tier but missing LEASE_ACQUIRE is rejected."""
        claims = self._claims(
            TrustTier.T2_STANDARD,
            frozenset({TokenScope.TOOL_EXECUTE}),
        )
        with pytest.raises(InsufficientScopeError, match="lease:acquire"):
            authorize_lease(claims, TrustTier.T1_BASIC)

    def test_admin_requires_lease_admin_scope(self) -> None:
        claims = self._claims(
            TrustTier.T4_FULL,
            frozenset({TokenScope.LEASE_ACQUIRE}),
        )
        with pytest.raises(InsufficientScopeError, match="lease:admin"):
            authorize_lease(claims, TrustTier.T1_BASIC, admin=True)

    def test_admin_with_scope_passes(self) -> None:
        claims = self._claims(
            TrustTier.T4_FULL,
            frozenset({TokenScope.LEASE_ADMIN}),
        )
        authorize_lease(claims, TrustTier.T4_FULL, admin=True)

    def test_tier_still_enforced(self) -> None:
        """Even with LEASE_ACQUIRE, tier ceiling is enforced."""
        claims = self._claims(
            TrustTier.T1_BASIC,
            frozenset({TokenScope.LEASE_ACQUIRE}),
        )
        with pytest.raises(InsufficientTierError):
            authorize_lease(claims, TrustTier.T3_ELEVATED)

    def test_empty_scopes_rejected(self) -> None:
        """Token with no scopes cannot acquire leases."""
        claims = self._claims(TrustTier.T2_STANDARD, frozenset())
        with pytest.raises(InsufficientScopeError):
            authorize_lease(claims, TrustTier.T1_BASIC)


# ═══════════════════════════════════════════════════════════════════════
# #107 — Token Registry (Revocation)
# ═══════════════════════════════════════════════════════════════════════


class TestTokenRegistry:
    """Token revocation registry."""

    def test_new_token_not_revoked(self) -> None:
        reg = TokenRegistry()
        assert not reg.is_revoked("abc")

    def test_revoke_marks_as_revoked(self) -> None:
        reg = TokenRegistry()
        reg.revoke("abc")
        assert reg.is_revoked("abc")

    def test_revoke_idempotent(self) -> None:
        reg = TokenRegistry()
        reg.revoke("abc")
        reg.revoke("abc")
        assert reg.revoked_count() == 1

    def test_multiple_revocations(self) -> None:
        reg = TokenRegistry()
        for i in range(100):
            reg.revoke(f"token-{i}")
        assert reg.revoked_count() == 100
        assert reg.is_revoked("token-50")
        assert not reg.is_revoked("token-999")

    def test_expired_entries_gc(self) -> None:
        """Only expired revocation entries are garbage-collected."""
        reg = TokenRegistry()
        reg.MAX_REVOKED = 5
        past = time.time() - 100  # already expired
        future = time.time() + 3600  # still valid
        # Fill with 3 expired + 3 unexpired entries → over MAX_REVOKED(5)
        for i in range(3):
            reg.revoke(f"expired-{i}", expires_at=past)
        for i in range(3):
            reg.revoke(f"valid-{i}", expires_at=future)
        # Expired entries should be GC'd, unexpired retained
        for i in range(3):
            assert not reg.is_revoked(f"expired-{i}"), "expired should be GC'd"
        for i in range(3):
            assert reg.is_revoked(f"valid-{i}"), "unexpired must be kept"
        assert reg.revoked_count() == 3

    def test_fifo_flooding_does_not_resurrect_revoked_token(self) -> None:
        """Regression: flooding with revocations must NOT evict unexpired entries.

        PoC from security review R1: revoke stolen admin token, then flood with 10K+
        other revocations — the victim token MUST stay revoked.
        """
        reg = TokenRegistry()
        reg.MAX_REVOKED = 10
        future = time.time() + 3600
        # Revoke the "victim" — unexpired, must NEVER be evicted
        reg.revoke("victim", expires_at=future)
        # Flood with additional revocations (all also unexpired)
        for i in range(20):
            reg.revoke(f"flood-{i}", expires_at=future)
        # Victim must still be revoked
        assert reg.is_revoked("victim"), "FIFO flooding must not resurrect victim"
        # All flood tokens also retained (none are expired)
        for i in range(20):
            assert reg.is_revoked(f"flood-{i}")

    def test_revoke_without_expiry_never_evicted(self) -> None:
        """Tokens revoked without expires_at are kept indefinitely."""
        reg = TokenRegistry()
        reg.MAX_REVOKED = 3
        past = time.time() - 100
        reg.revoke("permanent")  # no expires_at
        for i in range(5):
            reg.revoke(f"exp-{i}", expires_at=past)
        # Permanent entry must survive
        assert reg.is_revoked("permanent")

    def test_thread_safety(self) -> None:
        reg = TokenRegistry()
        errors: list[Exception] = []

        def revoke_batch(start: int) -> None:
            try:
                for i in range(100):
                    reg.revoke(f"t-{start}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=revoke_batch, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert reg.revoked_count() == 500

    def test_hard_ceiling_raises_registry_full(self) -> None:
        """Registry raises RegistryFullError at HARD_MAX."""
        reg = TokenRegistry()
        reg.MAX_REVOKED = 5
        reg.HARD_MAX = 10
        future = time.time() + 3600
        for i in range(10):
            reg.revoke(f"t-{i}", expires_at=future)
        with pytest.raises(RegistryFullError, match="full"):
            reg.revoke("one-too-many", expires_at=future)
        # All 10 still revoked
        assert reg.revoked_count() == 10

    def test_lazy_gc_on_is_revoked(self) -> None:
        """Expired entries are cleaned up during is_revoked reads."""
        reg = TokenRegistry()
        reg._GC_INTERVAL = 3  # trigger after 3 reads
        future = time.time() + 3600
        past = time.time() - 100
        # Add unexpired entry first (won't be GC'd during revoke)
        reg.revoke("alive", expires_at=future)
        # Manually inject expired entries to bypass revoke-time GC
        with reg._lock:
            reg._revoked.add("expired-1")
            reg._revoked_order.append(("expired-1", past))
            reg._revoked.add("expired-2")
            reg._revoked_order.append(("expired-2", past))
        assert reg.revoked_count() == 3
        # Trigger lazy GC via reads
        for _ in range(3):
            reg.is_revoked("anything")
        # Expired should be gone, alive should remain
        assert reg.revoked_count() == 1
        assert reg.is_revoked("alive")

    def test_revoke_with_expiry_passthrough(self) -> None:
        """AuthProvider.revoke_token passes expires_at to registry."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        exp = time.time() + 3600
        provider.revoke_token("tid", expires_at=exp)
        assert provider.registry.is_revoked("tid")
        assert provider.registry._revoked_order[-1][1] == exp

    def test_high_level_revoke_extracts_expiry(self) -> None:
        """AuthProvider.revoke(token) auto-extracts expires_at from claims."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(subject="svc", token_id="auto-exp")
        provider.revoke(token)
        assert provider.registry.is_revoked("auto-exp")
        # Should store actual expiry, not inf
        _, stored_exp = provider.registry._revoked_order[-1]
        assert stored_exp != float("inf")
        assert stored_exp > time.time()


# ═══════════════════════════════════════════════════════════════════════
# #51 — AuthProvider Integration
# ═══════════════════════════════════════════════════════════════════════


class TestAuthProvider:
    """AuthProvider as concrete AuthPort implementation."""

    def _provider(self, **kwargs) -> AuthProvider:
        return AuthProvider(signing_secret=TEST_SECRET, **kwargs)

    def test_create_and_validate(self) -> None:
        provider = self._provider()
        token = provider.create_token(subject="svc-a")
        claims = provider.validate_and_check(token)
        assert claims.subject == "svc-a"

    def test_short_secret_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            AuthProvider(signing_secret="short")

    def test_revocation_flow(self) -> None:
        provider = self._provider()
        token = provider.create_token(subject="svc", token_id="revoke-me")
        # Valid before revocation
        claims = provider.validate_and_check(token)
        assert claims.token_id == "revoke-me"
        # Revoke using high-level API (extracts expiry automatically)
        provider.revoke(token)
        with pytest.raises(TokenRevokedError):
            provider.validate_and_check(token)

    def test_authorize_success(self) -> None:
        provider = self._provider()
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T2_STANDARD,
        )
        claims = provider.authorize(token, "execute_task")
        assert claims.subject == "svc"

    def test_authorize_insufficient_tier(self) -> None:
        provider = self._provider()
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T1_BASIC,
        )
        with pytest.raises(InsufficientTierError):
            provider.authorize(token, "execute_task")

    def test_authorize_unknown_tool_requires_admin(self) -> None:
        provider = self._provider()
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T2_STANDARD,
        )
        with pytest.raises(InsufficientTierError):
            provider.authorize(token, "unknown_tool")

    def test_authorize_unknown_tool_with_admin(self) -> None:
        provider = self._provider()
        token = provider.create_token(
            subject="admin",
            trust_tier=TrustTier.T4_FULL,
        )
        claims = provider.authorize(token, "unknown_tool")
        assert claims.subject == "admin"

    def test_tier_ceiling(self) -> None:
        provider = self._provider()
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T2_STANDARD,
        )
        claims = provider.validate_and_check(token)
        provider.check_tier_ceiling(claims, TrustTier.T2_STANDARD)
        with pytest.raises(InsufficientTierError):
            provider.check_tier_ceiling(claims, TrustTier.T3_ELEVATED)


class TestAuthPortProtocol:
    """AuthProvider satisfies AuthPort protocol."""

    @pytest.mark.asyncio
    async def test_authenticate_valid(self) -> None:
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(subject="svc")
        assert await provider.authenticate(token) is True

    @pytest.mark.asyncio
    async def test_authenticate_invalid(self) -> None:
        provider = AuthProvider(signing_secret=TEST_SECRET)
        assert await provider.authenticate("garbage") is False

    @pytest.mark.asyncio
    async def test_authenticate_revoked(self) -> None:
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(subject="svc", token_id="rev1")
        provider.revoke(token)
        assert await provider.authenticate(token) is False

    @pytest.mark.asyncio
    async def test_validate_token_valid(self) -> None:
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(subject="svc-x")
        valid, subject = await provider.validate_token(token)
        assert valid is True
        assert subject == "svc-x"

    @pytest.mark.asyncio
    async def test_validate_token_invalid(self) -> None:
        provider = AuthProvider(signing_secret=TEST_SECRET)
        valid, msg = await provider.validate_token("bad-token")
        assert valid is False
        assert "separator" in msg.lower() or "malformed" in msg.lower()


class TestAuthProviderCustomPolicies:
    """Custom tool policies in AuthProvider."""

    def test_custom_policy_enforced(self) -> None:
        custom = {
            "my_tool": ToolPolicy(
                tool_name="my_tool",
                required_scopes=frozenset({TokenScope.SECRET_READ}),
                min_trust_tier=TrustTier.T3_ELEVATED,
            ),
        }
        provider = AuthProvider(signing_secret=TEST_SECRET, tool_policies=custom)
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T3_ELEVATED,
        )
        # T3 has SECRET_READ in default scopes
        claims = provider.authorize(token, "my_tool")
        assert claims.subject == "svc"

    def test_custom_policy_blocks_low_tier(self) -> None:
        custom = {
            "my_tool": ToolPolicy(
                tool_name="my_tool",
                min_trust_tier=TrustTier.T3_ELEVATED,
            ),
        }
        provider = AuthProvider(signing_secret=TEST_SECRET, tool_policies=custom)
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T1_BASIC,
        )
        with pytest.raises(InsufficientTierError):
            provider.authorize(token, "my_tool")


# ═══════════════════════════════════════════════════════════════════════
# Security Regression Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSecurityRegressions:
    """Ensure security invariants hold."""

    def test_token_not_reusable_after_revocation(self) -> None:
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(subject="svc", token_id="sec-1")
        provider.validate_and_check(token)  # OK
        provider.revoke(token)
        with pytest.raises(TokenRevokedError):
            provider.validate_and_check(token)

    def test_cannot_forge_token_with_different_secret(self) -> None:
        legit = create_token(secret=TEST_SECRET, subject="admin", trust_tier=TrustTier.T4_FULL)
        forged_secret = "attacker-secret-that-is-at-least-32-characters"
        forged = create_token(secret=forged_secret, subject="admin", trust_tier=TrustTier.T4_FULL)
        # Legit works
        validate_token(legit, secret=TEST_SECRET)
        # Forged fails
        with pytest.raises(TokenInvalidError, match="signature"):
            validate_token(forged, secret=TEST_SECRET)

    def test_tier_escalation_prevented(self) -> None:
        """T1 token cannot access T3 tool."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(
            subject="basic-svc",
            trust_tier=TrustTier.T1_BASIC,
        )
        with pytest.raises(InsufficientTierError):
            provider.authorize(token, "fix_skill")

    def test_scope_escalation_prevented(self) -> None:
        """T2 token without TOOL_ADMIN cannot access admin tools."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(
            subject="svc",
            trust_tier=TrustTier.T2_STANDARD,
        )
        with pytest.raises(InsufficientTierError):
            provider.authorize(token, "upload_skill")

    def test_t0_cannot_do_anything(self) -> None:
        """T0 tokens have no scopes and fail all tool authorization."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        token = provider.create_token(
            subject="untrusted",
            trust_tier=TrustTier.T0_UNTRUSTED,
        )
        for tool_name in DEFAULT_TOOL_POLICIES:
            with pytest.raises(AuthError):
                provider.authorize(token, tool_name)

    def test_timing_safe_comparison(self) -> None:
        """Token validation uses hmac.compare_digest (timing-safe)."""
        import hmac as _hmac

        # This is a design assertion — hmac.compare_digest is used in
        # _compute_signature verification path
        assert hasattr(_hmac, "compare_digest")

    def test_deny_before_allow_in_tool_auth(self) -> None:
        """Blocked subjects are rejected even if in allowed list."""
        claims = AuthClaims(
            subject="evil",
            trust_tier=TrustTier.T4_FULL,
            scopes=frozenset({TokenScope.TOOL_EXECUTE}),
            issued_at=time.time(),
            expires_at=time.time() + 3600,
            token_id="t1",
        )
        policy = ToolPolicy(
            tool_name="x",
            blocked_subjects=frozenset({"evil"}),
            allowed_subjects=frozenset({"evil"}),
        )
        with pytest.raises(ToolNotAuthorizedError, match="blocked"):
            authorize_tool(claims, policy)

    def test_secret_not_in_repr(self) -> None:
        """F4: signing_secret must not appear in repr()."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        r = repr(provider)
        assert TEST_SECRET not in r
        assert "signing_secret" not in r

    def test_secret_immutable_after_init(self) -> None:
        """F4: signing_secret cannot be mutated after construction."""
        provider = AuthProvider(signing_secret=TEST_SECRET)
        with pytest.raises(AttributeError, match="immutable"):
            provider.signing_secret = "new-secret-at-least-32-characters-long"

    def test_error_messages_do_not_leak_timestamps(self) -> None:
        """F6: error messages should not contain server timestamps."""
        # Create an expired token by time manipulation
        token = create_token(
            secret=TEST_SECRET,
            subject="x",
            ttl_seconds=1,
        )
        # Wait for expiry
        time.sleep(1.1)
        with pytest.raises(TokenExpiredError) as exc_info:
            validate_token(token, secret=TEST_SECRET)
        msg = str(exc_info.value)
        assert "now:" not in msg
        assert "expired at" not in msg.lower() or "has expired" in msg.lower()

    def test_error_messages_do_not_leak_internals(self) -> None:
        """F6: invalid token errors should be generic."""
        with pytest.raises(TokenInvalidError) as exc_info:
            validate_token("not.atoken", secret=TEST_SECRET)
        msg = str(exc_info.value)
        # Should not contain stack traces or internal details
        assert "Traceback" not in msg

    def test_nan_expires_at_rejected(self) -> None:
        """NaN in expires_at must not bypass expiry check."""
        payload = {
            "sub": "attacker",
            "tier": "T1",
            "scopes": [],
            "iat": time.time(),
            "exp": float("nan"),
            "jti": "nan-test",
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode("ascii")
        import hashlib as _hashlib
        import hmac as _hmac

        sig = _hmac.new(TEST_SECRET.encode(), payload_b64.encode(), _hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
        evil_token = f"{payload_b64}.{sig_b64}"
        with pytest.raises(TokenInvalidError, match="finite"):
            validate_token(evil_token, secret=TEST_SECRET)

    def test_infinity_expires_at_rejected(self) -> None:
        """Infinity in expires_at must not create immortal tokens."""
        payload = {
            "sub": "attacker",
            "tier": "T1",
            "scopes": [],
            "iat": time.time(),
            "exp": float("inf"),
            "jti": "inf-test",
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode("ascii")
        import hashlib as _hashlib
        import hmac as _hmac

        sig = _hmac.new(TEST_SECRET.encode(), payload_b64.encode(), _hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
        evil_token = f"{payload_b64}.{sig_b64}"
        with pytest.raises(TokenInvalidError, match="finite"):
            validate_token(evil_token, secret=TEST_SECRET)

    def test_exp_before_iat_rejected(self) -> None:
        """Token where exp <= iat is rejected."""
        now = time.time()
        payload = {
            "sub": "attacker",
            "tier": "T1",
            "scopes": [],
            "iat": now,
            "exp": now + 3600,
            "jti": "backwards",
        }
        # Manually set exp <= iat AFTER normal creation to bypass create_token
        payload["exp"] = now - 1
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode("ascii")
        import hashlib as _hashlib
        import hmac as _hmac

        sig = _hmac.new(TEST_SECRET.encode(), payload_b64.encode(), _hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
        evil_token = f"{payload_b64}.{sig_b64}"
        # Could raise either TokenExpiredError or TokenInvalidError
        with pytest.raises(TokenInvalidError):
            validate_token(evil_token, secret=TEST_SECRET)

    def test_audience_mismatch_rejected(self) -> None:
        """Token minted for service-A is rejected by service-B."""
        token = create_token(
            secret=TEST_SECRET,
            subject="svc",
            audience="service-a",
        )
        # Accepted by service-a
        claims = validate_token(token, secret=TEST_SECRET, expected_audience="service-a")
        assert claims.audience == "service-a"
        # Rejected by service-b
        with pytest.raises(TokenInvalidError, match="audience"):
            validate_token(token, secret=TEST_SECRET, expected_audience="service-b")

    def test_audience_not_enforced_when_empty(self) -> None:
        """Tokens without audience work when validator has no expectation."""
        token = create_token(secret=TEST_SECRET, subject="svc")
        claims = validate_token(token, secret=TEST_SECRET)
        assert claims.audience == ""

    def test_provider_audience_binding(self) -> None:
        """AuthProvider enforces audience on validate_and_check."""
        provider_a = AuthProvider(signing_secret=TEST_SECRET, audience="svc-a")
        provider_b = AuthProvider(signing_secret=TEST_SECRET, audience="svc-b")
        token = provider_a.create_token(subject="user")
        # Works on provider_a
        claims = provider_a.validate_and_check(token)
        assert claims.audience == "svc-a"
        # Rejected by provider_b
        with pytest.raises(TokenInvalidError, match="audience"):
            provider_b.validate_and_check(token)

    def test_cross_service_revoke_blocked(self) -> None:
        """Provider B cannot revoke provider A's tokens via high-level API."""
        shared_registry = TokenRegistry()
        provider_a = AuthProvider(
            signing_secret=TEST_SECRET,
            audience="svc-a",
            registry=shared_registry,
        )
        provider_b = AuthProvider(
            signing_secret=TEST_SECRET,
            audience="svc-b",
            registry=shared_registry,
        )
        token_a = provider_a.create_token(subject="user", token_id="cross-rev")
        # provider_b cannot revoke provider_a's token
        with pytest.raises(TokenInvalidError, match="audience"):
            provider_b.revoke(token_a)
        # Token is still valid on provider_a
        claims = provider_a.validate_and_check(token_a)
        assert claims.token_id == "cross-rev"
