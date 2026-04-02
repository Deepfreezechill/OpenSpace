"""Tests for openspace.secret.broker — EPIC 2.6.

Covers:
- #52: SecretBrokerPort concrete implementation
- Secret scoping (task, session, global)
- Lease-based access control via SecretCapability
- At-rest encryption/decryption
- Key validation and store bounds
- Thread safety
"""

from __future__ import annotations

import threading
import time

import pytest

from openspace.sandbox.leases import SecretCapability
from openspace.secret.broker import (
    SecretAccessDenied,
    SecretBroker,
    SecretBrokerError,
    SecretKeyInvalid,
    SecretScope,
    SecretStore,
    SecretStoreFull,
    SecretValueTooLarge,
    _SecretEncryptor,
    _validate_key,
)

# ═══════════════════════════════════════════════════════════════════════
# Key Validation
# ═══════════════════════════════════════════════════════════════════════


class TestKeyValidation:
    """Secret key naming rules."""

    def test_valid_keys(self) -> None:
        for key in ["api-key", "DB_PASSWORD", "my.secret/path:v1", "a"]:
            _validate_key(key)  # should not raise

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(SecretKeyInvalid, match="empty"):
            _validate_key("")

    def test_too_long_key_rejected(self) -> None:
        with pytest.raises(SecretKeyInvalid, match="maximum length"):
            _validate_key("x" * 257)

    def test_invalid_chars_rejected(self) -> None:
        with pytest.raises(SecretKeyInvalid, match="invalid characters"):
            _validate_key("key with spaces")

    def test_special_chars_rejected(self) -> None:
        for ch in ["$", "!", "@", "#", "%", "^", "&", "*", "(", ")"]:
            with pytest.raises(SecretKeyInvalid):
                _validate_key(f"key{ch}")


# ═══════════════════════════════════════════════════════════════════════
# At-Rest Encryption
# ═══════════════════════════════════════════════════════════════════════


class TestSecretEncryptor:
    """XOR-based at-rest encryption."""

    def test_roundtrip(self) -> None:
        enc = _SecretEncryptor(b"master-key-32-bytes-for-testing!")
        plaintext = "super-secret-value-123"
        encrypted = enc.encrypt(plaintext)
        assert enc.decrypt(encrypted) == plaintext

    def test_encrypted_differs_from_plaintext(self) -> None:
        enc = _SecretEncryptor(b"test-key-32-bytes-for-testing!!")
        plaintext = "my-api-key"
        encrypted = enc.encrypt(plaintext)
        assert plaintext.encode() not in encrypted

    def test_different_nonce_different_ciphertext(self) -> None:
        enc = _SecretEncryptor(b"test-key-32-bytes-for-testing!!")
        plaintext = "same-value"
        e1 = enc.encrypt(plaintext)
        e2 = enc.encrypt(plaintext)
        assert e1 != e2  # different nonce → different output
        assert enc.decrypt(e1) == enc.decrypt(e2) == plaintext

    def test_corrupt_data_rejected(self) -> None:
        enc = _SecretEncryptor(b"key")
        with pytest.raises(SecretBrokerError, match="Corrupt"):
            enc.decrypt(b"short")

    def test_empty_string(self) -> None:
        enc = _SecretEncryptor(b"test-key-32-bytes-for-testing!!")
        encrypted = enc.encrypt("")
        assert enc.decrypt(encrypted) == ""

    def test_unicode_roundtrip(self) -> None:
        enc = _SecretEncryptor(b"test-key-32-bytes-for-testing!!")
        plaintext = "héllo wörld 🔑"
        assert enc.decrypt(enc.encrypt(plaintext)) == plaintext

    def test_long_value(self) -> None:
        enc = _SecretEncryptor(b"test-key-32-bytes-for-testing!!")
        plaintext = "x" * 10000
        assert enc.decrypt(enc.encrypt(plaintext)) == plaintext


# ═══════════════════════════════════════════════════════════════════════
# Secret Store
# ═══════════════════════════════════════════════════════════════════════


class TestSecretStore:
    """Scoped, encrypted, bounded secret storage."""

    def test_put_and_get(self) -> None:
        store = SecretStore()
        store.put("api-key", "secret123", scope=SecretScope.TASK, owner="svc")
        assert store.get("api-key", scope=SecretScope.TASK) == "secret123"

    def test_get_missing_returns_none(self) -> None:
        store = SecretStore()
        assert store.get("nope", scope=SecretScope.TASK) is None

    def test_scopes_are_independent(self) -> None:
        store = SecretStore()
        store.put("key", "task-val", scope=SecretScope.TASK, owner="svc")
        store.put("key", "session-val", scope=SecretScope.SESSION, owner="svc")
        assert store.get("key", scope=SecretScope.TASK) == "task-val"
        assert store.get("key", scope=SecretScope.SESSION) == "session-val"
        assert store.get("key", scope=SecretScope.GLOBAL) is None

    def test_update_existing(self) -> None:
        store = SecretStore()
        store.put("key", "v1", scope=SecretScope.TASK, owner="svc")
        store.put("key", "v2", scope=SecretScope.TASK, owner="svc")
        assert store.get("key", scope=SecretScope.TASK) == "v2"

    def test_delete(self) -> None:
        store = SecretStore()
        store.put("key", "val", scope=SecretScope.TASK, owner="svc")
        assert store.delete("key", scope=SecretScope.TASK) is True
        assert store.get("key", scope=SecretScope.TASK) is None
        assert store.delete("key", scope=SecretScope.TASK) is False

    def test_list_keys(self) -> None:
        store = SecretStore()
        store.put("b-key", "1", scope=SecretScope.TASK, owner="svc")
        store.put("a-key", "2", scope=SecretScope.TASK, owner="svc")
        assert store.list_keys(scope=SecretScope.TASK) == ["a-key", "b-key"]

    def test_count(self) -> None:
        store = SecretStore()
        assert store.count(scope=SecretScope.TASK) == 0
        store.put("k1", "v", scope=SecretScope.TASK, owner="svc")
        store.put("k2", "v", scope=SecretScope.TASK, owner="svc")
        assert store.count(scope=SecretScope.TASK) == 2

    def test_clear_scope(self) -> None:
        store = SecretStore()
        store.put("k1", "v", scope=SecretScope.TASK, owner="svc")
        store.put("k2", "v", scope=SecretScope.TASK, owner="svc")
        store.put("k3", "v", scope=SecretScope.SESSION, owner="svc")
        assert store.clear_scope(SecretScope.TASK) == 2
        assert store.count(scope=SecretScope.TASK) == 0
        assert store.count(scope=SecretScope.SESSION) == 1

    def test_capacity_limit(self) -> None:
        store = SecretStore()
        store.MAX_SECRETS_PER_SCOPE = 3
        for i in range(3):
            store.put(f"k{i}", "v", scope=SecretScope.TASK, owner="svc")
        with pytest.raises(SecretStoreFull, match="full"):
            store.put("overflow", "v", scope=SecretScope.TASK, owner="svc")

    def test_update_does_not_count_toward_capacity(self) -> None:
        store = SecretStore()
        store.MAX_SECRETS_PER_SCOPE = 2
        store.put("k1", "v1", scope=SecretScope.TASK, owner="svc")
        store.put("k2", "v2", scope=SecretScope.TASK, owner="svc")
        # Update existing — should NOT fail
        store.put("k1", "v1-updated", scope=SecretScope.TASK, owner="svc")
        assert store.get("k1", scope=SecretScope.TASK) == "v1-updated"

    def test_value_too_long_rejected(self) -> None:
        store = SecretStore()
        with pytest.raises(SecretValueTooLarge, match="maximum length"):
            store.put("k", "x" * 70_000, scope=SecretScope.TASK, owner="svc")

    def test_value_too_long_bytes_not_chars(self) -> None:
        """Byte length is enforced, not character count. Multi-byte chars
        must be correctly measured against the 64KB limit."""
        store = SecretStore()
        # 4-byte emoji × 16385 = 65540 bytes > 64KB, but only 16385 chars
        value = "\U0001f600" * 16385
        assert len(value) < store.MAX_VALUE_LENGTH  # chars < limit
        assert len(value.encode("utf-8")) > store.MAX_VALUE_LENGTH  # bytes > limit
        with pytest.raises(SecretValueTooLarge, match="bytes"):
            store.put("k", value, scope=SecretScope.TASK, owner="svc")

    def test_lazy_expiry_on_get(self) -> None:
        store = SecretStore()
        past = time.time() - 10
        store.put("k", "v", scope=SecretScope.TASK, owner="svc", expires_at=past)
        assert store.get("k", scope=SecretScope.TASK) is None

    def test_lazy_expiry_on_list(self) -> None:
        store = SecretStore()
        past = time.time() - 10
        future = time.time() + 3600
        store.put("expired", "v", scope=SecretScope.TASK, owner="svc", expires_at=past)
        store.put("alive", "v", scope=SecretScope.TASK, owner="svc", expires_at=future)
        keys = store.list_keys(scope=SecretScope.TASK)
        assert keys == ["alive"]

    def test_thread_safety(self) -> None:
        store = SecretStore()
        errors: list[Exception] = []

        def write_batch(prefix: str) -> None:
            try:
                for i in range(50):
                    store.put(f"{prefix}-{i}", f"val-{i}", scope=SecretScope.TASK, owner="svc")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_batch, args=(f"t{n}",)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.count(scope=SecretScope.TASK) == 200

    def test_encryption_at_rest(self) -> None:
        """Stored values are encrypted — raw access doesn't reveal plaintext."""
        store = SecretStore()
        store.put("secret-key", "super-secret-password", scope=SecretScope.TASK, owner="svc")
        with store._lock:
            entry = store._store[SecretScope.TASK]["secret-key"]
        assert b"super-secret-password" not in entry.encrypted_value


# ═══════════════════════════════════════════════════════════════════════
# SecretBroker — Capability-Based Access Control
# ═══════════════════════════════════════════════════════════════════════


class TestSecretBrokerCapability:
    """SecretBroker enforces SecretCapability from leases."""

    def _t2_capability(self) -> SecretCapability:
        """T2-equivalent: 3 secrets, task scope only."""
        return SecretCapability(
            allowed_scopes=["task"],
            max_secrets=3,
        )

    def _t3_capability(self) -> SecretCapability:
        """T3-equivalent: 10 secrets, task + session scopes."""
        return SecretCapability(
            allowed_scopes=["task", "session"],
            max_secrets=10,
        )

    def _t4_capability(self) -> SecretCapability:
        """T4-equivalent: 50 secrets, all scopes."""
        return SecretCapability(
            allowed_scopes=["task", "session", "global"],
            max_secrets=50,
        )

    def _no_access_capability(self) -> SecretCapability:
        """T0/T1-equivalent: no secret access."""
        return SecretCapability(
            allowed_scopes=[],
            max_secrets=0,
        )

    @pytest.mark.asyncio
    async def test_get_with_valid_capability(self) -> None:
        broker = SecretBroker()
        cap = self._t2_capability()
        await broker.put_secret("api-key", "secret", capability=cap)
        result = await broker.get_secret("api-key", capability=cap)
        assert result == "secret"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self) -> None:
        broker = SecretBroker()
        cap = self._t2_capability()
        result = await broker.get_secret("nope", capability=cap)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_access_denied(self) -> None:
        broker = SecretBroker()
        cap = self._no_access_capability()
        with pytest.raises(SecretAccessDenied, match="no secret access"):
            await broker.get_secret("key", capability=cap)

    @pytest.mark.asyncio
    async def test_scope_denied(self) -> None:
        broker = SecretBroker()
        cap = self._t2_capability()  # task only
        with pytest.raises(SecretAccessDenied, match="not in allowed scopes"):
            await broker.get_secret("key", scope="session", capability=cap)

    @pytest.mark.asyncio
    async def test_t3_can_access_session(self) -> None:
        broker = SecretBroker()
        cap = self._t3_capability()
        await broker.put_secret("k", "v", scope="session", capability=cap)
        assert await broker.get_secret("k", scope="session", capability=cap) == "v"

    @pytest.mark.asyncio
    async def test_t3_cannot_access_global(self) -> None:
        broker = SecretBroker()
        cap = self._t3_capability()
        with pytest.raises(SecretAccessDenied, match="not in allowed scopes"):
            await broker.get_secret("key", scope="global", capability=cap)

    @pytest.mark.asyncio
    async def test_t4_can_access_all_scopes(self) -> None:
        broker = SecretBroker()
        cap = self._t4_capability()
        for scope in ["task", "session", "global"]:
            await broker.put_secret(f"k-{scope}", "v", scope=scope, capability=cap)
            assert await broker.get_secret(f"k-{scope}", scope=scope, capability=cap) == "v"

    @pytest.mark.asyncio
    async def test_allowed_keys_enforced(self) -> None:
        cap = SecretCapability(
            allowed_scopes=["task"],
            allowed_keys=["db-pass", "api-key"],
            max_secrets=5,
        )
        broker = SecretBroker()
        await broker.put_secret("db-pass", "secret", capability=cap)
        with pytest.raises(SecretAccessDenied, match="not in allowed keys"):
            await broker.get_secret("other-key", capability=cap)

    @pytest.mark.asyncio
    async def test_max_secrets_enforced(self) -> None:
        cap = SecretCapability(
            allowed_scopes=["task"],
            max_secrets=2,
        )
        broker = SecretBroker()
        await broker.put_secret("k1", "v1", capability=cap)
        await broker.put_secret("k2", "v2", capability=cap)
        with pytest.raises(SecretAccessDenied, match="max_secrets"):
            await broker.put_secret("k3", "v3", capability=cap)

    @pytest.mark.asyncio
    async def test_max_secrets_per_owner_not_scope(self) -> None:
        """max_secrets counts per-owner, not scope-wide. Another owner's
        secrets must not block a different caller's writes."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=2)
        broker_a = SecretBroker(store=store, default_capability=cap, owner="owner-a")
        broker_b = SecretBroker(store=store, default_capability=cap, owner="owner-b")
        # Owner A fills their quota
        await broker_a.put_secret("a1", "v")
        await broker_a.put_secret("a2", "v")
        # Owner B should NOT be blocked by owner A's secrets
        await broker_b.put_secret("b1", "v")
        await broker_b.put_secret("b2", "v")
        assert store.count(scope=SecretScope.TASK) == 4

    @pytest.mark.asyncio
    async def test_expired_secrets_dont_block_writes(self) -> None:
        """Expired entries are purged on write so they don't ghost-fill the scope."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=2)
        broker = SecretBroker(store=store)
        past = time.time() - 10
        await broker.put_secret("old1", "v", capability=cap, expires_at=past)
        await broker.put_secret("old2", "v", capability=cap, expires_at=past)
        # Both are expired — new writes should succeed after purge
        await broker.put_secret("new1", "v", capability=cap)
        await broker.put_secret("new2", "v", capability=cap)
        assert await broker.get_secret("new1", capability=cap) == "v"

    @pytest.mark.asyncio
    async def test_expired_key_resurrection_blocked(self) -> None:
        """Overwriting an expired key must count as a new insert for cap_limit.
        Prevents bypassing max_secrets by 'updating' expired keys back to life."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=1)
        broker = SecretBroker(store=store)
        past = time.time() - 10
        # Fill quota then let it expire
        await broker.put_secret("k1", "v1", capability=cap, expires_at=past)
        # Write a new live key — uses the freed slot
        await broker.put_secret("k2", "live", capability=cap)
        # Attempting to "resurrect" expired k1 must be denied (quota full)
        with pytest.raises(SecretAccessDenied, match="max_secrets"):
            await broker.put_secret("k1", "revived", capability=cap)

    @pytest.mark.asyncio
    async def test_update_existing_does_not_hit_limit(self) -> None:
        cap = SecretCapability(
            allowed_scopes=["task"],
            max_secrets=1,
        )
        broker = SecretBroker()
        await broker.put_secret("k1", "v1", capability=cap)
        # Update should work even at limit
        await broker.put_secret("k1", "v2", capability=cap)
        assert await broker.get_secret("k1", capability=cap) == "v2"

    @pytest.mark.asyncio
    async def test_invalid_scope_rejected(self) -> None:
        broker = SecretBroker()
        cap = self._t2_capability()
        with pytest.raises(SecretAccessDenied, match="Invalid scope"):
            await broker.get_secret("key", scope="invalid", capability=cap)


# ═══════════════════════════════════════════════════════════════════════
# SecretBroker — Revocation and Listing
# ═══════════════════════════════════════════════════════════════════════


class TestSecretBrokerOperations:
    """SecretBroker revoke and list operations."""

    def _cap(self) -> SecretCapability:
        return SecretCapability(
            allowed_scopes=["task", "session"],
            max_secrets=10,
        )

    @pytest.mark.asyncio
    async def test_revoke_existing(self) -> None:
        broker = SecretBroker()
        cap = self._cap()
        await broker.put_secret("k", "v", capability=cap)
        assert await broker.revoke("k", capability=cap) is True
        assert await broker.get_secret("k", capability=cap) is None

    @pytest.mark.asyncio
    async def test_revoke_nonexistent(self) -> None:
        broker = SecretBroker()
        cap = self._cap()
        assert await broker.revoke("nope", capability=cap) is False

    @pytest.mark.asyncio
    async def test_list_available(self) -> None:
        broker = SecretBroker()
        cap = self._cap()
        await broker.put_secret("b", "v", capability=cap)
        await broker.put_secret("a", "v", capability=cap)
        keys = broker.list_available(capability=cap)
        assert keys == ["a", "b"]

    @pytest.mark.asyncio
    async def test_list_filtered_by_allowed_keys(self) -> None:
        store = SecretStore()
        store.put("visible", "v", scope=SecretScope.TASK, owner="svc")
        store.put("hidden", "v", scope=SecretScope.TASK, owner="svc")
        cap = SecretCapability(
            allowed_scopes=["task"],
            allowed_keys=["visible"],
            max_secrets=5,
        )
        broker = SecretBroker(store=store, owner="svc")
        keys = broker.list_available(capability=cap)
        assert keys == ["visible"]

    @pytest.mark.asyncio
    async def test_list_empty_scope(self) -> None:
        broker = SecretBroker()
        cap = self._cap()
        assert broker.list_available(capability=cap) == []

    @pytest.mark.asyncio
    async def test_revoke_denied_without_scope(self) -> None:
        broker = SecretBroker()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        with pytest.raises(SecretAccessDenied):
            await broker.revoke("k", scope="session", capability=cap)


# ═══════════════════════════════════════════════════════════════════════
# SecretBroker — Default Capability
# ═══════════════════════════════════════════════════════════════════════


class TestSecretBrokerDefaults:
    """SecretBroker with default_capability."""

    @pytest.mark.asyncio
    async def test_uses_default_capability(self) -> None:
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker = SecretBroker(default_capability=cap)
        await broker.put_secret("k", "v")
        assert await broker.get_secret("k") == "v"

    @pytest.mark.asyncio
    async def test_default_zero_denies(self) -> None:
        broker = SecretBroker()  # default SecretCapability has max_secrets=0
        with pytest.raises(SecretAccessDenied, match="no secret access"):
            await broker.get_secret("k")


# ═══════════════════════════════════════════════════════════════════════
# Security Regression Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSecretBrokerSecurity:
    """Security invariants for the secret broker."""

    @pytest.mark.asyncio
    async def test_t0_t1_cannot_access_secrets(self) -> None:
        """T0/T1 equivalent capabilities deny all access."""
        broker = SecretBroker()
        for cap in [
            SecretCapability(allowed_scopes=[], max_secrets=0),
            SecretCapability(allowed_scopes=["task"], max_secrets=0),
        ]:
            with pytest.raises(SecretAccessDenied):
                await broker.get_secret("key", capability=cap)

    @pytest.mark.asyncio
    async def test_scope_escalation_prevented(self) -> None:
        """T2 (task-only) cannot read session secrets."""
        store = SecretStore()
        store.put("session-secret", "classified", scope=SecretScope.SESSION, owner="admin")
        t2_cap = SecretCapability(
            allowed_scopes=["task"],
            max_secrets=3,
        )
        broker = SecretBroker(store=store)
        with pytest.raises(SecretAccessDenied, match="not in allowed scopes"):
            await broker.get_secret("session-secret", scope="session", capability=t2_cap)

    @pytest.mark.asyncio
    async def test_key_restriction_enforced(self) -> None:
        """Allowed_keys list is a hard deny for unlisted keys."""
        cap = SecretCapability(
            allowed_scopes=["task"],
            allowed_keys=["safe-key"],
            max_secrets=5,
        )
        broker = SecretBroker()
        with pytest.raises(SecretAccessDenied, match="not in allowed keys"):
            await broker.put_secret("other-key", "v", capability=cap)

    @pytest.mark.asyncio
    async def test_encrypted_at_rest_via_broker(self) -> None:
        """Values stored through broker are encrypted in the store."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker = SecretBroker(store=store, default_capability=cap)
        await broker.put_secret("api-key", "super-secret-123")
        # Direct store access — value should be encrypted
        with store._lock:
            entry = store._store[SecretScope.TASK]["api-key"]
        assert b"super-secret-123" not in entry.encrypted_value
        # But broker decrypts it
        assert await broker.get_secret("api-key") == "super-secret-123"

    @pytest.mark.asyncio
    async def test_deny_before_allow(self) -> None:
        """Zero max_secrets denies even if scopes match."""
        cap = SecretCapability(
            allowed_scopes=["task", "session", "global"],
            max_secrets=0,
        )
        broker = SecretBroker()
        with pytest.raises(SecretAccessDenied, match="no secret access"):
            await broker.get_secret("key", capability=cap)

    def test_secret_store_values_not_in_repr(self) -> None:
        """SecretEntry encrypted_value should not leak plaintext."""
        store = SecretStore()
        store.put("key", "password123", scope=SecretScope.TASK, owner="svc")
        with store._lock:
            entry = store._store[SecretScope.TASK]["key"]
        r = repr(entry)
        assert "password123" not in r

    @pytest.mark.asyncio
    async def test_expired_secret_not_accessible(self) -> None:
        """Expired secrets return None even through broker."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker = SecretBroker(store=store, default_capability=cap)
        past = time.time() - 10
        await broker.put_secret("expired", "v", expires_at=past)
        assert await broker.get_secret("expired") is None

    def test_ciphertext_integrity_check(self) -> None:
        """Tampered ciphertext is detected by HMAC integrity tag."""
        enc = _SecretEncryptor(b"test-key-32bytes" * 2)
        encrypted = enc.encrypt("sensitive-data")
        # Flip a byte in the ciphertext region (after 16-byte nonce, before 32-byte tag)
        tampered = bytearray(encrypted)
        tampered[20] ^= 0xFF
        with pytest.raises(SecretBrokerError, match="integrity check failed"):
            enc.decrypt(bytes(tampered))

    def test_ciphertext_truncation_detected(self) -> None:
        """Truncated ciphertext is rejected."""
        enc = _SecretEncryptor(b"test-key-32bytes" * 2)
        with pytest.raises(SecretBrokerError, match="Corrupt"):
            enc.decrypt(b"\x00" * 16)  # nonce only, no ciphertext or tag

    @pytest.mark.asyncio
    async def test_concurrent_put_respects_cap_limit(self) -> None:
        """Concurrent writers must not exceed capability max_secrets (TOCTOU fix)."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker = SecretBroker(store=store, default_capability=cap)
        errors: list[Exception] = []
        success_count = 0
        lock = threading.Lock()

        import asyncio

        async def write(i: int) -> None:
            nonlocal success_count
            try:
                await broker.put_secret(
                    f"key-{i}",
                    f"val-{i}",
                )
                with lock:
                    success_count += 1
            except SecretAccessDenied:
                pass  # Expected once limit is reached
            except Exception as e:
                with lock:
                    errors.append(e)

        # Attempt 20 concurrent writes with cap of 5
        await asyncio.gather(*(write(i) for i in range(20)))
        assert not errors, f"Unexpected errors: {errors}"
        # Must not exceed cap
        assert store.count(scope=SecretScope.TASK) <= 5

    @pytest.mark.asyncio
    async def test_cross_owner_read_isolation(self) -> None:
        """Owner B cannot read owner A's secrets."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker_a = SecretBroker(store=store, default_capability=cap, owner="alice")
        broker_b = SecretBroker(store=store, default_capability=cap, owner="bob")
        await broker_a.put_secret("secret-key", "alice-secret")
        # Alice can read her own secret
        assert await broker_a.get_secret("secret-key") == "alice-secret"
        # Bob cannot read Alice's secret
        assert await broker_b.get_secret("secret-key") is None

    @pytest.mark.asyncio
    async def test_cross_owner_overwrite_blocked(self) -> None:
        """Owner B cannot overwrite owner A's secret."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker_a = SecretBroker(store=store, default_capability=cap, owner="alice")
        broker_b = SecretBroker(store=store, default_capability=cap, owner="bob")
        await broker_a.put_secret("shared-name", "alice-value")
        with pytest.raises(SecretAccessDenied, match="owned by"):
            await broker_b.put_secret("shared-name", "bob-takeover")
        # Alice's value is unchanged
        assert await broker_a.get_secret("shared-name") == "alice-value"

    @pytest.mark.asyncio
    async def test_cross_owner_revoke_blocked(self) -> None:
        """Owner B cannot revoke owner A's secret."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker_a = SecretBroker(store=store, default_capability=cap, owner="alice")
        broker_b = SecretBroker(store=store, default_capability=cap, owner="bob")
        await broker_a.put_secret("protected", "value")
        # Bob's revoke returns False (not his secret)
        assert await broker_b.revoke("protected") is False
        # Alice's secret still exists
        assert await broker_a.get_secret("protected") == "value"

    @pytest.mark.asyncio
    async def test_cross_owner_list_isolation(self) -> None:
        """Owners only see their own secrets in list_available."""
        store = SecretStore()
        cap = SecretCapability(allowed_scopes=["task"], max_secrets=5)
        broker_a = SecretBroker(store=store, default_capability=cap, owner="alice")
        broker_b = SecretBroker(store=store, default_capability=cap, owner="bob")
        await broker_a.put_secret("alice-key", "v")
        await broker_b.put_secret("bob-key", "v")
        assert broker_a.list_available() == ["alice-key"]
        assert broker_b.list_available() == ["bob-key"]
