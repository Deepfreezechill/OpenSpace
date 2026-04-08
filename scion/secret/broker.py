"""Concrete SecretBrokerPort implementation — EPIC 2.6.

Provides:
- Scoped secret storage (task, session, global) with lease-based access control.
- Encryption at rest using HMAC-derived Fernet keys (stdlib + cryptography-free).
- Thread-safe operations with bounded storage per scope.
- Integration with SecretCapability from lease system and auth token scopes.

Design decisions:
- Fail-closed: missing capability or insufficient scope → deny.
- Encryption uses HMAC-SHA256 derived keys with XOR cipher (no external deps).
- Secrets are stored in-memory only — no persistence across restarts.
- Scope hierarchy: task < session < global (each is independent namespace).
- Revocation is immediate and irreversible within a session.

Security requirements:
- Callers MUST present a valid SecretCapability from their lease.
- SECRET_READ / SECRET_WRITE token scopes gate read/write operations.
- T0/T1 tiers cannot access secrets (enforced by lease validation).
- Secret values are encrypted at rest in memory to resist heap inspection.

Issues:
- #52: SecretBrokerPort concrete implementation
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from scion.sandbox.leases import SecretCapability

# ---------------------------------------------------------------------------
# Secret Scope Model
# ---------------------------------------------------------------------------


class SecretScope(str, Enum):
    """Hierarchical secret scopes matching lease capability model."""

    TASK = "task"
    SESSION = "session"
    GLOBAL = "global"


# Scope hierarchy for validation (higher index = broader access)
_SCOPE_ORDER: list[SecretScope] = [
    SecretScope.TASK,
    SecretScope.SESSION,
    SecretScope.GLOBAL,
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecretBrokerError(Exception):
    """Base for all secret broker errors."""


class SecretAccessDenied(SecretBrokerError):
    """Caller lacks permission to access the requested secret."""


class SecretNotFoundError(SecretBrokerError):
    """Requested secret does not exist."""


class SecretStoreFull(SecretBrokerError):
    """Secret store has reached its capacity limit."""


class SecretKeyInvalid(SecretBrokerError):
    """Secret key is malformed or invalid."""


class SecretValueTooLarge(SecretBrokerError):
    """Secret value exceeds maximum allowed size."""


# ---------------------------------------------------------------------------
# At-Rest Encryption (no external crypto dependencies)
# ---------------------------------------------------------------------------


class _SecretEncryptor:
    """XOR-based at-rest encryption using HMAC-derived key stream.

    NOT a general-purpose cipher — this provides defense-in-depth against
    heap inspection only.  The real security boundary is access control
    via capabilities and token scopes.
    """

    def __init__(self, master_key: bytes) -> None:
        self._master_key = master_key

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt a secret value, returning nonce + ciphertext + HMAC tag.

        Layout: nonce (16) || ciphertext (N) || hmac_tag (32)
        Integrity is verified on decrypt (encrypt-then-MAC).
        """
        nonce = os.urandom(16)
        plaintext_bytes = plaintext.encode("utf-8")
        key_stream = self._derive_stream(nonce, len(plaintext_bytes))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext_bytes, key_stream))
        # Encrypt-then-MAC: HMAC over nonce + ciphertext
        tag = hmac.new(
            self._master_key,
            nonce + ciphertext,
            hashlib.sha256,
        ).digest()
        return nonce + ciphertext + tag

    _TAG_LEN = 32  # HMAC-SHA256 output length

    def decrypt(self, data: bytes) -> str:
        """Decrypt nonce + ciphertext + HMAC tag back to plaintext.

        Raises SecretBrokerError if data is corrupt or tampered with.
        """
        # Minimum: 16 (nonce) + 0 (ciphertext can be empty) + 32 (tag)
        if len(data) < 16 + self._TAG_LEN:
            raise SecretBrokerError("Corrupt encrypted data")
        nonce = data[:16]
        ciphertext = data[16 : -self._TAG_LEN]
        stored_tag = data[-self._TAG_LEN :]
        # Verify integrity before decryption
        expected_tag = hmac.new(
            self._master_key,
            nonce + ciphertext,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(stored_tag, expected_tag):
            raise SecretBrokerError("Encrypted data integrity check failed")
        key_stream = self._derive_stream(nonce, len(ciphertext))
        plaintext_bytes = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
        return plaintext_bytes.decode("utf-8")

    def _derive_stream(self, nonce: bytes, length: int) -> bytes:
        """Derive a key stream of given length from nonce + master key."""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hmac.new(
                self._master_key,
                nonce + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            stream += block
            counter += 1
        return stream[:length]


# ---------------------------------------------------------------------------
# Secret Entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretEntry:
    """Metadata and encrypted value for a stored secret."""

    key: str
    scope: SecretScope
    encrypted_value: bytes
    owner: str  # subject that created the secret
    created_at: float
    expires_at: float | None = None  # None = no expiry


# ---------------------------------------------------------------------------
# Secret Store (scoped, encrypted, bounded)
# ---------------------------------------------------------------------------

_MAX_KEY_LENGTH = 256
_KEY_PATTERN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./:")


def _validate_key(key: str) -> None:
    """Validate a secret key name."""
    if not key:
        raise SecretKeyInvalid("Secret key cannot be empty")
    if len(key) > _MAX_KEY_LENGTH:
        raise SecretKeyInvalid(f"Secret key exceeds maximum length ({_MAX_KEY_LENGTH})")
    invalid = set(key) - _KEY_PATTERN_CHARS
    if invalid:
        raise SecretKeyInvalid(f"Secret key contains invalid characters: {sorted(invalid)}")


class SecretStore:
    """Thread-safe, encrypted, scoped secret storage.

    Secrets are organized by scope (task/session/global) and encrypted
    at rest using HMAC-derived key streams.  Each scope has an
    independent namespace and capacity limit.
    """

    MAX_SECRETS_PER_SCOPE = 1000
    MAX_VALUE_LENGTH = 65_536  # 64KB per secret value

    def __init__(self, encryption_key: bytes | None = None) -> None:
        self._lock = threading.Lock()
        self._encryptor = _SecretEncryptor(encryption_key or secrets.token_bytes(32))
        # scope -> key -> SecretEntry
        self._store: dict[SecretScope, dict[str, SecretEntry]] = {scope: {} for scope in SecretScope}

    def put(
        self,
        key: str,
        value: str,
        *,
        scope: SecretScope,
        owner: str,
        expires_at: float | None = None,
    ) -> None:
        """Store or update a secret.

        Args:
            key: Secret key name.
            value: Plaintext secret value (encrypted before storage).
            scope: Secret scope namespace.
            owner: Subject identity of the caller.
            expires_at: Optional epoch expiry.

        Raises:
            SecretKeyInvalid: If key is malformed.
            SecretStoreFull: If the scope has reached capacity.
            ValueError: If value exceeds maximum length.
        """
        _validate_key(key)
        value_bytes_len = len(value.encode("utf-8"))
        if value_bytes_len > self.MAX_VALUE_LENGTH:
            raise SecretValueTooLarge(
                f"Secret value ({value_bytes_len} bytes) exceeds maximum length ({self.MAX_VALUE_LENGTH} bytes)"
            )

        encrypted = self._encryptor.encrypt(value)
        entry = SecretEntry(
            key=key,
            scope=scope,
            encrypted_value=encrypted,
            owner=owner,
            created_at=time.time(),
            expires_at=expires_at,
        )

        with self._lock:
            scope_store = self._store[scope]
            if key not in scope_store:
                # Purge expired entries before capacity check
                now = time.time()
                expired_keys = [k for k, e in scope_store.items() if e.expires_at is not None and e.expires_at <= now]
                for k in expired_keys:
                    del scope_store[k]
                if len(scope_store) >= self.MAX_SECRETS_PER_SCOPE:
                    raise SecretStoreFull(f"Scope '{scope.value}' is full ({self.MAX_SECRETS_PER_SCOPE} secrets)")
            scope_store[key] = entry

    def put_checked(
        self,
        key: str,
        value: str,
        *,
        scope: SecretScope,
        owner: str,
        expires_at: float | None = None,
        cap_limit: int,
    ) -> None:
        """Atomic put with capability-level count check.

        Same as put(), but additionally enforces a per-capability secret
        count limit *inside* the lock, eliminating TOCTOU races between
        count() and put() at the broker layer.

        Raises:
            SecretAccessDenied: If cap_limit would be exceeded for new keys.
            SecretStoreFull: If scope hard limit is reached.
        """
        _validate_key(key)
        value_bytes_len = len(value.encode("utf-8"))
        if value_bytes_len > self.MAX_VALUE_LENGTH:
            raise SecretValueTooLarge(
                f"Secret value ({value_bytes_len} bytes) exceeds maximum length ({self.MAX_VALUE_LENGTH} bytes)"
            )

        encrypted = self._encryptor.encrypt(value)
        entry = SecretEntry(
            key=key,
            scope=scope,
            encrypted_value=encrypted,
            owner=owner,
            created_at=time.time(),
            expires_at=expires_at,
        )

        with self._lock:
            scope_store = self._store[scope]
            now = time.time()
            # Purge expired entries first to avoid ghost fullness
            expired_keys = [k for k, e in scope_store.items() if e.expires_at is not None and e.expires_at <= now]
            for k in expired_keys:
                del scope_store[k]
            # After purge, expired keys are gone — is_new is accurate
            is_new = key not in scope_store
            if is_new:
                # Capability limit: count only secrets owned by this caller
                owner_count = sum(1 for e in scope_store.values() if e.owner == owner)
                if owner_count >= cap_limit:
                    raise SecretAccessDenied(f"Would exceed max_secrets limit ({cap_limit}) for scope '{scope.value}'")
                if len(scope_store) >= self.MAX_SECRETS_PER_SCOPE:
                    raise SecretStoreFull(f"Scope '{scope.value}' is full ({self.MAX_SECRETS_PER_SCOPE} secrets)")
            else:
                # Overwrite: only the owner can update their own secret
                existing = scope_store[key]
                if existing.owner != owner:
                    raise SecretAccessDenied(f"Cannot overwrite secret owned by '{existing.owner}'")
            scope_store[key] = entry

    def get(self, key: str, *, scope: SecretScope) -> str | None:
        """Retrieve and decrypt a secret value.

        Returns None if the key does not exist or has expired.
        Expired entries are lazily removed.
        """
        with self._lock:
            entry = self._store[scope].get(key)
            if entry is None:
                return None
            # Lazy expiry — consistent >= boundary (expired at exact expiry time)
            if entry.expires_at is not None and time.time() >= entry.expires_at:
                del self._store[scope][key]
                return None
            return self._encryptor.decrypt(entry.encrypted_value)

    def get_entry(self, key: str, *, scope: SecretScope) -> SecretEntry | None:
        """Retrieve a secret entry (metadata) without decrypting.

        Returns None if not found or expired.  Used by broker layer
        for owner-based access control.
        """
        with self._lock:
            entry = self._store[scope].get(key)
            if entry is None:
                return None
            if entry.expires_at is not None and time.time() >= entry.expires_at:
                del self._store[scope][key]
                return None
            return entry

    def get_owned(
        self,
        key: str,
        *,
        scope: SecretScope,
        owner: str,
    ) -> str | None:
        """Retrieve and decrypt a secret only if owned by the given owner.

        Returns None if not found, expired, or owned by someone else.
        """
        with self._lock:
            entry = self._store[scope].get(key)
            if entry is None:
                return None
            if entry.expires_at is not None and time.time() >= entry.expires_at:
                del self._store[scope][key]
                return None
            if entry.owner != owner:
                return None
            return self._encryptor.decrypt(entry.encrypted_value)

    def delete(self, key: str, *, scope: SecretScope) -> bool:
        """Delete a secret. Returns True if it existed."""
        with self._lock:
            return self._store[scope].pop(key, None) is not None

    def delete_owned(
        self,
        key: str,
        *,
        scope: SecretScope,
        owner: str,
    ) -> bool:
        """Delete a secret only if owned by the given owner."""
        with self._lock:
            entry = self._store[scope].get(key)
            if entry is None:
                return False
            if entry.owner != owner:
                return False
            del self._store[scope][key]
            return True

    def list_keys(self, *, scope: SecretScope) -> list[str]:
        """List all non-expired secret keys in a scope."""
        now = time.time()
        with self._lock:
            result = []
            expired = []
            for k, entry in self._store[scope].items():
                if entry.expires_at is not None and now >= entry.expires_at:
                    expired.append(k)
                else:
                    result.append(k)
            # Lazy cleanup
            for k in expired:
                del self._store[scope][k]
            return sorted(result)

    def list_keys_owned(self, *, scope: SecretScope, owner: str) -> list[str]:
        """List non-expired secret keys in a scope owned by the given owner."""
        now = time.time()
        with self._lock:
            result = []
            expired = []
            for k, entry in self._store[scope].items():
                if entry.expires_at is not None and now >= entry.expires_at:
                    expired.append(k)
                elif entry.owner == owner:
                    result.append(k)
            for k in expired:
                del self._store[scope][k]
            return sorted(result)

    def count(self, *, scope: SecretScope) -> int:
        """Number of non-expired secrets in a scope."""
        return len(self.list_keys(scope=scope))

    def clear_scope(self, scope: SecretScope) -> int:
        """Remove all secrets in a scope. Returns count removed."""
        with self._lock:
            count = len(self._store[scope])
            self._store[scope].clear()
            return count


# ---------------------------------------------------------------------------
# #52 — SecretBroker (concrete SecretBrokerPort implementation)
# ---------------------------------------------------------------------------


@dataclass
class SecretBroker:
    """Concrete implementation of SecretBrokerPort.

    Enforces lease-based access control via SecretCapability and
    integrates with the scoped SecretStore for encrypted storage.

    Access control layers:
    1. Capability check: caller's SecretCapability from lease
    2. Scope check: requested scope must be in capability's allowed_scopes
    3. Key check: if allowed_keys is non-empty, key must be listed
    4. Count check: caller cannot exceed max_secrets from capability
    5. Value encryption: all values encrypted at rest

    Security: ``owner`` is bound at construction by the container from the
    authenticated session identity.  It MUST NOT be caller-supplied — this
    prevents quota bypass via owner rotation.

    Args:
        store: The backing secret store (shared across brokers).
        default_capability: Fallback capability if none provided per-call.
        owner: Authenticated identity bound by the container (not caller-supplied).
    """

    store: SecretStore = field(default_factory=SecretStore)
    default_capability: SecretCapability = field(default_factory=SecretCapability)
    owner: str = "system"

    def _resolve_scope(self, scope: str) -> SecretScope:
        """Parse and validate a scope string."""
        try:
            return SecretScope(scope)
        except ValueError:
            raise SecretAccessDenied(
                f"Invalid scope '{scope}'. Must be one of: {', '.join(s.value for s in SecretScope)}"
            )

    def _check_capability(
        self,
        capability: SecretCapability,
        *,
        scope: str,
        key: str | None = None,
        writing: bool = False,
    ) -> SecretScope:
        """Validate access against a SecretCapability.

        Returns the validated SecretScope.

        Raises:
            SecretAccessDenied: If capability doesn't allow the operation.
        """
        # 1. Max secrets check (0 = no access at all)
        if capability.max_secrets <= 0:
            raise SecretAccessDenied("Capability grants no secret access")

        # 2. Scope check
        resolved_scope = self._resolve_scope(scope)
        if scope not in capability.allowed_scopes:
            raise SecretAccessDenied(f"Scope '{scope}' not in allowed scopes: {capability.allowed_scopes}")

        # 3. Key check — empty allowed_keys = unrestricted (all tiers use
        #    empty by default; non-empty means explicit whitelist)
        if key is not None and capability.allowed_keys:
            if key not in capability.allowed_keys:
                raise SecretAccessDenied(f"Key '{key}' not in allowed keys")

        return resolved_scope

    # --- SecretBrokerPort interface ---

    async def get_secret(
        self,
        key: str,
        *,
        scope: str = "task",
        capability: SecretCapability | None = None,
    ) -> Optional[str]:
        """Retrieve a secret value.

        Args:
            key: Secret key to retrieve.
            scope: Secret scope namespace.
            capability: Caller's lease capability (uses default if None).

        Returns:
            Decrypted secret value, or None if not found.

        Raises:
            SecretAccessDenied: If capability doesn't permit access.
            SecretKeyInvalid: If key is malformed.
        """
        _validate_key(key)
        cap = capability or self.default_capability
        resolved = self._check_capability(cap, scope=scope, key=key)
        return self.store.get_owned(key, scope=resolved, owner=self.owner)

    async def put_secret(
        self,
        key: str,
        value: str,
        *,
        scope: str = "task",
        capability: SecretCapability | None = None,
        expires_at: float | None = None,
    ) -> None:
        """Store a secret value.

        Args:
            key: Secret key name.
            value: Plaintext value to encrypt and store.
            scope: Secret scope namespace.
            capability: Caller's lease capability.
            expires_at: Optional epoch expiry for the secret.

        Raises:
            SecretAccessDenied: If capability doesn't permit write.
            SecretKeyInvalid: If key is malformed.
            SecretStoreFull: If scope is at capacity.
        """
        _validate_key(key)
        cap = capability or self.default_capability
        resolved = self._check_capability(
            cap,
            scope=scope,
            key=key,
            writing=True,
        )

        # Atomic put with capability count check inside the lock
        # owner is bound at construction — not caller-supplied
        self.store.put_checked(
            key,
            value,
            scope=resolved,
            owner=self.owner,
            expires_at=expires_at,
            cap_limit=cap.max_secrets,
        )

    async def revoke(
        self,
        key: str,
        *,
        scope: str = "task",
        capability: SecretCapability | None = None,
    ) -> bool:
        """Revoke (delete) a secret.

        Args:
            key: Secret key to revoke.
            scope: Secret scope namespace.
            capability: Caller's lease capability.

        Returns:
            True if the secret existed and was deleted.
        """
        _validate_key(key)
        cap = capability or self.default_capability
        resolved = self._check_capability(cap, scope=scope, key=key)
        return self.store.delete_owned(key, scope=resolved, owner=self.owner)

    def list_available(
        self,
        *,
        scope: str = "task",
        capability: SecretCapability | None = None,
    ) -> list[str]:
        """List available secret keys in a scope.

        Args:
            scope: Secret scope namespace.
            capability: Caller's lease capability.

        Returns:
            Sorted list of accessible key names.
        """
        cap = capability or self.default_capability
        resolved = self._check_capability(cap, scope=scope)

        all_keys = self.store.list_keys_owned(scope=resolved, owner=self.owner)

        # Filter to allowed_keys if set
        if cap.allowed_keys:
            allowed = set(cap.allowed_keys)
            return [k for k in all_keys if k in allowed]

        return all_keys
