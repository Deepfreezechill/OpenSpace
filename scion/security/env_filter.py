"""Environment variable filtering for sandbox execution.

Provides an explicit allowlist of environment variables that are safe to
pass into an E2B sandbox.  Everything else — API keys, tokens, database
URLs, credentials — is stripped so that skill code running in the sandbox
cannot exfiltrate host secrets.
"""

from __future__ import annotations

import os
import re
from typing import Dict, FrozenSet

__all__ = ["get_safe_env", "is_sensitive_key", "ENV_ALLOWLIST"]

# ---------------------------------------------------------------------------
# Allowlist — the ONLY env vars that may reach the sandbox
# ---------------------------------------------------------------------------

ENV_ALLOWLIST: FrozenSet[str] = frozenset(
    {
        "SCION_LOG_LEVEL",
        "PATH",
        "HOME",
        "LANG",
    }
)

# ---------------------------------------------------------------------------
# Sensitive-key heuristic
# ---------------------------------------------------------------------------

_SENSITIVE_FRAGMENTS = re.compile(
    r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH|PRIVATE"
    r"|DATABASE_URL|DB_URL|CONNECTION_STRING"
    r"|API_KEY|ACCESS_KEY|SIGNING"
    r"|REDIS_URL|MONGO_URL|POSTGRES_URL|MYSQL_URL"
    r"|CELERY_BROKER|SQLALCHEMY"
    r"|DSN|SENTRY_DSN",
    re.IGNORECASE,
)

# Some allowlisted keys contain sensitive fragments.
# They are explicitly permitted and must not be flagged.
_SENSITIVE_ALLOWLIST: FrozenSet[str] = frozenset()


def is_sensitive_key(key: str) -> bool:
    """Return ``True`` if *key* looks like it holds a secret.

    Uses a regex heuristic that matches common naming conventions for
    tokens, passwords, API keys, database URLs, etc.
    """
    if key in _SENSITIVE_ALLOWLIST:
        return False
    return bool(_SENSITIVE_FRAGMENTS.search(key))


def get_safe_env() -> Dict[str, str]:
    """Return a dict containing **only** allowlisted env vars.

    Any variable whose name is not in :data:`ENV_ALLOWLIST` is dropped.
    This is the environment dict that should be forwarded to the E2B
    sandbox — nothing else.
    """
    return {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
