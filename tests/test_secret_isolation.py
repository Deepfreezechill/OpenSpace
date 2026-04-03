"""Tests for EPIC 0.3b — Secret Isolation from Sandbox.

Covers:
  - Issue #13: get_safe_env() allowlist, is_sensitive_key() heuristic
  - Issue #14: AST blocklist severity upgrades + os.environ.get
  - Issue #15: End-to-end sandbox secret isolation
"""

from __future__ import annotations

import os
import textwrap

import pytest

from scion.security import check_code_safety
from scion.security.ast_scanner import (
    Severity,
    load_blocklist,
    scan_code,
)
from scion.security.env_filter import (
    ENV_ALLOWLIST,
    get_safe_env,
    is_sensitive_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _names(findings):
    return {f.pattern_name for f in findings}


def _severities_for(findings, pattern_name):
    return {f.severity for f in findings if f.pattern_name == pattern_name}


# ---------------------------------------------------------------------------
# Issue #13 — get_safe_env() allowlist
# ---------------------------------------------------------------------------


class TestGetSafeEnv:
    """get_safe_env() must return ONLY allowlisted variables."""

    def test_returns_only_allowlisted_keys(self, mock_env):
        mock_env.set("PATH", "/usr/bin")
        mock_env.set("HOME", "/home/test")
        mock_env.set("LANG", "en_US.UTF-8")
        mock_env.set("OPENSPACE_LOG_LEVEL", "DEBUG")
        mock_env.set("AWS_SECRET_ACCESS_KEY", "supersecret")
        mock_env.set("DATABASE_URL", "postgres://...")

        safe = get_safe_env()

        assert set(safe.keys()) <= ENV_ALLOWLIST
        assert "AWS_SECRET_ACCESS_KEY" not in safe
        assert "DATABASE_URL" not in safe

    def test_all_allowlist_vars_present_when_set(self, mock_env):
        for key in ENV_ALLOWLIST:
            mock_env.set(key, f"val-{key}")

        safe = get_safe_env()
        for key in ENV_ALLOWLIST:
            assert key in safe
            assert safe[key] == f"val-{key}"

    def test_missing_allowlist_var_is_omitted(self, mock_env):
        mock_env.delete("OPENSPACE_LOG_LEVEL")
        safe = get_safe_env()
        assert "OPENSPACE_LOG_LEVEL" not in safe

    @pytest.mark.parametrize(
        "dangerous_key",
        [
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "AZURE_CLIENT_SECRET",
            "DATABASE_URL",
            "DB_PASSWORD",
            "PRIVATE_KEY",
            "GH_TOKEN",
            "ANTHROPIC_API_KEY",
            "SLACK_BOT_TOKEN",
            "STRIPE_SECRET_KEY",
            "SENDGRID_API_KEY",
            "JWT_SECRET",
            "SESSION_SECRET",
            "SIGNING_KEY",
        ],
    )
    def test_dangerous_vars_stripped(self, mock_env, dangerous_key):
        mock_env.set(dangerous_key, "secret-value")
        safe = get_safe_env()
        assert dangerous_key not in safe

    def test_returns_dict_type(self, mock_env):
        safe = get_safe_env()
        assert isinstance(safe, dict)

    def test_empty_env_returns_empty(self, mock_env):
        for key in list(os.environ.keys()):
            mock_env.delete(key)
        safe = get_safe_env()
        assert safe == {}


# ---------------------------------------------------------------------------
# Issue #13 — is_sensitive_key() heuristic
# ---------------------------------------------------------------------------


class TestIsSensitiveKey:
    """Heuristic must catch common sensitive variable naming patterns."""

    @pytest.mark.parametrize(
        "key",
        [
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "DB_URL",
            "DB_PASSWORD",
            "AZURE_CLIENT_SECRET",
            "PRIVATE_KEY",
            "GH_AUTH_TOKEN",
            "MY_CREDENTIAL",
            "SIGNING_KEY",
            "CONNECTION_STRING",
            "ACCESS_KEY_ID",
        ],
    )
    def test_detects_sensitive_keys(self, key):
        assert is_sensitive_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "PATH",
            "HOME",
            "LANG",
            "SHELL",
            "TERM",
            "USER",
            "HOSTNAME",
            "OPENSPACE_LOG_LEVEL",
            "PYTHONPATH",
            "NODE_ENV",
        ],
    )
    def test_safe_keys_not_flagged(self, key):
        assert is_sensitive_key(key) is False

    def test_e2b_api_key_flagged_as_sensitive(self):
        """E2B_API_KEY contains 'KEY' and is now correctly flagged as sensitive."""
        assert is_sensitive_key("E2B_API_KEY") is True

    def test_case_insensitive(self):
        assert is_sensitive_key("my_secret_var") is True
        assert is_sensitive_key("Github_Token") is True


# ---------------------------------------------------------------------------
# Issue #14 — AST blocklist severity upgrades
# ---------------------------------------------------------------------------


class TestBlocklistSeverityUpgrades:
    """os.environ and os.getenv must be HIGH; os.environ.get must be present."""

    def test_env_access_is_high(self):
        findings = scan_code("import os\nos.environ")
        hits = [f for f in findings if f.pattern_name == "env_access"]
        assert len(hits) >= 1
        assert all(f.severity == Severity.HIGH for f in hits)

    def test_env_getenv_is_high(self):
        findings = scan_code("import os\nos.getenv('HOME')")
        hits = [f for f in findings if f.pattern_name == "env_getenv"]
        assert len(hits) >= 1
        assert all(f.severity == Severity.HIGH for f in hits)

    def test_env_environ_get_detected(self):
        findings = scan_code("import os\nos.environ.get('HOME')")
        hits = [f for f in findings if f.pattern_name == "env_environ_get"]
        assert len(hits) >= 1
        assert all(f.severity == Severity.HIGH for f in hits)

    def test_blocklist_has_environ_get_pattern(self):
        patterns = load_blocklist()
        names = {p.name for p in patterns}
        assert "env_environ_get" in names

    def test_all_env_patterns_are_high(self):
        patterns = load_blocklist()
        env_patterns = [p for p in patterns if p.name.startswith("env_")]
        assert len(env_patterns) >= 3
        for p in env_patterns:
            assert p.severity == Severity.HIGH, f"Pattern {p.name!r} should be HIGH, got {p.severity}"


# ---------------------------------------------------------------------------
# Issue #14 — check_code_safety with upgraded severity
# ---------------------------------------------------------------------------


class TestCheckCodeSafetyEnv:
    """HIGH env findings warn but do not block execution."""

    def test_env_access_does_not_block(self):
        is_safe, findings = check_code_safety("import os\nos.environ")
        assert is_safe is True
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_environ_get_does_not_block(self):
        is_safe, findings = check_code_safety("import os\nos.environ.get('HOME')")
        assert is_safe is True
        assert any(f.pattern_name == "env_environ_get" for f in findings)

    def test_getenv_does_not_block(self):
        is_safe, findings = check_code_safety("import os\nos.getenv('HOME')")
        assert is_safe is True
        assert any(f.pattern_name == "env_getenv" for f in findings)


# ---------------------------------------------------------------------------
# Issue #15 — End-to-end sandbox isolation
# ---------------------------------------------------------------------------


class TestSandboxSecretIsolation:
    """Verify that a realistic sandbox scenario strips all host secrets."""

    def test_realistic_host_env_stripped(self, mock_env):
        """Simulate a developer machine with many secrets."""
        mock_env.set("PATH", "/usr/local/bin:/usr/bin")
        mock_env.set("HOME", "/home/dev")
        mock_env.set("LANG", "en_US.UTF-8")
        mock_env.set("OPENSPACE_LOG_LEVEL", "INFO")
        # Secrets that MUST NOT leak
        mock_env.set("E2B_API_KEY", "e2b-test-key")
        mock_env.set("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG")
        mock_env.set("GITHUB_TOKEN", "ghp_xxxxxxxxxxxx")
        mock_env.set("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxx")
        mock_env.set("DATABASE_URL", "postgres://user:pass@host/db")
        mock_env.set("STRIPE_SECRET_KEY", "sk_live_xxxxxxxx")

        safe = get_safe_env()

        # Allowlisted vars present
        assert safe["PATH"] == "/usr/local/bin:/usr/bin"
        assert safe["HOME"] == "/home/dev"

        # All secrets stripped (including E2B_API_KEY — host-side only)
        for key in (
            "E2B_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "STRIPE_SECRET_KEY",
        ):
            assert key not in safe, f"{key} leaked into sandbox env!"

    def test_ast_scanner_flags_env_access_as_high(self):
        """Skill code that reads env vars should be flagged at HIGH severity."""
        code = textwrap.dedent("""\
            import os
            secret = os.environ.get("AWS_SECRET_KEY", "")
            token = os.getenv("GITHUB_TOKEN")
            all_env = os.environ
        """)
        findings = scan_code(code)
        env_findings = [f for f in findings if f.pattern_name in ("env_access", "env_getenv", "env_environ_get")]
        assert len(env_findings) >= 3
        assert all(f.severity == Severity.HIGH for f in env_findings)

    def test_proc_environ_flagged(self):
        """Reading /proc/self/environ should be flagged."""
        code = "open('/proc/self/environ')"
        findings = scan_code(code)
        assert len(findings) >= 1
        assert any(f.pattern_name == "sensitive_file_open" for f in findings)

    def test_allowlist_is_frozen(self):
        """ENV_ALLOWLIST must be immutable."""
        with pytest.raises(AttributeError):
            ENV_ALLOWLIST.add("HACK")  # type: ignore[attr-defined]

    def test_allowlist_exact_members(self):
        """Lock down the exact allowlist to prevent accidental expansion."""
        assert ENV_ALLOWLIST == frozenset(
            {
                "OPENSPACE_LOG_LEVEL",
                "PATH",
                "HOME",
                "LANG",
            }
        )

    def test_sandbox_connector_filters_env(self):
        """SandboxConnector must strip secrets from env before passing to sandbox."""
        from unittest.mock import MagicMock

        from scion.grounding.backends.mcp.transport.connectors.sandbox import SandboxConnector

        mock_sandbox = MagicMock()
        hostile_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "AWS_SECRET_ACCESS_KEY": "AKIAIOSFODNN7EXAMPLE",
            "GITHUB_TOKEN": "ghp_xxxx",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "OPENSPACE_LOG_LEVEL": "DEBUG",
        }

        connector = SandboxConnector(
            sandbox=mock_sandbox,
            command="python server.py",
            args=[],
            env=hostile_env,
        )

        assert "PATH" in connector.user_env
        assert "HOME" in connector.user_env
        assert "OPENSPACE_LOG_LEVEL" in connector.user_env
        assert "AWS_SECRET_ACCESS_KEY" not in connector.user_env
        assert "GITHUB_TOKEN" not in connector.user_env
        assert "DATABASE_URL" not in connector.user_env

    def test_heuristic_catches_url_secrets(self):
        """is_sensitive_key() must flag URL-based connection strings."""
        url_secrets = [
            "REDIS_URL",
            "MONGO_URL",
            "POSTGRES_URL",
            "MYSQL_URL",
            "CELERY_BROKER_URL",
            "SQLALCHEMY_DATABASE_URI",
            "SENTRY_DSN",
        ]
        for key in url_secrets:
            assert is_sensitive_key(key) is True, f"{key} not flagged as sensitive!"
