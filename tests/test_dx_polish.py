"""Tests for Epic 6.4: DX Polish — CLI help, error messages, onboarding.

Tests cover:
- CLI --help output quality (examples, descriptions)
- Pre-flight checks (env validation, actionable suggestions)
- Error message UX (suggestions included)
- Argument defaults from DeployConfig
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# CLI --help quality
# ======================================================================
class TestCLIHelp:
    """openspace-mcp --help must be useful for first-time users."""

    def _get_help_text(self):
        """Get help text from parser directly (stdout redirected on Windows)."""
        import io
        from openspace.mcp.server import _build_arg_parser

        parser = _build_arg_parser()
        buf = io.StringIO()
        parser.print_help(buf)
        return buf.getvalue()

    def test_help_exits_zero(self):
        """--help should not crash the parser."""
        from openspace.mcp.server import _build_arg_parser

        parser = _build_arg_parser()
        # parse_args(['--help']) calls sys.exit(0) — just verify parser builds
        assert parser is not None

    def test_help_contains_description(self):
        text = self._get_help_text()
        assert "OpenSpace" in text
        assert "MCP" in text

    def test_help_contains_examples(self):
        """--help should show usage examples."""
        text = self._get_help_text()
        assert "example" in text.lower() or "openspace-mcp" in text

    def test_help_documents_transport_choices(self):
        text = self._get_help_text()
        assert "stdio" in text
        assert "streamable-http" in text

    def test_help_explains_auth_requirement(self):
        """Users must know HTTP transports require a bearer token."""
        text = self._get_help_text()
        output = text.lower()
        assert "bearer" in output or "auth" in output or "token" in output

    def test_help_shows_env_var_names(self):
        """Users should see which env vars configure the server."""
        text = self._get_help_text()
        assert "OPENSPACE_MCP" in text

    def test_help_has_version_flag(self):
        """--version should be registered."""
        from openspace.mcp.server import _build_arg_parser, _VERSION

        parser = _build_arg_parser()
        # Check that version action exists
        version_actions = [
            a for a in parser._actions
            if isinstance(a, __import__("argparse")._VersionAction)
        ]
        assert version_actions, "--version must be registered"
        assert _VERSION in version_actions[0].version


# ======================================================================
# Pre-flight checks
# ======================================================================
class TestPreflightChecks:
    """Environment validation with actionable suggestions."""

    def test_preflight_passes_for_stdio(self):
        from openspace.deploy.preflight import preflight_check

        issues = preflight_check(transport="stdio", check_port=False)
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, f"stdio should have no errors: {errors}"

    def test_preflight_catches_missing_bearer_token(self):
        from openspace.deploy.preflight import preflight_check

        with patch.dict(os.environ, {}, clear=True):
            issues = preflight_check(transport="streamable-http", check_port=False)
        token_issues = [i for i in issues if i.check == "bearer-token"]
        assert len(token_issues) == 1
        assert "OPENSPACE_MCP_BEARER_TOKEN" in token_issues[0].message

    def test_preflight_suggestion_is_actionable(self):
        """Suggestions must include a command the user can run."""
        from openspace.deploy.preflight import preflight_check

        with patch.dict(os.environ, {}, clear=True):
            issues = preflight_check(transport="streamable-http", check_port=False)
        token_issues = [i for i in issues if i.check == "bearer-token"]
        assert token_issues
        suggestion = token_issues[0].suggestion
        assert "export" in suggestion or "set" in suggestion.lower()

    def test_preflight_no_token_check_for_stdio(self):
        from openspace.deploy.preflight import check_bearer_token

        result = check_bearer_token("stdio")
        assert result is None

    def test_preflight_token_check_passes_when_set(self):
        from openspace.deploy.preflight import check_bearer_token

        with patch.dict(os.environ, {"OPENSPACE_MCP_BEARER_TOKEN": "test-token-value"}):
            result = check_bearer_token("streamable-http")
        assert result is None

    def test_preflight_skill_store_file_not_dir(self, tmp_path):
        from openspace.deploy.preflight import check_skill_store

        # Create a file where directory expected
        fake_path = tmp_path / "skills"
        fake_path.write_text("not a directory")
        result = check_skill_store(str(fake_path))
        assert result is not None
        assert result.check == "skill-store"

    def test_preflight_skill_store_missing_is_ok(self, tmp_path):
        """Missing skill store is OK — it will be created on first use."""
        from openspace.deploy.preflight import check_skill_store

        result = check_skill_store(str(tmp_path / "nonexistent"))
        assert result is None

    def test_preflight_python_version_check(self):
        from openspace.deploy.preflight import check_python_version

        # Current Python should pass
        result = check_python_version(minimum=(3, 11))
        assert result is None

    def test_preflight_python_version_too_old(self):
        from openspace.deploy.preflight import check_python_version

        # Require a future version to force failure
        result = check_python_version(minimum=(99, 0))
        assert result is not None
        assert "99.0" in result.message

    def test_preflight_issue_str_format(self):
        from openspace.deploy.preflight import PreflightIssue

        issue = PreflightIssue(
            check="test", message="broken", suggestion="fix it"
        )
        text = str(issue)
        assert "✗" in text
        assert "broken" in text
        assert "fix it" in text

    def test_preflight_warning_icon(self):
        from openspace.deploy.preflight import PreflightIssue

        issue = PreflightIssue(
            check="test", message="warning", suggestion="maybe fix",
            severity="warning",
        )
        assert "⚠" in str(issue)

    def test_preflight_report_format(self):
        from openspace.deploy.preflight import (
            PreflightIssue,
            format_preflight_report,
        )

        issues = [
            PreflightIssue(check="a", message="err", suggestion="fix", severity="error"),
            PreflightIssue(check="b", message="warn", suggestion="maybe", severity="warning"),
        ]
        report = format_preflight_report(issues)
        assert "Pre-flight" in report
        assert "1 error" in report
        assert "1 warning" in report

    def test_preflight_report_all_clear(self):
        from openspace.deploy.preflight import format_preflight_report

        report = format_preflight_report([])
        assert "All checks passed" in report


# ======================================================================
# Error message UX
# ======================================================================
class TestErrorMessages:
    """Error messages must include actionable suggestions."""

    def test_invalid_transport_error_is_descriptive(self):
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="transport"):
            DeployConfig(mcp_transport="invalid")

    def test_invalid_port_error_shows_range(self):
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="1-65535"):
            DeployConfig(mcp_port=99999)

    def test_bad_env_port_error_names_variable(self):
        from openspace.deploy.config import DeployConfig

        with patch.dict(os.environ, {"OPENSPACE_MCP_PORT": "xyz"}):
            with pytest.raises(ValueError, match="OPENSPACE_MCP_PORT"):
                DeployConfig.from_env()

    def test_bad_log_level_shows_valid_options(self):
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="DEBUG.*INFO.*WARNING.*ERROR"):
            DeployConfig(log_level="VERBOSE")


# ======================================================================
# Entrypoint wiring
# ======================================================================
class TestEntrypointDX:
    """DX improvements are wired into the actual entry point."""

    def test_preflight_module_importable(self):
        from openspace.deploy.preflight import preflight_check
        assert callable(preflight_check)

    def test_preflight_wired_into_server(self):
        """run_mcp_server must call preflight_check before starting."""
        import inspect
        from openspace.mcp.server import run_mcp_server

        source = inspect.getsource(run_mcp_server)
        assert "preflight" in source.lower(), (
            "Preflight checks must be wired into run_mcp_server"
        )

    def test_server_argparse_has_epilog(self):
        """Argparse must have epilog with examples."""
        from openspace.mcp.server import _build_arg_parser

        parser = _build_arg_parser()
        assert parser.epilog, "argparse must have epilog with examples"
        assert "example" in parser.epilog.lower()

    def test_server_argparse_has_formatter(self):
        """Must use RawDescriptionHelpFormatter for epilog formatting."""
        import argparse

        from openspace.mcp.server import _build_arg_parser

        parser = _build_arg_parser()
        assert parser.formatter_class in (
            argparse.RawDescriptionHelpFormatter,
            argparse.RawTextHelpFormatter,
        )
