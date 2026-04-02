"""Tests for EPIC 0.2: E2B Sandbox Hardening.

Covers:
- #4: Sandbox enforcement (not optional bypass)
- #5: Config from env/config only (no user-supplied API keys)
- #6: Fallback behavior must be deny (not allow)
- #7: Documentation validation
- #8: Integration test for sandbox creation path
"""

import ast
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Issue #4: Sandbox must be enforced, not optional bypass
# ---------------------------------------------------------------------------


class TestSandboxEnforcement:
    """Sandbox defaults must be True everywhere."""

    def test_mcp_config_defaults_sandbox_true(self):
        """MCPConfig.sandbox defaults to True."""
        from openspace.config.grounding import MCPConfig

        cfg = MCPConfig()
        assert cfg.sandbox is True, "MCPConfig.sandbox must default to True"

    def test_config_grounding_json_sandbox_true(self):
        """config_grounding.json ships with sandbox: true."""
        config_path = ROOT / "openspace" / "config" / "config_grounding.json"
        data = json.loads(config_path.read_text())
        assert data["mcp"]["sandbox"] is True, "config_grounding.json mcp.sandbox must be true"

    def test_config_security_json_sandbox_enabled(self):
        """config_security.json ships with sandbox_enabled: true globally."""
        config_path = ROOT / "openspace" / "config" / "config_security.json"
        data = json.loads(config_path.read_text())
        policies = data["security_policies"]
        assert policies["global"]["sandbox_enabled"] is True, "Global sandbox_enabled must be true"
        assert policies["backend"]["shell"]["sandbox_enabled"] is True, "Shell sandbox_enabled must be true"
        assert policies["backend"]["mcp"]["sandbox_enabled"] is True, "MCP sandbox_enabled must be true"

    def test_mcp_client_constructor_defaults_sandbox_true(self):
        """MCPClient.__init__ defaults sandbox=True."""
        import inspect

        from openspace.grounding.backends.mcp.client import MCPClient

        sig = inspect.signature(MCPClient.__init__)
        default = sig.parameters["sandbox"].default
        assert default is True, f"MCPClient sandbox default must be True, got {default}"

    def test_mcp_client_from_dict_defaults_sandbox_true(self):
        """MCPClient.from_dict defaults sandbox=True."""
        import inspect

        from openspace.grounding.backends.mcp.client import MCPClient

        sig = inspect.signature(MCPClient.from_dict)
        default = sig.parameters["sandbox"].default
        assert default is True, f"from_dict sandbox default must be True, got {default}"

    def test_mcp_client_from_config_file_defaults_sandbox_true(self):
        """MCPClient.from_config_file defaults sandbox=True."""
        import inspect

        from openspace.grounding.backends.mcp.client import MCPClient

        sig = inspect.signature(MCPClient.from_config_file)
        default = sig.parameters["sandbox"].default
        assert default is True, f"from_config_file sandbox default must be True, got {default}"

    def test_create_connector_defaults_sandbox_true(self):
        """create_connector_from_config defaults sandbox=True."""
        import inspect

        from openspace.grounding.backends.mcp.config import create_connector_from_config

        sig = inspect.signature(create_connector_from_config)
        default = sig.parameters["sandbox"].default
        assert default is True, f"create_connector sandbox default must be True, got {default}"

    def test_provider_extracts_sandbox_default_true(self):
        """MCPProvider reads sandbox with default True."""
        source = (ROOT / "openspace" / "grounding" / "backends" / "mcp" / "provider.py").read_text()
        # Should have: get_config_value(config, "sandbox", True)
        assert 'get_config_value(config, "sandbox", True)' in source, (
            "Provider must default sandbox to True via get_config_value"
        )


# ---------------------------------------------------------------------------
# Issue #5: Config from env/config only, no user-supplied API keys
# ---------------------------------------------------------------------------


class TestConfigSourceRestriction:
    """Sandbox config must come from trusted sources only."""

    def test_e2b_sandbox_ignores_caller_api_key(self):
        """E2BSandbox reads API key from env only, not options."""
        source = (ROOT / "openspace" / "grounding" / "core" / "security" / "e2b_sandbox.py").read_text()
        # Must NOT contain options.get("api_key")
        assert 'options.get("api_key")' not in source, "E2BSandbox must not accept api_key from caller options"
        # Must use os.environ.get("E2B_API_KEY")
        assert 'os.environ.get("E2B_API_KEY")' in source, "E2BSandbox must read API key from E2B_API_KEY env var"

    def test_trusted_sandbox_options_strips_api_key(self):
        """_build_trusted_sandbox_options strips api_key from caller input."""
        from openspace.grounding.backends.mcp.config import _build_trusted_sandbox_options

        caller_options = {
            "api_key": "EVIL_INJECTED_KEY",
            "timeout": 120,
            "sandbox_template_id": "custom",
            "arbitrary_field": "should_be_dropped",
        }
        result = _build_trusted_sandbox_options(caller_options, 30.0, 300.0)
        assert "api_key" not in result, "api_key must be stripped"
        assert result["timeout"] == 120, "Trusted timeout should pass through"
        assert result["sandbox_template_id"] == "custom", "Template ID should pass through"
        assert "arbitrary_field" not in result, "Unknown fields must be dropped"

    def test_trusted_sandbox_options_with_none_input(self):
        """_build_trusted_sandbox_options handles None caller options."""
        from openspace.grounding.backends.mcp.config import _build_trusted_sandbox_options

        result = _build_trusted_sandbox_options(None, 30.0, 300.0)
        assert result["timeout"] == 30.0
        assert result["sse_read_timeout"] == 300.0


# ---------------------------------------------------------------------------
# Issue #6: Fallback behavior must be deny, not allow
# ---------------------------------------------------------------------------


class TestFailClosedBehavior:
    """All failure modes must deny execution, not silently allow."""

    @pytest.mark.asyncio
    async def test_unsandboxed_stdio_denied_by_default(self):
        """Stdio without sandbox raises RuntimeError."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config

        config = {"command": "python", "args": ["-m", "some_server"]}
        with patch.dict(os.environ, {}, clear=False):
            # Ensure OPENSPACE_ALLOW_UNSANDBOXED is not set
            os.environ.pop("OPENSPACE_ALLOW_UNSANDBOXED", None)
            with pytest.raises(RuntimeError, match="Unsandboxed stdio execution denied"):
                await create_connector_from_config(
                    config,
                    server_name="test",
                    sandbox=False,
                    check_dependencies=False,
                )

    @pytest.mark.asyncio
    async def test_unsandboxed_stdio_allowed_with_explicit_opt_out(self):
        """Stdio without sandbox works only with OPENSPACE_ALLOW_UNSANDBOXED=1."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config
        from openspace.grounding.backends.mcp.transport.connectors import StdioConnector

        config = {"command": "python", "args": ["-m", "some_server"]}
        with patch.dict(os.environ, {"OPENSPACE_ALLOW_UNSANDBOXED": "1"}):
            connector = await create_connector_from_config(
                config,
                server_name="test",
                sandbox=False,
                check_dependencies=False,
            )
            assert isinstance(connector, StdioConnector)

    @pytest.mark.asyncio
    async def test_unsandboxed_partial_value_still_denied(self):
        """OPENSPACE_ALLOW_UNSANDBOXED must be exactly '1'."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config

        config = {"command": "python", "args": ["-m", "some_server"]}
        for bad_value in ["true", "yes", "0", "", "  "]:
            with patch.dict(os.environ, {"OPENSPACE_ALLOW_UNSANDBOXED": bad_value}):
                with pytest.raises(RuntimeError, match="Unsandboxed stdio execution denied"):
                    await create_connector_from_config(
                        config,
                        server_name="test",
                        sandbox=False,
                        check_dependencies=False,
                    )

    @pytest.mark.asyncio
    async def test_sandbox_required_but_e2b_unavailable_raises(self):
        """When sandbox=True but E2B not installed, raises ImportError."""
        from openspace.grounding.backends.mcp import config as cfg_module
        from openspace.grounding.backends.mcp.config import create_connector_from_config as _create

        config = {"command": "python", "args": ["-m", "some_server"]}
        original = cfg_module.E2B_AVAILABLE
        try:
            cfg_module.E2B_AVAILABLE = False
            with pytest.raises(ImportError, match="E2B sandbox support not available"):
                await _create(
                    config,
                    server_name="test",
                    sandbox=True,
                    check_dependencies=False,
                )
        finally:
            cfg_module.E2B_AVAILABLE = original

    def test_config_loader_fails_closed_on_invalid_config(self):
        """Config loader raises on validation failure, not silently defaults."""
        source = (ROOT / "openspace" / "config" / "loader.py").read_text()
        # Must NOT contain "using default configuration"
        assert "using default configuration" not in source, "Config loader must not silently fall back to defaults"
        # Must raise RuntimeError on validation failure
        assert "raise RuntimeError" in source, "Config loader must raise on validation failure"

    def test_security_config_is_critical(self):
        """Security config file is loaded with critical=True."""
        source = (ROOT / "openspace" / "config" / "loader.py").read_text()
        assert "CONFIG_SECURITY" in source
        assert "_CRITICAL_CONFIG_FILES" in source
        assert "critical_files" in source, "Config loader must pass critical_files for security config"

    @pytest.mark.asyncio
    async def test_sandbox_enforcement_before_ensure_dependencies(self):
        """Sandbox enforcement must happen BEFORE ensure_dependencies."""
        source = (ROOT / "openspace" / "grounding" / "backends" / "mcp" / "config.py").read_text()
        # Find positions of key operations
        sandbox_check_pos = source.find("Sandbox enforcement BEFORE")
        deps_check_pos = source.find("ensure_dependencies")
        assert sandbox_check_pos != -1, "Sandbox enforcement comment must exist"
        assert deps_check_pos != -1, "ensure_dependencies must exist"
        assert sandbox_check_pos < deps_check_pos, "Sandbox enforcement must come before ensure_dependencies"

    @pytest.mark.asyncio
    async def test_unsandboxed_denied_before_deps_installed(self):
        """Unsandboxed stdio denied before any host-side install runs."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config

        install_called = False

        async def mock_ensure_deps(*a, **kw):
            nonlocal install_called
            install_called = True

        config = {"command": "python", "args": ["-m", "server"]}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENSPACE_ALLOW_UNSANDBOXED", None)
            with patch(
                "openspace.grounding.backends.mcp.installer.MCPInstallerManager.ensure_dependencies",
                side_effect=mock_ensure_deps,
            ):
                with pytest.raises(RuntimeError, match="Unsandboxed stdio execution denied"):
                    await create_connector_from_config(
                        config,
                        server_name="test",
                        sandbox=False,
                        check_dependencies=True,
                    )
        assert not install_called, "ensure_dependencies must NOT run before sandbox denial"

    def test_no_sandbox_false_defaults_in_config_json(self):
        """No config JSON file ships with sandbox disabled."""
        config_dir = ROOT / "openspace" / "config"
        for json_file in config_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            self._check_no_sandbox_false(data, str(json_file))

    def _check_no_sandbox_false(self, obj, path, key_path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                current = f"{key_path}.{k}" if key_path else k
                if k in ("sandbox", "sandbox_enabled") and v is False:
                    pytest.fail(f"{path} has {current}=false — sandbox must be enabled")
                self._check_no_sandbox_false(v, path, current)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._check_no_sandbox_false(item, path, f"{key_path}[{i}]")


# ---------------------------------------------------------------------------
# Issue #7: Documentation validation
# ---------------------------------------------------------------------------


class TestSandboxDocumentation:
    """Sandbox env/config must be properly documented."""

    def test_env_example_documents_e2b_api_key_required(self):
        """E2B_API_KEY is documented as required in .env.example."""
        env_example = (ROOT / "openspace" / ".env.example").read_text()
        assert "E2B_API_KEY" in env_example
        # Must not say "Optional"
        lines_around = [line for line in env_example.splitlines() if "E2B" in line.upper() or "sandbox" in line.lower()]
        text = "\n".join(lines_around).lower()
        assert "optional" not in text or "not recommended" in text, (
            ".env.example should not describe E2B as merely optional"
        )

    def test_env_example_documents_unsandboxed_opt_out(self):
        """OPENSPACE_ALLOW_UNSANDBOXED is documented."""
        env_example = (ROOT / "openspace" / ".env.example").read_text()
        assert "OPENSPACE_ALLOW_UNSANDBOXED" in env_example

    def test_config_readme_documents_sandbox(self):
        """config/README.md has E2B sandbox section."""
        readme = (ROOT / "openspace" / "config" / "README.md").read_text()
        assert "E2B Sandbox" in readme, "README must have E2B sandbox section"
        assert "E2B_API_KEY" in readme, "README must document E2B_API_KEY"
        assert "OPENSPACE_ALLOW_UNSANDBOXED" in readme, "README must document OPENSPACE_ALLOW_UNSANDBOXED"
        assert "Fail-closed" in readme or "fail-closed" in readme.lower(), "README must document fail-closed behavior"


# ---------------------------------------------------------------------------
# Issue #8: Integration test for sandbox creation path
# ---------------------------------------------------------------------------


class TestSandboxCreationIntegration:
    """Integration tests proving the sandbox creation path works end-to-end."""

    @pytest.mark.asyncio
    async def test_stdio_config_creates_sandbox_connector(self):
        """Stdio MCP config with sandbox=True creates SandboxConnector."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config
        from openspace.grounding.backends.mcp.transport.connectors import SandboxConnector

        mock_e2b = MagicMock()
        config = {"command": "python", "args": ["-m", "server"]}

        with (
            patch(
                "openspace.grounding.backends.mcp.config.E2BSandbox",
                return_value=mock_e2b,
            ),
            patch("openspace.grounding.backends.mcp.config.E2B_AVAILABLE", True),
        ):
            connector = await create_connector_from_config(
                config,
                server_name="test",
                sandbox=True,
                check_dependencies=False,
            )
            assert isinstance(connector, SandboxConnector), f"Expected SandboxConnector, got {type(connector).__name__}"

    @pytest.mark.asyncio
    async def test_sandbox_connector_receives_filtered_env(self):
        """SandboxConnector only gets ENV_ALLOWLIST vars."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config

        mock_e2b = MagicMock()
        config = {"command": "python", "args": ["-m", "server"]}

        with (
            patch(
                "openspace.grounding.backends.mcp.config.E2BSandbox",
                return_value=mock_e2b,
            ),
            patch("openspace.grounding.backends.mcp.config.E2B_AVAILABLE", True),
            patch.dict(
                os.environ,
                {
                    "SECRET_KEY": "leaked",
                    "PATH": "/usr/bin",
                },
            ),
        ):
            connector = await create_connector_from_config(
                config,
                server_name="test",
                sandbox=True,
                check_dependencies=False,
            )
            # SandboxConnector filters env in __init__
            if hasattr(connector, "user_env") and connector.user_env:
                assert "SECRET_KEY" not in connector.user_env, "SECRET_KEY must not leak into sandbox"

    @pytest.mark.asyncio
    async def test_http_config_unaffected_by_sandbox_enforcement(self):
        """HTTP-based MCP servers are unaffected by sandbox enforcement."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config
        from openspace.grounding.backends.mcp.transport.connectors import HttpConnector

        config = {"url": "http://localhost:8080"}
        connector = await create_connector_from_config(
            config,
            server_name="test",
            sandbox=True,
            check_dependencies=False,
        )
        assert isinstance(connector, HttpConnector)

    @pytest.mark.asyncio
    async def test_websocket_config_unaffected_by_sandbox_enforcement(self):
        """WebSocket-based MCP servers are unaffected by sandbox enforcement."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config
        from openspace.grounding.backends.mcp.transport.connectors import WebSocketConnector

        config = {"ws_url": "ws://localhost:8080"}
        connector = await create_connector_from_config(
            config,
            server_name="test",
            sandbox=True,
            check_dependencies=False,
        )
        assert isinstance(connector, WebSocketConnector)

    @pytest.mark.asyncio
    async def test_e2b_sandbox_constructed_with_trusted_options(self):
        """E2BSandbox receives options WITHOUT api_key from caller."""
        from openspace.grounding.backends.mcp.config import create_connector_from_config

        captured_options = {}

        def mock_e2b_init(options):
            captured_options.update(options)
            return MagicMock()

        config = {"command": "python", "args": ["-m", "server"]}
        caller_options = {
            "api_key": "SHOULD_BE_STRIPPED",
            "timeout": 120,
            "sandbox_template_id": "custom-template",
        }

        with (
            patch(
                "openspace.grounding.backends.mcp.config.E2BSandbox",
                side_effect=mock_e2b_init,
            ),
            patch("openspace.grounding.backends.mcp.config.E2B_AVAILABLE", True),
        ):
            await create_connector_from_config(
                config,
                server_name="test",
                sandbox=True,
                sandbox_options=caller_options,
                check_dependencies=False,
            )
            assert "api_key" not in captured_options, "api_key must not reach E2BSandbox from caller options"
            assert captured_options.get("timeout") == 120
            assert captured_options.get("sandbox_template_id") == "custom-template"


# ---------------------------------------------------------------------------
# Regression: AST scan for sandbox=False defaults in production code
# ---------------------------------------------------------------------------


class TestNoSandboxFalseInProduction:
    """Regression test: no production code should default sandbox to False."""

    PRODUCTION_DIRS = [
        ROOT / "openspace" / "grounding" / "backends" / "mcp",
        ROOT / "openspace" / "config",
    ]

    def test_no_sandbox_false_keyword_defaults(self):
        """No function signature in MCP backend defaults sandbox to False."""
        violations = []
        for prod_dir in self.PRODUCTION_DIRS:
            for py_file in prod_dir.rglob("*.py"):
                try:
                    source = py_file.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                tree = ast.parse(source, filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for arg, default in zip(
                            reversed(node.args.args),
                            reversed(node.args.defaults),
                        ):
                            if arg.arg == "sandbox" and isinstance(default, ast.Constant) and default.value is False:
                                violations.append(
                                    f"{py_file.relative_to(ROOT)}:{node.lineno} def {node.name}(sandbox=False)"
                                )
        assert not violations, "Found sandbox=False defaults in production code:\n" + "\n".join(violations)
