"""
Configuration loader for MCP session.

This module provides functionality to load MCP configuration from JSON files.

Security: Sandbox is enforced by default for all stdio-based MCP servers.
Unsandboxed execution requires explicit SCION_ALLOW_UNSANDBOXED=1 env var.
"""

import os
from typing import Any, Optional

from scion.config.utils import get_config_value
from scion.grounding.core.types import SandboxOptions

from .installer import MCPInstallerManager
from .transport.connectors import (
    HttpConnector,
    MCPBaseConnector,
    SandboxConnector,
    StdioConnector,
    WebSocketConnector,
)
from .transport.connectors.utils import is_stdio_server

# Import E2BSandbox
try:
    from scion.grounding.core.security import E2BSandbox

    E2B_AVAILABLE = True
except ImportError:
    E2BSandbox = None
    E2B_AVAILABLE = False


# Trusted sandbox config keys that may be sourced from config/env
_TRUSTED_SANDBOX_KEYS = frozenset(
    {
        "timeout",
        "sse_read_timeout",
        "supergateway_command",
        "port",
        "sandbox_template_id",
    }
)


def _build_trusted_sandbox_options(
    caller_options: SandboxOptions | None,
    default_timeout: float,
    default_sse_timeout: float,
) -> dict[str, Any]:
    """Build sandbox options from trusted sources only.

    Strips caller-supplied api_key (must come from env E2B_API_KEY).
    Only allows known config keys through; ignores anything else.
    """
    base: dict[str, Any] = {
        "timeout": default_timeout,
        "sse_read_timeout": default_sse_timeout,
    }
    if caller_options:
        for key in _TRUSTED_SANDBOX_KEYS:
            if key in caller_options:
                base[key] = caller_options[key]
    return base


async def create_connector_from_config(
    server_config: dict[str, Any],
    server_name: str = "unknown",
    sandbox: bool = True,
    sandbox_options: SandboxOptions | None = None,
    timeout: float = 30.0,
    sse_read_timeout: float = 300.0,
    installer: Optional[MCPInstallerManager] = None,
    check_dependencies: bool = True,
    tool_call_max_retries: int = 3,
    tool_call_retry_delay: float = 1.0,
) -> MCPBaseConnector:
    """Create a connector based on server configuration.

    For stdio-based servers, sandbox is ENFORCED. Unsandboxed stdio execution
    is denied by default. Set SCION_ALLOW_UNSANDBOXED=1 to explicitly
    opt out (development/testing only — NOT recommended for production).

    Args:
        server_config: The server configuration section
        server_name: Name of the MCP server (for display purposes)
        sandbox: Whether to use sandboxed execution mode for running MCP servers.
                 Defaults to True (enforced).
        sandbox_options: Optional sandbox configuration options.
        timeout: Timeout for operations in seconds (default: 30.0)
        sse_read_timeout: SSE read timeout in seconds (default: 300.0)
        installer: Optional installer manager for dependency installation
        check_dependencies: Whether to check and install dependencies (default: True)
        tool_call_max_retries: Maximum number of retries for tool calls (default: 3)
        tool_call_retry_delay: Initial delay between retries in seconds (default: 1.0)

    Returns:
        A configured connector instance

    Raises:
        RuntimeError: If sandbox is required but not available, or if
            dependencies are not installed and user declines installation
    """

    # Get original command and args from config
    original_command = get_config_value(server_config, "command")
    original_args = get_config_value(server_config, "args", [])

    # --- Sandbox enforcement BEFORE any host-side operations ---
    # Reject unsandboxed stdio early, before ensure_dependencies runs
    # npm/pip install on the host. This prevents a malicious server config
    # from triggering host-side package installs before being denied.
    if is_stdio_server(server_config) and not sandbox:
        allow_unsandboxed = os.environ.get("SCION_ALLOW_UNSANDBOXED", "").strip()
        if allow_unsandboxed != "1":
            raise RuntimeError(
                f"Unsandboxed stdio execution denied for server '{server_name}'. "
                "Sandbox is required for all stdio-based MCP servers. "
                "Set SCION_ALLOW_UNSANDBOXED=1 to override (NOT recommended)."
            )
        import logging

        logging.getLogger(__name__).warning(
            "SECURITY: Running server '%s' WITHOUT sandbox (SCION_ALLOW_UNSANDBOXED=1). "
            "This is NOT recommended for production use.",
            server_name,
        )

    if is_stdio_server(server_config) and sandbox and not E2B_AVAILABLE:
        raise ImportError(
            "E2B sandbox support not available. Please install e2b-code-interpreter: 'pip install e2b-code-interpreter'"
        )

    # --- Host-side operations (only after sandbox enforcement passes) ---
    # Check and install dependencies if needed (only for stdio servers)
    if is_stdio_server(server_config) and check_dependencies:
        # Use provided installer or get global instance
        if installer is None:
            from .installer import get_global_installer

            installer = get_global_installer()

        # Ensure dependencies are installed (using original command/args)
        await installer.ensure_dependencies(server_name, original_command, original_args)

    # Stdio connector — unsandboxed (only reachable with explicit opt-out)
    if is_stdio_server(server_config) and not sandbox:
        return StdioConnector(
            command=get_config_value(server_config, "command"),
            args=get_config_value(server_config, "args"),
            env=get_config_value(server_config, "env", None),
        )

    # Sandboxed connector (E2B_AVAILABLE already verified above)
    elif is_stdio_server(server_config):
        # Build sandbox options from trusted config/env only (never user input)
        _sandbox_options = _build_trusted_sandbox_options(sandbox_options, timeout, sse_read_timeout)
        e2b_sandbox = E2BSandbox(_sandbox_options)

        # Extract timeout values from trusted options
        connector_timeout = _sandbox_options.get("timeout", timeout)
        connector_sse_timeout = _sandbox_options.get("sse_read_timeout", sse_read_timeout)

        # Create and return sandbox connector
        return SandboxConnector(
            sandbox=e2b_sandbox,
            command=get_config_value(server_config, "command"),
            args=get_config_value(server_config, "args"),
            env=get_config_value(server_config, "env", None),
            supergateway_command=_sandbox_options.get("supergateway_command", "npx -y supergateway"),
            port=_sandbox_options.get("port", 3000),
            timeout=connector_timeout,
            sse_read_timeout=connector_sse_timeout,
        )

    # HTTP connector
    elif "url" in server_config:
        return HttpConnector(
            base_url=get_config_value(server_config, "url"),
            headers=get_config_value(server_config, "headers", None),
            auth_token=get_config_value(server_config, "auth_token", None),
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            tool_call_max_retries=tool_call_max_retries,
            tool_call_retry_delay=tool_call_retry_delay,
        )

    # WebSocket connector
    elif "ws_url" in server_config:
        return WebSocketConnector(
            url=get_config_value(server_config, "ws_url"),
            headers=get_config_value(server_config, "headers", None),
            auth_token=get_config_value(server_config, "auth_token", None),
        )

    raise ValueError("Cannot determine connector type from config")
