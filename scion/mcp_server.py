"""Scion MCP Server — thin entry point.

All MCP concerns live under ``scion/mcp/``:
  - ``scion.mcp.server``        — bootstrap, FastMCP, run_mcp_server()
  - ``scion.mcp.tool_handlers`` — execute_task, search_skills, fix_skill, upload_skill

This module exists for backward compatibility:
  - ``python -m scion.mcp_server`` still works
  - ``from scion.mcp_server import run_mcp_server`` still works
  - Existing test imports that reference this module still work

Usage:
    python -m scion.mcp_server                     # stdio (default)
    python -m scion.mcp_server --transport sse     # SSE on port 8080
    python -m scion.mcp_server --port 9090         # SSE on custom port
"""

from __future__ import annotations

# Re-export for backward compatibility
from scion.mcp.server import (  # noqa: F401
    _MCPSafeStdout,
    _cleanup_file_handles,
    create_mcp_app,
    run_mcp_server,
)

# Lazy module-level mcp instance for backward compat.
# Some code imports mcp_server.mcp directly. Created on first access
# to avoid wasting a FastMCP instance when only run_mcp_server() is needed.
_mcp = None


def _get_mcp():
    global _mcp
    if _mcp is None:
        _mcp = create_mcp_app()
    return _mcp


class _LazyMcpProxy:
    """Proxy that creates the FastMCP instance on first attribute access."""

    def __getattr__(self, name):
        return getattr(_get_mcp(), name)


mcp = _LazyMcpProxy()

if __name__ == "__main__":
    run_mcp_server(_get_mcp())
