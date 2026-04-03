"""OpenSpace MCP Server — thin entry point.

All MCP concerns live under ``openspace/mcp/``:
  - ``openspace.mcp.server``        — bootstrap, FastMCP, run_mcp_server()
  - ``openspace.mcp.tool_handlers`` — execute_task, search_skills, fix_skill, upload_skill

This module exists for backward compatibility:
  - ``python -m openspace.mcp_server`` still works
  - ``from openspace.mcp_server import run_mcp_server`` still works
  - Existing test imports that reference this module still work

Usage:
    python -m openspace.mcp_server                     # stdio (default)
    python -m openspace.mcp_server --transport sse     # SSE on port 8080
    python -m openspace.mcp_server --port 9090         # SSE on custom port
"""

from __future__ import annotations

# Re-export for backward compatibility
from openspace.mcp.server import (  # noqa: F401
    _MCPSafeStdout,
    _cleanup_file_handles,
    create_mcp_app,
    run_mcp_server,
)

# Create the module-level mcp instance for backward compatibility.
# Some code (e.g., test_mcp_auth.py) imports mcp_server and expects
# the FastMCP `mcp` instance to exist at module level.
mcp = create_mcp_app()

if __name__ == "__main__":
    run_mcp_server(mcp)
