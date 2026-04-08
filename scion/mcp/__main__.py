"""Allow ``python -m scion.mcp`` to launch the MCP server."""

from scion.mcp.server import create_mcp_app, run_mcp_server

if __name__ == "__main__":
    run_mcp_server(create_mcp_app())
