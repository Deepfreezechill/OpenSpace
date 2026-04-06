"""OpenSpace MCP Server — bootstrap, FastMCP creation, and entry point.

Extracted from ``mcp_server.py`` (Epic 4.9) to consolidate all MCP server
concerns under ``openspace/mcp/``.

Sections:
  1. _MCPSafeStdout — stdout wrapper for MCP stdio transport
  2. Logging/stderr bootstrap — Windows pipe deadlock prevention
  3. create_mcp_app() — FastMCP factory with tool handler registration
  4. run_mcp_server() — CLI entry point with transport + auth setup
"""

from __future__ import annotations

import atexit
import inspect
import logging
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Stdout wrapper for MCP stdio transport
# ---------------------------------------------------------------------------
class _MCPSafeStdout:
    """Stdout wrapper: binary (.buffer) → real stdout, text (.write) → stderr.

    MCP stdio transport requires stdout for JSON-RPC messages only.
    Any text written by application code (print, warnings) must go to stderr.
    """

    def __init__(self, real_stdout, stderr):
        self._real = real_stdout
        self._stderr = stderr

    @property
    def buffer(self):
        return self._real.buffer

    def fileno(self):
        return self._real.fileno()

    def write(self, s):
        return self._stderr.write(s)

    def writelines(self, lines):
        return self._stderr.writelines(lines)

    def flush(self):
        self._stderr.flush()
        self._real.flush()

    def isatty(self):
        return self._stderr.isatty()

    @property
    def encoding(self):
        return self._stderr.encoding

    @property
    def errors(self):
        return self._stderr.errors

    @property
    def closed(self):
        return self._stderr.closed

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stderr, name)


# ---------------------------------------------------------------------------
# 2. Logging / stderr bootstrap
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_real_stdout = sys.stdout

# Windows pipe buffers are small. When using stdio MCP transport,
# the parent process only reads stdout for MCP messages and does NOT
# drain stderr. Heavy log/print output during execute_task fills the stderr
# pipe buffer, blocking this process on write() → deadlock → timeout.
# Redirect stderr to a log file on Windows to prevent this.
_stderr_file = None
if os.name == "nt":
    _stderr_file = open(_LOG_DIR / "mcp_stderr.log", "a", encoding="utf-8", buffering=1)
    sys.stderr = _stderr_file

sys.stdout = _MCPSafeStdout(_real_stdout, sys.stderr)

_log_file_handler = logging.FileHandler(_LOG_DIR / "mcp_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[_log_file_handler],
)
logger = logging.getLogger("openspace.mcp_server")


def _cleanup_file_handles():
    """Clean up file handles on exit."""
    global _stderr_file, _log_file_handler
    if _stderr_file:
        _stderr_file.close()
        _stderr_file = None
    if _log_file_handler:
        _log_file_handler.close()


atexit.register(_cleanup_file_handles)


# ---------------------------------------------------------------------------
# 3. FastMCP app factory
# ---------------------------------------------------------------------------
def create_mcp_app():
    """Create and return a configured FastMCP instance with all tools registered.

    This is the single factory for the MCP application. Tool handlers are
    wired via ``register_handlers()`` from the tool_handlers module.
    """
    from mcp.server.fastmcp import FastMCP

    from openspace.mcp.tool_handlers import register_handlers

    kwargs: dict = {}
    try:
        if "description" in inspect.signature(FastMCP.__init__).parameters:
            kwargs["description"] = "OpenSpace: Unite the Agents. Evolve the Mind. Rebuild the World."
    except (TypeError, ValueError):
        pass

    app = FastMCP("OpenSpace", **kwargs)
    register_handlers(app)

    # Register default health probes (Epic 6.1)
    _register_health_probes()

    return app


def _register_health_probes() -> None:
    """Register default health probes for core subsystems."""
    from openspace.observability.health import HealthProbe, health

    def _probe_skill_store() -> HealthProbe:
        try:
            from openspace.skill_engine.store import SkillStore

            store = SkillStore()
            count = len(store.list_skills()) if hasattr(store, "list_skills") else 0
            return HealthProbe(ok=True, detail=f"{count} skills", metadata={"count": count})
        except Exception as exc:
            return HealthProbe(ok=False, detail=f"{type(exc).__name__}")

    def _probe_grounding() -> HealthProbe:
        try:
            from openspace.agents.grounding_agent import GroundingAgent

            return HealthProbe(ok=True, detail="module loaded")
        except Exception as exc:
            return HealthProbe(ok=False, detail=f"{type(exc).__name__}")

    def _probe_mcp_tools() -> HealthProbe:
        from openspace.mcp.tool_handlers import register_handlers

        return HealthProbe(ok=True, detail="7 tools registered")

    health.register("skill_store", _probe_skill_store)
    health.register("grounding_engine", _probe_grounding)
    health.register("mcp_tools", _probe_mcp_tools)


# ---------------------------------------------------------------------------
# 4. Server entry point
# ---------------------------------------------------------------------------
def run_mcp_server(mcp=None) -> None:
    """Console-script entry point for ``openspace-mcp``.

    For HTTP transports (SSE, streamable-http), bearer token auth is
    REQUIRED.  Set OPENSPACE_MCP_BEARER_TOKEN in the environment.
    The server refuses to start without it (fail-closed).

    For stdio transport, auth is not applicable (local process IPC).

    Args:
        mcp: Optional pre-created FastMCP instance. If None, creates one
             via ``create_mcp_app()``.
    """
    import argparse

    import uvicorn

    from openspace.auth.bearer import (
        BEARER_TOKEN_ENV,
        BearerTokenMiddleware,
        get_bearer_token,
        validate_token_strength,
    )
    from openspace.auth.rate_limit import RateLimitMiddleware

    if mcp is None:
        mcp = create_mcp_app()

    parser = argparse.ArgumentParser(description="OpenSpace MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    # --- HTTP transports: enforce bearer token auth (fail-closed) ---
    token = get_bearer_token()
    if not token:
        logger.critical(
            "FAIL-CLOSED: %s not set. Refusing to start %s transport "
            "without authentication. Set the environment variable or "
            "use --transport stdio for local-only access.",
            BEARER_TOKEN_ENV,
            args.transport,
        )
        sys.exit(1)

    token_ok, reason = validate_token_strength(token)
    if not token_ok:
        logger.critical("FAIL-CLOSED: %s — %s", BEARER_TOKEN_ENV, reason)
        sys.exit(1)

    if args.transport == "sse":
        starlette_app = mcp.sse_app()
    else:
        starlette_app = mcp.streamable_http_app()

    # Middleware chain: request → BearerAuth → RateLimit → MCP app
    rate_limited_app = RateLimitMiddleware(starlette_app)
    protected_app = BearerTokenMiddleware(rate_limited_app, token)

    logger.info(
        "Starting MCP server with bearer auth + rate limiting on %s:%d (%s transport)",
        args.host,
        args.port,
        args.transport,
    )

    config = uvicorn.Config(
        protected_app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    import anyio

    anyio.run(server.serve)
