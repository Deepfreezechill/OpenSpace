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

            if not hasattr(_probe_skill_store, "_cached"):
                _probe_skill_store._cached = SkillStore()
            store = _probe_skill_store._cached
            count = len(store.list_skills()) if hasattr(store, "list_skills") else 0
            return HealthProbe(ok=True, detail=f"{count} skills")
        except Exception:
            return HealthProbe(ok=False, detail="not available")

    def _probe_grounding() -> HealthProbe:
        try:
            from openspace.agents.grounding_agent import GroundingAgent

            return HealthProbe(ok=True, detail="module loaded")
        except Exception:
            return HealthProbe(ok=False, detail="not available")

    def _probe_mcp_tools() -> HealthProbe:
        return HealthProbe(ok=True, detail="tools registered")

    health.register("skill_store", _probe_skill_store)
    health.register("grounding_engine", _probe_grounding)
    health.register("mcp_tools", _probe_mcp_tools)


# ---------------------------------------------------------------------------
# 4. CLI argument parser (extracted for testability)
# ---------------------------------------------------------------------------
_VERSION = "0.1.0"  # TODO: read from pyproject.toml or importlib.metadata

_EPILOG = """\
examples:
  openspace-mcp                          # stdio transport (local, no auth)
  openspace-mcp --transport streamable-http  # HTTP (requires bearer token)
  openspace-mcp --transport sse --port 9000  # SSE on custom port

environment variables:
  OPENSPACE_MCP_HOST             Bind address (default: 0.0.0.0)
  OPENSPACE_MCP_PORT             Port number (default: 8000)
  OPENSPACE_MCP_TRANSPORT        Transport: stdio|sse|streamable-http
  OPENSPACE_MCP_BEARER_TOKEN     Auth token (REQUIRED for HTTP transports)
  OPENSPACE_LOG_LEVEL            Logging: DEBUG|INFO|WARNING|ERROR
  OPENSPACE_SHUTDOWN_TIMEOUT     Graceful shutdown seconds (default: 30)
  OPENSPACE_METRICS_ENABLED      Prometheus metrics: true|false

notes:
  HTTP transports (sse, streamable-http) require OPENSPACE_MCP_BEARER_TOKEN.
  Use --transport stdio for local development without authentication.
"""


def _build_arg_parser(deploy_cfg=None):
    """Build the CLI argument parser with rich help text.

    Extracted from run_mcp_server for testability. Returns an
    argparse.ArgumentParser ready for parse_args().
    """
    import argparse

    defaults = {
        "transport": "stdio",
        "port": 8000,
        "host": "0.0.0.0",
    }
    if deploy_cfg is not None:
        defaults["transport"] = deploy_cfg.mcp_transport
        defaults["port"] = deploy_cfg.mcp_port
        defaults["host"] = deploy_cfg.mcp_host

    parser = argparse.ArgumentParser(
        prog="openspace-mcp",
        description="OpenSpace MCP Server — AI agent framework with tool execution",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openspace-mcp {_VERSION}",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=defaults["transport"],
        help="MCP transport protocol (default: %(default)s). "
        "HTTP transports require OPENSPACE_MCP_BEARER_TOKEN.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=defaults["port"],
        help="Server port for HTTP transports (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=defaults["host"],
        help="Bind address for HTTP transports (default: %(default)s)",
    )
    return parser


# ---------------------------------------------------------------------------
# 5. Server entry point
# ---------------------------------------------------------------------------
def run_mcp_server(mcp=None) -> None:
    """Console-script entry point for ``openspace-mcp``.

    Uses DeployConfig for defaults, with CLI args as overrides.
    For HTTP transports, bearer token auth is REQUIRED.

    Args:
        mcp: Optional pre-created FastMCP instance. If None, creates one
             via ``create_mcp_app()``.
    """
    import uvicorn

    from openspace.auth.bearer import (
        BEARER_TOKEN_ENV,
        BearerTokenMiddleware,
        get_bearer_token,
        validate_token_strength,
    )
    from openspace.auth.rate_limit import RateLimitMiddleware
    from openspace.deploy.config import DeployConfig

    if mcp is None:
        mcp = create_mcp_app()

    # Load config from environment, then allow CLI overrides
    deploy_cfg = DeployConfig.from_env()

    parser = _build_arg_parser(deploy_cfg)
    args = parser.parse_args()

    # Pre-flight checks: validate environment before starting
    from openspace.deploy.preflight import (
        format_preflight_report,
        preflight_check,
    )

    issues = preflight_check(
        transport=args.transport,
        port=args.port,
        host=args.host,
        skill_store_path=deploy_cfg.skill_store_path,
        check_port=(args.transport != "stdio"),
    )
    if issues:
        errors = [i for i in issues if i.severity == "error"]
        report = format_preflight_report(issues)
        logger.info(report)
        if errors:
            sys.exit(1)

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

    # Auth middleware chain: request → BearerAuth → RateLimit → MCP app
    rate_limited_app = RateLimitMiddleware(starlette_app)
    protected_app = BearerTokenMiddleware(rate_limited_app, token)

    # /health is OUTSIDE auth — K8s probes can't send bearer tokens
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    from openspace.observability.health import health

    async def health_endpoint(request):
        result = health.check()
        status_code = 200 if result.get("status") == "healthy" else 503
        return JSONResponse(result, status_code=status_code)

    app = Starlette(
        routes=[
            Route("/health", health_endpoint, methods=["GET"]),
            Mount("/", app=protected_app),
        ]
    )

    logger.info(
        "Starting MCP server with bearer auth + rate limiting on %s:%d (%s transport)",
        args.host,
        args.port,
        args.transport,
    )

    # Graceful shutdown: uvicorn handles SIGTERM → drain HTTP → our hooks run after
    from openspace.deploy.shutdown import GracefulShutdownHandler

    shutdown_handler = GracefulShutdownHandler(timeout=deploy_cfg.shutdown_timeout)

    # Give uvicorn half the budget for HTTP draining, our handler gets the rest
    uvicorn_timeout = max(deploy_cfg.shutdown_timeout // 2, 1)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=deploy_cfg.log_level.lower(),
        timeout_graceful_shutdown=uvicorn_timeout,
    )
    server = uvicorn.Server(config)

    async def serve_with_shutdown():
        try:
            await server.serve()
        finally:
            await shutdown_handler.shutdown()

    import anyio

    anyio.run(serve_with_shutdown)
