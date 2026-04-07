"""Pre-flight environment checks for OpenSpace.

Validates the runtime environment before server startup and provides
actionable error messages with suggested fixes. Designed for DX — a
developer who misconfigures something should know exactly what to do.

Usage::

    from openspace.deploy.preflight import preflight_check

    issues = preflight_check(transport="streamable-http")
    if issues:
        for issue in issues:
            print(f"  ✗ {issue}")
        sys.exit(1)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreflightIssue:
    """A single pre-flight check failure with actionable fix."""

    check: str
    message: str
    suggestion: str
    severity: str = "error"  # error | warning

    def __str__(self) -> str:
        icon = "✗" if self.severity == "error" else "⚠"
        return f"{icon} [{self.check}] {self.message}\n  → {self.suggestion}"


def check_python_version(minimum: tuple = (3, 11)) -> Optional[PreflightIssue]:
    """Verify Python version meets minimum requirement."""
    if sys.version_info < minimum:
        return PreflightIssue(
            check="python-version",
            message=f"Python {minimum[0]}.{minimum[1]}+ required, got {sys.version_info.major}.{sys.version_info.minor}",
            suggestion=f"Install Python {minimum[0]}.{minimum[1]}+ from https://python.org/downloads/",
        )
    return None


def check_bearer_token(transport: str) -> Optional[PreflightIssue]:
    """Verify bearer token is set for HTTP transports."""
    if transport == "stdio":
        return None  # No auth needed for stdio

    token_env = "OPENSPACE_MCP_BEARER_TOKEN"
    token = os.environ.get(token_env, "").strip()
    if not token:
        if sys.platform == "win32":
            set_cmd = (
                f'$env:{token_env} = python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        else:
            set_cmd = (
                f'export {token_env}=$(python -c "import secrets; print(secrets.token_urlsafe(32))")'
            )
        return PreflightIssue(
            check="bearer-token",
            message=f"{token_env} not set (required for {transport} transport)",
            suggestion=(
                f"Set {token_env} to a strong random token:\n"
                f"    {set_cmd}\n"
                f"  Or use --transport stdio for local-only access (no auth required)"
            ),
        )
    return None


def check_skill_store(path: str) -> Optional[PreflightIssue]:
    """Verify skill store directory exists or can be created."""
    p = Path(path)
    if p.exists() and not p.is_dir():
        if sys.platform == "win32":
            fix_cmd = f"Remove-Item {path}; New-Item -ItemType Directory {path}"
        else:
            fix_cmd = f"rm {path} && mkdir -p {path}"
        return PreflightIssue(
            check="skill-store",
            message=f"Skill store path exists but is not a directory: {path}",
            suggestion=f"Remove the file and create directory:\n    {fix_cmd}",
        )
    return None


def check_port_available(port: int, host: str = "0.0.0.0") -> Optional[PreflightIssue]:
    """Check if the target port is likely available (best-effort)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
    except OSError:
        return PreflightIssue(
            check="port-available",
            message=f"Port {port} appears to be in use on {host}",
            suggestion=(
                f"Either stop the other process or use a different port:\n"
                f"    openspace-mcp --port {port + 1}\n"
                f"  Or set OPENSPACE_MCP_PORT={port + 1}"
            ),
            severity="warning",
        )
    return None


def preflight_check(
    transport: str = "stdio",
    port: int = 8000,
    host: str = "0.0.0.0",
    skill_store_path: str = "skills/",
    check_port: bool = True,
) -> List[PreflightIssue]:
    """Run all pre-flight checks and return any issues found.

    Returns an empty list if all checks pass.

    Args:
        transport: MCP transport type (stdio, sse, streamable-http)
        port: Server port (only checked for HTTP transports)
        host: Server bind address
        skill_store_path: Path to skill store directory
        check_port: Whether to check port availability
    """
    issues: List[PreflightIssue] = []

    result = check_python_version()
    if result:
        issues.append(result)

    result = check_bearer_token(transport)
    if result:
        issues.append(result)

    result = check_skill_store(skill_store_path)
    if result:
        issues.append(result)

    if transport != "stdio" and check_port:
        result = check_port_available(port, host)
        if result:
            issues.append(result)

    return issues


def format_preflight_report(issues: List[PreflightIssue]) -> str:
    """Format preflight issues into a readable report."""
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    lines = ["\n╔══ OpenSpace Pre-flight Check ══╗\n"]

    if errors:
        lines.append(f"  {len(errors)} error(s) found:\n")
        for issue in errors:
            lines.append(f"  {issue}\n")

    if warnings:
        lines.append(f"  {len(warnings)} warning(s):\n")
        for issue in warnings:
            lines.append(f"  {issue}\n")

    if not errors:
        lines.append("  ✓ All checks passed\n")

    lines.append("╚═══════════════════════════════╝")
    return "\n".join(lines)
