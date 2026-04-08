"""Centralized environment-based configuration for Scion deployments.

All deployment-relevant settings are read from environment variables with
sensible defaults. Configuration is frozen (immutable) after creation.

Environment variables:
    SCION_MCP_HOST          MCP server bind address (default: 0.0.0.0)
    SCION_MCP_PORT          MCP server port (default: 8000)
    SCION_MCP_TRANSPORT     Transport: stdio|sse|streamable-http (default: stdio)
    SCION_LOG_LEVEL         Log level: DEBUG|INFO|WARNING|ERROR (default: INFO)
    SCION_SHUTDOWN_TIMEOUT  Graceful shutdown timeout in seconds (default: 30)
    SCION_METRICS_ENABLED   Enable Prometheus metrics (default: true)
    SCION_SKILL_STORE_PATH  Path to skill store directory (default: skills/)
    SCION_DEBUG             Enable debug mode (default: false)

Usage::

    from scion.deploy.config import DeployConfig

    cfg = DeployConfig.from_env()
    print(cfg.mcp_port)  # 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

_VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with graceful fallback."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer"
        ) from None


@dataclass(frozen=True)
class DeployConfig:
    """Immutable deployment configuration.

    All values have sane defaults and can be overridden via environment
    variables using :meth:`from_env`.
    """

    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000
    mcp_transport: str = "stdio"
    log_level: str = "INFO"
    shutdown_timeout: int = 30
    metrics_enabled: bool = True
    skill_store_path: str = "skills/"
    debug: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.mcp_port <= 65535):
            raise ValueError(
                f"port must be 1-65535, got {self.mcp_port}"
            )
        if self.mcp_transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {_VALID_TRANSPORTS}, "
                f"got {self.mcp_transport!r}"
            )
        # Normalize log_level to uppercase
        normalized = self.log_level.upper()
        if normalized not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {_VALID_LOG_LEVELS}, "
                f"got {self.log_level!r}"
            )
        if normalized != self.log_level:
            object.__setattr__(self, "log_level", normalized)
        if self.shutdown_timeout < 1:
            raise ValueError(
                f"shutdown_timeout must be >= 1, got {self.shutdown_timeout}"
            )

    @classmethod
    def from_env(cls) -> DeployConfig:
        """Create config from environment variables.

        Handles empty strings, whitespace, and non-integer values
        gracefully with descriptive error messages.
        """
        return cls(
            mcp_host=os.environ.get("SCION_MCP_HOST", "0.0.0.0").strip() or "0.0.0.0",
            mcp_port=_int_env("SCION_MCP_PORT", 8000),
            mcp_transport=os.environ.get("SCION_MCP_TRANSPORT", "stdio").strip() or "stdio",
            log_level=(os.environ.get("SCION_LOG_LEVEL", "INFO").strip() or "INFO").upper(),
            shutdown_timeout=_int_env("SCION_SHUTDOWN_TIMEOUT", 30),
            metrics_enabled=os.environ.get(
                "SCION_METRICS_ENABLED", "true"
            ).strip().lower()
            in ("true", "1", "yes"),
            skill_store_path=os.environ.get(
                "SCION_SKILL_STORE_PATH", "skills/"
            ).strip() or "skills/",
            debug=os.environ.get("SCION_DEBUG", "false").strip().lower()
            in ("true", "1", "yes"),
        )

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize config without secrets.

        Returns only deployment-relevant fields. API keys, tokens,
        and other secrets are never included.
        """
        return {
            "mcp_host": self.mcp_host,
            "mcp_port": self.mcp_port,
            "mcp_transport": self.mcp_transport,
            "log_level": self.log_level,
            "shutdown_timeout": self.shutdown_timeout,
            "metrics_enabled": self.metrics_enabled,
            "skill_store_path": self.skill_store_path,
            "debug": self.debug,
        }
