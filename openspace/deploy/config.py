"""Centralized environment-based configuration for OpenSpace deployments.

All deployment-relevant settings are read from environment variables with
sensible defaults. Configuration is frozen (immutable) after creation.

Environment variables:
    OPENSPACE_MCP_HOST          MCP server bind address (default: 0.0.0.0)
    OPENSPACE_MCP_PORT          MCP server port (default: 8000)
    OPENSPACE_MCP_TRANSPORT     Transport: stdio|sse|streamable-http (default: stdio)
    OPENSPACE_LOG_LEVEL         Log level: DEBUG|INFO|WARNING|ERROR (default: INFO)
    OPENSPACE_SHUTDOWN_TIMEOUT  Graceful shutdown timeout in seconds (default: 30)
    OPENSPACE_METRICS_ENABLED   Enable Prometheus metrics (default: true)
    OPENSPACE_SKILL_STORE_PATH  Path to skill store directory (default: skills/)
    OPENSPACE_DEBUG             Enable debug mode (default: false)

Usage::

    from openspace.deploy.config import DeployConfig

    cfg = DeployConfig.from_env()
    print(cfg.mcp_port)  # 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

_VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


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
        if self.log_level.upper() not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {_VALID_LOG_LEVELS}, "
                f"got {self.log_level!r}"
            )
        if self.shutdown_timeout < 0:
            raise ValueError(
                f"shutdown_timeout must be non-negative, got {self.shutdown_timeout}"
            )

    @classmethod
    def from_env(cls) -> DeployConfig:
        """Create config from environment variables."""
        return cls(
            mcp_host=os.environ.get("OPENSPACE_MCP_HOST", "0.0.0.0"),
            mcp_port=int(os.environ.get("OPENSPACE_MCP_PORT", "8000")),
            mcp_transport=os.environ.get("OPENSPACE_MCP_TRANSPORT", "stdio"),
            log_level=os.environ.get("OPENSPACE_LOG_LEVEL", "INFO").upper(),
            shutdown_timeout=int(
                os.environ.get("OPENSPACE_SHUTDOWN_TIMEOUT", "30")
            ),
            metrics_enabled=os.environ.get(
                "OPENSPACE_METRICS_ENABLED", "true"
            ).lower()
            in ("true", "1", "yes"),
            skill_store_path=os.environ.get(
                "OPENSPACE_SKILL_STORE_PATH", "skills/"
            ),
            debug=os.environ.get("OPENSPACE_DEBUG", "false").lower()
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
