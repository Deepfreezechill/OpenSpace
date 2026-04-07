"""Tests for Epic 6.3: Deployment architecture.

Tests config management, graceful shutdown, Dockerfile validity,
and deployment readiness.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Config management
# ═══════════════════════════════════════════════════════════════════════


class TestDeployConfig:
    """Centralized env-based configuration."""

    def test_config_importable(self):
        from openspace.deploy.config import DeployConfig

        assert DeployConfig is not None

    def test_defaults_without_env(self):
        """Config provides sane defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            from openspace.deploy.config import DeployConfig

            cfg = DeployConfig()
            assert cfg.mcp_host == "0.0.0.0"
            assert cfg.mcp_port == 8000
            assert cfg.mcp_transport == "stdio"
            assert cfg.log_level == "INFO"
            assert cfg.shutdown_timeout == 30
            assert cfg.metrics_enabled is True

    def test_env_override(self):
        """Environment variables override defaults."""
        env = {
            "OPENSPACE_MCP_HOST": "127.0.0.1",
            "OPENSPACE_MCP_PORT": "9090",
            "OPENSPACE_MCP_TRANSPORT": "sse",
            "OPENSPACE_LOG_LEVEL": "DEBUG",
            "OPENSPACE_SHUTDOWN_TIMEOUT": "60",
            "OPENSPACE_METRICS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            from openspace.deploy.config import DeployConfig

            cfg = DeployConfig.from_env()
            assert cfg.mcp_host == "127.0.0.1"
            assert cfg.mcp_port == 9090
            assert cfg.mcp_transport == "sse"
            assert cfg.log_level == "DEBUG"
            assert cfg.shutdown_timeout == 60
            assert cfg.metrics_enabled is False

    def test_invalid_port_rejected(self):
        """Port outside valid range raises ValueError."""
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="port"):
            DeployConfig(mcp_port=99999)

    def test_invalid_transport_rejected(self):
        """Unknown transport raises ValueError."""
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="transport"):
            DeployConfig(mcp_transport="websocket")

    def test_invalid_log_level_rejected(self):
        """Invalid log level raises ValueError."""
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="log_level"):
            DeployConfig(log_level="VERBOSE")

    def test_negative_shutdown_timeout_rejected(self):
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="shutdown_timeout"):
            DeployConfig(shutdown_timeout=-1)

    def test_to_dict_excludes_secrets(self):
        """Serialization must not leak API keys."""
        from openspace.deploy.config import DeployConfig

        cfg = DeployConfig()
        d = cfg.to_safe_dict()
        # Ensure no secret fields are exposed
        for key in d:
            assert "key" not in key.lower() or "api" not in key.lower()
            assert "token" not in key.lower()
            assert "secret" not in key.lower()

    def test_frozen_config(self):
        """Config should be immutable after creation."""
        from openspace.deploy.config import DeployConfig

        cfg = DeployConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.mcp_port = 1234  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# Graceful shutdown
# ═══════════════════════════════════════════════════════════════════════


class TestGracefulShutdown:
    """Signal-aware graceful shutdown handler."""

    def test_shutdown_handler_importable(self):
        from openspace.deploy.shutdown import GracefulShutdownHandler

        assert GracefulShutdownHandler is not None

    @pytest.mark.asyncio
    async def test_shutdown_runs_hooks(self):
        """Shutdown must execute all registered hooks."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)
        hook1 = AsyncMock()
        hook2 = AsyncMock()
        handler.register_hook(hook1)
        handler.register_hook(hook2)

        await handler.shutdown()
        hook1.assert_awaited_once()
        hook2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_respects_timeout(self):
        """Hooks exceeding timeout must be cancelled."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=1)

        async def slow_hook():
            await asyncio.sleep(100)

        handler.register_hook(slow_hook)
        # Should complete within ~2 seconds (1s timeout + buffer), not hang
        await asyncio.wait_for(handler.shutdown(), timeout=5)

    @pytest.mark.asyncio
    async def test_shutdown_continues_on_hook_failure(self):
        """One failing hook must not prevent others from running."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)

        async def failing_hook():
            raise RuntimeError("hook failed")

        success_hook = AsyncMock()
        handler.register_hook(failing_hook)
        handler.register_hook(success_hook)

        await handler.shutdown()
        success_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Multiple shutdown calls must not re-run hooks."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)
        hook = AsyncMock()
        handler.register_hook(hook)

        await handler.shutdown()
        await handler.shutdown()  # second call is no-op
        hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_drains_in_flight(self):
        """Shutdown must wait for tracked in-flight tasks."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)
        completed = False

        async def in_flight_task():
            nonlocal completed
            await asyncio.sleep(0.5)
            completed = True

        handler.track_task(asyncio.create_task(in_flight_task()))
        await handler.shutdown()
        assert completed, "In-flight task was not drained before shutdown"


# ═══════════════════════════════════════════════════════════════════════
# Dockerfile
# ═══════════════════════════════════════════════════════════════════════


class TestDockerfile:
    """Dockerfile structure and correctness."""

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").is_file()

    def test_dockerfile_multi_stage(self):
        content = (ROOT / "Dockerfile").read_text()
        # Must have at least 2 FROM instructions (build + runtime)
        from_count = sum(1 for line in content.splitlines() if line.strip().startswith("FROM "))
        assert from_count >= 2, "Dockerfile should be multi-stage"

    def test_dockerfile_non_root_user(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "USER " in content, "Dockerfile must run as non-root user"

    def test_dockerfile_healthcheck(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content, "Dockerfile must include HEALTHCHECK"

    def test_dockerfile_no_secrets(self):
        """Dockerfile must not contain hardcoded secrets."""
        content = (ROOT / "Dockerfile").read_text().lower()
        for pattern in ["api_key=", "secret=", "password=", "token="]:
            assert pattern not in content, f"Dockerfile contains hardcoded secret: {pattern}"

    def test_dockerignore_exists(self):
        assert (ROOT / ".dockerignore").is_file()

    def test_dockerignore_excludes_tests(self):
        content = (ROOT / ".dockerignore").read_text()
        assert "tests" in content or "tests/" in content


# ═══════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════


class TestEntrypoint:
    """Application entrypoint wiring."""

    def test_deploy_package_importable(self):
        import openspace.deploy

        assert hasattr(openspace.deploy, "__name__")

    def test_config_from_env_integration(self):
        """Config → server wiring."""
        from openspace.deploy.config import DeployConfig

        cfg = DeployConfig()
        # Config provides all the values the MCP server needs
        assert hasattr(cfg, "mcp_host")
        assert hasattr(cfg, "mcp_port")
        assert hasattr(cfg, "mcp_transport")
        assert hasattr(cfg, "shutdown_timeout")


# ═══════════════════════════════════════════════════════════════════════
# Package completeness
# ═══════════════════════════════════════════════════════════════════════


class TestDeployPackageCompleteness:
    def test_all_modules_importable(self):
        import openspace.deploy.config
        import openspace.deploy.shutdown

        assert True

    def test_tool_count_unchanged(self):
        """Deployment must not change MCP tool count."""
        from unittest.mock import MagicMock

        from openspace.mcp.tool_handlers import register_handlers

        mock_mcp = MagicMock()
        register_handlers(mock_mcp)
        assert mock_mcp.tool.call_count == 8
