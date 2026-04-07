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

    def test_zero_shutdown_timeout_rejected(self):
        """timeout=0 would cause instant task cancellation."""
        from openspace.deploy.config import DeployConfig

        with pytest.raises(ValueError, match="shutdown_timeout"):
            DeployConfig(shutdown_timeout=0)

    def test_port_boundary_values(self):
        from openspace.deploy.config import DeployConfig

        # Valid boundaries
        assert DeployConfig(mcp_port=1).mcp_port == 1
        assert DeployConfig(mcp_port=65535).mcp_port == 65535
        # Invalid boundaries
        with pytest.raises(ValueError, match="port"):
            DeployConfig(mcp_port=0)
        with pytest.raises(ValueError, match="port"):
            DeployConfig(mcp_port=65536)

    def test_log_level_normalized_to_upper(self):
        from openspace.deploy.config import DeployConfig

        cfg = DeployConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_non_integer_port_env_raises(self):
        from openspace.deploy.config import DeployConfig

        with patch.dict(os.environ, {"OPENSPACE_MCP_PORT": "abc"}, clear=True):
            with pytest.raises(ValueError, match="OPENSPACE_MCP_PORT"):
                DeployConfig.from_env()

    def test_empty_port_env_uses_default(self):
        from openspace.deploy.config import DeployConfig

        with patch.dict(os.environ, {"OPENSPACE_MCP_PORT": ""}, clear=True):
            cfg = DeployConfig.from_env()
            assert cfg.mcp_port == 8000

    def test_boolean_env_parsing_variants(self):
        """All boolean truthy values should work."""
        from openspace.deploy.config import DeployConfig

        for val in ("true", "True", "TRUE", "1", "yes"):
            with patch.dict(os.environ, {"OPENSPACE_METRICS_ENABLED": val}, clear=True):
                cfg = DeployConfig.from_env()
                assert cfg.metrics_enabled is True, f"Failed for {val!r}"

        for val in ("false", "0", "no", ""):
            with patch.dict(os.environ, {"OPENSPACE_METRICS_ENABLED": val}, clear=True):
                cfg = DeployConfig.from_env()
                assert cfg.metrics_enabled is False, f"Failed for {val!r}"

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

    @pytest.mark.asyncio
    async def test_drain_survives_task_exception(self):
        """A throwing in-flight task must not crash shutdown."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)
        success_hook = AsyncMock()
        handler.register_hook(success_hook)

        async def exploding_task():
            raise RuntimeError("task boom")

        handler.track_task(asyncio.create_task(exploding_task()))
        await handler.shutdown()
        success_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_track_task_during_shutdown_auto_cancels(self):
        """Tasks tracked after shutdown starts are auto-cancelled."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)
        await handler.shutdown()

        # Now try to track a new task — should be cancelled
        task = asyncio.create_task(asyncio.sleep(100))
        handler.track_task(task)
        await asyncio.sleep(0.1)
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_with_no_hooks_no_tasks(self):
        """Empty shutdown completes without error."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        handler = GracefulShutdownHandler(timeout=5)
        await handler.shutdown()  # should not raise

    def test_shutdown_timeout_minimum(self):
        """timeout < 1 is rejected."""
        from openspace.deploy.shutdown import GracefulShutdownHandler

        with pytest.raises(ValueError, match="timeout"):
            GracefulShutdownHandler(timeout=0)


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
        # Must be a Dockerfile instruction, not a comment
        lines = content.splitlines()
        user_lines = [l for l in lines if l.strip().startswith("USER ")]
        assert user_lines, "Dockerfile must have USER instruction (not in comment)"

    def test_dockerfile_shutdown_timeout_under_docker_default(self):
        """Dockerfile OPENSPACE_SHUTDOWN_TIMEOUT must be < Docker's 10s default."""
        content = (ROOT / "Dockerfile").read_text()
        import re

        match = re.search(r"OPENSPACE_SHUTDOWN_TIMEOUT=(\d+)", content)
        assert match, "Dockerfile must set OPENSPACE_SHUTDOWN_TIMEOUT"
        timeout = int(match.group(1))
        assert timeout < 10, (
            f"Shutdown timeout {timeout}s must be < Docker's 10s stop_grace_period"
        )

    def test_dockerfile_healthcheck(self):
        content = (ROOT / "Dockerfile").read_text()
        lines = content.splitlines()
        hc_lines = [l for l in lines if l.strip().startswith("HEALTHCHECK")]
        assert hc_lines, "Dockerfile must have HEALTHCHECK instruction"

    def test_dockerfile_healthcheck_hits_real_endpoint(self):
        """HEALTHCHECK must hit /health HTTP endpoint, not just import."""
        content = (ROOT / "Dockerfile").read_text()
        assert "localhost:8000/health" in content, (
            "HEALTHCHECK must hit real /health endpoint for accurate checks"
        )

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
        assert hasattr(cfg, "mcp_host")
        assert hasattr(cfg, "mcp_port")
        assert hasattr(cfg, "mcp_transport")
        assert hasattr(cfg, "shutdown_timeout")

    def test_deploy_config_wired_into_server(self):
        """run_mcp_server imports DeployConfig (not dead code)."""
        import inspect

        from openspace.mcp.server import run_mcp_server

        source = inspect.getsource(run_mcp_server)
        assert "DeployConfig" in source, "DeployConfig must be used in run_mcp_server"

    def test_health_endpoint_outside_auth(self):
        """/health must be accessible WITHOUT bearer auth for K8s probes."""
        import inspect

        from openspace.mcp.server import run_mcp_server

        source = inspect.getsource(run_mcp_server)
        # /health route must be mounted BEFORE auth middleware in ASGI chain
        health_pos = source.find("Route(\"/health\"")
        bearer_pos = source.find("BearerTokenMiddleware")
        assert health_pos > 0, "/health route must exist"
        assert bearer_pos > 0, "BearerTokenMiddleware must exist"
        # In the code, auth wraps the MCP app; health wraps auth+MCP at top level
        # So health_endpoint definition comes AFTER protected_app is built
        assert health_pos > bearer_pos, (
            "/health must be outside auth — route should wrap protected_app"
        )

    def test_shutdown_runs_after_uvicorn(self):
        """GracefulShutdownHandler.shutdown() must run after server.serve()."""
        import inspect

        from openspace.mcp.server import run_mcp_server

        source = inspect.getsource(run_mcp_server)
        assert "serve_with_shutdown" in source, (
            "Shutdown must be wired via serve_with_shutdown wrapper"
        )
        assert "finally:" in source, (
            "Shutdown must run in finally block to guarantee execution"
        )


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
