"""Graceful shutdown handler for Scion.

Provides signal-aware shutdown that:
1. Stops accepting new requests
2. Drains in-flight tasks (with timeout)
3. Runs registered cleanup hooks
4. Exits cleanly

Uses a single monotonic deadline for the entire shutdown sequence to
ensure Docker's stop_grace_period (default 10s) isn't exceeded.

Usage::

    from scion.deploy.shutdown import GracefulShutdownHandler

    handler = GracefulShutdownHandler(timeout=30)
    handler.register_hook(flush_metrics)
    handler.register_hook(close_connections)

    # In your server setup:
    handler.install_signal_handlers(loop)

    # On SIGTERM/SIGINT, handler.shutdown() runs automatically.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from typing import Any, Awaitable, Callable, List, Optional, Set

logger = logging.getLogger(__name__)

ShutdownHook = Callable[[], Awaitable[Any]]


class GracefulShutdownHandler:
    """Manages graceful shutdown with a global timeout budget.

    The timeout is shared across drain + hooks phases to ensure
    total shutdown time stays within Docker/K8s stop_grace_period.
    """

    def __init__(self, timeout: int = 30) -> None:
        if timeout < 1:
            raise ValueError(f"timeout must be >= 1 second, got {timeout}")
        self._timeout = timeout
        self._hooks: List[ShutdownHook] = []
        self._in_flight: Set[asyncio.Task[Any]] = set()
        self._shutting_down = False
        self._shutdown_complete = False

    def register_hook(self, hook: ShutdownHook) -> None:
        """Register an async cleanup hook to run during shutdown."""
        self._hooks.append(hook)

    def track_task(self, task: asyncio.Task[Any]) -> None:
        """Track an in-flight task that must complete before shutdown.

        Rejects new tasks once shutdown has started to prevent leaks.
        """
        if self._shutting_down:
            task.cancel()
            return
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)

    def install_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Install SIGTERM/SIGINT handlers on the event loop."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

        if sys.platform == "win32":
            signal.signal(signal.SIGINT, self._sync_signal_handler)
        else:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.ensure_future(self.shutdown())
                )

    def _sync_signal_handler(self, signum: int, frame: Any) -> None:
        """Fallback signal handler for Windows."""
        if self._shutting_down:
            return
        logger.info("Signal %d received, initiating shutdown...", signum)
        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(self.shutdown())
        except RuntimeError:
            pass  # No running loop — process is exiting anyway

    async def shutdown(self) -> None:
        """Execute graceful shutdown with global timeout budget.

        Allocates ~70% of timeout to draining tasks, ~30% to hooks.
        Idempotent: safe to call multiple times.
        """
        if self._shutdown_complete or self._shutting_down:
            return
        self._shutting_down = True

        deadline = time.monotonic() + self._timeout
        drain_budget = self._timeout * 0.7
        logger.info(
            "Graceful shutdown started (timeout=%ds, in_flight=%d, hooks=%d)",
            self._timeout,
            len(self._in_flight),
            len(self._hooks),
        )

        # Phase 1: Drain in-flight tasks (70% of budget)
        if self._in_flight:
            snapshot = set(self._in_flight)
            logger.info("Draining %d in-flight tasks...", len(snapshot))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*snapshot, return_exceptions=True),
                    timeout=drain_budget,
                )
            except asyncio.TimeoutError:
                logger.warning("Drain timeout, cancelling remaining tasks...")
                for task in snapshot:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*snapshot, return_exceptions=True)

        # Phase 2: Run cleanup hooks (remaining budget)
        for hook in self._hooks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Shutdown deadline exceeded, skipping remaining hooks")
                break
            try:
                await asyncio.wait_for(hook(), timeout=max(remaining, 0.1))
            except asyncio.TimeoutError:
                logger.warning("Shutdown hook %s timed out", hook.__name__)
            except Exception:
                logger.exception("Shutdown hook %s failed", hook.__name__)

        self._shutdown_complete = True
        logger.info("Graceful shutdown complete")
