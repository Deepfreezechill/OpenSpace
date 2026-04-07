"""Graceful shutdown handler for OpenSpace.

Provides signal-aware shutdown that:
1. Stops accepting new requests
2. Drains in-flight tasks (with timeout)
3. Runs registered cleanup hooks
4. Exits cleanly

Usage::

    from openspace.deploy.shutdown import GracefulShutdownHandler

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
from typing import Any, Awaitable, Callable, List, Optional, Set

logger = logging.getLogger(__name__)

ShutdownHook = Callable[[], Awaitable[Any]]


class GracefulShutdownHandler:
    """Manages graceful shutdown with timeout and hook execution.

    Tracks in-flight async tasks and ensures they complete (or are
    cancelled) before running cleanup hooks.
    """

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout
        self._hooks: List[ShutdownHook] = []
        self._in_flight: Set[asyncio.Task[Any]] = set()
        self._shutting_down = False
        self._shutdown_complete = False

    def register_hook(self, hook: ShutdownHook) -> None:
        """Register an async cleanup hook to run during shutdown."""
        self._hooks.append(hook)

    def track_task(self, task: asyncio.Task[Any]) -> None:
        """Track an in-flight task that must complete before shutdown."""
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)

    def install_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Install SIGTERM/SIGINT handlers on the event loop.

        On Windows, signal handlers are limited — only SIGINT is
        supported via signal.signal(). On Unix, uses loop.add_signal_handler().
        """
        if loop is None:
            loop = asyncio.get_event_loop()

        if sys.platform == "win32":
            # Windows: use signal module (only SIGINT supported)
            signal.signal(signal.SIGINT, self._sync_signal_handler)
        else:
            # Unix: proper async signal handling
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.ensure_future(self.shutdown())
                )

    def _sync_signal_handler(self, signum: int, frame: Any) -> None:
        """Fallback signal handler for Windows."""
        if self._shutting_down:
            return
        logger.info("Signal %d received, initiating shutdown...", signum)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self.shutdown())
        else:
            loop.run_until_complete(self.shutdown())

    async def shutdown(self) -> None:
        """Execute graceful shutdown sequence.

        1. Drain in-flight tasks (with timeout)
        2. Run cleanup hooks (best-effort, with timeout)
        3. Mark shutdown complete

        Idempotent: safe to call multiple times.
        """
        if self._shutdown_complete:
            return
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info(
            "Graceful shutdown started (timeout=%ds, in_flight=%d)",
            self._timeout,
            len(self._in_flight),
        )

        # Phase 1: Drain in-flight tasks
        if self._in_flight:
            logger.info("Waiting for %d in-flight tasks...", len(self._in_flight))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._in_flight, return_exceptions=True),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout draining %d in-flight tasks, cancelling...",
                    len(self._in_flight),
                )
                for task in self._in_flight:
                    task.cancel()
                # Give cancelled tasks a moment to clean up
                await asyncio.gather(*self._in_flight, return_exceptions=True)

        # Phase 2: Run cleanup hooks
        for hook in self._hooks:
            try:
                await asyncio.wait_for(hook(), timeout=self._timeout)
            except asyncio.TimeoutError:
                logger.warning("Shutdown hook %s timed out", hook.__name__)
            except Exception:
                logger.exception("Shutdown hook %s failed", hook.__name__)

        self._shutdown_complete = True
        logger.info("Graceful shutdown complete")
