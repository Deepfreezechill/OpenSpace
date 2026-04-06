"""Aggregated health checks for OpenSpace components.

Collects health status from multiple subsystems (MCP server, grounding
engine, skill store, evolution pipeline) into a single structured response
suitable for load balancers, uptime monitors, and ``/health`` endpoints.

Usage::

    from openspace.observability.health import health, HealthProbe

    # Register probes
    health.register("skill_store", lambda: HealthProbe(ok=True, detail="42 skills"))
    health.register("llm", lambda: HealthProbe(ok=ping_llm(), detail="gpt-4o"))

    # Check aggregate
    status = health.check()
    # {"status": "healthy", "checks": {"skill_store": {...}, "llm": {...}}}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HealthStatus(str, Enum):
    """Overall system health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthProbe:
    """Result from a single health probe."""

    ok: bool
    detail: str = ""
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"ok": self.ok, "detail": self.detail}
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 2)
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class HealthAggregator:
    """Collects and evaluates health probes from registered subsystems.

    Probes are callables that return a :class:`HealthProbe`. They are
    evaluated lazily when :meth:`check` is called.

    Status logic:
      - All probes OK → ``healthy``
      - ≥ 1 probe failed, but majority OK → ``degraded``
      - Majority failed → ``unhealthy``
    """

    def __init__(self) -> None:
        self._probes: Dict[str, Callable[[], HealthProbe]] = {}
        self._start_time = time.time()

    def register(self, name: str, probe: Callable[[], HealthProbe]) -> None:
        """Register a named health probe."""
        self._probes[name] = probe

    def unregister(self, name: str) -> None:
        """Remove a health probe."""
        self._probes.pop(name, None)

    @property
    def probe_names(self) -> List[str]:
        return list(self._probes.keys())

    def check(self) -> Dict[str, Any]:
        """Run all probes and return structured health status."""
        results: Dict[str, Dict[str, Any]] = {}
        failed = 0
        total = len(self._probes)

        for name, probe_fn in self._probes.items():
            start = time.monotonic()
            try:
                probe = probe_fn()
                probe.latency_ms = (time.monotonic() - start) * 1000
                results[name] = probe.to_dict()
                if not probe.ok:
                    failed += 1
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                results[name] = HealthProbe(
                    ok=False,
                    detail=f"probe error: {type(exc).__name__}",
                    latency_ms=elapsed,
                ).to_dict()
                failed += 1

        if total == 0:
            status = HealthStatus.HEALTHY
        elif failed == 0:
            status = HealthStatus.HEALTHY
        elif failed < total / 2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return {
            "status": status.value,
            "checks": results,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "total_probes": total,
            "failed_probes": failed,
        }


# Module-level singleton
health = HealthAggregator()
