"""Prometheus-compatible metrics for OpenSpace.

Defines counters, histograms, and gauges covering the three key observability
axes: grounding execution, skill engine, and evolution pipeline.

All metrics are registered on a single :class:`CollectorRegistry` so they
can be rendered at ``/metrics`` without colliding with the default registry
(important for test isolation).

Usage::

    from openspace.observability.metrics import metrics

    metrics.execution_latency.labels(agent="GroundingAgent").observe(1.23)
    metrics.skill_hits.labels(skill_id="web-search").inc()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Dedicated registry — avoids pollution of the global default registry
# and ensures test isolation via fresh MetricsRegistry instances.
_BUCKETS_LATENCY = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
_BUCKETS_ITERATIONS = (1, 2, 3, 5, 8, 10, 15)


class MetricsRegistry:
    """Container for all OpenSpace Prometheus metrics.

    Instantiate once at app startup; pass the same instance to all
    instrumented components.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or CollectorRegistry()

        # ── Grounding execution ──────────────────────────────────────
        self.execution_latency = Histogram(
            "openspace_execution_latency_seconds",
            "End-to-end execution latency for a grounding task",
            labelnames=["agent", "status"],
            buckets=_BUCKETS_LATENCY,
            registry=self._registry,
        )

        self.execution_iterations = Histogram(
            "openspace_execution_iterations",
            "Number of LLM iterations per grounding execution",
            labelnames=["agent"],
            buckets=_BUCKETS_ITERATIONS,
            registry=self._registry,
        )

        self.execution_total = Counter(
            "openspace_execution_total",
            "Total grounding executions",
            labelnames=["agent", "status"],
            registry=self._registry,
        )

        self.execution_in_flight = Gauge(
            "openspace_execution_in_flight",
            "Currently running grounding executions",
            labelnames=["agent"],
            registry=self._registry,
        )

        # ── Skill engine ─────────────────────────────────────────────
        self.skill_hits = Counter(
            "openspace_skill_hits_total",
            "Tasks resolved by a matching skill (skill-first path)",
            labelnames=["skill_id"],
            registry=self._registry,
        )

        self.skill_misses = Counter(
            "openspace_skill_misses_total",
            "Tasks that fell through to the grounding fallback",
            registry=self._registry,
        )

        self.skill_search_latency = Histogram(
            "openspace_skill_search_latency_seconds",
            "Skill search/retrieval latency",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self._registry,
        )

        # ── Evolution pipeline ───────────────────────────────────────
        self.evolution_total = Counter(
            "openspace_evolution_total",
            "Total evolution attempts",
            labelnames=["trigger", "outcome"],
            registry=self._registry,
        )

        self.evolution_latency = Histogram(
            "openspace_evolution_latency_seconds",
            "Evolution pipeline latency",
            labelnames=["trigger"],
            buckets=_BUCKETS_LATENCY,
            registry=self._registry,
        )

        # ── Tool execution ───────────────────────────────────────────
        self.tool_calls_total = Counter(
            "openspace_tool_calls_total",
            "Total tool invocations across all backends",
            labelnames=["backend", "tool_name", "status"],
            registry=self._registry,
        )

        self.tool_call_latency = Histogram(
            "openspace_tool_call_latency_seconds",
            "Individual tool call latency",
            labelnames=["backend"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

    def render(self) -> bytes:
        """Render all metrics in Prometheus exposition format."""
        return generate_latest(self._registry)

    @contextmanager
    def track_execution(
        self, agent: str = "GroundingAgent"
    ) -> Generator[None, None, None]:
        """Context manager that tracks execution latency, count, and in-flight gauge."""
        self.execution_in_flight.labels(agent=agent).inc()
        start = time.monotonic()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.monotonic() - start
            self.execution_latency.labels(agent=agent, status=status).observe(elapsed)
            self.execution_total.labels(agent=agent, status=status).inc()
            self.execution_in_flight.labels(agent=agent).dec()


# Module-level singleton for convenience (tests should create their own).
metrics = MetricsRegistry()
