"""openspace.observability — Metrics, tracing, and health for OpenSpace.

Provides Prometheus-compatible metrics, structured execution traces,
and aggregated health checks for the MCP server and grounding engine.

Epic 6.1 — Phase 6 (Operability).
"""

from openspace.observability.health import HealthAggregator, HealthStatus
from openspace.observability.metrics import MetricsRegistry
from openspace.observability.tracing import ExecutionTracer, trace_async

__all__ = [
    "MetricsRegistry",
    "ExecutionTracer",
    "trace_async",
    "HealthAggregator",
    "HealthStatus",
]
