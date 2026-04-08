"""scion.observability — Metrics, tracing, health, and SLOs for Scion.

Provides Prometheus-compatible metrics, structured execution traces,
aggregated health checks, and SLO tracking for the MCP server and
grounding engine.

Epics 6.1–6.2 — Phase 6 (Operability).
"""

from scion.observability.health import HealthAggregator, HealthStatus
from scion.observability.metrics import MetricsRegistry
from scion.observability.slos import SLOEvaluator
from scion.observability.tracing import ExecutionTracer, trace_async

__all__ = [
    "MetricsRegistry",
    "ExecutionTracer",
    "trace_async",
    "HealthAggregator",
    "HealthStatus",
    "SLOEvaluator",
]
