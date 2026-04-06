"""Tests for openspace.observability — metrics, tracing, health.

Epic 6.1: Observability (metrics, tracing, health).
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from openspace.observability.health import HealthAggregator, HealthProbe, HealthStatus
from openspace.observability.metrics import MetricsRegistry
from openspace.observability.tracing import ExecutionTracer, Span, Trace, trace_async


# ═══════════════════════════════════════════════════════════════════════
# Metrics tests
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsRegistry:
    def test_creates_with_isolated_registry(self):
        reg = MetricsRegistry()
        assert reg.registry is not None
        assert isinstance(reg.registry, CollectorRegistry)

    def test_render_produces_prometheus_format(self):
        reg = MetricsRegistry()
        reg.execution_total.labels(agent="test", status="success").inc()
        output = reg.render()
        assert b"openspace_execution_total" in output
        assert b'agent="test"' in output

    def test_execution_latency_histogram(self):
        reg = MetricsRegistry()
        reg.execution_latency.labels(agent="GA", status="success").observe(1.5)
        output = reg.render()
        assert b"openspace_execution_latency_seconds" in output

    def test_skill_hits_counter(self):
        reg = MetricsRegistry()
        reg.skill_hits.labels(skill_id="web-search").inc()
        reg.skill_hits.labels(skill_id="web-search").inc()
        output = reg.render()
        assert b"openspace_skill_hits_total" in output
        assert b'skill_id="web-search"' in output

    def test_evolution_metrics(self):
        reg = MetricsRegistry()
        reg.evolution_total.labels(trigger="analysis", outcome="success").inc()
        reg.evolution_latency.labels(trigger="analysis").observe(2.3)
        output = reg.render()
        assert b"openspace_evolution_total" in output
        assert b"openspace_evolution_latency_seconds" in output

    def test_tool_call_metrics(self):
        reg = MetricsRegistry()
        reg.tool_calls_total.labels(backend="gui", tool_name="click", status="ok").inc()
        reg.tool_call_latency.labels(backend="gui").observe(0.5)
        output = reg.render()
        assert b"openspace_tool_calls_total" in output

    def test_track_execution_context_manager_success(self):
        reg = MetricsRegistry()
        with reg.track_execution(agent="test"):
            pass  # simulate work
        output = reg.render().decode()
        assert 'agent="test"' in output
        assert 'status="success"' in output

    def test_track_execution_context_manager_error(self):
        reg = MetricsRegistry()
        with pytest.raises(ValueError):
            with reg.track_execution(agent="err"):
                raise ValueError("boom")
        output = reg.render().decode()
        assert 'status="error"' in output

    def test_in_flight_gauge(self):
        reg = MetricsRegistry()
        # Before
        sample_before = reg.execution_in_flight.labels(agent="test")._value.get()
        assert sample_before == 0.0

    def test_separate_registries_are_isolated(self):
        r1 = MetricsRegistry()
        r2 = MetricsRegistry()
        r1.execution_total.labels(agent="a", status="success").inc()
        output2 = r2.render()
        # r2 should not have r1's data
        assert b'agent="a"' not in output2


# ═══════════════════════════════════════════════════════════════════════
# Tracing tests
# ═══════════════════════════════════════════════════════════════════════


class TestSpan:
    def test_span_creation(self):
        s = Span(name="test", trace_id="abc")
        assert s.name == "test"
        assert s.trace_id == "abc"
        assert s.end_time is None
        assert s.duration_ms is None

    def test_span_finish(self):
        s = Span(name="test", trace_id="abc")
        s.finish()
        assert s.end_time is not None
        assert s.duration_ms is not None
        assert s.duration_ms >= 0
        assert s.status == "ok"

    def test_span_finish_with_error(self):
        s = Span(name="test", trace_id="abc")
        s.finish(status="error")
        assert s.status == "error"

    def test_span_add_event(self):
        s = Span(name="test", trace_id="abc")
        s.add_event("tool_call", tool="click")
        assert len(s.events) == 1
        assert s.events[0]["name"] == "tool_call"
        assert s.events[0]["tool"] == "click"

    def test_span_to_dict(self):
        s = Span(name="test", trace_id="abc")
        s.finish()
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["trace_id"] == "abc"
        assert "duration_ms" in d


class TestTrace:
    def test_trace_has_unique_id(self):
        t1 = Trace()
        t2 = Trace()
        assert t1.trace_id != t2.trace_id

    def test_trace_to_dict(self):
        t = Trace()
        s = Span(name="root", trace_id=t.trace_id)
        t.spans.append(s)
        d = t.to_dict()
        assert d["trace_id"] == t.trace_id
        assert d["total_spans"] == 1
        assert d["root_span"]["name"] == "root"


class TestExecutionTracer:
    def test_start_and_finish_trace(self):
        tracer = ExecutionTracer()
        trace = tracer.start_trace("test")
        assert tracer.current_trace() is trace
        finished = tracer.finish_trace()
        assert finished is trace
        assert tracer.current_trace() is None

    def test_recent_traces_bounded(self):
        tracer = ExecutionTracer(max_traces=3)
        for i in range(5):
            tracer.start_trace(f"trace-{i}")
            tracer.finish_trace()
        assert len(tracer.recent_traces) == 3

    def test_span_context_manager(self):
        tracer = ExecutionTracer()
        tracer.start_trace("root")
        with tracer.span("child", key="value") as child:
            assert child.name == "child"
            assert child.attributes["key"] == "value"
            assert child.parent_id is not None
        trace = tracer.finish_trace()
        assert len(trace.spans) == 2  # root + child

    def test_nested_spans(self):
        tracer = ExecutionTracer()
        tracer.start_trace("root")
        with tracer.span("level1") as l1:
            with tracer.span("level2") as l2:
                assert l2.parent_id == l1.span_id
        trace = tracer.finish_trace()
        assert len(trace.spans) == 3

    def test_span_error_status(self):
        tracer = ExecutionTracer()
        tracer.start_trace("root")
        with pytest.raises(ValueError):
            with tracer.span("failing"):
                raise ValueError("oops")
        trace = tracer.finish_trace()
        failing = [s for s in trace.spans if s.name == "failing"][0]
        assert failing.status == "error"

    def test_clear(self):
        tracer = ExecutionTracer()
        tracer.start_trace("root")
        tracer.finish_trace()
        assert len(tracer.recent_traces) == 1
        tracer.clear()
        assert len(tracer.recent_traces) == 0

    @pytest.mark.asyncio
    async def test_async_span_context_manager(self):
        tracer = ExecutionTracer()
        tracer.start_trace("root")
        async with tracer.span("async_child") as child:
            assert child.name == "async_child"
        trace = tracer.finish_trace()
        assert len(trace.spans) == 2


class TestTraceAsyncDecorator:
    @pytest.mark.asyncio
    async def test_decorator_creates_span(self):
        test_tracer = ExecutionTracer()

        @trace_async("my_func", tracer_instance=test_tracer)
        async def my_func():
            return 42

        result = await my_func()
        assert result == 42
        assert len(test_tracer.recent_traces) == 1
        trace = test_tracer.recent_traces[0]
        assert any(s.name == "my_func" for s in trace.spans)

    @pytest.mark.asyncio
    async def test_decorator_propagates_exceptions(self):
        test_tracer = ExecutionTracer()

        @trace_async("failing", tracer_instance=test_tracer)
        async def failing():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await failing()


# ═══════════════════════════════════════════════════════════════════════
# Health tests
# ═══════════════════════════════════════════════════════════════════════


class TestHealthProbe:
    def test_healthy_probe(self):
        p = HealthProbe(ok=True, detail="42 skills")
        d = p.to_dict()
        assert d["ok"] is True
        assert d["detail"] == "42 skills"

    def test_unhealthy_probe(self):
        p = HealthProbe(ok=False, detail="connection refused")
        d = p.to_dict()
        assert d["ok"] is False

    def test_probe_with_metadata(self):
        p = HealthProbe(ok=True, detail="ok", metadata={"count": 5})
        d = p.to_dict()
        assert d["metadata"]["count"] == 5

    def test_probe_with_latency(self):
        p = HealthProbe(ok=True, detail="ok", latency_ms=12.345)
        d = p.to_dict()
        assert d["latency_ms"] == 12.35


class TestHealthAggregator:
    def test_healthy_when_all_probes_ok(self):
        agg = HealthAggregator()
        agg.register("a", lambda: HealthProbe(ok=True, detail="ok"))
        agg.register("b", lambda: HealthProbe(ok=True, detail="ok"))
        result = agg.check()
        assert result["status"] == "healthy"
        assert result["failed_probes"] == 0

    def test_degraded_when_minority_fail(self):
        agg = HealthAggregator()
        agg.register("a", lambda: HealthProbe(ok=True, detail="ok"))
        agg.register("b", lambda: HealthProbe(ok=True, detail="ok"))
        agg.register("c", lambda: HealthProbe(ok=False, detail="down"))
        result = agg.check()
        assert result["status"] == "degraded"
        assert result["failed_probes"] == 1

    def test_unhealthy_when_majority_fail(self):
        agg = HealthAggregator()
        agg.register("a", lambda: HealthProbe(ok=False, detail="down"))
        agg.register("b", lambda: HealthProbe(ok=False, detail="down"))
        agg.register("c", lambda: HealthProbe(ok=True, detail="ok"))
        result = agg.check()
        assert result["status"] == "unhealthy"

    def test_healthy_when_no_probes(self):
        agg = HealthAggregator()
        result = agg.check()
        assert result["status"] == "healthy"
        assert result["total_probes"] == 0

    def test_probe_exception_is_caught(self):
        agg = HealthAggregator()

        def bad_probe():
            raise ConnectionError("timeout")

        agg.register("flaky", bad_probe)
        result = agg.check()
        assert result["status"] == "unhealthy"
        assert result["checks"]["flaky"]["ok"] is False
        assert "ConnectionError" in result["checks"]["flaky"]["detail"]

    def test_uptime_is_positive(self):
        agg = HealthAggregator()
        result = agg.check()
        assert result["uptime_seconds"] >= 0

    def test_unregister_probe(self):
        agg = HealthAggregator()
        agg.register("a", lambda: HealthProbe(ok=True))
        agg.unregister("a")
        assert "a" not in agg.probe_names

    def test_probe_names(self):
        agg = HealthAggregator()
        agg.register("x", lambda: HealthProbe(ok=True))
        agg.register("y", lambda: HealthProbe(ok=True))
        assert set(agg.probe_names) == {"x", "y"}

    def test_check_measures_probe_latency(self):
        def slow_probe():
            time.sleep(0.01)
            return HealthProbe(ok=True, detail="slow")

        agg = HealthAggregator()
        agg.register("slow", slow_probe)
        result = agg.check()
        assert result["checks"]["slow"]["latency_ms"] >= 5


# ═══════════════════════════════════════════════════════════════════════
# MCP tool handler tests
# ═══════════════════════════════════════════════════════════════════════


class TestMCPObservabilityTools:
    @pytest.mark.asyncio
    async def test_health_check_returns_json(self):
        from openspace.mcp.tool_handlers import health_check

        result = await health_check()
        parsed = json.loads(result)
        assert "status" in parsed
        assert "checks" in parsed

    @pytest.mark.asyncio
    async def test_get_metrics_returns_prometheus_format(self):
        from openspace.mcp.tool_handlers import get_metrics

        result = await get_metrics()
        assert isinstance(result, str)
        # Should contain at least one metric name
        assert "openspace_" in result

    @pytest.mark.asyncio
    async def test_get_execution_traces_returns_json(self):
        from openspace.mcp.tool_handlers import get_execution_traces

        result = await get_execution_traces(limit=3)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @pytest.mark.asyncio
    async def test_get_execution_traces_limit_clamped(self):
        from openspace.mcp.tool_handlers import get_execution_traces

        result = await get_execution_traces(limit=999)
        parsed = json.loads(result)
        # Should not crash, just return what's available
        assert isinstance(parsed, list)


# ═══════════════════════════════════════════════════════════════════════
# Integration: execution instrumentation
# ═══════════════════════════════════════════════════════════════════════


class TestExecutionInstrumentation:
    def test_execution_module_imports_observability(self):
        """Verify the execution loop has observability wired in."""
        import openspace.agents.grounding.execution as ex

        assert hasattr(ex, "_metrics")
        assert hasattr(ex, "_tracer")

    def test_metrics_singleton_is_accessible(self):
        from openspace.observability.metrics import metrics

        assert isinstance(metrics, MetricsRegistry)

    def test_tracer_singleton_is_accessible(self):
        from openspace.observability.tracing import tracer

        assert isinstance(tracer, ExecutionTracer)


# ═══════════════════════════════════════════════════════════════════════
# Package completeness
# ═══════════════════════════════════════════════════════════════════════


class TestPackageCompleteness:
    def test_package_exports(self):
        import openspace.observability as obs

        assert hasattr(obs, "MetricsRegistry")
        assert hasattr(obs, "ExecutionTracer")
        assert hasattr(obs, "trace_async")
        assert hasattr(obs, "HealthAggregator")
        assert hasattr(obs, "HealthStatus")

    def test_mcp_registers_observability_tools(self):
        """Verify register_handlers wires observability tools."""
        mock_mcp = MagicMock()
        from openspace.mcp.tool_handlers import register_handlers

        register_handlers(mock_mcp)
        # 4 original + 3 observability = 7 calls
        assert mock_mcp.tool.call_count == 7
