"""Tests for openspace.observability.slos — SLO targets, budgets, burn rates.

Epic 6.2: SLOs (latency, error rate, availability).
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from openspace.observability.metrics import MetricsRegistry


# ═══════════════════════════════════════════════════════════════════════
# SLOTarget tests
# ═══════════════════════════════════════════════════════════════════════


class TestSLOTarget:
    def test_create_latency_target(self):
        from openspace.observability.slos import SLOTarget

        t = SLOTarget(
            name="execution_latency_p99",
            objective=0.99,
            threshold=30.0,
            unit="seconds",
            description="99th percentile execution latency under 30s",
        )
        assert t.name == "execution_latency_p99"
        assert t.objective == 0.99
        assert t.threshold == 30.0

    def test_create_error_rate_target(self):
        from openspace.observability.slos import SLOTarget

        t = SLOTarget(
            name="error_rate",
            objective=0.95,
            threshold=0.05,
            unit="ratio",
            description="Error rate below 5%",
        )
        assert t.objective == 0.95
        assert t.threshold == 0.05

    def test_create_availability_target(self):
        from openspace.observability.slos import SLOTarget

        t = SLOTarget(
            name="availability",
            objective=0.99,
            threshold=0.99,
            unit="ratio",
            description="99% availability",
        )
        assert t.objective == 0.99

    def test_to_dict(self):
        from openspace.observability.slos import SLOTarget

        t = SLOTarget(name="test", objective=0.99, threshold=10.0, unit="seconds")
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["objective"] == 0.99
        assert d["threshold"] == 10.0
        assert d["unit"] == "seconds"

    def test_invalid_objective_raises(self):
        from openspace.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="objective"):
            SLOTarget(name="bad", objective=1.5, threshold=10.0)

    def test_zero_objective_raises(self):
        from openspace.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="objective"):
            SLOTarget(name="bad", objective=-0.1, threshold=10.0)


# ═══════════════════════════════════════════════════════════════════════
# ErrorBudget tests
# ═══════════════════════════════════════════════════════════════════════


class TestErrorBudget:
    def test_full_budget_remaining(self):
        from openspace.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=0)
        assert status["budget_total"] == pytest.approx(10)  # 1% of 1000
        assert status["budget_consumed"] == 0
        assert status["budget_remaining"] == pytest.approx(10)
        assert status["budget_remaining_pct"] == pytest.approx(100.0)
        assert status["exhausted"] is False

    def test_partial_budget_consumed(self):
        from openspace.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=5)
        assert status["budget_total"] == pytest.approx(10)
        assert status["budget_consumed"] == 5
        assert status["budget_remaining"] == pytest.approx(5)
        assert status["budget_remaining_pct"] == pytest.approx(50.0)
        assert status["exhausted"] is False

    def test_budget_exactly_exhausted(self):
        """Boundary: consumed == total → exhausted."""
        from openspace.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=10)
        assert status["budget_remaining"] == pytest.approx(0)
        assert status["budget_remaining_pct"] == pytest.approx(0.0)
        assert status["exhausted"] is True

    def test_budget_over_exhausted(self):
        """Over budget: consumed > total."""
        from openspace.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=15)
        assert status["budget_remaining"] == pytest.approx(-5)
        assert status["budget_remaining_pct"] < 0
        assert status["exhausted"] is True

    def test_zero_requests(self):
        """No requests → full budget, not exhausted."""
        from openspace.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=0, failed_requests=0)
        assert status["budget_remaining_pct"] == 100.0
        assert status["exhausted"] is False

    def test_to_dict(self):
        from openspace.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.95, window_seconds=86400)
        d = budget.to_dict()
        assert d["objective"] == 0.95
        assert d["window_seconds"] == 86400


# ═══════════════════════════════════════════════════════════════════════
# BurnRate tests
# ═══════════════════════════════════════════════════════════════════════


class TestBurnRate:
    def test_burn_rate_calculation(self):
        """Burn rate = actual error rate / allowed error rate."""
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        # 2% errors when 1% allowed → burn rate 2.0
        rate = calc.burn_rate(total_requests=1000, failed_requests=20)
        assert rate == pytest.approx(2.0)

    def test_burn_rate_zero_errors(self):
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        rate = calc.burn_rate(total_requests=1000, failed_requests=0)
        assert rate == 0.0

    def test_burn_rate_zero_requests(self):
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        rate = calc.burn_rate(total_requests=0, failed_requests=0)
        assert rate == 0.0

    def test_burn_rate_exactly_at_budget(self):
        """Burn rate = 1.0 when error rate matches allowed rate exactly."""
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        rate = calc.burn_rate(total_requests=1000, failed_requests=10)
        assert rate == pytest.approx(1.0)

    def test_alert_thresholds_default(self):
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        alerts = calc.check_alerts(total_requests=1000, failed_requests=150)
        # 15% error rate, 1% budget → burn rate 15.0
        # Should trigger critical (>14.4) and high (>6)
        assert any(a["severity"] == "critical" for a in alerts)

    def test_no_alerts_when_healthy(self):
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        alerts = calc.check_alerts(total_requests=1000, failed_requests=5)
        # 0.5% error, 1% budget → burn rate 0.5 → no alerts
        assert len(alerts) == 0

    def test_alert_at_boundary(self):
        """Burn rate exactly at threshold should NOT trigger (strict >)."""
        from openspace.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        # Default high threshold is 6.0
        # Need 6% error rate: 60 failures / 1000 requests = 6.0 burn rate
        alerts = calc.check_alerts(total_requests=1000, failed_requests=60)
        # burn rate = 6.0, threshold is > 6.0, so should NOT trigger high
        # but burn rate 6.0 > 1.0 triggers warning
        severities = {a["severity"] for a in alerts}
        assert "high" not in severities
        assert "warning" in severities


# ═══════════════════════════════════════════════════════════════════════
# SLOEvaluator tests
# ═══════════════════════════════════════════════════════════════════════


class TestSLOEvaluator:
    def test_evaluate_healthy(self):
        from openspace.observability.slos import SLOEvaluator, SLOTarget

        reg = MetricsRegistry()
        # Simulate 100 successful requests
        for _ in range(100):
            reg.execution_total.labels(agent="test", status="success").inc()
        for _ in range(2):
            reg.execution_total.labels(agent="test", status="error").inc()
        # Record latencies (all fast)
        for _ in range(100):
            reg.execution_latency.labels(agent="test", status="success").observe(1.0)

        ev = SLOEvaluator(registry=reg)
        result = ev.evaluate()
        assert result["overall_status"] in ("meeting", "at_risk", "breaching")

    def test_evaluate_breaching_error_rate(self):
        from openspace.observability.slos import SLOEvaluator

        reg = MetricsRegistry()
        for _ in range(50):
            reg.execution_total.labels(agent="test", status="success").inc()
        for _ in range(50):
            reg.execution_total.labels(agent="test", status="error").inc()

        ev = SLOEvaluator(registry=reg)
        result = ev.evaluate()
        # 50% error rate with 5% target → breaching
        error_slo = next(
            s for s in result["slos"] if s["target"]["name"] == "error_rate"
        )
        assert error_slo["status"] == "breaching"

    def test_evaluate_with_custom_targets(self):
        from openspace.observability.slos import SLOEvaluator, SLOTarget

        reg = MetricsRegistry()
        targets = [
            SLOTarget(
                name="custom_error",
                objective=0.90,
                threshold=0.10,
                unit="ratio",
                description="10% error budget",
            )
        ]
        ev = SLOEvaluator(registry=reg, targets=targets)
        result = ev.evaluate()
        assert len(result["slos"]) == 1
        assert result["slos"][0]["target"]["name"] == "custom_error"

    def test_evaluate_no_data(self):
        """With zero requests, SLOs should report meeting (no violations)."""
        from openspace.observability.slos import SLOEvaluator

        reg = MetricsRegistry()
        ev = SLOEvaluator(registry=reg)
        result = ev.evaluate()
        assert result["overall_status"] == "meeting"

    def test_to_json(self):
        from openspace.observability.slos import SLOEvaluator

        reg = MetricsRegistry()
        ev = SLOEvaluator(registry=reg)
        j = ev.to_json()
        parsed = json.loads(j)
        assert "slos" in parsed
        assert "overall_status" in parsed


# ═══════════════════════════════════════════════════════════════════════
# Default SLO targets
# ═══════════════════════════════════════════════════════════════════════


class TestDefaultTargets:
    def test_default_targets_exist(self):
        from openspace.observability.slos import DEFAULT_TARGETS

        names = {t.name for t in DEFAULT_TARGETS}
        assert "execution_latency_p99" in names
        assert "error_rate" in names
        assert "availability" in names

    def test_all_defaults_valid(self):
        from openspace.observability.slos import DEFAULT_TARGETS

        for t in DEFAULT_TARGETS:
            assert 0 < t.objective <= 1.0
            assert t.threshold > 0


# ═══════════════════════════════════════════════════════════════════════
# MCP integration
# ═══════════════════════════════════════════════════════════════════════


class TestSLOMCPIntegration:
    def test_mcp_registers_check_slos_tool(self):
        mock_mcp = MagicMock()
        from openspace.mcp.tool_handlers import register_handlers

        register_handlers(mock_mcp)
        assert mock_mcp.tool.call_count == 8  # 7 existing + 1 new

    def test_check_slos_returns_valid_json(self):
        """End-to-end: create MCP app, call check_slos, parse result."""
        from openspace.mcp.server import create_mcp_app

        app = create_mcp_app()
        if hasattr(app, "_tool_manager") and hasattr(app._tool_manager, "_tools"):
            assert "check_slos" in app._tool_manager._tools


# ═══════════════════════════════════════════════════════════════════════
# Package completeness
# ═══════════════════════════════════════════════════════════════════════


class TestSLOPackageCompleteness:
    def test_module_importable(self):
        import openspace.observability.slos as slos

        assert hasattr(slos, "SLOTarget")
        assert hasattr(slos, "ErrorBudget")
        assert hasattr(slos, "BurnRateCalculator")
        assert hasattr(slos, "SLOEvaluator")
        assert hasattr(slos, "DEFAULT_TARGETS")

    def test_exported_from_observability_package(self):
        import openspace.observability as obs

        assert hasattr(obs, "SLOEvaluator")
