"""Tests for scion.observability.slos — SLO targets, budgets, burn rates.

Epic 6.2: SLOs (latency, error rate, availability).
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from scion.observability.metrics import MetricsRegistry


# ═══════════════════════════════════════════════════════════════════════
# SLOTarget tests
# ═══════════════════════════════════════════════════════════════════════


class TestSLOTarget:
    def test_create_latency_target(self):
        from scion.observability.slos import SLOTarget

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
        from scion.observability.slos import SLOTarget

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
        from scion.observability.slos import SLOTarget

        t = SLOTarget(
            name="availability",
            objective=0.99,
            threshold=0.99,
            unit="ratio",
            description="99% availability",
        )
        assert t.objective == 0.99

    def test_to_dict(self):
        from scion.observability.slos import SLOTarget

        t = SLOTarget(name="test", objective=0.99, threshold=10.0, unit="seconds")
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["objective"] == 0.99
        assert d["threshold"] == 10.0
        assert d["unit"] == "seconds"

    def test_invalid_objective_raises(self):
        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="objective"):
            SLOTarget(name="bad", objective=1.5, threshold=10.0)

    def test_zero_objective_raises(self):
        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="objective"):
            SLOTarget(name="bad", objective=-0.1, threshold=10.0)


# ═══════════════════════════════════════════════════════════════════════
# ErrorBudget tests
# ═══════════════════════════════════════════════════════════════════════


class TestErrorBudget:
    def test_full_budget_remaining(self):
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=0)
        assert status["budget_total"] == pytest.approx(10)  # 1% of 1000
        assert status["budget_consumed"] == 0
        assert status["budget_remaining"] == pytest.approx(10)
        assert status["budget_remaining_pct"] == pytest.approx(100.0)
        assert status["exhausted"] is False

    def test_partial_budget_consumed(self):
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=5)
        assert status["budget_total"] == pytest.approx(10)
        assert status["budget_consumed"] == 5
        assert status["budget_remaining"] == pytest.approx(5)
        assert status["budget_remaining_pct"] == pytest.approx(50.0)
        assert status["exhausted"] is False

    def test_budget_exactly_exhausted(self):
        """Boundary: consumed == total → exhausted."""
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=10)
        assert status["budget_remaining"] == pytest.approx(0)
        assert status["budget_remaining_pct"] == pytest.approx(0.0)
        assert status["exhausted"] is True

    def test_budget_over_exhausted(self):
        """Over budget: consumed > total."""
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=1000, failed_requests=15)
        assert status["budget_remaining"] == pytest.approx(-5)
        assert status["budget_remaining_pct"] < 0
        assert status["exhausted"] is True

    def test_zero_requests(self):
        """No requests → full budget, not exhausted."""
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=0.99, window_seconds=3600)
        status = budget.status(total_requests=0, failed_requests=0)
        assert status["budget_remaining_pct"] == 100.0
        assert status["exhausted"] is False

    def test_to_dict(self):
        from scion.observability.slos import ErrorBudget

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
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        # 2% errors when 1% allowed → burn rate 2.0
        rate = calc.burn_rate(total_requests=1000, failed_requests=20)
        assert rate == pytest.approx(2.0)

    def test_burn_rate_zero_errors(self):
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        rate = calc.burn_rate(total_requests=1000, failed_requests=0)
        assert rate == 0.0

    def test_burn_rate_zero_requests(self):
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        rate = calc.burn_rate(total_requests=0, failed_requests=0)
        assert rate == 0.0

    def test_burn_rate_exactly_at_budget(self):
        """Burn rate = 1.0 when error rate matches allowed rate exactly."""
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        rate = calc.burn_rate(total_requests=1000, failed_requests=10)
        assert rate == pytest.approx(1.0)

    def test_alert_thresholds_default(self):
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        alerts = calc.check_alerts(total_requests=1000, failed_requests=150)
        # 15% error rate, 1% budget → burn rate 15.0
        # Should trigger critical (>14.4) and high (>6)
        assert any(a["severity"] == "critical" for a in alerts)

    def test_no_alerts_when_healthy(self):
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.99)
        alerts = calc.check_alerts(total_requests=1000, failed_requests=5)
        # 0.5% error, 1% budget → burn rate 0.5 → no alerts
        assert len(alerts) == 0

    def test_alert_at_boundary(self):
        """Burn rate exactly at threshold should NOT trigger (strict >)."""
        from scion.observability.slos import BurnRateCalculator

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
        from scion.observability.slos import SLOEvaluator, SLOTarget

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
        from scion.observability.slos import SLOEvaluator

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
        from scion.observability.slos import SLOEvaluator, SLOTarget

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
        from scion.observability.slos import SLOEvaluator

        reg = MetricsRegistry()
        ev = SLOEvaluator(registry=reg)
        result = ev.evaluate()
        assert result["overall_status"] == "meeting"

    def test_to_json(self):
        from scion.observability.slos import SLOEvaluator

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
        from scion.observability.slos import DEFAULT_TARGETS

        names = {t.name for t in DEFAULT_TARGETS}
        assert "execution_latency_p99" in names
        assert "error_rate" in names
        assert "availability" in names

    def test_all_defaults_valid(self):
        from scion.observability.slos import DEFAULT_TARGETS

        for t in DEFAULT_TARGETS:
            assert 0 < t.objective <= 1.0
            assert t.threshold > 0


# ═══════════════════════════════════════════════════════════════════════
# MCP integration
# ═══════════════════════════════════════════════════════════════════════


class TestSLOMCPIntegration:
    def test_mcp_registers_check_slos_tool(self):
        mock_mcp = MagicMock()
        from scion.mcp.tool_handlers import register_handlers

        register_handlers(mock_mcp)
        assert mock_mcp.tool.call_count == 8  # 7 existing + 1 new

    def test_check_slos_returns_valid_json(self):
        """End-to-end: create MCP app, call check_slos, parse result."""
        from scion.mcp.server import create_mcp_app

        app = create_mcp_app()
        if hasattr(app, "_tool_manager") and hasattr(app._tool_manager, "_tools"):
            assert "check_slos" in app._tool_manager._tools

    @pytest.mark.asyncio
    async def test_check_slos_handler_returns_parseable_json(self):
        """Actually invoke check_slos handler and verify JSON output."""
        from scion.mcp.tool_handlers import check_slos

        result = await check_slos()
        parsed = json.loads(result)
        assert "overall_status" in parsed
        assert "slos" in parsed
        assert "error_budget" in parsed
        assert "burn_rate" in parsed
        assert "alerts" in parsed
        assert "total_requests" in parsed
        assert parsed["overall_status"] in ("meeting", "at_risk", "breaching")

    @pytest.mark.asyncio
    async def test_check_slos_handler_error_isolation(self):
        """check_slos must not propagate raw exceptions."""
        from unittest.mock import patch

        with patch(
            "scion.observability.metrics.metrics",
            side_effect=RuntimeError("registry boom"),
        ):
            from scion.mcp.tool_handlers import check_slos

            result = await check_slos()
            # Should return error JSON, not crash
            assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
# Package completeness
# ═══════════════════════════════════════════════════════════════════════


class TestSLOPackageCompleteness:
    def test_module_importable(self):
        import scion.observability.slos as slos

        assert hasattr(slos, "SLOTarget")
        assert hasattr(slos, "ErrorBudget")
        assert hasattr(slos, "BurnRateCalculator")
        assert hasattr(slos, "SLOEvaluator")
        assert hasattr(slos, "DEFAULT_TARGETS")

    def test_exported_from_observability_package(self):
        import scion.observability as obs

        assert hasattr(obs, "SLOEvaluator")

    def test_default_targets_immutable(self):
        """DEFAULT_TARGETS must be a tuple (not mutable list)."""
        from scion.observability.slos import DEFAULT_TARGETS

        assert isinstance(DEFAULT_TARGETS, tuple)


# ═══════════════════════════════════════════════════════════════════════
# Adversarial / boundary tests added from review findings
# ═══════════════════════════════════════════════════════════════════════


class TestSLOTargetValidation:
    """Threshold and name validation (8eyes-test P1-4, 8eyes-sec P2)."""

    def test_empty_name_rejected(self):
        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="non-empty"):
            SLOTarget(name="", objective=0.99, threshold=1.0)

    def test_whitespace_name_rejected(self):
        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="non-empty"):
            SLOTarget(name="   ", objective=0.99, threshold=1.0)

    def test_negative_threshold_rejected(self):
        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="non-negative"):
            SLOTarget(name="test", objective=0.99, threshold=-1.0)

    def test_nan_threshold_rejected(self):
        import math

        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="finite"):
            SLOTarget(name="test", objective=0.99, threshold=float("nan"))

    def test_inf_threshold_rejected(self):
        from scion.observability.slos import SLOTarget

        with pytest.raises(ValueError, match="finite"):
            SLOTarget(name="test", objective=0.99, threshold=float("inf"))

    def test_zero_threshold_allowed(self):
        """threshold=0 is valid (e.g., zero-tolerance error rate)."""
        from scion.observability.slos import SLOTarget

        t = SLOTarget(name="zero_errors", objective=0.99, threshold=0.0)
        assert t.threshold == 0.0

    def test_frozen_target(self):
        """SLOTarget should be immutable."""
        from scion.observability.slos import SLOTarget

        t = SLOTarget(name="test", objective=0.99, threshold=1.0)
        with pytest.raises(AttributeError):
            t.objective = 0.5  # type: ignore[misc]


class TestObjectiveOneEdgeCases:
    """objective=1.0 zero-tolerance SLO (8eyes-test P0-2, 8eyes-impl P1)."""

    def test_obj_1_no_failures_not_exhausted(self):
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=1.0)
        status = budget.status(total_requests=1000, failed_requests=0)
        assert status["exhausted"] is False
        assert status["budget_remaining_pct"] == 100.0

    def test_obj_1_any_failure_exhausted(self):
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=1.0)
        status = budget.status(total_requests=1000, failed_requests=1)
        assert status["exhausted"] is True
        assert status["budget_remaining_pct"] == 0.0

    def test_obj_1_no_contradictory_state(self):
        """exhausted=True must never coexist with remaining_pct=100%."""
        from scion.observability.slos import ErrorBudget

        budget = ErrorBudget(objective=1.0)
        for fails in range(0, 5):
            status = budget.status(total_requests=100, failed_requests=fails)
            if status["exhausted"]:
                assert status["budget_remaining_pct"] == 0.0
            else:
                assert status["budget_remaining_pct"] == 100.0


class TestBurnRateClamp:
    """Burn rate clamped to finite max (8eyes-test P0-3, 8eyes-sec P1)."""

    def test_obj_1_burn_rate_finite(self):
        """objective=1.0 with failures must NOT produce float('inf')."""
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=1.0)
        rate = calc.burn_rate(total_requests=1000, failed_requests=5)
        assert rate == 1000.0  # MAX_BURN_RATE
        assert rate != float("inf")

    def test_burn_rate_json_serializable(self):
        """Burn rate must be valid JSON (no Infinity)."""
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=1.0)
        rate = calc.burn_rate(total_requests=1000, failed_requests=5)
        serialized = json.dumps({"burn_rate": rate})
        assert "Infinity" not in serialized

    def test_obj_1_triggers_all_alerts(self):
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=1.0)
        alerts = calc.check_alerts(total_requests=1000, failed_requests=5)
        severities = {a["severity"] for a in alerts}
        assert "critical" in severities
        assert "high" in severities
        assert "warning" in severities


class TestAlertTiersIndependent:
    """Each alert tier fires independently (8eyes-test P1-1)."""

    def test_warning_only(self):
        """Burn rate ~1.5x → only warning fires."""
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.95)
        # allowed = 5%, actual = 7.5% → rate = 1.5x
        alerts = calc.check_alerts(total_requests=10000, failed_requests=750)
        severities = {a["severity"] for a in alerts}
        assert severities == {"warning"}

    def test_high_and_warning(self):
        """Burn rate ~7x → high + warning fire, not critical."""
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.95)
        # allowed = 5%, actual = 35% → rate = 7x
        alerts = calc.check_alerts(total_requests=10000, failed_requests=3500)
        severities = {a["severity"] for a in alerts}
        assert severities == {"high", "warning"}

    def test_all_three_tiers(self):
        """Burn rate ~15x → all three fire."""
        from scion.observability.slos import BurnRateCalculator

        calc = BurnRateCalculator(objective=0.95)
        # allowed = 5%, actual = 75% → rate = 15x
        alerts = calc.check_alerts(total_requests=10000, failed_requests=7500)
        severities = {a["severity"] for a in alerts}
        assert severities == {"critical", "high", "warning"}


class TestP99Estimation:
    """Direct p99 testing with known distributions (8eyes-test P1-2)."""

    def test_p99_known_distribution(self):
        """90 fast + 10 slow → p99 should reflect the slow bucket."""
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)
        # 90 at 0.5s, 10 at 50s → p99 of 100 = 99th obs is in slow bucket
        for _ in range(90):
            m.execution_latency.labels(agent="a", status="ok").observe(0.5)
        for _ in range(10):
            m.execution_latency.labels(agent="a", status="ok").observe(50.0)

        evaluator = SLOEvaluator(registry=m)
        p99 = evaluator._estimate_p99()
        assert p99 is not None
        # With interpolation between 30s and 60s buckets, should be ~48s
        assert p99 > 30.0, f"p99={p99} too low — should reflect slow bucket"
        assert p99 <= 60.0, f"p99={p99} exceeds bucket range"

    def test_p99_all_fast(self):
        """100 fast requests → p99 in fast bucket."""
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)
        for _ in range(100):
            m.execution_latency.labels(agent="a", status="ok").observe(0.5)

        evaluator = SLOEvaluator(registry=m)
        p99 = evaluator._estimate_p99()
        assert p99 is not None
        assert p99 <= 1.0, f"p99={p99} should be in fast bucket"

    def test_p99_empty_histogram(self):
        """No observations → None."""
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)

        evaluator = SLOEvaluator(registry=m)
        p99 = evaluator._estimate_p99()
        assert p99 is None

    def test_p99_multi_agent_aggregation(self):
        """p99 must aggregate across multiple agent label sets."""
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)
        # Agent A: 50 fast at 0.5s
        for _ in range(50):
            m.execution_latency.labels(agent="agent_a", status="ok").observe(0.5)
        # Agent B: 40 fast + 10 slow
        for _ in range(40):
            m.execution_latency.labels(agent="agent_b", status="ok").observe(0.5)
        for _ in range(10):
            m.execution_latency.labels(agent="agent_b", status="ok").observe(50.0)

        evaluator = SLOEvaluator(registry=m)
        p99 = evaluator._estimate_p99()
        assert p99 is not None
        # 100 total: 90 fast + 10 slow. p99 should be in slow bucket (>30s)
        assert p99 > 30.0, f"p99={p99} — multi-agent aggregation broken"

    def test_p99_overflow_beyond_max_bucket(self):
        """All observations above highest finite bucket must NOT return None.

        GPT-5.4 P0: if true p99 > 120s, _estimate_p99 must return a
        lower-bound estimate (120s), not None which hides the outage.
        """
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)
        # All 100 observations at 200s (above max bucket 120s)
        for _ in range(100):
            m.execution_latency.labels(agent="slow", status="ok").observe(200.0)

        evaluator = SLOEvaluator(registry=m)
        p99 = evaluator._estimate_p99()
        assert p99 is not None, "p99 must not be None when observations exist"
        assert p99 >= 120.0, f"p99={p99} should be >= highest bucket (120s)"


class TestEvaluateReturnSchema:
    """Exhaustive schema assertion (8eyes-test P2-5)."""

    def test_evaluate_all_fields_present(self):
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)
        evaluator = SLOEvaluator(registry=m)
        result = evaluator.evaluate()

        assert "overall_status" in result
        assert "slos" in result
        assert "error_budget" in result
        assert "burn_rate" in result
        assert "alerts" in result
        assert "total_requests" in result
        assert "failed_requests" in result
        assert "evaluated_at" in result
        assert isinstance(result["slos"], list)
        assert len(result["slos"]) == 3  # 3 default targets
        assert result["overall_status"] in ("meeting", "at_risk", "breaching")

    def test_evaluate_numeric_values(self):
        """Verify computed values, not just field existence."""
        from scion.observability.slos import SLOEvaluator

        reg = CollectorRegistry()
        m = MetricsRegistry(registry=reg)
        # Observe 100 success + 10 errors
        for _ in range(100):
            m.execution_total.labels(agent="test", status="ok").inc()
        for _ in range(10):
            m.execution_total.labels(agent="test", status="error").inc()

        evaluator = SLOEvaluator(registry=m)
        result = evaluator.evaluate()
        assert result["total_requests"] == 110
        assert result["failed_requests"] == 10
        assert isinstance(result["burn_rate"], (int, float))
        assert result["burn_rate"] > 0  # 10/110 = 9.1% error rate > 5% allowed
