"""SLO (Service Level Objective) tracking for OpenSpace.

Defines SLO targets, error budgets, burn-rate alerting, and an evaluator
that reads from the Prometheus metrics registry to compute live SLO status.

Implements Google's multi-window burn-rate alerting model:
  - Critical (14.4x): exhausts 30-day budget in 2 hours
  - High (6x): exhausts budget in 5 hours
  - Warning (1x): on-pace to exhaust budget by window end

Usage::

    from openspace.observability.slos import SLOEvaluator
    from openspace.observability.metrics import metrics

    evaluator = SLOEvaluator(registry=metrics)
    result = evaluator.evaluate()
    # {"overall_status": "meeting", "slos": [...], "burn_rates": [...]}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openspace.observability.metrics import MetricsRegistry


# ── SLO Target ───────────────────────────────────────────────────────


@dataclass
class SLOTarget:
    """Definition of a single SLO target."""

    name: str
    objective: float  # e.g., 0.99 for 99%
    threshold: float  # e.g., 30.0 seconds for latency, 0.05 for 5% error rate
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not (0 < self.objective <= 1.0):
            raise ValueError(
                f"objective must be in (0, 1.0], got {self.objective}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "threshold": self.threshold,
            "unit": self.unit,
            "description": self.description,
        }


# ── Default targets ──────────────────────────────────────────────────

DEFAULT_TARGETS: List[SLOTarget] = [
    SLOTarget(
        name="execution_latency_p99",
        objective=0.99,
        threshold=30.0,
        unit="seconds",
        description="99th percentile execution latency under 30 seconds",
    ),
    SLOTarget(
        name="error_rate",
        objective=0.95,
        threshold=0.05,
        unit="ratio",
        description="Execution error rate below 5%",
    ),
    SLOTarget(
        name="availability",
        objective=0.99,
        threshold=0.99,
        unit="ratio",
        description="System availability above 99%",
    ),
]


# ── Error Budget ─────────────────────────────────────────────────────


@dataclass
class ErrorBudget:
    """Tracks error budget consumption over a rolling window.

    Error budget = (1 - objective) * total_requests.
    Consumed budget = failed_requests within the window.
    """

    objective: float
    window_seconds: int = 2_592_000  # 30 days default

    def status(
        self, total_requests: int, failed_requests: int
    ) -> Dict[str, Any]:
        """Compute current error budget status."""
        allowed_error_rate = 1 - self.objective
        budget_total = allowed_error_rate * total_requests

        if total_requests == 0:
            return {
                "budget_total": 0,
                "budget_consumed": 0,
                "budget_remaining": 0,
                "budget_remaining_pct": 100.0,
                "exhausted": False,
            }

        consumed = failed_requests
        remaining = budget_total - consumed
        remaining_pct = (remaining / budget_total * 100) if budget_total > 0 else 100.0

        return {
            "budget_total": budget_total,
            "budget_consumed": consumed,
            "budget_remaining": remaining,
            "budget_remaining_pct": round(remaining_pct, 2),
            "exhausted": remaining <= 1e-9,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "window_seconds": self.window_seconds,
        }


# ── Burn Rate ────────────────────────────────────────────────────────

# Google SRE multi-window burn rate thresholds:
# Critical: 14.4x → exhausts 30-day budget in ~2 hours
# High:     6.0x  → exhausts budget in ~5 hours
# Warning:  1.0x  → on-pace to exhaust budget by window end
_DEFAULT_ALERT_THRESHOLDS = [
    {"severity": "critical", "burn_rate_threshold": 14.4},
    {"severity": "high", "burn_rate_threshold": 6.0},
    {"severity": "warning", "burn_rate_threshold": 1.0},
]


@dataclass
class BurnRateCalculator:
    """Calculates SLO burn rate and generates alerts.

    Burn rate = (actual error rate) / (allowed error rate).
    A burn rate of 1.0 means you're consuming budget at exactly the
    allowed pace. >1.0 means you'll exhaust budget before the window ends.
    """

    objective: float
    alert_thresholds: List[Dict[str, Any]] = field(
        default_factory=lambda: list(_DEFAULT_ALERT_THRESHOLDS)
    )

    def burn_rate(self, total_requests: int, failed_requests: int) -> float:
        """Calculate current burn rate."""
        if total_requests == 0:
            return 0.0
        allowed_error_rate = 1 - self.objective
        if allowed_error_rate == 0:
            return float("inf") if failed_requests > 0 else 0.0
        actual_error_rate = failed_requests / total_requests
        return actual_error_rate / allowed_error_rate

    def check_alerts(
        self, total_requests: int, failed_requests: int
    ) -> List[Dict[str, Any]]:
        """Return list of triggered alerts based on current burn rate."""
        rate = self.burn_rate(total_requests, failed_requests)
        alerts = []
        for threshold in self.alert_thresholds:
            if rate > threshold["burn_rate_threshold"]:
                alerts.append(
                    {
                        "severity": threshold["severity"],
                        "burn_rate": round(rate, 2),
                        "threshold": threshold["burn_rate_threshold"],
                    }
                )
        return alerts


# ── SLO Evaluator ────────────────────────────────────────────────────


class SLOEvaluator:
    """Evaluates SLO compliance by reading from the metrics registry.

    Reads counters/histograms from the Prometheus registry to compute
    current error rates, latency percentiles, and availability.
    """

    def __init__(
        self,
        registry: MetricsRegistry | None = None,
        targets: List[SLOTarget] | None = None,
    ) -> None:
        self._registry = registry or MetricsRegistry()
        self._targets = targets if targets is not None else list(DEFAULT_TARGETS)
        self._burn_calc = BurnRateCalculator(
            objective=self._get_error_objective()
        )

    def _get_error_objective(self) -> float:
        """Get the error rate objective from targets."""
        for t in self._targets:
            if t.name == "error_rate":
                return t.objective
        return 0.95  # default

    def _get_request_counts(self) -> tuple[int, int]:
        """Read total and failed request counts from the registry."""
        total = 0
        failed = 0
        try:
            # Sum across all agent labels
            for sample in self._registry.execution_total.collect()[0].samples:
                val = int(sample.value)
                total += val
                if sample.labels.get("status") == "error":
                    failed += val
        except (IndexError, AttributeError):
            pass
        return total, failed

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all SLO targets against current metrics."""
        total, failed = self._get_request_counts()
        slo_results = []

        for target in self._targets:
            result = self._evaluate_target(target, total, failed)
            slo_results.append(result)

        # Overall status: worst of all SLOs
        statuses = [r["status"] for r in slo_results]
        if "breaching" in statuses:
            overall = "breaching"
        elif "at_risk" in statuses:
            overall = "at_risk"
        else:
            overall = "meeting"

        # Error budget
        budget = ErrorBudget(objective=self._get_error_objective())
        budget_status = budget.status(total, failed)

        # Burn rate alerts
        alerts = self._burn_calc.check_alerts(total, failed)

        return {
            "overall_status": overall,
            "slos": slo_results,
            "error_budget": budget_status,
            "burn_rate": self._burn_calc.burn_rate(total, failed),
            "alerts": alerts,
            "total_requests": total,
            "failed_requests": failed,
            "evaluated_at": time.time(),
        }

    def _evaluate_target(
        self, target: SLOTarget, total: int, failed: int
    ) -> Dict[str, Any]:
        """Evaluate a single SLO target."""
        if total == 0:
            return {
                "target": target.to_dict(),
                "current_value": None,
                "status": "meeting",
                "detail": "No data",
            }

        if target.name == "error_rate":
            current = failed / total
            status = "meeting" if current <= target.threshold else "breaching"
            # At risk if >50% of error budget consumed
            if status == "meeting":
                budget = ErrorBudget(objective=target.objective)
                bs = budget.status(total, failed)
                if bs["budget_remaining_pct"] < 50:
                    status = "at_risk"
            return {
                "target": target.to_dict(),
                "current_value": round(current, 4),
                "status": status,
                "detail": f"{current:.1%} error rate (threshold: {target.threshold:.1%})",
            }

        elif target.name == "execution_latency_p99":
            # Approximate p99 from histogram buckets
            p99 = self._estimate_p99()
            if p99 is None:
                return {
                    "target": target.to_dict(),
                    "current_value": None,
                    "status": "meeting",
                    "detail": "No latency data",
                }
            status = "meeting" if p99 <= target.threshold else "breaching"
            if status == "meeting" and p99 > target.threshold * 0.8:
                status = "at_risk"
            return {
                "target": target.to_dict(),
                "current_value": round(p99, 2),
                "status": status,
                "detail": f"p99={p99:.1f}s (threshold: {target.threshold}s)",
            }

        elif target.name == "availability":
            success = total - failed
            availability = success / total if total > 0 else 1.0
            status = "meeting" if availability >= target.threshold else "breaching"
            if status == "meeting" and availability < target.threshold * 1.01:
                status = "at_risk"
            return {
                "target": target.to_dict(),
                "current_value": round(availability, 4),
                "status": status,
                "detail": f"{availability:.2%} available (target: {target.threshold:.2%})",
            }

        # Custom target — no specific evaluation logic
        return {
            "target": target.to_dict(),
            "current_value": None,
            "status": "meeting",
            "detail": "Custom target — no auto-evaluation",
        }

    def _estimate_p99(self) -> Optional[float]:
        """Estimate p99 latency from histogram buckets."""
        try:
            samples = self._registry.execution_latency.collect()[0].samples
            buckets = []
            count = 0
            for sample in samples:
                if sample.name.endswith("_bucket"):
                    le = sample.labels.get("le")
                    if le and le != "+Inf":
                        buckets.append((float(le), sample.value))
                elif sample.name.endswith("_count"):
                    count += sample.value

            if count == 0:
                return None

            target_count = count * 0.99
            for le, bucket_count in sorted(buckets):
                if bucket_count >= target_count:
                    return le
            return None
        except (IndexError, AttributeError, ValueError):
            return None

    def to_json(self) -> str:
        """Return evaluation result as JSON string."""
        return json.dumps(self.evaluate(), default=str, indent=2)
