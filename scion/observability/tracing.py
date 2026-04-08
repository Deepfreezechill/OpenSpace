"""Structured execution tracing for Scion.

Captures multi-step execution traces using contextvars so trace context
propagates correctly across ``asyncio`` task boundaries.  Each trace is a
tree of spans with timing, metadata, and parent–child relationships.

Integrates with ``scion.domain.logging`` context variables when
available, carrying ``task_id`` and ``correlation_id`` into every span.

Usage::

    from scion.observability.tracing import tracer, trace_async

    @trace_async("grounding.process")
    async def process(self, context):
        with tracer.span("load_tools"):
            tools = await self._get_available_tools(...)
        ...

    # After execution, retrieve the trace:
    trace = tracer.current_trace()
"""

from __future__ import annotations

import collections
import contextvars
import functools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ── Span data model ──────────────────────────────────────────────────


@dataclass
class Span:
    """A single unit of work within a trace."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: str = "ok") -> None:
        if self.end_time is not None:
            return  # Already finished — idempotent
        self.end_time = time.monotonic()
        self.status = status

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "time": time.monotonic(), **attrs})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


# ── Trace (collection of spans) ─────────────────────────────────────


@dataclass
class Trace:
    """An ordered collection of spans forming a complete execution trace."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:32])
    spans: List[Span] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "total_spans": len(self.spans),
            "root_span": self.spans[0].to_dict() if self.spans else None,
        }


# ── Context variables ────────────────────────────────────────────────

_current_trace: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "scion_trace", default=None
)
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "scion_span", default=None
)

# ── Execution tracer ─────────────────────────────────────────────────

# Ring buffer size — keep last N traces in memory for debugging
_MAX_TRACES = 50
_MAX_SPANS_PER_TRACE = 500


class ExecutionTracer:
    """Lightweight tracer that captures execution spans in-process.

    Traces are stored in a bounded ring buffer for post-hoc debugging.
    Not a replacement for distributed tracing (OpenTelemetry) — this is
    a lightweight, zero-dependency alternative for development and
    single-node deployments.
    """

    def __init__(self, max_traces: int = _MAX_TRACES) -> None:
        self._max_traces = max_traces
        self._traces: collections.deque[Trace] = collections.deque(maxlen=max_traces)
        self._lock = threading.Lock()

    def start_trace(self, name: str = "root", **attrs: Any) -> Trace:
        """Begin a new trace with a root span.

        If a trace is already active, it is finished and stored before
        the new one starts (prevents orphaned traces).
        """
        existing = _current_trace.get()
        if existing is not None:
            self.finish_trace()

        trace = Trace()
        root = Span(
            name=name,
            trace_id=trace.trace_id,
            attributes=attrs,
        )
        trace.spans.append(root)
        _current_trace.set(trace)
        _current_span.set(root)
        return trace

    def span(self, name: str, **attrs: Any):
        """Context manager that creates a child span under the current span."""
        return _SpanContext(self, name, attrs)

    def finish_trace(self) -> Optional[Trace]:
        """Finish the current trace, closing the root span."""
        trace = _current_trace.get()
        if trace is None:
            return None

        # Close any open spans (preserving their current status)
        for s in trace.spans:
            if s.end_time is None:
                s.finish(status=s.status)

        # Store in ring buffer (thread-safe deque with maxlen)
        with self._lock:
            self._traces.append(trace)

        _current_trace.set(None)
        _current_span.set(None)
        return trace

    def current_trace(self) -> Optional[Trace]:
        return _current_trace.get()

    def current_span(self) -> Optional[Span]:
        return _current_span.get()

    @property
    def recent_traces(self) -> List[Trace]:
        with self._lock:
            return list(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
        _current_trace.set(None)
        _current_span.set(None)


class _SpanContext:
    """Context manager for creating child spans."""

    def __init__(self, tracer: ExecutionTracer, name: str, attrs: Dict[str, Any]):
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._span: Optional[Span] = None
        self._parent: Optional[Span] = None

    def __enter__(self) -> Span:
        trace = _current_trace.get()
        if trace is None:
            trace = self._tracer.start_trace(self._name, **self._attrs)
            self._span = trace.spans[0]
            return self._span

        self._parent = _current_span.get()
        # Guard against unbounded span growth — return a no-op sentinel
        if len(trace.spans) >= _MAX_SPANS_PER_TRACE:
            self._span = Span(
                name=self._name,
                trace_id=trace.trace_id,
                parent_id=self._parent.span_id if self._parent else None,
            )
            # NOT appended to trace.spans — mutations are silently discarded
            return self._span
        self._span = Span(
            name=self._name,
            trace_id=trace.trace_id,
            parent_id=self._parent.span_id if self._parent else None,
            attributes=self._attrs,
        )
        trace.spans.append(self._span)
        _current_span.set(self._span)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            self._span.finish("error" if exc_type else "ok")
        if self._parent:
            _current_span.set(self._parent)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


# ── Decorator ────────────────────────────────────────────────────────


def trace_async(
    name: str, tracer_instance: Optional[ExecutionTracer] = None
) -> Callable:
    """Decorator that wraps an async function in a trace span.

    If no trace is active, starts a new trace and finishes it when done.
    Otherwise, creates a child span under the current trace.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t = tracer_instance or tracer
            is_root = t.current_trace() is None
            if is_root:
                t.start_trace(name)
            try:
                if not is_root:
                    async with t.span(name):
                        return await fn(*args, **kwargs)
                else:
                    return await fn(*args, **kwargs)
            except Exception:
                # Mark root span as error before finish
                if is_root:
                    root = t.current_span()
                    if root:
                        root.status = "error"
                raise
            finally:
                if is_root:
                    t.finish_trace()

        return wrapper

    return decorator


# Module-level singleton
tracer = ExecutionTracer()
