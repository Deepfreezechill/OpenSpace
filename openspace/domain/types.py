"""Frozen domain value objects and data-transfer types.

All types here are **immutable** (``frozen=True``) so they can be safely
shared across async boundaries, cached, and hashed.  Mutations produce
new instances via ``dataclasses.replace()``.

Existing mutable dataclasses in ``openspace.skill_engine.types`` remain
for backward compatibility.  New code should prefer these frozen variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, FrozenSet, Optional, Tuple


def _deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers to immutable equivalents.

    - dict  → tuple of (key, frozen_value) pairs
    - list  → tuple of frozen values
    - set   → frozenset of frozen values
    - scalar / already-frozen → returned as-is
    """
    if isinstance(value, dict):
        return tuple((k, _deep_freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(v) for v in value)
    return value


# ─── Task Execution Types ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """Immutable request to execute a task."""

    task: str
    task_id: str = ""
    workspace_dir: str = ""
    max_iterations: Optional[int] = None
    search_scope: str = "all"
    skill_dirs: Tuple[str, ...] = ()
    context: Tuple[Tuple[str, Any], ...] = ()

    @property
    def context_dict(self) -> Dict[str, Any]:
        return dict(self.context)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRequest":
        ctx = data.get("context") or {}
        return cls(
            task=data["task"],
            task_id=data.get("task_id", ""),
            workspace_dir=data.get("workspace_dir", ""),
            max_iterations=data.get("max_iterations"),
            search_scope=data.get("search_scope", "all"),
            skill_dirs=tuple(data.get("skill_dirs") or []),
            context=_deep_freeze(ctx) if isinstance(ctx, dict) else (),
        )


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Record of a single tool call within a task."""

    tool_name: str
    arguments: Tuple[Tuple[str, Any], ...] = ()
    status: str = "success"
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Immutable result of a task execution."""

    task_id: str
    status: str  # "success" | "error" | "timeout"
    response: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0
    iterations: int = 0
    skills_used: Tuple[str, ...] = ()
    evolved_skills: Tuple[str, ...] = ()
    tool_executions: Tuple[ToolExecution, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "response": self.response,
            "execution_time": self.execution_time,
            "iterations": self.iterations,
            "skills_used": list(self.skills_used),
        }
        if self.error:
            d["error"] = self.error
        if self.evolved_skills:
            d["evolved_skills"] = list(self.evolved_skills)
        if self.tool_executions:
            d["tool_executions"] = [
                {
                    "tool_name": te.tool_name,
                    "arguments": dict(te.arguments),
                    "status": te.status,
                    "duration_ms": te.duration_ms,
                    **({"error": te.error} if te.error else {}),
                }
                for te in self.tool_executions
            ]
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        tool_execs = tuple(
            ToolExecution(
                tool_name=te["tool_name"],
                arguments=_deep_freeze(te.get("arguments", {})) if isinstance(te.get("arguments"), dict) else (),
                status=te.get("status", "success"),
                duration_ms=te.get("duration_ms", 0.0),
                error=te.get("error"),
            )
            for te in (data.get("tool_executions") or [])
        )
        return cls(
            task_id=data.get("task_id", ""),
            status=data.get("status", "error"),
            response=data.get("response", ""),
            error=data.get("error"),
            execution_time=data.get("execution_time", 0.0),
            iterations=data.get("iterations", 0),
            skills_used=tuple(data.get("skills_used") or []),
            evolved_skills=tuple(data.get("evolved_skills") or []),
            tool_executions=tool_execs,
            warnings=tuple(data.get("warnings") or []),
        )


# ─── Skill Identity & Metadata ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillIdentity:
    """Lightweight, hashable skill reference."""

    skill_id: str
    name: str
    description: str = ""
    category: str = "workflow"
    source: str = "local"

    def __hash__(self) -> int:
        return hash(self.skill_id)


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Full skill metadata — immutable snapshot."""

    skill_id: str
    name: str
    description: str
    path: str = ""
    is_active: bool = True
    category: str = "workflow"
    visibility: str = "private"
    creator_id: str = ""
    tags: Tuple[str, ...] = ()
    tool_dependencies: Tuple[str, ...] = ()
    critical_tools: Tuple[str, ...] = ()

    # Lineage
    origin: str = "imported"
    generation: int = 0
    parent_skill_ids: Tuple[str, ...] = ()
    source_task_id: str = ""
    change_summary: str = ""

    # Counters (snapshot at freeze-time)
    total_selections: int = 0
    total_applied: int = 0
    total_completions: int = 0
    total_fallbacks: int = 0

    # Timestamps
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @property
    def effective_rate(self) -> float:
        if self.total_selections == 0:
            return 0.0
        return self.total_applied / self.total_selections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "is_active": self.is_active,
            "category": self.category,
            "visibility": self.visibility,
            "origin": self.origin,
            "generation": self.generation,
            "tags": list(self.tags),
            "total_selections": self.total_selections,
            "total_applied": self.total_applied,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


# ─── Evolution Types ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvolutionRequest:
    """Immutable request to evolve one or more skills."""

    evolution_type: str  # "fix" | "derived" | "captured"
    trigger: str  # "analysis" | "tool_degradation" | "metric_monitor"
    target_skill_ids: Tuple[str, ...] = ()
    source_task_id: str = ""
    direction: str = ""
    category: Optional[str] = None
    tool_issue_summary: str = ""
    metric_summary: str = ""


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """Immutable outcome of an evolution attempt."""

    success: bool
    evolved_skill_id: Optional[str] = None
    evolved_skill_name: Optional[str] = None
    parent_skill_ids: Tuple[str, ...] = ()
    evolution_type: str = ""
    change_summary: str = ""
    error: Optional[str] = None


# ─── Analysis Types ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillJudgmentSnapshot:
    """Immutable judgment of a skill's performance on a task."""

    skill_id: str
    skill_applied: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class EvolutionSuggestionSnapshot:
    """Immutable suggestion for evolution from an analysis."""

    evolution_type: str
    target_skill_ids: Tuple[str, ...] = ()
    category: Optional[str] = None
    direction: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionAnalysisSnapshot:
    """Immutable snapshot of a post-execution analysis."""

    task_id: str
    timestamp: datetime
    task_completed: bool = False
    execution_note: str = ""
    tool_issues: Tuple[str, ...] = ()
    skill_judgments: Tuple[SkillJudgmentSnapshot, ...] = ()
    evolution_suggestions: Tuple[EvolutionSuggestionSnapshot, ...] = ()
    analyzed_by: str = ""
    analyzed_at: Optional[datetime] = None


# ─── Search Types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillSearchResult:
    """A single skill search hit."""

    skill_id: str
    name: str
    description: str
    score: float = 0.0
    source: str = "local"
    body: str = ""


@dataclass(frozen=True, slots=True)
class SkillSearchResponse:
    """Collection of search results."""

    query: str
    results: Tuple[SkillSearchResult, ...] = ()
    total_count: int = 0
    search_mode: str = "hybrid"


# ─── Sandbox / Security Types ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Immutable sandbox policy snapshot."""

    sandbox_enabled: bool = True
    trust_tier: str = "untrusted"
    allowed_commands: FrozenSet[str] = frozenset()
    blocked_commands: FrozenSet[str] = frozenset()
    allowed_domains: FrozenSet[str] = frozenset()
    blocked_domains: FrozenSet[str] = frozenset()
    max_execution_time_s: int = 300
    max_memory_mb: int = 512


@dataclass(frozen=True, slots=True)
class CapabilityLease:
    """A time-bounded, revocable capability grant (Phase 2)."""

    lease_id: str
    capability: str
    granted_to: str
    trust_tier: str = "basic"
    expires_at: Optional[datetime] = None
    revoked: bool = False


# ─── Tool Types ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Lightweight tool identity for protocol boundaries."""

    name: str
    description: str = ""
    backend_type: str = "not_set"
    parameters_schema: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Immutable result of a tool invocation."""

    status: str  # "success" | "error"
    content: str = ""
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Tuple[Tuple[str, Any], ...] = ()


__all__ = [
    "CapabilityLease",
    "EvolutionRequest",
    "EvolutionResult",
    "EvolutionSuggestionSnapshot",
    "ExecutionAnalysisSnapshot",
    "SandboxPolicy",
    "SkillIdentity",
    "SkillJudgmentSnapshot",
    "SkillManifest",
    "SkillSearchResponse",
    "SkillSearchResult",
    "TaskRequest",
    "TaskResult",
    "ToolCallResult",
    "ToolDescriptor",
    "ToolExecution",
    "_deep_freeze",
]
