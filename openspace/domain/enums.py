"""Consolidated domain enumerations.

Re-exports existing enums from their current locations for backward
compatibility, and adds new enums that were previously magic strings.
Existing code can continue importing from the original modules; new code
should prefer ``openspace.domain.enums``.
"""

from __future__ import annotations

from enum import Enum

# ── Re-exports from existing locations ────────────────────────────────
from openspace.skill_engine.types import (
    EvolutionType,
    SkillCategory,
    SkillOrigin,
    SkillVisibility,
)
from openspace.grounding.core.types import (
    BackendType,
    SessionStatus,
    ToolStatus,
)
from openspace.grounding.core.exceptions import ErrorCode as GroundingErrorCode


# ── New enums (were magic strings) ────────────────────────────────────

class TaskStatus(str, Enum):
    """Execution task lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


class SearchMode(str, Enum):
    """Skill / tool search strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchScope(str, Enum):
    """Where to search for skills."""

    LOCAL = "local"
    CLOUD = "cloud"
    ALL = "all"


class TrustTier(str, Enum):
    """Capability trust level for sandbox decisions."""

    UNTRUSTED = "untrusted"
    BASIC = "basic"
    STANDARD = "standard"
    PRIVILEGED = "privileged"


class SkillStatus(str, Enum):
    """Lifecycle status of a skill record."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


class PatchType(str, Enum):
    """How a skill patch is applied."""

    AUTO = "auto"
    FULL = "full"
    DIFF = "diff"
    PATCH = "patch"


class MCPErrorCode(str, Enum):
    """Error codes surfaced through the MCP server layer."""

    EXECUTION_ERROR = "EXECUTION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"


__all__ = [
    # Re-exported
    "BackendType",
    "EvolutionType",
    "GroundingErrorCode",
    "SessionStatus",
    "SkillCategory",
    "SkillOrigin",
    "SkillVisibility",
    "ToolStatus",
    # New
    "MCPErrorCode",
    "PatchType",
    "SearchMode",
    "SearchScope",
    "SkillStatus",
    "TaskStatus",
    "TrustTier",
]
