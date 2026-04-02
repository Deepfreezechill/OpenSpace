"""SDK API contract tests — EPIC 1.8.

Issues:
- #56: REST API specification defined in docs/sdk-api-spec.md
- #57: Public API surface documented in docs/sdk-public-surface.md
- #58: API contract tests (test-first, implementations in Phase 6)

These tests validate the **contract** (data shapes, error codes, envelope
structure) without requiring a running server.  They test the specification
itself: can our domain types serialize to the documented schema?  Do error
envelopes match the spec?

Phase 6 will add integration tests against a live server.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# SDK envelope helpers (will become openspace.sdk.envelope in Phase 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class APIError:
    """Standard error payload."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class APIEnvelope:
    """Standard response envelope per sdk-api-spec.md."""

    ok: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[APIError] = None
    request_id: str = ""

    def to_json(self) -> str:
        d: dict[str, Any] = {"ok": self.ok, "data": self.data, "request_id": self.request_id}
        d["error"] = asdict(self.error) if self.error else None
        return json.dumps(d)

    @classmethod
    def success(cls, data: dict[str, Any], request_id: str = "") -> "APIEnvelope":
        return cls(ok=True, data=data, request_id=request_id)

    @classmethod
    def failure(cls, code: str, message: str, request_id: str = "", **details: Any) -> "APIEnvelope":
        return cls(ok=False, error=APIError(code=code, message=message, details=details), request_id=request_id)


# ---------------------------------------------------------------------------
# SDK request/response types (will become openspace.sdk.types in Phase 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """SDK task execution request."""

    task: str
    workspace_dir: Optional[str] = None
    max_iterations: Optional[int] = None
    skill_dirs: Optional[list[str]] = None
    search_scope: str = "all"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True, slots=True)
class ToolUsageRecord:
    """SDK record of a tool invocation within a task."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass(frozen=True, slots=True)
class TaskResultData:
    """SDK task result payload."""

    task_id: str
    status: str
    success: Optional[bool] = None
    output: Optional[str] = None
    tools_used: Optional[list[ToolUsageRecord]] = None
    skill_used: Optional[str] = None
    evolved_skills: Optional[list[str]] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """SDK skill info payload (list/summary view)."""

    id: str
    name: str
    version: str
    active: bool
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class SkillDetail:
    """SDK skill detail payload (single-resource view).

    Extends SkillInfo with manifest and lineage for GET /skills/{id}.
    """

    id: str
    name: str
    version: str
    active: bool
    created_at: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SkillSearchResult:
    """SDK skill search result."""

    id: str
    name: str
    description: str
    source: str
    score: float
    imported: bool = False


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """SDK health check payload."""

    status: str
    version: str
    initialized: bool
    backends: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error code registry (must match sdk-api-spec.md)
# ---------------------------------------------------------------------------

VALID_ERROR_CODES = frozenset(
    {
        "AUTH_REQUIRED",
        "RATE_LIMITED",
        "TASK_NOT_FOUND",
        "SKILL_NOT_FOUND",
        "TASK_FAILED",
        "NOT_INITIALIZED",
        "VALIDATION_ERROR",
        "EVOLUTION_NOT_FOUND",
        "INVALID_STATE",
    }
)

VALID_TASK_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})

VALID_SEARCH_SCOPES = frozenset({"all", "local", "cloud"})

VALID_HEALTH_STATUSES = frozenset({"healthy", "degraded", "unhealthy"})


# ===========================================================================
# Contract Tests
# ===========================================================================


class TestAPIEnvelopeContract:
    """The response envelope must match the documented schema."""

    def test_success_envelope_shape(self) -> None:
        env = APIEnvelope.success({"task_id": "abc"}, request_id="req-1")
        parsed = json.loads(env.to_json())

        assert parsed["ok"] is True
        assert parsed["data"] == {"task_id": "abc"}
        assert parsed["error"] is None
        assert parsed["request_id"] == "req-1"

    def test_error_envelope_shape(self) -> None:
        env = APIEnvelope.failure("TASK_NOT_FOUND", "Task xyz not found", request_id="req-2")
        parsed = json.loads(env.to_json())

        assert parsed["ok"] is False
        assert parsed["data"] is None
        assert parsed["error"]["code"] == "TASK_NOT_FOUND"
        assert parsed["error"]["message"] == "Task xyz not found"
        assert isinstance(parsed["error"]["details"], dict)
        assert parsed["request_id"] == "req-2"

    def test_envelope_always_has_four_keys(self) -> None:
        """Spec requires exactly: ok, data, error, request_id."""
        for env in [
            APIEnvelope.success({}),
            APIEnvelope.failure("AUTH_REQUIRED", "No token"),
        ]:
            parsed = json.loads(env.to_json())
            assert set(parsed.keys()) == {"ok", "data", "error", "request_id"}

    def test_success_and_error_mutually_exclusive(self) -> None:
        success = APIEnvelope.success({"x": 1})
        assert success.ok is True and success.error is None

        failure = APIEnvelope.failure("TASK_FAILED", "boom")
        assert failure.ok is False and failure.data is None


class TestErrorCodeContract:
    """All error codes used must be in the documented registry."""

    def test_known_error_codes(self) -> None:
        for code in VALID_ERROR_CODES:
            env = APIEnvelope.failure(code, f"Test {code}")
            assert env.error is not None
            assert env.error.code in VALID_ERROR_CODES

    def test_error_code_count(self) -> None:
        """Guard: if you add an error code, update the spec first."""
        assert len(VALID_ERROR_CODES) == 9


class TestTaskRequestContract:
    """TaskRequest must serialize to the documented schema."""

    def test_minimal_request(self) -> None:
        req = TaskRequest(task="Hello world")
        d = req.to_dict()
        assert d == {"task": "Hello world", "search_scope": "all"}

    def test_full_request(self) -> None:
        req = TaskRequest(
            task="Build a calculator",
            workspace_dir="/tmp/ws",
            max_iterations=5,
            skill_dirs=["/skills/a"],
            search_scope="local",
        )
        d = req.to_dict()
        assert d["task"] == "Build a calculator"
        assert d["workspace_dir"] == "/tmp/ws"
        assert d["max_iterations"] == 5
        assert d["skill_dirs"] == ["/skills/a"]
        assert d["search_scope"] == "local"

    def test_search_scope_validation(self) -> None:
        for scope in VALID_SEARCH_SCOPES:
            req = TaskRequest(task="test", search_scope=scope)
            assert req.search_scope in VALID_SEARCH_SCOPES


class TestTaskResultContract:
    """TaskResultData must match the documented response schema."""

    def test_queued_result(self) -> None:
        result = TaskResultData(task_id="t-1", status="queued")
        assert result.status in VALID_TASK_STATUSES
        assert result.success is None  # not yet determined

    def test_completed_result(self) -> None:
        result = TaskResultData(
            task_id="t-2",
            status="completed",
            success=True,
            output="Done",
            skill_used="calculator-v1",
            evolved_skills=["calculator-v2"],
            duration_ms=1500,
        )
        assert result.status in VALID_TASK_STATUSES
        assert result.success is True
        assert result.duration_ms > 0

    def test_failed_result(self) -> None:
        result = TaskResultData(
            task_id="t-3",
            status="failed",
            success=False,
            error="Timeout after 30s",
        )
        assert result.status in VALID_TASK_STATUSES
        assert result.success is False
        assert result.error is not None

    def test_all_statuses_valid(self) -> None:
        for status in VALID_TASK_STATUSES:
            result = TaskResultData(task_id="x", status=status)
            assert result.status == status


class TestSkillInfoContract:
    """SkillInfo must match the documented response schema."""

    def test_skill_info_fields(self) -> None:
        info = SkillInfo(
            id="skill-1",
            name="Calculator",
            version="1.0.0",
            active=True,
            created_at="2026-01-01T00:00:00Z",
        )
        d = asdict(info)
        assert set(d.keys()) == {"id", "name", "version", "active", "created_at"}

    def test_skill_serialization_roundtrip(self) -> None:
        info = SkillInfo(id="s1", name="Test", version="0.1", active=False)
        j = json.dumps(asdict(info))
        parsed = json.loads(j)
        assert parsed["id"] == "s1"
        assert parsed["active"] is False


class TestSkillSearchContract:
    """SkillSearchResult must match the documented response schema."""

    def test_search_result_fields(self) -> None:
        result = SkillSearchResult(
            id="s-1",
            name="Web Scraper",
            description="Scrapes websites",
            source="cloud",
            score=0.95,
            imported=False,
        )
        d = asdict(result)
        assert set(d.keys()) == {"id", "name", "description", "source", "score", "imported"}
        assert result.source in ("local", "cloud")
        assert 0.0 <= result.score <= 1.0

    def test_imported_flag(self) -> None:
        local = SkillSearchResult(id="x", name="X", description="", source="local", score=1.0, imported=True)
        assert local.imported is True


class TestHealthContract:
    """HealthStatus must match the documented response schema."""

    def test_health_fields(self) -> None:
        health = HealthStatus(
            status="healthy",
            version="0.1.0",
            initialized=True,
            backends=["shell", "mcp"],
        )
        assert health.status in VALID_HEALTH_STATUSES
        assert isinstance(health.backends, list)

    def test_all_health_statuses(self) -> None:
        for status in VALID_HEALTH_STATUSES:
            h = HealthStatus(status=status, version="0.1", initialized=True)
            assert h.status == status


class TestEnvelopeIntegration:
    """Verify domain data wraps correctly in the envelope."""

    def test_task_result_in_envelope(self) -> None:
        result = TaskResultData(task_id="t-1", status="completed", success=True, output="Done")
        env = APIEnvelope.success(asdict(result), request_id="r-1")
        parsed = json.loads(env.to_json())

        assert parsed["ok"] is True
        assert parsed["data"]["task_id"] == "t-1"
        assert parsed["data"]["status"] == "completed"

    def test_skill_list_in_envelope(self) -> None:
        skills = [
            asdict(SkillInfo(id="s1", name="A", version="1.0", active=True)),
            asdict(SkillInfo(id="s2", name="B", version="2.0", active=False)),
        ]
        env = APIEnvelope.success({"skills": skills, "total": 2, "limit": 50, "offset": 0})
        parsed = json.loads(env.to_json())

        assert parsed["data"]["total"] == 2
        assert len(parsed["data"]["skills"]) == 2

    def test_error_wrapping(self) -> None:
        env = APIEnvelope.failure(
            "SKILL_NOT_FOUND",
            "Skill 'foo' not found",
            request_id="r-3",
            skill_id="foo",
        )
        parsed = json.loads(env.to_json())
        assert parsed["ok"] is False
        assert parsed["error"]["code"] == "SKILL_NOT_FOUND"
        assert parsed["error"]["details"]["skill_id"] == "foo"


class TestSDKTypeConsistency:
    """Cross-check that SDK types stay consistent with the spec."""

    def test_task_request_has_all_spec_fields(self) -> None:
        """TaskRequest must have exactly the fields from the spec."""
        expected = {"task", "workspace_dir", "max_iterations", "skill_dirs", "search_scope"}
        actual = {f.name for f in TaskRequest.__dataclass_fields__.values()}
        assert actual == expected

    def test_task_result_has_all_spec_fields(self) -> None:
        expected = {
            "task_id",
            "status",
            "success",
            "output",
            "tools_used",
            "skill_used",
            "evolved_skills",
            "duration_ms",
            "error",
        }
        actual = {f.name for f in TaskResultData.__dataclass_fields__.values()}
        assert actual == expected

    def test_skill_info_has_all_spec_fields(self) -> None:
        expected = {"id", "name", "version", "active", "created_at"}
        actual = {f.name for f in SkillInfo.__dataclass_fields__.values()}
        assert actual == expected

    def test_health_has_all_spec_fields(self) -> None:
        expected = {"status", "version", "initialized", "backends"}
        actual = {f.name for f in HealthStatus.__dataclass_fields__.values()}
        assert actual == expected

    def test_skill_detail_has_all_spec_fields(self) -> None:
        expected = {"id", "name", "version", "active", "created_at", "manifest", "lineage"}
        actual = {f.name for f in SkillDetail.__dataclass_fields__.values()}
        assert actual == expected

    def test_tool_usage_record_has_all_spec_fields(self) -> None:
        expected = {"tool_name", "arguments", "success"}
        actual = {f.name for f in ToolUsageRecord.__dataclass_fields__.values()}
        assert actual == expected


class TestSkillDetailContract:
    """SkillDetail must match the documented GET /skills/{id} response."""

    def test_detail_includes_manifest_and_lineage(self) -> None:
        detail = SkillDetail(
            id="s-1",
            name="Calculator",
            version="2.0",
            active=True,
            created_at="2026-01-01T00:00:00Z",
            manifest={"entry_point": "main.py", "tools": ["calc"]},
            lineage=["s-0", "s-parent"],
        )
        d = asdict(detail)
        assert "manifest" in d
        assert "lineage" in d
        assert isinstance(d["manifest"], dict)
        assert isinstance(d["lineage"], list)

    def test_detail_superset_of_info(self) -> None:
        """SkillDetail must contain all SkillInfo fields."""
        info_fields = {f.name for f in SkillInfo.__dataclass_fields__.values()}
        detail_fields = {f.name for f in SkillDetail.__dataclass_fields__.values()}
        assert info_fields.issubset(detail_fields)


class TestToolUsageContract:
    """ToolUsageRecord must match the documented tools_used schema."""

    def test_tool_usage_fields(self) -> None:
        record = ToolUsageRecord(tool_name="bash", arguments={"command": "ls"}, success=True)
        d = asdict(record)
        assert d["tool_name"] == "bash"
        assert d["arguments"] == {"command": "ls"}
        assert d["success"] is True

    def test_task_result_with_tools(self) -> None:
        tools = [
            ToolUsageRecord(tool_name="bash", arguments={"cmd": "ls"}),
            ToolUsageRecord(tool_name="python", arguments={"code": "1+1"}, success=False),
        ]
        result = TaskResultData(task_id="t-5", status="completed", success=True, tools_used=tools)
        assert result.tools_used is not None
        assert len(result.tools_used) == 2
        assert result.tools_used[1].success is False


class TestSecurityContract:
    """Security-critical contract assertions."""

    def test_auth_error_envelope(self) -> None:
        """401 responses must use AUTH_REQUIRED error code."""
        env = APIEnvelope.failure("AUTH_REQUIRED", "Missing or invalid bearer token")
        parsed = json.loads(env.to_json())
        assert parsed["ok"] is False
        assert parsed["error"]["code"] == "AUTH_REQUIRED"

    def test_rate_limit_error_envelope(self) -> None:
        """429 responses must use RATE_LIMITED error code."""
        env = APIEnvelope.failure("RATE_LIMITED", "Request rate limit exceeded")
        parsed = json.loads(env.to_json())
        assert parsed["error"]["code"] == "RATE_LIMITED"

    def test_invalid_state_error_envelope(self) -> None:
        """409 responses for invalid operations (e.g., cancel completed task)."""
        env = APIEnvelope.failure(
            "INVALID_STATE",
            "Cannot cancel task in 'completed' state",
            request_id="r-9",
            task_id="t-done",
            current_state="completed",
        )
        parsed = json.loads(env.to_json())
        assert parsed["error"]["code"] == "INVALID_STATE"
        assert parsed["error"]["details"]["current_state"] == "completed"

    _SENSITIVE_KEYS: frozenset[str] = frozenset(
        {
            "token",
            "bearer_token",
            "api_key",
            "secret",
            "password",
            "credential",
            "private_key",
            "mcp_bearer_token",
        }
    )

    def test_config_must_not_expose_sensitive_fields(self) -> None:
        """GET /config must never contain tokens, secrets, or credentials."""
        # Simulate a config response with ONLY safe fields
        safe_config = {
            "model": "gpt-4",
            "max_iterations": 10,
            "search_scope": "all",
            "sandbox_enabled": True,
        }
        env = APIEnvelope.success(safe_config)
        parsed = json.loads(env.to_json())

        for key in parsed["data"]:
            normalized = key.lower()
            assert normalized not in self._SENSITIVE_KEYS, f"Config response contains sensitive key: {key}"

    def test_error_responses_never_leak_tokens(self) -> None:
        """Error messages must not contain token values."""
        token = "sk-secret-abc123-very-long-token-value"
        env = APIEnvelope.failure("AUTH_REQUIRED", "Invalid bearer token provided")
        serialized = env.to_json()
        assert token not in serialized
