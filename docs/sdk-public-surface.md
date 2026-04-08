# SkillGuard SDK — Public API Surface

> **Version:** 2.0.0
> **Status:** Stable

## Overview

This document defines which SkillGuard operations are SDK-accessible, the
authentication model, rate limits, and data types exposed to SDK consumers.

---

## SDK-Accessible Operations

### Tier 1 — Core Operations (launch priority)

| Operation | Method | Endpoint | Description |
|-----------|--------|----------|-------------|
| Execute task | POST | `/api/v2/tasks` | Run task through full pipeline |
| Poll task | GET | `/api/v2/tasks/{id}` | Get execution status/result |
| Cancel task | DELETE | `/api/v2/tasks/{id}` | Cancel running task |
| Search skills | GET | `/api/v2/skills/search` | Query local + cloud skills |
| List skills | GET | `/api/v2/skills` | List registered skills |
| Get skill | GET | `/api/v2/skills/{id}` | Skill details + manifest |
| Health check | GET | `/api/v2/health` | System health + version |

### Tier 2 — Management Operations

| Operation | Method | Endpoint | Description |
|-----------|--------|----------|-------------|
| Fix skill | POST | `/api/v2/skills/{id}/fix` | Trigger fix-evolution |
| Upload skill | POST | `/api/v2/skills/{id}/upload` | Publish to cloud |
| Poll evolution | GET | `/api/v2/evolutions/{id}` | Evolution status |
| Get config | GET | `/api/v2/config` | Non-sensitive config |

### Not SDK-Accessible (internal only)

| Operation | Reason |
|-----------|--------|
| `initialize()` | Server lifecycle — managed by host process |
| `cleanup()` | Server lifecycle — managed by host process |
| Direct LLM calls | Security boundary — must go through task pipeline |
| Sandbox control | Security boundary — managed by policy engine |
| Secret access | Security boundary — never exposed via API |
| Recording management | Internal telemetry — not for consumers |

---

## Authentication Model

### Bearer Token Authentication

```
Authorization: Bearer <token>
```

**Token lifecycle:**
1. Admin generates token (min 32 chars) and sets `SCION_MCP_BEARER_TOKEN`
2. SDK client includes token in every request
3. Server validates via constant-time HMAC comparison
4. Invalid/missing token → 401 immediately (fail-closed)

**Security properties:**
- Constant-time comparison prevents timing attacks
- Minimum length enforced at startup
- No token = server refuses all requests
- Tokens are never logged or included in error responses

### Future Auth Enhancements (Phase 5)

| Feature | Phase | Notes |
|---------|-------|-------|
| API key rotation | 5 | Zero-downtime key rotation |
| OAuth2 / OIDC | 5 | For multi-tenant deployments |
| Scoped permissions | 5 | Read-only vs execute tokens |
| mTLS | 5 | Certificate-based for infra |

---

## Rate Limits

| Scope | Default | Env Var |
|-------|---------|---------|
| Per-token | 60 req/min | `SCION_RATE_LIMIT_PER_TOKEN` |
| Per-IP | 120 req/min | `SCION_RATE_LIMIT_PER_IP` |
| Window | 60 seconds | `SCION_RATE_LIMIT_WINDOW` |

**Behavior on limit:**
- HTTP 429 with `Retry-After` header
- Sliding window (not fixed buckets)
- Per-identity and per-IP limits enforced independently

---

## SDK Data Types

These types form the SDK's data model. All are immutable dataclasses
serialized as JSON.

### Request Types

```python
@dataclass(frozen=True)
class TaskRequest:
    task: str
    workspace_dir: str | None = None
    max_iterations: int | None = None
    skill_dirs: list[str] | None = None
    search_scope: str = "all"  # "all" | "local" | "cloud"

@dataclass(frozen=True)
class SkillSearchRequest:
    query: str
    source: str = "all"  # "all" | "local" | "cloud"
    limit: int = 20
    auto_import: bool = True

@dataclass(frozen=True)
class SkillFixRequest:
    direction: str

@dataclass(frozen=True)
class SkillUploadRequest:
    visibility: str = "public"  # "public" | "private"
    tags: list[str] | None = None
    change_summary: str | None = None
```

### Response Types

```python
@dataclass(frozen=True)
class ToolUsageRecord:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = True

@dataclass(frozen=True)
class TaskResult:
    task_id: str
    status: str  # "queued" | "running" | "completed" | "failed" | "cancelled"
    success: bool | None = None
    output: str | None = None
    tools_used: list[ToolUsageRecord] | None = None
    skill_used: str | None = None
    evolved_skills: list[str] | None = None
    duration_ms: int | None = None
    error: str | None = None

@dataclass(frozen=True)
class SkillInfo:
    id: str
    name: str
    version: str
    active: bool
    created_at: str  # ISO-8601

@dataclass(frozen=True)
class SkillDetail:
    """Extended skill view for GET /skills/{id}."""
    id: str
    name: str
    version: str
    active: bool
    created_at: str  # ISO-8601
    manifest: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class SkillSearchResult:
    id: str
    name: str
    description: str
    source: str  # "local" | "cloud"
    score: float
    imported: bool = False

@dataclass(frozen=True)
class HealthStatus:
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    initialized: bool
    backends: list[str]
```

### Domain Types (re-exported via SDK)

| Type | Module | Description |
|------|--------|-------------|
| `SkillIdentity` | `scion.domain.types` | Skill ID + version |
| `SkillManifest` | `scion.domain.types` | Full skill metadata |
| `ToolDescriptor` | `scion.domain.types` | Tool name + schema |
| `ToolCallResult` | `scion.domain.types` | Tool execution result |
| `SandboxPolicy` | `scion.domain.types` | Security policy config |

---

## SDK Client Interface (Python)

> Implementation in Phase 6 (`scion-sdk` package)

```python
class SkillGuardClient:
    """Async Python client for the SkillGuard API."""

    def __init__(self, base_url: str, token: str) -> None: ...

    # Tier 1 — Core
    async def execute(self, request: TaskRequest) -> TaskResult: ...
    async def poll_task(self, task_id: str) -> TaskResult: ...
    async def cancel_task(self, task_id: str) -> TaskResult: ...
    async def search_skills(self, request: SkillSearchRequest) -> list[SkillSearchResult]: ...
    async def list_skills(self, *, active_only: bool = False) -> list[SkillInfo]: ...
    async def get_skill(self, skill_id: str) -> SkillDetail: ...
    async def health(self) -> HealthStatus: ...

    # Tier 2 — Management
    async def fix_skill(self, skill_id: str, request: SkillFixRequest) -> str: ...
    async def upload_skill(self, skill_id: str, request: SkillUploadRequest) -> dict: ...
    async def poll_evolution(self, evolution_id: str) -> dict: ...
    async def get_config(self) -> dict: ...

    # Context manager
    async def __aenter__(self) -> "SkillGuardClient": ...
    async def __aexit__(self, *exc: Any) -> None: ...
```

---

## Backward Compatibility

- `/api/v1/` (dashboard API) remains unchanged
- `/api/v2/` is the SDK surface — breaking changes require version bump
- MCP tool interface continues to work alongside REST API
- Both APIs share the same SkillGuard instance
