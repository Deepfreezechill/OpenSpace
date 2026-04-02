# SkillGuard SDK — REST API Specification

> **Phase:** 1 (design only — implementation in Phase 6)
> **Status:** Draft
> **Version:** 0.1.0

## Overview

The SkillGuard SDK API provides programmatic access to the SkillGuard skill
execution engine.  All endpoints live under the `/api/v2/` prefix and use
JSON request/response bodies.  The API is **async-first**: long-running
operations return a `task_id` for polling.

---

## Base URL

```
https://<host>:<port>/api/v2
```

---

## Authentication

| Mechanism | Header | Notes |
|-----------|--------|-------|
| Bearer token | `Authorization: Bearer <token>` | Required for all endpoints |
| Rate limiting | Per-IP + per-identity sliding window | Configurable via env vars |

Token requirements:
- Minimum 32 characters
- Set via `OPENSPACE_MCP_BEARER_TOKEN` environment variable
- Constant-time comparison (HMAC-based)
- Fail-closed: missing/weak token → 401

---

## Common Response Envelope

All responses use a standard envelope:

```json
{
  "ok": true,
  "data": { ... },
  "error": null,
  "request_id": "uuid-v4"
}
```

Error responses:

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "SKILL_NOT_FOUND",
    "message": "Skill 'foo' not found in registry",
    "details": {}
  },
  "request_id": "uuid-v4"
}
```

---

## Endpoints

### Tasks

#### `POST /api/v2/tasks`

Execute a task through the SkillGuard pipeline.

**Request:**
```json
{
  "task": "string (required)",
  "workspace_dir": "string | null",
  "max_iterations": "integer | null",
  "skill_dirs": ["string"] | null,
  "search_scope": "all | local | cloud"
}
```

**Response (202 Accepted):**
```json
{
  "ok": true,
  "data": {
    "task_id": "uuid-v4",
    "status": "queued",
    "poll_url": "/api/v2/tasks/{task_id}"
  }
}
```

#### `GET /api/v2/tasks/{task_id}`

Poll task execution status and results.

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "task_id": "uuid-v4",
    "status": "running | completed | failed | cancelled",
    "result": {
      "success": true,
      "output": "string",
      "tools_used": [{"tool_name": "string", "arguments": {}, "success": true}],
      "skill_used": "string | null",
      "evolved_skills": ["string"],
      "duration_ms": 1234
    },
    "error": null
  }
}
```

#### `DELETE /api/v2/tasks/{task_id}`

Cancel a running task.

**Response (200):**
```json
{
  "ok": true,
  "data": {"task_id": "uuid-v4", "status": "cancelled"}
}
```

---

### Skills

#### `GET /api/v2/skills`

List skills in the registry.

**Query params:** `?active_only=true&limit=50&offset=0`

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "skills": [
      {
        "id": "string",
        "name": "string",
        "version": "string",
        "active": true,
        "created_at": "ISO-8601"
      }
    ],
    "total": 42,
    "limit": 50,
    "offset": 0
  }
}
```

#### `GET /api/v2/skills/{skill_id}`

Get detailed skill information.

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "id": "string",
    "name": "string",
    "version": "string",
    "active": true,
    "created_at": "ISO-8601",
    "manifest": { "...SkillManifest fields..." },
    "lineage": ["parent_id_1", "parent_id_2"]
  }
}
```

#### `GET /api/v2/skills/search`

Search skills across local and cloud registries.

**Query params:** `?q=query&source=all|local|cloud&limit=20&auto_import=true`

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "results": [
      {
        "id": "string",
        "name": "string",
        "description": "string",
        "source": "local | cloud",
        "score": 0.95,
        "imported": false
      }
    ],
    "total": 15
  }
}
```

#### `POST /api/v2/skills/{skill_id}/fix`

Trigger fix-evolution on a broken skill.

**Request:**
```json
{
  "direction": "string (required)"
}
```

**Response (202 Accepted):**
```json
{
  "ok": true,
  "data": {
    "evolution_id": "uuid-v4",
    "status": "queued",
    "poll_url": "/api/v2/evolutions/{evolution_id}"
  }
}
```

#### `POST /api/v2/skills/{skill_id}/upload`

Upload a skill to cloud.

**Request:**
```json
{
  "visibility": "public | private",
  "tags": ["string"],
  "change_summary": "string | null"
}
```

**Response (201 Created):**
```json
{
  "ok": true,
  "data": {
    "cloud_id": "string",
    "url": "string",
    "visibility": "public"
  }
}
```

---

### Evolutions

#### `GET /api/v2/evolutions/{evolution_id}`

Poll evolution status.

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "evolution_id": "uuid-v4",
    "skill_id": "string",
    "status": "running | completed | failed",
    "result": {
      "improved": true,
      "changes_summary": "string",
      "new_version": "string"
    }
  }
}
```

---

### System

#### `GET /api/v2/health`

Health check.

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "status": "healthy",
    "version": "0.1.0",
    "initialized": true,
    "backends": ["shell", "mcp", "gui"]
  }
}
```

#### `GET /api/v2/config`

Get current configuration (non-sensitive fields only).

> **Security:** This endpoint MUST NOT expose bearer tokens, API keys,
> secret values, internal file paths, or database credentials.  Only
> operational settings are returned.

**Response (200):**
```json
{
  "ok": true,
  "data": {
    "model": "string",
    "max_iterations": 10,
    "search_scope": "all",
    "sandbox_enabled": true
  }
}
```

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `AUTH_REQUIRED` | 401 | Missing or invalid bearer token |
| `RATE_LIMITED` | 429 | Request rate limit exceeded |
| `TASK_NOT_FOUND` | 404 | Task ID does not exist |
| `SKILL_NOT_FOUND` | 404 | Skill ID not in registry |
| `TASK_FAILED` | 500 | Task execution failed |
| `NOT_INITIALIZED` | 503 | SkillGuard not initialized |
| `VALIDATION_ERROR` | 422 | Invalid request body |
| `EVOLUTION_NOT_FOUND` | 404 | Evolution ID does not exist |
| `INVALID_STATE` | 409 | Operation invalid for current state (e.g., cancel completed task) |

---

## Rate Limiting

- **Per-token:** Configurable via `OPENSPACE_RATE_LIMIT_PER_TOKEN` (default: 60/min)
- **Per-IP:** Configurable via `OPENSPACE_RATE_LIMIT_PER_IP` (default: 120/min)
- **Window:** Configurable via `OPENSPACE_RATE_LIMIT_WINDOW` (default: 60s)

Rate limit headers on every response:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1680000000
```

---

## Versioning

- API version in URL path (`/api/v2/`)
- Breaking changes require version bump
- v1 (dashboard API) remains available for backward compatibility
