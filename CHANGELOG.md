# Changelog

All notable changes to OpenSpace are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0] — 2026-04-07

### Overview

OpenSpace v2.0.0 is a ground-up architectural overhaul of the self-evolving skill engine for AI agents. Across 8 phases (P0–P7) and 50+ epics, the project was hardened, decomposed, and extended from a monolithic prototype into a production-grade, security-first platform. The codebase grew from ~500 tests to **2,174 tests with zero failures**. Every epic was reviewed via multi-agent code review (/8eyes + /collab) before merge.

---

### Phase 0 — Emergency Hardening
_11 epics · PRs #433–#467 (upstream) · Immediate security stabilization_

- **Bearer token auth** on all MCP HTTP endpoints (Epic 0.1, PR #433)
- **Sliding-window rate limiting** on MCP HTTP endpoints (Epic 0.1b, PR #434)
- **E2B sandbox enforcement** — fail-closed defaults, trusted config only (Epic 0.2)
- **Traceback scrubbing** — removed stack trace exposure from all MCP error paths (Epic 0.3a)
- **Env var isolation** — sandbox execution can no longer leak host environment (Epic 0.3b)
- **Cloud auto-import disabled** by default — skills must be explicitly trusted (Epic 0.4)
- **AST safety scanner** — static analysis for dangerous API usage (os.exec, subprocess, eval) (Epic 0.5)
- **Test infrastructure** — foundational test harness and fixtures (Epic 0.6)
- **MCP end-to-end tests** — execution paths, error paths, coverage gate (Epic 0.7)
- **GitHub Actions CI** — automated pipeline with lint, test, type-check, security audit (Epic 0.8)
- **Benchmark decoupling** — extracted benchmark code from production paths (Epic 0.9)
- **Dependency pinning** — all deps pinned with `pip-audit` infrastructure; blocked litellm PYSEC-2026-2 supply-chain vector (Epic 0.10)

### Phase 1 — Foundation Architecture
_8 epics (7 bullets — Epics 1.1 + 1.2 combined) · PRs #451–#459 (upstream) · Clean architecture layer_

- **Domain layer** — Protocol-based interfaces and frozen dataclass types for all core concepts (Epics 1.1 + 1.2)
- **AppContainer** — composition root with lifecycle hooks, dependency injection, graceful startup/shutdown (Epic 1.3)
- **Property delegation** — clean public accessor surface on AppContainer (Epic 1.4)
- **Exception hierarchy** — domain-specific exceptions with centralized error mapping to HTTP/MCP codes (Epic 1.5)
- **Structured logging** — structlog integration with context propagation, automatic PII redaction, and correlation IDs (Epic 1.6)
- **Architecture boundary tests** — enforced layering rules (no domain→infra imports, no circular deps) (Epic 1.7)
- **SDK API design** — public surface documentation and contract tests for stable API guarantees (Epic 1.8)

### Phase 2 — Smart Sandbox
_6 epics · PRs #461–#471 + PR #1 · Capability-based security model_

- **Capability lease system** — schema-driven capability grants with tier defaults, expiration, and revocation (Epic 2.1, PR #461)
- **Filesystem broker** — jail-based file access with TOCTOU race protection and path traversal prevention (Epic 2.2, PR #463)
- **Network proxy** — domain allowlisting, port enforcement, connection limits; blocks SSRF via loopback/link-local and apex domain rebinding (Epic 2.3, PR #465)
- **Process broker** — command allowlisting, shell execution control, syscall filtering (Epic 2.4, PR #469)
- **MCP auth advanced** — HMAC token signing, per-tool authorization, tier-gated access control (Epic 2.5, PR #471)
- **Secret broker** — scoped storage with encryption at rest, owner isolation, capability-based access. 64 tests, 5 review rounds (Epic 2.6, PR #1)

### Phase 3 — Skill Store Decomposition
_7 epics · PRs #12–#28 · 1,356 tests passing_

Decomposed the monolithic `store.py` (1,345 lines) into 5 focused modules:

- **SkillSchema + SkillVersion models** — frozen dataclasses with validation (Epic 3.1, PR #12)
- **SkillRepository** — CRUD operations with LIKE-escape safety and shared-lock architecture (Epic 3.2, PR #15)
- **LineageTracker** — parent-child derivation graph with cycle detection and diamond DAG support (Epic 3.3, PR #17)
- **AnalysisStore** — quality scores, evolution candidates, atomic bulk inserts, dedup protection (Epic 3.4, PR #19)
- **TagIndex + SearchEngine** — tag dedup, match_all queries, skill analytics (Epic 3.5, PR #21)
- **MigrationManager** — DDL consolidation, WAL recovery, atomic migrations, injection guard (Epic 3.6, PR #23)
- **P3 integration + cleanup** — store.py reduced to re-export facade; 7 E2E blockers fixed; thread-safe close(); batch hydration + mypy fixes (Epic 3.7, PRs #25–#28)

### Phase 4 — Tool Layer + MCP Decomposition
_10 epics (2 skipped) · PRs #31–#38 · 1,408 tests passing_

Decomposed `tool_layer.py` (788 lines) and `mcp_server.py` (992 lines):

- **ToolRegistry** — tool registration, discovery, and backend listing extracted (Epic 4.1, PR #31)
- **ExecutionEngine** — task execution with skill injection and fallback orchestration (Epic 4.3, PR #32)
- **RecordingService** — execution trace capture, analysis triggers, evolution hooks (Epic 4.4, PR #33)
- **LLMFactory** — provider-agnostic LLM client initialization (Epic 4.5, PR #34)
- **OpenSpace facade** — tool_layer.py reduced to thin composition facade preserving public API (Epic 4.6, PR #35)
- **MCPToolHandlers** — all 4 MCP tool implementations (execute, search, fix, upload) extracted (Epic 4.7, PR #36)
- **MCPServer package** — `openspace/mcp/` package with clean server lifecycle and endpoint wiring (Epic 4.9, PR #37)
- **P4 integration tests** — 26 outcome-focused tests + MCP protocol conformance (Epic 4.10, PR #38)
- _Skipped:_ Epic 4.2 (SkillSelector — covered by 4.1), Epic 4.8 (MCPAuthMiddleware — covered by P3)

### Phase 5 — Skill Engine / Grounding Decomposition
_11 epics · PRs #39–#49 · Two sub-phases (P5a: evolver, P5b: grounding agent)_

**P5a — Evolution Engine** (evolver.py → `openspace/skill_engine/evolution/` package):
- **EvolutionModels** — domain types: EvolutionTrigger, EvolutionContext, name sanitizer (Epic 5.1, PR #39)
- **EvolutionOrchestrator** — scheduling, dispatch, background execution (Epic 5.2, PR #40)
- **Evolution triggers** — analysis, tool degradation, and metric monitor as separate modules (Epic 5.3, PR #41)
- **EvolutionConfirmation** — LLM-based approval flow with negation pattern parsing (Epic 5.4, PR #42)
- **Evolution engines** — FIX/DERIVED/CAPTURED strategies in loop.py + strategies.py (Epic 5.5, PR #43)
- **P5a capstone** — evolver.py reduced to thin facade (**80% code reduction**); formatting helpers extracted (Epic 5.6, PR #44)

**P5b — Grounding Agent** (grounding_agent.py → `openspace/agents/grounding/` package):
- **SkillContext + MessageSafety** — context management and message truncation with cap safety (Epic 5.7, PR #45)
- **ExecutionLoop + PromptBuilder** — core reasoning loop decoupled from tool/visual concerns (Epic 5.8, PR #46)
- **ToolCatalog + Visual + Workspace** — tool discovery, visual analysis callbacks, workspace scanning with MRO safety (Epic 5.9, PR #47)
- **ResultFormatter + Telemetry** — result building and execution recording; stripped str(e) leaks (Epic 5.10, PR #48)
- **P5b capstone** — grounding_agent.py replaced with package; private symbols cleaned from `__all__` (Epic 5.11, PR #49)

### Phase 6 — Production Readiness
_4 epics · PRs #50–#53 · 1,957 tests passing_

- **Observability** — Prometheus-compatible metrics (execution latency, skill hit rate, evolution success rate), distributed tracing for multi-step workflows, health check endpoints. Isolated from execution path per adversarial review (Epic 6.1, PR #50)
- **SLOs** — error budget tracking, multi-window burn-rate alerting, NaN-safe JSON serialization, MCP tool for SLO queries (Epic 6.2, PR #51)
- **Deployment architecture** — multi-stage Dockerfile (non-root), `DeployConfig` frozen dataclass, `GracefulShutdownHandler` with 70/30 drain/hook budget, `/health` endpoint bypasses auth for K8s probes, `HEALTHCHECK` instruction (Epic 6.3, PR #52)
- **DX polish** — Rich `--help` with examples/env vars, `--version`, preflight checks (bearer token, skill store, port, Python version), platform-aware suggestions (PowerShell on Windows, bash on Unix), errors write to `sys.__stderr__` for console visibility (Epic 6.4, PR #53)

### Phase 7 — Safety & Launch
_3 epics · PRs #54–#56 · 2,174 tests passing_

- **ReviewGate** — 5-layer security gate for skill evolution: path traversal detection, extension allowlist, size limits, HIGH+CRITICAL AST blocking, lineage validation. Windows drive-relative paths and reserved device names handled. 63 tests (45 adversarial), 3 review rounds with 12 agent-reviews (Epic 7.1, PR #54)
- **SkillGuard** — pre-persist quality gates wrapping ReviewGate + SkillStore. All skill mutations (evolve_fix, evolve_derived, evolve_captured) routed through guard. Deep disk rollback via rglob snapshot on rejection. **40+ AST blocklist patterns** covering: `builtins` bypass, `__builtins__` attribute access, `sys.modules` manipulation, `os.exec*`/`os.spawn*`, `pty`/`code` modules, HTTP exfiltration, `marshal`/`runpy`, `asyncio.subprocess`, `signal`, `webbrowser`, `multiprocessing`, `shutil`, filesystem destructors, `os.chmod`/`os.chown`. Bare-name alias bypass prevention (Epic 7.2, PR #55)
- **Launch checklist** — security audit, documentation, changelog, license verification, dependency audit (Epic 7.3, in progress)

---

### Security

Key security improvements across the v2.0.0 release:

- **Fail-closed by default** — all security decisions default to deny
- **Bearer token + HMAC auth** on all MCP endpoints with per-tool authorization
- **Capability lease model** — tier-gated access with expiration and revocation
- **Filesystem jailing** — TOCTOU-safe path enforcement with traversal prevention
- **Network SSRF protection** — blocks loopback, link-local, and apex domain rebinding
- **AST-based code scanning** — 40+ blocklist patterns for RCE, sandbox escape, data exfiltration
- **5-layer ReviewGate** — every evolved skill passes path, extension, size, AST, and lineage checks
- **Deep disk rollback** — failed skill mutations are fully reverted including subdirectory files
- **PII redaction** — structured logging automatically strips sensitive fields
- **Supply-chain protection** — pinned deps, `pip-audit` in CI, blocked PYSEC-2026-2

### Dependencies

- **litellm** — bumped to ≥1.83.0 (fixes CVE-2026-35029 + CVE-2026-35030; skips PYSEC-2026-2 supply-chain attack in 1.82.7/1.82.8)
- **structlog** — added for structured logging with context propagation
- **E2B sandbox** — enforced as default execution environment
- **GitHub Actions** — full CI pipeline: lint (ruff), type-check (mypy), test (pytest), security audit (pip-audit)
- **Docker** — multi-stage build, non-root user, health check probe

### Migration Notes

- `store.py` is now a re-export facade — direct imports still work but should migrate to specific modules (`skill_repository`, `lineage_tracker`, `analysis_store`, `tag_index_search`, `migration_manager`)
- `tool_layer.py` is now a thin facade — migrate to `tool_registry`, `execution_engine`, `recording_service`, `llm_adapter`
- `mcp_server.py` is now a backward-compat shim — migrate to `openspace/mcp/` package
- `evolver.py` is now a thin facade — migrate to `openspace/skill_engine/evolution/` package
- `grounding_agent.py` replaced by `openspace/agents/grounding/` package
- All MCP endpoints now require bearer token auth (set `OPENSPACE_BEARER_TOKEN`)
- Cloud auto-import disabled by default (set `OPENSPACE_CLOUD_IMPORT=true` to re-enable)
- E2B sandbox enforced by default for skill execution

### Stats

| Metric | Value |
|--------|-------|
| Phases | 8 (P0–P7) |
| Epics completed | 50+ |
| Pull requests | 55+ |
| Test count | 2,174 (zero failures) |
| Review rounds | Multi-agent (/8eyes + /collab) on every PR |
| Monoliths decomposed | 5 (store.py, tool_layer.py, mcp_server.py, evolver.py, grounding_agent.py) |
| AST blocklist patterns | 40+ |
| Security layers | 5 (ReviewGate) + capability leases + auth |
