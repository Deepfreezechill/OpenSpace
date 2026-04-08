"""End-to-end OUTCOME tests for OpenSpace v2.0.0.

These tests validate that the system DOES WHAT USERS EXPECT, not just
that individual functions return correct types.  Real objects are used
wherever possible; only LLM and cloud calls are mocked.

Test Matrix (10 outcome categories):
  1. Health Check     — system reports healthy with correct version
  2. Platform Info    — server returns real platform metadata
  3. Skill Store      — full CRUD + search + lineage lifecycle
  4. Review Gate      — blocks unsafe code, passes safe code
  5. Execution Engine — submit task → get structured result (orchestration-level;
                        grounding agent is mocked since it requires a live LLM)
  6. MCP Server       — tools are registered and discoverable
  7. Evolution        — skill evolves, lineage preserved, quarantine works
  8. Command Execute  — server runs commands and returns output
  9. Version          — consistent version across all entry points
 10. Analysis         — execution analysis persistence and counters

NOTE on local server auth: The Flask desktop server (local_server/main.py)
runs on 127.0.0.1 by design — it is a localhost-only desktop automation
endpoint, not a network-facing service.  Auth is enforced on the MCP server
path (see test_auth_integration.py).
"""

from __future__ import annotations

import asyncio
import platform as _platform_mod
import textwrap
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# Canonical version — single source of truth for all version assertions
from openspace import __version__ as EXPECTED_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine from sync test code.

    Handles both standalone and pytest-asyncio environments.
    NOTE: SkillStore uses asyncio.to_thread internally, which creates its
    own connections, so cross-thread SQLite access is safe.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _make_skill_record(
    *,
    skill_id=None,
    name="test_skill",
    description="A test skill",
    origin="imported",
    generation=0,
    parent_skill_ids=None,
    content_snapshot=None,
    is_active=True,
    tags=None,
):
    """Build a SkillRecord with sensible defaults."""
    from openspace.skill_engine.types import (
        SkillCategory,
        SkillLineage,
        SkillOrigin,
        SkillRecord,
    )

    if skill_id is None:
        skill_id = f"{name}__e2e_{uuid.uuid4().hex[:8]}"
    if parent_skill_ids is None:
        parent_skill_ids = []
    if content_snapshot is None:
        content_snapshot = {
            "SKILL.md": f"# {name}\n\nA test skill.\n\n## Steps\n\n1. Do something.\n",
            "helper.py": "# Safe helper\ndef greet():\n    return 'hello'\n",
        }

    return SkillRecord(
        skill_id=skill_id,
        name=name,
        description=description,
        path=f"/skills/{name}",
        is_active=is_active,
        category=SkillCategory.WORKFLOW,
        tags=tags or ["test"],
        lineage=SkillLineage(
            origin=SkillOrigin(origin),
            generation=generation,
            parent_skill_ids=parent_skill_ids,
            content_snapshot=content_snapshot,
            change_summary="initial import" if not parent_skill_ids else "evolved",
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 1: Health Check — "Can I trust the system is healthy?"
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthCheckOutcome:
    """User expectation: GET / returns status=ok with version 2.0.0."""

    def test_health_returns_ok_status(self):
        """OUTCOME: Health endpoint says the system is operational."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.get("/")
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "ok"
        assert data["service"] == "OpenSpace Desktop Server"

    def test_health_reports_correct_version(self):
        """OUTCOME: Reported version matches the release we shipped."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.get("/")
        data = resp.get_json()

        assert data["version"] == EXPECTED_VERSION, (
            f"Health endpoint reports {data['version']}, expected {EXPECTED_VERSION}"
        )

    def test_health_includes_features(self):
        """OUTCOME: User can see what capabilities are available."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.get("/")
        data = resp.get_json()

        assert "features" in data, "Health response must include features dict"
        assert isinstance(data["features"], dict)
        assert len(data["features"]) > 0, (
            "Features dict must not be empty — system should report capabilities"
        )

    def test_health_includes_fresh_timestamp(self):
        """OUTCOME: User can verify the response is fresh, not cached stale."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.get("/")
        data = resp.get_json()

        assert "timestamp" in data
        ts = datetime.fromisoformat(data["timestamp"])
        delta = abs((datetime.now() - ts).total_seconds())
        assert delta < 5, f"Timestamp is {delta}s old — expected fresh (<5s)"


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 2: Platform Info — "What system am I running on?"
# ═══════════════════════════════════════════════════════════════════════════


class TestPlatformInfoOutcome:
    """User expectation: GET /platform returns real system metadata."""

    def test_platform_returns_system_info(self):
        """OUTCOME: User gets real, actionable platform information."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.get("/platform")
        data = resp.get_json()

        assert resp.status_code == 200
        assert "system" in data
        assert data["system"] in ("Windows", "Linux", "Darwin"), (
            f"Unexpected platform: {data['system']}"
        )
        assert "release" in data
        assert "machine" in data


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 3: Skill Store Lifecycle — "Can I manage skills?"
# ═══════════════════════════════════════════════════════════════════════════


class TestSkillStoreLifecycleOutcome:
    """User expectation: I can create, find, browse, and evolve skills
    using a real database — not mocks."""

    def test_create_and_retrieve_skill(self, in_memory_store):
        """OUTCOME: User saves a skill and can retrieve it later."""
        record = _make_skill_record(name="weather_lookup")
        _run(in_memory_store.save_record(record))

        loaded = in_memory_store.load_record(record.skill_id)
        assert loaded is not None, "Saved skill must be retrievable"
        assert loaded.name == "weather_lookup"
        assert loaded.description == "A test skill"
        assert loaded.is_active is True

    def test_search_skills_by_tag(self, in_memory_store):
        """OUTCOME: User searches for skills by tag and finds matches."""
        skills = [
            _make_skill_record(name="api_caller", tags=["api", "http"]),
            _make_skill_record(name="data_parser", tags=["data", "json"]),
            _make_skill_record(name="api_tester", tags=["api", "testing"]),
        ]
        for s in skills:
            _run(in_memory_store.save_record(s))

        # find_skills_by_tags returns List[str] (skill IDs)
        matching_ids = in_memory_store.find_skills_by_tags(["api"])
        # Load the matching records to check names
        matching_names = []
        for sid in matching_ids:
            rec = in_memory_store.load_record(sid)
            if rec:
                matching_names.append(rec.name)

        assert "api_caller" in matching_names, "Tag search must find matching skills"
        assert "api_tester" in matching_names
        assert "data_parser" not in matching_names, "Non-matching skills must be excluded"

    def test_evolve_skill_creates_new_version(self, in_memory_store):
        """OUTCOME: When a skill evolves, a new version exists and the old one
        is deactivated (for FIXED origin)."""
        parent = _make_skill_record(name="code_review")
        _run(in_memory_store.save_record(parent))

        child = _make_skill_record(
            name="code_review",
            origin="fixed",
            generation=1,
            parent_skill_ids=[parent.skill_id],
            content_snapshot={
                "SKILL.md": "# code_review\n\nImproved version.\n\n## Steps\n\n1. Better review.\n",
                "helper.py": "def review():\n    return 'improved'\n",
            },
        )
        _run(in_memory_store.evolve_skill(child, [parent.skill_id]))

        # New version is active
        loaded_child = in_memory_store.load_record(child.skill_id)
        assert loaded_child is not None
        assert loaded_child.is_active is True
        assert loaded_child.lineage.generation == 1

        # Old version is deactivated
        loaded_parent = in_memory_store.load_record(parent.skill_id)
        assert loaded_parent is not None
        assert loaded_parent.is_active is False, (
            "FIXED evolution must deactivate the parent version"
        )

    def test_skill_count_reflects_reality(self, in_memory_store):
        """OUTCOME: User can count how many skills exist in the system."""
        for i in range(5):
            _run(in_memory_store.save_record(
                _make_skill_record(name=f"skill_{i}")
            ))

        count = in_memory_store.count()
        assert count == 5, f"Expected 5 skills, got {count}"

    def test_delete_skill_removes_it(self, in_memory_store):
        """OUTCOME: User deletes a skill and it's gone."""
        record = _make_skill_record(name="obsolete_skill")
        _run(in_memory_store.save_record(record))
        assert in_memory_store.load_record(record.skill_id) is not None

        _run(in_memory_store.delete_record(record.skill_id))
        assert in_memory_store.load_record(record.skill_id) is None, (
            "Deleted skill must not be retrievable"
        )

    def test_lineage_ancestry_chain(self, in_memory_store):
        """OUTCOME: User can trace back through a skill's full evolution history."""
        # Create a 3-generation lineage: v0 → v1 → v2
        v0 = _make_skill_record(name="auto_deploy", tags=["deploy"])
        _run(in_memory_store.save_record(v0))

        v1 = _make_skill_record(
            name="auto_deploy",
            origin="fixed",
            generation=1,
            parent_skill_ids=[v0.skill_id],
            tags=["deploy"],
        )
        _run(in_memory_store.evolve_skill(v1, [v0.skill_id]))

        v2 = _make_skill_record(
            name="auto_deploy",
            origin="fixed",
            generation=2,
            parent_skill_ids=[v1.skill_id],
            tags=["deploy"],
        )
        _run(in_memory_store.evolve_skill(v2, [v1.skill_id]))

        # Only v2 should be active
        all_records = in_memory_store.load_all()  # Dict[str, SkillRecord]
        active = [r for r in all_records.values() if r.is_active and r.name == "auto_deploy"]
        assert len(active) == 1, "Only the latest version should be active"
        assert active[0].skill_id == v2.skill_id

        # Full ancestry exists
        ancestors = in_memory_store.get_ancestry(v2.skill_id)
        ancestor_ids = [a.skill_id for a in ancestors]
        assert v1.skill_id in ancestor_ids, "v1 must be in v2's ancestry"
        assert v0.skill_id in ancestor_ids, "v0 must be in v2's ancestry"


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 4: Review Gate — "Does the system block unsafe code?"
# ═══════════════════════════════════════════════════════════════════════════


class TestReviewGateSecurityOutcome:
    """User expectation: Unsafe skills are caught and blocked before
    they can harm the system. Safe skills pass through."""

    def test_safe_skill_passes_review(self):
        """OUTCOME: A well-formed, safe skill is approved by the gate."""
        from openspace.skill_engine.review_gate import ReviewGate

        record = _make_skill_record(
            name="safe_skill",
            content_snapshot={
                "SKILL.md": "# Safe Skill\n\nDoes safe things.\n",
                "main.py": "def run():\n    return 'safe'\n",
            },
        )

        gate = ReviewGate()
        result = gate.review(record)

        assert result.passed, (
            f"Safe skill should pass review, but got: "
            f"{[(c.name, c.verdict, c.detail) for c in result.checks]}"
        )

    def test_dangerous_code_blocked(self):
        """OUTCOME: A skill with dangerous AST patterns is REJECTED."""
        from openspace.skill_engine.review_gate import ReviewGate

        record = _make_skill_record(
            name="evil_skill",
            content_snapshot={
                "SKILL.md": "# Evil Skill\n\nDoes bad things.\n",
                "exploit.py": textwrap.dedent("""\
                    import subprocess
                    subprocess.call(['id'])
                """),  # nosec B603 — test data for AST scanner validation
            },
        )

        gate = ReviewGate()
        result = gate.review(record)

        assert not result.passed, "Dangerous code MUST be blocked by ReviewGate"
        ast_check = next((c for c in result.checks if c.name == "ast-safety"), None)
        assert ast_check is not None
        assert ast_check.verdict == "fail"

    def test_path_traversal_blocked(self):
        """OUTCOME: A skill trying to escape its directory is REJECTED."""
        from openspace.skill_engine.review_gate import ReviewGate

        record = _make_skill_record(
            name="traversal_skill",
            content_snapshot={
                "SKILL.md": "# Traversal\n\nNotes.\n",
                "../../etc/passwd": "root:x:0:0:root:/root:/bin/bash",
            },
        )

        gate = ReviewGate()
        result = gate.review(record)

        assert not result.passed, "Path traversal MUST be blocked"

    def test_disallowed_file_types_blocked(self):
        """OUTCOME: A skill containing forbidden file types is REJECTED."""
        from openspace.skill_engine.review_gate import ReviewGate

        record = _make_skill_record(
            name="template_skill",
            content_snapshot={
                "SKILL.md": "# Template\n\nUses Jinja.\n",
                "template.jinja": "{{ user_input | safe }}",  # SSTI risk
            },
        )

        gate = ReviewGate()
        result = gate.review(record)

        assert not result.passed, "Jinja template files MUST be blocked (SSTI risk)"

    def test_missing_skill_md_blocked(self):
        """OUTCOME: A skill without SKILL.md is incomplete and REJECTED."""
        from openspace.skill_engine.review_gate import ReviewGate

        record = _make_skill_record(
            name="incomplete_skill",
            content_snapshot={
                "main.py": "def run():\n    pass\n",
            },
        )

        gate = ReviewGate()
        result = gate.review(record)

        assert not result.passed, "Skill without SKILL.md MUST fail content check"
        content_check = next((c for c in result.checks if c.name == "content"), None)
        assert content_check is not None
        assert content_check.verdict == "fail"

    def test_windows_reserved_names_blocked(self):
        """OUTCOME: Skills with Windows reserved device names are REJECTED."""
        from openspace.skill_engine.review_gate import ReviewGate

        record = _make_skill_record(
            name="windows_exploit",
            content_snapshot={
                "SKILL.md": "# WinExploit\n\nBad.\n",
                "CON.py": "print('this hangs on Windows')",
            },
        )

        gate = ReviewGate()
        result = gate.review(record)

        assert not result.passed, "Windows reserved names MUST be blocked"


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 5: Execution Engine — "Can I submit a task and get results?"
# ═══════════════════════════════════════════════════════════════════════════


class TestExecutionEngineOutcome:
    """User expectation: Submit a task → get a structured result with
    status, response, execution_time."""

    @pytest.fixture
    def fake_grounding_agent(self):
        """A fake grounding agent that returns canned responses."""
        agent = AsyncMock()
        agent.process = AsyncMock(return_value={
            "status": "completed",
            "response": "The weather in Seattle is 65°F and cloudy.",
        })
        agent.clear_skill_context = MagicMock()
        agent._active_skill_ids = set()
        agent._last_tools = []
        return agent

    @pytest.fixture
    def engine(self, fake_grounding_agent, tmp_path):
        """Real ExecutionEngine with a fake grounding agent."""
        from openspace.execution_engine import ExecutionEngine

        config = MagicMock()
        config.grounding_max_iterations = 10
        config.workspace = str(tmp_path)

        return ExecutionEngine(
            config=config,
            grounding_agent=fake_grounding_agent,
            grounding_client=None,
        )

    def test_task_returns_structured_result(self, engine, fake_grounding_agent):
        """OUTCOME: User submits a task and gets a response with expected fields.

        NOTE: The grounding agent is mocked — this validates the ExecutionEngine
        orchestration shell (busy-wait, task ID, recording, workspace), not the
        LLM reasoning path.  LLM-level E2E requires a live model.
        """
        result = _run(engine.execute("What is the weather in Seattle?"))

        assert isinstance(result, dict), "Result must be a dict"
        assert "status" in result, "Result must include status"
        assert result["status"] == "completed"
        assert "response" in result

    def test_tasks_get_unique_ids_and_count_increments(self, engine, fake_grounding_agent):
        """OUTCOME: Each task gets a unique identifier and the system tracks count.

        Verifies engine-generated IDs (not caller-supplied) are unique by
        inspecting the execution_context passed to the grounding agent.
        """
        assert engine.execution_count == 0

        # Let engine auto-generate task IDs (don't supply them)
        _run(engine.execute("Task 1"))
        assert engine.execution_count == 1
        ctx1 = fake_grounding_agent.process.call_args_list[0][0][0]

        _run(engine.execute("Task 2"))
        assert engine.execution_count == 2
        ctx2 = fake_grounding_agent.process.call_args_list[1][0][0]

        # Verify the engine assigned unique, non-empty task IDs
        id1 = ctx1.get("task_id", "")
        id2 = ctx2.get("task_id", "")
        assert id1, "Engine must assign a task_id to execution context"
        assert id2, "Engine must assign a task_id to execution context"
        assert id1 != id2, f"Engine-generated task IDs must be unique: {id1!r} == {id2!r}"

    def test_no_grounding_agent_raises(self, tmp_path):
        """OUTCOME: System clearly tells user it's not ready if not initialized."""
        from openspace.execution_engine import ExecutionEngine

        config = MagicMock()
        config.grounding_max_iterations = 10

        engine = ExecutionEngine(
            config=config,
            grounding_agent=None,
            grounding_client=None,
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            _run(engine.execute("This should fail"))


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 6: MCP Server — "Does the agent integration actually work?"
# ═══════════════════════════════════════════════════════════════════════════


class TestMCPServerOutcome:
    """User expectation: Connect my agent to OpenSpace via MCP and the
    expected tools are available."""

    def test_mcp_app_creates_successfully(self):
        """OUTCOME: MCP server can be instantiated without errors."""
        from openspace.mcp.server import create_mcp_app

        app = create_mcp_app()
        assert app is not None
        assert app.name == "OpenSpace"

    def test_mcp_has_expected_tools(self):
        """OUTCOME: All documented tools are registered and discoverable."""
        from openspace.mcp.server import create_mcp_app

        app = create_mcp_app()

        # The tools that users expect based on SDK docs
        expected_tools = {
            "execute_task",
            "search_skills",
            "fix_skill",
            "upload_skill",
            "health_check",
            "get_metrics",
            "get_execution_traces",
            "check_slos",
        }

        # Use FastMCP's public list_tools() API
        # NOTE: If FastMCP changes this API, we WANT the test to break — it means
        # the tool discovery contract changed.
        assert hasattr(app, "list_tools"), (
            "FastMCP must expose list_tools() — tool discovery contract broken"
        )
        tools = _run(app.list_tools())
        registered = {t.name for t in tools}

        missing = expected_tools - registered
        assert not missing, (
            f"Missing MCP tools: {missing}. Registered: {registered}"
        )

    def test_multiple_mcp_apps_are_independent(self):
        """OUTCOME: Creating multiple instances doesn't cause cross-contamination."""
        from openspace.mcp.server import create_mcp_app

        app1 = create_mcp_app()
        app2 = create_mcp_app()

        assert app1 is not app2, "Each call should produce an independent instance"

        # Verify both have the same tools registered (independent copies)
        tools1 = {t.name for t in _run(app1.list_tools())}
        tools2 = {t.name for t in _run(app2.list_tools())}
        assert tools1 == tools2, "Independent apps should have identical tool sets"


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 7: Evolution Pipeline — "Can broken skills be fixed?"
# ═══════════════════════════════════════════════════════════════════════════


class TestEvolutionPipelineOutcome:
    """User expectation: A degraded skill can be evolved into a better
    version, with full lineage tracking and security review."""

    def test_evolution_with_review_gate_pass(self, in_memory_store):
        """OUTCOME: A skill evolves, passes review, and the new version
        is stored with correct lineage."""
        from openspace.skill_engine.review_gate import ReviewGate

        # Save the parent skill
        parent = _make_skill_record(name="data_fetcher", tags=["data"])
        _run(in_memory_store.save_record(parent))

        # Simulate evolution — create an improved child
        child = _make_skill_record(
            name="data_fetcher",
            origin="fixed",
            generation=1,
            parent_skill_ids=[parent.skill_id],
            content_snapshot={
                "SKILL.md": "# data_fetcher\n\nFetches data with retry.\n\n## Steps\n\n1. Fetch with retry.\n",
                "helper.py": "import time\ndef fetch_with_retry():\n    return 'data'\n",
            },
            tags=["data"],
        )

        # Review gate check
        gate = ReviewGate()
        result = gate.review(child)
        assert result.passed, (
            f"Evolved skill should pass review: "
            f"{[(c.name, c.verdict, c.detail) for c in result.checks]}"
        )

        # Persist the evolution
        _run(in_memory_store.evolve_skill(child, [parent.skill_id]))

        # Verify the outcome
        loaded_child = in_memory_store.load_record(child.skill_id)
        assert loaded_child.is_active is True
        assert loaded_child.lineage.origin.value == "fixed"
        assert loaded_child.lineage.generation == 1
        assert loaded_child.lineage.parent_skill_ids == [parent.skill_id]

        loaded_parent = in_memory_store.load_record(parent.skill_id)
        assert loaded_parent.is_active is False

    def test_evolution_with_review_gate_block(self, in_memory_store):
        """OUTCOME: If an evolved skill contains dangerous code, it's blocked
        and the original skill remains active."""
        from openspace.skill_engine.review_gate import ReviewGate

        parent = _make_skill_record(name="safe_processor")
        _run(in_memory_store.save_record(parent))

        # Create a "poisoned" evolution
        poisoned = _make_skill_record(
            name="safe_processor",
            origin="fixed",
            generation=1,
            parent_skill_ids=[parent.skill_id],
            content_snapshot={
                "SKILL.md": "# safe_processor\n\nProcesses data.\n",
                "exploit.py": "import subprocess\nsubprocess.call(['id'])",  # nosec B603 — test data
            },
        )

        gate = ReviewGate()
        result = gate.review(poisoned)
        assert not result.passed, "Poisoned evolution MUST be blocked"

        # Gate blocked it → do NOT save. Original must still be the only active record.
        all_records = in_memory_store.load_all()
        active = [r for r in all_records.values() if r.is_active and r.name == "safe_processor"]
        assert len(active) == 1, "Only original skill should exist"
        assert active[0].skill_id == parent.skill_id

    def test_evolution_quarantine_after_save(self, in_memory_store):
        """OUTCOME: Even if a bad skill was persisted, quarantine deactivates it."""
        from openspace.skill_engine.review_gate import ReviewResult, quarantine_skill, CheckResult

        record = _make_skill_record(name="bad_skill")
        _run(in_memory_store.save_record(record))

        failed_result = ReviewResult(
            verdict="fail",
            checks=[CheckResult(name="ast-safety", verdict="fail", detail="dangerous patterns")],
        )

        quarantined = _run(quarantine_skill(in_memory_store, record.skill_id, failed_result))
        assert quarantined is True

        loaded = in_memory_store.load_record(record.skill_id)
        assert loaded.is_active is False, "Quarantined skill must be deactivated"

    def test_derived_skill_multi_parent_evolution(self, in_memory_store):
        """OUTCOME: A DERIVED skill merges two parents and both remain active."""
        from openspace.skill_engine.review_gate import ReviewGate

        parent_a = _make_skill_record(name="weather_skill", tags=["weather"])
        parent_b = _make_skill_record(name="geocoding_skill", tags=["geo"])
        _run(in_memory_store.save_record(parent_a))
        _run(in_memory_store.save_record(parent_b))

        derived = _make_skill_record(
            name="location_forecast",
            origin="derived",
            generation=1,
            parent_skill_ids=[parent_a.skill_id, parent_b.skill_id],
            content_snapshot={
                "SKILL.md": "# location_forecast\n\nCombines weather + geo.\n\n## Steps\n\n1. Geocode. 2. Forecast.\n",
                "main.py": "def forecast(city):\n    return f'Sunny in {city}'\n",
            },
            tags=["weather", "geo"],
        )

        gate = ReviewGate()
        result = gate.review(derived)
        assert result.passed, f"Derived skill should pass: {[(c.name, c.detail) for c in result.checks]}"

        _run(in_memory_store.evolve_skill(derived, [parent_a.skill_id, parent_b.skill_id]))

        # For DERIVED: parents remain active, child is also active
        loaded_a = in_memory_store.load_record(parent_a.skill_id)
        loaded_b = in_memory_store.load_record(parent_b.skill_id)
        loaded_d = in_memory_store.load_record(derived.skill_id)

        assert loaded_a.is_active is True, "DERIVED parent A must stay active"
        assert loaded_b.is_active is True, "DERIVED parent B must stay active"
        assert loaded_d.is_active is True, "Derived child must be active"
        assert loaded_d.lineage.generation == 1

    def test_lineage_validation_catches_invalid_evolution(self):
        """OUTCOME: System rejects evolution attempts with broken lineage."""
        from openspace.skill_engine.types import SkillLineage, SkillOrigin, ValidationError

        # FIXED must have exactly 1 parent
        bad_lineage = SkillLineage(
            origin=SkillOrigin.FIXED,
            generation=1,
            parent_skill_ids=[],  # Missing parent!
        )
        with pytest.raises(ValidationError, match="exactly 1 parent"):
            bad_lineage.validate()

        # DERIVED must have at least 1 parent
        bad_derived = SkillLineage(
            origin=SkillOrigin.DERIVED,
            generation=1,
            parent_skill_ids=[],  # Missing parent!
        )
        with pytest.raises(ValidationError, match="at least 1 parent"):
            bad_derived.validate()

        # IMPORTED must have no parents
        bad_imported = SkillLineage(
            origin=SkillOrigin.IMPORTED,
            generation=0,
            parent_skill_ids=["some_parent"],  # Shouldn't have one!
        )
        with pytest.raises(ValidationError, match="no parents"):
            bad_imported.validate()


# ═══════════════════════════════════════════════════════════════════════════
#  E2E Outcome 8: Command Execution — "Can the server run commands?"
# ═══════════════════════════════════════════════════════════════════════════


class TestCommandExecutionOutcome:
    """User expectation: POST /execute runs a command and returns the output."""

    def test_execute_returns_output(self):
        """OUTCOME: User sends a command and gets stdout back."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.post(
            "/execute",
            json={"command": "echo hello", "shell": True, "timeout": 10},
        )
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "success"
        assert "hello" in data["output"]

    def test_execute_captures_returncode(self):
        """OUTCOME: User can check if the command succeeded or failed."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.post(
            "/execute",
            json={"command": "echo success", "shell": True, "timeout": 10},
        )
        data = resp.get_json()

        assert data["returncode"] == 0, "Successful command must return exit code 0"

    def test_execute_reports_errors(self):
        """OUTCOME: Failed commands report stderr and non-zero exit code."""
        from openspace.local_server.main import app

        client = app.test_client()
        if _platform_mod.system() == "Windows":
            cmd = "cmd /c exit 1"
        else:
            cmd = "false"

        resp = client.post(
            "/execute",
            json={"command": cmd, "shell": True, "timeout": 10},
        )
        data = resp.get_json()

        assert resp.status_code == 200  # HTTP 200 even for failed commands
        assert data["returncode"] != 0, "Failed command must have non-zero exit code"
        assert "error" in data or "stderr" in data, (
            "Response must include an error/stderr field for failed commands"
        )

    def test_execute_timeout_enforcement(self):
        """OUTCOME: A long-running command is killed after the timeout."""
        from openspace.local_server.main import app

        client = app.test_client()
        if _platform_mod.system() == "Windows":
            cmd = "ping -n 30 127.0.0.1"
        else:
            cmd = "sleep 30"

        resp = client.post(
            "/execute",
            json={"command": cmd, "shell": True, "timeout": 1},
        )
        data = resp.get_json()

        # Should return 408 (timeout) or 200 with error status
        assert resp.status_code in (200, 408), f"Unexpected status: {resp.status_code}"
        if resp.status_code == 408:
            assert "timeout" in data.get("message", "").lower()
        else:
            assert data.get("status") == "error"

    def test_local_server_binds_localhost_only(self):
        """OUTCOME: The desktop server defaults to 127.0.0.1 — not exposed to network.

        This is a security-by-design decision: the Flask desktop server runs
        shell commands and does NOT have auth middleware.  It must only bind
        to localhost.
        """
        from openspace.local_server import main as local_main

        # Verify run_server defaults to 127.0.0.1
        import inspect
        sig = inspect.signature(local_main.run_server)
        host_default = sig.parameters["host"].default
        assert host_default == "127.0.0.1", (
            f"Local server must default to 127.0.0.1, got {host_default}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Cross-cutting: Version Consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionConsistencyOutcome:
    """User expectation: The version is consistent everywhere — no stale
    strings leaking through different entry points."""

    def test_package_version(self):
        """OUTCOME: Python package reports correct version."""
        assert EXPECTED_VERSION == "2.0.0", (
            f"Canonical version from openspace.__version__ is {EXPECTED_VERSION}"
        )

    def test_mcp_server_version(self):
        """OUTCOME: MCP server advertises correct version."""
        from openspace.mcp.server import _VERSION
        assert _VERSION == EXPECTED_VERSION

    def test_local_server_health_version(self):
        """OUTCOME: Local HTTP server health endpoint reports correct version."""
        from openspace.local_server.main import app

        client = app.test_client()
        resp = client.get("/")
        data = resp.get_json()
        assert data["version"] == EXPECTED_VERSION

    def test_pyproject_version(self):
        """OUTCOME: pyproject.toml (packaging source of truth) matches."""
        import tomllib
        from openspace.config.constants import PROJECT_ROOT

        pyproject = PROJECT_ROOT / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            pkg_version = data.get("project", {}).get("version", "")
            assert pkg_version == EXPECTED_VERSION, (
                f"pyproject.toml version={pkg_version}, expected {EXPECTED_VERSION}"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Cross-cutting: Execution Analysis Persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestExecutionAnalysisOutcome:
    """User expectation: After a task runs, I can see what happened —
    which skills were used, whether they worked, and what to evolve."""

    def test_record_and_retrieve_analysis(self, in_memory_store):
        """OUTCOME: An execution analysis is persisted and retrievable."""
        from openspace.skill_engine.types import ExecutionAnalysis, SkillJudgment

        # Save a skill first
        record = _make_skill_record(name="analyzer_test")
        _run(in_memory_store.save_record(record))

        # Record an analysis
        analysis = ExecutionAnalysis(
            task_id="task_e2e_001",
            timestamp=datetime.now(),
            task_completed=True,
            execution_note="Task completed successfully using analyzer_test skill",
            skill_judgments=[
                SkillJudgment(
                    skill_id=record.skill_id,
                    skill_applied=True,
                    note="Skill was applied correctly",
                )
            ],
        )
        _run(in_memory_store.record_analysis(analysis))

        # Verify counters were updated (exactly 1 analysis recorded)
        loaded = in_memory_store.load_record(record.skill_id)
        assert loaded.total_selections == 1, "Selection counter must be exactly 1"
        assert loaded.total_applied == 1, "Applied counter must be exactly 1"
        assert loaded.total_completions == 1, "Completion counter must be exactly 1"

    def test_analysis_with_failed_skill(self, in_memory_store):
        """OUTCOME: When a skill fails, the system records the failure for learning."""
        from openspace.skill_engine.types import ExecutionAnalysis, SkillJudgment

        record = _make_skill_record(name="flaky_skill")
        _run(in_memory_store.save_record(record))

        analysis = ExecutionAnalysis(
            task_id="task_e2e_002",
            timestamp=datetime.now(),
            task_completed=False,
            execution_note="Task failed — skill did not apply correctly",
            skill_judgments=[
                SkillJudgment(
                    skill_id=record.skill_id,
                    skill_applied=False,
                    note="Skill instructions were outdated",
                )
            ],
        )
        _run(in_memory_store.record_analysis(analysis))

        loaded = in_memory_store.load_record(record.skill_id)
        assert loaded.total_selections == 1
        assert loaded.total_applied == 0, "Skill was NOT applied"
        assert loaded.total_fallbacks == 1, "Fallback counter must be exactly 1"
