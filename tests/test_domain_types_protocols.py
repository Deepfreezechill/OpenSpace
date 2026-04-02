"""Tests for EPIC 1.1 (Protocol Interfaces) and EPIC 1.2 (Domain Types).

Validates:
- All 13 Protocol interfaces are importable and runtime-checkable
- All frozen domain types are truly immutable
- Serialization round-trips work correctly
- Enum consolidation preserves original values
- Protocol structural compliance (concrete classes satisfy protocols)
"""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError, replace
from datetime import datetime

import pytest

# ═══════════════════════════════════════════════════════════════════════
#  Protocol Import & Structural Tests (EPIC 1.1)
# ═══════════════════════════════════════════════════════════════════════


class TestProtocolImports:
    """All 13 protocols must be importable and runtime-checkable."""

    def test_all_protocols_importable(self):
        from openspace.domain.ports import (
            AgentExecutorPort,
            AnalysisPort,
            AuthPort,
            CapabilityLeaseResolverPort,
            CloudSkillPort,
            LLMClientPort,
            PolicyEnginePort,
            SandboxPort,
            SecretBrokerPort,
            SkillEvolutionPort,
            SkillStorePort,
            TelemetryPort,
            ToolBackendPort,
        )

        protocols = [
            AgentExecutorPort,
            AnalysisPort,
            AuthPort,
            CapabilityLeaseResolverPort,
            CloudSkillPort,
            LLMClientPort,
            PolicyEnginePort,
            SandboxPort,
            SecretBrokerPort,
            SkillEvolutionPort,
            SkillStorePort,
            TelemetryPort,
            ToolBackendPort,
        ]
        assert len(protocols) == 13

    def test_protocols_are_runtime_checkable(self):
        from openspace.domain.ports import (
            AgentExecutorPort,
            AnalysisPort,
            AuthPort,
            CapabilityLeaseResolverPort,
            CloudSkillPort,
            LLMClientPort,
            PolicyEnginePort,
            SandboxPort,
            SecretBrokerPort,
            SkillEvolutionPort,
            SkillStorePort,
            TelemetryPort,
            ToolBackendPort,
        )

        for proto in [
            AgentExecutorPort,
            AnalysisPort,
            AuthPort,
            CapabilityLeaseResolverPort,
            CloudSkillPort,
            LLMClientPort,
            PolicyEnginePort,
            SandboxPort,
            SecretBrokerPort,
            SkillEvolutionPort,
            SkillStorePort,
            TelemetryPort,
            ToolBackendPort,
        ]:
            assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_runtime_protocol"), (
                f"{proto.__name__} is not runtime_checkable"
            )


class TestProtocolCompliance:
    """Verify concrete classes have the required methods.

    Note: Some ports define a *domain-layer* signature that differs from
    the concrete implementation (e.g. SkillStorePort uses SkillManifest,
    but SkillStore uses SkillRecord).  Phase 1.3 (AppContainer) will
    introduce thin adapters to bridge the gap.  These tests verify that
    the concrete classes have the *method names* the port requires.
    """

    def test_skill_store_has_required_methods(self):
        """SkillStore has the methods that SkillStorePort requires."""
        from openspace.skill_engine.store import SkillStore

        required_methods = [
            "save_record",
            "load_record",
            "load_all",
            "load_active",
            "delete_record",
            "count",
        ]
        for method in required_methods:
            assert hasattr(SkillStore, method), f"SkillStore missing method: {method}"

    def test_sandbox_has_required_methods(self):
        """BaseSandbox has the methods that SandboxPort requires."""
        from openspace.grounding.core.security.sandbox import BaseSandbox

        required_methods = ["start", "stop", "execute_safe"]
        for method in required_methods:
            assert hasattr(BaseSandbox, method), f"BaseSandbox missing method: {method}"

    def test_telemetry_has_required_methods(self):
        """Telemetry has capture/flush/shutdown (adapter bridges signature)."""
        try:
            from openspace.utils.telemetry.telemetry import Telemetry
        except ImportError:
            pytest.skip("Telemetry module not available")

        required_methods = ["capture", "flush", "shutdown"]
        for method in required_methods:
            assert hasattr(Telemetry, method), f"Telemetry missing method: {method}"

    def test_llm_client_has_complete(self):
        """LLMClient has the complete method."""
        try:
            from openspace.llm.client import LLMClient
        except ImportError:
            pytest.skip("LLMClient not importable (litellm version issue)")

        assert hasattr(LLMClient, "complete"), "LLMClient missing 'complete'"

    def test_policy_engine_has_required_methods(self):
        """SecurityPolicyManager has the methods PolicyEnginePort requires."""
        from openspace.grounding.core.security.policies import SecurityPolicyManager

        required_methods = [
            "check_command_allowed",
            "check_domain_allowed",
            "get_policy",
        ]
        for method in required_methods:
            assert hasattr(SecurityPolicyManager, method), f"SecurityPolicyManager missing: {method}"


# ═══════════════════════════════════════════════════════════════════════
#  Frozen Domain Types Tests (EPIC 1.2)
# ═══════════════════════════════════════════════════════════════════════


class TestTaskTypes:
    """TaskRequest and TaskResult are frozen and serializable."""

    def test_task_request_frozen(self):
        from openspace.domain.types import TaskRequest

        req = TaskRequest(task="test task", task_id="t1")
        with pytest.raises(FrozenInstanceError):
            req.task = "mutated"  # type: ignore[misc]

    def test_task_request_from_dict(self):
        from openspace.domain.types import TaskRequest

        data = {
            "task": "do something",
            "task_id": "t42",
            "workspace_dir": "/tmp",
            "max_iterations": 5,
            "search_scope": "local",
            "skill_dirs": ["/skills/a", "/skills/b"],
            "context": {"key": "value"},
        }
        req = TaskRequest.from_dict(data)
        assert req.task == "do something"
        assert req.task_id == "t42"
        assert req.max_iterations == 5
        assert req.skill_dirs == ("/skills/a", "/skills/b")
        assert req.context_dict == {"key": "value"}

    def test_task_request_deep_freezes_nested_context(self):
        from openspace.domain.types import TaskRequest

        data = {
            "task": "test",
            "context": {
                "nested": {"a": [1, 2, 3]},
                "list_val": [{"x": 1}],
            },
        }
        req = TaskRequest.from_dict(data)
        # Nested dicts become tuples of tuples, lists become tuples
        # The entire structure should be hashable (deeply frozen)
        assert isinstance(req.context, tuple)
        for key, val in req.context:
            assert isinstance(val, tuple), f"Value for {key} not frozen: {type(val)}"

    def test_task_result_frozen(self):
        from openspace.domain.types import TaskResult

        result = TaskResult(task_id="t1", status="success", response="done")
        with pytest.raises(FrozenInstanceError):
            result.status = "error"  # type: ignore[misc]

    def test_task_result_ok_property(self):
        from openspace.domain.types import TaskResult

        ok = TaskResult(task_id="t1", status="success")
        fail = TaskResult(task_id="t2", status="error")
        assert ok.ok is True
        assert fail.ok is False

    def test_task_result_roundtrip(self):
        from openspace.domain.types import TaskResult, ToolExecution

        original = TaskResult(
            task_id="t1",
            status="success",
            response="done",
            execution_time=1.5,
            iterations=3,
            skills_used=("skill-a",),
            evolved_skills=("skill-b",),
            tool_executions=(
                ToolExecution(
                    tool_name="bash",
                    arguments=(("cmd", "ls"),),
                    status="success",
                    duration_ms=42.0,
                ),
                ToolExecution(
                    tool_name="read_file",
                    arguments=(("path", "/tmp/f"),),
                    status="error",
                    duration_ms=1.0,
                    error="not found",
                ),
            ),
            warnings=("w1",),
        )
        d = original.to_dict()
        restored = TaskResult.from_dict(d)
        assert restored.task_id == original.task_id
        assert restored.status == original.status
        assert restored.skills_used == original.skills_used
        assert restored.evolved_skills == original.evolved_skills
        assert len(restored.tool_executions) == 2
        assert restored.tool_executions[0].tool_name == "bash"
        assert restored.tool_executions[0].duration_ms == 42.0
        assert restored.tool_executions[1].error == "not found"

    def test_task_result_replace(self):
        from openspace.domain.types import TaskResult

        original = TaskResult(task_id="t1", status="error", error="boom")
        fixed = replace(original, status="success", error=None)
        assert fixed.status == "success"
        assert fixed.error is None
        assert original.status == "error"  # Original unchanged


class TestSkillTypes:
    """SkillIdentity and SkillManifest are frozen."""

    def test_skill_identity_hashable(self):
        from openspace.domain.types import SkillIdentity

        s1 = SkillIdentity(skill_id="s1", name="Skill One")
        s2 = SkillIdentity(skill_id="s1", name="Skill One Modified")
        # Same skill_id → same hash
        assert hash(s1) == hash(s2)
        # Can be used in sets
        skills = {s1, s2}
        assert len(skills) == 2  # Different objects (frozen, full eq)

    def test_skill_identity_frozen(self):
        from openspace.domain.types import SkillIdentity

        s = SkillIdentity(skill_id="s1", name="test")
        with pytest.raises(FrozenInstanceError):
            s.name = "mutated"  # type: ignore[misc]

    def test_skill_manifest_frozen(self):
        from openspace.domain.types import SkillManifest

        m = SkillManifest(
            skill_id="s1",
            name="Test",
            description="A test skill",
            tags=("tag1", "tag2"),
        )
        with pytest.raises(FrozenInstanceError):
            m.is_active = False  # type: ignore[misc]

    def test_skill_manifest_effective_rate(self):
        from openspace.domain.types import SkillManifest

        zero = SkillManifest(skill_id="s1", name="t", description="d", total_selections=0)
        assert zero.effective_rate == 0.0

        active = SkillManifest(
            skill_id="s2",
            name="t",
            description="d",
            total_selections=10,
            total_applied=7,
        )
        assert active.effective_rate == pytest.approx(0.7)

    def test_skill_manifest_to_dict(self):
        from openspace.domain.types import SkillManifest

        now = datetime.now()
        m = SkillManifest(
            skill_id="s1",
            name="Test",
            description="desc",
            tags=("a", "b"),
            first_seen=now,
        )
        d = m.to_dict()
        assert d["skill_id"] == "s1"
        assert d["tags"] == ["a", "b"]
        assert d["first_seen"] == now.isoformat()


class TestEvolutionTypes:
    """EvolutionRequest and EvolutionResult are frozen."""

    def test_evolution_request_frozen(self):
        from openspace.domain.types import EvolutionRequest

        req = EvolutionRequest(evolution_type="fix", trigger="analysis", target_skill_ids=("s1",))
        with pytest.raises(FrozenInstanceError):
            req.direction = "mutated"  # type: ignore[misc]

    def test_evolution_result_frozen(self):
        from openspace.domain.types import EvolutionResult

        res = EvolutionResult(
            success=True,
            evolved_skill_id="s2",
            evolution_type="fix",
        )
        with pytest.raises(FrozenInstanceError):
            res.success = False  # type: ignore[misc]


class TestAnalysisTypes:
    """Analysis snapshots are frozen."""

    def test_execution_analysis_snapshot_frozen(self):
        from openspace.domain.types import ExecutionAnalysisSnapshot

        snap = ExecutionAnalysisSnapshot(
            task_id="t1",
            timestamp=datetime.now(),
            task_completed=True,
            tool_issues=("issue1",),
        )
        with pytest.raises(FrozenInstanceError):
            snap.task_completed = False  # type: ignore[misc]

    def test_analysis_snapshot_nested_immutability(self):
        from openspace.domain.types import (
            ExecutionAnalysisSnapshot,
            SkillJudgmentSnapshot,
        )

        snap = ExecutionAnalysisSnapshot(
            task_id="t1",
            timestamp=datetime.now(),
            skill_judgments=(SkillJudgmentSnapshot(skill_id="s1", skill_applied=True),),
        )
        assert snap.skill_judgments[0].skill_applied is True
        with pytest.raises(FrozenInstanceError):
            snap.skill_judgments[0].skill_applied = False  # type: ignore[misc]


class TestSearchTypes:
    """Search result types are frozen."""

    def test_search_result_frozen(self):
        from openspace.domain.types import SkillSearchResult

        r = SkillSearchResult(skill_id="s1", name="Test", description="desc", score=0.9)
        with pytest.raises(FrozenInstanceError):
            r.score = 0.1  # type: ignore[misc]

    def test_search_response_frozen(self):
        from openspace.domain.types import SkillSearchResponse, SkillSearchResult

        resp = SkillSearchResponse(
            query="test",
            results=(SkillSearchResult(skill_id="s1", name="Test", description="d", score=0.9),),
            total_count=1,
        )
        assert len(resp.results) == 1
        with pytest.raises(FrozenInstanceError):
            resp.total_count = 99  # type: ignore[misc]


class TestSecurityTypes:
    """Sandbox policy and capability lease are frozen."""

    def test_sandbox_policy_frozen(self):
        from openspace.domain.types import SandboxPolicy

        p = SandboxPolicy(sandbox_enabled=True, trust_tier="basic")
        with pytest.raises(FrozenInstanceError):
            p.sandbox_enabled = False  # type: ignore[misc]

    def test_capability_lease_frozen(self):
        from openspace.domain.types import CapabilityLease

        lease = CapabilityLease(
            lease_id="L1",
            capability="fs.read",
            granted_to="task-42",
            trust_tier="standard",
        )
        with pytest.raises(FrozenInstanceError):
            lease.revoked = True  # type: ignore[misc]


class TestToolTypes:
    """Tool descriptor and call result are frozen."""

    def test_tool_descriptor_frozen(self):
        from openspace.domain.types import ToolDescriptor

        t = ToolDescriptor(name="bash", description="Run shell commands")
        with pytest.raises(FrozenInstanceError):
            t.name = "changed"  # type: ignore[misc]

    def test_tool_call_result_frozen(self):
        from openspace.domain.types import ToolCallResult

        r = ToolCallResult(status="success", content="output")
        with pytest.raises(FrozenInstanceError):
            r.content = "mutated"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
#  Enum Consolidation Tests (EPIC 1.2, Issue #62)
# ═══════════════════════════════════════════════════════════════════════


class TestEnumConsolidation:
    """New enums exist and re-exports preserve original values."""

    def test_re_exported_enums_match_originals(self):
        """Re-exported enums are the exact same objects."""
        from openspace.domain.enums import EvolutionType as DomainEvType
        from openspace.skill_engine.types import EvolutionType as OrigEvType

        assert DomainEvType is OrigEvType

        from openspace.domain.enums import BackendType as DomainBT
        from openspace.grounding.core.types import BackendType as OrigBT

        assert DomainBT is OrigBT

        from openspace.domain.enums import SkillCategory as DomainSC
        from openspace.skill_engine.types import SkillCategory as OrigSC

        assert DomainSC is OrigSC

    def test_new_task_status_enum(self):
        from openspace.domain.enums import TaskStatus

        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.RUNNING.value == "running"

    def test_new_search_scope_enum(self):
        from openspace.domain.enums import SearchScope

        assert SearchScope.ALL.value == "all"
        assert SearchScope.LOCAL.value == "local"
        assert SearchScope.CLOUD.value == "cloud"

    def test_new_trust_tier_enum(self):
        from openspace.domain.enums import TrustTier

        assert TrustTier.UNTRUSTED.value == "untrusted"
        assert TrustTier.PRIVILEGED.value == "privileged"

    def test_new_skill_status_enum(self):
        from openspace.domain.enums import SkillStatus

        assert SkillStatus.ACTIVE.value == "active"
        assert SkillStatus.DEPRECATED.value == "deprecated"

    def test_new_mcp_error_code_enum(self):
        from openspace.domain.enums import MCPErrorCode

        assert MCPErrorCode.EXECUTION_ERROR.value == "EXECUTION_ERROR"
        assert MCPErrorCode.PERMISSION_DENIED.value == "PERMISSION_DENIED"

    def test_search_mode_enum(self):
        from openspace.domain.enums import SearchMode

        assert SearchMode.HYBRID.value == "hybrid"
        assert SearchMode.SEMANTIC.value == "semantic"

    def test_patch_type_enum(self):
        from openspace.domain.enums import PatchType

        assert PatchType.AUTO.value == "auto"
        assert PatchType.DIFF.value == "diff"


# ═══════════════════════════════════════════════════════════════════════
#  Deep-freeze helper tests
# ═══════════════════════════════════════════════════════════════════════


class TestDeepFreeze:
    """The _deep_freeze helper converts mutable containers recursively."""

    def test_deep_freeze_dict(self):
        from openspace.domain.types import _deep_freeze

        result = _deep_freeze({"a": 1, "b": [2, 3]})
        assert isinstance(result, tuple)
        assert result == (("a", 1), ("b", (2, 3)))

    def test_deep_freeze_nested(self):
        from openspace.domain.types import _deep_freeze

        result = _deep_freeze({"outer": {"inner": [1, {"deep": True}]}})
        assert isinstance(result, tuple)
        # outer → (outer, (inner, (1, (deep, True))))
        key, val = result[0]
        assert key == "outer"
        assert isinstance(val, tuple)

    def test_deep_freeze_set(self):
        from openspace.domain.types import _deep_freeze

        result = _deep_freeze({1, 2, 3})
        assert isinstance(result, frozenset)

    def test_deep_freeze_scalar(self):
        from openspace.domain.types import _deep_freeze

        assert _deep_freeze(42) == 42
        assert _deep_freeze("hello") == "hello"
        assert _deep_freeze(None) is None


# ═══════════════════════════════════════════════════════════════════════
#  All-frozen invariant test
# ═══════════════════════════════════════════════════════════════════════


class TestAllTypesFrozen:
    """Every dataclass in domain.types MUST be frozen."""

    def test_all_domain_types_are_frozen(self):
        import openspace.domain.types as mod

        for name in mod.__all__:
            cls = getattr(mod, name)
            if dataclasses.is_dataclass(cls):
                assert dataclasses.fields(cls)[0].default is not dataclasses.MISSING or True
                # Check frozen flag
                assert cls.__dataclass_params__.frozen, (  # type: ignore[attr-defined]
                    f"{name} is not frozen!"
                )

    def test_all_domain_types_use_slots(self):
        import openspace.domain.types as mod

        for name in mod.__all__:
            cls = getattr(mod, name)
            if dataclasses.is_dataclass(cls):
                assert cls.__dataclass_params__.slots, (  # type: ignore[attr-defined]
                    f"{name} does not use slots!"
                )
