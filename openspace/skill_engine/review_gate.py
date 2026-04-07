"""Eight-Eyes Review Gate for skill evolution.

Intercepts skill activation after evolution and runs multiple review
checks before allowing the skill to become active. Skills that fail
review are quarantined (deactivated) until manually approved.

Checks:
  1. AST Safety — scans Python files for dangerous patterns
  2. Content    — validates required fields (name, description, SKILL.md)
  3. Lineage    — validates parent chain integrity for evolved skills

Usage::

    from openspace.skill_engine.review_gate import ReviewGate, quarantine_skill

    gate = ReviewGate()
    result = gate.review(evolved_record)
    if not result.passed:
        await quarantine_skill(store, evolved_record.skill_id, result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from openspace.skill_engine.types import SkillOrigin, SkillRecord, ValidationError

logger = logging.getLogger(__name__)

# Executable extensions that must not appear in skill snapshots
_BLOCKED_EXTENSIONS = frozenset({
    ".sh", ".bash", ".bat", ".cmd", ".ps1", ".pyw", ".js", ".rb", ".pl",
})

# Max individual file size (bytes) before AST scan is refused
_MAX_FILE_SIZE = 512 * 1024  # 512 KB


@dataclass
class CheckResult:
    """Result of a single review check."""

    name: str
    verdict: str  # "pass" | "fail"
    detail: str = ""


@dataclass
class ReviewResult:
    """Aggregated result of all review checks."""

    verdict: str  # "pass" | "fail" | "quarantine"
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    @classmethod
    def from_checks(cls, checks: List[CheckResult]) -> ReviewResult:
        """Aggregate individual check results into a final verdict."""
        any_fail = any(c.verdict == "fail" for c in checks)
        return cls(
            verdict="fail" if any_fail else "pass",
            checks=list(checks),
        )


# ======================================================================
# Individual review checks
# ======================================================================


def check_ast_safety(record: SkillRecord) -> CheckResult:
    """Scan Python files in the skill's content snapshot for dangerous patterns.

    Uses the existing AST scanner from openspace.security.
    Fails-closed: if the scanner is unavailable, the check FAILS (not skips).
    Also rejects executable non-Python files (.sh, .bat, .ps1, etc.).
    """
    snapshot = record.lineage.content_snapshot or {}

    # Block executable non-Python files (shell scripts, batch files, etc.)
    blocked = [
        k for k in snapshot
        if any(k.lower().endswith(ext) for ext in _BLOCKED_EXTENSIONS)
    ]
    if blocked:
        return CheckResult(
            name="ast-safety",
            verdict="fail",
            detail=f"Blocked executable files: {', '.join(sorted(blocked))}",
        )

    py_files = {
        k: v for k, v in snapshot.items() if k.lower().endswith(".py")
    }

    if not py_files:
        return CheckResult(name="ast-safety", verdict="pass", detail="no Python files")

    # Fail-closed: if scanner unavailable, refuse the skill
    try:
        from openspace.security import check_code_safety
    except ImportError:
        return CheckResult(
            name="ast-safety",
            verdict="fail",
            detail="AST scanner not available — cannot verify safety",
        )

    violations = []
    for filename, source in py_files.items():
        if len(source) > _MAX_FILE_SIZE:
            violations.append(f"{filename}: file too large ({len(source)} bytes, max {_MAX_FILE_SIZE})")
            continue
        is_safe, findings = check_code_safety(source)
        if not is_safe:
            descs = [f.description for f in findings]
            violations.append(f"{filename}: {'; '.join(descs) or 'unsafe'}")

    if violations:
        return CheckResult(
            name="ast-safety",
            verdict="fail",
            detail=f"Dangerous code detected: {'; '.join(violations)}",
        )
    return CheckResult(name="ast-safety", verdict="pass", detail="all files clean")


def check_content(record: SkillRecord) -> CheckResult:
    """Validate skill has required content fields."""
    issues = []

    if not record.name or not record.name.strip():
        issues.append("skill name is empty")

    if not record.description or not record.description.strip():
        issues.append("skill description is empty")

    snapshot = record.lineage.content_snapshot or {}
    if not snapshot or "SKILL.md" not in snapshot:
        issues.append("SKILL.md missing from content snapshot")

    if issues:
        return CheckResult(
            name="content",
            verdict="fail",
            detail=f"Content validation failed: {'; '.join(issues)}",
        )
    return CheckResult(name="content", verdict="pass", detail="all required fields present")


def check_lineage(record: SkillRecord) -> CheckResult:
    """Validate lineage integrity for evolved skills.

    Delegates to SkillLineage.validate() for per-origin cardinality rules,
    then applies additional ReviewGate-specific checks.
    """
    # Run the canonical validation from SkillLineage
    try:
        record.lineage.validate()
    except ValidationError as exc:
        return CheckResult(
            name="lineage",
            verdict="fail",
            detail=f"Lineage validation failed: {exc}",
        )

    origin = record.lineage.origin

    # Non-evolved skills pass after validate() confirms no parents
    if origin in (SkillOrigin.IMPORTED, SkillOrigin.CAPTURED):
        return CheckResult(
            name="lineage", verdict="pass", detail="non-evolved skill (validated)"
        )

    # Evolved skills must have generation > 0
    if record.lineage.generation < 1:
        return CheckResult(
            name="lineage",
            verdict="fail",
            detail=f"Evolved skill ({origin.value}) has generation={record.lineage.generation}, expected >= 1",
        )

    return CheckResult(
        name="lineage",
        verdict="pass",
        detail=f"valid {origin.value} chain (gen={record.lineage.generation})",
    )


# ======================================================================
# ReviewGate — orchestrates all checks
# ======================================================================


class ReviewGate:
    """Runs multi-check review on evolved skills before activation.

    Each check is independent — the gate runs ALL checks even if earlier
    ones fail, so the developer gets a complete picture of issues.
    """

    _DEFAULT_CHECKS = [check_ast_safety, check_content, check_lineage]

    def __init__(self, checks: Optional[List[Callable]] = None) -> None:
        if checks is not None and len(checks) == 0:
            raise ValueError("ReviewGate requires at least one check — empty list not allowed")
        self._checks = checks if checks is not None else list(self._DEFAULT_CHECKS)

    def review(self, record: SkillRecord) -> ReviewResult:
        """Run all review checks on a skill record.

        Returns a ReviewResult with the aggregate verdict and per-check details.
        Always runs ALL checks — does not short-circuit on first failure.
        """
        results = []
        for check_fn in self._checks:
            try:
                result = check_fn(record)
                results.append(result)
            except Exception as exc:
                results.append(
                    CheckResult(
                        name=getattr(check_fn, "__name__", "unknown"),
                        verdict="fail",
                        detail=f"Check raised exception: {type(exc).__name__}",
                    )
                )

        return ReviewResult.from_checks(results)


# ======================================================================
# Quarantine — deactivate skills that fail review
# ======================================================================


async def quarantine_skill(
    store,
    skill_id: str,
    review_result: ReviewResult,
) -> bool:
    """Quarantine a skill that failed review by deactivating it.

    Args:
        store: SkillStore instance with deactivate_record()
        skill_id: ID of the skill to quarantine
        review_result: The review result (quarantine only if failed)

    Returns:
        True if skill was quarantined, False if review passed or quarantine failed.
    """
    if review_result.passed:
        return False

    failed_checks = [c for c in review_result.checks if c.verdict == "fail"]
    check_names = ", ".join(c.name for c in failed_checks)

    logger.warning(
        "Quarantining skill %s — failed checks: %s",
        skill_id,
        check_names,
    )
    for check in failed_checks:
        logger.warning("  [%s] %s", check.name, check.detail)

    try:
        success = await store.deactivate_record(skill_id)
        if not success:
            logger.error(
                "Quarantine FAILED for %s — deactivate_record returned False",
                skill_id,
            )
            return False
        return True
    except Exception as exc:
        logger.error("Quarantine FAILED for %s: %s", skill_id, type(exc).__name__)
        return False
