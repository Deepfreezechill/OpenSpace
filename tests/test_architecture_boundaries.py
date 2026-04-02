"""Architecture boundary tests — EPIC 1.7.

Issues:
- #80: Import-graph test: domain layer imports ZERO infrastructure modules
- #81: MCP handlers access ZERO private fields (_store, _registry, etc.)
- #82: File-size guard: no .py file exceeds 15 KB (warning-only in Phase 1)
- #83: CI integration (tests run in pytest → already in CI Tier 1)

Phase 1 approach:
- Import violations in domain/ are **hard failures** (ZERO tolerance)
- Private-field access in MCP handlers is a **hard failure**
- File-size violations emit ``pytest.warnings`` but do NOT fail
  (enforcement deferred to Phase 7)
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OPENSPACE = _REPO_ROOT / "openspace"
_DOMAIN = _OPENSPACE / "domain"

# stdlib top-level modules that are always allowed everywhere
_STDLIB_PREFIXES: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "ast",
        "asyncio",
        "base64",
        "collections",
        "contextlib",
        "copy",
        "contextvars",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "hashlib",
        "hmac",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "operator",
        "os",
        "pathlib",
        "pprint",
        "re",
        "secrets",
        "shutil",
        "signal",
        "socket",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "types",
        "typing",
        "typing_extensions",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
    }
)

# Third-party packages explicitly allowed in domain layer
_DOMAIN_ALLOWED_THIRD_PARTY: frozenset[str] = frozenset(
    {
        "structlog",  # structured logging (EPIC 1.6)
    }
)

# All allowed import roots for domain layer
_DOMAIN_ALLOWED_ROOTS: frozenset[str] = _STDLIB_PREFIXES | _DOMAIN_ALLOWED_THIRD_PARTY | frozenset({"openspace.domain"})


def _collect_py_files(directory: Path) -> list[Path]:
    """Recursively collect all .py files under *directory*."""
    return sorted(directory.rglob("*.py"))


def _extract_imports(filepath: Path) -> list[str]:
    """Return top-level module names imported by *filepath* using AST."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        # Fail loudly — a syntax error in domain/ means the file can't be
        # validated and should not silently pass boundary checks.
        raise AssertionError(f"SyntaxError in {filepath} — cannot validate imports: {exc}") from exc

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) are intra-package — always allowed
            if node.level and node.level > 0:
                continue
            if node.module:
                modules.append(node.module)
    return modules


def _import_root(module: str) -> str:
    """Return the top-level package/module name."""
    return module.split(".")[0]


def _is_allowed_domain_import(module: str) -> bool:
    """Check if *module* is allowed inside the domain layer."""
    root = _import_root(module)
    # stdlib or allowed third-party
    if root in _DOMAIN_ALLOWED_ROOTS:
        return True
    # intra-domain (openspace.domain.*)
    if module.startswith("openspace.domain"):
        return True
    # relative imports within domain resolve to openspace.domain
    # (already handled by ast — from .ports import X becomes module="ports")
    # Single-segment names matching stdlib
    if root in _STDLIB_PREFIXES:
        return True
    return False


# ---------------------------------------------------------------------------
# #80 — Domain Import Purity
# ---------------------------------------------------------------------------


class TestDomainImportPurity:
    """Domain layer (openspace/domain/) must import ZERO infrastructure modules.

    Allowed: stdlib, structlog, openspace.domain.*
    Forbidden: anything else in openspace.* or third-party infra packages.
    """

    def test_known_cross_layer_count(self) -> None:
        """Guard: the known-violations allowlist must not silently grow."""
        assert len(self._KNOWN_CROSS_LAYER) == 3, (
            f"If you fixed a cross-layer import, remove it from _KNOWN_CROSS_LAYER. "
            f"If you added one, stop — fix it instead. Count: {len(self._KNOWN_CROSS_LAYER)}"
        )

    @staticmethod
    def _domain_files() -> list[Path]:
        return _collect_py_files(_DOMAIN)

    def test_domain_files_exist(self) -> None:
        """Sanity: domain directory has Python files to test."""
        files = self._domain_files()
        assert len(files) >= 3, f"Expected ≥3 domain files, got {len(files)}"

    # Known Phase 1 cross-layer imports (tech debt — to be fixed in later phases).
    # The test ensures NO NEW violations are introduced.
    _KNOWN_CROSS_LAYER: frozenset[tuple[str, str]] = frozenset(
        {
            ("openspace/domain/enums.py", "openspace.skill_engine.types"),
            ("openspace/domain/enums.py", "openspace.grounding.core.types"),
            ("openspace/domain/enums.py", "openspace.grounding.core.exceptions"),
        }
    )

    def test_domain_imports_no_infrastructure(self) -> None:
        """Every import in openspace/domain/ must be stdlib, structlog, or domain-local."""
        violations: list[str] = []

        for filepath in self._domain_files():
            rel = filepath.relative_to(_REPO_ROOT)
            rel_posix = rel.as_posix()
            for module in _extract_imports(filepath):
                if not _is_allowed_domain_import(module):
                    if (rel_posix, module) not in self._KNOWN_CROSS_LAYER:
                        violations.append(f"  {rel}: imports '{module}'")

        if violations:
            detail = "\n".join(violations)
            pytest.fail(f"Domain layer has {len(violations)} NEW forbidden import(s):\n{detail}")

    def test_domain_does_not_import_openspace_infra(self) -> None:
        """Domain must not import from openspace.* outside openspace.domain."""
        violations: list[str] = []

        for filepath in self._domain_files():
            rel = filepath.relative_to(_REPO_ROOT)
            rel_posix = rel.as_posix()
            for module in _extract_imports(filepath):
                if module.startswith("openspace.") and not module.startswith("openspace.domain"):
                    if (rel_posix, module) not in self._KNOWN_CROSS_LAYER:
                        violations.append(f"  {rel}: imports '{module}'")

        if violations:
            detail = "\n".join(violations)
            pytest.fail(f"Domain layer has {len(violations)} cross-layer import(s):\n{detail}")


# ---------------------------------------------------------------------------
# #81 — MCP Handler Private-Field Guard
# ---------------------------------------------------------------------------

# Private fields that belong to OpenSpace internals
_PRIVATE_FIELD_PATTERNS: frozenset[str] = frozenset(
    {
        "_store",
        "_registry",
        "_config",
        "_llm",
        "_llm_client",
        "_evolver",
        "_grounding_client",
        "_grounding_config",
        "_skill_registry",
        "_skill_store",
        "_skill_evolver",
        "_telemetry",
        "_auth_provider",
        "_sandbox",
        "_container",
    }
)

_MCP_HANDLER_FILES: list[Path] = [
    _OPENSPACE / "mcp_server.py",
]


class TestMCPHandlerBoundary:
    """MCP handlers must access OpenSpace through public API only.

    They must never reach into private fields (_store, _registry, etc.)
    """

    def test_known_private_access_count(self) -> None:
        """Guard: the known-violations allowlist must not silently grow."""
        assert len(self._KNOWN_PRIVATE_ACCESS) == 9, (
            f"If you fixed a private-field access, remove it from _KNOWN_PRIVATE_ACCESS. "
            f"If you added one, stop — use a public property. Count: {len(self._KNOWN_PRIVATE_ACCESS)}"
        )

    def test_mcp_handler_files_exist(self) -> None:
        """Sanity: MCP handler files we're guarding actually exist."""
        for f in _MCP_HANDLER_FILES:
            assert f.exists(), f"Expected MCP handler file: {f}"

    # Known Phase 1 private-field access (tech debt — to be replaced with
    # public property accessors as delegation is fully wired in Phase 4).
    _KNOWN_PRIVATE_ACCESS: frozenset[tuple[str, int, str]] = frozenset(
        {
            ("openspace/mcp_server.py", 185, "_skill_store"),
            ("openspace/mcp_server.py", 203, "_grounding_config"),
            ("openspace/mcp_server.py", 296, "_skill_registry"),
            ("openspace/mcp_server.py", 430, "_skill_registry"),
            ("openspace/mcp_server.py", 734, "_skill_registry"),
            ("openspace/mcp_server.py", 635, "_skill_registry"),
            ("openspace/mcp_server.py", 753, "_skill_evolver"),
            ("openspace/mcp_server.py", 737, "_skill_evolver"),
            ("openspace/mcp_server.py", 419, "_grounding_config"),
        }
    )

    def test_no_private_field_access(self) -> None:
        """MCP handlers must not access private fields of OpenSpace."""
        violations: list[str] = []

        for filepath in _MCP_HANDLER_FILES:
            rel = filepath.relative_to(_REPO_ROOT)
            rel_posix = rel.as_posix()
            source = filepath.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source, filename=str(filepath))
            except SyntaxError as exc:
                raise AssertionError(f"SyntaxError in {filepath} — cannot validate boundary: {exc}") from exc

            for node in ast.walk(tree):
                # Detect obj._field attribute access
                if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
                    attr = node.attr
                    if attr in _PRIVATE_FIELD_PATTERNS:
                        # Exclude self-references (class defining its own privates)
                        if isinstance(node.value, ast.Name) and node.value.id == "self":
                            continue
                        if (rel_posix, node.lineno, attr) in self._KNOWN_PRIVATE_ACCESS:
                            continue
                        violations.append(f"  {rel}:{node.lineno}: accesses '.{attr}'")

                # Detect getattr(obj, "_field") calls
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "getattr" and len(node.args) >= 2:
                        arg = node.args[1]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value in _PRIVATE_FIELD_PATTERNS:
                                if (rel_posix, node.lineno, arg.value) in self._KNOWN_PRIVATE_ACCESS:
                                    continue
                                violations.append(f"  {rel}:{node.lineno}: getattr(..., '{arg.value}')")

        if violations:
            detail = "\n".join(violations)
            pytest.fail(f"MCP handlers access {len(violations)} private field(s):\n{detail}")


# ---------------------------------------------------------------------------
# #82 — File Size Guard (warning-only in Phase 1)
# ---------------------------------------------------------------------------

_FILE_SIZE_LIMIT_KB = 15
_FILE_SIZE_LIMIT_BYTES = _FILE_SIZE_LIMIT_KB * 1024


class TestFileSizeGuard:
    """No .py file should exceed 15 KB.

    Phase 1: warn only (test always passes).
    Phase 7: this becomes a hard failure.
    """

    @staticmethod
    def _all_source_files() -> list[Path]:
        """Collect all .py files under openspace/."""
        return _collect_py_files(_OPENSPACE)

    def test_source_files_exist(self) -> None:
        """Sanity: we have source files to check."""
        assert len(self._all_source_files()) >= 10

    def test_file_sizes_within_limit(self) -> None:
        """Warn about files exceeding 15 KB (does NOT fail in Phase 1)."""
        oversized: list[tuple[str, int]] = []

        for filepath in self._all_source_files():
            size = filepath.stat().st_size
            if size > _FILE_SIZE_LIMIT_BYTES:
                rel = str(filepath.relative_to(_REPO_ROOT))
                oversized.append((rel, size))

        if oversized:
            oversized.sort(key=lambda x: -x[1])
            for rel, size in oversized:
                kb = size / 1024
                warnings.warn(
                    f"File exceeds {_FILE_SIZE_LIMIT_KB}KB: {rel} ({kb:.1f}KB)",
                    stacklevel=1,
                )
            # Phase 1: warn only — do NOT fail
            # Phase 7 will change this to:
            #   pytest.fail(f"{len(oversized)} file(s) exceed {_FILE_SIZE_LIMIT_KB}KB")


# ---------------------------------------------------------------------------
# #83 — CI Integration (meta-test)
# ---------------------------------------------------------------------------


class TestCIIntegration:
    """Verify architecture tests are discoverable by pytest (CI Tier 1).

    Since these tests live in tests/ and CI runs ``pytest tests/``,
    they are automatically part of CI Tier 1. This meta-test validates
    that assumption by checking our test file is in the right location.
    """

    def test_boundary_tests_in_tests_directory(self) -> None:
        """This file must live under tests/ to be CI-discoverable."""
        this_file = Path(__file__).resolve()
        tests_dir = _REPO_ROOT / "tests"
        assert str(this_file).startswith(str(tests_dir)), f"Boundary tests must be under {tests_dir}"

    def test_architecture_test_count(self) -> None:
        """We should have a meaningful number of architecture checks."""
        # Count test methods in this module (sanity that we haven't
        # accidentally disabled everything)
        import inspect

        test_classes = [
            TestDomainImportPurity,
            TestMCPHandlerBoundary,
            TestFileSizeGuard,
        ]
        test_count = sum(
            1
            for cls in test_classes
            for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith("test_")
        )
        assert test_count >= 8, f"Expected ≥8 architecture tests, got {test_count}"
