"""Tests for EPIC 0.9 — Benchmark extraction from production code.

Verifies that production ``openspace/`` modules have no runtime coupling
to ``gdpval_bench`` (the benchmark harness).  The benchmark package is
CLI-only and must never be imported during MCP server startup or
tool execution.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import List, Set

import pytest

# Production package root
_OPENSPACE_ROOT = Path(__file__).resolve().parent.parent / "openspace"

# Files that are explicitly allowed to reference gdpval_bench
# (dashboard is not part of the MCP server)
_ALLOWED_FILES: Set[str] = {
    "dashboard_server.py",
}


def _find_python_files(root: Path) -> List[Path]:
    """Recursively find all .py files under *root*."""
    return sorted(root.rglob("*.py"))


def _has_gdpval_import(source: str) -> List[str]:
    """Return list of gdpval_bench import statements found in *source*."""
    findings: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gdpval_bench"):
                    findings.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("gdpval_bench"):
                names = ", ".join(a.name for a in node.names)
                findings.append(f"from {node.module} import {names} (line {node.lineno})")
    return findings


class TestNoBenchmarkImports:
    """Production code must not import gdpval_bench."""

    def test_no_gdpval_imports_in_production(self):
        """Scan all .py files under openspace/ for gdpval_bench imports."""
        violations: List[str] = []
        for py_file in _find_python_files(_OPENSPACE_ROOT):
            if py_file.name in _ALLOWED_FILES:
                continue
            source = py_file.read_text(encoding="utf-8", errors="replace")
            findings = _has_gdpval_import(source)
            if findings:
                rel = py_file.relative_to(_OPENSPACE_ROOT.parent)
                for f in findings:
                    violations.append(f"{rel}: {f}")

        assert violations == [], "Production code must not import gdpval_bench:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

    def test_allowed_files_list_is_minimal(self):
        """Ensure _ALLOWED_FILES only contains files that actually exist."""
        for name in _ALLOWED_FILES:
            assert (_OPENSPACE_ROOT / name).exists(), f"{name} is in _ALLOWED_FILES but doesn't exist"

    @pytest.mark.parametrize(
        "module_path",
        [
            "openspace.llm.client",
            "openspace.skill_engine.registry",
            "openspace.skill_engine.evolver",
            "openspace.skill_engine.analyzer",
            "openspace.grounding.core.quality.manager",
        ],
    )
    def test_previously_coupled_modules_clean(self, module_path):
        """Verify the 5 modules that previously imported gdpval_bench are clean."""
        py_file = _OPENSPACE_ROOT.parent / module_path.replace(".", "/")
        py_file = py_file.with_suffix(".py")
        assert py_file.exists(), f"{module_path} not found at {py_file}"

        source = py_file.read_text(encoding="utf-8")
        findings = _has_gdpval_import(source)
        assert findings == [], f"{module_path} still imports gdpval_bench:\n" + "\n".join(f"  - {f}" for f in findings)


class TestBenchmarkIsolation:
    """gdpval_bench must be completely separate from openspace runtime."""

    def test_gdpval_bench_not_in_sys_modules_after_openspace_import(self):
        """Importing openspace modules must not pull in gdpval_bench."""
        # Clear any cached gdpval_bench modules
        gdpval_mods = [k for k in sys.modules if k.startswith("gdpval_bench")]
        saved = {k: sys.modules.pop(k) for k in gdpval_mods}

        try:
            # Force reimport of the previously-coupled modules
            for mod_name in [
                "openspace.skill_engine.registry",
                "openspace.skill_engine.evolver",
                "openspace.skill_engine.analyzer",
            ]:
                try:
                    if mod_name in sys.modules:
                        importlib.reload(sys.modules[mod_name])
                    else:
                        importlib.import_module(mod_name)
                except (ImportError, ModuleNotFoundError):
                    # Skip modules with missing optional deps (e.g. litellm)
                    pass

            # Verify no gdpval_bench modules crept in
            leaked = [k for k in sys.modules if k.startswith("gdpval_bench")]
            assert leaked == [], f"gdpval_bench leaked into sys.modules: {leaked}"
        finally:
            # Restore
            sys.modules.update(saved)

    def test_token_tracker_not_referenced_in_strings(self):
        """No string references to token_tracker in production code."""
        for py_file in _find_python_files(_OPENSPACE_ROOT):
            if py_file.name in _ALLOWED_FILES:
                continue
            source = py_file.read_text(encoding="utf-8", errors="replace")
            if "token_tracker" in source:
                rel = py_file.relative_to(_OPENSPACE_ROOT.parent)
                pytest.fail(f"{rel} still references 'token_tracker' — benchmark coupling not fully removed")
