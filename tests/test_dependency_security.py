"""Tests for EPIC 0.10 — Dependency security.

Verifies that all dependencies in pyproject.toml have upper-bound
version constraints and that critical security infrastructure
(Dependabot, pip-audit CI job) is in place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Version constraint tests
# ---------------------------------------------------------------------------

_UPPER_BOUND_RE = re.compile(r"<\d|!=")


class TestDependencyPinning:
    """All deps must have upper-bound version constraints."""

    def _check_deps(self, deps: list[str], section: str):
        violations = []
        for dep in deps:
            # Strip extras markers like ; sys_platform == 'darwin'
            spec = dep.split(";")[0].strip()
            if not _UPPER_BOUND_RE.search(spec):
                violations.append(spec)
        assert violations == [], f"[{section}] deps without upper bounds:\n" + "\n".join(f"  - {v}" for v in violations)

    def test_core_deps_have_upper_bounds(self, pyproject):
        deps = pyproject["project"]["dependencies"]
        self._check_deps(deps, "dependencies")

    def test_dev_deps_have_upper_bounds(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["dev"]
        self._check_deps(deps, "dev")

    def test_macos_deps_have_upper_bounds(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["macos"]
        self._check_deps(deps, "macos")

    def test_linux_deps_have_upper_bounds(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["linux"]
        self._check_deps(deps, "linux")

    def test_windows_deps_have_upper_bounds(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["windows"]
        self._check_deps(deps, "windows")

    def test_litellm_has_security_cap(self, pyproject):
        """litellm must pin >=1.83.0 (fixes CVE-2026-35029/35030, skips PYSEC-2026-2)."""
        deps = pyproject["project"]["dependencies"]
        litellm_specs = [d for d in deps if d.startswith("litellm")]
        assert len(litellm_specs) == 1
        assert ">=1.83.0" in litellm_specs[0]


# ---------------------------------------------------------------------------
# Infrastructure tests
# ---------------------------------------------------------------------------


class TestSecurityInfrastructure:
    """Dependabot and pip-audit must be configured."""

    def test_dependabot_config_exists(self):
        path = _PROJECT_ROOT / ".github" / "dependabot.yml"
        assert path.exists(), "Missing .github/dependabot.yml"
        content = path.read_text()
        assert "pip" in content, "Dependabot must monitor pip ecosystem"
        assert "github-actions" in content, "Dependabot must monitor GH Actions"

    def test_ci_has_pip_audit_job(self):
        ci_path = _PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), "Missing CI workflow"
        content = ci_path.read_text()
        assert "pip-audit" in content, "CI must include pip-audit job"

    def test_requirements_txt_synced(self):
        """requirements.txt must match pyproject.toml core deps."""
        req_path = _PROJECT_ROOT / "requirements.txt"
        assert req_path.exists()
        req_content = req_path.read_text()

        pyproject_data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
        core_deps = pyproject_data["project"]["dependencies"]

        for dep in core_deps:
            # Extract package name (before any version specifier)
            pkg_name = re.split(r"[><=!~\[]", dep)[0].strip().lower()
            assert pkg_name in req_content.lower(), f"{pkg_name} in pyproject.toml but missing from requirements.txt"

    def test_no_black_or_flake8_in_dev_deps(self, pyproject):
        """Dev deps should use ruff, not black+flake8 (replaced in EPIC 0.8)."""
        dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
        dep_names = [re.split(r"[><=!~\[]", d)[0].strip().lower() for d in dev_deps]
        assert "black" not in dep_names, "Use ruff instead of black"
        assert "flake8" not in dep_names, "Use ruff instead of flake8"
