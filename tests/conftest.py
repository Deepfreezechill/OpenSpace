"""Shared pytest fixtures for the Scion test suite.

Provides:
  - ``mock_llm_client`` — deterministic LLM mock (no real API calls)
  - ``in_memory_store`` — SQLite-backed SkillStore using :memory:
  - ``work_dir`` — clean temporary directory for filesystem-heavy tests
  - ``temp_skill_dir`` — temporary directory pre-populated with sample skills
  - ``mock_env`` — safely set / restore environment variables
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Dict, Generator

import pytest

from tests.mocks.llm import MockLLMClient

# ---------------------------------------------------------------------------
# mock_llm_client — deterministic LLM responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Return a ``MockLLMClient`` with a single default response.

    Override by creating the fixture with custom responses::

        @pytest.fixture
        def mock_llm_client():
            return MockLLMClient(responses=["custom answer"])
    """
    return MockLLMClient()


# ---------------------------------------------------------------------------
# in_memory_store — ephemeral SkillStore backed by :memory: SQLite
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_store(tmp_path: Path):
    """Create a SkillStore backed by a temporary SQLite database.

    Uses a temp-file database (not ``:memory:``) so that the SkillStore
    class can open its own connection with the path it expects.
    The file is cleaned up automatically by pytest's ``tmp_path``.
    """
    from scion.skill_engine.store import SkillStore

    db_path = tmp_path / "test_scion.db"
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# work_dir — clean temporary directory for filesystem-heavy tests
# ---------------------------------------------------------------------------


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Return a clean temporary directory for filesystem-heavy tests.

    NOTE: This is a convenience alias for ``tmp_path``, NOT a security
    sandbox.  Code under test can still access paths outside this
    directory.  For true filesystem isolation, use E2B or a chroot.
    """
    return tmp_path


# ---------------------------------------------------------------------------
# temp_skill_dir — directory with sample SKILL.md files
# ---------------------------------------------------------------------------

_SAMPLE_SKILL_TEMPLATE = textwrap.dedent("""\
    ---
    name: {name}
    description: {description}
    ---

    # {name}

    This is a sample skill for testing purposes.

    ## Steps

    1. Do the first thing.
    2. Do the second thing.
    3. Verify the result.
""")


@pytest.fixture
def temp_skill_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with three sample skills.

    Directory layout::

        tmp/skills/
        ├── weather_lookup/
        │   └── SKILL.md
        ├── code_review/
        │   └── SKILL.md
        └── file_organizer/
            └── SKILL.md
    """
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    samples = [
        ("weather_lookup", "Look up current weather for a city"),
        ("code_review", "Review code changes and suggest improvements"),
        ("file_organizer", "Organize files in a directory by type"),
    ]

    for name, description in samples:
        skill_dir = skills_root / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            _SAMPLE_SKILL_TEMPLATE.format(name=name, description=description),
            encoding="utf-8",
        )

    return skills_root


# ---------------------------------------------------------------------------
# mock_env — safe environment variable manipulation
# ---------------------------------------------------------------------------


class _EnvPatcher:
    """Context-manager / callable that sets env vars and restores originals."""

    def __init__(self) -> None:
        self._originals: Dict[str, str | None] = {}

    def set(self, key: str, value: str) -> None:
        """Set an environment variable, recording the original value."""
        if key not in self._originals:
            self._originals[key] = os.environ.get(key)
        os.environ[key] = value

    def delete(self, key: str) -> None:
        """Remove an environment variable, recording the original value."""
        if key not in self._originals:
            self._originals[key] = os.environ.get(key)
        os.environ.pop(key, None)

    def restore(self) -> None:
        """Restore all modified variables to their original values."""
        for key, original in self._originals.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        self._originals.clear()


@pytest.fixture
def mock_env() -> Generator[_EnvPatcher, None, None]:
    """Fixture that provides an ``_EnvPatcher`` for safe env-var manipulation.

    Usage::

        def test_something(mock_env):
            mock_env.set("MY_VAR", "value")
            mock_env.delete("OTHER_VAR")
            # ...test runs with modified env...
        # originals automatically restored
    """
    patcher = _EnvPatcher()
    yield patcher
    patcher.restore()
