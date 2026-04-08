"""Tests for Epic 3.2 — SkillRepository CRUD extraction.

Verifies:
- save / upsert skill records
- get skill by ID
- delete skill by ID
- list all skills (with optional filters)
- search skills by name/tags
- count skills
- check if skill exists
- validation enforcement (invalid records rejected)
- edge cases (not found, duplicates, empty DB)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from scion.skill_engine.types import (
    SkillCategory,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillVisibility,
    ValidationError,
)


def _make_record(
    skill_id: str = "test__abc123",
    name: str = "test_skill",
    description: str = "A test skill",
    tags: list[str] | None = None,
    category: SkillCategory = SkillCategory.WORKFLOW,
    is_active: bool = True,
) -> SkillRecord:
    """Helper: create a minimal valid SkillRecord."""
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        description=description,
        path=f"/skills/{name}/SKILL.md",
        is_active=is_active,
        category=category,
        tags=tags or [],
        visibility=SkillVisibility.PRIVATE,
        lineage=SkillLineage(origin=SkillOrigin.IMPORTED),
    )


@pytest.fixture
def repo(tmp_path: Path):
    """Create a SkillRepository backed by a temp SQLite database."""
    from scion.skill_engine.skill_repository import SkillRepository

    db_path = tmp_path / "test_repo.db"
    repository = SkillRepository(db_path=db_path)
    yield repository
    repository.close()


# ═══════════════════════════════════════════════════════════════════════
#  Save / Upsert
# ═══════════════════════════════════════════════════════════════════════


class TestSave:
    """save() should insert new records and upsert existing ones."""

    def test_save_new_record(self, repo):
        record = _make_record()
        repo.save(record)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.skill_id == record.skill_id
        assert loaded.name == record.name
        assert loaded.description == record.description

    def test_save_upsert_updates_existing(self, repo):
        record = _make_record()
        repo.save(record)

        # Update description via upsert
        from dataclasses import replace

        updated = replace(record, description="Updated description")
        repo.save(updated)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.description == "Updated description"
        assert repo.count() == 1  # No duplicate created

    def test_save_multiple_records(self, repo):
        records = [
            _make_record(skill_id="s1", name="skill_one"),
            _make_record(skill_id="s2", name="skill_two"),
            _make_record(skill_id="s3", name="skill_three"),
        ]
        repo.save_many(records)
        assert repo.count() == 3

    def test_save_preserves_tags(self, repo):
        record = _make_record(tags=["python", "automation", "testing"])
        repo.save(record)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert set(loaded.tags) == {"python", "automation", "testing"}

    def test_save_preserves_category(self, repo):
        record = _make_record(category=SkillCategory.TOOL_GUIDE)
        repo.save(record)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.category == SkillCategory.TOOL_GUIDE


# ═══════════════════════════════════════════════════════════════════════
#  Validation Enforcement
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """save() must call validate() and reject invalid records."""

    def test_reject_empty_skill_id(self, repo):
        record = _make_record(skill_id="")
        with pytest.raises(ValidationError):
            repo.save(record)

    def test_reject_empty_name(self, repo):
        record = _make_record(name="")
        with pytest.raises(ValidationError):
            repo.save(record)

    def test_reject_invalid_counters(self, repo):
        record = _make_record()
        # Force invalid counters: applied > selections
        record.total_selections = 5
        record.total_applied = 10
        with pytest.raises(ValidationError):
            repo.save(record)


# ═══════════════════════════════════════════════════════════════════════
#  Get by ID
# ═══════════════════════════════════════════════════════════════════════


class TestGet:
    """get() should retrieve by skill_id, or return None."""

    def test_get_existing(self, repo):
        record = _make_record()
        repo.save(record)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.skill_id == record.skill_id

    def test_get_nonexistent_returns_none(self, repo):
        assert repo.get("nonexistent__id") is None

    def test_get_returns_full_record(self, repo):
        record = _make_record(
            tags=["web", "api"],
            category=SkillCategory.REFERENCE,
        )
        repo.save(record)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.category == SkillCategory.REFERENCE
        assert set(loaded.tags) == {"web", "api"}
        assert loaded.lineage.origin == SkillOrigin.IMPORTED


# ═══════════════════════════════════════════════════════════════════════
#  Delete by ID
# ═══════════════════════════════════════════════════════════════════════


class TestDelete:
    """delete() should remove the record and return success/failure."""

    def test_delete_existing(self, repo):
        record = _make_record()
        repo.save(record)
        assert repo.delete(record.skill_id) is True
        assert repo.get(record.skill_id) is None

    def test_delete_nonexistent_returns_false(self, repo):
        assert repo.delete("nonexistent__id") is False

    def test_delete_does_not_affect_others(self, repo):
        r1 = _make_record(skill_id="s1", name="keep_me")
        r2 = _make_record(skill_id="s2", name="delete_me")
        repo.save_many([r1, r2])

        repo.delete("s2")
        assert repo.get("s1") is not None
        assert repo.get("s2") is None
        assert repo.count() == 1


# ═══════════════════════════════════════════════════════════════════════
#  List All (with filters)
# ═══════════════════════════════════════════════════════════════════════


class TestListAll:
    """list_all() should return all records, optionally filtered."""

    def test_list_all_empty_db(self, repo):
        result = repo.list_all()
        assert result == {}

    def test_list_all_returns_dict(self, repo):
        r1 = _make_record(skill_id="s1", name="alpha")
        r2 = _make_record(skill_id="s2", name="beta")
        repo.save_many([r1, r2])

        result = repo.list_all()
        assert len(result) == 2
        assert "s1" in result
        assert "s2" in result

    def test_list_all_active_only(self, repo):
        active = _make_record(skill_id="s1", name="active", is_active=True)
        inactive = _make_record(skill_id="s2", name="inactive", is_active=False)
        repo.save_many([active, inactive])

        all_records = repo.list_all(active_only=False)
        assert len(all_records) == 2

        active_records = repo.list_all(active_only=True)
        assert len(active_records) == 1
        assert "s1" in active_records


# ═══════════════════════════════════════════════════════════════════════
#  Search by Name / Tags
# ═══════════════════════════════════════════════════════════════════════


class TestSearch:
    """search() should find skills by name pattern or tags."""

    def test_search_by_name(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="weather_lookup"),
            _make_record(skill_id="s2", name="code_review"),
            _make_record(skill_id="s3", name="weather_forecast"),
        ])

        results = repo.search(name="weather")
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"weather_lookup", "weather_forecast"}

    def test_search_by_tags(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="a", tags=["python", "api"]),
            _make_record(skill_id="s2", name="b", tags=["javascript"]),
            _make_record(skill_id="s3", name="c", tags=["python", "testing"]),
        ])

        results = repo.search(tags=["python"])
        assert len(results) == 2
        ids = {r.skill_id for r in results}
        assert ids == {"s1", "s3"}

    def test_search_no_match(self, repo):
        repo.save(_make_record(skill_id="s1", name="weather"))
        results = repo.search(name="nonexistent")
        assert results == []

    def test_search_by_category(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="a", category=SkillCategory.TOOL_GUIDE),
            _make_record(skill_id="s2", name="b", category=SkillCategory.WORKFLOW),
        ])

        results = repo.search(category=SkillCategory.TOOL_GUIDE)
        assert len(results) == 1
        assert results[0].skill_id == "s1"


# ═══════════════════════════════════════════════════════════════════════
#  Count
# ═══════════════════════════════════════════════════════════════════════


class TestCount:
    """count() should return the total number of records."""

    def test_count_empty(self, repo):
        assert repo.count() == 0

    def test_count_all(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="a"),
            _make_record(skill_id="s2", name="b"),
        ])
        assert repo.count() == 2

    def test_count_active_only(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="a", is_active=True),
            _make_record(skill_id="s2", name="b", is_active=False),
            _make_record(skill_id="s3", name="c", is_active=True),
        ])
        assert repo.count(active_only=True) == 2
        assert repo.count(active_only=False) == 3


# ═══════════════════════════════════════════════════════════════════════
#  Exists
# ═══════════════════════════════════════════════════════════════════════


class TestExists:
    """exists() should check if a skill_id is present."""

    def test_exists_true(self, repo):
        repo.save(_make_record(skill_id="s1", name="present"))
        assert repo.exists("s1") is True

    def test_exists_false(self, repo):
        assert repo.exists("nonexistent") is False

    def test_exists_after_delete(self, repo):
        repo.save(_make_record(skill_id="s1", name="temp"))
        repo.delete("s1")
        assert repo.exists("s1") is False


# ═══════════════════════════════════════════════════════════════════════
#  Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Various edge cases and boundary conditions."""

    def test_save_empty_batch(self, repo):
        """save_many with empty list should not error."""
        repo.save_many([])
        assert repo.count() == 0

    def test_search_empty_db(self, repo):
        assert repo.search(name="anything") == []
        assert repo.search(tags=["anything"]) == []

    def test_save_and_get_preserves_lineage(self, repo):
        lin = SkillLineage(
            origin=SkillOrigin.IMPORTED,
            generation=0,
            change_summary="Initial import",
        )
        record = _make_record()
        from dataclasses import replace

        record = replace(record, lineage=lin)
        repo.save(record)

        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.lineage.origin == SkillOrigin.IMPORTED
        assert loaded.lineage.change_summary == "Initial import"

    def test_duplicate_save_same_id_is_upsert(self, repo):
        """Saving twice with same ID should update, not duplicate."""
        repo.save(_make_record(skill_id="dup", name="v1"))
        repo.save(_make_record(skill_id="dup", name="v2"))
        assert repo.count() == 1
        loaded = repo.get("dup")
        assert loaded is not None
        assert loaded.name == "v2"

    def test_close_prevents_further_operations(self, repo):
        repo.close()
        with pytest.raises(RuntimeError):
            repo.get("anything")


# ═══════════════════════════════════════════════════════════════════════
#  F2: Shared Lock (dual-mutex fix)
# ═══════════════════════════════════════════════════════════════════════


class TestSharedLock:
    """When given an external lock, SkillRepository should use it."""

    def test_uses_provided_lock(self, tmp_path):
        from scion.skill_engine.skill_repository import SkillRepository
        import threading

        external_lock = threading.Lock()
        db_path = tmp_path / "shared_lock.db"
        repo = SkillRepository(db_path=db_path, lock=external_lock)
        assert repo._mu is external_lock
        repo.close()

    def test_creates_own_lock_when_none_provided(self, tmp_path):
        from scion.skill_engine.skill_repository import SkillRepository
        import threading

        db_path = tmp_path / "own_lock.db"
        repo = SkillRepository(db_path=db_path)
        assert isinstance(repo._mu, type(threading.Lock()))
        repo.close()

    def test_shared_conn_and_lock(self, tmp_path):
        """Shared connection + shared lock should work without deadlock."""
        from scion.skill_engine.skill_repository import SkillRepository
        import sqlite3
        import threading

        db_path = tmp_path / "shared.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        lock = threading.Lock()

        repo = SkillRepository(conn=conn, lock=lock)
        # Init tables manually since shared conn skips _init_db
        repo._owns_conn = True  # temporarily to init
        repo._db_path = db_path
        repo._init_db()
        repo._owns_conn = False
        repo._db_path = Path(":shared:")

        record = _make_record()
        repo.save(record)
        loaded = repo.get(record.skill_id)
        assert loaded is not None
        assert loaded.skill_id == record.skill_id

        repo.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  F3: LIKE Wildcard Escaping
# ═══════════════════════════════════════════════════════════════════════


class TestSearchEscaping:
    """search() must escape SQL LIKE wildcards in user input."""

    def test_percent_in_name_does_not_match_broadly(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="100%_done"),
            _make_record(skill_id="s2", name="100_other_tasks"),
            _make_record(skill_id="s3", name="something_else"),
        ])

        results = repo.search(name="100%_done")
        assert len(results) == 1
        assert results[0].skill_id == "s1"

    def test_underscore_in_name_is_literal(self, repo):
        repo.save_many([
            _make_record(skill_id="s1", name="my_skill"),
            _make_record(skill_id="s2", name="myXskill"),
        ])

        # Without escaping, "_" would match any single char including "X"
        results = repo.search(name="my_skill")
        assert len(results) == 1
        assert results[0].skill_id == "s1"

    def test_backslash_in_name_is_literal(self, repo):
        repo.save(_make_record(skill_id="s1", name="path\\to\\skill"))
        results = repo.search(name="path\\to")
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════
#  F4: JSON Safety in _to_record
# ═══════════════════════════════════════════════════════════════════════


class TestJsonSafety:
    """_to_record should survive malformed JSON in lineage_content_snapshot."""

    def test_malformed_snapshot_falls_back_to_empty_dict(self, repo):
        """Insert a record, corrupt the JSON, then load it."""
        record = _make_record(skill_id="corrupt1", name="corrupt_test")
        repo.save(record)

        # Directly corrupt the snapshot column
        repo._conn.execute(
            "UPDATE skill_records SET lineage_content_snapshot = ? WHERE skill_id = ?",
            ("NOT VALID JSON {{{", "corrupt1"),
        )
        repo._conn.commit()

        loaded = repo.get("corrupt1")
        assert loaded is not None
        assert loaded.lineage.content_snapshot == {}

    def test_empty_snapshot_falls_back_to_empty_dict(self, repo):
        record = _make_record(skill_id="empty1", name="empty_test")
        repo.save(record)

        repo._conn.execute(
            "UPDATE skill_records SET lineage_content_snapshot = '' WHERE skill_id = ?",
            ("empty1",),
        )
        repo._conn.commit()

        loaded = repo.get("empty1")
        assert loaded is not None
        assert loaded.lineage.content_snapshot == {}
