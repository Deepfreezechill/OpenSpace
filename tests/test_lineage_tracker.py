"""Tests for Epic 3.3 — LineageTracker extraction from monolithic SkillStore.

Verifies:
- record_derivation (parent-child relationship recording)
- get_children (find direct children of a skill)
- get_ancestors (walk up the lineage tree)
- get_evolution_chain (all versions of a named skill by generation)
- get_lineage_tree (JSON-friendly downward tree)
- Edge cases (orphans, circular refs, missing records, empty DB)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from openspace.skill_engine.types import (
    SkillCategory,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillVisibility,
)


def _make_record(
    skill_id: str = "test__abc123",
    name: str = "test_skill",
    description: str = "A test skill",
    generation: int = 0,
    origin: SkillOrigin = SkillOrigin.IMPORTED,
    parent_skill_ids: list[str] | None = None,
    is_active: bool = True,
    change_summary: str = "",
) -> SkillRecord:
    """Helper: create a minimal valid SkillRecord with lineage."""
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        description=description,
        path=f"/skills/{name}/SKILL.md",
        is_active=is_active,
        category=SkillCategory.WORKFLOW,
        tags=[],
        visibility=SkillVisibility.PRIVATE,
        lineage=SkillLineage(
            origin=origin,
            generation=generation,
            parent_skill_ids=parent_skill_ids or [],
            change_summary=change_summary,
        ),
    )


@pytest.fixture
def tracker(tmp_path: Path):
    """Create a LineageTracker backed by a temp SQLite database."""
    from openspace.skill_engine.lineage_tracker import LineageTracker

    db_path = tmp_path / "test_lineage.db"
    lt = LineageTracker(db_path=db_path)
    yield lt
    lt.close()


@pytest.fixture
def seeded_tracker(tracker):
    """Tracker pre-seeded with a 3-generation chain: root → child → grandchild."""
    from openspace.skill_engine.lineage_tracker import LineageTracker

    lt: LineageTracker = tracker

    root = _make_record(
        skill_id="skill__root",
        name="weather_guide",
        generation=0,
        origin=SkillOrigin.IMPORTED,
    )
    child = _make_record(
        skill_id="skill__child",
        name="weather_guide",
        generation=1,
        origin=SkillOrigin.FIXED,
        parent_skill_ids=["skill__root"],
        is_active=True,
        change_summary="Fixed curl params",
    )
    grandchild = _make_record(
        skill_id="skill__grandchild",
        name="weather_guide",
        generation=2,
        origin=SkillOrigin.FIXED,
        parent_skill_ids=["skill__child"],
        is_active=True,
        change_summary="Added error handling",
    )

    # Use the internal repo to persist records
    lt._repo.save(root)
    lt._repo.save(child)
    lt._repo.save(grandchild)

    return lt


# ═══════════════════════════════════════════════════════════════════════
#  record_derivation
# ═══════════════════════════════════════════════════════════════════════


class TestRecordDerivation:
    """record_derivation() should persist parent-child relationships."""

    def test_record_single_parent(self, tracker):
        parent = _make_record(skill_id="p1", name="parent_skill")
        tracker._repo.save(parent)

        child = _make_record(
            skill_id="c1",
            name="parent_skill",
            generation=1,
            origin=SkillOrigin.FIXED,
            parent_skill_ids=["p1"],
        )
        tracker.record_derivation(child, parent_skill_ids=["p1"])

        children = tracker.get_children("p1")
        assert "c1" in children

    def test_record_multiple_parents(self, tracker):
        p1 = _make_record(skill_id="p1", name="skill_a")
        p2 = _make_record(skill_id="p2", name="skill_b")
        tracker._repo.save(p1)
        tracker._repo.save(p2)

        derived = _make_record(
            skill_id="d1",
            name="composed_skill",
            generation=1,
            origin=SkillOrigin.DERIVED,
            parent_skill_ids=["p1", "p2"],
        )
        tracker.record_derivation(derived, parent_skill_ids=["p1", "p2"])

        assert "d1" in tracker.get_children("p1")
        assert "d1" in tracker.get_children("p2")

    def test_record_derivation_deactivates_parent_for_fixed(self, tracker):
        parent = _make_record(skill_id="p1", name="my_skill", is_active=True)
        tracker._repo.save(parent)

        child = _make_record(
            skill_id="c1",
            name="my_skill",
            generation=1,
            origin=SkillOrigin.FIXED,
            parent_skill_ids=["p1"],
        )
        tracker.record_derivation(child, parent_skill_ids=["p1"])

        # Parent should be deactivated for FIXED origin
        loaded_parent = tracker._repo.get("p1")
        assert loaded_parent is not None
        assert not loaded_parent.is_active

    def test_record_derivation_keeps_parent_active_for_derived(self, tracker):
        parent = _make_record(skill_id="p1", name="skill_a", is_active=True)
        tracker._repo.save(parent)

        derived = _make_record(
            skill_id="d1",
            name="new_composed",
            generation=1,
            origin=SkillOrigin.DERIVED,
            parent_skill_ids=["p1"],
        )
        tracker.record_derivation(derived, parent_skill_ids=["p1"])

        loaded_parent = tracker._repo.get("p1")
        assert loaded_parent is not None
        assert loaded_parent.is_active


# ═══════════════════════════════════════════════════════════════════════
#  get_children
# ═══════════════════════════════════════════════════════════════════════


class TestGetChildren:
    """get_children() should return skill_ids of direct children."""

    def test_no_children(self, tracker):
        orphan = _make_record(skill_id="orphan", name="lonely_skill")
        tracker._repo.save(orphan)
        assert tracker.get_children("orphan") == []

    def test_multiple_children(self, tracker):
        parent = _make_record(skill_id="p1", name="parent")
        tracker._repo.save(parent)

        for i in range(3):
            child = _make_record(
                skill_id=f"c{i}",
                name=f"child_{i}",
                generation=1,
                origin=SkillOrigin.DERIVED,
                parent_skill_ids=["p1"],
            )
            tracker.record_derivation(child, parent_skill_ids=["p1"])

        children = tracker.get_children("p1")
        assert set(children) == {"c0", "c1", "c2"}

    def test_nonexistent_parent(self, tracker):
        assert tracker.get_children("nonexistent") == []


# ═══════════════════════════════════════════════════════════════════════
#  get_ancestors
# ═══════════════════════════════════════════════════════════════════════


class TestGetAncestors:
    """get_ancestors() should walk up the lineage tree, oldest-first."""

    def test_no_ancestors(self, tracker):
        root = _make_record(skill_id="root", name="root_skill")
        tracker._repo.save(root)
        assert tracker.get_ancestors("root") == []

    def test_single_parent(self, seeded_tracker):
        ancestors = seeded_tracker.get_ancestors("skill__child")
        assert len(ancestors) == 1
        assert ancestors[0].skill_id == "skill__root"

    def test_full_chain(self, seeded_tracker):
        ancestors = seeded_tracker.get_ancestors("skill__grandchild")
        assert len(ancestors) == 2
        ids = [a.skill_id for a in ancestors]
        assert ids == ["skill__root", "skill__child"]

    def test_max_depth_limit(self, seeded_tracker):
        ancestors = seeded_tracker.get_ancestors("skill__grandchild", max_depth=1)
        assert len(ancestors) == 1
        assert ancestors[0].skill_id == "skill__child"

    def test_nonexistent_skill(self, tracker):
        assert tracker.get_ancestors("ghost") == []


# ═══════════════════════════════════════════════════════════════════════
#  get_evolution_chain
# ═══════════════════════════════════════════════════════════════════════


class TestGetEvolutionChain:
    """get_evolution_chain() returns all versions of a named skill by generation."""

    def test_single_version(self, tracker):
        record = _make_record(skill_id="s1", name="unique_skill")
        tracker._repo.save(record)

        chain = tracker.get_evolution_chain("unique_skill")
        assert len(chain) == 1
        assert chain[0].skill_id == "s1"

    def test_multi_version_chain(self, seeded_tracker):
        chain = seeded_tracker.get_evolution_chain("weather_guide")
        assert len(chain) == 3
        generations = [r.lineage.generation for r in chain]
        assert generations == [0, 1, 2]

    def test_no_versions(self, tracker):
        chain = tracker.get_evolution_chain("nonexistent")
        assert chain == []


# ═══════════════════════════════════════════════════════════════════════
#  get_lineage_tree
# ═══════════════════════════════════════════════════════════════════════


class TestGetLineageTree:
    """get_lineage_tree() builds a JSON-friendly downward tree."""

    def test_leaf_node(self, seeded_tracker):
        tree = seeded_tracker.get_lineage_tree("skill__grandchild")
        assert tree["skill_id"] == "skill__grandchild"
        assert tree["children"] == []

    def test_root_with_descendants(self, seeded_tracker):
        tree = seeded_tracker.get_lineage_tree("skill__root")
        assert tree["skill_id"] == "skill__root"
        assert len(tree["children"]) == 1
        child = tree["children"][0]
        assert child["skill_id"] == "skill__child"
        assert len(child["children"]) == 1
        assert child["children"][0]["skill_id"] == "skill__grandchild"

    def test_max_depth_truncation(self, seeded_tracker):
        tree = seeded_tracker.get_lineage_tree("skill__root", max_depth=1)
        assert tree["skill_id"] == "skill__root"
        assert len(tree["children"]) == 1
        child = tree["children"][0]
        assert child["children"] == []

    def test_nonexistent_skill(self, tracker):
        tree = tracker.get_lineage_tree("ghost")
        assert tree["skill_id"] == "ghost"
        assert tree["name"] == "?"
        assert tree["children"] == []

    def test_tree_structure_fields(self, seeded_tracker):
        tree = seeded_tracker.get_lineage_tree("skill__root")
        assert "skill_id" in tree
        assert "name" in tree
        assert "generation" in tree
        assert "origin" in tree
        assert "is_active" in tree
        assert "children" in tree


# ═══════════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases: orphans, circular refs, diamond inheritance."""

    def test_orphan_skill(self, tracker):
        """A skill with no parents and no children is handled gracefully."""
        orphan = _make_record(skill_id="orphan", name="orphan_skill")
        tracker._repo.save(orphan)

        assert tracker.get_children("orphan") == []
        assert tracker.get_ancestors("orphan") == []
        tree = tracker.get_lineage_tree("orphan")
        assert tree["children"] == []

    def test_diamond_inheritance(self, tracker):
        """A → B, A → C, B+C → D (diamond pattern)."""
        a = _make_record(skill_id="a", name="skill_a")
        tracker._repo.save(a)

        b = _make_record(
            skill_id="b", name="skill_b", generation=1,
            origin=SkillOrigin.DERIVED, parent_skill_ids=["a"],
        )
        tracker.record_derivation(b, parent_skill_ids=["a"])

        c = _make_record(
            skill_id="c", name="skill_c", generation=1,
            origin=SkillOrigin.DERIVED, parent_skill_ids=["a"],
        )
        tracker.record_derivation(c, parent_skill_ids=["a"])

        d = _make_record(
            skill_id="d", name="skill_d", generation=2,
            origin=SkillOrigin.DERIVED, parent_skill_ids=["b", "c"],
        )
        tracker.record_derivation(d, parent_skill_ids=["b", "c"])

        # D has ancestors: A, B, C (sorted by generation)
        ancestors = tracker.get_ancestors("d")
        ancestor_ids = [a.skill_id for a in ancestors]
        assert "a" in ancestor_ids
        assert "b" in ancestor_ids
        assert "c" in ancestor_ids

        # A has children B and C
        children_of_a = tracker.get_children("a")
        assert set(children_of_a) == {"b", "c"}

    def test_shared_conn_injection(self, tmp_path):
        """LineageTracker works with an injected connection + lock."""
        import sqlite3
        import threading

        from openspace.skill_engine.lineage_tracker import LineageTracker
        from openspace.skill_engine.skill_repository import SkillRepository

        db_path = tmp_path / "shared.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        lock = threading.Lock()

        # Create tables via SkillRepository (which owns DDL)
        repo = SkillRepository(db_path=db_path)

        # Now create LineageTracker sharing the repo's conn
        lt = LineageTracker(conn=repo._conn, lock=lock)

        record = _make_record(skill_id="shared1", name="shared_skill")
        repo.save(record)

        children = lt.get_children("shared1")
        assert children == []

        repo.close()
        conn.close()

    def test_get_lineage(self, seeded_tracker):
        """get_lineage() returns the SkillLineage for a given skill_id."""
        lineage = seeded_tracker.get_lineage("skill__child")
        assert lineage is not None
        assert lineage.origin == SkillOrigin.FIXED
        assert lineage.generation == 1
        assert lineage.parent_skill_ids == ["skill__root"]

    def test_get_lineage_missing(self, tracker):
        """get_lineage() returns None for nonexistent skill."""
        assert tracker.get_lineage("ghost") is None
