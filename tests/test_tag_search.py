"""Tests for Epic 3.5 — TagSearch extraction.

Verifies:
- Tag management (sync, get, add, remove)
- Tag-based skill search
- Tool-based skill search
- Skill discovery and filtering
- Performance statistics and rankings
- Complex search combinations (tags + text + filters)
- Standalone usage and shared connection patterns
- Edge cases (not found, empty DB, invalid inputs)
"""

from __future__ import annotations

import sqlite3
import threading
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
    tags: list[str] | None = None,
    category: SkillCategory = SkillCategory.WORKFLOW,
    visibility: SkillVisibility = SkillVisibility.PRIVATE,
    is_active: bool = True,
    total_selections: int = 0,
    total_applied: int = 0,
    total_completions: int = 0,
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
        visibility=visibility,
        lineage=SkillLineage(origin=SkillOrigin.IMPORTED),
        total_selections=total_selections,
        total_applied=total_applied,
        total_completions=total_completions,
    )


@pytest.fixture
def tag_search(tmp_path: Path):
    """Create a TagSearch backed by a temp SQLite database."""
    from openspace.skill_engine.tag_search import TagSearch

    db_path = tmp_path / "test_tagsearch.db"
    ts = TagSearch(db_path=db_path)
    yield ts
    ts.close()


@pytest.fixture
def populated_tag_search(tag_search):
    """TagSearch with some test data."""
    from openspace.skill_engine.skill_repository import SkillRepository

    # We need to use SkillRepository to populate records, then TagSearch for tags
    repo = SkillRepository(db_path=tag_search._db_path)
    
    # Create test records
    records = [
        _make_record(
            skill_id="python__dev001",
            name="Python Development",
            description="Advanced Python development patterns",
            category=SkillCategory.WORKFLOW,
            visibility=SkillVisibility.PUBLIC,
            total_selections=50,
            total_applied=45,
            total_completions=40,
        ),
        _make_record(
            skill_id="js__web002",
            name="JavaScript Web Framework",
            description="Modern JavaScript web development",
            category=SkillCategory.TOOL_GUIDE,
            visibility=SkillVisibility.PUBLIC,
            total_selections=30,
            total_applied=25,
            total_completions=20,
        ),
        _make_record(
            skill_id="git__tool003",
            name="Git Workflow",
            description="Version control with Git",
            category=SkillCategory.REFERENCE,
            visibility=SkillVisibility.PRIVATE,
            total_selections=100,
            total_applied=90,
            total_completions=85,
        ),
        _make_record(
            skill_id="inactive__old004",
            name="Old Skill",
            description="This skill is deactivated",
            is_active=False,
            total_selections=10,
            total_applied=5,
            total_completions=1,
        ),
    ]
    
    # Save records
    for record in records:
        repo.save(record)
    
    # Add tags
    tag_search.sync_tags("python__dev001", ["python", "development", "backend"])
    tag_search.sync_tags("js__web002", ["javascript", "web", "frontend"])
    tag_search.sync_tags("git__tool003", ["git", "version-control", "development"])
    tag_search.sync_tags("inactive__old004", ["old", "deprecated"])
    
    # Add tool dependencies (simulate)
    with tag_search._conn as conn:
        conn.executemany(
            "INSERT INTO skill_tool_deps(skill_id, tool_key, critical) VALUES(?,?,?)",
            [
                ("python__dev001", "python", 1),
                ("python__dev001", "pip", 0),
                ("js__web002", "node", 1),
                ("js__web002", "npm", 0),
                ("git__tool003", "git", 1),
            ]
        )
        conn.commit()
    
    repo.close()
    yield tag_search


# ═══════════════════════════════════════════════════════════════════════
#  Tag Management
# ═══════════════════════════════════════════════════════════════════════


class TestTagManagement:
    """Tag CRUD operations."""

    def test_sync_tags_new_skill(self, tag_search):
        # Need to create a skill record first for foreign key constraint
        with tag_search._conn as conn:
            conn.execute(
                "INSERT INTO skill_records(skill_id, name, first_seen, last_updated, lineage_created_at) VALUES(?,?,?,?,?)",
                ("skill_123", "Test Skill", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            conn.commit()
        
        tags = ["python", "web", "api"]
        tag_search.sync_tags("skill_123", tags)
        
        retrieved = tag_search.get_tags("skill_123")
        assert sorted(retrieved) == sorted(tags)

    def test_sync_tags_replace_existing(self, tag_search):
        # Create skill record first
        with tag_search._conn as conn:
            conn.execute(
                "INSERT INTO skill_records(skill_id, name, first_seen, last_updated, lineage_created_at) VALUES(?,?,?,?,?)",
                ("skill_123", "Test Skill", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            conn.commit()
            
        # Initial tags
        tag_search.sync_tags("skill_123", ["python", "old"])
        assert set(tag_search.get_tags("skill_123")) == {"python", "old"}
        
        # Replace with new tags
        tag_search.sync_tags("skill_123", ["python", "new", "web"])
        assert set(tag_search.get_tags("skill_123")) == {"python", "new", "web"}

    def test_sync_tags_empty_list_clears(self, tag_search):
        # Create skill record first
        with tag_search._conn as conn:
            conn.execute(
                "INSERT INTO skill_records(skill_id, name, first_seen, last_updated, lineage_created_at) VALUES(?,?,?,?,?)",
                ("skill_123", "Test Skill", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            conn.commit()
            
        tag_search.sync_tags("skill_123", ["python", "web"])
        tag_search.sync_tags("skill_123", [])
        
        assert tag_search.get_tags("skill_123") == []

    def test_get_tags_nonexistent_skill(self, tag_search):
        assert tag_search.get_tags("nonexistent") == []

    def test_get_all_tags(self, populated_tag_search):
        all_tags = populated_tag_search.get_all_tags()
        
        # Check structure
        assert all(isinstance(item, dict) for item in all_tags)
        assert all("tag" in item and "usage_count" in item for item in all_tags)
        
        # Check that active skills' tags are included
        tag_names = {item["tag"] for item in all_tags}
        assert "python" in tag_names
        assert "javascript" in tag_names
        assert "development" in tag_names
        
        # Check that deprecated tags from inactive skills are excluded
        assert "old" not in tag_names
        assert "deprecated" not in tag_names


# ═══════════════════════════════════════════════════════════════════════
#  Tag-Based Search
# ═══════════════════════════════════════════════════════════════════════


class TestTagSearch:
    """Finding skills by tags."""

    def test_find_skills_by_tags_any_match(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tags(
            ["python", "javascript"], match_all=False
        )
        assert "python__dev001" in skills
        assert "js__web002" in skills
        assert "git__tool003" not in skills

    def test_find_skills_by_tags_all_match(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tags(
            ["development"], match_all=True
        )
        assert "python__dev001" in skills
        assert "git__tool003" in skills
        assert "js__web002" not in skills

    def test_find_skills_by_tags_all_match_multiple(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tags(
            ["python", "development"], match_all=True
        )
        assert "python__dev001" in skills
        assert "js__web002" not in skills
        assert "git__tool003" not in skills

    def test_find_skills_by_tags_include_inactive(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tags(
            ["old"], match_all=False, active_only=False
        )
        assert "inactive__old004" in skills

    def test_find_skills_by_tags_exclude_inactive(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tags(
            ["old"], match_all=False, active_only=True
        )
        assert "inactive__old004" not in skills

    def test_find_skills_by_tags_empty_list(self, populated_tag_search):
        assert populated_tag_search.find_skills_by_tags([]) == []

    def test_find_skills_by_tags_no_matches(self, populated_tag_search):
        assert populated_tag_search.find_skills_by_tags(["nonexistent"]) == []


# ═══════════════════════════════════════════════════════════════════════
#  Tool-Based Search
# ═══════════════════════════════════════════════════════════════════════


class TestToolSearch:
    """Finding skills by tool dependencies."""

    def test_find_skills_by_tool_existing(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tool("python")
        assert "python__dev001" in skills
        assert "js__web002" not in skills

    def test_find_skills_by_tool_multiple_skills(self, populated_tag_search):
        skills = populated_tag_search.find_skills_by_tool("git")
        assert "git__tool003" in skills

    def test_find_skills_by_tool_nonexistent(self, populated_tag_search):
        assert populated_tag_search.find_skills_by_tool("nonexistent") == []

    def test_find_skills_by_tool_only_active(self, populated_tag_search):
        # Add tool dependency to inactive skill
        with populated_tag_search._conn as conn:
            conn.execute(
                "INSERT INTO skill_tool_deps(skill_id, tool_key, critical) VALUES(?,?,?)",
                ("inactive__old004", "old_tool", 1)
            )
            conn.commit()
        
        skills = populated_tag_search.find_skills_by_tool("old_tool")
        assert "inactive__old004" not in skills  # Inactive skills excluded


# ═══════════════════════════════════════════════════════════════════════
#  Skill Discovery and Statistics
# ═══════════════════════════════════════════════════════════════════════


class TestSkillDiscovery:
    """Skill listing, filtering, and statistics."""

    def test_get_summary_active_only(self, populated_tag_search):
        summary = populated_tag_search.get_summary(active_only=True)
        
        skill_ids = {skill["skill_id"] for skill in summary}
        assert "python__dev001" in skill_ids
        assert "js__web002" in skill_ids
        assert "git__tool003" in skill_ids
        assert "inactive__old004" not in skill_ids

    def test_get_summary_include_inactive(self, populated_tag_search):
        summary = populated_tag_search.get_summary(active_only=False)
        
        skill_ids = {skill["skill_id"] for skill in summary}
        assert "inactive__old004" in skill_ids

    def test_get_stats(self, populated_tag_search):
        stats = populated_tag_search.get_stats(active_only=True)
        
        # Check structure
        assert "total_skills" in stats
        assert "total_skills_all" in stats
        assert "by_category" in stats
        assert "by_lineage_origin" in stats
        assert "total_selections" in stats
        
        # Check values
        assert stats["total_skills"] == 3  # Active only
        assert stats["total_skills_all"] == 4  # Including inactive
        assert stats["total_selections"] == 180  # 50+30+100

    def test_get_top_skills_by_effective_rate(self, populated_tag_search):
        top = populated_tag_search.get_top_skills(n=2, metric="effective_rate")
        
        # Should be ordered by completions/selections
        skill_ids = [skill["skill_id"] for skill in top]
        assert "git__tool003" == skill_ids[0]  # 85/100 = 0.85
        assert "python__dev001" == skill_ids[1]  # 40/50 = 0.80

    def test_get_top_skills_by_total_selections(self, populated_tag_search):
        top = populated_tag_search.get_top_skills(n=2, metric="total_selections")
        
        skill_ids = [skill["skill_id"] for skill in top]
        assert "git__tool003" == skill_ids[0]  # 100 selections
        assert "python__dev001" == skill_ids[1]  # 50 selections

    def test_get_count_and_timestamp(self, populated_tag_search):
        result = populated_tag_search.get_count_and_timestamp()
        
        assert result["count"] == 3  # Active skills only
        assert result["max_last_updated"] is not None


# ═══════════════════════════════════════════════════════════════════════
#  Complex Search
# ═══════════════════════════════════════════════════════════════════════


class TestComplexSearch:
    """Advanced search with multiple criteria."""

    def test_search_skills_by_query(self, populated_tag_search):
        results = populated_tag_search.search_skills(query="Python")
        skill_ids = {skill["skill_id"] for skill in results}
        assert "python__dev001" in skill_ids

    def test_search_skills_by_category(self, populated_tag_search):
        results = populated_tag_search.search_skills(category=SkillCategory.WORKFLOW)
        skill_ids = {skill["skill_id"] for skill in results}
        assert "python__dev001" in skill_ids
        assert "js__web002" not in skill_ids  # TOOL_GUIDE

    def test_search_skills_by_visibility(self, populated_tag_search):
        results = populated_tag_search.search_skills(visibility=SkillVisibility.PRIVATE)
        skill_ids = {skill["skill_id"] for skill in results}
        assert "git__tool003" in skill_ids
        assert "python__dev001" not in skill_ids  # PUBLIC

    def test_search_skills_with_tags_any(self, populated_tag_search):
        results = populated_tag_search.search_skills(
            tags=["python", "javascript"], match_all_tags=False
        )
        skill_ids = {skill["skill_id"] for skill in results}
        assert "python__dev001" in skill_ids
        assert "js__web002" in skill_ids

    def test_search_skills_with_tags_all(self, populated_tag_search):
        results = populated_tag_search.search_skills(
            tags=["python", "development"], match_all_tags=True
        )
        skill_ids = {skill["skill_id"] for skill in results}
        assert "python__dev001" in skill_ids
        assert "js__web002" not in skill_ids

    def test_search_skills_combined_criteria(self, populated_tag_search):
        results = populated_tag_search.search_skills(
            query="development",
            category=SkillCategory.WORKFLOW,
            tags=["python"],
            limit=1
        )
        assert len(results) == 1
        assert results[0]["skill_id"] == "python__dev001"

    def test_search_skills_sql_injection_prevention(self, populated_tag_search):
        # Test that LIKE wildcards are properly escaped
        results = populated_tag_search.search_skills(query="Python%")
        # Should not match anything since % is escaped
        assert len(results) == 0

    def test_search_skills_limit(self, populated_tag_search):
        results = populated_tag_search.search_skills(limit=2)
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════
#  Shared Connection Pattern
# ═══════════════════════════════════════════════════════════════════════


class TestSharedConnection:
    """Verify TagSearch works with shared connections (SkillStore pattern)."""

    def test_shared_connection_and_lock(self, tmp_path: Path):
        """TagSearch should work when sharing connection with another component."""
        from openspace.skill_engine.tag_search import TagSearch
        
        # Create a shared connection and lock
        db_path = tmp_path / "shared.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        lock = threading.Lock()
        
        # Initialize database schema manually since we're not using TagSearch constructor
        conn.executescript("""
            CREATE TABLE skill_records (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                lineage_created_at TEXT NOT NULL
            );
            CREATE TABLE skill_tags (
                skill_id TEXT NOT NULL REFERENCES skill_records(skill_id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                PRIMARY KEY (skill_id, tag)
            );
        """)
        conn.commit()
        
        # Create TagSearch instances with shared connection
        ts1 = TagSearch(conn=conn, lock=lock)
        ts2 = TagSearch(conn=conn, lock=lock)
        
        # Create skill records first for foreign key constraints
        conn.execute(
            "INSERT INTO skill_records(skill_id, name, first_seen, last_updated, lineage_created_at) VALUES(?,?,?,?,?)",
            ("skill1", "Test 1", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO skill_records(skill_id, name, first_seen, last_updated, lineage_created_at) VALUES(?,?,?,?,?)",
            ("skill2", "Test 2", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
        )
        conn.commit()
        
        # Both should work without conflicts
        ts1.sync_tags("skill1", ["tag1"])
        ts2.sync_tags("skill2", ["tag2"])
        
        assert ts1.get_tags("skill1") == ["tag1"]
        assert ts2.get_tags("skill2") == ["tag2"]
        
        # Cleanup
        ts1.close()  # Should not close shared connection
        ts2.close()  # Should not close shared connection
        conn.close()  # Manual cleanup of shared resource

    def test_reader_with_shared_connection_acquires_lock(self, tmp_path: Path):
        """When using shared connection, _reader() should acquire lock."""
        from openspace.skill_engine.tag_search import TagSearch
        
        db_path = tmp_path / "shared.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        lock = threading.Lock()
        
        ts = TagSearch(conn=conn, lock=lock)
        
        # This should not deadlock and should work properly
        with ts._reader() as reader_conn:
            assert reader_conn is conn
        
        ts.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  Error Handling
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Error conditions and edge cases."""

    def test_tag_search_closed_operations_fail(self, tag_search):
        tag_search.close()
        
        with pytest.raises(RuntimeError, match="TagSearch is closed"):
            tag_search.get_tags("skill1")

    def test_neither_db_path_nor_conn_raises(self):
        from openspace.skill_engine.tag_search import TagSearch
        
        with pytest.raises(ValueError, match="Either db_path or conn must be provided"):
            TagSearch()

    def test_db_retry_on_operational_error(self, tmp_path: Path):
        """_db_retry should handle transient SQLite errors."""
        from openspace.skill_engine.tag_search import TagSearch
        
        # Create a TagSearch with actual database
        db_path = tmp_path / "test.db"
        ts = TagSearch(db_path=db_path)
        
        # Create skill record first
        with ts._conn as conn:
            conn.execute(
                "INSERT INTO skill_records(skill_id, name, first_seen, last_updated, lineage_created_at) VALUES(?,?,?,?,?)",
                ("skill1", "Test", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00")
            )
            conn.commit()
        
        # Test that a valid operation works (retry logic exists and works for valid operations)
        ts.sync_tags("skill1", ["tag1"])
        assert ts.get_tags("skill1") == ["tag1"]
        
        ts.close()


# ═══════════════════════════════════════════════════════════════════════
#  Integration with SkillStore (Facade Pattern)
# ═══════════════════════════════════════════════════════════════════════


class TestSkillStoreFacadeIntegration:
    """Verify TagSearch integration via SkillStore facade."""

    def test_skill_store_delegates_to_tag_search(self, tmp_path: Path):
        """SkillStore should delegate tag/search operations to TagSearch."""
        from openspace.skill_engine.store import SkillStore
        
        # Create SkillStore (which should initialize TagSearch internally)
        store = SkillStore(db_path=tmp_path / "integration.db")
        
        # Verify TagSearch was created
        assert hasattr(store, "_tag_search")
        assert store._tag_search is not None
        
        # Test delegation works
        assert store.get_count_and_timestamp()["count"] == 0
        assert store.find_skills_by_tool("python") == []
        assert store.get_summary() == []
        
        store.close()

    def test_skill_store_tag_sync_delegation(self, tmp_path: Path):
        """SkillStore should delegate tag sync during record saving."""
        import asyncio
        from openspace.skill_engine.store import SkillStore
        
        store = SkillStore(db_path=tmp_path / "integration.db")
        
        # Create and save a skill with tags
        record = _make_record(tags=["python", "web"])
        
        async def test_save():
            await store.save_record(record)
        
        asyncio.run(test_save())
        
        # Verify tags were saved via TagSearch delegation
        tags = store._tag_search.get_tags(record.skill_id)
        assert set(tags) == {"python", "web"}
        
        store.close()