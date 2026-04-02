"""Tests for AnalysisStore (Epic 3.4) — execution analysis persistence."""

import pytest
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from openspace.skill_engine.analysis_store import AnalysisStore
from openspace.skill_engine.types import (
    ExecutionAnalysis,
    EvolutionSuggestion,
    EvolutionType, 
    SkillJudgment,
    SkillRecord,
    SkillCategory,
    SkillOrigin,
    SkillLineage,
    SkillVisibility,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database file for testing."""
    return tmp_path / "test_analysis_store.db"


@pytest.fixture
def analysis_store(temp_db_path):
    """AnalysisStore instance for testing."""
    store = AnalysisStore(db_path=temp_db_path)
    yield store
    store.close()


@pytest.fixture
def sample_analysis():
    """Sample ExecutionAnalysis for testing."""
    return ExecutionAnalysis(
        task_id="test_task_123",
        timestamp=datetime.now(),
        task_completed=True,
        execution_note="Test execution completed successfully",
        tool_issues=["tool1", "tool2"],
        skill_judgments=[
            SkillJudgment(
                skill_id="skill_1",
                skill_applied=True,
                note="Skill applied successfully"
            ),
            SkillJudgment(
                skill_id="skill_2", 
                skill_applied=False,
                note="Skill was not needed"
            )
        ],
        evolution_suggestions=[
            EvolutionSuggestion(
                evolution_type=EvolutionType.FIX,
                target_skill_ids=["skill_1"],
                direction="Improve efficiency - the skill could be more efficient"
            )
        ],
        analyzed_by="test_model",
        analyzed_at=datetime.now(),
    )


@pytest.fixture
def sample_skill_record():
    """Sample SkillRecord for testing hydration."""
    return SkillRecord(
        skill_id="skill_123",
        name="Test Skill",
        description="A test skill",
        path="/test/path",
        is_active=True,
        category=SkillCategory.WORKFLOW,
        visibility=SkillVisibility.PRIVATE,
        creator_id="test_creator",
        lineage=SkillLineage(
            origin=SkillOrigin.IMPORTED,
            generation=0,
            source_task_id=None,
            change_summary="Initial import",
            content_diff="",
            content_snapshot={},
            created_at=datetime.now(),
            created_by="test_creator",
        ),
        total_selections=0,
        total_applied=0,
        total_completions=0,
        total_fallbacks=0,
        first_seen=datetime.now(),
        last_updated=datetime.now(),
        recent_analyses=[],
        tool_dependencies=[],
        tags=[],
    )


class TestAnalysisStoreStandalone:
    """Test AnalysisStore in standalone mode."""

    def test_init_standalone(self, temp_db_path):
        """Test standalone initialization."""
        store = AnalysisStore(db_path=temp_db_path)
        assert store._owns_conn
        assert store._db_path == temp_db_path
        assert not store._closed
        store.close()

    def test_init_no_args_raises_error(self):
        """Test that initialization without args raises ValueError."""
        with pytest.raises(ValueError, match="Either db_path or conn must be provided"):
            AnalysisStore()

    def test_record_and_load_analysis(self, analysis_store, sample_analysis):
        """Test recording and loading an analysis."""
        # Record the analysis
        analysis_store.record_execution_analysis_sync(sample_analysis)
        
        # Load it back
        loaded = analysis_store.load_analyses_for_task(sample_analysis.task_id)
        assert loaded is not None
        assert loaded.task_id == sample_analysis.task_id
        assert loaded.task_completed == sample_analysis.task_completed
        assert len(loaded.skill_judgments) == 2
        assert len(loaded.evolution_suggestions) == 1

    def test_load_analyses_by_skill(self, analysis_store, sample_analysis):
        """Test loading analyses filtered by skill."""
        analysis_store.record_execution_analysis_sync(sample_analysis)
        
        # Load analyses for skill_1 
        analyses = analysis_store.load_analyses(skill_id="skill_1", limit=10)
        assert len(analyses) == 1
        assert analyses[0].task_id == sample_analysis.task_id

        # Load analyses for skill that doesn't exist
        analyses = analysis_store.load_analyses(skill_id="nonexistent", limit=10)
        assert len(analyses) == 0

    def test_load_all_analyses(self, analysis_store):
        """Test loading all analyses."""
        # Create multiple analyses
        analyses = []
        for i in range(3):
            analysis = ExecutionAnalysis(
                task_id=f"task_{i}",
                timestamp=datetime.now(),
                task_completed=i % 2 == 0,
                execution_note=f"Task {i} note",
                skill_judgments=[
                    SkillJudgment(skill_id="skill_1", skill_applied=True)
                ]
            )
            analyses.append(analysis)
            analysis_store.record_execution_analysis_sync(analysis)

        # Load all analyses
        all_analyses = analysis_store.load_all_analyses(limit=10)
        assert len(all_analyses) == 3

    def test_load_evolution_candidates(self, analysis_store):
        """Test loading evolution candidates."""
        # Analysis with evolution suggestion (candidate)
        candidate_analysis = ExecutionAnalysis(
            task_id="candidate_task",
            timestamp=datetime.now(),
            evolution_suggestions=[
                EvolutionSuggestion(
                    evolution_type=EvolutionType.FIX,
                    target_skill_ids=["skill_1"],
                    direction="Test suggestion"
                )
            ]
        )
        
        # Analysis without evolution suggestions (not candidate)
        regular_analysis = ExecutionAnalysis(
            task_id="regular_task",
            timestamp=datetime.now(),
        )
        
        analysis_store.record_execution_analysis_sync(candidate_analysis)
        analysis_store.record_execution_analysis_sync(regular_analysis)
        
        # Load evolution candidates
        candidates = analysis_store.load_evolution_candidates(limit=10)
        assert len(candidates) == 1
        assert candidates[0].task_id == "candidate_task"

    def test_load_recent_analyses_for_skill(self, analysis_store):
        """Test loading recent analyses for a specific skill."""
        # Create analyses with different skills
        for i in range(3):
            analysis = ExecutionAnalysis(
                task_id=f"task_{i}",
                timestamp=datetime.now(),
                skill_judgments=[
                    SkillJudgment(skill_id="target_skill", skill_applied=True)
                ]
            )
            analysis_store.record_execution_analysis_sync(analysis)
        
        # Load recent analyses for target_skill
        analyses = analysis_store.load_recent_analyses_for_skill("target_skill", limit=2)
        assert len(analyses) == 2

    def test_hydrate_recent_analyses(self, analysis_store, sample_skill_record, sample_analysis):
        """Test hydrating a skill record with recent analyses."""
        # Record an analysis that involves the skill
        sample_analysis.skill_judgments = [
            SkillJudgment(skill_id=sample_skill_record.skill_id, skill_applied=True)
        ]
        analysis_store.record_execution_analysis_sync(sample_analysis)
        
        # Hydrate the skill record
        hydrated_record = analysis_store.hydrate_recent_analyses(sample_skill_record)
        
        # Check that recent_analyses was populated
        assert len(hydrated_record.recent_analyses) == 1
        assert hydrated_record.recent_analyses[0].task_id == sample_analysis.task_id
        
        # Check that other fields were preserved
        assert hydrated_record.skill_id == sample_skill_record.skill_id
        assert hydrated_record.name == sample_skill_record.name

    def test_bulk_upsert_analyses(self, analysis_store):
        """Test bulk upserting analyses."""
        analyses = []
        for i in range(3):
            analysis = ExecutionAnalysis(
                task_id=f"bulk_task_{i}",
                timestamp=datetime.now(),
                execution_note=f"Bulk task {i}"
            )
            analyses.append(analysis)
        
        # Bulk upsert
        analysis_store.bulk_upsert_analyses(analyses)
        
        # Verify all were inserted
        all_analyses = analysis_store.load_all_analyses(limit=10)
        task_ids = [a.task_id for a in all_analyses]
        for i in range(3):
            assert f"bulk_task_{i}" in task_ids

    def test_get_analysis_stats(self, analysis_store):
        """Test getting analysis statistics."""
        # Initially should be empty
        stats = analysis_store.get_analysis_stats()
        assert stats["total_analyses"] == 0
        assert stats["evolution_candidates"] == 0
        
        # Add some analyses
        regular_analysis = ExecutionAnalysis(
            task_id="regular", 
            timestamp=datetime.now()
        )
        candidate_analysis = ExecutionAnalysis(
            task_id="candidate",
            timestamp=datetime.now(),
            evolution_suggestions=[
                EvolutionSuggestion(
                    evolution_type=EvolutionType.FIX,
                    target_skill_ids=["skill_1"],
                    direction="Test"
                )
            ]
        )
        
        analysis_store.record_execution_analysis_sync(regular_analysis)
        analysis_store.record_execution_analysis_sync(candidate_analysis)
        
        # Check updated stats
        stats = analysis_store.get_analysis_stats()
        assert stats["total_analyses"] == 2
        assert stats["evolution_candidates"] == 1

    def test_get_task_skill_summary(self, analysis_store, sample_analysis):
        """Test getting task skill summary."""
        analysis_store.record_execution_analysis_sync(sample_analysis)
        
        summary = analysis_store.get_task_skill_summary(sample_analysis.task_id)
        
        assert summary["task_id"] == sample_analysis.task_id
        assert summary["task_completed"] == sample_analysis.task_completed
        assert summary["execution_note"] == sample_analysis.execution_note
        assert len(summary["judgments"]) == 2
        assert summary["judgments"][0]["skill_id"] == "skill_1"
        
        # Test nonexistent task
        empty_summary = analysis_store.get_task_skill_summary("nonexistent")
        assert empty_summary == {}

    def test_clear_all_analyses(self, analysis_store, sample_analysis):
        """Test clearing all analyses."""
        # Add some analyses
        analysis_store.record_execution_analysis_sync(sample_analysis)
        
        # Verify they exist
        all_analyses = analysis_store.load_all_analyses()
        assert len(all_analyses) == 1
        
        # Clear them
        analysis_store.clear_all_analyses()
        
        # Verify they're gone
        all_analyses = analysis_store.load_all_analyses()
        assert len(all_analyses) == 0

    def test_close_and_reopen(self, temp_db_path, sample_analysis):
        """Test data persistence across close/reopen."""
        # Create store, add data, close
        store1 = AnalysisStore(db_path=temp_db_path)
        store1.record_execution_analysis_sync(sample_analysis)
        store1.close()
        
        # Reopen and verify data persists
        store2 = AnalysisStore(db_path=temp_db_path)
        loaded = store2.load_analyses_for_task(sample_analysis.task_id)
        assert loaded is not None
        assert loaded.task_id == sample_analysis.task_id
        store2.close()


class TestAnalysisStoreSharedConnection:
    """Test AnalysisStore with shared connection (embedded mode)."""

    def test_shared_connection_mode(self, temp_db_path):
        """Test AnalysisStore with shared connection."""
        import sqlite3
        import threading
        
        # Create shared connection and lock
        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row
        lock = threading.Lock()
        
        # Initialize the schema manually since we're not using standalone mode
        from openspace.skill_engine.migration_manager import MigrationManager
        migration_manager = MigrationManager(conn=conn, lock=lock)
        migration_manager.ensure_current_schema()
        
        # Create AnalysisStore with shared connection
        store = AnalysisStore(conn=conn, lock=lock)
        assert not store._owns_conn
        assert store._conn is conn
        assert store._mu is lock
        
        # Test basic operations work
        analysis = ExecutionAnalysis(
            task_id="shared_test",
            timestamp=datetime.now(),
        )
        store.record_execution_analysis_sync(analysis)
        
        loaded = store.load_analyses_for_task("shared_test")
        assert loaded is not None
        assert loaded.task_id == "shared_test"
        
        # Close should not close shared connection
        store.close()
        # Connection should still be usable
        result = conn.execute("SELECT COUNT(*) FROM execution_analyses").fetchone()
        assert result[0] == 1
        
        conn.close()


def test_analysis_store_integration_with_skill_store(temp_db_path):
    """Test AnalysisStore integration with SkillStore."""
    from openspace.skill_engine.store import SkillStore
    
    store = SkillStore(db_path=temp_db_path)
    
    # Verify AnalysisStore is properly initialized
    assert hasattr(store, '_analyses')
    assert isinstance(store._analyses, AnalysisStore)
    assert not store._analyses._owns_conn  # Should be sharing connection
    assert store._analyses._conn is store._conn
    assert store._analyses._mu is store._mu
    
    # Test facade methods delegate properly
    analysis = ExecutionAnalysis(
        task_id="integration_test",
        timestamp=datetime.now(),
        skill_judgments=[
            SkillJudgment(skill_id="test_skill", skill_applied=True)
        ]
    )
    
    # This should delegate to _analyses.record_execution_analysis_sync
    import asyncio
    asyncio.run(store.record_analysis(analysis))
    
    # These should delegate to AnalysisStore methods
    loaded = store.load_analyses_for_task("integration_test")
    assert loaded is not None
    
    all_analyses = store.load_all_analyses()
    assert len(all_analyses) == 1
    
    summary = store.get_task_skill_summary("integration_test") 
    assert summary["task_id"] == "integration_test"
    
    store.close()


def test_duplicate_judgments_no_duplicate_analysis_rows(temp_db_path):
    """Test that multiple judgments for different skills in same analysis don't create duplicate analysis rows."""
    store = AnalysisStore(db_path=temp_db_path)
    
    # Create analysis with judgments for different skills (no duplicates)
    analysis = ExecutionAnalysis(
        task_id="no_duplicate_judgment_test",
        timestamp=datetime.now(),
        skill_judgments=[
            SkillJudgment(skill_id="first_skill", skill_applied=True, note="First judgment"),
            SkillJudgment(skill_id="second_skill", skill_applied=False, note="Second judgment"),
            SkillJudgment(skill_id="third_skill", skill_applied=True, note="Third judgment"),
        ]
    )
    
    store.record_execution_analysis_sync(analysis)
    
    # Load analyses for the first skill - should only get one analysis back
    analyses = store.load_analyses(skill_id="first_skill", limit=10)
    assert len(analyses) == 1, "Should get exactly one analysis row"
    assert analyses[0].task_id == "no_duplicate_judgment_test"
    
    # Load recent analyses - should also not have duplicates  
    recent_analyses = store.load_recent_analyses_for_skill("first_skill", limit=10)
    assert len(recent_analyses) == 1, "Should get exactly one analysis row in recent analyses"
    
    store.close()


def test_duplicate_judgments_prevented_by_unique_constraint(temp_db_path):
    """Test that the UNIQUE constraint prevents duplicate judgments for same skill in same analysis."""
    store = AnalysisStore(db_path=temp_db_path)
    
    # Create analysis with duplicate judgments for the same skill (should fail)
    analysis = ExecutionAnalysis(
        task_id="duplicate_judgment_test",
        timestamp=datetime.now(),
        skill_judgments=[
            SkillJudgment(skill_id="shared_skill", skill_applied=True, note="First judgment"),
            SkillJudgment(skill_id="shared_skill", skill_applied=False, note="Second judgment"),  # Duplicate!
            SkillJudgment(skill_id="other_skill", skill_applied=True, note="Other judgment"),
        ]
    )
    
    # This should fail due to UNIQUE constraint on (analysis_id, skill_id)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        store.record_execution_analysis_sync(analysis)
    
    store.close()


def test_bulk_upsert_atomicity_with_validation_error(temp_db_path):
    """Test that bulk_upsert with mixed valid/invalid analyses is all-or-nothing."""
    import sqlite3
    import threading
    
    # Test in shared connection mode where atomicity issue exists
    conn = sqlite3.connect(str(temp_db_path))
    conn.row_factory = sqlite3.Row
    lock = threading.Lock()
    
    # Initialize schema
    from openspace.skill_engine.migration_manager import MigrationManager
    migration_manager = MigrationManager(conn=conn, lock=lock)
    migration_manager.ensure_current_schema()
    
    store = AnalysisStore(conn=conn, lock=lock)
    
    # Create mix of valid and invalid analyses
    valid_analysis1 = ExecutionAnalysis(
        task_id="valid_1",
        timestamp=datetime.now(),
        execution_note="Valid analysis 1"
    )
    
    valid_analysis2 = ExecutionAnalysis(
        task_id="valid_2", 
        timestamp=datetime.now(),
        execution_note="Valid analysis 2"
    )
    
    # Create invalid analysis (missing required task_id)
    invalid_analysis = ExecutionAnalysis(
        task_id="",  # Invalid empty task_id
        timestamp=datetime.now(),
        execution_note="Invalid analysis"
    )
    
    analyses = [valid_analysis1, invalid_analysis, valid_analysis2]
    
    # Bulk upsert should fail due to invalid analysis
    with pytest.raises(Exception):  # Should raise validation error
        with lock:
            conn.execute("BEGIN")
            try:
                store.bulk_upsert_analyses(analyses)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    # Verify NO analyses were inserted (atomicity)
    all_analyses = store.load_all_analyses()
    assert len(all_analyses) == 0, "No analyses should be inserted when bulk operation fails"
    
    conn.close()


def test_store_calls_public_insert_analysis_method(tmp_path):
    """Test that store.py calls the public insert_analysis method (not private)."""
    # This test verifies the layering fix - store.py should call public methods
    temp_db_path = tmp_path / "test_public_method.db"
    
    from openspace.skill_engine.store import SkillStore
    
    store = SkillStore(db_path=temp_db_path)
    
    # Verify the AnalysisStore has public insert_analysis method
    assert hasattr(store._analyses, 'insert_analysis'), "insert_analysis should be public"
    
    # Verify it's callable (not private)
    assert callable(store._analyses.insert_analysis), "insert_analysis should be callable"
    
    # This should work without accessing private methods
    analysis = ExecutionAnalysis(
        task_id="public_method_test", 
        timestamp=datetime.now(),
        skill_judgments=[
            SkillJudgment(skill_id="test_skill", skill_applied=True)
        ]
    )
    
    import asyncio
    asyncio.run(store.record_analysis(analysis))
    
    # Verify it was recorded successfully
    loaded = store.load_analyses_for_task("public_method_test")
    assert loaded is not None
    assert loaded.task_id == "public_method_test"
    
    store.close()