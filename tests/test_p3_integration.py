"""Phase 3 Integration Tests — Epic 3.7

Comprehensive tests exercising full workflows through the SkillStore facade,
verifying all extracted modules work together:
- SkillStore (facade) → store.py
- MigrationManager → migration_manager.py
- SkillRepository → skill_repository.py
- LineageTracker → lineage_tracker.py
- AnalysisStore → analysis_store.py
- TagSearch → tag_search.py

Tests full end-to-end workflows, not individual module boundaries.
"""

import asyncio
import concurrent.futures
import gc
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

from openspace.skill_engine.store import SkillStore
from openspace.skill_engine.types import (
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
    SkillCategory,
    SkillJudgment,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillVisibility,
)


@pytest.fixture
def temp_db():
    """Temporary database for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def store(temp_db):
    """Clean SkillStore instance for each test."""
    store = SkillStore(temp_db)
    try:
        yield store
    finally:
        try:
            store.close()
        except Exception:
            pass
        # Force garbage collection to clean up any lingering connections
        gc.collect()
        # Small delay to let Windows finish closing file handles
        time.sleep(0.1)


@pytest.fixture
def sample_record():
    """Sample SkillRecord for testing."""
    return SkillRecord(
        skill_id="test_skill_v1",
        name="test_skill",
        description="Test skill for integration testing",
        category=SkillCategory.TOOL_GUIDE,
        visibility=SkillVisibility.PUBLIC,
        path="/test/skill.py",
        lineage=SkillLineage(
            origin=SkillOrigin.IMPORTED,
            generation=0,
            content_snapshot={"skill.py": "def test_skill():\n    pass"},
        ),
        tool_dependencies=[],
        critical_tools=[],
        total_selections=0,
        total_applied=0,
        total_completions=0,
        total_fallbacks=0,
        recent_analyses=[],
        first_seen=datetime.now(),
        last_updated=datetime.now(),
    )


@pytest.fixture
def sample_analysis(sample_record):
    """Sample ExecutionAnalysis for testing."""
    return ExecutionAnalysis(
        task_id="task_123",
        timestamp=datetime.now(),
        task_completed=True,
        execution_note="Test execution completed successfully",
        skill_judgments=[
            SkillJudgment(
                skill_id="test_skill_v1",
                skill_applied=True,
                note="Skill executed successfully",
            )
        ],
        analyzed_at=datetime.now(),
    )


class TestSkillLifecycleWorkflow:
    """Test complete skill lifecycle: create → save → get → update → delete."""

    @pytest.mark.asyncio
    async def test_basic_crud_operations(self, store, sample_record):
        """Test basic create, read, update, delete through facade."""
        # CREATE: Save record
        await store.save_record(sample_record)
        
        # READ: Load record back
        loaded = store.load_record(sample_record.skill_id)
        assert loaded is not None
        assert loaded.skill_id == sample_record.skill_id
        assert loaded.name == sample_record.name
        assert loaded.description == sample_record.description
        
        # UPDATE: Modify and save again
        sample_record.description = "Updated description"
        sample_record.total_selections = 5
        await store.save_record(sample_record)
        
        updated = store.load_record(sample_record.skill_id)
        assert updated.description == "Updated description"
        assert updated.total_selections == 5
        
        # DEACTIVATE: Soft delete
        result = await store.deactivate_record(sample_record.skill_id)
        assert result is True
        
        # Should not appear in active queries
        active_skills = store.load_active()
        assert sample_record.skill_id not in active_skills
        
        # But should still exist in full query
        all_skills = store.load_all(active_only=False)
        assert sample_record.skill_id in all_skills
        
        # REACTIVATE
        result = await store.reactivate_record(sample_record.skill_id)
        assert result is True
        
        active_skills = store.load_active()
        assert sample_record.skill_id in active_skills
        
        # HARD DELETE
        result = await store.delete_record(sample_record.skill_id)
        assert result is True
        
        deleted = store.load_record(sample_record.skill_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_batch_operations(self, store):
        """Test batch save and load operations."""
        # Create multiple records
        records = []
        for i in range(5):
            record = SkillRecord(
                skill_id=f"batch_skill_v{i}",
                name=f"batch_skill_{i}",
                description=f"Batch skill {i}",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path=f"/test/batch_{i}.py",
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={f"batch_{i}.py": f"def batch_skill_{i}():\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=i,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            records.append(record)
        
        # Batch save
        await store.save_records(records)
        
        # Verify all were saved
        all_skills = store.load_all()
        for record in records:
            assert record.skill_id in all_skills
            loaded = all_skills[record.skill_id]
            assert loaded.name == record.name
            assert loaded.total_selections == record.total_selections


class TestLineageWorkflow:
    """Test lineage workflow: create → evolve → record → traverse."""

    @pytest.mark.asyncio
    async def test_evolution_and_lineage_tracking(self, store, sample_record):
        """Test skill evolution with lineage tracking."""
        # Save initial skill
        await store.save_record(sample_record)
        
        # Evolve the skill
        evolved_record = SkillRecord(
            skill_id="evolved_test_skill_v1",
            name="evolved_test_skill",
            description="Evolved test skill with improvements",
            category=sample_record.category,
            visibility=sample_record.visibility,
            path=sample_record.path,
            lineage=SkillLineage(
                origin=SkillOrigin.DERIVED,
                generation=1,
                parent_skill_ids=[sample_record.skill_id],
                change_summary="Added return value",
                content_snapshot={"skill.py": "def evolved_test_skill():\n    return 'improved'"},
            ),
            tool_dependencies=[],
            critical_tools=[],
            total_selections=0,
            total_applied=0,
            total_completions=0,
            total_fallbacks=0,
            recent_analyses=[],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
        )
        
        await store.evolve_skill(evolved_record, [sample_record.skill_id])
        evolved_skill_id = evolved_record.skill_id
        
        assert evolved_skill_id != sample_record.skill_id
        
        # Load evolved skill
        evolved = store.load_record(evolved_skill_id)
        assert evolved is not None
        assert evolved.name == "evolved_test_skill"
        assert evolved.lineage.generation == 1
        assert evolved.lineage.origin == SkillOrigin.DERIVED
        
        # Check lineage relationships
        children = store.find_children(sample_record.skill_id)
        assert evolved_skill_id in children
        
        # Get ancestry
        ancestry = store.get_ancestry(evolved_skill_id)
        assert len(ancestry) == 1  # just parent
        assert ancestry[0].skill_id == sample_record.skill_id
        
        # Get lineage tree
        tree = store.get_lineage_tree(sample_record.skill_id)
        assert tree["skill_id"] == sample_record.skill_id
        assert len(tree["children"]) == 1
        assert tree["children"][0]["skill_id"] == evolved_skill_id

    @pytest.mark.asyncio
    async def test_multi_generation_lineage(self, store, sample_record):
        """Test multiple generations of evolution."""
        # Generation 0: Original
        await store.save_record(sample_record)
        
        # Generation 1: First evolution
        gen1_record = SkillRecord(
            skill_id="gen1_skill_v1",
            name="gen1_skill",
            description="First generation evolution",
            category=sample_record.category,
            visibility=sample_record.visibility,
            path=sample_record.path,
            lineage=SkillLineage(
                origin=SkillOrigin.DERIVED,
                generation=1,
                parent_skill_ids=[sample_record.skill_id],
                change_summary="First evolution",
                content_snapshot={"skill.py": "def gen1():\n    return 1"},
            ),
            tool_dependencies=[],
            critical_tools=[],
            total_selections=0,
            total_applied=0,
            total_completions=0,
            total_fallbacks=0,
            recent_analyses=[],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
        )
        
        await store.evolve_skill(gen1_record, [sample_record.skill_id])
        gen1_id = gen1_record.skill_id
        
        # Generation 2: Second evolution
        gen2_record = SkillRecord(
            skill_id="gen2_skill_v1",
            name="gen2_skill",
            description="Second generation evolution",
            category=sample_record.category,
            visibility=sample_record.visibility,
            path=sample_record.path,
            lineage=SkillLineage(
                origin=SkillOrigin.DERIVED,
                generation=2,
                parent_skill_ids=[gen1_id],
                change_summary="Second evolution",
                content_snapshot={"skill.py": "def gen2():\n    return 2"},
            ),
            tool_dependencies=[],
            critical_tools=[],
            total_selections=0,
            total_applied=0,
            total_completions=0,
            total_fallbacks=0,
            recent_analyses=[],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
        )
        
        await store.evolve_skill(gen2_record, [gen1_id])
        gen2_id = gen2_record.skill_id
        
        # Verify generations
        gen2 = store.load_record(gen2_id)
        assert gen2.lineage.generation == 2
        
        # Check full ancestry
        ancestry = store.get_ancestry(gen2_id)
        assert len(ancestry) == 2  # gen1 + gen0 (ancestors only)
        assert ancestry[0].skill_id == sample_record.skill_id  # gen0 (oldest)
        assert ancestry[1].skill_id == gen1_id                # gen1 (newer)


class TestAnalysisWorkflow:
    """Test analysis workflow: save skill → record analysis → retrieve → evolve."""

    @pytest.mark.asyncio
    async def test_analysis_recording_and_retrieval(self, store, sample_record, sample_analysis):
        """Test recording and retrieving execution analyses."""
        # Save skill first
        await store.save_record(sample_record)
        
        # Record analysis
        await store.record_analysis(sample_analysis)
        
        # Load analysis back
        loaded_analysis = store.load_analyses_for_task(sample_analysis.task_id)
        assert loaded_analysis is not None
        assert loaded_analysis.task_id == sample_analysis.task_id
        assert loaded_analysis.execution_note == sample_analysis.execution_note
        assert len(loaded_analysis.skill_judgments) == 1
        
        # Load analyses for skill
        skill_analyses = store.load_analyses(sample_record.skill_id)
        assert len(skill_analyses) == 1
        assert skill_analyses[0].task_id == sample_analysis.task_id
        
        # Get all analyses
        all_analyses = store.load_all_analyses()
        assert len(all_analyses) == 1
        assert all_analyses[0].task_id == sample_analysis.task_id

    @pytest.mark.asyncio
    async def test_evolution_candidates(self, store, sample_record):
        """Test getting evolution candidates from analysis data."""
        # Save skill
        await store.save_record(sample_record)
        
        # Create analysis with failure and evolution suggestion
        failed_analysis = ExecutionAnalysis(
            task_id="failed_task",
            timestamp=datetime.now(),
            task_completed=False,  # Failed task
            execution_note="Skill failed to execute properly",
            skill_judgments=[
                SkillJudgment(
                    skill_id=sample_record.skill_id,
                    skill_applied=False,  # Failed to apply
                    note="Skill failed to execute",
                )
            ],
            evolution_suggestions=[
                EvolutionSuggestion(
                    evolution_type=EvolutionType.FIX,
                    target_skill_ids=[sample_record.skill_id],
                    direction="Fix skill execution issues",
                )
            ],
            analyzed_at=datetime.now(),
        )
        
        await store.record_analysis(failed_analysis)
        
        # Get evolution candidates (should include skills with failures)
        candidates = store.load_evolution_candidates()
        assert len(candidates) > 0
        
        # Should include our failed analysis
        task_ids = [c.task_id for c in candidates]
        assert failed_analysis.task_id in task_ids


class TestTagSearchWorkflow:
    """Test tag/search workflow: save with tags → search by tags → search by query."""

    @pytest.mark.asyncio
    async def test_tag_operations(self, store, sample_record):
        """Test tag synchronization and retrieval."""
        # Save skill
        await store.save_record(sample_record)
        
        # Add tags
        tags = ["python", "testing", "integration"]
        store.sync_tags(sample_record.skill_id, tags)
        
        # Get tags back
        loaded_tags = store.get_tags(sample_record.skill_id)
        assert set(loaded_tags) == set(tags)
        
        # Get all tags
        all_tags = store.get_all_tags()
        tag_names = [tag["tag"] for tag in all_tags]
        for tag in tags:
            assert tag in tag_names
        
        # Find by tags
        found_skills = store.find_skills_by_tags(["python"])
        assert sample_record.skill_id in found_skills
        
        found_skills = store.find_skills_by_tags(["python", "testing"], match_all=True)
        assert sample_record.skill_id in found_skills
        
        found_skills = store.find_skills_by_tags(["nonexistent"])
        assert sample_record.skill_id not in found_skills

    @pytest.mark.asyncio
    async def test_comprehensive_search(self, store):
        """Test comprehensive search functionality."""
        # Create skills with different attributes
        skills_data = [
            {
                "skill_id": "tool_skill_v1",
                "name": "tool_helper",
                "description": "Tool utility functions",
                "category": SkillCategory.TOOL_GUIDE,
                "tags": ["tool", "utility"]
            },
            {
                "skill_id": "workflow_skill_v1", 
                "name": "workflow_helper",
                "description": "Workflow helper functions",
                "category": SkillCategory.WORKFLOW,
                "tags": ["workflow", "utility"]
            },
            {
                "skill_id": "private_skill_v1",
                "name": "private_helper",
                "description": "Private internal tool",
                "category": SkillCategory.REFERENCE,
                "tags": ["internal"],
                "visibility": SkillVisibility.PRIVATE
            }
        ]
        
        for skill_data in skills_data:
            record = SkillRecord(
                skill_id=skill_data["skill_id"],
                name=skill_data["name"], 
                description=skill_data["description"],
                category=skill_data["category"],
                visibility=skill_data.get("visibility", SkillVisibility.PUBLIC),
                path=f"/test/{skill_data['name']}.py",
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={f"{skill_data['name']}.py": f"def {skill_data['name']}():\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            await store.save_record(record)
            store.sync_tags(skill_data["skill_id"], skill_data["tags"])
        
        # Search by text query
        results = store.search_skills("tool")
        assert len(results) >= 1
        skill_ids = [r["skill_id"] for r in results]
        assert "tool_skill_v1" in skill_ids
        
        # Search by category
        results = store.search_skills(category=SkillCategory.TOOL_GUIDE)
        assert len(results) == 1
        assert results[0]["skill_id"] == "tool_skill_v1"
        
        # Search by tags
        results = store.search_skills(tags=["utility"])
        assert len(results) == 2  # tool and workflow skills
        
        # Search with visibility filter
        results = store.search_skills(visibility=SkillVisibility.PUBLIC)
        skill_ids = [r["skill_id"] for r in results]
        assert "private_skill_v1" not in skill_ids
        
        # Comprehensive search with multiple filters
        results = store.search_skills(
            query="utility",
            tags=["tool"],
            category=SkillCategory.TOOL_GUIDE,
            limit=10
        )
        assert len(results) == 1
        assert results[0]["skill_id"] == "tool_skill_v1"


class TestMigrationWorkflow:
    """Test migration workflow: fresh DB → schema creation → all modules work."""

    def test_fresh_database_initialization(self, temp_db):
        """Test that a fresh database is properly initialized."""
        # Create store which should initialize schema
        store = SkillStore(temp_db)
        
        try:
            # Verify schema version
            version = store.get_schema_version()
            assert version == 1
            
            # Verify we can perform basic operations
            stats = store.get_stats()
            assert stats["total_skills"] == 0
            assert stats["total_analyses"] == 0
            
            # Test that all module operations work
            all_skills = store.load_all()
            assert len(all_skills) == 0
            
            all_tags = store.get_all_tags()
            assert len(all_tags) == 0
            
        finally:
            store.close()

    def test_schema_migration_methods(self, store):
        """Test schema migration facade methods."""
        # Test version operations
        current_version = store.get_schema_version()
        assert current_version == 1
        
        # Test ensure current schema (should be idempotent)
        store.ensure_current_schema()
        assert store.get_schema_version() == 1
        
        # Test migration method exists
        store.migrate_to_version(1)  # Should be no-op
        assert store.get_schema_version() == 1


class TestCrossModuleWorkflow:
    """Test cross-module workflow: save → tag → analyze → evolve → comprehensive retrieval."""

    @pytest.mark.asyncio
    async def test_complete_skill_workflow(self, store, sample_record):
        """Test a complete workflow touching all modules."""
        # 1. REPOSITORY: Save initial skill
        await store.save_record(sample_record)
        
        # 2. TAG SEARCH: Add tags
        tags = ["python", "testing", "v1"]
        store.sync_tags(sample_record.skill_id, tags)
        
        # 3. ANALYSIS STORE: Record successful execution
        analysis = ExecutionAnalysis(
            task_id="workflow_test_123",
            timestamp=datetime.now(),
            task_completed=True,
            execution_note="Executed successfully in workflow test",
            skill_judgments=[
                SkillJudgment(
                    skill_id=sample_record.skill_id,
                    skill_applied=True,
                    note="Executed successfully in workflow test",
                )
            ],
            analyzed_at=datetime.now(),
        )
        await store.record_analysis(analysis)
        
        # 4. LINEAGE TRACKER: Evolve the skill
        evolved_record = SkillRecord(
            skill_id="evolved_workflow_skill_v1",
            name="evolved_workflow_skill",
            description="Enhanced workflow skill for testing",
            category=sample_record.category,
            visibility=sample_record.visibility,
            path=sample_record.path,
            lineage=SkillLineage(
                origin=SkillOrigin.DERIVED,
                generation=1,
                parent_skill_ids=[sample_record.skill_id],
                change_summary="Enhanced for workflow testing",
                content_snapshot={"skill.py": "def evolved_workflow_skill():\n    return 'enhanced'"},
            ),
            tool_dependencies=[],
            critical_tools=[],
            total_selections=0,
            total_applied=0,
            total_completions=0,
            total_fallbacks=0,
            recent_analyses=[],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
        )
        
        await store.evolve_skill(evolved_record, [sample_record.skill_id])
        evolved_id = evolved_record.skill_id
        
        # 5. COMPREHENSIVE VERIFICATION through facade
        
        # Repository operations
        original = store.load_record(sample_record.skill_id)
        evolved = store.load_record(evolved_id)
        assert original is not None
        assert evolved is not None
        assert evolved.lineage.generation == 1
        
        # Tag operations
        original_tags = store.get_tags(sample_record.skill_id)
        assert set(original_tags) == set(tags)
        
        # Search operations
        found_by_tag = store.find_skills_by_tags(["python"])
        assert sample_record.skill_id in found_by_tag
        
        search_results = store.search_skills("testing")
        skill_ids = [r["skill_id"] for r in search_results]
        assert sample_record.skill_id in skill_ids
        
        # Analysis operations
        skill_analyses = store.load_analyses(sample_record.skill_id)
        assert len(skill_analyses) == 1
        assert skill_analyses[0].task_id == analysis.task_id
        
        # Lineage operations
        children = store.find_children(sample_record.skill_id)
        assert evolved_id in children
        
        ancestry = store.get_ancestry(evolved_id)
        assert len(ancestry) == 1  # just the parent
        
        # Summary stats should reflect all data
        stats = store.get_stats()
        assert stats["total_skills"] == 2  # original + evolved
        assert stats["total_analyses"] == 1


class TestConcurrentAccess:
    """Test concurrent access to SkillStore."""

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, store, sample_record):
        """Test that concurrent operations don't interfere."""
        # Save initial skill
        await store.save_record(sample_record)
        
        async def reader_task(worker_id: int, iterations: int):
            """Read operations in concurrent task."""
            results = []
            for i in range(iterations):
                # Mix of read operations
                record = store.load_record(sample_record.skill_id)
                all_skills = store.load_all()
                stats = store.get_stats()
                results.append({
                    'worker': worker_id,
                    'iteration': i,
                    'record_found': record is not None,
                    'total_skills': len(all_skills),
                    'stats_count': stats.get('total_skills', 0)
                })
            return results
        
        async def writer_task(worker_id: int, iterations: int):
            """Write operations in concurrent task."""
            for i in range(iterations):
                # Create unique records
                record = SkillRecord(
                    skill_id=f"concurrent_skill_w{worker_id}_i{i}",
                    name=f"concurrent_skill_{worker_id}_{i}",
                    description=f"Concurrent test skill worker {worker_id} iteration {i}",
                    category=SkillCategory.TOOL_GUIDE,
                    visibility=SkillVisibility.PUBLIC,
                    path=f"/test/concurrent_{worker_id}_{i}.py",
                    lineage=SkillLineage(
                        origin=SkillOrigin.IMPORTED,
                        generation=0,
                        content_snapshot={f"concurrent_{worker_id}_{i}.py": f"def concurrent_skill_{worker_id}_{i}():\n    pass"},
                    ),
                    tool_dependencies=[],
                    critical_tools=[],
                    total_selections=0,
                    total_applied=0,
                    total_completions=0,
                    total_fallbacks=0,
                    recent_analyses=[],
                    first_seen=datetime.now(),
                    last_updated=datetime.now(),
                )
                await store.save_record(record)
        
        # Run concurrent readers and writers
        num_readers = 3
        num_writers = 2
        iterations = 5
        
        tasks = []
        
        # Start reader tasks
        for i in range(num_readers):
            tasks.append(reader_task(i, iterations))
        
        # Start writer tasks  
        for i in range(num_writers):
            tasks.append(writer_task(i, iterations))
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent task failed: {result}")
        
        # Verify final state
        final_count = store.count()
        expected_min = 1 + (num_writers * iterations)  # initial + written records
        assert final_count >= expected_min

    def test_concurrent_threading(self, store, sample_record):
        """Test concurrent access from multiple threads using threading."""
        # Save initial skill synchronously
        asyncio.run(store.save_record(sample_record))
        
        results = {}
        errors = []
        
        def reader_thread(thread_id: int):
            """Thread function for read operations."""
            try:
                for i in range(10):
                    record = store.load_record(sample_record.skill_id)
                    stats = store.get_stats()
                    results[f"reader_{thread_id}_{i}"] = {
                        'found': record is not None,
                        'stats': stats.get('total_skills', 0)
                    }
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Reader {thread_id}: {e}")
        
        def writer_thread(thread_id: int):
            """Thread function for write operations."""
            try:
                async def write_operations():
                    for i in range(5):
                        record = SkillRecord(
                            skill_id=f"thread_skill_{thread_id}_{i}",
                            name=f"thread_skill_{thread_id}_{i}",
                            description=f"Thread test skill {thread_id}-{i}",
                            category=SkillCategory.TOOL_GUIDE,
                            visibility=SkillVisibility.PUBLIC,
                            path=f"/test/thread_{thread_id}_{i}.py",
                            lineage=SkillLineage(
                                origin=SkillOrigin.IMPORTED,
                                generation=0,
                                content_snapshot={f"thread_{thread_id}_{i}.py": f"def thread_skill_{thread_id}_{i}():\n    pass"},
                            ),
                            tool_dependencies=[],
                            critical_tools=[],
                            total_selections=0,
                            total_applied=0,
                            total_completions=0,
                            total_fallbacks=0,
                            recent_analyses=[],
                            first_seen=datetime.now(),
                            last_updated=datetime.now(),
                        )
                        await store.save_record(record)
                
                # Run async operations in thread
                asyncio.run(write_operations())
                    
            except Exception as e:
                errors.append(f"Writer {thread_id}: {e}")
        
        # Start threads
        threads = []
        
        # Start 2 reader threads
        for i in range(2):
            thread = threading.Thread(target=reader_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Start 1 writer thread
        writer = threading.Thread(target=writer_thread, args=(0,))
        threads.append(writer)
        writer.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                pytest.fail("Thread did not complete within timeout")
        
        # Check for errors
        if errors:
            pytest.fail(f"Concurrent threading errors: {errors}")
        
        # Verify some operations succeeded
        assert len(results) > 0
        
        # Verify final database state
        final_count = store.count()
        assert final_count >= 1  # At least the initial record


class TestEdgeCases:
    """Test edge cases and error conditions in integrated workflows."""

    @pytest.mark.asyncio
    async def test_nonexistent_skill_operations(self, store):
        """Test operations on nonexistent skills."""
        nonexistent_id = "nonexistent_skill_v1"
        
        # Load nonexistent
        record = store.load_record(nonexistent_id)
        assert record is None
        
        # Deactivate nonexistent
        result = await store.deactivate_record(nonexistent_id)
        assert result is False
        
        # Delete nonexistent
        result = await store.delete_record(nonexistent_id)
        assert result is False
        
        # Get tags for nonexistent
        tags = store.get_tags(nonexistent_id)
        assert tags == []
        
        # Find children of nonexistent
        children = store.find_children(nonexistent_id)
        assert children == []

    @pytest.mark.asyncio
    async def test_empty_database_operations(self, store):
        """Test operations on empty database."""
        # Load operations on empty DB
        all_skills = store.load_all()
        assert len(all_skills) == 0
        
        active_skills = store.load_active()
        assert len(active_skills) == 0
        
        # Search operations on empty DB
        results = store.search_skills("anything")
        assert len(results) == 0
        
        found = store.find_skills_by_tags(["any", "tags"])
        assert len(found) == 0
        
        # Analysis operations on empty DB
        analyses = store.load_all_analyses()
        assert len(analyses) == 0
        
        candidates = store.load_evolution_candidates()
        assert len(candidates) == 0
        
        # Stats on empty DB
        stats = store.get_stats()
        assert stats["total_skills"] == 0
        assert stats["total_analyses"] == 0
        
        count = store.count()
        assert count == 0


class TestExternalReaderVisibility:
    """Test that data committed by one SkillStore instance is visible to a SEPARATE SkillStore instance (external WAL reader)."""

    @pytest.mark.asyncio
    async def test_save_visible_to_external_reader(self, sample_record):
        """Store A saves a record, Store B (same db_path, separate instance) can load it."""
        db_path = tempfile.mktemp(suffix='.db')
        store_a = None
        store_b = None
        
        try:
            # Store A saves a record
            store_a = SkillStore(db_path)
            await store_a.save_record(sample_record)
            
            # Store B (separate instance) should see it
            store_b = SkillStore(db_path)
            loaded_record = store_b.load_record(sample_record.skill_id)
            
            assert loaded_record is not None
            assert loaded_record.skill_id == sample_record.skill_id
            assert loaded_record.name == sample_record.name
            
        finally:
            if store_a:
                try:
                    store_a.close()
                except Exception:
                    pass
            if store_b:
                try:
                    store_b.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_sync_tags_visible_to_external_reader(self, sample_record):
        """Store A saves + sync_tags, Store B sees the tags."""
        db_path = tempfile.mktemp(suffix='.db')
        store_a = None
        store_b = None
        
        try:
            # Store A saves a record and syncs tags
            store_a = SkillStore(db_path)
            await store_a.save_record(sample_record)
            store_a.sync_tags(sample_record.skill_id, ["test-tag", "integration-tag"])
            
            # Store B should see the tags
            store_b = SkillStore(db_path)
            tags = store_b.get_tags(sample_record.skill_id)
            
            assert "test-tag" in tags
            assert "integration-tag" in tags
            
            # And should be able to find the skill by tags
            found_skills = store_b.find_skills_by_tags(["test-tag"])
            assert len(found_skills) == 1
            assert found_skills[0] == sample_record.skill_id
            
        finally:
            if store_a:
                try:
                    store_a.close()
                except Exception:
                    pass
            if store_b:
                try:
                    store_b.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_evolve_visible_to_external_reader(self, sample_record, sample_analysis):
        """Store A does save → sync_tags → record_analysis → evolve, Store B can load the evolved record."""
        db_path = tempfile.mktemp(suffix='.db')
        store_a = None
        store_b = None
        
        try:
            # Store A does full workflow
            store_a = SkillStore(db_path)
            await store_a.save_record(sample_record)
            store_a.sync_tags(sample_record.skill_id, ["evolution-test"])
            await store_a.record_analysis(sample_analysis)
            
            # Create evolved record
            evolved_record = SkillRecord(
                skill_id="evolved_external_test_v1",
                name="evolved_external_test",
                description="Evolved record for external reader visibility test",
                category=sample_record.category,
                visibility=sample_record.visibility,
                path=sample_record.path,
                lineage=SkillLineage(
                    origin=SkillOrigin.DERIVED,
                    generation=1,
                    parent_skill_ids=[sample_record.skill_id],
                    content_snapshot={"skill.py": "def evolved_test_skill():\n    try:\n        pass\n    except Exception:\n        pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            await store_a.evolve_skill(evolved_record, [sample_record.skill_id])
            
            # Store B should see the evolved record
            store_b = SkillStore(db_path)
            loaded_evolved = store_b.load_record(evolved_record.skill_id)
            
            assert loaded_evolved is not None
            assert loaded_evolved.skill_id == evolved_record.skill_id
            assert loaded_evolved.lineage.generation == 1
            assert loaded_evolved.lineage.parent_skill_ids == [sample_record.skill_id]
            
            # And should see the parent relationship
            children = store_b.find_children(sample_record.skill_id)
            assert len(children) == 1
            assert children[0] == evolved_record.skill_id
            
        finally:
            if store_a:
                try:
                    store_a.close()
                except Exception:
                    pass
            if store_b:
                try:
                    store_b.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_full_workflow_external_parity(self, sample_record, sample_analysis):
        """Store A does save → sync_tags → record_analysis → evolve → deactivate parent → reactivate parent. Store B sees ALL state changes."""
        db_path = tempfile.mktemp(suffix='.db')
        store_a = None
        store_b = None
        
        try:
            # Store A does full workflow
            store_a = SkillStore(db_path)
            await store_a.save_record(sample_record)
            store_a.sync_tags(sample_record.skill_id, ["workflow-test", "complete"])
            await store_a.record_analysis(sample_analysis)
            
            # Create evolved record
            evolved_record = SkillRecord(
                skill_id="evolved_workflow_test_v1",
                name="evolved_workflow_test",
                description="Evolved record for full workflow test",
                category=sample_record.category,
                visibility=sample_record.visibility,
                path=sample_record.path,
                lineage=SkillLineage(
                    origin=SkillOrigin.DERIVED,
                    generation=1,
                    parent_skill_ids=[sample_record.skill_id],
                    content_snapshot={"skill.py": "import logging\ndef test_skill():\n    logging.info('test')\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            await store_a.evolve_skill(evolved_record, [sample_record.skill_id])
            
            # Deactivate parent
            deactivate_result = await store_a.deactivate_record(sample_record.skill_id)
            assert deactivate_result is True
            
            # Reactivate parent
            reactivate_result = await store_a.reactivate_record(sample_record.skill_id)
            assert reactivate_result is True
            
            # Store B should see ALL state changes
            store_b = SkillStore(db_path)
            
            # Original record should be active
            original = store_b.load_record(sample_record.skill_id)
            assert original is not None
            assert original.is_active is True
            
            # Evolved record should exist
            evolved = store_b.load_record(evolved_record.skill_id)
            assert evolved is not None
            
            # Tags should be visible
            tags = store_b.get_tags(sample_record.skill_id)
            assert "workflow-test" in tags
            assert "complete" in tags
            
            # Analysis should be visible
            analyses = store_b.load_analyses(sample_record.skill_id)
            assert len(analyses) == 1
            assert analyses[0].task_id == sample_analysis.task_id
            
            # Lineage should be intact
            children = store_b.find_children(sample_record.skill_id)
            assert len(children) == 1
            assert children[0] == evolved.skill_id
            
        finally:
            if store_a:
                try:
                    store_a.close()
                except Exception:
                    pass
            if store_b:
                try:
                    store_b.close()
                except Exception:
                    pass
            gc.collect()


class TestRollbackRecovery:
    """Test that failed operations don't poison the connection for subsequent operations."""

    @pytest.mark.asyncio
    async def test_failed_save_doesnt_poison_connection(self, temp_db):
        """Force a save failure (invalid record), then do a successful save. The successful save must be visible."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            
            # Create an invalid record (missing required fields)
            invalid_record = SkillRecord(
                skill_id="invalid_skill_v1",
                name="",  # Empty name should cause validation error
                description="Test invalid record",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="",  # Empty path should cause validation error
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={},  # Empty snapshot
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            # Try to save invalid record - should fail
            with pytest.raises(Exception):
                await store.save_record(invalid_record)
            
            # Now save a valid record - should succeed
            valid_record = SkillRecord(
                skill_id="valid_after_failure_v1",
                name="valid_after_failure",
                description="Valid record after failed save",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/test/valid_after_failure.py",
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={"skill.py": "def valid_skill():\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            # This should succeed
            await store.save_record(valid_record)
            
            # Verify it's actually saved and visible
            loaded = store.load_record(valid_record.skill_id)
            assert loaded is not None
            assert loaded.skill_id == valid_record.skill_id
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_failed_evolve_doesnt_poison_connection(self, temp_db, sample_record):
        """Force an evolve failure (missing parent), then do a successful evolve. Must work."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            
            # Save a valid parent record first
            await store.save_record(sample_record)
            
            # Try to evolve from a non-existent parent - should fail because of lineage validation
            invalid_evolved_record = SkillRecord(
                skill_id="",  # Empty skill_id should cause validation failure
                name="invalid_evolved",
                description="Invalid evolved record",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/test/invalid_evolved.py",
                lineage=SkillLineage(
                    origin=SkillOrigin.DERIVED,
                    generation=1,
                    parent_skill_ids=["nonexistent_parent_v1"],
                    content_snapshot={"skill.py": "def invalid_evolved():\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            with pytest.raises(Exception):
                await store.evolve_skill(invalid_evolved_record, ["nonexistent_parent_v1"])
            
            # Now do a successful evolution - should work
            valid_evolved_record = SkillRecord(
                skill_id="valid_evolved_after_failure_v1",
                name="valid_evolved_after_failure",
                description="Valid evolved record after failure",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/test/valid_evolved_after_failure.py",
                lineage=SkillLineage(
                    origin=SkillOrigin.DERIVED,
                    generation=1,
                    parent_skill_ids=[sample_record.skill_id],
                    content_snapshot={"skill.py": "def valid_evolved():\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            await store.evolve_skill(valid_evolved_record, [sample_record.skill_id])
            
            # Verify the evolution worked
            evolved = store.load_record(valid_evolved_record.skill_id)
            assert evolved is not None
            assert evolved.lineage.parent_skill_ids == [sample_record.skill_id]
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_savepoint_rollback_isolation(self, temp_db, sample_record, sample_analysis):
        """Within a workflow, one step fails but prior committed steps survive. Specifically: save succeeds, record_analysis succeeds, evolve fails (bad record) → analysis and save are still visible, evolved record is NOT."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            
            # Step 1: Save - should succeed
            await store.save_record(sample_record)
            
            # Step 2: Record analysis - should succeed  
            await store.record_analysis(sample_analysis)
            
            # Verify both are committed and visible
            saved_record = store.load_record(sample_record.skill_id)
            assert saved_record is not None
            
            analyses = store.load_analyses(sample_record.skill_id)
            assert len(analyses) == 1
            assert analyses[0].task_id == sample_analysis.task_id
            
            # Step 3: Evolve with invalid data - should fail
            invalid_evolved_record = SkillRecord(
                skill_id="",  # Empty skill_id should cause validation failure
                name="invalid_evolved_isolation",
                description="Invalid evolved record for isolation test",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/test/invalid_evolved_isolation.py",
                lineage=SkillLineage(
                    origin=SkillOrigin.DERIVED,
                    generation=1,
                    parent_skill_ids=[sample_record.skill_id],
                    content_snapshot={"skill.py": "def invalid_evolved():\n    pass"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            with pytest.raises(Exception):
                await store.evolve_skill(invalid_evolved_record, [sample_record.skill_id])
            
            # Verify the previous steps are still intact (savepoint isolation worked)
            record_after_failure = store.load_record(sample_record.skill_id)
            assert record_after_failure is not None
            assert record_after_failure.skill_id == sample_record.skill_id
            
            analyses_after_failure = store.load_analyses(sample_record.skill_id)
            assert len(analyses_after_failure) == 1
            assert analyses_after_failure[0].task_id == sample_analysis.task_id
            
            # Verify no evolved record was created
            children = store.find_children(sample_record.skill_id)
            assert len(children) == 0
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()


class TestSavepointNesting:
    """Test the SAVEPOINT pattern works for nested transaction scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_operations_in_sequence(self, temp_db, sample_record):
        """Test multiple operations work correctly in sequence, ensuring SAVEPOINT pattern handles internal nesting."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            await store.save_record(sample_record)
            
            # Perform multiple operations that each internally use savepoints
            # This tests that savepoints can be nested without issues
            
            # 1. Deactivate
            result1 = await store.deactivate_record(sample_record.skill_id)
            assert result1 is True
            
            # 2. Reactivate (this should work after deactivate)
            result2 = await store.reactivate_record(sample_record.skill_id)
            assert result2 is True
            
            # 3. Deactivate again
            result3 = await store.deactivate_record(sample_record.skill_id)
            assert result3 is True
            
            # Verify final state
            record = store.load_record(sample_record.skill_id)
            assert record is not None
            assert record.is_active is False
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_concurrent_count_during_write(self, temp_db, sample_record):
        """Call count() from one thread while save_record is executing on another. Verify no crash and count returns a valid number."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            
            # Create a barrier to coordinate timing
            barrier = threading.Barrier(2)
            results = {}
            exceptions = {}
            
            def count_worker():
                """Worker that calls count() during a save operation."""
                try:
                    barrier.wait()  # Wait for save to start
                    # Multiple counts to increase chance of hitting the save operation
                    for i in range(10):
                        count = store.count()
                        results[f'count_{i}'] = count
                        time.sleep(0.01)  # Small delay
                except Exception as e:
                    exceptions['count_worker'] = e
            
            async def save_worker():
                """Worker that does a slow save operation."""
                try:
                    barrier.wait()  # Signal that save is starting
                    # Create multiple records with small delays to make a "slow" save
                    for i in range(5):
                        record = SkillRecord(
                            skill_id=f"concurrent_save_test_{i}_v1",
                            name=f"concurrent_save_test_{i}",
                            description=f"Concurrent save test record {i}",
                            category=SkillCategory.TOOL_GUIDE,
                            visibility=SkillVisibility.PUBLIC,
                            path=f"/test/concurrent_{i}.py",
                            lineage=SkillLineage(
                                origin=SkillOrigin.IMPORTED,
                                generation=0,
                                content_snapshot={f"concurrent_{i}.py": f"def concurrent_{i}():\n    pass"},
                            ),
                            tool_dependencies=[],
                            critical_tools=[],
                            total_selections=0,
                            total_applied=0,
                            total_completions=0,
                            total_fallbacks=0,
                            recent_analyses=[],
                            first_seen=datetime.now(),
                            last_updated=datetime.now(),
                        )
                        await store.save_record(record)
                        await asyncio.sleep(0.01)  # Small async delay
                except Exception as e:
                    exceptions['save_worker'] = e
            
            # Start count worker in thread
            count_thread = threading.Thread(target=count_worker)
            count_thread.start()
            
            # Start save worker
            await save_worker()
            
            # Wait for count worker to finish
            count_thread.join()
            
            # Verify no exceptions occurred
            if exceptions:
                pytest.fail(f"Concurrent operations failed: {exceptions}")
            
            # Verify count results are valid (non-negative integers)
            for key, count in results.items():
                assert isinstance(count, int)
                assert count >= 0
            
            # Verify final state is consistent
            final_count = store.count()
            assert final_count >= 5  # At least the 5 records we saved
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()


class TestLikeWildcardEscape:
    """Test LIKE wildcard injection is prevented."""

    @pytest.mark.asyncio
    async def test_load_record_by_path_escapes_percent(self, temp_db):
        """Save a record with path `/skills/100%_complete/SKILL.md`, verify `load_record_by_path("/skills/100")` does NOT match it (the % is literal, not a wildcard)."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            
            # Create record with % in path  
            record_with_percent = SkillRecord(
                skill_id="percent_path_test_v1",
                name="percent_path_test",
                description="Test record with % in path",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/skills/100%_complete/SKILL.md",  # % should be treated literally
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={"SKILL.md": "# Skill with % in path"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            await store.save_record(record_with_percent)
            
            # Also create a record that would match if % was treated as wildcard
            record_normal = SkillRecord(
                skill_id="normal_path_test_v1", 
                name="normal_path_test",
                description="Test record with normal path",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/skills/100x_complete/SKILL.md",  # Would match "/skills/100" + % wildcard
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={"SKILL.md": "# Normal skill"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            await store.save_record(record_normal)
            
            # Search with path prefix - should NOT match the % record if % is escaped
            match = store.load_record_by_path("/skills/100")
            
            # If there's a match, it should NOT be the %_complete record
            if match:
                assert match.path != "/skills/100%_complete/SKILL.md", "% was treated as wildcard - security issue!"
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()

    @pytest.mark.asyncio
    async def test_load_record_by_path_escapes_underscore(self, temp_db):
        """Save records with paths `/skills/a_b/SKILL.md` and `/skills/axb/SKILL.md`, verify `load_record_by_path("/skills/a_b")` only matches the first."""
        store = None
        
        try:
            store = SkillStore(temp_db)
            
            # Record with literal underscore
            record_underscore = SkillRecord(
                skill_id="underscore_path_test_v1",
                name="underscore_path_test", 
                description="Test record with _ in path",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/skills/a_b/SKILL.md",  # _ should be treated literally
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={"SKILL.md": "# Skill with _ in path"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            # Record that would match if _ was treated as single-char wildcard
            record_would_match = SkillRecord(
                skill_id="would_match_test_v1",
                name="would_match_test",
                description="Record that would match if _ is wildcard",
                category=SkillCategory.TOOL_GUIDE,
                visibility=SkillVisibility.PUBLIC,
                path="/skills/axb/SKILL.md",  # Would match "a_b" if _ is wildcard
                lineage=SkillLineage(
                    origin=SkillOrigin.IMPORTED,
                    generation=0,
                    content_snapshot={"SKILL.md": "# Would match if _ is wildcard"},
                ),
                tool_dependencies=[],
                critical_tools=[],
                total_selections=0,
                total_applied=0,
                total_completions=0,
                total_fallbacks=0,
                recent_analyses=[],
                first_seen=datetime.now(),
                last_updated=datetime.now(),
            )
            
            await store.save_record(record_underscore)
            await store.save_record(record_would_match)
            
            # Search for exact underscore path
            match = store.load_record_by_path("/skills/a_b")
            
            if match:
                # Should match the literal underscore path
                assert match.path == "/skills/a_b/SKILL.md", "Literal underscore path should match"
                
                # Verify we didn't get the "axb" path (which would mean _ was treated as wildcard)
                assert match.path != "/skills/axb/SKILL.md", "_ was treated as wildcard - security issue!"
            
        finally:
            if store:
                try:
                    store.close()
                except Exception:
                    pass
            gc.collect()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])