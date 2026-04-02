"""Tests for MigrationManager (Epic 3.6) — schema creation and migration."""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from openspace.skill_engine.migration_manager import MigrationManager, CURRENT_VERSION


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database file for testing."""
    return tmp_path / "test_migration_manager.db"


@pytest.fixture
def migration_manager(temp_db_path):
    """MigrationManager instance for testing."""
    manager = MigrationManager(db_path=temp_db_path)
    yield manager
    manager.close()


@pytest.fixture
def shared_connection(temp_db_path):
    """Shared SQLite connection for embedding tests."""
    conn = sqlite3.connect(str(temp_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class TestMigrationManagerConstructor:
    """Test MigrationManager construction and initialization."""

    def test_standalone_construction(self, temp_db_path):
        """Test creating MigrationManager with db_path."""
        manager = MigrationManager(db_path=temp_db_path)
        
        assert manager._db_path == temp_db_path
        assert manager._owns_conn is True
        assert manager._closed is False
        
        manager.close()
        assert manager._closed is True

    def test_embedded_construction(self, shared_connection):
        """Test creating MigrationManager with shared connection."""
        import threading
        lock = threading.Lock()
        
        manager = MigrationManager(conn=shared_connection, lock=lock)
        
        assert manager._db_path == Path(":shared:")
        assert manager._owns_conn is False
        assert manager._closed is False
        assert manager._mu is lock
        
        manager.close()
        # Should not close shared connection
        assert not manager._closed  # close() is no-op when we don't own conn

    def test_invalid_construction(self):
        """Test that construction fails without db_path or conn."""
        with pytest.raises(ValueError, match="Either db_path or conn must be provided"):
            MigrationManager()

    def test_ensure_open_after_close(self, migration_manager):
        """Test that operations fail after close."""
        migration_manager.close()
        
        with pytest.raises(RuntimeError, match="MigrationManager is closed"):
            migration_manager.initialize_schema()


class TestSchemaInitialization:
    """Test schema creation and initialization."""

    def test_initialize_schema_creates_tables(self, migration_manager):
        """Test that initialize_schema creates all required tables."""
        migration_manager.initialize_schema()
        
        # Query the database to verify tables were created
        with migration_manager._conn:
            cursor = migration_manager._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'skill_records',
            'skill_lineage_parents',
            'execution_analyses',
            'skill_judgments',
            'skill_tool_deps',
            'skill_tags'
        ]
        
        for table in expected_tables:
            assert table in tables

    def test_initialize_schema_creates_indexes(self, migration_manager):
        """Test that initialize_schema creates all required indexes."""
        migration_manager.initialize_schema()
        
        # Query the database to verify indexes were created
        with migration_manager._conn:
            cursor = migration_manager._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
        
        expected_indexes = [
            'idx_sr_category',
            'idx_sr_updated',
            'idx_sr_active',
            'idx_sr_name',
            'idx_lp_parent',
            'idx_ea_task',
            'idx_ea_ts',
            'idx_sj_skill',
            'idx_sj_analysis',
            'idx_td_tool'
        ]
        
        for index in expected_indexes:
            assert index in indexes

    def test_initialize_schema_idempotent(self, migration_manager):
        """Test that initialize_schema is safe to call multiple times."""
        # Should not raise any errors
        migration_manager.initialize_schema()
        migration_manager.initialize_schema()
        migration_manager.initialize_schema()
        
        # Tables should still exist and be functional
        with migration_manager._conn:
            cursor = migration_manager._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
        assert 'skill_records' in tables

    def test_pragmas_set_correctly(self, migration_manager):
        """Test that PRAGMA settings are applied correctly."""
        # Check that WAL mode is enabled
        cursor = migration_manager._conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.upper() == "WAL"
        
        # Check that foreign keys are enabled
        cursor = migration_manager._conn.execute("PRAGMA foreign_keys")
        foreign_keys = cursor.fetchone()[0]
        assert foreign_keys == 1

    def test_schema_functional_after_init(self, migration_manager):
        """Test that tables are functional after initialization."""
        migration_manager.initialize_schema()
        
        # Try inserting test data to verify schema works
        with migration_manager._conn:
            migration_manager._conn.execute("""
                INSERT INTO skill_records (
                    skill_id, name, first_seen, last_updated, lineage_created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, ("test_123", "Test Skill", "2024-01-01", "2024-01-01", "2024-01-01"))
            
            # Verify the insert worked
            cursor = migration_manager._conn.execute(
                "SELECT name FROM skill_records WHERE skill_id = ?", 
                ("test_123",)
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "Test Skill"


class TestSchemaVersioning:
    """Test schema version management."""

    def test_initial_schema_version_is_zero(self, migration_manager):
        """Test that new database starts with version 0."""
        version = migration_manager.get_schema_version()
        assert version == 0

    def test_set_and_get_schema_version(self, migration_manager):
        """Test setting and retrieving schema versions."""
        migration_manager.set_schema_version(5)
        version = migration_manager.get_schema_version()
        assert version == 5
        
        migration_manager.set_schema_version(10)
        version = migration_manager.get_schema_version()
        assert version == 10

    def test_set_negative_version_fails(self, migration_manager):
        """Test that setting negative version raises error."""
        with pytest.raises(ValueError, match="version must be non-negative"):
            migration_manager.set_schema_version(-1)

    def test_version_persists_across_connections(self, temp_db_path):
        """Test that schema version persists when reopening database."""
        # Set version in first instance
        manager1 = MigrationManager(db_path=temp_db_path)
        manager1.initialize_schema()  # Create tables first
        manager1.set_schema_version(7)
        manager1.close()
        
        # Verify version persists in second instance
        manager2 = MigrationManager(db_path=temp_db_path)
        version = manager2.get_schema_version()
        assert version == 7
        manager2.close()


class TestSchemaMigration:
    """Test schema migration functionality."""

    def test_migrate_from_zero_to_one(self, migration_manager):
        """Test migration from version 0 to 1."""
        # Start at version 0
        assert migration_manager.get_schema_version() == 0
        
        # Migrate to version 1
        migration_manager.migrate_to_version(1)
        
        # Should initialize schema and set version
        assert migration_manager.get_schema_version() == 1
        
        # Tables should exist
        cursor = migration_manager._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert 'skill_records' in tables

    def test_migrate_same_version_is_noop(self, migration_manager):
        """Test that migrating to current version is no-op."""
        migration_manager.set_schema_version(3)
        
        # Should not raise error and version should remain unchanged
        migration_manager.migrate_to_version(3)
        assert migration_manager.get_schema_version() == 3

    def test_migrate_to_lower_version_fails(self, migration_manager):
        """Test that downgrading version is not allowed."""
        migration_manager.set_schema_version(5)
        
        with pytest.raises(ValueError, match="Cannot downgrade from version 5 to 3"):
            migration_manager.migrate_to_version(3)

    def test_unsupported_migration_path_fails(self, migration_manager):
        """Test that unsupported migration paths raise errors."""
        migration_manager.set_schema_version(2)
        
        with pytest.raises(RuntimeError, match="Migration from version 2 to 5 not supported"):
            migration_manager.migrate_to_version(5)

    def test_ensure_current_schema_migrates_when_needed(self, migration_manager):
        """Test that ensure_current_schema auto-migrates."""
        # Start at version 0
        assert migration_manager.get_schema_version() == 0
        
        # Should auto-migrate to version 1
        with patch.object(migration_manager, 'migrate_to_version') as mock_migrate:
            migration_manager.ensure_current_schema(1)
            mock_migrate.assert_called_once_with(1)

    def test_ensure_current_schema_handles_newer_version(self, migration_manager):
        """Test warning when database is newer than expected."""
        migration_manager.set_schema_version(5)
        
        # Should log warning but not fail
        with patch('openspace.skill_engine.migration_manager.logger') as mock_logger:
            migration_manager.ensure_current_schema(3)
            mock_logger.warning.assert_called_once_with(
                "Database schema version 5 is newer than expected 3"
            )

    def test_ensure_current_schema_defaults_to_version_one(self, migration_manager):
        """Test that ensure_current_schema defaults to version 1."""
        with patch.object(migration_manager, 'migrate_to_version') as mock_migrate:
            migration_manager.ensure_current_schema()
            mock_migrate.assert_called_once_with(1)


class TestEmbeddedMode:
    """Test MigrationManager when embedded in SkillStore."""

    def test_embedded_shared_connection(self, shared_connection):
        """Test that embedded mode uses shared connection."""
        import threading
        lock = threading.Lock()
        
        manager = MigrationManager(conn=shared_connection, lock=lock)
        manager.initialize_schema()
        
        # Should use the shared connection
        assert manager._conn is shared_connection
        
        # Verify tables were created on shared connection
        cursor = shared_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert 'skill_records' in tables

    def test_embedded_uses_shared_lock(self, shared_connection):
        """Test that embedded mode uses shared lock."""
        import threading
        lock = threading.Lock()
        
        manager = MigrationManager(conn=shared_connection, lock=lock)
        
        # Should use the provided lock
        assert manager._mu is lock

    def test_embedded_close_does_not_close_shared_connection(self, shared_connection):
        """Test that close() doesn't close shared connection."""
        import threading
        lock = threading.Lock()
        
        manager = MigrationManager(conn=shared_connection, lock=lock)
        manager.close()
        
        # Shared connection should still be usable
        cursor = shared_connection.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_operations_fail_when_closed(self, migration_manager):
        """Test that operations fail after manager is closed."""
        migration_manager.close()
        
        operations = [
            lambda: migration_manager.initialize_schema(),
            lambda: migration_manager.get_schema_version(),
            lambda: migration_manager.set_schema_version(1),
            lambda: migration_manager.migrate_to_version(1),
            lambda: migration_manager.ensure_current_schema(1),
        ]
        
        for operation in operations:
            with pytest.raises(RuntimeError, match="MigrationManager is closed"):
                operation()


class TestSkillStoreFacadeIntegration:
    """Test integration with SkillStore facade methods."""

    def test_skill_store_initializes_with_migration_manager(self, temp_db_path):
        """Test that SkillStore creates and uses MigrationManager."""
        from openspace.skill_engine.store import SkillStore
        
        store = SkillStore(db_path=temp_db_path)
        
        # Should have a MigrationManager instance
        assert hasattr(store, '_migrations')
        assert isinstance(store._migrations, MigrationManager)
        
        # Schema should be initialized
        assert store.get_schema_version() == 1
        
        store.close()

    def test_skill_store_facade_methods(self, temp_db_path):
        """Test that SkillStore facade methods work correctly."""
        from openspace.skill_engine.store import SkillStore
        
        store = SkillStore(db_path=temp_db_path)
        
        # Test facade methods
        version = store.get_schema_version()
        assert version == 1
        
        # Test setting version
        store.set_schema_version(2)
        assert store.get_schema_version() == 2
        
        # Test ensure_current_schema with same version (should be no-op)
        store.ensure_current_schema(2)
        assert store.get_schema_version() == 2
        
        # Test ensure_current_schema with lower version (should warn but not fail)
        store.ensure_current_schema(1)
        assert store.get_schema_version() == 2  # Should remain at 2
        
        store.close()


class TestSecurityAndRobustness:
    """Tests for security fixes and robustness improvements."""

    def test_set_schema_version_rejects_non_int(self, migration_manager):
        """Test that _set_schema_version rejects non-integer inputs."""
        # Test various non-integer types that could be used for injection
        invalid_inputs = [
            "5; DROP TABLE skill_records; --",
            5.0,
            "5",
            [],
            {},
            None,
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(TypeError, match="version must be int"):
                migration_manager._set_schema_version(invalid_input)

    def test_set_schema_version_bounds_checking(self, migration_manager):
        """Test that _set_schema_version enforces bounds checking."""
        # Test negative values
        with pytest.raises(ValueError, match="version must be non-negative"):
            migration_manager._set_schema_version(-1)
            
        # Test values exceeding reasonable limit
        with pytest.raises(ValueError, match="version \\d+ exceeds reasonable limit"):
            migration_manager._set_schema_version(100)

    def test_migration_atomic_on_failure(self, migration_manager):
        """Test that migration failures don't leave database in inconsistent state."""
        # Start at version 0
        assert migration_manager.get_schema_version() == 0
        
        # Patch the DDL statements to include a failing statement
        with patch('openspace.skill_engine.migration_manager._DDL_STATEMENTS') as mock_statements:
            mock_statements.__iter__ = lambda x: iter([
                "CREATE TABLE test_table (id INTEGER PRIMARY KEY)",
                "INVALID SQL THAT WILL FAIL",  # This will cause the migration to fail
                "CREATE TABLE another_table (id INTEGER PRIMARY KEY)"
            ])
            
            with pytest.raises(sqlite3.OperationalError):
                migration_manager.migrate_to_version(1)
        
        # Verify version wasn't bumped due to rollback
        assert migration_manager.get_schema_version() == 0
        
        # Verify no partial tables were created
        with migration_manager._conn:
            cursor = migration_manager._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            tables = cursor.fetchall()
            assert len(tables) == 0, "Partial table creation detected - transaction not atomic"

    def test_set_schema_version_deprecated_warning(self, migration_manager):
        """Test that public set_schema_version shows deprecation warning."""
        with pytest.warns(DeprecationWarning, match="set_schema_version is deprecated"):
            migration_manager.set_schema_version(1)


class TestDDLSingleSourceOfTruth:
    """Test that DDL consolidation is properly enforced."""

    def test_ddl_single_source_of_truth(self):
        """Verify no other module has CREATE TABLE strings (architecture invariant)."""
        import openspace.skill_engine.skill_repository as repo_module
        import openspace.skill_engine.analysis_store as analysis_module
        import openspace.skill_engine.tag_search as tag_module
        import inspect
        
        # Check module source code for CREATE TABLE strings
        modules_to_check = [
            (repo_module, "skill_repository.py"),
            (analysis_module, "analysis_store.py"),
            (tag_module, "tag_search.py"),
        ]
        
        for module, module_name in modules_to_check:
            source = inspect.getsource(module)
            assert "CREATE TABLE" not in source, f"{module_name} still contains CREATE TABLE DDL"
            assert "_DDL" not in source, f"{module_name} still contains _DDL constant"

    def test_standalone_modules_use_migration_manager(self, temp_db_path):
        """Verify SkillRepository/AnalysisStore/TagSearch standalone mode delegates DDL to MigrationManager."""
        from openspace.skill_engine.skill_repository import SkillRepository
        from openspace.skill_engine.analysis_store import AnalysisStore
        from openspace.skill_engine.tag_search import TagSearch
        
        # Each module should create schema via MigrationManager in standalone mode
        modules = [
            SkillRepository,
            AnalysisStore,
            TagSearch,
        ]
        
        for i, ModuleClass in enumerate(modules):
            db_path = temp_db_path.parent / f"test_standalone_{i}.db"
            
            # Create instance (should delegate to MigrationManager)
            instance = ModuleClass(db_path=db_path)
            
            # Verify schema exists
            with instance._conn:
                cursor = instance._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                
            # All modules should have core schema tables
            assert 'skill_records' in tables
            assert 'execution_analyses' in tables
            assert 'skill_judgments' in tables
            
            instance.close()

    def test_schema_consistency_across_modules(self, temp_db_path):
        """Verify that all modules create identical schemas."""
        from openspace.skill_engine.skill_repository import SkillRepository
        from openspace.skill_engine.analysis_store import AnalysisStore
        from openspace.skill_engine.tag_search import TagSearch
        
        modules = [SkillRepository, AnalysisStore, TagSearch]
        schemas = []
        
        for i, ModuleClass in enumerate(modules):
            db_path = temp_db_path.parent / f"test_consistency_{i}.db"
            instance = ModuleClass(db_path=db_path)
            
            # Get schema definition
            with instance._conn:
                cursor = instance._conn.execute("""
                    SELECT sql FROM sqlite_master 
                    WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
                schema = [row[0] for row in cursor.fetchall() if row[0]]
                schemas.append(schema)
                
            instance.close()
        
        # All schemas should be identical
        for i in range(1, len(schemas)):
            assert schemas[0] == schemas[i], f"Schema mismatch between modules {0} and {i}"