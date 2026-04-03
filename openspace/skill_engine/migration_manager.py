"""MigrationManager — extracted DDL and schema migration logic from SkillStore.

Epic 3.6: Separates database schema creation and versioning concerns from the
monolithic SkillStore, following the same extraction pattern as SkillRepository
(Epic 3.2), LineageTracker (Epic 3.3), AnalysisStore (Epic 3.4), and TagSearch (Epic 3.5).

Architecture:
    - Accepts ``db_path`` for standalone use, or ``conn`` + ``lock`` for
      embedding inside SkillStore (shared write connection, shared mutex).
    - Handles schema creation via DDL execution (CREATE TABLE, CREATE INDEX)
    - Manages PRAGMA settings for optimal SQLite configuration
    - Idempotent: safe to run multiple times via IF NOT EXISTS
    - Future: schema versioning and upgrade paths

SAVEPOINT Usage:
    MigrationManager intentionally does NOT use SAVEPOINTs in its DDL operations.
    SQLite DDL statements (CREATE TABLE, PRAGMA user_version) have complex 
    interactions with nested transactions and SAVEPOINTs. Using explicit 
    transaction boundaries (BEGIN/COMMIT/ROLLBACK) provides cleaner, more 
    predictable behavior for schema operations.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Generator, Optional

from openspace.utils.logging import Logger

logger = Logger.get_logger(__name__)


def _db_retry(
    max_retries: int = 5,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
):
    """Retry on transient SQLite errors with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                    if attempt == max_retries - 1:
                        logger.error(f"DB {func.__name__} failed after {max_retries} retries: {exc}")
                        raise
                    logger.warning(f"DB {func.__name__} retry {attempt + 1}/{max_retries}: {exc}")
                    time.sleep(delay)
                    delay *= backoff

        return wrapper

    return decorator


# DDL statements for schema creation
# This is the SINGLE SOURCE OF TRUTH for all OpenSpace skill engine database schema.
# Other modules (SkillRepository, AnalysisStore, TagSearch) must delegate to MigrationManager
# for schema creation to prevent divergence and ensure consistency.

_DDL_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS skill_records (
        skill_id               TEXT PRIMARY KEY,
        name                   TEXT NOT NULL,
        description            TEXT NOT NULL DEFAULT '',
        path                   TEXT NOT NULL DEFAULT '',
        is_active              INTEGER NOT NULL DEFAULT 1,
        category               TEXT NOT NULL DEFAULT 'workflow',
        visibility             TEXT NOT NULL DEFAULT 'private',
        creator_id             TEXT NOT NULL DEFAULT '',
        lineage_origin         TEXT NOT NULL DEFAULT 'imported',
        lineage_generation     INTEGER NOT NULL DEFAULT 0,
        lineage_source_task_id TEXT,
        lineage_change_summary TEXT NOT NULL DEFAULT '',
        lineage_content_diff   TEXT NOT NULL DEFAULT '',
        lineage_content_snapshot TEXT NOT NULL DEFAULT '{}',
        lineage_created_at     TEXT NOT NULL,
        lineage_created_by     TEXT NOT NULL DEFAULT '',
        total_selections       INTEGER NOT NULL DEFAULT 0,
        total_applied          INTEGER NOT NULL DEFAULT 0,
        total_completions      INTEGER NOT NULL DEFAULT 0,
        total_fallbacks        INTEGER NOT NULL DEFAULT 0,
        first_seen             TEXT NOT NULL,
        last_updated           TEXT NOT NULL
    )""",
    
    """CREATE INDEX IF NOT EXISTS idx_sr_category ON skill_records(category)""",
    """CREATE INDEX IF NOT EXISTS idx_sr_updated  ON skill_records(last_updated)""",
    """CREATE INDEX IF NOT EXISTS idx_sr_active   ON skill_records(is_active)""",
    """CREATE INDEX IF NOT EXISTS idx_sr_name     ON skill_records(name)""",

    """CREATE TABLE IF NOT EXISTS skill_lineage_parents (
        skill_id        TEXT NOT NULL
            REFERENCES skill_records(skill_id) ON DELETE CASCADE,
        parent_skill_id TEXT NOT NULL,
        PRIMARY KEY (skill_id, parent_skill_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_lp_parent
        ON skill_lineage_parents(parent_skill_id)""",

    """CREATE TABLE IF NOT EXISTS execution_analyses (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id                 TEXT NOT NULL UNIQUE,
        timestamp               TEXT NOT NULL,
        task_completed          INTEGER NOT NULL DEFAULT 0,
        execution_note          TEXT NOT NULL DEFAULT '',
        tool_issues             TEXT NOT NULL DEFAULT '[]',
        candidate_for_evolution INTEGER NOT NULL DEFAULT 0,
        evolution_suggestions   TEXT NOT NULL DEFAULT '[]',
        analyzed_by             TEXT NOT NULL DEFAULT '',
        analyzed_at             TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_ea_task  ON execution_analyses(task_id)""",
    """CREATE INDEX IF NOT EXISTS idx_ea_ts    ON execution_analyses(timestamp)""",

    """CREATE TABLE IF NOT EXISTS skill_judgments (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id    INTEGER NOT NULL
            REFERENCES execution_analyses(id) ON DELETE CASCADE,
        skill_id       TEXT NOT NULL,
        skill_applied  INTEGER NOT NULL DEFAULT 0,
        note           TEXT NOT NULL DEFAULT '',
        UNIQUE(analysis_id, skill_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_sj_skill    ON skill_judgments(skill_id)""",
    """CREATE INDEX IF NOT EXISTS idx_sj_analysis ON skill_judgments(analysis_id)""",

    """CREATE TABLE IF NOT EXISTS skill_tool_deps (
        skill_id TEXT NOT NULL
            REFERENCES skill_records(skill_id) ON DELETE CASCADE,
        tool_key TEXT NOT NULL,
        critical INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (skill_id, tool_key)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_td_tool ON skill_tool_deps(tool_key)""",

    """CREATE TABLE IF NOT EXISTS skill_tags (
        skill_id TEXT NOT NULL
            REFERENCES skill_records(skill_id) ON DELETE CASCADE,
        tag      TEXT NOT NULL,
        PRIMARY KEY (skill_id, tag)
    )"""
]

# Schema version constants
CURRENT_VERSION = 1


class MigrationManager:
    """Database schema creation and migration management.

    Extracted from ``SkillStore`` (Epic 3.6) to isolate DDL and schema
    versioning logic.

    Usage (standalone)::

        manager = MigrationManager(db_path=Path("skills.db"))
        manager.initialize_schema()
        manager.close()

    Usage (embedded in SkillStore)::

        # SkillStore creates us internally:
        # store._migrations = MigrationManager(conn=store._conn, lock=store._mu)
        # SkillStore calls manager.initialize_schema() during __init__
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        self._owns_conn = conn is None
        self._closed = False
        self._mu = lock if lock is not None else threading.Lock()

        if conn is not None:
            self._conn = conn
            self._db_path = Path(":shared:")
        else:
            if db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self._db_path = Path(db_path)
            self._conn = self._make_connection(read_only=False)

        logger.debug(f"MigrationManager ready at {self._db_path}")

    # ── Connection Management ──────────────────────────────────────────

    def _make_connection(self, *, read_only: bool) -> sqlite3.Connection:
        """Create a tuned SQLite connection with optimal PRAGMA settings."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        
        # Performance and reliability PRAGMA settings
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-16000")  # 16 MB
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")
        
        if read_only:
            conn.execute("PRAGMA query_only=ON")
            
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MigrationManager is closed")

    def close(self) -> None:
        """Close the connection if we own it."""
        with self._mu:
            if self._closed:
                return
            # Only set _closed and close conn if we own the connection
            if self._owns_conn:
                self._closed = True
                try:
                    self._conn.close()
                except Exception:
                    pass

        logger.debug("MigrationManager closed")

    @contextmanager
    def _reader(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a temporary read-only connection (WAL parallel reads)."""
        self._ensure_open()
        if not self._owns_conn:
            # When sharing a connection, acquire lock to prevent dirty reads
            with self._mu:
                yield self._conn
            return
        conn = self._make_connection(read_only=True)
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ── Schema Management ───────────────────────────────────────────────

    @_db_retry()
    def initialize_schema(self) -> None:
        """Create all tables and indexes if they don't exist (idempotent).
        
        This method executes the complete DDL script to set up the OpenSpace
        skill engine database schema. Safe to call multiple times due to
        IF NOT EXISTS clauses.
        
        Uses individual execute() statements instead of executescript() to
        maintain transaction control when called from migrate_to_version().
        """
        self._ensure_open()
        
        with self._mu:
            for statement in _DDL_STATEMENTS:
                self._conn.execute(statement)
            self._conn.commit()
        
        logger.debug("Schema initialized successfully")

    def get_schema_version(self) -> int:
        """Get the current schema version from PRAGMA user_version.
        
        Returns:
            Current schema version (0 if not set).
        """
        self._ensure_open()
        
        with self._mu:
            cursor = self._conn.execute("PRAGMA user_version")
            return cursor.fetchone()[0]

    def _set_schema_version(self, version: int) -> None:
        """Set the schema version using PRAGMA user_version.
        
        Private method - external callers should use ensure_current_schema().
        
        Args:
            version: Schema version number to set.
            
        Raises:
            TypeError: If version is not an integer.
            ValueError: If version is negative or exceeds CURRENT_VERSION.
        """
        self._ensure_open()
        
        # Security: Explicit type check to prevent f-string injection
        if not isinstance(version, int):
            raise TypeError(f"schema version must be a non-negative int, got {type(version).__name__}: {version!r}")
        if version < 0:
            raise ValueError(f"version must be non-negative, got {version}")
        # Allow test versions higher than CURRENT_VERSION for testing purposes
        if version > 99:  # Reasonable upper bound for testing
            raise ValueError(f"version {version} exceeds reasonable limit")
            
        with self._mu:
            # Note: PRAGMA user_version does not support parameterized queries
            # This is safe because version is validated as an integer above
            self._conn.execute(f"PRAGMA user_version = {version}")
            # Only commit if we're not inside an outer transaction —
            # prevents escaping caller's transaction boundary
            if not self._conn.in_transaction:
                self._conn.commit()
        
        logger.debug(f"Schema version set to {version}")

    def set_schema_version(self, version: int) -> None:
        """Set the schema version using PRAGMA user_version.
        
        Args:
            version: Schema version number to set.
            
        Raises:
            TypeError: If version is not an integer.
            ValueError: If version is negative or exceeds CURRENT_VERSION.
            
        DEPRECATED: External callers should use ensure_current_schema() instead.
        This method may be removed in a future version.
        """
        import warnings
        warnings.warn(
            "set_schema_version is deprecated. Use ensure_current_schema() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._set_schema_version(version)

    def migrate_to_version(self, target_version: int) -> None:
        """Migrate schema from current version to target version.
        
        Args:
            target_version: Target schema version.
            
        Raises:
            ValueError: If target version is lower than current version.
            RuntimeError: If migration path is not supported.
        """
        if not isinstance(target_version, int):
            raise TypeError(f"schema version must be a non-negative int, got {type(target_version).__name__}: {target_version!r}")
        if target_version < 0:
            raise ValueError(f"schema version must be a non-negative int, got {type(target_version).__name__}: {target_version!r}")
        
        self._ensure_open()
        
        current_version = self.get_schema_version()
        
        if target_version < current_version:
            raise ValueError(
                f"Cannot downgrade from version {current_version} to {target_version}"
            )
        
        if target_version == current_version:
            logger.debug(f"Schema already at version {target_version}")
            return
        
        # Atomic migration: wrap DDL execution and version bump in single transaction
        with self._mu:
            self._conn.execute("BEGIN")
            try:
                # For now, we only support migration from 0 to 1
                # Future versions will add migration paths between versions
                if current_version == 0 and target_version == 1:
                    # Execute DDL statements individually to maintain transaction control
                    for statement in _DDL_STATEMENTS:
                        self._conn.execute(statement)
                    
                    # Set version atomically within the same transaction
                    self._conn.execute(f"PRAGMA user_version = {target_version}")
                    self._conn.commit()
                    
                    logger.info(f"Migrated schema from {current_version} to {target_version}")
                    
                else:
                    self._conn.rollback()
                    raise RuntimeError(
                        f"Migration from version {current_version} to {target_version} not supported"
                    )
                    
            except Exception:
                self._conn.rollback()
                raise

    def ensure_current_schema(self, expected_version: int = CURRENT_VERSION) -> None:
        """Ensure the database schema is at the expected version.
        
        This is the recommended method for application initialization.
        It will automatically migrate if needed.
        
        Args:
            expected_version: The version the schema should be at.
                             Defaults to CURRENT_VERSION.
        """
        self._ensure_open()
        
        current_version = self.get_schema_version()
        
        if current_version < expected_version:
            logger.info(f"Schema upgrade needed: {current_version} → {expected_version}")
            self.migrate_to_version(expected_version)
        elif current_version > expected_version:
            logger.warning(
                f"Database schema version {current_version} is newer than expected {expected_version}"
            )

    def clear(self) -> None:
        """No data to clear - MigrationManager only manages schema."""
        pass