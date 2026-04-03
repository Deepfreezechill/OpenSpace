"""SkillStore — SQLite persistence facade for skill quality tracking and evolution.

Architecture (Phase 3 decomposition):

SkillStore (facade) — store.py
├── MigrationManager — Schema creation & versioning (migration_manager.py)
├── SkillRepository — CRUD operations (skill_repository.py)  
├── LineageTracker — Lineage traversal & evolution (lineage_tracker.py)
├── AnalysisStore — Execution analysis persistence (analysis_store.py)
└── TagSearch — Tag indexing & search operations (tag_search.py)

Storage location: <project_root>/.openspace/openspace.db

Tables:
  skill_records          — SkillRecord main table
  skill_lineage_parents  — Lineage parent-child relationships (many-to-many)
  execution_analyses     — ExecutionAnalysis records (one per task)
  skill_judgments         — Per-skill judgments within an analysis
  skill_tool_deps        — Tool dependencies
  skill_tags             — Auxiliary tags

The SkillStore class serves as a unified facade that:
1. Manages the persistent SQLite connection and transaction safety
2. Delegates specialized operations to focused modules
3. Coordinates cross-module workflows (e.g., evolve_skill touches both lineage & repository)
4. Provides async API via asyncio.to_thread for thread-safe database access
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from openspace.config.constants import PROJECT_ROOT
from openspace.utils.logging import Logger

from .analysis_store import AnalysisStore
from .lineage_tracker import LineageTracker
from .migration_manager import MigrationManager
from .patch import collect_skill_snapshot, compute_unified_diff
from .skill_repository import SkillRepository
from .tag_search import TagSearch
from .types import (
    EvolutionSuggestion,
    ExecutionAnalysis,
    SkillCategory,
    SkillJudgment,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillVisibility,
    ValidationError,
)

logger = Logger.get_logger(__name__)


def _db_retry(
    max_retries: int = 5,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
):
    """Retry on transient SQLite errors with exponential backoff.

    Catches ``OperationalError`` (e.g. "database is locked") and
    ``DatabaseError`` but NOT programming errors like ``InterfaceError``.
    """

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





class SkillStore:
    """SQLite persistence facade for skill quality tracking and evolution.

    Phase 3 Architecture:
    Delegates specialized operations to focused modules while maintaining unified API.
    All modules share the same connection and lock for transaction consistency.

    Concurrency:
        Write path: async method → asyncio.to_thread → _xxx_sync → self._mu lock → self._conn
        Read path: sync method → self._reader() → independent short connection (WAL parallel read)

    Lifecycle: ``__init__()`` → use → ``close()``
    Also supports async context manager:
        async with SkillStore() as store:
            await store.save_record(record)
            rec = store.load_record(skill_id)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_dir = PROJECT_ROOT / ".openspace"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "openspace.db"

        self._db_path = Path(db_path)
        self._mu = threading.Lock()
        self._closed = False

        # Crash recovery: clean up stale WAL/SHM from unclean shutdown
        self._cleanup_wal_on_startup()

        # Persistent write connection
        self._conn = self._make_connection(read_only=False)
        
        # Migration manager (Epic 3.6) — delegates schema creation and versioning.
        self._migrations = MigrationManager(conn=self._conn, lock=self._mu)
        self._migrations.ensure_current_schema()

        # CRUD repository (Epic 3.2) — delegates simple CRUD operations.
        # Shares our lock to avoid dual-mutex on the same connection.
        self._repo = SkillRepository(conn=self._conn, lock=self._mu)

        # Lineage tracker (Epic 3.3) — delegates lineage traversal/evolution.
        self._lineage = LineageTracker(conn=self._conn, lock=self._mu)

        # Analysis store (Epic 3.4) — delegates execution analyses and judgments.
        self._analyses = AnalysisStore(conn=self._conn, lock=self._mu)

        # Tag search (Epic 3.5) — delegates tag indexing and skill search operations.
        self._tag_search = TagSearch(conn=self._conn, lock=self._mu)

        logger.debug(f"SkillStore ready at {self._db_path}")

    def _make_connection(self, *, read_only: bool) -> sqlite3.Connection:
        """Create a tuned SQLite connection.

        Write connection: ``check_same_thread=False`` for cross-thread
        usage via ``asyncio.to_thread()``.

        Read connection: ``query_only=ON`` pragma for safety.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            check_same_thread=False,
        )
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

    @contextmanager
    def _reader(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a temporary read-only connection.

        WAL mode allows concurrent readers and one writer.
        Each read operation gets its own connection so reads never
        block the event loop and never contend with the write lock.
        """
        self._ensure_open()
        conn = self._make_connection(read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    def _cleanup_wal_on_startup(self) -> None:
        """Remove stale WAL/SHM left by unclean shutdown.

        If the main DB file is empty (0 bytes) but WAL/SHM companions
        exist, the database is unrecoverable — delete the companions
        so SQLite can start fresh.
        """
        if not self._db_path.exists():
            return
        wal = Path(f"{self._db_path}-wal")
        shm = Path(f"{self._db_path}-shm")
        if self._db_path.stat().st_size == 0 and (wal.exists() or shm.exists()):
            logger.warning("Empty DB with WAL/SHM — removing for crash recovery")
            for f in (wal, shm):
                if f.exists():
                    f.unlink()

    # Lifecycle
    def close(self) -> None:
        """Close the persistent connection. Subsequent ops will raise.

        Performs a WAL checkpoint before closing so that all committed
        data is flushed from the WAL file into the main ``.db`` file.
        This ensures external tools (DB browsers, backup scripts) see
        complete data without needing to understand SQLite WAL mode.
        """
        if self._closed:
            return
        with self._mu:
            self._closed = True
        try:
            self._repo.close()
        except Exception:
            pass
        try:
            self._lineage.close()
        except Exception:
            pass
        try:
            self._analyses.close()
        except Exception:
            pass
        try:
            self._tag_search.close()
        except Exception:
            pass
        try:
            self._migrations.close()
        except Exception:
            pass
        try:
            # Flush WAL → main DB so external readers see all data
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
        except Exception:
            pass
        logger.debug("SkillStore closed (WAL checkpointed)")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SkillStore is closed")

    # Write API (async, offloaded via asyncio.to_thread)
    async def save_record(self, record: SkillRecord) -> None:
        """Upsert a single :class:`SkillRecord`."""
        await asyncio.to_thread(self._save_record_sync, record)

    async def save_records(self, records: List[SkillRecord]) -> None:
        """Batch upsert in a single transaction."""
        await asyncio.to_thread(self._save_records_sync, records)

    async def sync_from_registry(
        self,
        discovered_skills: List[Any],
    ) -> int:
        """Ensure every discovered skill has an initial DB record.

        For each skill in *discovered_skills* (``SkillMeta`` objects
        from :meth:`SkillRegistry.discover`), if no record with the
        same ``skill_id`` already exists, a new :class:`SkillRecord` is
        created (``origin=IMPORTED``, ``generation=0``).

        Existing records (including evolved ones) are left untouched.

        Args:
            discovered_skills: List of ``SkillMeta`` objects.
        """
        return await asyncio.to_thread(
            self._sync_from_registry_sync,
            discovered_skills,
        )

    @_db_retry()
    def _sync_from_registry_sync(
        self,
        discovered_skills: List[Any],
    ) -> int:
        """Delegate to SkillRepository for sync logic."""
        return self._repo.sync_from_registry(discovered_skills)

    async def record_analysis(self, analysis: ExecutionAnalysis) -> None:
        """Atomic observation: insert analysis + judgments + increment counters.

        1. INSERT a row in ``execution_analyses`` (one per task).
        2. INSERT rows in ``skill_judgments`` for each skill assessed.
        3. For each judgment, atomically increment the matching
           ``skill_records`` counters:
           - total_selections  += 1         (always)
           - total_applied     += 1         (if skill_applied)
           - total_completions += 1         (if applied and completed)
           - total_fallbacks   += 1         (if not applied and not completed)
           - last_updated = now

        Note: record_analysis() and evolve_skill() are individually atomic but
        not jointly atomic. If evolve_skill() fails after record_analysis()
        succeeded, the analysis persists (this is intentional — analysis records
        what happened, regardless of whether evolution succeeds).
        """
        await asyncio.to_thread(self._record_analysis_sync, analysis)

    async def evolve_skill(
        self,
        new_record: SkillRecord,
        parent_skill_ids: List[str],
    ) -> None:
        """Atomic evolution: insert new version + deactivate old version.

        **FIXED** — Same-name skill fix:
          - ``new_record.name`` is the same as parent
          - ``new_record.path`` is the same as parent
          - parent is set to ``is_active=False``
          - ``new_record.is_active=True``

        **DERIVED** — New skill derived:
          - ``new_record.name`` is a new name
          - parent is kept ``is_active=True`` (it is still the latest version of its line)
          - ``new_record.is_active=True``

        In the same SQL transaction, guaranteed by ``self._mu``.

        Note: record_analysis() and evolve_skill() are individually atomic but
        not jointly atomic. If evolve_skill() fails after record_analysis()
        succeeded, the analysis persists (this is intentional — analysis records
        what happened, regardless of whether evolution succeeds).

        Args:
        new_record : SkillRecord
            New version record, ``lineage.parent_skill_ids`` must be non-empty.
        parent_skill_ids : list[str]
            Parent skill_id list (FIXED exactly 1, DERIVED ≥ 1).
            For FIXED, parent is automatically deactivated.
        """
        await asyncio.to_thread(self._evolve_skill_sync, new_record, parent_skill_ids)

    async def deactivate_record(self, skill_id: str) -> bool:
        """Set a specific record's ``is_active`` to False."""
        return await asyncio.to_thread(self._deactivate_record_sync, skill_id)

    async def reactivate_record(self, skill_id: str) -> bool:
        """Set a specific record's ``is_active`` to True (revert / rollback)."""
        return await asyncio.to_thread(self._reactivate_record_sync, skill_id)

    async def delete_record(self, skill_id: str) -> bool:
        """Delete a skill and all related data (CASCADE)."""
        return await asyncio.to_thread(self._delete_record_sync, skill_id)

    # Sync write implementations (thread-safe via self._mu)
    @_db_retry()
    def _save_record_sync(self, record: SkillRecord) -> None:
        """Persist a SkillRecord including its recent_analyses.

        NOT delegated to SkillRepository — analyses persistence is
        handled by SkillStore._upsert until Epic 3.4 (AnalysisStore).
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_save_record")
            try:
                self._upsert(record)
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_save_record")
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_save_record")
                raise

    @_db_retry()
    def _save_records_sync(self, records: List[SkillRecord]) -> None:
        """Batch persist SkillRecords including their recent_analyses.

        NOT delegated to SkillRepository — analyses persistence is
        handled by SkillStore._upsert until Epic 3.4 (AnalysisStore).
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_save_records")
            try:
                for r in records:
                    self._upsert(r)
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_save_records")
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_save_records")
                raise

    @_db_retry()
    def _record_analysis_sync(self, analysis: ExecutionAnalysis) -> None:
        """Persist an analysis and update skill quality counters.

        ``SkillJudgment.skill_id`` is the **true skill_id** (e.g.
        ``weather__imp_a1b2c3d4``), the same identifier used as the DB
        primary key.  The analysis LLM receives skill_ids in its prompt
        and outputs them verbatim.

        We update counters via ``WHERE skill_id = ?`` — exact match, no
        ambiguity.
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_record_analysis")
            try:
                # Delegate analysis storage to AnalysisStore (Epic 3.4)
                self._analyses.insert_analysis(analysis)

                # Update skill counters in skill_records (remains in SkillStore)
                now_iso = datetime.now().isoformat()
                for j in analysis.skill_judgments:
                    applied = 1 if j.skill_applied else 0
                    completed = 1 if (j.skill_applied and analysis.task_completed) else 0
                    fallback = 1 if (not j.skill_applied and not analysis.task_completed) else 0
                    self._conn.execute(
                        """
                        UPDATE skill_records SET
                            total_selections  = total_selections + 1,
                            total_applied     = total_applied + ?,
                            total_completions = total_completions + ?,
                            total_fallbacks   = total_fallbacks + ?,
                            last_updated      = ?
                        WHERE skill_id = ?
                        """,
                        (applied, completed, fallback, now_iso, j.skill_id),
                    )

                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_record_analysis")
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_record_analysis")
                raise

    @_db_retry()
    def _evolve_skill_sync(
        self,
        new_record: SkillRecord,
        parent_skill_ids: List[str],
    ) -> None:
        """Atomic: insert new version + deactivate parents (for FIXED).

        Delegates to :class:`LineageTracker` (Epic 3.3).

        Note: evolve_skill() is individually atomic but not jointly atomic
        with record_analysis(). If evolve fails after analysis succeeded,
        the analysis persists — this is intentional (analysis records what
        happened regardless of evolution outcome).
        """
        self._ensure_open()
        # Note: We don't acquire self._mu here because the lineage tracker
        # will acquire it and both use the same mutex (would cause deadlock).
        # The lineage tracker handles its own transaction management.
        self._lineage.record_derivation(new_record, parent_skill_ids)

    @_db_retry()
    def _deactivate_record_sync(self, skill_id: str) -> bool:
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_deactivate_record")
            try:
                cur = self._conn.execute(
                    "UPDATE skill_records SET is_active=0, last_updated=? WHERE skill_id=?",
                    (datetime.now().isoformat(), skill_id),
                )
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_deactivate_record")
                return cur.rowcount > 0
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_deactivate_record")
                raise

    @_db_retry()
    def _reactivate_record_sync(self, skill_id: str) -> bool:
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_reactivate_record")
            try:
                cur = self._conn.execute(
                    "UPDATE skill_records SET is_active=1, last_updated=? WHERE skill_id=?",
                    (datetime.now().isoformat(), skill_id),
                )
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_reactivate_record")
                return cur.rowcount > 0
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_reactivate_record")
                raise

    @_db_retry()
    def _delete_record_sync(self, skill_id: str) -> bool:
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_delete_record")
            try:
                # Clean up dependent tables BEFORE deleting skill_records to prevent ghost data
                # Order: judgments (references analyses) → analyses → tags → lineage_parents → tool_deps → skill_records
                
                # Step 1: Get analysis_ids for skill_judgments that reference this skill_id
                analysis_ids = self._conn.execute(
                    "SELECT DISTINCT analysis_id FROM skill_judgments WHERE skill_id=?",
                    (skill_id,)
                ).fetchall()
                analysis_ids = [row["analysis_id"] for row in analysis_ids]
                
                # Step 2: Delete skill_judgments that reference this skill_id
                self._conn.execute("DELETE FROM skill_judgments WHERE skill_id=?", (skill_id,))
                
                # Step 3: Delete execution_analyses that have no more judgments referencing them
                for analysis_id in analysis_ids:
                    remaining_judgments = self._conn.execute(
                        "SELECT COUNT(*) as count FROM skill_judgments WHERE analysis_id=?",
                        (analysis_id,)
                    ).fetchone()
                    if remaining_judgments["count"] == 0:
                        self._conn.execute("DELETE FROM execution_analyses WHERE id=?", (analysis_id,))
                
                # Step 4: Delete from other tables (these have CASCADE but be explicit for clarity)
                self._conn.execute("DELETE FROM skill_tags WHERE skill_id=?", (skill_id,))
                self._conn.execute("DELETE FROM skill_lineage_parents WHERE skill_id=?", (skill_id,))
                # Also clean reverse lineage refs: children pointing TO this skill as parent
                self._conn.execute("DELETE FROM skill_lineage_parents WHERE parent_skill_id=?", (skill_id,))
                self._conn.execute("DELETE FROM skill_tool_deps WHERE skill_id=?", (skill_id,))
                
                # Step 5: Finally delete from skill_records
                cur = self._conn.execute("DELETE FROM skill_records WHERE skill_id=?", (skill_id,))
                result = cur.rowcount > 0
                
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_delete_record")
                return result
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_delete_record")
                raise

    # Read API (sync, each call opens its own read-only conn)
    @_db_retry()
    def load_record(self, skill_id: str) -> Optional[SkillRecord]:
        """Load a single :class:`SkillRecord` by id, including recent_analyses.

        NOT delegated to SkillRepository — analysis hydration is
        handled by SkillStore._to_record until Epic 3.4 (AnalysisStore).
        """
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?",
                (skill_id,),
            ).fetchone()
            return self._to_record(conn, row) if row else None

    @_db_retry()
    def load_all(self, *, active_only: bool = False) -> Dict[str, SkillRecord]:
        """Load skill records, keyed by ``skill_id``, including recent_analyses.

        NOT delegated to SkillRepository — analysis hydration is
        handled by SkillStore._to_record until Epic 3.4 (AnalysisStore).

        Args:
            active_only: If True, only return records with ``is_active=True``.
        """
        with self._reader() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM skill_records WHERE is_active=1"
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM skill_records").fetchall()
            result: Dict[str, SkillRecord] = {}
            for row in rows:
                rec = self._to_record(conn, row)
                result[rec.skill_id] = rec
            logger.info(f"Loaded {len(result)} skill records (active_only={active_only})")
            return result

    @_db_retry()
    def load_active(self) -> Dict[str, SkillRecord]:
        """Load only active skill records, keyed by ``skill_id``.

        Convenience wrapper for ``load_all(active_only=True)``.
        """
        return self.load_all(active_only=True)

    @_db_retry()
    def load_record_by_path(self, skill_dir: str) -> Optional[SkillRecord]:
        """Load the most recent active SkillRecord whose ``path`` is inside *skill_dir*.

        Used by ``upload_skill`` to retrieve pre-computed upload metadata
        (origin, parents, change_summary, etc.) from the DB when
        ``.upload_meta.json`` is missing.

        The match uses ``path LIKE '{skill_dir}%'`` so both
        ``/a/b/SKILL.md`` and ``/a/b/scenarios/x.md`` match ``/a/b``.
        Returns the newest active record (by ``last_updated DESC``).

        The trailing ``/`` in the LIKE pattern prevents prefix collisions:
        ``/a/`` does NOT match ``/ab/SKILL.md``.
        """
        normalized = skill_dir.rstrip("/").rstrip("\\")
        # Escape LIKE wildcards to prevent %, _, and \ injection
        # Backslash must be escaped FIRST (it's our ESCAPE char)
        escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM skill_records WHERE path LIKE ? ESCAPE '\\' AND is_active=1 ORDER BY last_updated DESC LIMIT 1",
                (f"{escaped}/%",),
            ).fetchone()
            return self._to_record(conn, row) if row else None

    @_db_retry()
    def get_versions(self, name: str) -> List[SkillRecord]:
        """Load all versions of a named skill (active + inactive), sorted by generation.

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        # Fix: Fully hydrate records after delegation (tags, tool_deps, critical_tools, recent_analyses)
        records = self._lineage.get_evolution_chain(name)
        return self._hydrate_records(records)

    @_db_retry()
    def load_by_category(self, category: SkillCategory, *, active_only: bool = True) -> List[SkillRecord]:
        """Delegate to SkillRepository for category queries."""
        records = self._repo.load_by_category(category, active_only=active_only)
        return self._hydrate_records(records)

    @_db_retry()
    def load_analyses(
        self,
        skill_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionAnalysis]:
        """Load recent analyses.

        Args:
            skill_id: True ``skill_id`` (e.g. ``weather__imp_a1b2c3d4``).
                ``skill_judgments.skill_id`` now stores the true skill_id,
                so filtering uses exact match.
                If None, return pure-execution analyses (no judgments).
        """
        return self._analyses.load_analyses(skill_id=skill_id, limit=limit)

    @_db_retry()
    def load_analyses_for_task(self, task_id: str) -> Optional[ExecutionAnalysis]:
        """Load the analysis for a specific task, or None."""
        return self._analyses.load_analyses_for_task(task_id)

    @_db_retry()
    def load_all_analyses(self, limit: int = 200) -> List[ExecutionAnalysis]:
        """Load recent analyses across all tasks."""
        return self._analyses.load_all_analyses(limit)

    @_db_retry()
    def load_evolution_candidates(self, limit: int = 50) -> List[ExecutionAnalysis]:
        """Load analyses marked as evolution candidates."""
        return self._analyses.load_evolution_candidates(limit)

    @_db_retry()
    def find_skills_by_tool(self, tool_key: str) -> List[str]:
        """
        Only returns active records — deactivated (superseded) versions
        are excluded so that Trigger 2 never re-processes old versions.
        
        Delegates to :class:`TagSearch` (Epic 3.5).
        """
        return self._tag_search.find_skills_by_tool(tool_key)

    @_db_retry()
    def find_children(self, parent_skill_id: str) -> List[str]:
        """Find skill_ids derived from the given parent.

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        return self._lineage.get_children(parent_skill_id)

    @_db_retry()
    def count(self, *, active_only: bool = False) -> int:
        """Total number of skill records."""
        return self._repo.count(active_only=active_only)

    # Analytics / Summary
    @_db_retry()
    def get_summary(self, *, active_only: bool = True) -> List[Dict[str, Any]]:
        """Lightweight summary of skills (no analyses/deps loaded).

        Default filters to active skills only.
        
        Delegates to :class:`TagSearch` (Epic 3.5).
        """
        return self._tag_search.get_summary(active_only=active_only)

    @_db_retry()
    def get_stats(self, *, active_only: bool = True) -> Dict[str, Any]:
        """Aggregate statistics across skills.
        
        Delegates to :class:`TagSearch` (Epic 3.5) for skill stats and
        :class:`AnalysisStore` (Epic 3.4) for analysis stats.
        """
        # Get skill-related stats from TagSearch
        stats = self._tag_search.get_stats(active_only=active_only)
        
        # Merge in analysis stats from AnalysisStore
        analysis_stats = self._analyses.get_analysis_stats()
        stats.update({
            "total_analyses": analysis_stats["total_analyses"],
            "evolution_candidates": analysis_stats["evolution_candidates"],
        })
        
        return stats

    @_db_retry()
    def get_task_skill_summary(self, task_id: str) -> Dict[str, Any]:
        """Per-task summary: task-level fields + per-skill judgments.

        Useful for understanding how multiple skills contributed to a
        single task execution.

        Returns:
            dict: ``{"task_id", "task_completed", "execution_note",
                "tool_issues", "judgments": [{skill_id, skill_applied, note}],
                ...}`` or empty dict if the task has no analysis.
        """
        return self._analyses.get_task_skill_summary(task_id)

    @_db_retry()
    def get_top_skills(
        self,
        n: int = 10,
        metric: str = "effective_rate",
        min_selections: int = 1,
        *,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Top-N skills ranked by the chosen metric.

        Metrics:
            ``effective_rate``  — completions / selections
            ``applied_rate``    — applied / selections
            ``completion_rate`` — completions / applied
            ``total_selections``— raw count
            
        Delegates to :class:`TagSearch` (Epic 3.5).
        """
        return self._tag_search.get_top_skills(
            n=n, metric=metric, min_selections=min_selections, active_only=active_only
        )

    @_db_retry()
    def get_count_and_timestamp(self, *, active_only: bool = True) -> Dict[str, Any]:
        """Skill count + newest ``last_updated`` for cheap change detection.
        
        Delegates to :class:`TagSearch` (Epic 3.5).
        """
        return self._tag_search.get_count_and_timestamp(active_only=active_only)

    # Lineage / Ancestry
    @_db_retry()
    def get_ancestry(self, skill_id: str, max_depth: int = 10) -> List[SkillRecord]:
        """Walk up the lineage tree; returns ancestors nearest-first.

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        # Fix: Fully hydrate records after delegation (tags, tool_deps, critical_tools, recent_analyses)
        records = self._lineage.get_ancestors(skill_id, max_depth=max_depth)
        return self._hydrate_records(records)

    @_db_retry()
    def get_lineage_tree(self, skill_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Build a JSON-friendly tree rooted at *skill_id* (downward).

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        return self._lineage.get_lineage_tree(skill_id, max_depth=max_depth)

    # Maintenance
    def clear(self) -> None:
        """Delete all data (keeps schema).
        
        Orchestrates clearing across all modules - each module clears 
        its own data in a coordinated transaction.
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_clear")
            try:
                # Clear each module's data
                self._repo.clear()           # Clears skill_records + CASCADE
                self._analyses.clear_all_analyses()  # Clears execution_analyses + skill_judgments
                self._lineage.clear()        # No-op (data is in skill_records)
                self._tag_search.clear()     # No-op (data is in skill_tags, CASCADE)
                self._migrations.clear()     # No-op (schema only)
                
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_clear")
                logger.info("SkillStore cleared")
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_clear")
                raise

    def vacuum(self) -> None:
        """Compact the database file.
        
        DB-level maintenance, not module-specific - operates on the entire
        SQLite database file to reclaim space and optimize performance.
        """
        self._ensure_open()
        with self._mu:
            self._conn.execute("VACUUM")

    # Internal: Upsert / Insert / Deserialize
    def _upsert(self, record: SkillRecord) -> None:
        """Insert or update skill_records + sync related rows.

        Called within a transaction holding ``self._mu``.
        """
        try:
            record.validate()
        except ValidationError as exc:
            raise ValidationError(
                f"Cannot persist invalid SkillRecord '{record.skill_id}': {exc}"
            ) from exc
        lin = record.lineage
        # content_snapshot is Dict[str, str]; store as JSON text
        snapshot_json = json.dumps(lin.content_snapshot, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO skill_records (
                skill_id, name, description, path, is_active, category,
                visibility, creator_id,
                lineage_origin, lineage_generation,
                lineage_source_task_id, lineage_change_summary,
                lineage_content_diff, lineage_content_snapshot,
                lineage_created_at, lineage_created_by,
                total_selections, total_applied,
                total_completions, total_fallbacks,
                first_seen, last_updated
            ) VALUES (?,?,?,?,?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?,?,?, ?,?)
            ON CONFLICT(skill_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                path=excluded.path,
                is_active=excluded.is_active,
                category=excluded.category,
                visibility=excluded.visibility,
                creator_id=excluded.creator_id,
                lineage_origin=excluded.lineage_origin,
                lineage_generation=excluded.lineage_generation,
                lineage_source_task_id=excluded.lineage_source_task_id,
                lineage_change_summary=excluded.lineage_change_summary,
                lineage_content_diff=excluded.lineage_content_diff,
                lineage_content_snapshot=excluded.lineage_content_snapshot,
                lineage_created_at=excluded.lineage_created_at,
                lineage_created_by=excluded.lineage_created_by,
                total_selections=excluded.total_selections,
                total_applied=excluded.total_applied,
                total_completions=excluded.total_completions,
                total_fallbacks=excluded.total_fallbacks,
                last_updated=excluded.last_updated
            """,
            (
                record.skill_id,
                record.name,
                record.description,
                record.path,
                int(record.is_active),
                record.category.value,
                record.visibility.value,
                record.creator_id,
                lin.origin.value,
                lin.generation,
                lin.source_task_id,
                lin.change_summary,
                lin.content_diff,
                snapshot_json,
                lin.created_at.isoformat(),
                lin.created_by,
                record.total_selections,
                record.total_applied,
                record.total_completions,
                record.total_fallbacks,
                record.first_seen.isoformat(),
                record.last_updated.isoformat(),
            ),
        )

        # Sync lineage parents
        self._conn.execute(
            "DELETE FROM skill_lineage_parents WHERE skill_id=?",
            (record.skill_id,),
        )
        for pid in lin.parent_skill_ids:
            self._conn.execute(
                "INSERT INTO skill_lineage_parents(skill_id, parent_skill_id) VALUES(?,?)",
                (record.skill_id, pid),
            )

        # Sync tool dependencies
        self._conn.execute(
            "DELETE FROM skill_tool_deps WHERE skill_id=?",
            (record.skill_id,),
        )
        critical_set = set(record.critical_tools)
        for tk in record.tool_dependencies:
            self._conn.execute(
                "INSERT INTO skill_tool_deps(skill_id, tool_key, critical) VALUES(?,?,?)",
                (record.skill_id, tk, 1 if tk in critical_set else 0),
            )

        # Sync tags - delegate to TagSearch (Epic 3.5)
        self._tag_search.sync_tags(record.skill_id, record.tags)

        # Sync analyses (insert only NEW ones, dedup by task_id) - delegate to AnalysisStore
        self._analyses.bulk_upsert_analyses(record.recent_analyses)

    # Fix 5: Helper method to hydrate recent_analyses for records from LineageTracker
    def _hydrate_recent_analyses(self, record: SkillRecord) -> SkillRecord:
        """Hydrate recent_analyses for a SkillRecord from AnalysisStore delegation."""
        return self._analyses.hydrate_recent_analyses(record)

    def _hydrate_record(self, record: SkillRecord) -> SkillRecord:
        """Fully hydrate a SkillRecord with tags, tool_deps, critical_tools, and recent_analyses.
        
        Used by facade methods that delegate to modules returning partially hydrated records.
        """
        with self._reader() as conn:
            # Hydrate tool dependencies and critical tools
            dep_rows = conn.execute(
                "SELECT tool_key, critical FROM skill_tool_deps WHERE skill_id=?",
                (record.skill_id,),
            ).fetchall()
            
            tool_dependencies = [r["tool_key"] for r in dep_rows]
            critical_tools = [r["tool_key"] for r in dep_rows if r["critical"]]
            
        # Hydrate tags using TagSearch
        tags = self._tag_search.get_tags(record.skill_id)
        
        # Hydrate recent_analyses using AnalysisStore
        hydrated_record = self._analyses.hydrate_recent_analyses(record)
        
        # Return a new record with all fields hydrated
        return dataclasses.replace(
            hydrated_record,
            tags=tags,
            tool_dependencies=tool_dependencies,
            critical_tools=critical_tools,
        )
        
    def _hydrate_records(self, records: List[SkillRecord]) -> List[SkillRecord]:
        """Batch hydrate multiple SkillRecords — O(1) queries instead of O(N).

        Batches tag, tool_dep, and analysis queries across all records.
        """
        if not records:
            return records

        skill_ids = [r.skill_id for r in records]
        placeholders = ",".join("?" * len(skill_ids))

        with self._reader() as conn:
            # Batch 1: tool_deps — 1 query
            dep_rows = conn.execute(
                f"SELECT skill_id, tool_key, critical FROM skill_tool_deps "
                f"WHERE skill_id IN ({placeholders})",
                skill_ids,
            ).fetchall()

        deps_by_skill: Dict[str, List[str]] = {}
        critical_by_skill: Dict[str, List[str]] = {}
        for row in dep_rows:
            sid = row["skill_id"]
            deps_by_skill.setdefault(sid, []).append(row["tool_key"])
            if row["critical"]:
                critical_by_skill.setdefault(sid, []).append(row["tool_key"])

        # Batch 2: tags — 1 query (via TagSearch)
        tags_by_skill = self._tag_search.get_tags_batch(skill_ids)

        # Batch 3: analyses — 2 queries (via AnalysisStore)
        analyses_by_skill = self._analyses.batch_load_recent_analyses(
            skill_ids, SkillRecord.MAX_RECENT
        )

        # Assemble
        return [
            dataclasses.replace(
                record,
                tags=tags_by_skill.get(record.skill_id, []),
                tool_dependencies=deps_by_skill.get(record.skill_id, []),
                critical_tools=critical_by_skill.get(record.skill_id, []),
                recent_analyses=analyses_by_skill.get(record.skill_id, []),
            )
            for record in records
        ]

    # Deserialization
    def _to_record(self, conn: sqlite3.Connection, row: sqlite3.Row) -> SkillRecord:
        """Deserialize a skill_records row + related rows → SkillRecord."""
        sid = row["skill_id"]

        parents = [
            r["parent_skill_id"]
            for r in conn.execute(
                "SELECT parent_skill_id FROM skill_lineage_parents WHERE skill_id=?",
                (sid,),
            ).fetchall()
        ]

        # Deserialize content_snapshot: stored as JSON dict
        # mapping relative file paths to their text content
        raw_snapshot = row["lineage_content_snapshot"] or "{}"
        try:
            snapshot: Dict[str, str] = json.loads(raw_snapshot)
        except json.JSONDecodeError:
            snapshot = {}

        lineage = SkillLineage(
            origin=SkillOrigin(row["lineage_origin"]),
            generation=row["lineage_generation"],
            parent_skill_ids=parents,
            source_task_id=row["lineage_source_task_id"],
            change_summary=row["lineage_change_summary"],
            content_diff=row["lineage_content_diff"],
            content_snapshot=snapshot,
            created_at=datetime.fromisoformat(row["lineage_created_at"]),
            created_by=row["lineage_created_by"],
        )

        dep_rows = conn.execute(
            "SELECT tool_key, critical FROM skill_tool_deps WHERE skill_id=?",
            (sid,),
        ).fetchall()

        # Load tags - delegate to TagSearch (Epic 3.5)
        tags = self._tag_search.get_tags(sid)

        # Load recent analyses involving this skill - delegate to AnalysisStore (Epic 3.4)
        recent_analyses = self._analyses.load_recent_analyses_for_skill(
            sid, SkillRecord.MAX_RECENT
        )

        return SkillRecord(
            skill_id=sid,
            name=row["name"],
            description=row["description"],
            path=row["path"],
            is_active=bool(row["is_active"]),
            category=SkillCategory(row["category"]),
            tags=tags,
            visibility=(SkillVisibility(row["visibility"]) if row["visibility"] else SkillVisibility.PRIVATE),
            creator_id=row["creator_id"] or "",
            lineage=lineage,
            tool_dependencies=[r["tool_key"] for r in dep_rows],
            critical_tools=[r["tool_key"] for r in dep_rows if r["critical"]],
            total_selections=row["total_selections"],
            total_applied=row["total_applied"],
            total_completions=row["total_completions"],
            total_fallbacks=row["total_fallbacks"],
            recent_analyses=recent_analyses,
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_updated=datetime.fromisoformat(row["last_updated"]),
        )

    # ── Tag Search Facade Methods (Epic 3.5) ──────────────────────────────
    # Fix Finding 3: Complete SkillStore facade for all extracted TagSearch methods

    def find_skills_by_tags(
        self,
        tags: List[str],
        *,
        match_all: bool = False,
        active_only: bool = True,
    ) -> List[str]:
        """Find skills by tags (facade to TagSearch)."""
        return self._tag_search.find_skills_by_tags(
            tags, match_all=match_all, active_only=active_only
        )

    def search_skills(
        self,
        query: Optional[str] = None,
        *,
        category: Optional[SkillCategory] = None,
        visibility: Optional[SkillVisibility] = None,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
        active_only: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search skills by multiple criteria (facade to TagSearch)."""
        return self._tag_search.search_skills(
            query,
            category=category,
            visibility=visibility,
            tags=tags,
            match_all_tags=match_all_tags,
            active_only=active_only,
            limit=limit,
        )

    def get_tags(self, skill_id: str) -> List[str]:
        """Get all tags for a skill (facade to TagSearch)."""
        return self._tag_search.get_tags(skill_id)

    def get_tags_batch(self, skill_ids: List[str]) -> Dict[str, List[str]]:
        """Batch-get tags for multiple skills (facade to TagSearch)."""
        return self._tag_search.get_tags_batch(skill_ids)

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags with usage counts (facade to TagSearch)."""
        return self._tag_search.get_all_tags()

    def sync_tags(self, skill_id: str, tags: List[str]) -> None:
        """Synchronize tags for a skill (facade to TagSearch).

        Must manage transaction explicitly — TagSearch shared-mode
        delegates commit responsibility to the caller (us).
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_sync_tags")
            try:
                self._tag_search.sync_tags(skill_id, tags)
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_sync_tags")
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_sync_tags")
                raise

    # ── Migration Management (Epic 3.6) ─────────────────────────────────

    def initialize_schema(self) -> None:
        """Initialize database schema (facade to MigrationManager)."""
        return self._migrations.initialize_schema()

    def get_schema_version(self) -> int:
        """Get current schema version (facade to MigrationManager)."""
        return self._migrations.get_schema_version()

    def set_schema_version(self, version: int) -> None:
        """Set schema version (facade to MigrationManager).
        
        DEPRECATED: Use ensure_current_schema() instead.
        This method is kept for backward compatibility with existing tests.
        """
        return self._migrations._set_schema_version(version)

    def migrate_to_version(self, target_version: int) -> None:
        """Migrate schema to target version (facade to MigrationManager)."""
        return self._migrations.migrate_to_version(target_version)

    def ensure_current_schema(self, expected_version: int = 1) -> None:
        """Ensure schema is at expected version (facade to MigrationManager)."""
        return self._migrations.ensure_current_schema(expected_version)
