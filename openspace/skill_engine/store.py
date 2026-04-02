"""
Storage location: <project_root>/.openspace/openspace.db
Tables:
  skill_records          — SkillRecord main table
  skill_lineage_parents  — Lineage parent-child relationships (many-to-many)
  execution_analyses     — ExecutionAnalysis records (one per task)
  skill_judgments         — Per-skill judgments within an analysis
  skill_tool_deps        — Tool dependencies
  skill_tags             — Auxiliary tags
"""

from __future__ import annotations

import asyncio
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


_DDL = """
CREATE TABLE IF NOT EXISTS skill_records (
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
);
CREATE INDEX IF NOT EXISTS idx_sr_category ON skill_records(category);
CREATE INDEX IF NOT EXISTS idx_sr_updated  ON skill_records(last_updated);
CREATE INDEX IF NOT EXISTS idx_sr_active   ON skill_records(is_active);
CREATE INDEX IF NOT EXISTS idx_sr_name     ON skill_records(name);

CREATE TABLE IF NOT EXISTS skill_lineage_parents (
    skill_id        TEXT NOT NULL
        REFERENCES skill_records(skill_id) ON DELETE CASCADE,
    parent_skill_id TEXT NOT NULL,
    PRIMARY KEY (skill_id, parent_skill_id)
);
CREATE INDEX IF NOT EXISTS idx_lp_parent
    ON skill_lineage_parents(parent_skill_id);

-- One row per task.  task_id is UNIQUE (at most one analysis per task).
CREATE TABLE IF NOT EXISTS execution_analyses (
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
);
CREATE INDEX IF NOT EXISTS idx_ea_task  ON execution_analyses(task_id);
CREATE INDEX IF NOT EXISTS idx_ea_ts    ON execution_analyses(timestamp);

-- Per-skill judgments within an analysis.
-- FK to execution_analyses.id (CASCADE delete).
-- skill_id is a plain TEXT — no FK to skill_records so that
-- historical judgments survive skill deletion.
CREATE TABLE IF NOT EXISTS skill_judgments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id    INTEGER NOT NULL
        REFERENCES execution_analyses(id) ON DELETE CASCADE,
    skill_id       TEXT NOT NULL,
    skill_applied  INTEGER NOT NULL DEFAULT 0,
    note           TEXT NOT NULL DEFAULT '',
    UNIQUE(analysis_id, skill_id)
);
CREATE INDEX IF NOT EXISTS idx_sj_skill    ON skill_judgments(skill_id);
CREATE INDEX IF NOT EXISTS idx_sj_analysis ON skill_judgments(analysis_id);

CREATE TABLE IF NOT EXISTS skill_tool_deps (
    skill_id TEXT NOT NULL
        REFERENCES skill_records(skill_id) ON DELETE CASCADE,
    tool_key TEXT NOT NULL,
    critical INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (skill_id, tool_key)
);
CREATE INDEX IF NOT EXISTS idx_td_tool ON skill_tool_deps(tool_key);

CREATE TABLE IF NOT EXISTS skill_tags (
    skill_id TEXT NOT NULL
        REFERENCES skill_records(skill_id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (skill_id, tag)
);
"""


class SkillStore:
    """SQLite persistence engine — Skill quality tracking and evolution ledger.

    Architecture:
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
        self._init_db()

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

    @_db_retry()
    def _init_db(self) -> None:
        """Create tables if they don't exist (idempotent via IF NOT EXISTS)."""
        with self._mu:
            self._conn.executescript(_DDL)
            self._conn.commit()

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
        self._ensure_open()
        created = 0
        refreshed = 0
        with self._mu:
            self._conn.execute("BEGIN")
            try:
                # Fetch all existing records keyed by skill_id
                rows = self._conn.execute(
                    "SELECT skill_id, name, description, lineage_content_snapshot FROM skill_records"
                ).fetchall()
                existing: Dict[str, Any] = {r[0]: r for r in rows}

                # Also fetch all paths with an active record.
                # After FIX evolution the DB skill_id changes but the
                # filesystem path stays the same.  Matching by path
                # prevents creating a duplicate imported record on restart.
                path_rows = self._conn.execute("SELECT path FROM skill_records WHERE is_active=1").fetchall()
                existing_active_paths: set = {r[0] for r in path_rows}

                for meta in discovered_skills:
                    path_str = str(meta.path)
                    skill_dir = meta.path.parent

                    if meta.skill_id in existing:
                        # Refresh name/description if frontmatter changed,
                        # and backfill empty content_snapshot
                        row = existing[meta.skill_id]
                        updates: List[str] = []
                        params: list = []

                        if row["name"] != meta.name:
                            updates.append("name=?")
                            params.append(meta.name)
                        if row["description"] != meta.description:
                            updates.append("description=?")
                            params.append(meta.description)

                        raw_snap = row["lineage_content_snapshot"] or ""
                        if raw_snap in ("", "{}"):
                            try:
                                snap = collect_skill_snapshot(skill_dir)
                                if snap:
                                    updates.append("lineage_content_snapshot=?")
                                    params.append(json.dumps(snap, ensure_ascii=False))
                                    diff = "\n".join(
                                        compute_unified_diff("", text, filename=name)
                                        for name, text in sorted(snap.items())
                                        if compute_unified_diff("", text, filename=name)
                                    )
                                    if diff:
                                        updates.append("lineage_content_diff=?")
                                        params.append(diff)
                            except Exception as e:
                                logger.warning(f"sync_from_registry: snapshot backfill failed for {meta.skill_id}: {e}")

                        if updates:
                            params.append(meta.skill_id)
                            self._conn.execute(
                                f"UPDATE skill_records SET {', '.join(updates)} WHERE skill_id=?",
                                params,
                            )
                            refreshed += 1
                        continue

                    # Path already covered by an evolved record
                    if path_str in existing_active_paths:
                        continue

                    # Snapshot the directory so this version can be restored later
                    snapshot: Dict[str, str] = {}
                    content_diff = ""
                    try:
                        snapshot = collect_skill_snapshot(skill_dir)
                        content_diff = "\n".join(
                            compute_unified_diff("", text, filename=name)
                            for name, text in sorted(snapshot.items())
                            if compute_unified_diff("", text, filename=name)
                        )
                    except Exception as e:
                        logger.warning(f"sync_from_registry: failed to snapshot {skill_dir}: {e}")

                    record = SkillRecord(
                        skill_id=meta.skill_id,
                        name=meta.name,
                        description=meta.description,
                        path=path_str,
                        is_active=True,
                        lineage=SkillLineage(
                            origin=SkillOrigin.IMPORTED,
                            generation=0,
                            content_snapshot=snapshot,
                            content_diff=content_diff,
                        ),
                    )
                    self._upsert(record)
                    created += 1
                    logger.debug(f"sync_from_registry: created {meta.name} [{meta.skill_id}]")

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        if created or refreshed:
            logger.info(
                f"sync_from_registry: {created} new record(s) created, "
                f"{refreshed} refreshed, "
                f"{len(discovered_skills) - created - refreshed} unchanged"
            )
        return created

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
            self._conn.execute("BEGIN")
            try:
                self._upsert(record)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @_db_retry()
    def _save_records_sync(self, records: List[SkillRecord]) -> None:
        """Batch persist SkillRecords including their recent_analyses.

        NOT delegated to SkillRepository — analyses persistence is
        handled by SkillStore._upsert until Epic 3.4 (AnalysisStore).
        """
        self._ensure_open()
        with self._mu:
            self._conn.execute("BEGIN")
            try:
                for r in records:
                    self._upsert(r)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
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
            self._conn.execute("BEGIN")
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

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @_db_retry()
    def _evolve_skill_sync(
        self,
        new_record: SkillRecord,
        parent_skill_ids: List[str],
    ) -> None:
        """Atomic: insert new version + deactivate parents (for FIXED).

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        self._lineage.record_derivation(new_record, parent_skill_ids)

    @_db_retry()
    def _deactivate_record_sync(self, skill_id: str) -> bool:
        self._ensure_open()
        with self._mu:
            cur = self._conn.execute(
                "UPDATE skill_records SET is_active=0, last_updated=? WHERE skill_id=?",
                (datetime.now().isoformat(), skill_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    @_db_retry()
    def _reactivate_record_sync(self, skill_id: str) -> bool:
        self._ensure_open()
        with self._mu:
            cur = self._conn.execute(
                "UPDATE skill_records SET is_active=1, last_updated=? WHERE skill_id=?",
                (datetime.now().isoformat(), skill_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    @_db_retry()
    def _delete_record_sync(self, skill_id: str) -> bool:
        self._ensure_open()
        return self._repo.delete(skill_id)

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
        """
        normalized = skill_dir.rstrip("/")
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM skill_records WHERE path LIKE ? AND is_active=1 ORDER BY last_updated DESC LIMIT 1",
                (f"{normalized}%",),
            ).fetchone()
            return self._to_record(conn, row) if row else None

    @_db_retry()
    def get_versions(self, name: str) -> List[SkillRecord]:
        """Load all versions of a named skill (active + inactive), sorted by generation.

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        # Fix 5: Hydrate recent_analyses after delegation
        records = self._lineage.get_evolution_chain(name)
        return [self._hydrate_recent_analyses(record) for record in records]

    @_db_retry()
    def load_by_category(self, category: SkillCategory, *, active_only: bool = True) -> List[SkillRecord]:
        """Load skill records filtered by category.

        Args:
            active_only: If True (default), only return active records.
        """
        with self._reader() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM skill_records WHERE category=? AND is_active=1",
                    (category.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM skill_records WHERE category=?",
                    (category.value,),
                ).fetchall()
            return [self._to_record(conn, r) for r in rows]

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
        """Walk up the lineage tree; returns ancestors oldest-first.

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        # Fix 5: Hydrate recent_analyses after delegation
        records = self._lineage.get_ancestors(skill_id, max_depth=max_depth)
        return [self._hydrate_recent_analyses(record) for record in records]

    @_db_retry()
    def get_lineage_tree(self, skill_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Build a JSON-friendly tree rooted at *skill_id* (downward).

        Delegates to :class:`LineageTracker` (Epic 3.3).
        """
        return self._lineage.get_lineage_tree(skill_id, max_depth=max_depth)

    # Maintenance
    def clear(self) -> None:
        """Delete all data (keeps schema)."""
        self._ensure_open()
        with self._mu:
            self._conn.execute("BEGIN")
            try:
                # CASCADE on skill_records cleans up: lineage_parents, tool_deps, tags
                self._conn.execute("DELETE FROM skill_records")
                # Delegate analysis clearing to AnalysisStore (Epic 3.4)
                self._analyses.clear_all_analyses()
                self._conn.commit()
                logger.info("SkillStore cleared")
            except Exception:
                self._conn.rollback()
                raise

    def vacuum(self) -> None:
        """Compact the database file."""
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

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags with usage counts (facade to TagSearch)."""
        return self._tag_search.get_all_tags()

    def sync_tags(self, skill_id: str, tags: List[str]) -> None:
        """Synchronize tags for a skill (facade to TagSearch)."""
        return self._tag_search.sync_tags(skill_id, tags)
