"""SkillRepository — extracted CRUD operations for SkillRecord persistence.

Epic 3.2: Separates data-access concerns from the monolithic SkillStore.
SkillStore delegates all CRUD calls here via the facade pattern.

Architecture:
    - Owns the SQLite connection lifecycle (or accepts one)
    - All reads use a short-lived read-only connection (WAL parallel reads)
    - All writes go through the persistent write connection with a mutex
    - validate() is called before every write
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from openspace.utils.logging import Logger

from .types import (
    SkillCategory,
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


class SkillRepository:
    """Pure CRUD repository for :class:`SkillRecord` persistence.

    Extracted from ``SkillStore`` (Epic 3.2) to isolate data-access logic.

    Usage::

        repo = SkillRepository(db_path=Path("skills.db"))
        repo.save(record)
        loaded = repo.get("skill__abc123")
        repo.close()

    Args:
        db_path: Path to the SQLite database file.
        conn: Optional existing SQLite connection (for embedding in SkillStore).
               If provided, the repository will NOT own or close this connection.
        lock: Optional :class:`threading.Lock` to use instead of creating a
              private one.  When sharing a connection with another component
              (e.g. ``SkillStore``), pass the same lock to avoid dual-mutex.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
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
            self._init_db()

        logger.debug(f"SkillRepository ready at {self._db_path}")

    # ── Connection Management ──────────────────────────────────────────

    def _make_connection(self, *, read_only: bool) -> sqlite3.Connection:
        """Create a tuned SQLite connection."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-16000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")
        if read_only:
            conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _reader(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a temporary read-only connection (WAL parallel reads)."""
        self._ensure_open()
        if not self._owns_conn:
            # When sharing a connection, just use it directly
            yield self._conn
            return
        conn = self._make_connection(read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    @_db_retry()
    def _init_db(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        with self._mu:
            self._conn.executescript(_DDL)
            self._conn.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SkillRepository is closed")

    def close(self) -> None:
        """Close the repository. Only closes the connection if we own it."""
        if self._closed:
            return
        self._closed = True
        if self._owns_conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.close()
            except Exception:
                pass
        logger.debug("SkillRepository closed")

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── CRUD: Save / Upsert ───────────────────────────────────────────

    @_db_retry()
    def save(self, record: SkillRecord) -> None:
        """Upsert a single :class:`SkillRecord`.

        Calls ``record.validate()`` before writing to enforce data integrity.
        Raises :class:`ValidationError` if the record is invalid.
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
    def save_many(self, records: List[SkillRecord]) -> None:
        """Batch upsert multiple records in a single transaction.

        All records are validated before any writes. If any record fails
        validation, no records are written (atomic).
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

    # ── CRUD: Get by ID ───────────────────────────────────────────────

    @_db_retry()
    def get(self, skill_id: str) -> Optional[SkillRecord]:
        """Load a single :class:`SkillRecord` by its ``skill_id``.

        Returns:
            The record, or ``None`` if not found.
        """
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?",
                (skill_id,),
            ).fetchone()
            return self._to_record(conn, row) if row else None

    # ── CRUD: Delete ──────────────────────────────────────────────────

    @_db_retry()
    def delete(self, skill_id: str) -> bool:
        """Delete a skill record and all related data (CASCADE).

        Returns:
            ``True`` if a record was deleted, ``False`` if not found.
        """
        self._ensure_open()
        with self._mu:
            cur = self._conn.execute(
                "DELETE FROM skill_records WHERE skill_id=?",
                (skill_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ── CRUD: List All ────────────────────────────────────────────────

    @_db_retry()
    def list_all(self, *, active_only: bool = False) -> Dict[str, SkillRecord]:
        """Load all skill records, keyed by ``skill_id``.

        Args:
            active_only: If ``True``, only return records with ``is_active=True``.
        """
        with self._reader() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM skill_records WHERE is_active=1"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM skill_records"
                ).fetchall()
            result: Dict[str, SkillRecord] = {}
            for row in rows:
                rec = self._to_record(conn, row)
                result[rec.skill_id] = rec
            return result

    # ── CRUD: Search ──────────────────────────────────────────────────

    @_db_retry()
    def search(
        self,
        *,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[SkillCategory] = None,
        active_only: bool = True,
    ) -> List[SkillRecord]:
        """Search skills by name pattern, tags, and/or category.

        Filters are combined with AND logic. A ``name`` filter uses
        SQL ``LIKE '%name%'`` for substring matching.

        Args:
            name: Substring to match in the skill name.
            tags: List of tags; skills must have at least one matching tag.
            category: Filter by :class:`SkillCategory`.
            active_only: If ``True`` (default), only search active records.

        Returns:
            List of matching :class:`SkillRecord` objects.
        """
        with self._reader() as conn:
            conditions: List[str] = []
            params: List[Any] = []

            if active_only:
                conditions.append("sr.is_active = 1")

            if name is not None:
                escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                conditions.append("sr.name LIKE ? ESCAPE '\\'")
                params.append(f"%{escaped}%")

            if category is not None:
                conditions.append("sr.category = ?")
                params.append(category.value)

            if tags:
                placeholders = ",".join("?" for _ in tags)
                conditions.append(
                    f"sr.skill_id IN (SELECT skill_id FROM skill_tags WHERE tag IN ({placeholders}))"
                )
                params.extend(tags)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            rows = conn.execute(
                f"SELECT sr.* FROM skill_records sr {where} ORDER BY sr.last_updated DESC",
                params,
            ).fetchall()

            return [self._to_record(conn, row) for row in rows]

    # ── CRUD: Count ───────────────────────────────────────────────────

    @_db_retry()
    def count(self, *, active_only: bool = False) -> int:
        """Return the total number of skill records.

        Args:
            active_only: If ``True``, only count active records.
        """
        with self._reader() as conn:
            if active_only:
                return conn.execute(
                    "SELECT COUNT(*) FROM skill_records WHERE is_active=1"
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM skill_records"
            ).fetchone()[0]

    # ── CRUD: Exists ──────────────────────────────────────────────────

    @_db_retry()
    def exists(self, skill_id: str) -> bool:
        """Check whether a record with the given ``skill_id`` exists.

        More efficient than ``get()`` — doesn't deserialize the full record.
        """
        with self._reader() as conn:
            row = conn.execute(
                "SELECT 1 FROM skill_records WHERE skill_id=? LIMIT 1",
                (skill_id,),
            ).fetchone()
            return row is not None

    # ── Internal: Upsert ──────────────────────────────────────────────

    def _upsert(self, record: SkillRecord) -> None:
        """Insert or replace a skill_records row + sync related tables.

        Must be called within a transaction holding ``self._mu``.
        """
        try:
            record.validate()
        except ValidationError as exc:
            raise ValidationError(
                f"Cannot persist invalid SkillRecord '{record.skill_id}': {exc}"
            ) from exc

        lin = record.lineage
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

        # Sync tags
        self._conn.execute(
            "DELETE FROM skill_tags WHERE skill_id=?",
            (record.skill_id,),
        )
        for tag in record.tags:
            self._conn.execute(
                "INSERT INTO skill_tags(skill_id, tag) VALUES(?,?)",
                (record.skill_id, tag),
            )

    # ── Internal: Deserialization ─────────────────────────────────────

    @staticmethod
    def _to_record(conn: sqlite3.Connection, row: sqlite3.Row) -> SkillRecord:
        """Deserialize a ``skill_records`` row + related rows → SkillRecord."""
        sid = row["skill_id"]

        parents = [
            r["parent_skill_id"]
            for r in conn.execute(
                "SELECT parent_skill_id FROM skill_lineage_parents WHERE skill_id=?",
                (sid,),
            ).fetchall()
        ]

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

        tag_rows = conn.execute(
            "SELECT tag FROM skill_tags WHERE skill_id=?",
            (sid,),
        ).fetchall()

        return SkillRecord(
            skill_id=sid,
            name=row["name"],
            description=row["description"],
            path=row["path"],
            is_active=bool(row["is_active"]),
            category=SkillCategory(row["category"]),
            tags=[r["tag"] for r in tag_rows],
            visibility=(
                SkillVisibility(row["visibility"])
                if row["visibility"]
                else SkillVisibility.PRIVATE
            ),
            creator_id=row["creator_id"] or "",
            lineage=lineage,
            tool_dependencies=[r["tool_key"] for r in dep_rows],
            critical_tools=[r["tool_key"] for r in dep_rows if r["critical"]],
            total_selections=row["total_selections"],
            total_applied=row["total_applied"],
            total_completions=row["total_completions"],
            total_fallbacks=row["total_fallbacks"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_updated=datetime.fromisoformat(row["last_updated"]),
        )
