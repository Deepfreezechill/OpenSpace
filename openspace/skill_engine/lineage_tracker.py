"""LineageTracker — extracted lineage/parent-child tracking from SkillStore.

Epic 3.3: Separates lineage traversal and evolution recording from the
monolithic SkillStore, following the same extraction pattern as
SkillRepository (Epic 3.2).

Architecture:
    - Accepts ``db_path`` for standalone use, or ``conn`` + ``lock`` for
      embedding inside SkillStore (shared write connection, shared mutex).
    - All reads use a short-lived read-only connection (WAL parallel reads)
      when owning the connection, or the shared connection when injected.
    - All writes go through the persistent write connection with a mutex.
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

from .migration_manager import MigrationManager
from .types import (
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillCategory,
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
                        logger.error(
                            f"DB {func.__name__} failed after {max_retries} retries: {exc}"
                        )
                        raise
                    logger.warning(
                        f"DB {func.__name__} retry {attempt + 1}/{max_retries}: {exc}"
                    )
                    time.sleep(delay)
                    delay *= backoff

        return wrapper

    return decorator


class LineageTracker:
    """Tracks parent-child lineage relationships between skill versions.

    Extracted from ``SkillStore`` (Epic 3.3) to isolate lineage traversal
    and evolution recording logic.

    Usage (standalone)::

        tracker = LineageTracker(db_path=Path("skills.db"))
        tracker.record_derivation(child_record, parent_skill_ids=["parent_id"])
        ancestors = tracker.get_ancestors("child_id")
        tracker.close()

    Usage (embedded in SkillStore)::

        tracker = LineageTracker(conn=self._conn, lock=self._mu)
        # shares write connection and mutex with SkillStore

    Args:
        db_path: Path to the SQLite database file.
        conn: Optional existing SQLite connection (for embedding in SkillStore).
               If provided, the tracker will NOT own or close this connection.
        lock: Optional :class:`threading.Lock` to use instead of creating a
              private one.  When sharing a connection with another component,
              pass the same lock to avoid dual-mutex.
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
            # When sharing a connection, the caller owns DDL
        else:
            if db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self._db_path = Path(db_path)
            self._conn = sqlite3.connect(str(db_path), timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            # Standalone mode: ensure schema exists
            mm = MigrationManager(conn=self._conn, lock=self._mu)
            mm.initialize_schema()

        logger.debug(f"LineageTracker ready at {self._db_path}")

    # ── Connection helpers ─────────────────────────────────────────────

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
            # Fix 3: Acquire lock when using shared connection to prevent dirty reads
            with self._mu:
                yield self._conn
            return
        conn = self._make_connection(read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LineageTracker is closed")

    def close(self) -> None:
        """Close the tracker. Only closes owned resources."""
        if self._closed:
            return
        self._closed = True
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:
                pass
        logger.debug("LineageTracker closed")

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── Internal: Record persistence ─────────────────────────────────
    
    def _upsert_skill_record(self, record: SkillRecord) -> None:
        """Insert/update skill_records row + sync lineage_parents.
        
        LineageTracker-specific upsert that only handles skill_records
        and skill_lineage_parents tables. Does not handle tool deps or tags.
        
        Must be called within a transaction holding self._mu.
        """
        try:
            record.validate()
        except ValidationError as exc:
            raise ValidationError(
                f"Cannot persist invalid SkillRecord '{record.skill_id}': {exc}"
            ) from exc

        lin = record.lineage
        snapshot_json = json.dumps(lin.content_snapshot, ensure_ascii=False)

        # Insert or update skill_records
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
                record.skill_id, record.name, record.description, record.path,
                int(record.is_active), record.category.value, record.visibility.value,
                record.creator_id, lin.origin.value, lin.generation,
                lin.source_task_id, lin.change_summary,
                lin.content_diff, snapshot_json,
                lin.created_at.isoformat(), lin.created_by,
                record.total_selections, record.total_applied,
                record.total_completions, record.total_fallbacks,
                record.first_seen.isoformat(), record.last_updated.isoformat(),
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

    def _to_record(self, conn: sqlite3.Connection, row: sqlite3.Row) -> SkillRecord:
        """Deserialize a skill_records row + related rows → SkillRecord.
        
        LineageTracker-specific deserialization that includes lineage parents
        but omits tool dependencies and tags.
        """
        sid = row["skill_id"]

        parents = [
            r["parent_skill_id"]
            for r in conn.execute(
                "SELECT parent_skill_id FROM skill_lineage_parents WHERE skill_id=?",
                (sid,),
            ).fetchall()
        ]

        # Parse lineage content snapshot 
        snapshot_raw = row["lineage_content_snapshot"] or "{}"
        try:
            snapshot = json.loads(snapshot_raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid JSON in lineage_content_snapshot for {sid}: {snapshot_raw!r}")
            snapshot = {}

        # Parse dates
        created_at = datetime.fromisoformat(row["lineage_created_at"])
        first_seen = datetime.fromisoformat(row["first_seen"])
        last_updated = datetime.fromisoformat(row["last_updated"])

        lineage = SkillLineage(
            origin=SkillOrigin(row["lineage_origin"]),
            generation=row["lineage_generation"],
            source_task_id=row["lineage_source_task_id"],
            change_summary=row["lineage_change_summary"],
            content_diff=row["lineage_content_diff"],
            content_snapshot=snapshot,
            created_at=created_at,
            created_by=row["lineage_created_by"],
            parent_skill_ids=parents,
        )

        return SkillRecord(
            skill_id=sid,
            name=row["name"],
            description=row["description"],
            path=row["path"],
            is_active=bool(row["is_active"]),
            category=SkillCategory(row["category"]),
            visibility=SkillVisibility(row["visibility"]),
            creator_id=row["creator_id"],
            lineage=lineage,
            tool_dependencies=[],  # Not loaded by LineageTracker
            critical_tools=[],    # Not loaded by LineageTracker  
            tags=[],              # Not loaded by LineageTracker
            total_selections=row["total_selections"],
            total_applied=row["total_applied"],
            total_completions=row["total_completions"],
            total_fallbacks=row["total_fallbacks"],
            first_seen=first_seen,
            last_updated=last_updated,
        )

    # ── Public API: Save ──────────────────────────────────────────────
    
    @_db_retry()
    def save(self, record: SkillRecord) -> None:
        """Upsert a single :class:`SkillRecord`.
        
        Calls ``record.validate()`` before writing to enforce data integrity.
        Raises :class:`ValidationError` if the record is invalid.
        
        Note: This only syncs skill_records and skill_lineage_parents.
        Tool dependencies and tags are not managed by LineageTracker.
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            try:
                self._upsert_skill_record(record)
                if owns_txn:
                    self._conn.commit()
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                raise

    @_db_retry()
    def get(self, skill_id: str) -> Optional[SkillRecord]:
        """Retrieve a single :class:`SkillRecord` by skill_id.
        
        Args:
            skill_id: The skill_id to look up.
        
        Returns:
            The SkillRecord if found, None otherwise.
        
        Note: Tool dependencies and tags are not loaded by LineageTracker.
        """
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?", (skill_id,)
            ).fetchone()
            if not row:
                return None
            return self._to_record(conn, row)

    # ── Lineage recording ─────────────────────────────────────────────

    @_db_retry()
    def record_derivation(
        self,
        new_record: SkillRecord,
        parent_skill_ids: List[str],
    ) -> None:
        """Record a derivation: persist new skill + parent-child links.

        For ``FIXED`` origin, parent skills are deactivated (superseded).
        For ``DERIVED`` origin, parents remain active.

        Args:
            new_record: The new skill record to persist.
            parent_skill_ids: List of parent skill_ids.
        """
        self._ensure_open()
        with self._mu:
            owns_txn = not self._conn.in_transaction
            if owns_txn:
                self._conn.execute("BEGIN")
            else:
                self._conn.execute("SAVEPOINT sp_record_derivation")
            try:
                # For FIXED: deactivate same-name parents (superseded)
                if new_record.lineage.origin == SkillOrigin.FIXED:
                    for pid in parent_skill_ids:
                        self._conn.execute(
                            "UPDATE skill_records SET is_active=0, last_updated=? "
                            "WHERE skill_id=?",
                            (datetime.now().isoformat(), pid),
                        )

                new_record.lineage.parent_skill_ids = list(parent_skill_ids)
                new_record.is_active = True

                self._upsert_skill_record(new_record)
                if owns_txn:
                    self._conn.commit()
                else:
                    self._conn.execute("RELEASE sp_record_derivation")

                origin = new_record.lineage.origin.value
                logger.info(
                    f"record_derivation ({origin}): "
                    f"{new_record.name}@gen{new_record.lineage.generation} "
                    f"[{new_record.skill_id}] ← parents={parent_skill_ids}"
                )
            except Exception:
                if owns_txn:
                    self._conn.rollback()
                else:
                    self._conn.execute("ROLLBACK TO sp_record_derivation")
                raise

    # ── Lineage queries ───────────────────────────────────────────────

    @_db_retry()
    def get_lineage(self, skill_id: str) -> Optional[SkillLineage]:
        """Return the :class:`SkillLineage` for the given skill, or ``None``.

        Args:
            skill_id: The skill to look up.

        Returns:
            The lineage metadata, or ``None`` if the skill doesn't exist.
        """
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM skill_records WHERE skill_id=?", (skill_id,)
            ).fetchone()
            if not row:
                return None
            record = self._to_record(conn, row)
            return record.lineage

    @_db_retry()
    def get_children(self, parent_skill_id: str) -> List[str]:
        """Find skill_ids directly derived from the given parent.

        Args:
            parent_skill_id: The parent skill to query.

        Returns:
            List of child skill_ids (may be empty).
        """
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT skill_id FROM skill_lineage_parents WHERE parent_skill_id=?",
                (parent_skill_id,),
            ).fetchall()
            return [r["skill_id"] for r in rows]

    @_db_retry()
    def get_ancestors(
        self, skill_id: str, max_depth: int = 10
    ) -> List[SkillRecord]:
        """Walk up the lineage tree; returns ancestors nearest-first.

        Uses BFS to traverse parent links, respecting ``max_depth`` to
        prevent runaway traversal on deep or cyclic graphs.

        Args:
            skill_id: The skill to find ancestors for.
            max_depth: Maximum number of generations to traverse.

        Returns:
            List of ancestor :class:`SkillRecord` objects, sorted by
            generation (nearest first).
        """
        with self._reader() as conn:
            visited: set[str] = {skill_id}  # Fix 1: Seed with starting skill_id to prevent cycles
            ancestors: List[SkillRecord] = []
            frontier = [skill_id]

            for _ in range(max_depth):
                next_frontier: List[str] = []
                for sid in frontier:
                    for pr in conn.execute(
                        "SELECT parent_skill_id FROM skill_lineage_parents "
                        "WHERE skill_id=?",
                        (sid,),
                    ).fetchall():
                        pid = pr["parent_skill_id"]
                        if pid in visited:
                            continue
                        visited.add(pid)
                        row = conn.execute(
                            "SELECT * FROM skill_records WHERE skill_id=?",
                            (pid,),
                        ).fetchone()
                        if row:
                            ancestors.append(
                                self._to_record(conn, row)
                            )
                            next_frontier.append(pid)
                frontier = next_frontier
                if not frontier:
                    break

            # Sort by generation descending (nearest first)
            ancestors.sort(key=lambda r: r.lineage.generation, reverse=True)
            return ancestors

    @_db_retry()
    def get_evolution_chain(self, name: str) -> List[SkillRecord]:
        """Load all versions of a named skill, sorted by generation.

        Returns both active and inactive versions to show the full
        evolution history.

        Args:
            name: The skill name to search for.

        Returns:
            List of :class:`SkillRecord` sorted by ``lineage.generation`` ASC.
        """
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT * FROM skill_records WHERE name=? "
                "ORDER BY lineage_generation ASC",
                (name,),
            ).fetchall()
            return [self._to_record(conn, r) for r in rows]

    @_db_retry()
    def get_lineage_tree(
        self, skill_id: str, max_depth: int = 5
    ) -> Dict[str, Any]:
        """Build a JSON-friendly tree rooted at *skill_id* (downward).

        Each node contains: ``skill_id``, ``name``, ``generation``,
        ``origin``, ``is_active``, ``children``.

        Args:
            skill_id: Root of the tree.
            max_depth: Maximum depth to traverse.

        Returns:
            Nested dict representing the lineage tree.
        """
        with self._reader() as conn:
            return self._subtree(conn, skill_id, max_depth, set())

    def _subtree(
        self,
        conn: sqlite3.Connection,
        sid: str,
        depth: int,
        visited: set,
    ) -> Dict[str, Any]:
        """Recursive helper for :meth:`get_lineage_tree`."""
        visited.add(sid)
        row = conn.execute(
            "SELECT skill_id, name, lineage_generation, lineage_origin, "
            "is_active FROM skill_records WHERE skill_id=?",
            (sid,),
        ).fetchone()
        node: Dict[str, Any] = {
            "skill_id": sid,
            "name": row["name"] if row else "?",
            "generation": row["lineage_generation"] if row else -1,
            "origin": row["lineage_origin"] if row else "unknown",
            "is_active": bool(row["is_active"]) if row else False,
            "children": [],
        }
        if depth <= 0:
            return node
        for cr in conn.execute(
            "SELECT skill_id FROM skill_lineage_parents "
            "WHERE parent_skill_id=?",
            (sid,),
        ).fetchall():
            cid = cr["skill_id"]
            if cid not in visited:
                # Fix 2: Pass copy of visited to prevent diamond DAG edge loss
                node["children"].append(
                    self._subtree(conn, cid, depth - 1, visited.copy())
                )
        return node

    def clear(self) -> None:
        """No independent data to clear - all lineage data is in skill_records."""
        pass
