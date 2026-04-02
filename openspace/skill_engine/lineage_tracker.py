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

import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from openspace.utils.logging import Logger

from .skill_repository import SkillRepository
from .types import (
    SkillLineage,
    SkillOrigin,
    SkillRecord,
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
        conn: Optional[sqlite3.Connection] = None,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        self._owns_conn = conn is None
        self._closed = False
        self._mu = lock if lock is not None else threading.Lock()

        if conn is not None:
            self._conn = conn
            self._db_path = Path(":shared:")
            # When sharing a connection, the caller owns DDL and the repo
            self._repo = SkillRepository(conn=conn, lock=self._mu)
        else:
            if db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self._db_path = Path(db_path)
            self._repo = SkillRepository(db_path=db_path)
            self._conn = self._repo._conn
            self._mu = self._repo._mu

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
                self._repo.close()
            except Exception:
                pass
        logger.debug("LineageTracker closed")

    @property
    def db_path(self) -> Path:
        return self._db_path

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
            self._conn.execute("BEGIN")
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

                self._repo._upsert(new_record)
                self._conn.commit()

                origin = new_record.lineage.origin.value
                logger.info(
                    f"record_derivation ({origin}): "
                    f"{new_record.name}@gen{new_record.lineage.generation} "
                    f"[{new_record.skill_id}] ← parents={parent_skill_ids}"
                )
            except Exception:
                self._conn.rollback()
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
        record = self._repo.get(skill_id)
        return record.lineage if record else None

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
        """Walk up the lineage tree; returns ancestors oldest-first.

        Uses BFS to traverse parent links, respecting ``max_depth`` to
        prevent runaway traversal on deep or cyclic graphs.

        Args:
            skill_id: The skill to find ancestors for.
            max_depth: Maximum number of generations to traverse.

        Returns:
            List of ancestor :class:`SkillRecord` objects, sorted by
            generation (oldest first).
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
                                SkillRepository.to_record(conn, row)
                            )
                            next_frontier.append(pid)
                frontier = next_frontier
                if not frontier:
                    break

            ancestors.sort(key=lambda r: r.lineage.generation)
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
            return [SkillRepository.to_record(conn, r) for r in rows]

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
