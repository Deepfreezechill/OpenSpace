"""TagSearch — extracted tag indexing and search operations.

Epic 3.5: Separates tag management and skill search concerns from the monolithic SkillStore.
SkillStore delegates all tag and search calls here via the facade pattern.

Architecture:
    - Owns the SQLite connection lifecycle (or accepts one)
    - All reads use a short-lived read-only connection (WAL parallel reads)
    - All writes go through the persistent write connection with a mutex
    - Follows established patterns from SkillRepository and other extracted modules
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from openspace.utils.logging import Logger

from .skill_repository import SkillRepository
from .types import SkillCategory, SkillVisibility

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


class TagSearch:
    """Tag indexing and search operations for skills.

    Extracted from ``SkillStore`` (Epic 3.5) to isolate tag management and
    search logic.

    Usage::

        tag_search = TagSearch(db_path=Path("skills.db"))
        skills = tag_search.find_skills_by_tool("git")
        tag_search.sync_tags("skill_123", ["python", "dev"])
        tag_search.close()

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

        logger.debug(f"TagSearch ready at {self._db_path}")

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
            # When sharing a connection, acquire lock to prevent dirty reads
            with self._mu:
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
            raise RuntimeError("TagSearch is closed")

    def close(self) -> None:
        """Close owned resources. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._owns_conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.close()
            except sqlite3.Error:
                pass  # Already closed

    # ── Tag Management ─────────────────────────────────────────────────

    @_db_retry()
    def sync_tags(self, skill_id: str, tags: List[str]) -> None:
        """Synchronize tags for a skill (replace existing tags).
        
        When used standalone (owns connection), commits automatically.
        When used with shared connection, assumes caller manages transactions.
        """
        self._ensure_open()
        if self._owns_conn:
            with self._mu:
                # Clear existing tags
                self._conn.execute(
                    "DELETE FROM skill_tags WHERE skill_id=?",
                    (skill_id,),
                )
                # Insert new tags
                for tag in tags:
                    self._conn.execute(
                        "INSERT INTO skill_tags(skill_id, tag) VALUES(?,?)",
                        (skill_id, tag),
                    )
                self._conn.commit()
        else:
            # Shared connection - assume caller manages transaction
            # Clear existing tags
            self._conn.execute(
                "DELETE FROM skill_tags WHERE skill_id=?",
                (skill_id,),
            )
            # Insert new tags
            for tag in tags:
                self._conn.execute(
                    "INSERT INTO skill_tags(skill_id, tag) VALUES(?,?)",
                    (skill_id, tag),
                )

    @_db_retry()
    def get_tags(self, skill_id: str) -> List[str]:
        """Get all tags for a skill."""
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT tag FROM skill_tags WHERE skill_id=? ORDER BY tag",
                (skill_id,),
            ).fetchall()
            return [r["tag"] for r in rows]

    @_db_retry()
    def find_skills_by_tags(
        self,
        tags: List[str],
        *,
        match_all: bool = False,
        active_only: bool = True,
    ) -> List[str]:
        """Find skills that have any/all of the given tags.

        Args:
            tags: List of tags to search for
            match_all: If True, skill must have ALL tags. If False, ANY tag matches.
            active_only: If True, only return active skills.

        Returns:
            List of skill_ids matching the criteria.
        """
        if not tags:
            return []

        with self._reader() as conn:
            if match_all:
                # Skills that have ALL tags
                placeholders = ",".join("?" * len(tags))
                active_clause = " AND sr.is_active=1" if active_only else ""
                query = f"""
                    SELECT st.skill_id
                    FROM skill_tags st
                    JOIN skill_records sr ON st.skill_id = sr.skill_id
                    WHERE st.tag IN ({placeholders}){active_clause}
                    GROUP BY st.skill_id
                    HAVING COUNT(DISTINCT st.tag) = ?
                """
                rows = conn.execute(query, tags + [len(tags)]).fetchall()
            else:
                # Skills that have ANY tag
                placeholders = ",".join("?" * len(tags))
                active_clause = " AND sr.is_active=1" if active_only else ""
                query = f"""
                    SELECT DISTINCT st.skill_id
                    FROM skill_tags st
                    JOIN skill_records sr ON st.skill_id = sr.skill_id
                    WHERE st.tag IN ({placeholders}){active_clause}
                """
                rows = conn.execute(query, tags).fetchall()

            return [r["skill_id"] for r in rows]

    @_db_retry()
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags with usage counts."""
        with self._reader() as conn:
            rows = conn.execute(
                """
                SELECT st.tag, COUNT(*) as usage_count
                FROM skill_tags st
                JOIN skill_records sr ON st.skill_id = sr.skill_id
                WHERE sr.is_active = 1
                GROUP BY st.tag
                ORDER BY usage_count DESC, st.tag
                """
            ).fetchall()
            return [{"tag": r["tag"], "usage_count": r["usage_count"]} for r in rows]

    # ── Tool-Based Search ─────────────────────────────────────────────

    @_db_retry()
    def find_skills_by_tool(self, tool_key: str) -> List[str]:
        """Find active skills that depend on a specific tool.

        Only returns active records — deactivated (superseded) versions
        are excluded so that Trigger 2 never re-processes old versions.
        """
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT sd.skill_id "
                "FROM skill_tool_deps sd "
                "JOIN skill_records sr ON sd.skill_id = sr.skill_id "
                "WHERE sd.tool_key=? AND sr.is_active=1",
                (tool_key,),
            ).fetchall()
            return [r["skill_id"] for r in rows]

    # ── Skill Discovery and Filtering ─────────────────────────────────────

    @_db_retry()
    def get_summary(self, *, active_only: bool = True) -> List[Dict[str, Any]]:
        """Lightweight summary of skills (no analyses/deps loaded).

        Default filters to active skills only.
        """
        with self._reader() as conn:
            where = "WHERE is_active=1 " if active_only else ""
            rows = conn.execute(
                f"""
                SELECT skill_id, name, description, category, is_active,
                       visibility, creator_id,
                       lineage_origin, lineage_generation,
                       total_selections, total_applied,
                       total_completions, total_fallbacks,
                       first_seen, last_updated
                FROM skill_records
                {where}
                ORDER BY last_updated DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    @_db_retry()
    def get_stats(self, *, active_only: bool = True) -> Dict[str, Any]:
        """Aggregate statistics across skills."""
        with self._reader() as conn:
            where = " WHERE is_active=1" if active_only else ""
            total = conn.execute(f"SELECT COUNT(*) FROM skill_records{where}").fetchone()[0]

            by_category = {
                r["category"]: r["cnt"]
                for r in conn.execute(
                    f"SELECT category, COUNT(*) AS cnt FROM skill_records{where} GROUP BY category"
                ).fetchall()
            }
            by_origin = {
                r["lineage_origin"]: r["cnt"]
                for r in conn.execute(
                    f"SELECT lineage_origin, COUNT(*) AS cnt FROM skill_records{where} GROUP BY lineage_origin"
                ).fetchall()
            }

            agg = conn.execute(
                f"""
                SELECT SUM(total_selections)  AS sel,
                       SUM(total_applied)      AS app,
                       SUM(total_completions)  AS comp,
                       SUM(total_fallbacks)    AS fb
                FROM skill_records{where}
                """
            ).fetchone()

            # Also report total (including inactive) for context
            total_all = conn.execute("SELECT COUNT(*) FROM skill_records").fetchone()[0]

            return {
                "total_skills": total,
                "total_skills_all": total_all,
                "by_category": by_category,
                "by_lineage_origin": by_origin,
                "total_selections": agg["sel"] or 0,
                "total_applied": agg["app"] or 0,
                "total_completions": agg["comp"] or 0,
                "total_fallbacks": agg["fb"] or 0,
            }

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
        """
        rate_exprs = {
            "effective_rate": ("CAST(total_completions AS REAL) / total_selections"),
            "applied_rate": ("CAST(total_applied AS REAL) / total_selections"),
            "completion_rate": (
                "CASE WHEN total_applied > 0 THEN CAST(total_completions AS REAL) / total_applied ELSE 0.0 END"
            ),
            "total_selections": "total_selections",
        }
        expr = rate_exprs.get(metric, rate_exprs["effective_rate"])
        active_clause = " AND is_active=1" if active_only else ""

        with self._reader() as conn:
            rows = conn.execute(
                f"SELECT *, ({expr}) AS _rank "
                f"FROM skill_records "
                f"WHERE total_selections >= ?{active_clause} "
                f"ORDER BY _rank DESC LIMIT ?",
                (min_selections, n),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d.pop("_rank", None)
                results.append(d)
            return results

    @_db_retry()
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
        """Search skills by multiple criteria.

        Args:
            query: Text to search in name and description (case-insensitive LIKE)
            category: Filter by skill category
            visibility: Filter by skill visibility
            tags: Filter by tags (if provided)
            match_all_tags: If True, skill must have ALL tags. If False, ANY tag matches.
            active_only: If True, only return active skills.
            limit: Maximum number of results to return.

        Returns:
            List of skill records matching the criteria.
        """
        with self._reader() as conn:
            conditions = []
            params = []

            # Active filter
            if active_only:
                conditions.append("sr.is_active = 1")

            # Text search in name and description
            if query:
                # Escape SQL LIKE wildcards in user input
                escaped_query = query.replace("%", "\\%").replace("_", "\\_")
                conditions.append("(sr.name LIKE ? ESCAPE '\\' OR sr.description LIKE ? ESCAPE '\\')")
                like_pattern = f"%{escaped_query}%"
                params.extend([like_pattern, like_pattern])

            # Category filter
            if category:
                conditions.append("sr.category = ?")
                params.append(category.value)

            # Visibility filter
            if visibility:
                conditions.append("sr.visibility = ?")
                params.append(visibility.value)

            # Base query
            query_sql = "SELECT DISTINCT sr.* FROM skill_records sr"
            
            # Tag filtering requires JOIN
            if tags:
                if match_all_tags:
                    # For ALL tags, we need the skill to have ALL specified tags
                    placeholders = ",".join("?" * len(tags))
                    query_sql += f"""
                        JOIN skill_tags st ON sr.skill_id = st.skill_id
                        WHERE st.tag IN ({placeholders})
                    """
                    params = tags + params
                    # Group by skill_id and ensure it has all tags
                    if conditions:
                        query_sql += " AND " + " AND ".join(conditions)
                    query_sql += f" GROUP BY sr.skill_id HAVING COUNT(DISTINCT st.tag) = {len(tags)}"
                else:
                    # For ANY tags, simpler JOIN
                    placeholders = ",".join("?" * len(tags))
                    query_sql += f"""
                        JOIN skill_tags st ON sr.skill_id = st.skill_id
                        WHERE st.tag IN ({placeholders})
                    """
                    params = tags + params
                    if conditions:
                        query_sql += " AND " + " AND ".join(conditions)
            else:
                # No tag filtering
                if conditions:
                    query_sql += " WHERE " + " AND ".join(conditions)

            # Ordering and limit
            query_sql += " ORDER BY sr.last_updated DESC"
            if limit:
                query_sql += " LIMIT ?"
                params.append(limit)

            rows = conn.execute(query_sql, params).fetchall()
            return [dict(r) for r in rows]

    @_db_retry()
    def get_count_and_timestamp(self, *, active_only: bool = True) -> Dict[str, Any]:
        """Skill count + newest ``last_updated`` for cheap change detection."""
        with self._reader() as conn:
            where = " WHERE is_active=1" if active_only else ""
            row = conn.execute(
                f"SELECT COUNT(*) AS cnt, MAX(last_updated) AS max_ts FROM skill_records{where}"
            ).fetchone()
            return {
                "count": row["cnt"] if row else 0,
                "max_last_updated": row["max_ts"] if row else None,
            }