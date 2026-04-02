"""AnalysisStore — extracted execution analysis persistence from SkillStore.

Epic 3.4: Separates analysis recording and retrieval concerns from the
monolithic SkillStore, following the same extraction pattern as
SkillRepository (Epic 3.2) and LineageTracker (Epic 3.3).

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
    ExecutionAnalysis,
    EvolutionSuggestion,
    SkillJudgment,
    SkillRecord,
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





class AnalysisStore:
    """Execution analysis persistence and retrieval.

    Extracted from ``SkillStore`` (Epic 3.4) to isolate analysis recording
    and quality tracking logic.

    Usage (standalone)::

        store = AnalysisStore(db_path=Path("analyses.db"))
        store.record_execution_analysis(analysis)
        recent = store.load_analyses(skill_id="some_skill")
        store.close()

    Usage (embedded in SkillStore)::

        # SkillStore creates us internally:
        # store._analyses = AnalysisStore(conn=store._conn, lock=store._mu)
        # All facade methods delegate to us
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

        logger.debug(f"AnalysisStore ready at {self._db_path}")

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
        """Create tables if they don't exist (idempotent).
        
        In standalone mode, delegates to MigrationManager to ensure schema consistency.
        In embedded mode (shared connection), assumes schema already exists.
        """
        # Create a MigrationManager to handle schema creation
        # This ensures DDL consistency across all modules
        migration_manager = MigrationManager(conn=self._conn, lock=self._mu)
        migration_manager.ensure_current_schema()
        logger.debug("Schema initialization delegated to MigrationManager")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AnalysisStore is closed")

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

    # ── Analysis Recording ─────────────────────────────────────────────

    @_db_retry()
    def record_execution_analysis_sync(self, analysis: ExecutionAnalysis) -> None:
        """Persist an analysis and update skill quality counters.

        ``SkillJudgment.skill_id`` is the **true skill_id** (e.g.
        ``weather__imp_a1b2c3d4``), the same identifier used as the DB
        primary key.  The analysis LLM receives skill_ids in its prompt
        and outputs them verbatim.

        We update counters via ``WHERE skill_id = ?`` — exact match, no
        ambiguity.
        
        Note: This method only handles the analyses tables. Counter updates
        for skill_records are handled by SkillStore since they belong to
        the skill repository domain.
        """
        self._ensure_open()
        with self._mu:
            self._conn.execute("BEGIN")
            try:
                analysis_id = self.insert_analysis(analysis)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

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
        with self._reader() as conn:
            if skill_id is not None:
                # Use DISTINCT to avoid duplicate analyses when multiple judgments match
                rows = conn.execute(
                    "SELECT DISTINCT ea.* FROM execution_analyses ea "
                    "JOIN skill_judgments sj ON ea.id = sj.analysis_id "
                    "WHERE sj.skill_id = ? "
                    "ORDER BY ea.timestamp DESC LIMIT ?",
                    (skill_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ea.* FROM execution_analyses ea "
                    "LEFT JOIN skill_judgments sj ON ea.id = sj.analysis_id "
                    "WHERE sj.id IS NULL "
                    "ORDER BY ea.timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._to_analysis(conn, r) for r in reversed(rows)]

    @_db_retry()
    def load_analyses_for_task(self, task_id: str) -> Optional[ExecutionAnalysis]:
        """Load the analysis for a specific task, or None."""
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM execution_analyses WHERE task_id=?",
                (task_id,),
            ).fetchone()
            return self._to_analysis(conn, row) if row else None

    @_db_retry()
    def load_all_analyses(self, limit: int = 200) -> List[ExecutionAnalysis]:
        """Load recent analyses across all tasks."""
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_analyses ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._to_analysis(conn, r) for r in reversed(rows)]

    @_db_retry()
    def load_evolution_candidates(self, limit: int = 50) -> List[ExecutionAnalysis]:
        """Load analyses marked as evolution candidates."""
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_analyses WHERE candidate_for_evolution=1 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._to_analysis(conn, r) for r in reversed(rows)]

    @_db_retry()
    def load_recent_analyses_for_skill(self, skill_id: str, limit: int = 5) -> List[ExecutionAnalysis]:
        """Load recent analyses involving this skill (via skill_judgments).
        
        Args:
            skill_id: True ``skill_id`` (e.g. ``weather__imp_a1b2c3d4``).
                ``skill_judgments.skill_id`` stores the true skill_id,
                so filtering uses exact match.
        """
        with self._reader() as conn:
            # Use DISTINCT to avoid duplicate analyses when multiple judgments match
            analysis_rows = conn.execute(
                "SELECT DISTINCT ea.* FROM execution_analyses ea "
                "JOIN skill_judgments sj ON ea.id = sj.analysis_id "
                "WHERE sj.skill_id = ? "
                "ORDER BY ea.timestamp DESC LIMIT ?",
                (skill_id, limit),
            ).fetchall()
            
            return [self._to_analysis(conn, row) for row in analysis_rows]

    def hydrate_recent_analyses(self, record: SkillRecord) -> SkillRecord:
        """Hydrate recent_analyses for a SkillRecord from other components."""
        recent_analyses = self.load_recent_analyses_for_skill(
            record.skill_id, 
            SkillRecord.MAX_RECENT
        )
        
        # Create a new record with hydrated recent_analyses
        return SkillRecord(
            skill_id=record.skill_id,
            name=record.name,
            description=record.description,
            path=record.path,
            is_active=record.is_active,
            category=record.category,
            visibility=record.visibility,
            creator_id=record.creator_id,
            lineage=record.lineage,
            total_selections=record.total_selections,
            total_applied=record.total_applied,
            total_completions=record.total_completions,
            total_fallbacks=record.total_fallbacks,
            first_seen=record.first_seen,
            last_updated=record.last_updated,
            recent_analyses=recent_analyses,  # This is what we're hydrating
            tool_dependencies=record.tool_dependencies,
            tags=record.tags,
        )

    # ── Bulk Operations ────────────────────────────────────────────────

    @_db_retry()
    def bulk_upsert_analyses(self, analyses: List[ExecutionAnalysis]) -> None:
        """Bulk insert/update multiple analyses (used during migrations).
        
        Note: This method can be called standalone or within an existing transaction
        when used by SkillStore._upsert().
        """
        self._ensure_open()
        
        if self._owns_conn:
            # Standalone mode - we own the connection and need our own transaction
            with self._mu:
                self._conn.execute("BEGIN")
                try:
                    for a in analyses:
                        existing = self._conn.execute(
                            "SELECT id FROM execution_analyses WHERE task_id=?",
                            (a.task_id,),
                        ).fetchone()
                        if existing is None:
                            self.insert_analysis(a)
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
        else:
            # Shared connection mode - assume caller has transaction and lock
            # Validate ALL analyses first before inserting any to ensure atomicity
            for a in analyses:
                a.validate()
            
            # Then insert all (caller's transaction guarantees atomicity)
            for a in analyses:
                existing = self._conn.execute(
                    "SELECT id FROM execution_analyses WHERE task_id=?",
                    (a.task_id,),
                ).fetchone()
                if existing is None:
                    self.insert_analysis(a)

    @_db_retry()
    def clear_all_analyses(self) -> None:
        """Clear all analyses (CASCADE cleans up skill_judgments).
        
        Note: This method can be called standalone or within an existing transaction
        when used by SkillStore.clear().
        """
        self._ensure_open()
        
        if self._owns_conn:
            # Standalone mode - we own the connection and need our own transaction
            with self._mu:
                self._conn.execute("BEGIN")
                try:
                    self._conn.execute("DELETE FROM execution_analyses")
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
        else:
            # Shared connection mode - assume caller has transaction and lock
            self._conn.execute("DELETE FROM execution_analyses")

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
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM execution_analyses WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if not row:
                return {}

            judgment_rows = conn.execute(
                "SELECT skill_id, skill_applied, note FROM skill_judgments WHERE analysis_id=?",
                (row["id"],),
            ).fetchall()

            try:
                evo_suggestions = json.loads(row["evolution_suggestions"] or "[]")
            except json.JSONDecodeError:
                evo_suggestions = []

            return {
                "task_id": row["task_id"],
                "timestamp": row["timestamp"],
                "task_completed": bool(row["task_completed"]),
                "execution_note": row["execution_note"],
                "tool_issues": AnalysisStore._safe_json_loads(row["tool_issues"], []),
                "candidate_for_evolution": bool(row["candidate_for_evolution"]),
                "evolution_suggestions": evo_suggestions,
                "analyzed_by": row["analyzed_by"],
                "analyzed_at": row["analyzed_at"],
                "judgments": [
                    {
                        "skill_id": jr["skill_id"],
                        "skill_applied": bool(jr["skill_applied"]),
                        "note": jr["note"],
                    }
                    for jr in judgment_rows
                ],
            }

    @_db_retry()
    def get_analysis_stats(self) -> Dict[str, int]:
        """Get analysis statistics."""
        with self._reader() as conn:
            n_analyses = conn.execute("SELECT COUNT(*) FROM execution_analyses").fetchone()[0]
            n_candidates = conn.execute(
                "SELECT COUNT(*) FROM execution_analyses WHERE candidate_for_evolution=1"
            ).fetchone()[0]
            
            return {
                "total_analyses": n_analyses,
                "evolution_candidates": n_candidates,
            }

    @staticmethod
    def _safe_json_loads(json_str: str, default: Any) -> Any:
        """Safely parse JSON string, returning default on error."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return default

    # ── Private Helpers ────────────────────────────────────────────────

    def insert_analysis(self, a: ExecutionAnalysis) -> int:
        """Insert an execution_analyses row + its skill_judgments.

        Called within a transaction holding ``self._mu``.

        Returns:
            int: The ``execution_analyses.id`` of the newly inserted row.
        """
        try:
            a.validate()
        except ValidationError as exc:
            raise ValidationError(
                f"Cannot persist invalid ExecutionAnalysis '{a.task_id}': {exc}"
            ) from exc
        cur = self._conn.execute(
            """
            INSERT INTO execution_analyses (
                task_id, timestamp,
                task_completed, execution_note,
                tool_issues, candidate_for_evolution,
                evolution_suggestions, analyzed_by, analyzed_at
            ) VALUES (?,?, ?,?, ?,?, ?,?,?)
            """,
            (
                a.task_id,
                a.timestamp.isoformat(),
                int(a.task_completed),
                a.execution_note,
                json.dumps(a.tool_issues, ensure_ascii=False),
                int(a.candidate_for_evolution),
                json.dumps(
                    [s.to_dict() for s in a.evolution_suggestions],
                    ensure_ascii=False,
                ),
                a.analyzed_by,
                a.analyzed_at.isoformat(),
            ),
        )
        analysis_id = cur.lastrowid

        for j in a.skill_judgments:
            self._conn.execute(
                "INSERT INTO skill_judgments (analysis_id, skill_id, skill_applied, note) VALUES (?,?,?,?)",
                (analysis_id, j.skill_id, int(j.skill_applied), j.note),
            )

        return analysis_id

    @staticmethod
    def _to_analysis(conn: sqlite3.Connection, row: sqlite3.Row) -> ExecutionAnalysis:
        """Deserialize an execution_analyses row + judgments → ExecutionAnalysis."""
        analysis_id = row["id"]

        judgment_rows = conn.execute(
            "SELECT skill_id, skill_applied, note FROM skill_judgments WHERE analysis_id=?",
            (analysis_id,),
        ).fetchall()

        suggestions: list[EvolutionSuggestion] = []
        raw_suggestions = row["evolution_suggestions"]
        if raw_suggestions:
            try:
                suggestions = [EvolutionSuggestion.from_dict(s) for s in json.loads(raw_suggestions)]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return ExecutionAnalysis(
            task_id=row["task_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            task_completed=bool(row["task_completed"]),
            execution_note=row["execution_note"],
            tool_issues=AnalysisStore._safe_json_loads(row["tool_issues"], []),
            skill_judgments=[
                SkillJudgment(
                    skill_id=jr["skill_id"],
                    skill_applied=bool(jr["skill_applied"]),
                    note=jr["note"],
                )
                for jr in judgment_rows
            ],
            evolution_suggestions=suggestions,
            analyzed_by=row["analyzed_by"],
            analyzed_at=datetime.fromisoformat(row["analyzed_at"]),
        )