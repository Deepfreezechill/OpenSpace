"""MCP tool handler implementations for OpenSpace.

Extracted from ``mcp_server.py`` (Epic 4.7) to isolate tool handler logic
from server bootstrap and transport concerns.

Four MCP tools:
  - execute_task   — Full grounding engine delegation
  - search_skills  — Standalone skill search/discovery
  - fix_skill      — Manual skill fix (FIX evolution only)
  - upload_skill   — Upload local skill to cloud

Call ``register_handlers(mcp)`` to wire all handlers to a FastMCP instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openspace.mcp_server")

# ---------------------------------------------------------------------------
# Module-level state (singleton pattern, same as original mcp_server.py)
# ---------------------------------------------------------------------------
_openspace_instance = None
_openspace_lock = asyncio.Lock()
_standalone_store = None

# Tracks bot skill directories already registered this session.
_registered_skill_dirs: set = set()

_UPLOAD_META_FILENAME = ".upload_meta.json"


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
async def _get_openspace():
    """Lazy-initialise the OpenSpace engine."""
    global _openspace_instance
    if _openspace_instance is not None and _openspace_instance.is_initialized():
        return _openspace_instance

    async with _openspace_lock:
        if _openspace_instance is not None and _openspace_instance.is_initialized():
            return _openspace_instance

        logger.info("Initializing OpenSpace engine ...")
        from openspace.host_detection import build_grounding_config_path, build_llm_kwargs
        from openspace.tool_layer import OpenSpace, OpenSpaceConfig

        env_model = os.environ.get("OPENSPACE_MODEL", "")
        workspace = os.environ.get("OPENSPACE_WORKSPACE")
        max_iter = int(os.environ.get("OPENSPACE_MAX_ITERATIONS", "20"))
        enable_rec = os.environ.get("OPENSPACE_ENABLE_RECORDING", "true").lower() in ("true", "1", "yes")

        backend_scope_raw = os.environ.get("OPENSPACE_BACKEND_SCOPE")
        backend_scope = [b.strip() for b in backend_scope_raw.split(",") if b.strip()] if backend_scope_raw else None

        config_path = build_grounding_config_path()
        model, llm_kwargs = build_llm_kwargs(env_model)

        _pkg_root = str(Path(__file__).resolve().parent.parent.parent)
        recording_base = workspace or _pkg_root
        recording_log_dir = str(Path(recording_base) / "logs" / "recordings")

        config = OpenSpaceConfig(
            llm_model=model,
            llm_kwargs=llm_kwargs,
            workspace_dir=workspace,
            grounding_max_iterations=max_iter,
            enable_recording=enable_rec,
            recording_backends=["shell"] if enable_rec else None,
            recording_log_dir=recording_log_dir,
            backend_scope=backend_scope,
            grounding_config_path=config_path,
        )

        _openspace_instance = OpenSpace(config=config)
        await _openspace_instance.initialize()
        logger.info("OpenSpace engine ready (model=%s).", model)

        # Auto-register host bot skill directories from env
        host_skill_dirs_raw = os.environ.get("OPENSPACE_HOST_SKILL_DIRS", "")
        if host_skill_dirs_raw:
            dirs = [d.strip() for d in host_skill_dirs_raw.split(",") if d.strip()]
            if dirs:
                await _auto_register_skill_dirs(dirs)
                logger.info("Auto-registered host skill dirs from OPENSPACE_HOST_SKILL_DIRS: %s", dirs)

        return _openspace_instance


def _get_store():
    """Get SkillStore — reuses OpenSpace's internal instance when available."""
    global _standalone_store
    if _openspace_instance and _openspace_instance.is_initialized():
        internal = getattr(_openspace_instance, "_skill_store", None)
        if internal and not internal._closed:
            return internal
    if _standalone_store is None or _standalone_store._closed:
        from openspace.skill_engine import SkillStore

        _standalone_store = SkillStore()
    return _standalone_store


def _is_auto_import_enabled() -> bool:
    """Check whether cloud auto-import is enabled in SkillConfig.

    Returns ``False`` (safe default) if the config is unavailable or the
    flag is not explicitly set to ``True``.
    """
    if _openspace_instance and _openspace_instance.is_initialized():
        gc = getattr(_openspace_instance, "_grounding_config", None)
        if gc and gc.skills:
            return gc.skills.auto_import_enabled
    return False


def _get_cloud_client():
    """Get a OpenSpaceClient instance (raises CloudError if not configured)."""
    from openspace.cloud.auth import get_openspace_auth
    from openspace.cloud.client import OpenSpaceClient

    auth_headers, api_base = get_openspace_auth()
    return OpenSpaceClient(auth_headers, api_base)


# ---------------------------------------------------------------------------
# Upload metadata helpers
# ---------------------------------------------------------------------------
def _write_upload_meta(skill_dir: Path, info: Dict[str, Any]) -> None:
    """Write ``.upload_meta.json`` so ``upload_skill`` can read pre-saved metadata."""
    meta = {
        "origin": info.get("origin", "imported"),
        "parent_skill_ids": info.get("parent_skill_ids", []),
        "change_summary": info.get("change_summary", ""),
        "created_by": info.get("created_by", "openspace"),
        "tags": info.get("tags", []),
    }
    meta_path = skill_dir / _UPLOAD_META_FILENAME
    try:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.debug(f"Wrote upload metadata to {meta_path}")
    except Exception as e:
        logger.warning(f"Failed to write upload metadata: {e}")


def _read_upload_meta(skill_dir: Path) -> Dict[str, Any]:
    """Read upload metadata with three-tier fallback.

    Resolution order:
      1. ``.upload_meta.json`` sidecar file
      2. SkillStore DB lookup by path
      3. Empty dict (caller applies defaults)
    """
    # Tier 1: sidecar file
    meta_path = skill_dir / _UPLOAD_META_FILENAME
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if data:
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read upload metadata file: {e}")

    # Tier 2: DB lookup
    try:
        store = _get_store()
        rec = store.load_record_by_path(str(skill_dir))
        if rec:
            logger.debug(f"Upload metadata resolved from DB for {skill_dir}")
            return {
                "origin": rec.lineage.origin.value,
                "parent_skill_ids": rec.lineage.parent_skill_ids,
                "change_summary": rec.lineage.change_summary,
                "created_by": rec.lineage.created_by or "",
                "tags": rec.tags,
            }
    except Exception as e:
        logger.debug(f"DB upload metadata lookup failed: {e}")

    return {}


# ---------------------------------------------------------------------------
# Auto-registration / cloud import helpers
# ---------------------------------------------------------------------------
async def _auto_register_skill_dirs(skill_dirs: List[str]) -> int:
    """Register bot skill directories into OpenSpace's SkillRegistry + DB."""
    global _registered_skill_dirs

    valid_dirs = [Path(d) for d in skill_dirs if Path(d).is_dir()]
    if not valid_dirs:
        return 0

    openspace = await _get_openspace()
    registry = openspace._skill_registry
    if not registry:
        logger.warning("_auto_register_skill_dirs: SkillRegistry not initialized")
        return 0

    added = registry.discover_from_dirs(valid_dirs)

    db_created = 0
    if added:
        store = _get_store()
        db_created = await store.sync_from_registry(added)

    is_first = any(d not in _registered_skill_dirs for d in skill_dirs)
    for d in skill_dirs:
        _registered_skill_dirs.add(d)

    if added:
        action = "Auto-registered" if is_first else "Re-scanned & found"
        logger.info(f"{action} {len(added)} skill(s) from {len(valid_dirs)} dir(s), {db_created} new DB record(s)")
    return len(added)


async def _cloud_search_and_import(task: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Search cloud for skills relevant to *task* and auto-import top hits.

    Stage 1 of two-stage pipeline: cloud BM25+embedding → import top-N locally.
    Returns empty list when ``auto_import_enabled`` is ``False``.
    """
    if not _is_auto_import_enabled():
        logger.debug("Cloud auto-import is disabled (auto_import_enabled=False)")
        return []
    try:
        from openspace.cloud.embedding import generate_embedding, resolve_embedding_api
        from openspace.cloud.search import (
            SkillSearchEngine,
            build_cloud_candidates,
        )

        client = _get_cloud_client()
        embedding_api_key, _ = resolve_embedding_api()
        has_embedding = bool(embedding_api_key)

        items = await asyncio.to_thread(
            client.fetch_metadata,
            include_embedding=has_embedding,
            limit=200,
        )
        if not items:
            return []

        candidates = build_cloud_candidates(items)
        if not candidates:
            return []

        query_embedding: Optional[List[float]] = None
        if has_embedding:
            query_embedding = await asyncio.to_thread(
                generate_embedding,
                task,
            )

        engine = SkillSearchEngine()
        results = engine.search(task, candidates, query_embedding=query_embedding, limit=limit * 2)

        cloud_hits = [
            r
            for r in results
            if r.get("source") == "cloud" and r.get("visibility", "public") == "public" and r.get("skill_id")
        ][:limit]

        import_results: List[Dict[str, Any]] = []
        for hit in cloud_hits:
            try:
                imp = await _do_import_cloud_skill(skill_id=hit["skill_id"])
                import_results.append(
                    {
                        "skill_id": hit["skill_id"],
                        "name": hit.get("name", ""),
                        "import_status": imp.get("status", "error"),
                        "local_path": imp.get("local_path", ""),
                    }
                )
            except Exception as e:
                logger.warning(f"Cloud import failed for {hit['skill_id']}: {e}")

        if import_results:
            logger.info(f"Cloud search imported {len(import_results)} skill(s)")
        return import_results

    except Exception as e:
        logger.warning(f"_cloud_search_and_import failed (non-fatal): {e}")
        return []


async def _do_import_cloud_skill(skill_id: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
    """Download a cloud skill and register it locally.

    Refuses to proceed when ``auto_import_enabled`` is ``False``.
    """
    if not _is_auto_import_enabled():
        return {"status": "blocked", "reason": "auto_import_enabled is False"}
    client = _get_cloud_client()

    if target_dir:
        base_dir = Path(target_dir)
    else:
        host_ws = os.environ.get("NANOBOT_WORKSPACE") or os.environ.get("OPENCLAW_STATE_DIR")
        if host_ws:
            base_dir = Path(host_ws) / "skills"
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            openspace = await _get_openspace()
            skill_cfg = openspace._grounding_config.skills if openspace._grounding_config else None
            if skill_cfg and skill_cfg.skill_dirs:
                base_dir = Path(skill_cfg.skill_dirs[0])
            else:
                base_dir = Path(__file__).resolve().parent.parent / "skills"

    result = await asyncio.to_thread(client.import_skill, skill_id, base_dir)

    skill_dir = Path(result.get("local_path", ""))
    if skill_dir.exists():
        openspace = await _get_openspace()
        registry = openspace._skill_registry
        if registry:
            meta = registry.register_skill_dir(skill_dir)
            if meta:
                store = _get_store()
                await store.sync_from_registry([meta])
                result["registered"] = True

    result.setdefault("registered", False)
    return result


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def _format_task_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format an OpenSpace execution result for MCP transport."""
    tool_execs = result.get("tool_executions", [])
    tool_summary = [
        {
            "tool": te.get("tool_name", te.get("tool", "")),
            "status": te.get("status", ""),
            "error": te.get("error", "")[:200] if te.get("error") else None,
        }
        for te in tool_execs[:20]
    ]

    output: Dict[str, Any] = {
        "status": result.get("status", "unknown"),
        "response": result.get("response", ""),
        "execution_time": round(result.get("execution_time", 0), 2),
        "iterations": result.get("iterations", 0),
        "skills_used": result.get("skills_used", []),
        "task_id": result.get("task_id", ""),
        "tool_call_count": len(tool_execs),
        "tool_summary": tool_summary,
    }
    if result.get("warning"):
        output["warning"] = result["warning"]

    # Format evolved_skills with skill_dir and upload_ready flag
    raw_evolved = result.get("evolved_skills", [])
    if raw_evolved:
        formatted_evolved = []
        for es in raw_evolved:
            skill_path = es.get("path", "")
            skill_dir = str(Path(skill_path).parent) if skill_path else ""
            formatted_evolved.append(
                {
                    "skill_dir": skill_dir,
                    "name": es.get("name", ""),
                    "origin": es.get("origin", ""),
                    "change_summary": es.get("change_summary", ""),
                    "upload_ready": bool(skill_dir),
                }
            )
        output["evolved_skills"] = formatted_evolved
        names = [es["name"] for es in formatted_evolved if es.get("upload_ready")]
        if names:
            output["action_required"] = (
                f"OpenSpace auto-evolved {len(names)} skill(s): {', '.join(names)}. "
                f"Follow the 'When to upload' rules in your delegate-task skill to "
                f"decide visibility, then upload via upload_skill. "
                f"Tell the user what you evolved and what you uploaded."
            )

    return output


def _json_ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _json_error(error_msg: str, *, error_code: str = "VALIDATION_ERROR") -> str:
    """Return a structured error for validation / not-found cases."""
    from openspace.errors import safe_error_response

    return safe_error_response(error_code, error_msg)


# ---------------------------------------------------------------------------
# MCP Tool Handlers (4 tools)
# ---------------------------------------------------------------------------
async def execute_task(
    task: str,
    workspace_dir: str | None = None,
    max_iterations: int | None = None,
    skill_dirs: list[str] | None = None,
    search_scope: str = "all",
) -> str:
    """Execute a task with OpenSpace's full grounding engine.

    OpenSpace will:
    1. Auto-register bot skills from skill_dirs (if provided)
    2. Search for relevant skills (scope controls local vs cloud+local)
    3. Attempt skill-guided execution → fallback to pure tools
    4. Auto-analyze → auto-evolve (FIX/DERIVED/CAPTURED) if needed

    If skills are auto-evolved, the response includes ``evolved_skills``
    with ``upload_ready: true``.  Call ``upload_skill`` with just the
    ``skill_dir`` + ``visibility`` to upload — metadata is pre-saved.

    Note: This call blocks until the task completes (may take minutes).
    Set MCP client tool-call timeout ≥ 600 seconds.

    Args:
        task: The task instruction (natural language).
        workspace_dir: Working directory. Defaults to OPENSPACE_WORKSPACE env.
        max_iterations: Max agent iterations (default: 20).
        skill_dirs: Bot's skill directories to auto-register so OpenSpace
                    can select and track them.  Directories are re-scanned
                    on every call to discover skills created since the last
                    invocation.
        search_scope: Skill search scope before execution.
                      "all" (default) — local + cloud; falls back to local
                      if no API key is configured.
                      "local" — local SkillRegistry only (fast, no cloud).
    """
    try:
        openspace = await _get_openspace()

        # Re-scan host skill directories (from env)
        host_skill_dirs_raw = os.environ.get("OPENSPACE_HOST_SKILL_DIRS", "")
        if host_skill_dirs_raw:
            env_dirs = [d.strip() for d in host_skill_dirs_raw.split(",") if d.strip()]
            if env_dirs:
                await _auto_register_skill_dirs(env_dirs)

        # Auto-register bot skill directories (from call parameter)
        if skill_dirs:
            await _auto_register_skill_dirs(skill_dirs)

        # Cloud search + import (if requested)
        imported_skills: List[Dict[str, Any]] = []
        if search_scope == "all":
            imported_skills = await _cloud_search_and_import(task)

        # Execute
        result = await openspace.execute(
            task=task,
            workspace_dir=workspace_dir,
            max_iterations=max_iterations,
        )

        # Write .upload_meta.json for each evolved skill
        for es in result.get("evolved_skills", []):
            skill_path = es.get("path", "")
            if skill_path:
                _write_upload_meta(Path(skill_path).parent, es)

        formatted = _format_task_result(result)
        if imported_skills:
            formatted["imported_skills"] = imported_skills
        return _json_ok(formatted)

    except Exception as e:
        from openspace.errors import EXECUTION_ERROR, handle_mcp_exception

        return handle_mcp_exception(e, tool_name="execute_task", error_code=EXECUTION_ERROR)


async def search_skills(
    query: str,
    source: str = "all",
    limit: int = 20,
    auto_import: bool = True,
) -> str:
    """Search skills across local registry and cloud community.

    Standalone search for browsing / discovery.  Use this when the bot
    wants to find available skills, then decide whether to handle the
    task locally or delegate to ``execute_task``.

    **Scope difference from execute_task**:
      - ``search_skills`` returns results to the bot for decision-making.
      - ``execute_task``'s internal search feeds directly into execution
        (the bot never sees the search results).

    Uses hybrid ranking: BM25 → embedding re-rank → lexical boost.
    Embedding requires OPENAI_API_KEY; falls back to lexical-only without it.

    Args:
        query: Search query text (natural language or keywords).
        source: "all" (cloud + local), "local", or "cloud".  Default: "all".
        limit: Maximum results to return (default: 20).
        auto_import: Auto-download top public cloud skills (default: True).
    """
    try:
        from openspace.cloud.search import hybrid_search_skills

        q = query.strip()
        if not q:
            return _json_ok({"results": [], "count": 0})

        # Re-scan host skill directories so newly created skills are searchable.
        local_skills = None
        store = None
        if source in ("all", "local"):
            openspace = await _get_openspace()

            host_skill_dirs_raw = os.environ.get("OPENSPACE_HOST_SKILL_DIRS", "")
            if host_skill_dirs_raw:
                env_dirs = [d.strip() for d in host_skill_dirs_raw.split(",") if d.strip()]
                if env_dirs:
                    await _auto_register_skill_dirs(env_dirs)

            registry = openspace._skill_registry
            if registry:
                local_skills = registry.list_skills()
                store = _get_store()

        results = await hybrid_search_skills(
            query=q,
            local_skills=local_skills,
            store=store,
            source=source,
            limit=limit,
        )

        _AUTO_IMPORT_MAX = 3
        import_summary: List[Dict[str, Any]] = []
        if auto_import and _is_auto_import_enabled():
            cloud_results = [
                r
                for r in results
                if r.get("source") == "cloud" and r.get("visibility", "public") == "public" and r.get("skill_id")
            ][:_AUTO_IMPORT_MAX]
            for cr in cloud_results:
                try:
                    imp_result = await _do_import_cloud_skill(skill_id=cr["skill_id"])
                    status = imp_result.get("status", "error")
                    import_summary.append(
                        {
                            "skill_id": cr["skill_id"],
                            "name": cr.get("name", ""),
                            "import_status": status,
                            "local_path": imp_result.get("local_path", ""),
                        }
                    )
                    if status in ("success", "already_exists"):
                        cr["auto_imported"] = True
                        cr["local_path"] = imp_result.get("local_path", "")
                except Exception as imp_err:
                    logger.warning(f"auto_import failed for {cr['skill_id']}: {imp_err}", exc_info=True)
                    import_summary.append(
                        {
                            "skill_id": cr["skill_id"],
                            "import_status": "error",
                            "error": "Cloud skill import failed",
                        }
                    )

        output: Dict[str, Any] = {"results": results, "count": len(results)}
        if import_summary:
            output["auto_import_summary"] = import_summary
        return _json_ok(output)

    except Exception as e:
        from openspace.errors import EXECUTION_ERROR, handle_mcp_exception

        return handle_mcp_exception(e, tool_name="search_skills", error_code=EXECUTION_ERROR)


async def fix_skill(
    skill_dir: str,
    direction: str,
) -> str:
    """Manually fix a broken skill.

    This is the **only** manual evolution entry point.  DERIVED and
    CAPTURED evolutions are triggered automatically by ``execute_task``
    (they need a task to run).  Use ``fix_skill`` when:

      - A skill's instructions are wrong or outdated
      - The bot knows exactly which skill is broken and what to fix
      - Auto-evolution inside ``execute_task`` didn't catch the issue

    The skill does NOT need to be pre-registered in OpenSpace —
    provide the skill directory path and OpenSpace will register it
    automatically before fixing.

    After fixing, the new skill is saved locally and ``.upload_meta.json``
    is pre-written.  Call ``upload_skill`` with just ``skill_dir`` +
    ``visibility`` to upload.

    Args:
        skill_dir: Path to the broken skill directory (must contain SKILL.md).
        direction: What's broken and how to fix it.  Be specific:
                   e.g. "The API endpoint changed from v1 to v2" or
                   "Add retry logic for HTTP 429 rate limit errors".
    """
    try:
        from openspace.skill_engine.evolution.models import EvolutionContext, EvolutionTrigger
        from openspace.skill_engine.types import EvolutionSuggestion, EvolutionType

        if not direction:
            return _json_error("direction is required — describe what to fix.")

        skill_path = Path(skill_dir)
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            return _json_error("SKILL.md not found in the specified skill directory", error_code="SKILL_NOT_FOUND")

        openspace = await _get_openspace()
        registry = openspace._skill_registry
        if not registry:
            return _json_error("SkillRegistry not initialized", error_code="INTERNAL_ERROR")
        if not openspace._skill_evolver:
            return _json_error("Skill evolution is not enabled", error_code="INTERNAL_ERROR")

        # Step 1: Register the skill (idempotent)
        meta = registry.register_skill_dir(skill_path)
        if not meta:
            return _json_error("Failed to register skill from the specified directory", error_code="SKILL_NOT_FOUND")

        store = _get_store()
        await store.sync_from_registry([meta])

        # Step 2: Load record + content
        rec = store.load_record(meta.skill_id)
        if not rec:
            return _json_error("Failed to load skill record", error_code="SKILL_NOT_FOUND")

        evolver = openspace._skill_evolver
        content = evolver._load_skill_content(rec)
        if not content:
            return _json_error("Cannot load content for the specified skill", error_code="SKILL_NOT_FOUND")

        # Step 3: Run FIX evolution
        recent = store.load_analyses(skill_id=meta.skill_id, limit=5)

        ctx = EvolutionContext(
            trigger=EvolutionTrigger.ANALYSIS,
            suggestion=EvolutionSuggestion(
                evolution_type=EvolutionType.FIX,
                target_skill_ids=[meta.skill_id],
                direction=direction,
            ),
            skill_records=[rec],
            skill_contents=[content],
            skill_dirs=[skill_path],
            recent_analyses=recent,
            available_tools=evolver._available_tools,
        )

        logger.info(f"fix_skill: {meta.skill_id} — {direction[:100]}")
        new_record = await evolver.evolve(ctx)

        if not new_record:
            return _json_ok(
                {
                    "status": "failed",
                    "error": "Evolution did not produce a new skill.",
                }
            )

        # Step 4: Write .upload_meta.json
        new_skill_dir = Path(new_record.path).parent if new_record.path else skill_path
        _write_upload_meta(
            new_skill_dir,
            {
                "origin": new_record.lineage.origin.value,
                "parent_skill_ids": new_record.lineage.parent_skill_ids,
                "change_summary": new_record.lineage.change_summary,
                "created_by": new_record.lineage.created_by or "openspace",
                "tags": new_record.tags,
            },
        )

        return _json_ok(
            {
                "status": "success",
                "new_skill": {
                    "skill_dir": str(new_skill_dir),
                    "name": new_record.name,
                    "origin": new_record.lineage.origin.value,
                    "change_summary": new_record.lineage.change_summary,
                    "upload_ready": True,
                },
            }
        )

    except Exception as e:
        from openspace.errors import EXECUTION_ERROR, handle_mcp_exception

        return handle_mcp_exception(e, tool_name="fix_skill", error_code=EXECUTION_ERROR)


async def upload_skill(
    skill_dir: str,
    visibility: str = "public",
    origin: str | None = None,
    parent_skill_ids: list[str] | None = None,
    tags: list[str] | None = None,
    created_by: str | None = None,
    change_summary: str | None = None,
) -> str:
    """Upload a local skill to the cloud.

    For evolved skills (from ``execute_task`` or ``fix_skill``), most
    metadata is **pre-saved** in ``.upload_meta.json``.  The bot only
    needs to provide:

      - ``skill_dir`` — path to the skill directory
      - ``visibility`` — "public" or "private"

    All other parameters are optional overrides.  If omitted, pre-saved
    values are used.  If no pre-saved values exist, sensible defaults
    are applied.

    **origin + parent_skill_ids constraints** (enforced by cloud):
      - imported / captured → parent_skill_ids must be empty
      - derived → at least 1 parent
      - fixed → exactly 1 parent

    Args:
        skill_dir: Path to skill directory (must contain SKILL.md).
        visibility: "public" or "private".  This is the one thing the
                    bot MUST decide.
        origin: Override origin.  Default: from .upload_meta.json or "imported".
        parent_skill_ids: Override parents.  Default: from .upload_meta.json.
        tags: Override tags.  Default: from .upload_meta.json.
        created_by: Override creator.  Default: from .upload_meta.json.
        change_summary: Override summary.  Default: from .upload_meta.json.
    """
    try:
        skill_path = Path(skill_dir)
        if not (skill_path / "SKILL.md").exists():
            return _json_error("SKILL.md not found in the specified skill directory", error_code="SKILL_NOT_FOUND")

        # Read pre-saved metadata (written by execute_task/fix_skill)
        meta = _read_upload_meta(skill_path)

        # Merge: explicit params override pre-saved values
        final_origin = origin if origin is not None else meta.get("origin", "imported")
        final_parents = parent_skill_ids if parent_skill_ids is not None else meta.get("parent_skill_ids", [])
        final_tags = tags if tags is not None else meta.get("tags", [])
        final_created_by = created_by if created_by is not None else meta.get("created_by", "")
        final_change_summary = change_summary if change_summary is not None else meta.get("change_summary", "")

        client = _get_cloud_client()
        result = await asyncio.to_thread(
            client.upload_skill,
            skill_path,
            visibility=visibility,
            origin=final_origin,
            parent_skill_ids=final_parents,
            tags=final_tags,
            created_by=final_created_by,
            change_summary=final_change_summary,
        )
        return _json_ok(result)

    except Exception as e:
        from openspace.errors import EXECUTION_ERROR, handle_mcp_exception

        return handle_mcp_exception(e, tool_name="upload_skill", error_code=EXECUTION_ERROR)


# ---------------------------------------------------------------------------
# Observability tools (Epic 6.1)
# ---------------------------------------------------------------------------


async def health_check() -> str:
    """Check OpenSpace system health.

    Returns structured health status including all registered subsystem
    probes (skill store, LLM connectivity, grounding engine, etc.).
    """
    from openspace.observability.health import health

    result = health.check()
    return json.dumps(result, indent=2)


async def get_metrics() -> str:
    """Get Prometheus-compatible metrics.

    Returns all OpenSpace metrics in Prometheus exposition format, suitable
    for scraping by Prometheus, Grafana Agent, or similar tools.
    """
    from openspace.observability.metrics import metrics

    return metrics.render().decode("utf-8")


async def get_execution_traces(limit: int = 5) -> str:
    """Get recent execution traces for debugging.

    Returns the last N execution traces with span trees, timings, and
    metadata for debugging multi-step grounding workflows.

    Args:
        limit: Maximum number of traces to return (default: 5, max: 50)
    """
    from openspace.observability.tracing import tracer

    limit = min(max(1, limit), 50)
    traces = tracer.recent_traces[-limit:]
    return json.dumps([t.to_dict() for t in traces], indent=2, default=str)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_handlers(mcp) -> None:
    """Wire all tool handlers to the given FastMCP instance.

    Called by ``create_mcp_app()`` in ``openspace.mcp.server``.
    """
    mcp.tool()(execute_task)
    mcp.tool()(search_skills)
    mcp.tool()(fix_skill)
    mcp.tool()(upload_skill)
    # Observability (Epic 6.1)
    mcp.tool()(health_check)
    mcp.tool()(get_metrics)
    mcp.tool()(get_execution_traces)
