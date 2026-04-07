"""Evolution strategies — FIX, DERIVED, and CAPTURED.

Each strategy takes an ``evolver`` (``SkillEvolver`` instance) and an
``EvolutionContext``, runs an LLM agent loop, applies the edit with
retry, persists the new ``SkillRecord``, and registers the result.

All internal calls go through ``evolver._method()`` to preserve method
resolution order for subclass / hook / telemetry compatibility.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from openspace.prompts import SkillEnginePrompts
from openspace.utils.logging import Logger

from ..patch import (
    SKILL_FILENAME,
    PatchType,
    create_skill,
    derive_skill,
    fix_skill,
)
from ..registry import write_skill_id
from ..skill_utils import (
    extract_change_summary as _extract_change_summary,
    get_frontmatter_field as _extract_frontmatter_field,
    set_frontmatter_field as _set_frontmatter_field,
    truncate as _truncate,
)
from ..types import (
    SkillCategory,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
)
from .confirmation import _SKILL_CONTENT_MAX_CHARS
from .models import EvolutionContext, _sanitize_skill_name
from .triggers import _ANALYSIS_CONTEXT_MAX

logger = Logger.get_logger("openspace.skill_engine.evolver")


async def evolve_fix(evolver, ctx: EvolutionContext) -> Optional[SkillRecord]:
    """In-place fix: same name, same directory, new version record.

    Uses agent loop for information gathering + apply-retry cycle.
    """
    if not ctx.skill_records or not ctx.skill_contents or not ctx.skill_dirs:
        logger.warning("FIX requires exactly 1 parent (skill_records/contents/dirs)")
        return None

    parent = ctx.skill_records[0]
    parent_content = ctx.skill_contents[0]
    parent_dir = ctx.skill_dirs[0]

    # Build prompt with full directory content for multi-file skills
    dir_content = evolver._format_skill_dir_content(parent_dir)
    prompt = SkillEnginePrompts.evolution_fix(
        current_content=_truncate(dir_content or parent_content, _SKILL_CONTENT_MAX_CHARS),
        direction=ctx.suggestion.direction,
        failure_context=evolver._format_analysis_context(ctx.recent_analyses),
        tool_issue_summary=ctx.tool_issue_summary,
        metric_summary=ctx.metric_summary,
    )

    # Agent loop: LLM can gather information via tools before generating edits
    new_content = await evolver._run_evolution_loop(prompt, ctx)
    if not new_content:
        return None

    # Extract change_summary from LLM output (first line if prefixed)
    new_content, change_summary = _extract_change_summary(new_content)

    # Apply-retry cycle
    edit_result = await evolver._apply_with_retry(
        apply_fn=lambda content: fix_skill(parent_dir, content, PatchType.AUTO),
        initial_content=new_content,
        skill_dir=parent_dir,
        ctx=ctx,
        prompt=prompt,
    )
    if edit_result is None or not edit_result.ok:
        return None

    # Re-read name/description from the updated SKILL.md on disk —
    # the LLM may have refined the description (or even name) during the fix.
    updated_skill_md = edit_result.content_snapshot.get(SKILL_FILENAME, "")
    fixed_name = _extract_frontmatter_field(updated_skill_md, "name") or parent.name
    fixed_desc = _extract_frontmatter_field(updated_skill_md, "description") or parent.description

    new_id = f"{fixed_name}__v{parent.lineage.generation + 1}_{uuid.uuid4().hex[:8]}"
    model = evolver._model or evolver._llm_client.model

    new_record = SkillRecord(
        skill_id=new_id,
        name=fixed_name,
        description=fixed_desc,
        path=parent.path,
        category=parent.category,
        tags=list(parent.tags),
        visibility=parent.visibility,
        creator_id=parent.creator_id,
        lineage=SkillLineage(
            origin=SkillOrigin.FIXED,
            generation=parent.lineage.generation + 1,
            parent_skill_ids=[parent.skill_id],
            source_task_id=ctx.source_task_id,
            change_summary=change_summary or ctx.suggestion.direction,
            content_diff=edit_result.content_diff,
            content_snapshot=edit_result.content_snapshot,
            created_by=model,
        ),
        tool_dependencies=list(parent.tool_dependencies),
        critical_tools=list(parent.critical_tools),
    )

    result = await evolver._guard.guarded_evolve(new_record, [parent.skill_id])

    if not result.passed:
        # Guard rejected — dangerous code is on disk from _apply_with_retry.
        # Restore the parent directory from the ORIGINAL content_snapshot
        # that existed before our edit attempt.
        logger.warning(
            f"FIX: guard rejected {new_id} — skill NOT persisted or registered"
        )
        return None

    # Stamp the new skill_id into the sidecar file so next discover()
    write_skill_id(parent_dir, new_id)

    from ..registry import SkillMeta

    new_meta = SkillMeta(
        skill_id=new_id,
        name=fixed_name,
        description=fixed_desc,
        path=Path(parent.path),
    )
    evolver._registry.update_skill(parent.skill_id, new_meta)

    logger.info(
        f"FIX: {parent.name} gen{parent.lineage.generation} → gen{new_record.lineage.generation} [{new_id}]"
    )
    return new_record


async def evolve_derived(evolver, ctx: EvolutionContext) -> Optional[SkillRecord]:
    """Create enhanced version in a new directory.

    Supports single-parent (enhance) and multi-parent (merge/fuse).
    Uses agent loop for information gathering + apply-retry cycle.
    """
    if not ctx.skill_records or not ctx.skill_contents or not ctx.skill_dirs:
        logger.warning("DERIVED requires at least one parent skill_record + content + dir")
        return None

    first_parent = ctx.skill_records[0]  # For fallback defaults only
    is_merge = len(ctx.skill_records) > 1

    # Build prompt — include all parent contents for multi-parent merge
    if is_merge:
        parent_sections = []
        for i, (rec, sd) in enumerate(zip(ctx.skill_records, ctx.skill_dirs)):
            dir_content = evolver._format_skill_dir_content(sd)
            label = f"Parent {i + 1}: {rec.name}"
            parent_sections.append(
                f"## {label}\n{_truncate(dir_content or ctx.skill_contents[i], _SKILL_CONTENT_MAX_CHARS)}"
            )
        combined_content = "\n\n---\n\n".join(parent_sections)
    else:
        dir_content = evolver._format_skill_dir_content(ctx.skill_dirs[0])
        combined_content = _truncate(dir_content or ctx.skill_contents[0], _SKILL_CONTENT_MAX_CHARS)

    prompt = SkillEnginePrompts.evolution_derived(
        parent_content=combined_content,
        direction=ctx.suggestion.direction,
        execution_insights=evolver._format_analysis_context(ctx.recent_analyses),
        metric_summary=ctx.metric_summary,
    )

    # Agent loop
    new_content = await evolver._run_evolution_loop(prompt, ctx)
    if not new_content:
        return None

    new_content, change_summary = _extract_change_summary(new_content)

    # Determine new skill name from frontmatter, or generate one
    new_name = _extract_frontmatter_field(new_content, "name")
    if not new_name or new_name == first_parent.name:
        suffix = "-merged" if is_merge else "-enhanced"
        new_name = f"{first_parent.name}{suffix}"
        new_content = _set_frontmatter_field(new_content, "name", new_name)

    # Cap name length to avoid ever-growing chains like
    # "panel-component-enhanced-enhanced-merged_abc123"
    new_name = _sanitize_skill_name(new_name)
    new_content = _set_frontmatter_field(new_content, "name", new_name)

    # Directory name always matches the skill name
    target_dir = ctx.skill_dirs[0].parent / new_name
    if target_dir.exists():
        new_name = f"{new_name}-{uuid.uuid4().hex[:6]}"
        new_name = _sanitize_skill_name(new_name)
        target_dir = ctx.skill_dirs[0].parent / new_name
        new_content = _set_frontmatter_field(new_content, "name", new_name)

    # Apply-retry cycle for derive_skill
    edit_result = await evolver._apply_with_retry(
        apply_fn=lambda content: derive_skill(ctx.skill_dirs, target_dir, content, PatchType.AUTO),
        initial_content=new_content,
        skill_dir=target_dir,
        ctx=ctx,
        prompt=prompt,
        cleanup_on_retry=target_dir,  # Remove failed target dir before retry
    )
    if edit_result is None or not edit_result.ok:
        return None

    # Extract description from new content
    new_desc = _extract_frontmatter_field(new_content, "description") or first_parent.description

    # Collect parent info from ALL parents
    parent_ids = [r.skill_id for r in ctx.skill_records]
    max_gen = max(r.lineage.generation for r in ctx.skill_records)
    all_tool_deps: set = set()
    all_critical: set = set()
    all_tags: set = set()
    for rec in ctx.skill_records:
        all_tool_deps.update(rec.tool_dependencies)
        all_critical.update(rec.critical_tools)
        all_tags.update(rec.tags)

    new_id = f"{new_name}__v0_{uuid.uuid4().hex[:8]}"
    model = evolver._model or evolver._llm_client.model

    new_record = SkillRecord(
        skill_id=new_id,
        name=new_name,
        description=new_desc,
        path=str(target_dir / SKILL_FILENAME),
        category=ctx.suggestion.category or first_parent.category,
        tags=sorted(all_tags),
        visibility=first_parent.visibility,
        creator_id=first_parent.creator_id,
        lineage=SkillLineage(
            origin=SkillOrigin.DERIVED,
            generation=max_gen + 1,
            parent_skill_ids=parent_ids,
            source_task_id=ctx.source_task_id,
            change_summary=change_summary or ctx.suggestion.direction,
            content_diff=edit_result.content_diff,
            content_snapshot=edit_result.content_snapshot,
            created_by=model,
        ),
        tool_dependencies=sorted(all_tool_deps),
        critical_tools=sorted(all_critical),
    )

    result = await evolver._guard.guarded_evolve(new_record, parent_ids)

    if not result.passed:
        # Guard rejected — remove the newly created target directory
        logger.warning(
            f"DERIVED: guard rejected {new_id} — cleaning up {target_dir}"
        )
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return None

    # Stamp skill_id sidecar so discover() uses this ID on restart
    write_skill_id(target_dir, new_id)

    # Register the new skill so it's immediately available for selection
    from ..registry import SkillMeta

    new_meta = SkillMeta(
        skill_id=new_id,
        name=new_name,
        description=new_desc,
        path=target_dir / SKILL_FILENAME,
    )
    evolver._registry.add_skill(new_meta)

    parent_names = " + ".join(r.name for r in ctx.skill_records)
    logger.info(f"DERIVED: {parent_names} → {new_name} [{new_id}]")
    return new_record


async def evolve_captured(evolver, ctx: EvolutionContext) -> Optional[SkillRecord]:
    """Capture a novel pattern as a brand-new skill.

    Uses agent loop for information gathering + apply-retry cycle.
    """
    # Build prompt and call LLM
    # For CAPTURED, we use analyses as context (the tasks where the pattern was observed)
    task_descriptions = []
    for a in ctx.recent_analyses[:_ANALYSIS_CONTEXT_MAX]:
        if a.execution_note:
            task_descriptions.append(f"- task={a.task_id}: {a.execution_note[:200]}")

    prompt = SkillEnginePrompts.evolution_captured(
        direction=ctx.suggestion.direction,
        category=(ctx.suggestion.category or SkillCategory.WORKFLOW).value,
        execution_highlights="\n".join(task_descriptions) if task_descriptions else "(no task context available)",
    )

    # Agent loop
    new_content = await evolver._run_evolution_loop(prompt, ctx)
    if not new_content:
        return None

    new_content, change_summary = _extract_change_summary(new_content)

    # Extract name/description from the generated content
    new_name = _extract_frontmatter_field(new_content, "name")
    new_desc = _extract_frontmatter_field(new_content, "description")
    if not new_name:
        logger.warning("CAPTURED: LLM did not produce a valid skill name")
        return None

    # Sanitize name (enforce length limit + valid chars)
    new_name = _sanitize_skill_name(new_name)
    new_content = _set_frontmatter_field(new_content, "name", new_name)

    # Create new skill directory via create_skill (handles multi-file FULL)
    skill_dirs = evolver._registry._skill_dirs
    if not skill_dirs:
        logger.warning("CAPTURED: no skill directories configured")
        return None

    # Directory name always matches the skill name
    base_dir = skill_dirs[0]  # Primary user skill directory
    target_dir = base_dir / new_name
    if target_dir.exists():
        new_name = f"{new_name}-{uuid.uuid4().hex[:6]}"
        new_name = _sanitize_skill_name(new_name)
        target_dir = base_dir / new_name
        new_content = _set_frontmatter_field(new_content, "name", new_name)

    # Apply-retry cycle for create_skill
    edit_result = await evolver._apply_with_retry(
        apply_fn=lambda content: create_skill(target_dir, content, PatchType.AUTO),
        initial_content=new_content,
        skill_dir=target_dir,
        ctx=ctx,
        prompt=prompt,
        cleanup_on_retry=target_dir,
    )
    if edit_result is None or not edit_result.ok:
        return None

    snapshot = edit_result.content_snapshot
    add_all_diff = edit_result.content_diff

    new_id = f"{new_name}__v0_{uuid.uuid4().hex[:8]}"
    model = evolver._model or evolver._llm_client.model

    new_record = SkillRecord(
        skill_id=new_id,
        name=new_name,
        description=new_desc or new_name,
        path=str(target_dir / SKILL_FILENAME),
        category=ctx.suggestion.category or SkillCategory.WORKFLOW,
        lineage=SkillLineage(
            origin=SkillOrigin.CAPTURED,
            generation=0,
            parent_skill_ids=[],
            source_task_id=ctx.source_task_id,
            change_summary=change_summary or ctx.suggestion.direction,
            content_diff=add_all_diff,
            content_snapshot=snapshot,
            created_by=model,
        ),
    )

    result = await evolver._guard.guarded_save(new_record)

    if not result.passed:
        # Guard rejected — remove the newly created target directory
        logger.warning(
            f"CAPTURED: guard rejected {new_id} — cleaning up {target_dir}"
        )
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return None

    # Stamp skill_id sidecar so discover() uses this ID on restart
    write_skill_id(target_dir, new_id)

    # Register the new skill so it's immediately available
    from ..registry import SkillMeta

    new_meta = SkillMeta(
        skill_id=new_id,
        name=new_name,
        description=new_desc or new_name,
        path=target_dir / SKILL_FILENAME,
    )
    evolver._registry.add_skill(new_meta)

    logger.info(f"CAPTURED: {new_name} [{new_id}]")
    return new_record
