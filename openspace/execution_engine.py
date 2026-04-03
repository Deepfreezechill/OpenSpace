"""ExecutionEngine — task execution, two-phase orchestration, and post-exec analysis.

Extracted from ``OpenSpace`` (tool_layer.py) in Epic 4.3.  Owns:

- Task execution lifecycle (busy-wait, dispatch, error handling)
- Skill-first → tool-fallback two-phase orchestration
- Post-execution analysis and skill evolution triggers
- Workspace resolution and recording integration
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openspace.recording import RecordingManager
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.agents import GroundingAgent
    from openspace.grounding.core.grounding_client import GroundingClient
    from openspace.skill_engine import ExecutionAnalyzer, SkillRegistry, SkillStore
    from openspace.skill_engine.evolver import SkillEvolver
    from openspace.tool_layer import OpenSpaceConfig
    from openspace.tool_registry import ToolRegistry

logger = Logger.get_logger(__name__)


class ExecutionEngine:
    """Manages task execution lifecycle and two-phase skill orchestration.

    Parameters
    ----------
    config:
        The ``OpenSpaceConfig`` for iteration budgets, workspace, etc.
    grounding_agent:
        The agent that processes tasks.
    grounding_client:
        The grounding client (for quality manager access).
    tool_registry:
        Optional ``ToolRegistry`` for skill selection/injection.
    skill_registry:
        Optional ``SkillRegistry`` (presence gates skill-first phase).
    skill_store:
        Optional ``SkillStore`` for quality metric forwarding.
    recording_manager:
        Optional ``RecordingManager`` for task recording.
    execution_analyzer:
        Optional ``ExecutionAnalyzer`` for post-exec analysis.
    skill_evolver:
        Optional ``SkillEvolver`` for skill evolution triggers.
    """

    def __init__(
        self,
        *,
        config: OpenSpaceConfig,
        grounding_agent: Optional[GroundingAgent],
        grounding_client: Optional[GroundingClient],
        tool_registry: Optional[ToolRegistry] = None,
        skill_registry: Optional[SkillRegistry] = None,
        skill_store: Optional[SkillStore] = None,
        recording_manager: Optional[RecordingManager] = None,
        execution_analyzer: Optional[ExecutionAnalyzer] = None,
        skill_evolver: Optional[SkillEvolver] = None,
    ) -> None:
        self._config = config
        self._grounding_agent = grounding_agent
        self._grounding_client = grounding_client
        self._tool_registry = tool_registry
        self._skill_registry = skill_registry
        self._skill_store = skill_store
        self._recording_manager = recording_manager
        self._execution_analyzer = execution_analyzer
        self._skill_evolver = skill_evolver

        self._running = False
        self._task_done = asyncio.Event()
        self._task_done.set()
        self._last_evolved_skills: List[Dict[str, Any]] = []
        self._execution_count: int = 0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def last_evolved_skills(self) -> List[Dict[str, Any]]:
        return list(self._last_evolved_skills)

    # ── Main execution ────────────────────────────────────────────────

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[str] = None,
        max_iterations: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a task with two-phase skill orchestration.

        Phase 1 (Skill-First): If skills are available, select and inject
        relevant skills, then run the agent.  If the skill-guided phase
        fails, clean up workspace artifacts and fall back to Phase 2.

        Phase 2 (Tool-Fallback): Run the agent with no skill context,
        relying on raw tool capabilities.

        Returns a result dict with status, execution_time, skills_used, etc.
        """
        if not self._grounding_agent:
            raise RuntimeError(
                "ExecutionEngine not initialized. No grounding agent available."
            )

        _TASK_WAIT_TIMEOUT = 660
        if self._running:
            logger.info(
                "OpenSpace is busy — waiting up to %ds for the current task to finish...",
                _TASK_WAIT_TIMEOUT,
            )
            try:
                await asyncio.wait_for(
                    self._task_done.wait(), timeout=_TASK_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"OpenSpace is still running after waiting {_TASK_WAIT_TIMEOUT}s. "
                    f"Please try again later."
                )

        logger.info("=" * 60)
        logger.info(f"Task: {task[:100]}...")
        logger.info("=" * 60)

        self._running = True
        self._task_done.clear()
        self._last_evolved_skills = []
        start_time = asyncio.get_event_loop().time()

        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        logger.info(f"Task ID: {task_id}")

        result: Dict[str, Any] = {}

        try:
            execution_context = context or {}
            execution_context["task_id"] = task_id
            execution_context["instruction"] = task

            if max_iterations is not None:
                execution_context["max_iterations"] = max_iterations

            # Recording setup
            if self._recording_manager:
                if self._recording_manager.recording_status:
                    await self._recording_manager.stop()
                    logger.debug("Stopped previous recording session")

                self._recording_manager.task_id = task_id
                await self._recording_manager.start()
                await self._recording_manager.add_metadata("instruction", task)
                logger.info(f"Recording started: {task_id}")

            # Workspace resolution
            self._resolve_workspace(execution_context, workspace_dir, task_id)

            # Sync shell session working dir
            resolved_ws = execution_context["workspace_dir"]
            try:
                from openspace.grounding.core.types import BackendType as _BT

                shell_prov = self._grounding_client._registry.get(_BT.SHELL)
                for sess in shell_prov._sessions.values():
                    sess.default_working_dir = resolved_ws
            except Exception:
                pass

            # Resolve iteration budget
            configured_max = self._config.grounding_max_iterations
            if max_iterations:
                max_iterations = max(max_iterations, configured_max)
            else:
                max_iterations = configured_max

            # Two-phase execution: Skill-First → Tool-Fallback
            has_skills = False

            if self._skill_registry and self._tool_registry:
                has_skills = await self._tool_registry.select_and_inject(
                    task,
                    agent=self._grounding_agent,
                    store=self._skill_store,
                    recording_mgr=self._recording_manager,
                )

            if has_skills:
                result = await self._execute_skill_first(
                    task, execution_context, max_iterations
                )
            else:
                logger.info(
                    f"Executing with GroundingAgent "
                    f"(max {max_iterations} iterations, no skills)..."
                )
                result = await self._grounding_agent.process(execution_context)

            execution_time = asyncio.get_event_loop().time() - start_time
            self._log_result(result, execution_time)

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            tb = traceback.format_exc(limit=10)
            logger.error(f"Task execution failed: {e}", exc_info=True)

            result = {
                "status": "error",
                "error": str(e),
                "traceback": tb,
                "response": f"Task execution error: {str(e)}",
                "execution_time": execution_time,
                "task_id": task_id,
                "iterations": 0,
                "tool_executions": [],
            }

        finally:
            recording_dir = None
            if (
                self._recording_manager
                and self._recording_manager.recording_status
            ):
                recording_dir = self._recording_manager.trajectory_dir

                try:
                    exec_time = asyncio.get_event_loop().time() - start_time
                    await self._recording_manager.save_execution_outcome(
                        status=result.get("status", "unknown"),
                        iterations=result.get("iterations", 0),
                        execution_time=exec_time,
                    )
                except Exception:
                    pass

                try:
                    await self._recording_manager.stop()
                    logger.debug(f"Recording stopped: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to stop recording: {e}")

            await self._maybe_analyze_execution(task_id, recording_dir, result)
            await self._maybe_evolve_quality()

            final_result = {
                **result,
                "task_id": task_id,
                "execution_time": execution_time,
                "skills_used": result.get("active_skills", []),
                "evolved_skills": list(self._last_evolved_skills),
            }

            self._running = False
            self._task_done.set()

            return final_result

    # ── Skill-first phase ─────────────────────────────────────────────

    async def _execute_skill_first(
        self,
        task: str,
        execution_context: Dict[str, Any],
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Run skill-guided execution with fallback on failure."""
        logger.info(
            f"[Phase 1 — Skill] Executing with skill guidance "
            f"(max {max_iterations} iterations)..."
        )
        execution_context_p1 = {**execution_context}
        execution_context_p1["max_iterations"] = max_iterations

        # Snapshot workspace for cleanup on failure
        workspace_path = execution_context.get("workspace_dir", "")
        pre_skill_files: set = set()
        if workspace_path:
            try:
                ws = Path(workspace_path)
                pre_skill_files = (
                    {f.name for f in ws.iterdir()} if ws.exists() else set()
                )
            except Exception:
                pass

        injected_skill_ids = list(self._grounding_agent._active_skill_ids)
        skill_phase_result = await self._grounding_agent.process(
            execution_context_p1
        )
        skill_status = skill_phase_result.get("status", "unknown")
        skill_iterations = skill_phase_result.get("iterations", 0)

        self._grounding_agent.clear_skill_context()

        if skill_status == "success":
            result = skill_phase_result
            result["active_skills"] = injected_skill_ids
            logger.info(
                f"[Phase 1 — Skill] Completed successfully "
                f"({skill_iterations} iterations)"
            )
            return result

        # Skill failed — fall back
        logger.warning(
            f"[Phase 1 — Skill] {skill_status} after {skill_iterations} "
            f"iterations, falling back to tool-only execution "
            f"(budget: {max_iterations})"
        )

        self._cleanup_workspace(workspace_path, pre_skill_files)

        execution_context_p2 = {**execution_context}
        execution_context_p2["max_iterations"] = max_iterations

        result = await self._grounding_agent.process(execution_context_p2)
        result["active_skills"] = injected_skill_ids
        logger.info(
            f"[Phase 2 — Fallback] {result.get('status', 'unknown')} "
            f"({result.get('iterations', 0)} iterations)"
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────────

    def _resolve_workspace(
        self,
        execution_context: Dict[str, Any],
        workspace_dir: Optional[str],
        task_id: str,
    ) -> None:
        """Resolve workspace directory into execution_context."""
        if workspace_dir:
            execution_context["workspace_dir"] = workspace_dir
            logger.info(f"Workspace: {workspace_dir}")
        elif self._config.workspace_dir:
            execution_context["workspace_dir"] = self._config.workspace_dir
            logger.info(f"Workspace: {self._config.workspace_dir}")
        elif (
            self._recording_manager
            and self._recording_manager.trajectory_dir
        ):
            execution_context["workspace_dir"] = (
                self._recording_manager.trajectory_dir
            )
            logger.info(f"Workspace: {execution_context['workspace_dir']}")
        else:
            import tempfile

            workspace = (
                Path(tempfile.gettempdir()) / "openspace_workspace" / task_id
            )
            workspace.mkdir(parents=True, exist_ok=True)
            execution_context["workspace_dir"] = str(workspace)
            logger.info(f"Workspace: {execution_context['workspace_dir']}")

    @staticmethod
    def _cleanup_workspace(
        workspace_path: str, pre_skill_files: set
    ) -> None:
        """Remove artifacts created by the failed skill phase."""
        if not workspace_path:
            return
        try:
            import shutil

            ws = Path(workspace_path)
            removed = 0
            if ws.exists():
                for f in list(ws.iterdir()):
                    if f.name not in pre_skill_files:
                        if f.is_dir():
                            shutil.rmtree(f, ignore_errors=True)
                        else:
                            f.unlink(missing_ok=True)
                        removed += 1
            if removed:
                logger.info(
                    f"[Phase 2 — Fallback] Cleaned {removed} artifact(s) "
                    f"from failed skill-guided phase"
                )
        except Exception as e:
            logger.debug(f"Workspace cleanup failed: {e}")

    @staticmethod
    def _log_result(
        result: Dict[str, Any], execution_time: float
    ) -> None:
        """Log execution outcome."""
        status = result.get("status", "unknown")
        iterations = result.get("iterations", 0)
        tool_count = len(result.get("tool_executions", []))

        logger.info("=" * 60)
        if status == "success":
            logger.info(
                f"Task completed successfully! "
                f"({iterations} iterations, {tool_count} tool calls, "
                f"{execution_time:.2f}s)"
            )
        elif status == "incomplete":
            logger.warning(
                f"Task incomplete after {iterations} iterations. "
                f"Consider increasing max_iterations."
            )
        else:
            logger.error(
                f"Task failed: {result.get('error', 'Unknown error')}"
            )
        logger.info("=" * 60)

    # ── Post-execution hooks ──────────────────────────────────────────

    async def _maybe_analyze_execution(
        self,
        task_id: str,
        recording_dir: Optional[str],
        execution_result: Dict[str, Any],
    ) -> None:
        """Run post-execution analysis if enabled."""
        if not self._execution_analyzer or not recording_dir:
            return
        try:
            agent_tools = (
                getattr(self._grounding_agent, "_last_tools", [])
                if self._grounding_agent
                else []
            )

            analysis = await self._execution_analyzer.analyze_execution(
                task_id=task_id,
                recording_dir=recording_dir,
                execution_result=execution_result,
                available_tools=agent_tools,
            )
            if not analysis:
                return

            if analysis.candidate_for_evolution and self._skill_evolver:
                self._skill_evolver.set_available_tools(agent_tools)

                evo_summary = ", ".join(
                    f"{s.evolution_type.value}"
                    f"({'+'.join(s.target_skill_ids) or 'new'})"
                    for s in analysis.evolution_suggestions
                )
                logger.info(f"[Skill Evolution] Suggestions: {evo_summary}")
                evolved_records = await self._skill_evolver.process_analysis(
                    analysis
                )

                for rec in evolved_records:
                    self._last_evolved_skills.append(
                        {
                            "skill_id": rec.skill_id,
                            "name": rec.name,
                            "description": rec.description,
                            "path": str(rec.path) if rec.path else "",
                            "origin": rec.lineage.origin.value,
                            "generation": rec.lineage.generation,
                            "parent_skill_ids": rec.lineage.parent_skill_ids,
                            "change_summary": rec.lineage.change_summary,
                        }
                    )

        except Exception as e:
            logger.debug(f"Execution analysis skipped: {e}")

    async def _maybe_evolve_quality(self) -> None:
        """Trigger quality evolution based on global execution count."""
        self._execution_count += 1
        quality_mgr = (
            self._grounding_client.quality_manager
            if self._grounding_client
            else None
        )

        if self._skill_evolver and self._grounding_agent:
            agent_tools = getattr(self._grounding_agent, "_last_tools", [])
            if agent_tools:
                self._skill_evolver.set_available_tools(agent_tools)

        if quality_mgr and quality_mgr.should_evolve():
            try:
                report = await self._grounding_client.evolve_quality()
                if report.get("recommendations"):
                    logger.info(
                        f"Quality evolution: {report['recommendations']}"
                    )

                if self._skill_evolver:
                    problematic = quality_mgr.get_problematic_tools()
                    if problematic:
                        logger.info(
                            f"[Trigger:tool_degradation] "
                            f"{len(problematic)} problematic tool(s) detected"
                        )
                        self._skill_evolver.schedule_background(
                            self._skill_evolver.process_tool_degradation(
                                problematic
                            ),
                            label="trigger2_tool_degradation",
                        )

            except Exception as e:
                logger.debug(f"Quality evolution skipped: {e}")

        if self._skill_evolver and self._execution_count % 5 == 0:
            try:
                self._skill_evolver.schedule_background(
                    self._skill_evolver.process_metric_check(),
                    label="trigger3_metric_check",
                )
            except Exception as e:
                logger.debug(f"Skill metric check skipped: {e}")
