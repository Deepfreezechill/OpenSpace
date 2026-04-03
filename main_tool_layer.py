from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openspace.agents import GroundingAgent
from openspace.app.container import AppContainer
from openspace.config import get_config, load_config
from openspace.config.loader import get_agent_config
from openspace.grounding.core.grounding_client import GroundingClient
from openspace.execution_engine import ExecutionEngine
from openspace.llm import LLMClient
from openspace.llm_factory import LLMFactory
from openspace.recording_service import RecordingService
from openspace.skill_engine import ExecutionAnalyzer, SkillRegistry, SkillStore
from openspace.skill_engine.evolver import SkillEvolver
from openspace.tool_registry import ToolRegistry
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.recording import RecordingManager

logger = Logger.get_logger(__name__)


@dataclass
class OpenSpaceConfig:
    # LLM Configuration
    llm_model: str = "openrouter/anthropic/claude-sonnet-4.5"
    llm_enable_thinking: bool = False
    llm_timeout: float = 120.0
    llm_max_retries: int = 3
    llm_rate_limit_delay: float = 0.0
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Separate models for specific tasks (None = use llm_model)
    tool_retrieval_model: Optional[str] = None  # Model for tool retrieval LLM filter
    visual_analysis_model: Optional[str] = None  # Model for visual analysis

    # Skill Engine Models ΓÇö names map to class names (None = use llm_model)
    skill_registry_model: Optional[str] = None  # SkillRegistry: skill selection
    execution_analyzer_model: Optional[str] = None  # ExecutionAnalyzer: post-execution analysis
    skill_evolver_model: Optional[str] = None  # (future) SkillEvolver: skill evolution

    # Grounding Configuration
    grounding_config_path: Optional[str] = None
    grounding_max_iterations: int = 20
    grounding_system_prompt: Optional[str] = None

    # Backend Configuration
    backend_scope: Optional[List[str]] = None  # None = All backends ["shell", "gui", "mcp", "web", "system"]
    use_clawwork_productivity: bool = False  # If True, add ClawWork productivity tools (search_web, create_file, etc.) for fair comparison with ClawWork; requires livebench installed.

    # Workspace Configuration
    workspace_dir: Optional[str] = None

    # Recording Configuration
    enable_recording: bool = True
    recording_backends: Optional[List[str]] = None
    recording_log_dir: str = "./logs/recordings"
    enable_screenshot: bool = False
    enable_video: bool = False
    enable_conversation_log: bool = True  # Save LLM conversations to conversations.jsonl

    # Skill Evolution
    evolution_max_concurrent: int = 3  # Max parallel evolutions per trigger

    # Logging Configuration
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration"""
        if not self.llm_model:
            raise ValueError("llm_model is required")

        logger.debug(f"OpenSpaceConfig initialized with model: {self.llm_model}")


class OpenSpace:
    """High-level SkillGuard orchestration facade.

    Supports two creation paths:

    **Legacy** (backward-compatible)::

        cs = OpenSpace(config=OpenSpaceConfig(...))
        await cs.initialize()

    **Container-based** (Phase 1 seam ΓÇö Phase 4 wires initialize)::

        container = await build_container(config, llm=..., ...)
        cs = OpenSpace.from_container(container, config=config)
        await cs.initialize()  # still constructs services internally

    .. warning::

        In Phase 1, ``initialize()`` does **not** resolve services from
        the container.  The container is stored for use by callers that
        need typed access (via public properties) and will be fully
        wired in Phase 4.  Do **not** assume that injecting services
        into the container enforces them at runtime until Phase 4.

    Parameters
    ----------
    config:
        Application config.  Defaults to ``OpenSpaceConfig()``.
    container:
        Optional :class:`AppContainer`.  In Phase 1 the container is
        stored for property access only ΓÇö ``initialize()`` still
        constructs services internally.  Phase 4 will wire
        ``initialize()`` to resolve services from the container.
    """

    def __init__(
        self,
        config: Optional[OpenSpaceConfig] = None,
        *,
        container: Optional[AppContainer] = None,
    ):
        self.config = config or OpenSpaceConfig()
        self._container = container or AppContainer()

        self._llm_client: Optional[LLMClient] = None
        self._llm_factory: Optional[LLMFactory] = None
        self._grounding_client: Optional[GroundingClient] = None
        self._grounding_config = None  # GroundingConfig reference for skill settings
        self._grounding_agent: Optional[GroundingAgent] = None
        self._recording_manager: Optional[RecordingManager] = None
        self._recording_service: Optional[RecordingService] = None
        self._skill_registry: Optional[SkillRegistry] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._execution_engine: Optional[ExecutionEngine] = None
        self._skill_store: Optional[SkillStore] = None
        self._execution_analyzer: Optional[ExecutionAnalyzer] = None
        self._skill_evolver: Optional[SkillEvolver] = None
        self._initialized = False
        self._running = False  # Fallback for pre-init; delegates to engine after init

        logger.debug("OpenSpace instance created")

    # ΓöÇΓöÇ Factory methods ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    @classmethod
    def from_container(
        cls,
        container: AppContainer,
        config: Optional[OpenSpaceConfig] = None,
    ) -> "OpenSpace":
        """Create an OpenSpace instance backed by an AppContainer.

        .. note::

            Phase 1 only stores the container for property access.
            ``initialize()`` still constructs its own services.
            Phase 4 will wire ``initialize()`` to resolve from the
            container, making injected services authoritative.
        """
        return cls(config=config, container=container)

    # ΓöÇΓöÇ Public property accessors (replace private field access) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    @property
    def container(self) -> AppContainer:
        """The underlying :class:`AppContainer`."""
        return self._container

    @property
    def llm_client(self) -> Optional[LLMClient]:
        """The LLM client, or ``None`` if not initialized."""
        return self._llm_client

    @property
    def grounding_client(self) -> Optional[GroundingClient]:
        """The grounding client, or ``None`` if not initialized."""
        return self._grounding_client

    @property
    def grounding_config(self) -> Any:
        """The grounding configuration object."""
        return self._grounding_config

    @property
    def skill_registry(self) -> Optional[SkillRegistry]:
        """The skill registry, or ``None`` if skills are disabled."""
        return self._skill_registry

    @property
    def skill_store(self) -> Optional[SkillStore]:
        """The skill persistence store, or ``None`` if not initialized."""
        return self._skill_store

    @property
    def skill_evolver(self) -> Optional[SkillEvolver]:
        """The skill evolution engine, or ``None`` if not initialized."""
        return self._skill_evolver

    async def initialize(self) -> None:
        if self._initialized:
            logger.warning("OpenSpace already initialized")
            return

        logger.info("Initializing OpenSpace...")

        try:
            self._llm_factory = LLMFactory(config=self.config)
            self._llm_client = self._llm_factory.create_main()
            logger.info(f"Γ£ô LLM Client: {self.config.llm_model}")

            # Load grounding config
            # If custom config is provided, merge it with default configs
            # load_config supports multiple files and deep merges them (later files override earlier ones)
            if self.config.grounding_config_path:
                from openspace.config.constants import CONFIG_GROUNDING, CONFIG_SECURITY
                from openspace.config.loader import CONFIG_DIR

                # Load default configs + custom config (custom values will override defaults)
                grounding_config = load_config(
                    CONFIG_DIR / CONFIG_GROUNDING, CONFIG_DIR / CONFIG_SECURITY, self.config.grounding_config_path
                )
                logger.info(f"Merged custom grounding config: {self.config.grounding_config_path}")
            else:
                # Load default configs only
                grounding_config = get_config()

            # Optional: enable ClawWork productivity tools for fair benchmark comparison
            if getattr(self.config, "use_clawwork_productivity", False):
                shell_cfg = grounding_config.shell.model_copy(
                    update={
                        "use_clawwork_productivity": True,
                        "working_dir": self.config.workspace_dir or grounding_config.shell.working_dir,
                    }
                )
                grounding_config = grounding_config.model_copy(update={"shell": shell_cfg})
                logger.info("ClawWork productivity tools enabled (shell.working_dir used as sandbox root)")

            # Resolve backend_scope early so we can skip initializing
            # providers that are not in scope (e.g. web when only shell is needed).
            agent_config = get_agent_config("GroundingAgent")
            _cli_max_iter = self.config.grounding_max_iterations
            _default_max_iter = OpenSpaceConfig().grounding_max_iterations  # dataclass default (20)
            if agent_config:
                cfg_max_iter = agent_config.get("max_iterations", _default_max_iter)
                if _cli_max_iter != _default_max_iter:
                    max_iterations = _cli_max_iter
                else:
                    max_iterations = cfg_max_iter
                backend_scope = (
                    self.config.backend_scope
                    or agent_config.get("backend_scope")
                    or ["gui", "shell", "mcp", "web", "system"]
                )
                visual_analysis_timeout = agent_config.get("visual_analysis_timeout", 30.0)
                self.config.grounding_max_iterations = max_iterations
                logger.info(
                    f"Loaded GroundingAgent config from config_agents.json (max_iterations={max_iterations}, visual_analysis_timeout={visual_analysis_timeout}s)"
                )
            else:
                max_iterations = self.config.grounding_max_iterations
                backend_scope = self.config.backend_scope or ["gui", "shell", "mcp", "web", "system"]
                visual_analysis_timeout = 30.0
                logger.warning(f"config_agents.json not found, using default config (max_iterations={max_iterations})")

            # Filter enabled_backends in grounding config to only those in scope,
            # so providers outside scope (e.g. web) are never registered/initialized.
            if grounding_config.enabled_backends:
                scope_set = set(backend_scope)
                filtered = [
                    entry for entry in grounding_config.enabled_backends if entry.get("name", "").lower() in scope_set
                ]
                if len(filtered) != len(grounding_config.enabled_backends):
                    skipped = [
                        entry.get("name")
                        for entry in grounding_config.enabled_backends
                        if entry.get("name", "").lower() not in scope_set
                    ]
                    logger.info(f"Skipping backends not in scope: {skipped}")
                    grounding_config = grounding_config.model_copy(update={"enabled_backends": filtered})

            self._grounding_config = grounding_config
            self._grounding_client = GroundingClient(config=grounding_config)
            await self._grounding_client.initialize_all_providers()

            backends = list(self._grounding_client.list_providers().keys())
            logger.info(f"Γ£ô Grounding Client: {len(backends)} backends")
            logger.debug(f"  Available backends: {[b.value for b in backends]}")

            if self.config.enable_recording:
                self._recording_service = RecordingService(config=self.config)
                self._recording_manager = self._recording_service.create(llm_client=self._llm_client)
                self._recording_service.wire(grounding_client=self._grounding_client)
                logger.info(f"Γ£ô Recording enabled: {len(self._recording_manager.backends or [])} backends")

            # Create separate LLM client for tool retrieval if configured
            tool_retrieval_llm = self._llm_factory.create_tool_retrieval()
            if tool_retrieval_llm:
                logger.info(f"Γ£ô Tool retrieval LLM: {self.config.tool_retrieval_model}")

            self._grounding_agent = GroundingAgent(
                name="OpenSpace-GroundingAgent",
                backend_scope=backend_scope,
                llm_client=self._llm_client,
                grounding_client=self._grounding_client,
                recording_manager=self._recording_manager,
                system_prompt=self.config.grounding_system_prompt,
                max_iterations=max_iterations,
                visual_analysis_timeout=visual_analysis_timeout,
                tool_retrieval_llm=tool_retrieval_llm,
                visual_analysis_model=self.config.visual_analysis_model,
            )
            logger.info(f"Γ£ô GroundingAgent: {', '.join(backend_scope)}")

            # Initialize SkillRegistry (settings from config_grounding.json ΓåÆ skills)
            if self._grounding_config and self._grounding_config.skills.enabled:
                self._tool_registry = ToolRegistry(
                    config=self.config,
                    grounding_config=self._grounding_config,
                    llm_client=self._llm_client,
                )
                if self._tool_registry.discover():
                    self._skill_registry = self._tool_registry.registry
                    skills = self._skill_registry.list_skills()
                    logger.info(f"Γ£ô Skills: {len(skills)} discovered")
                    self._grounding_agent.set_skill_registry(self._skill_registry)
                else:
                    self._skill_registry = None

            # Initialize ExecutionAnalyzer (requires recording + skills)
            if self.config.enable_recording and self._skill_registry:
                try:
                    skill_store = SkillStore()
                    self._skill_store = skill_store  # Expose for MCP server reuse

                    # Sync filesystem skills ΓåÆ DB (creates initial records
                    # for newly discovered skills so that analysis stats
                    # can be recorded against them from the very first run).
                    await skill_store.sync_from_registry(self._skill_registry.list_skills())

                    # Bridge: pass quality_manager so analysis can feed back
                    # LLM-identified tool issues to the tool quality system.
                    quality_mgr = self._grounding_client.quality_manager if self._grounding_client else None
                    self._execution_analyzer = ExecutionAnalyzer(
                        store=skill_store,
                        llm_client=self._llm_client,
                        model=self.config.execution_analyzer_model,
                        skill_registry=self._skill_registry,
                        quality_manager=quality_mgr,
                    )
                    logger.info("Γ£ô Execution analysis enabled")

                    # Share store with GroundingAgent so retrieve_skill
                    # can access quality metrics for LLM selection.
                    self._grounding_agent._skill_store = skill_store

                    # Initialize SkillEvolver (reuses the same store & registry)
                    # available_tools will be updated before each evolution cycle
                    self._skill_evolver = SkillEvolver(
                        store=skill_store,
                        registry=self._skill_registry,
                        llm_client=self._llm_client,
                        model=self.config.skill_evolver_model,
                        max_concurrent=self.config.evolution_max_concurrent,
                    )
                    logger.info(f"Γ£ô Skill evolution enabled (concurrent={self.config.evolution_max_concurrent})")
                except Exception as e:
                    logger.warning(f"Execution analyzer init failed (non-fatal): {e}")

            # Create ExecutionEngine with all dependencies
            self._execution_engine = ExecutionEngine(
                config=self.config,
                grounding_agent=self._grounding_agent,
                grounding_client=self._grounding_client,
                tool_registry=self._tool_registry,
                skill_registry=self._skill_registry,
                skill_store=self._skill_store,
                recording_manager=self._recording_manager,
                execution_analyzer=self._execution_analyzer,
                skill_evolver=self._skill_evolver,
            )

            self._initialized = True
            logger.info("=" * 60)
            logger.info("OpenSpace ready to use!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to initialize OpenSpace: {e}")
            await self.cleanup()
            raise

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[str] = None,
        max_iterations: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a task with OpenSpace.  Delegates to ExecutionEngine."""
        if not self._initialized or not self._execution_engine:
            raise RuntimeError(
                "OpenSpace not initialized. Call await tool_layer.initialize() first or use async with."
            )
        return await self._execution_engine.execute(
            task,
            context=context,
            workspace_dir=workspace_dir,
            max_iterations=max_iterations,
            task_id=task_id,
        )

    # NOTE: _init_skill_registry, _select_and_inject_skills, and
    # _get_skill_selection_llm have been extracted to ToolRegistry
    # (openspace/tool_registry.py) in Epic 4.1.
    #
    # execute(), _maybe_analyze_execution(), and _maybe_evolve_quality()
    # have been extracted to ExecutionEngine
    # (openspace/execution_engine.py) in Epic 4.3.

    async def cleanup(self) -> None:
        """
        Close all sessions and release resources.
        Automatically called when using context manager.
        """
        logger.info("Cleaning up OpenSpace resources...")

        try:
            # Wait for background evolution tasks before tearing down
            if self._skill_evolver:
                await self._skill_evolver.wait_background()

            if self._grounding_client:
                await self._grounding_client.close_all_sessions()
                logger.debug("All grounding sessions closed")

            if self._recording_service:
                await self._recording_service.cleanup()

            if self._execution_analyzer:
                try:
                    self._execution_analyzer.close()
                    logger.debug("Execution analyzer closed")
                except Exception as e:
                    logger.debug(f"Failed to close execution analyzer: {e}")

            self._initialized = False
            self._running = False
            if self._execution_engine:
                self._execution_engine._running = False
                self._execution_engine._task_done.set()

            logger.info("OpenSpace cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    def is_initialized(self) -> bool:
        return self._initialized

    def is_running(self) -> bool:
        if self._execution_engine:
            return self._execution_engine.is_running
        return self._running

    def get_config(self) -> OpenSpaceConfig:
        return self.config

    def list_backends(self) -> List[str]:
        if not self._initialized:
            raise RuntimeError("OpenSpace not initialized")
        return [backend.value for backend in self._grounding_client.list_providers().keys()]

    def list_sessions(self) -> List[str]:
        if not self._initialized:
            raise RuntimeError("OpenSpace not initialized")
        return self._grounding_client.list_sessions()

    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.cleanup()
        return False

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        if self.is_running():
            status = "running"
        backends = ", ".join(self.config.backend_scope) if self.config.backend_scope else "all"
        return f"<OpenSpace(status={status}, backends={backends}, model={self.config.llm_model})>"
