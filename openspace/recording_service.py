"""RecordingService — factory and lifecycle for execution recording.

Extracted from OpenSpace.initialize() in Epic 4.4.  Owns:
  • RecordingManager creation from config
  • Wiring into GroundingClient and LLMClient
  • Graceful cleanup / stop
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from openspace.recording import RecordingManager
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.grounding.core.grounding_client import GroundingClient
    from openspace.llm import LLMClient
    from openspace.tool_layer import OpenSpaceConfig

logger = Logger.get_logger(__name__)


class RecordingService:
    """Creates and manages the RecordingManager lifecycle."""

    def __init__(self, *, config: OpenSpaceConfig) -> None:
        self._config = config
        self._manager: Optional[RecordingManager] = None

    @property
    def manager(self) -> Optional[RecordingManager]:
        return self._manager

    def create(self, *, llm_client: LLMClient) -> Optional[RecordingManager]:
        """Create RecordingManager from config. Returns None if recording disabled."""
        if not self._config.enable_recording:
            return None

        self._manager = RecordingManager(
            enabled=True,
            task_id="",
            log_dir=self._config.recording_log_dir,
            backends=self._config.recording_backends,
            enable_screenshot=self._config.enable_screenshot,
            enable_video=self._config.enable_video,
            enable_conversation_log=self._config.enable_conversation_log,
            agent_name="OpenSpace",
        )
        self._manager.register_to_llm(llm_client)
        return self._manager

    def wire(self, *, grounding_client: GroundingClient) -> None:
        """Inject recording manager into grounding client for GUI intermediate steps."""
        if self._manager:
            grounding_client.recording_manager = self._manager

    async def cleanup(self) -> None:
        """Stop active recording session, if any."""
        if not self._manager:
            return
        if self._manager.recording_status:
            try:
                await self._manager.stop()
            except Exception as e:
                logger.warning("Failed to stop recording: %s", e)
