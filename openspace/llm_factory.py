"""LLMFactory — centralized LLM client creation from config.

Extracted from OpenSpace.initialize() in Epic 4.5.  Owns:
  • Main LLM client creation with full config
  • Optional tool retrieval LLM creation (shares credentials via llm_kwargs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from openspace.llm import LLMClient
from openspace.utils.logging import Logger

if TYPE_CHECKING:
    from openspace.tool_layer import OpenSpaceConfig

logger = Logger.get_logger(__name__)


class LLMFactory:
    """Creates LLM clients from OpenSpaceConfig."""

    def __init__(self, *, config: OpenSpaceConfig) -> None:
        self._config = config
        self._llm_client: Optional[LLMClient] = None
        self._tool_retrieval_llm: Optional[LLMClient] = None

    @property
    def llm_client(self) -> Optional[LLMClient]:
        return self._llm_client

    @property
    def tool_retrieval_llm(self) -> Optional[LLMClient]:
        return self._tool_retrieval_llm

    def create_main(self) -> LLMClient:
        """Create the primary LLM client from config."""
        self._llm_client = LLMClient(
            model=self._config.llm_model,
            enable_thinking=self._config.llm_enable_thinking,
            rate_limit_delay=self._config.llm_rate_limit_delay,
            max_retries=self._config.llm_max_retries,
            timeout=self._config.llm_timeout,
            **self._config.llm_kwargs,
        )
        return self._llm_client

    def create_tool_retrieval(self) -> Optional[LLMClient]:
        """Create optional tool retrieval LLM. Returns None if not configured.

        Inherits llm_kwargs (api_key, api_base, etc.) so credentials
        from the host agent are shared across all internal LLM clients.

        Note: intentionally omits ``enable_thinking`` and ``rate_limit_delay``
        since tool retrieval is a simple selection task, not a reasoning task.
        """
        if not self._config.tool_retrieval_model:
            return None

        self._tool_retrieval_llm = LLMClient(
            model=self._config.tool_retrieval_model,
            timeout=self._config.llm_timeout,
            max_retries=self._config.llm_max_retries,
            **self._config.llm_kwargs,
        )
        return self._tool_retrieval_llm
