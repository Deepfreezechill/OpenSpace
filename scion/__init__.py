from importlib import import_module as _imp
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any
from typing import Dict as _Dict

if _TYPE_CHECKING:
    from scion.agents import GroundingAgent as GroundingAgent
    from scion.llm import LLMClient as LLMClient
    from scion.recording import RecordingManager as RecordingManager
    from scion.tool_layer import OpenSpace as OpenSpace
    from scion.tool_layer import OpenSpaceConfig as OpenSpaceConfig

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Main API
    "OpenSpace",
    "OpenSpaceConfig",
    # Core Components
    "GroundingAgent",
    "GroundingClient",
    "LLMClient",
    "BaseTool",
    "ToolResult",
    "BackendType",
    # Recording System
    "RecordingManager",
    "RecordingViewer",
]

# Map attribute → sub-module that provides it
_attr_to_module: _Dict[str, str] = {
    # Main API
    "OpenSpace": "scion.tool_layer",
    "OpenSpaceConfig": "scion.tool_layer",
    # Core Components
    "GroundingAgent": "scion.agents",
    "GroundingClient": "scion.grounding.core.grounding_client",
    "LLMClient": "scion.llm",
    "BaseTool": "scion.grounding.core.tool.base",
    "ToolResult": "scion.grounding.core.types",
    "BackendType": "scion.grounding.core.types",
    # Recording System
    "RecordingManager": "scion.recording",
    "RecordingViewer": "scion.recording.viewer",
}


def __getattr__(name: str) -> _Any:
    """Dynamically import sub-modules on first attribute access.

    This keeps the *initial* package import lightweight and avoids raising
    `ModuleNotFoundError` for optional / heavy dependencies until the
    corresponding functionality is explicitly used.
    """
    if name not in _attr_to_module:
        raise AttributeError(f"module 'scion' has no attribute '{name}'")

    module_name = _attr_to_module[name]
    module = _imp(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_attr_to_module.keys()))
