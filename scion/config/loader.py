import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from scion.utils.logging import Logger

from .constants import CONFIG_AGENTS, CONFIG_DEV, CONFIG_GROUNDING, CONFIG_MCP, CONFIG_SECURITY
from .grounding import GroundingConfig
from .utils import load_json_file
from .utils import save_json_file as save_json

logger = Logger.get_logger(__name__)


CONFIG_DIR = Path(__file__).parent

# Global configuration singleton
_config: GroundingConfig | None = None
_config_lock = threading.RLock()  # Use RLock to support recursive locking


def _deep_merge_dict(base: dict, update: dict) -> dict:
    """Deep merge two dictionaries, update's values will override base's values"""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _load_json_file(path: Path, *, critical: bool = False) -> Dict[str, Any]:
    """Load single JSON configuration file.

    This function wraps the generic load_json_file and adds global
    configuration specific error handling and logging.

    Args:
        path: Path to JSON file.
        critical: If True, raise on parse errors instead of returning {}.
                  Use for security-critical config files where silent
                  fallback to empty dict could weaken security posture.
    """
    if not path.exists():
        if critical:
            raise FileNotFoundError(
                f"Critical configuration file missing: {path}. Cannot start with potentially insecure defaults."
            )
        logger.debug(f"Configuration file does not exist, skipping: {path}")
        return {}

    try:
        data = load_json_file(path)
        logger.info(f"Loaded configuration file: {path}")
        return data
    except Exception as e:
        if critical:
            raise RuntimeError(
                f"Failed to parse critical configuration file {path}: {e}. "
                "Refusing to start with potentially insecure defaults."
            ) from e
        logger.warning(f"Failed to load configuration file {path}: {e}")
        return {}


def _load_multiple_files(paths: Iterable[Path], critical_files: frozenset[str] = frozenset()) -> Dict[str, Any]:
    """Load configuration from multiple files.

    Args:
        paths: Config file paths to load and merge.
        critical_files: Filenames (not full paths) that must not fail silently.
    """
    merged = {}
    for path in paths:
        is_critical = path.name in critical_files
        data = _load_json_file(path, critical=is_critical)
        if data:
            merged = _deep_merge_dict(merged, data)
    return merged


# Security config must not fail silently — malformed security config
# could silently disable sandbox enforcement.
_CRITICAL_CONFIG_FILES = frozenset({CONFIG_SECURITY})


def load_config(*config_paths: Union[str, Path]) -> GroundingConfig:
    """
    Load configuration files
    """
    global _config

    with _config_lock:
        if config_paths:
            paths = [Path(p) for p in config_paths]
        else:
            paths = [
                CONFIG_DIR / CONFIG_GROUNDING,
                CONFIG_DIR / CONFIG_SECURITY,
                CONFIG_DIR / CONFIG_DEV,  # Optional: development environment configuration
            ]

        # Load and merge configuration
        # Security config is marked critical — parse errors raise instead of
        # silently falling back to defaults (which could disable sandbox).
        raw_data = _load_multiple_files(paths, critical_files=_CRITICAL_CONFIG_FILES)

        # Load MCP configuration (separate processing)
        # Check if mcpServers already provided in merged custom configs
        has_custom_mcp_servers = "mcpServers" in raw_data

        if has_custom_mcp_servers:
            # Use mcpServers from custom config
            if "mcp" not in raw_data:
                raw_data["mcp"] = {}
            raw_data["mcp"]["servers"] = raw_data.pop("mcpServers")
            logger.debug(f"Using custom MCP servers from provided config ({len(raw_data['mcp']['servers'])} servers)")
        else:
            # Load default MCP servers from config_mcp.json
            mcp_data = _load_json_file(CONFIG_DIR / CONFIG_MCP)
            if mcp_data and "mcpServers" in mcp_data:
                if "mcp" not in raw_data:
                    raw_data["mcp"] = {}
                raw_data["mcp"]["servers"] = mcp_data["mcpServers"]
                logger.debug(
                    f"Loaded MCP servers from default config_mcp.json ({len(raw_data['mcp']['servers'])} servers)"
                )

        # Validate and create configuration object
        # Fail-closed: invalid config raises instead of silently falling back
        # to defaults (which could disable sandbox enforcement)
        try:
            _config = GroundingConfig.model_validate(raw_data)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise RuntimeError(
                f"GroundingConfig validation failed — refusing to start with "
                f"potentially insecure defaults. Fix the config or remove it "
                f"to use secure defaults. Error: {e}"
            ) from e

        # Adjust log level according to configuration
        if _config.debug:
            Logger.set_debug(2)
        elif _config.log_level:
            try:
                Logger.configure(level=_config.log_level)
            except Exception as e:
                logger.warning(f"Failed to set log level {_config.log_level}: {e}")

    return _config


def get_config() -> GroundingConfig:
    """
    Get global configuration instance.

    Usage:
        - Get configuration in Provider: get_config().get_backend_config('shell')
        - Get security policy in Tool: get_config().get_security_policy('shell')
    """
    global _config

    if _config is None:
        with _config_lock:
            if _config is None:
                load_config()

    return _config


def reset_config() -> None:
    """Reset configuration (for testing)"""
    global _config
    with _config_lock:
        _config = None


def save_config(config: GroundingConfig, path: Union[str, Path]) -> None:
    save_json(config.model_dump(), path)
    logger.info(f"Configuration saved to: {path}")


def load_agents_config() -> Dict[str, Any]:
    agents_config_path = CONFIG_DIR / CONFIG_AGENTS
    return _load_json_file(agents_config_path)


def get_agent_config(agent_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration of the specified agent
    """
    agents_config = load_agents_config()

    if "agents" not in agents_config:
        logger.warning(f"No 'agents' key found in {CONFIG_AGENTS}")
        return None

    for agent_cfg in agents_config.get("agents", []):
        if agent_cfg.get("name") == agent_name:
            return agent_cfg

    logger.warning(f"Agent '{agent_name}' not found in {CONFIG_AGENTS}")
    return None


__all__ = [
    "CONFIG_DIR",
    "load_config",
    "get_config",
    "reset_config",
    "save_config",
    "load_agents_config",
    "get_agent_config",
]
