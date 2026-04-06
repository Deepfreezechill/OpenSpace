"""Workspace scanning helpers for GroundingAgent.

Provides workspace path resolution, file scanning, and artifact
checking for grounding agent task execution.
Extracted from grounding_agent.py (Epic 5.9).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Optional

from openspace.utils.logging import Logger

logger = Logger.get_logger("openspace.agents.grounding_agent")


def _get_workspace_path(context: Dict[str, Any]) -> Optional[str]:
    """Get workspace directory path from context.

    Pure function — does not require an agent instance.
    """
    return context.get("workspace_dir")


def _scan_workspace_files(
    workspace_path: str,
    recent_threshold: int = 600,  # seconds
) -> Dict[str, Any]:
    """Scan workspace directory and collect file information.

    Pure function — does not require an agent instance.

    Args:
        workspace_path: Path to workspace directory
        recent_threshold: Threshold in seconds for recent files

    Returns:
        Dictionary with file information:
            - files: List of all filenames
            - file_details: Dict mapping filename to file info (size, modified, age_seconds)
            - recent_files: List of recently modified filenames
    """
    result: Dict[str, Any] = {"files": [], "file_details": {}, "recent_files": []}

    if not workspace_path or not os.path.exists(workspace_path):
        return result

    # Recording system files to exclude from workspace scanning
    excluded_files = {"metadata.json", "traj.jsonl"}

    try:
        current_time = time.time()

        for filename in os.listdir(workspace_path):
            filepath = os.path.join(workspace_path, filename)
            if os.path.isfile(filepath) and filename not in excluded_files:
                result["files"].append(filename)

                # Get file stats
                stat = os.stat(filepath)
                file_info = {
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "age_seconds": current_time - stat.st_mtime,
                }
                result["file_details"][filename] = file_info

                # Track recently created/modified files
                if file_info["age_seconds"] < recent_threshold:
                    result["recent_files"].append(filename)

        result["files"] = sorted(result["files"])

    except Exception as e:
        logger.debug(f"Error scanning workspace files: {e}")

    return result


async def _check_workspace_artifacts(agent, context: Dict[str, Any]) -> Dict[str, Any]:
    """Check workspace directory for existing artifacts that might be relevant to the task.

    Enhanced to detect if task might already be completed.
    """
    workspace_info: Dict[str, Any] = {"has_files": False, "files": [], "file_details": {}, "recent_files": []}

    try:
        # Get workspace path — route through agent for MRO preservation
        workspace_path = agent._get_workspace_path(context)

        # Scan workspace files — route through agent for MRO preservation
        scan_result = agent._scan_workspace_files(workspace_path, recent_threshold=600)

        if scan_result["files"]:
            workspace_info["has_files"] = True
            workspace_info["files"] = scan_result["files"]
            workspace_info["file_details"] = scan_result["file_details"]
            workspace_info["recent_files"] = scan_result["recent_files"]

            logger.info(
                f"Grounding Agent: Found {len(scan_result['files'])} existing files in workspace "
                f"({len(scan_result['recent_files'])} recent)"
            )

            # Check if instruction mentions specific filenames
            instruction = context.get("instruction", "")
            if instruction:
                # Look for potential file references in instruction
                potential_outputs = []
                # Match common file patterns: filename.ext, "filename", 'filename'
                file_patterns = re.findall(r'["\']?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)["\']?', instruction)
                for pattern in file_patterns:
                    if pattern in scan_result["files"]:
                        potential_outputs.append(pattern)

                if potential_outputs:
                    workspace_info["matching_files"] = potential_outputs
                    logger.info(
                        f"Grounding Agent: Found {len(potential_outputs)} files matching task: {potential_outputs}"
                    )

    except Exception as e:
        logger.debug(f"Could not check workspace artifacts: {e}")

    return workspace_info
