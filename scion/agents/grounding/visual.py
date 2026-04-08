"""Visual analysis helpers for GroundingAgent.

Handles screenshot capture, visual analysis callbacks, and screenshot
selection for grounding agent workflows.
Extracted from grounding_agent.py (Epic 5.9).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from scion.grounding.core.types import ToolResult
from scion.platforms.screenshot import ScreenshotClient
from scion.prompts import GroundingAgentPrompts
from scion.utils.logging import Logger

if TYPE_CHECKING:
    pass

logger = Logger.get_logger("scion.agents.grounding_agent")


async def _visual_analysis_callback(
    agent, result: ToolResult, tool_name: str, tool_call: Dict, backend: str
) -> ToolResult:
    """Callback for LLMClient to handle visual analysis after tool execution."""
    # 1. Check if LLM requested to skip visual analysis
    skip_visual_analysis = False
    try:
        arguments = tool_call.function.arguments
        if isinstance(arguments, str):
            args = json.loads(arguments.strip() or "{}")
        else:
            args = arguments

        if isinstance(args, dict) and args.get("skip_visual_analysis"):
            skip_visual_analysis = True
            logger.info(f"Visual analysis skipped for {tool_name} (meta-parameter set by LLM)")
    except Exception as e:
        logger.debug(f"Could not parse tool arguments: {e}")

    # 2. If skip requested, return original result
    if skip_visual_analysis:
        return result

    # 3. Check if this backend needs visual analysis
    if backend != "gui":
        return result

    # 4. Check if tool has visual data
    metadata = getattr(result, "metadata", None)
    has_screenshots = metadata and (metadata.get("screenshot") or metadata.get("screenshots"))

    # 5. If no visual data, try to capture a screenshot
    if not has_screenshots:
        try:
            logger.info(f"No visual data from {tool_name}, capturing screenshot...")
            screenshot_client = ScreenshotClient()
            screenshot_bytes = await screenshot_client.capture()

            if screenshot_bytes:
                # Add screenshot to result metadata
                if metadata is None:
                    result.metadata = {}
                    metadata = result.metadata
                metadata["screenshot"] = screenshot_bytes
                has_screenshots = True
                logger.info("Screenshot captured for visual analysis")
            else:
                logger.warning("Failed to capture screenshot")
        except Exception as e:
            logger.warning(f"Error capturing screenshot: {e}")

    # 6. If still no screenshots, return original result
    if not has_screenshots:
        logger.debug(f"No visual data available for {tool_name}")
        return result

    # 7. Perform visual analysis
    return await agent._enhance_result_with_visual_context(result, tool_name)


async def _enhance_result_with_visual_context(
    agent, result: ToolResult, tool_name: str
) -> ToolResult:
    """Enhance tool result with visual analysis for grounding agent workflows."""
    import asyncio
    import base64

    import litellm

    try:
        metadata = getattr(result, "metadata", None)
        if not metadata:
            return result

        # Collect all screenshots
        screenshots_bytes = []

        # Check for multiple screenshots first
        if metadata.get("screenshots"):
            screenshots_list = metadata["screenshots"]
            if isinstance(screenshots_list, list):
                screenshots_bytes = [s for s in screenshots_list if s]
        # Fall back to single screenshot
        elif metadata.get("screenshot"):
            screenshots_bytes = [metadata["screenshot"]]

        if not screenshots_bytes:
            return result

        # Select key screenshots if there are too many
        selected_screenshots = _select_key_screenshots(screenshots_bytes, max_count=3)

        # Convert to base64
        visual_b64_list = []
        for visual_data in selected_screenshots:
            if isinstance(visual_data, bytes):
                visual_b64_list.append(base64.b64encode(visual_data).decode("utf-8"))
            else:
                visual_b64_list.append(visual_data)  # Already base64

        # Build prompt based on number of screenshots
        num_screenshots = len(visual_b64_list)

        prompt = GroundingAgentPrompts.visual_analysis(
            tool_name=tool_name,
            num_screenshots=num_screenshots,
            task_description=getattr(agent, "_current_instruction", ""),
        )

        # Build content with text prompt + all images
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for visual_b64 in visual_b64_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{visual_b64}"}})

        # Use dedicated visual analysis model if configured, otherwise use main LLM model
        visual_model = agent._visual_analysis_model or (
            agent._llm_client.model if agent._llm_client else "openrouter/anthropic/claude-sonnet-4.5"
        )
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=visual_model,
                messages=[{"role": "user", "content": content}],
                timeout=agent._visual_analysis_timeout,
            ),
            timeout=agent._visual_analysis_timeout + 5,
        )

        analysis = response.choices[0].message.content.strip()

        # Inject visual analysis into content
        original_content = result.content or "(no text output)"
        enhanced_content = f"{original_content}\n\n**Visual content**: {analysis}"

        # Create enhanced result
        enhanced_result = ToolResult(
            status=result.status,
            content=enhanced_content,
            error=result.error,
            metadata={**metadata, "visual_analyzed": True, "visual_analysis": analysis},
            execution_time=result.execution_time,
        )

        logger.info(f"Enhanced {tool_name} result with visual analysis ({num_screenshots} screenshot(s))")
        return enhanced_result

    except asyncio.TimeoutError:
        logger.warning(f"Visual analysis timed out for {tool_name}, returning original result")
        return result
    except Exception as e:
        logger.warning(f"Failed to analyze visual content for {tool_name}: {e}")
        return result


def _select_key_screenshots(screenshots: List[bytes], max_count: int = 3) -> List[bytes]:
    """Select key screenshots if there are too many.

    Pure function — does not require an agent instance.
    """
    if len(screenshots) <= max_count:
        return screenshots

    selected_indices: set[int] = set()

    # Always include last (final state)
    selected_indices.add(len(screenshots) - 1)

    # If room, include first (initial state)
    if max_count >= 2:
        selected_indices.add(0)

    # Fill remaining slots with evenly spaced middle screenshots
    remaining_slots = max_count - len(selected_indices)
    if remaining_slots > 0:
        # Calculate spacing
        available_indices = [i for i in range(1, len(screenshots) - 1) if i not in selected_indices]

        if available_indices:
            step = max(1, len(available_indices) // (remaining_slots + 1))
            for i in range(remaining_slots):
                idx = min((i + 1) * step, len(available_indices) - 1)
                if idx < len(available_indices):
                    selected_indices.add(available_indices[idx])

    # Return screenshots in original order
    selected = [screenshots[i] for i in sorted(selected_indices)]

    logger.debug(
        f"Selected {len(selected)} screenshots at indices {sorted(selected_indices)} "
        f"from total of {len(screenshots)}"
    )

    return selected
