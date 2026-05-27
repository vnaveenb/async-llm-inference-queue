from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import litellm

from .config import get_config
from .models import InferenceJob

logger = logging.getLogger(__name__)

# Mock responses for demo mode
_MOCK_RESPONSES = [
    "The answer to your question involves several key considerations...",
    "Based on the available information, I can provide the following analysis...",
    "Here's a comprehensive response to your query...",
    "Let me break this down into key points for clarity...",
    "After careful consideration, here's what I've determined...",
]


async def run_inference(job: InferenceJob) -> dict[str, Any]:
    """Execute inference for a job. Returns result dict with tokens and latency.

    Supports two modes via config:
    - mock: simulated responses with random delay (no API key needed)
    - real: calls LiteLLM acompletion (integrates with Project 06 tracker)
    """
    cfg = get_config()

    if cfg.inference.mock:
        return await _mock_inference(job)
    else:
        return await _real_inference(job)


async def _mock_inference(job: InferenceJob) -> dict[str, Any]:
    """Simulate inference with random delay and canned response."""
    # Simulate variable latency (200ms - 3s)
    delay = random.uniform(0.2, 3.0)
    await asyncio.sleep(delay)

    response_text = random.choice(_MOCK_RESPONSES)
    # Simulate token counts
    input_tokens = len(job.prompt.split()) * 2
    output_tokens = len(response_text.split()) * 2

    return {
        "result": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": int(delay * 1000),
    }


async def _real_inference(job: InferenceJob) -> dict[str, Any]:
    """Call LiteLLM for real inference."""
    cfg = get_config()
    start = time.time()

    messages = [{"role": "user", "content": job.prompt}]

    kwargs: dict[str, Any] = {
        "model": job.model,
        "messages": messages,
        "max_tokens": job.max_tokens,
        "temperature": job.temperature,
        "timeout": cfg.inference.timeout_seconds,
    }

    # Add metadata for Project 06 tracker integration
    if cfg.tracker.enabled:
        kwargs["metadata"] = {
            "project_id": cfg.tracker.project_id,
            "endpoint": "/jobs",
            "job_id": job.id,
        }

    response = await litellm.acompletion(**kwargs)

    latency_ms = int((time.time() - start) * 1000)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    result_text = response.choices[0].message.content or ""

    return {
        "result": result_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
    }


def setup_tracker() -> None:
    """Initialize Project 06 cost tracker if available and enabled."""
    cfg = get_config()
    if not cfg.tracker.enabled:
        return

    try:
        import sys
        import os

        # Try importing the tracker from Project 06 (available when deployed together)
        tracker_path = os.environ.get("TRACKER_MODULE_PATH")
        if tracker_path and tracker_path not in sys.path:
            sys.path.insert(0, tracker_path)

        from token_cost_dashboard.tracker import register_callbacks
        register_callbacks(project_id=cfg.tracker.project_id)
        logger.info("Project 06 cost tracker registered")
    except ImportError:
        logger.info("Project 06 tracker not available — cost tracking disabled")
