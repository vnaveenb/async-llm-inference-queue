from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from .config import get_config
from .queue import queue_depth

logger = logging.getLogger(__name__)


class BackpressureMiddleware(BaseHTTPMiddleware):
    """Rejects new job submissions with 429 when queue depth exceeds threshold."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only apply to job submission endpoint
        if request.method == "POST" and request.url.path == "/jobs":
            cfg = get_config()
            depth = await queue_depth()
            if depth >= cfg.queue.backpressure_threshold:
                logger.warning(f"Backpressure active: queue depth {depth} >= {cfg.queue.backpressure_threshold}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Queue backpressure active",
                        "queue_depth": depth,
                        "threshold": cfg.queue.backpressure_threshold,
                        "message": "Too many pending requests. Retry later.",
                    },
                )
        return await call_next(request)


async def check_backpressure() -> Optional[dict]:
    """Check if backpressure is active. Returns None if OK, or error dict."""
    cfg = get_config()
    depth = await queue_depth()
    if depth >= cfg.queue.backpressure_threshold:
        return {
            "queue_depth": depth,
            "threshold": cfg.queue.backpressure_threshold,
        }
    return None
