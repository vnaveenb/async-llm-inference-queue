from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .backpressure import BackpressureMiddleware
from .config import get_config, reload_config
from .inference import setup_tracker
from .models import (
    InferenceJob, JobRequest, JobResponse, JobStatus,
    Priority, QueueStats, SubmitResponse,
)
from .queue import (
    QueueFullError, cancel_job, close_redis, complete_job,
    dequeue, dlq_size, enqueue, fail_job, get_dlq_jobs,
    get_job, get_redis, get_stats, queue_depth,
    queue_depth_by_priority, retry_job, running_count,
)
from .worker import start_workers, stop_workers, workers_active

logger = logging.getLogger(__name__)

# ── Security & Rate Limiting ──────────────────────────────────────────────
_IP_DAILY_CALLS: dict[str, list[float]] = defaultdict(list)
_DAILY_LIMIT = int(os.getenv("RATE_LIMIT_PER_IP_PER_DAY", "10"))
_BURST_LIMIT = int(os.getenv("RATE_LIMIT_PER_IP_PER_MINUTE", "5"))
_ADMIN_KEY = os.getenv("ADMIN_API_KEY", "")


def _get_client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def _check_rate_limit(request: Request) -> None:
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "1":
        return  # Bypass during automated pytest runs

    admin_key = request.headers.get("x-admin-key", "")
    if _ADMIN_KEY and admin_key == _ADMIN_KEY:
        return  # Admin bypasses rate limits

    ip = _get_client_ip(request)
    now = time.time()
    cutoff_day = now - 86400
    cutoff_min = now - 60

    calls = _IP_DAILY_CALLS[ip]
    _IP_DAILY_CALLS[ip] = [t for t in calls if t > cutoff_day]
    calls = _IP_DAILY_CALLS[ip]

    burst_calls = sum(1 for t in calls if t > cutoff_min)
    if burst_calls >= _BURST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Burst rate limit exceeded ({_BURST_LIMIT} requests/minute). Please wait.",
        )

    if len(calls) >= _DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily rate limit reached ({_DAILY_LIMIT} requests/day).",
        )

    _IP_DAILY_CALLS[ip].append(now)


def _require_admin(request: Request) -> None:
    if not _ADMIN_KEY:
        return  # If unconfigured in dev/test, allow
    key = request.headers.get("x-admin-key", "")
    if key != _ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Valid X-Admin-Key required")


def reset_rate_limits() -> None:
    _IP_DAILY_CALLS.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    setup_tracker()
    await get_redis()  # Verify connection
    await start_workers()
    logger.info("Async Inference Queue ready")
    yield
    await stop_workers()
    await close_redis()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Async Inference Queue",
    description="Priority-based async job queue for LLM inference with backpressure and DLQ",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(BackpressureMiddleware)

# Mount static dashboard
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Job Endpoints ──────────────────────────────────────────────────────────


@app.post("/jobs", response_model=SubmitResponse)
async def submit_job(request: JobRequest, raw_request: Request):
    """Submit a new inference job to the queue."""
    _check_rate_limit(raw_request)

    admin_key = raw_request.headers.get("x-admin-key", "")
    if not (_ADMIN_KEY and admin_key == _ADMIN_KEY):
        if len(request.prompt) > 2000:
            request.prompt = request.prompt[:2000]
        request.max_tokens = min(request.max_tokens or 500, 500)

    cfg = get_config()

    job = InferenceJob(
        prompt=request.prompt,
        model=request.model or cfg.inference.default_model,
        priority=request.priority,
        max_tokens=request.max_tokens or cfg.inference.max_tokens,
        temperature=request.temperature if request.temperature is not None else cfg.inference.temperature,
        metadata=request.metadata,
        max_retries=cfg.worker.max_retries,
    )

    try:
        position = await enqueue(job)
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e))

    return SubmitResponse(job_id=job.id, status=job.status, position=position)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status and result of a job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse(**job.model_dump())


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    status: Optional[JobStatus] = Query(None),
    priority: Optional[Priority] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List jobs. Note: scans Redis keys — use filters for efficiency."""
    r = await get_redis()
    prefix = get_config().redis.prefix
    cursor = 0
    jobs = []

    while True:
        cursor, keys = await r.scan(cursor, match=f"{prefix}job:*", count=100)
        for key in keys:
            data = await r.get(key)
            if not data:
                continue
            job = InferenceJob.model_validate_json(data)
            if status and job.status != status:
                continue
            if priority and job.priority != priority:
                continue
            jobs.append(JobResponse(**job.model_dump()))
            if len(jobs) >= limit:
                break
        if cursor == 0 or len(jobs) >= limit:
            break

    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job_endpoint(job_id: str, raw_request: Request):
    """Cancel a pending or queued job."""
    _require_admin(raw_request)
    job = await cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=400, detail="Job not found or not cancellable (must be pending/queued)")
    return JobResponse(**job.model_dump())


@app.post("/jobs/{job_id}/retry", response_model=JobResponse)
async def retry_job_endpoint(job_id: str, raw_request: Request):
    """Retry a failed or dead job."""
    _require_admin(raw_request)
    job = await retry_job(job_id)
    if not job:
        raise HTTPException(status_code=400, detail="Job not found or not retryable (must be failed/dead)")
    return JobResponse(**job.model_dump())


# ─── Queue Endpoints ─────────────────────────────────────────────────────────


@app.get("/queue/stats", response_model=QueueStats)
async def get_queue_stats():
    """Get current queue statistics."""
    cfg = get_config()
    depth = await queue_depth()
    by_priority = await queue_depth_by_priority()
    running = await running_count()
    dlq = await dlq_size()
    stats = await get_stats()

    return QueueStats(
        total_depth=depth,
        by_priority=by_priority,
        running=running,
        completed_last_hour=stats.get(f"completed:{_hour_bucket()}", 0),
        failed_last_hour=stats.get(f"failed:{_hour_bucket()}", 0),
        dlq_size=dlq,
        workers_active=workers_active(),
        backpressure_active=depth >= cfg.queue.backpressure_threshold,
    )


@app.get("/queue/dlq", response_model=list[JobResponse])
async def list_dlq():
    """List all jobs in the Dead Letter Queue."""
    jobs = await get_dlq_jobs()
    return [JobResponse(**j.model_dump()) for j in jobs]


@app.post("/queue/dlq/{job_id}/retry", response_model=JobResponse)
async def retry_dlq_job(job_id: str, raw_request: Request):
    """Move a DLQ job back to the queue for retry."""
    _require_admin(raw_request)
    job = await retry_job(job_id)
    if not job:
        raise HTTPException(status_code=400, detail="Job not found in DLQ or not retryable")
    return JobResponse(**job.model_dump())


# ─── System Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check — verifies Redis connectivity."""
    try:
        r = await get_redis()
        await r.ping()
        return {
            "status": "healthy",
            "redis": "connected",
            "workers": workers_active(),
            "queue_depth": await queue_depth(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "redis": str(e)},
        )


@app.post("/reload-config")
async def reload_config_endpoint():
    """Hot-reload config from disk."""
    cfg = reload_config()
    return {"status": "reloaded", "config": cfg.model_dump()}


# ─── Dashboard ───────────────────────────────────────────────────────────────


@app.get("/")
async def dashboard():
    """Serve the monitoring dashboard."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return {"message": "Async Inference Queue API", "docs": "/docs"}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _hour_bucket() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")
