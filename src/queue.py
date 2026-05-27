from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import redis.asyncio as redis

from .config import get_config
from .models import InferenceJob, JobStatus, Priority, PRIORITY_SCORES

logger = logging.getLogger(__name__)

_redis: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get or create the Redis connection."""
    global _redis
    if _redis is None:
        cfg = get_config()
        _redis = redis.from_url(cfg.redis.url, decode_responses=True)
    return _redis


async def set_redis(client: redis.Redis) -> None:
    """Override Redis client (for testing with fakeredis)."""
    global _redis
    _redis = client


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def _prefix() -> str:
    return get_config().redis.prefix


def _queue_key() -> str:
    return f"{_prefix()}queue"


def _job_key(job_id: str) -> str:
    return f"{_prefix()}job:{job_id}"


def _dlq_key() -> str:
    return f"{_prefix()}dlq"


def _running_key() -> str:
    return f"{_prefix()}running"


def _stats_key() -> str:
    return f"{_prefix()}stats"


async def enqueue(job: InferenceJob) -> int:
    """Add a job to the priority queue. Returns queue position."""
    r = await get_redis()
    cfg = get_config()

    # Check backpressure
    depth = await queue_depth()
    if depth >= cfg.queue.max_depth:
        raise QueueFullError(f"Queue at max capacity ({cfg.queue.max_depth})")

    job.status = JobStatus.queued
    # Store job data
    await r.set(_job_key(job.id), job.model_dump_json(), ex=cfg.queue.job_ttl_hours * 3600)

    # Add to sorted set with priority score + timestamp for FIFO within same priority
    score = PRIORITY_SCORES[job.priority] * 1e12 + time.time()
    await r.zadd(_queue_key(), {job.id: score})

    position = await r.zrank(_queue_key(), job.id)
    return (position or 0) + 1


async def dequeue() -> Optional[InferenceJob]:
    """Pop the highest-priority job from the queue."""
    r = await get_redis()

    # Get lowest score (highest priority) job
    results = await r.zpopmin(_queue_key(), count=1)
    if not results:
        return None

    job_id, _ = results[0]
    job_data = await r.get(_job_key(job_id))
    if not job_data:
        return None

    job = InferenceJob.model_validate_json(job_data)
    job.status = JobStatus.running
    job.started_at = datetime.now(timezone.utc).isoformat()
    job.attempts += 1

    # Track as running
    await r.sadd(_running_key(), job.id)
    await r.set(_job_key(job.id), job.model_dump_json(), ex=get_config().queue.job_ttl_hours * 3600)

    return job


async def complete_job(job: InferenceJob, result: str, input_tokens: int = 0, output_tokens: int = 0, latency_ms: int = 0) -> None:
    """Mark a job as successfully completed."""
    r = await get_redis()
    cfg = get_config()

    job.status = JobStatus.completed
    job.completed_at = datetime.now(timezone.utc).isoformat()
    job.result = result
    job.input_tokens = input_tokens
    job.output_tokens = output_tokens
    job.latency_ms = latency_ms

    await r.set(_job_key(job.id), job.model_dump_json(), ex=cfg.queue.job_ttl_hours * 3600)
    await r.srem(_running_key(), job.id)

    # Increment stats
    await r.hincrby(_stats_key(), "completed", 1)
    await r.hincrby(_stats_key(), f"completed:{_hour_bucket()}", 1)


async def fail_job(job: InferenceJob, error: str) -> None:
    """Mark a job as failed. Moves to DLQ if retries exhausted."""
    r = await get_redis()
    cfg = get_config()

    job.error = error
    await r.srem(_running_key(), job.id)

    if job.attempts >= job.max_retries:
        # Move to Dead Letter Queue
        job.status = JobStatus.dead
        await r.set(_job_key(job.id), job.model_dump_json(), ex=cfg.queue.job_ttl_hours * 3600)
        await r.lpush(_dlq_key(), job.id)
        await r.hincrby(_stats_key(), "dead", 1)
        logger.warning(f"Job {job.id} moved to DLQ after {job.attempts} attempts")
    else:
        # Re-enqueue for retry
        job.status = JobStatus.queued
        job.started_at = None
        await r.set(_job_key(job.id), job.model_dump_json(), ex=cfg.queue.job_ttl_hours * 3600)
        score = PRIORITY_SCORES[job.priority] * 1e12 + time.time()
        await r.zadd(_queue_key(), {job.id: score})
        logger.info(f"Job {job.id} re-queued (attempt {job.attempts}/{job.max_retries})")

    await r.hincrby(_stats_key(), "failed", 1)
    await r.hincrby(_stats_key(), f"failed:{_hour_bucket()}", 1)


async def get_job(job_id: str) -> Optional[InferenceJob]:
    """Get a job by ID."""
    r = await get_redis()
    data = await r.get(_job_key(job_id))
    if not data:
        return None
    return InferenceJob.model_validate_json(data)


async def cancel_job(job_id: str) -> Optional[InferenceJob]:
    """Cancel a pending/queued job."""
    r = await get_redis()
    job = await get_job(job_id)
    if not job:
        return None
    if job.status not in (JobStatus.pending, JobStatus.queued):
        return None

    job.status = JobStatus.cancelled
    job.completed_at = datetime.now(timezone.utc).isoformat()
    await r.set(_job_key(job.id), job.model_dump_json(), ex=get_config().queue.job_ttl_hours * 3600)
    await r.zrem(_queue_key(), job.id)
    return job


async def retry_job(job_id: str) -> Optional[InferenceJob]:
    """Retry a failed/dead job by re-enqueuing it."""
    r = await get_redis()
    job = await get_job(job_id)
    if not job:
        return None
    if job.status not in (JobStatus.failed, JobStatus.dead):
        return None

    job.status = JobStatus.queued
    job.error = None
    job.started_at = None
    job.attempts = 0

    await r.set(_job_key(job.id), job.model_dump_json(), ex=get_config().queue.job_ttl_hours * 3600)
    score = PRIORITY_SCORES[job.priority] * 1e12 + time.time()
    await r.zadd(_queue_key(), {job.id: score})

    # Remove from DLQ if present
    await r.lrem(_dlq_key(), 0, job.id)
    return job


async def get_dlq_jobs() -> list[InferenceJob]:
    """Get all jobs in the Dead Letter Queue."""
    r = await get_redis()
    job_ids = await r.lrange(_dlq_key(), 0, -1)
    jobs = []
    for jid in job_ids:
        job = await get_job(jid)
        if job:
            jobs.append(job)
    return jobs


async def queue_depth() -> int:
    """Total number of jobs waiting in the queue."""
    r = await get_redis()
    return await r.zcard(_queue_key())


async def queue_depth_by_priority() -> dict[str, int]:
    """Job count per priority level."""
    r = await get_redis()
    counts = {}
    for priority in Priority:
        score_min = PRIORITY_SCORES[priority] * 1e12
        score_max = (PRIORITY_SCORES[priority] + 1) * 1e12
        count = await r.zcount(_queue_key(), score_min, score_max)
        counts[priority.value] = count
    return counts


async def running_count() -> int:
    """Number of currently running jobs."""
    r = await get_redis()
    return await r.scard(_running_key())


async def dlq_size() -> int:
    """Number of jobs in the Dead Letter Queue."""
    r = await get_redis()
    return await r.llen(_dlq_key())


async def get_stats() -> dict[str, int]:
    """Get cumulative stats."""
    r = await get_redis()
    raw = await r.hgetall(_stats_key())
    return {k: int(v) for k, v in raw.items()}


def _hour_bucket() -> str:
    """Current hour bucket for time-based stats."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")


class QueueFullError(Exception):
    pass
