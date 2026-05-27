"""Tests for the Redis queue operations."""
from __future__ import annotations

import asyncio
import pytest
import fakeredis.aioredis

from src.models import InferenceJob, JobStatus, Priority
from src.queue import (
    cancel_job,
    close_redis,
    complete_job,
    dequeue,
    dlq_size,
    enqueue,
    fail_job,
    get_dlq_jobs,
    get_job,
    queue_depth,
    queue_depth_by_priority,
    retry_job,
    running_count,
    set_redis,
)


@pytest.fixture(autouse=True)
async def redis_client():
    """Provide a fresh fakeredis instance for each test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await set_redis(client)
    yield client
    await client.aclose()
    await close_redis()


@pytest.fixture
def make_job():
    """Factory for creating test jobs."""
    def _make(prompt="test prompt", priority=Priority.normal, model="test/model"):
        return InferenceJob(prompt=prompt, priority=priority, model=model)
    return _make


@pytest.mark.asyncio
async def test_enqueue_and_dequeue(make_job):
    job = make_job()
    position = await enqueue(job)
    assert position == 1

    depth = await queue_depth()
    assert depth == 1

    dequeued = await dequeue()
    assert dequeued is not None
    assert dequeued.id == job.id
    assert dequeued.status == JobStatus.running
    assert dequeued.attempts == 1


@pytest.mark.asyncio
async def test_dequeue_empty_returns_none():
    result = await dequeue()
    assert result is None


@pytest.mark.asyncio
async def test_priority_ordering(make_job):
    low = make_job(prompt="low", priority=Priority.low)
    high = make_job(prompt="high", priority=Priority.high)
    critical = make_job(prompt="critical", priority=Priority.critical)
    normal = make_job(prompt="normal", priority=Priority.normal)

    await enqueue(low)
    await enqueue(high)
    await enqueue(critical)
    await enqueue(normal)

    first = await dequeue()
    assert first.priority == Priority.critical
    second = await dequeue()
    assert second.priority == Priority.high
    third = await dequeue()
    assert third.priority == Priority.normal
    fourth = await dequeue()
    assert fourth.priority == Priority.low


@pytest.mark.asyncio
async def test_fifo_within_same_priority(make_job):
    job1 = make_job(prompt="first")
    job2 = make_job(prompt="second")

    await enqueue(job1)
    await asyncio.sleep(0.01)  # Ensure different timestamp
    await enqueue(job2)

    first = await dequeue()
    assert first.id == job1.id
    second = await dequeue()
    assert second.id == job2.id


@pytest.mark.asyncio
async def test_complete_job(make_job):
    job = make_job()
    await enqueue(job)
    job = await dequeue()

    await complete_job(job, result="hello", input_tokens=10, output_tokens=5, latency_ms=100)

    stored = await get_job(job.id)
    assert stored.status == JobStatus.completed
    assert stored.result == "hello"
    assert stored.input_tokens == 10
    assert stored.output_tokens == 5
    assert stored.latency_ms == 100
    assert stored.completed_at is not None


@pytest.mark.asyncio
async def test_fail_job_requeues_under_max_retries(make_job):
    job = make_job()
    job.max_retries = 3
    await enqueue(job)
    job = await dequeue()

    await fail_job(job, "timeout")

    # Job should be back in queue
    depth = await queue_depth()
    assert depth == 1

    requeued = await get_job(job.id)
    assert requeued.status == JobStatus.queued
    assert requeued.error == "timeout"


@pytest.mark.asyncio
async def test_fail_job_moves_to_dlq_after_max_retries(make_job):
    job = make_job()
    job.max_retries = 2
    job.attempts = 2  # Already at max
    await enqueue(job)
    job = await dequeue()

    await fail_job(job, "permanent failure")

    stored = await get_job(job.id)
    assert stored.status == JobStatus.dead

    dlq = await dlq_size()
    assert dlq == 1


@pytest.mark.asyncio
async def test_cancel_job(make_job):
    job = make_job()
    await enqueue(job)

    cancelled = await cancel_job(job.id)
    assert cancelled is not None
    assert cancelled.status == JobStatus.cancelled

    depth = await queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_cancel_running_job_fails(make_job):
    job = make_job()
    await enqueue(job)
    await dequeue()  # Now running

    result = await cancel_job(job.id)
    assert result is None  # Can't cancel running jobs


@pytest.mark.asyncio
async def test_retry_dead_job(make_job):
    job = make_job()
    job.max_retries = 1
    job.attempts = 1
    await enqueue(job)
    job = await dequeue()
    await fail_job(job, "error")

    retried = await retry_job(job.id)
    assert retried is not None
    assert retried.status == JobStatus.queued
    assert retried.attempts == 0
    assert retried.error is None

    depth = await queue_depth()
    assert depth == 1


@pytest.mark.asyncio
async def test_queue_depth_by_priority(make_job):
    await enqueue(make_job(priority=Priority.critical))
    await enqueue(make_job(priority=Priority.critical))
    await enqueue(make_job(priority=Priority.high))
    await enqueue(make_job(priority=Priority.normal))

    counts = await queue_depth_by_priority()
    assert counts["critical"] == 2
    assert counts["high"] == 1
    assert counts["normal"] == 1
    assert counts["low"] == 0


@pytest.mark.asyncio
async def test_running_count(make_job):
    job1 = make_job()
    job2 = make_job()
    await enqueue(job1)
    await enqueue(job2)
    await dequeue()
    await dequeue()

    count = await running_count()
    assert count == 2


@pytest.mark.asyncio
async def test_get_dlq_jobs(make_job):
    job = make_job()
    job.max_retries = 1
    job.attempts = 1
    await enqueue(job)
    job = await dequeue()
    await fail_job(job, "dead")

    dlq_jobs = await get_dlq_jobs()
    assert len(dlq_jobs) == 1
    assert dlq_jobs[0].id == job.id
