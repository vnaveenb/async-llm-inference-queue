"""Tests for the worker pool."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import fakeredis.aioredis

from src.models import InferenceJob, JobStatus, Priority
from src.queue import (
    close_redis,
    complete_job,
    dequeue,
    dlq_size,
    enqueue,
    get_job,
    queue_depth,
    set_redis,
)
from src.worker import _process_job, start_workers, stop_workers, workers_active


@pytest.fixture(autouse=True)
async def redis_client():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await set_redis(client)
    yield client
    await client.aclose()
    await close_redis()


@pytest.fixture
def make_job():
    def _make(prompt="test", priority=Priority.normal):
        return InferenceJob(prompt=prompt, priority=priority, model="test/model", max_retries=2)
    return _make


@pytest.mark.asyncio
async def test_process_job_success(make_job):
    job = make_job()
    await enqueue(job)
    job = await dequeue()

    mock_result = {
        "result": "test response",
        "input_tokens": 10,
        "output_tokens": 20,
        "latency_ms": 150,
    }

    with patch("src.worker.run_inference", new_callable=AsyncMock, return_value=mock_result):
        await _process_job(job, worker_id=0)

    stored = await get_job(job.id)
    assert stored.status == JobStatus.completed
    assert stored.result == "test response"
    assert stored.input_tokens == 10
    assert stored.output_tokens == 20


@pytest.mark.asyncio
async def test_process_job_failure_requeues(make_job):
    job = make_job()
    job.max_retries = 3
    await enqueue(job)
    job = await dequeue()

    with patch("src.worker.run_inference", new_callable=AsyncMock, side_effect=RuntimeError("API timeout")):
        await _process_job(job, worker_id=0)

    # Job should be re-queued
    depth = await queue_depth()
    assert depth == 1


@pytest.mark.asyncio
async def test_process_job_exhausted_retries_to_dlq(make_job):
    job = make_job()
    job.max_retries = 1
    job.attempts = 1  # Already used the one retry
    await enqueue(job)
    job = await dequeue()

    with patch("src.worker.run_inference", new_callable=AsyncMock, side_effect=RuntimeError("permanent")):
        await _process_job(job, worker_id=0)

    stored = await get_job(job.id)
    assert stored.status == JobStatus.dead

    dlq = await dlq_size()
    assert dlq == 1


@pytest.mark.asyncio
async def test_start_and_stop_workers():
    await start_workers()
    assert workers_active() == 4  # Default concurrency

    await stop_workers()
    assert workers_active() == 0


@pytest.mark.asyncio
async def test_workers_process_queued_jobs(make_job):
    job = make_job()
    await enqueue(job)

    mock_result = {
        "result": "done",
        "input_tokens": 5,
        "output_tokens": 10,
        "latency_ms": 50,
    }

    with patch("src.worker.run_inference", new_callable=AsyncMock, return_value=mock_result):
        await start_workers()
        await asyncio.sleep(0.3)  # Give workers time to pick up the job
        await stop_workers()

    stored = await get_job(job.id)
    assert stored.status == JobStatus.completed
    assert stored.result == "done"
