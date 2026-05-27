from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from .config import get_config
from .inference import run_inference
from .models import InferenceJob
from .queue import dequeue, complete_job, fail_job, running_count

logger = logging.getLogger(__name__)

_workers: list[asyncio.Task] = []
_shutdown_event: Optional[asyncio.Event] = None


async def start_workers() -> None:
    """Start the worker pool based on config concurrency."""
    global _shutdown_event
    cfg = get_config()
    _shutdown_event = asyncio.Event()

    for i in range(cfg.worker.concurrency):
        task = asyncio.create_task(_worker_loop(i), name=f"worker-{i}")
        _workers.append(task)

    logger.info(f"Started {cfg.worker.concurrency} inference workers")


async def stop_workers() -> None:
    """Gracefully stop all workers."""
    if _shutdown_event:
        _shutdown_event.set()

    for task in _workers:
        task.cancel()

    if _workers:
        await asyncio.gather(*_workers, return_exceptions=True)
    _workers.clear()
    logger.info("All workers stopped")


async def _worker_loop(worker_id: int) -> None:
    """Main worker loop: poll queue, execute inference, handle results."""
    cfg = get_config()
    poll_interval = cfg.worker.poll_interval_ms / 1000.0

    logger.info(f"Worker-{worker_id} started")

    while not (_shutdown_event and _shutdown_event.is_set()):
        try:
            job = await dequeue()
            if job is None:
                await asyncio.sleep(poll_interval)
                continue

            logger.info(f"Worker-{worker_id} processing job {job.id} (attempt {job.attempts})")
            await _process_job(job, worker_id)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Worker-{worker_id} unexpected error: {e}")
            await asyncio.sleep(poll_interval)

    logger.info(f"Worker-{worker_id} stopped")


async def _process_job(job: InferenceJob, worker_id: int) -> None:
    """Execute a single job with retry-aware error handling."""
    cfg = get_config()

    try:
        result = await run_inference(job)
        await complete_job(
            job,
            result=result["result"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=result["latency_ms"],
        )
        logger.info(
            f"Worker-{worker_id} completed job {job.id} "
            f"({result['input_tokens']}+{result['output_tokens']} tokens, {result['latency_ms']}ms)"
        )

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Worker-{worker_id} job {job.id} failed: {error_msg}")

        # Apply backoff delay before allowing retry
        if job.attempts < job.max_retries:
            delay = cfg.worker.base_delay * (2 ** (job.attempts - 1))
            if cfg.worker.jitter:
                delay += random.uniform(0, 0.5)
            await asyncio.sleep(delay)

        await fail_job(job, error_msg)


def workers_active() -> int:
    """Number of currently alive worker tasks."""
    return sum(1 for t in _workers if not t.done())
