from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Priority(str, Enum):
    critical = "critical"
    high = "high"
    normal = "normal"
    low = "low"


class JobStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    dead = "dead"  # In DLQ after max retries


PRIORITY_SCORES = {
    Priority.critical: 0,
    Priority.high: 1,
    Priority.normal: 2,
    Priority.low: 3,
}


class JobRequest(BaseModel):
    """Client-submitted inference request."""
    prompt: str
    model: Optional[str] = None
    priority: Priority = Priority.normal
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceJob(BaseModel):
    """Full job record stored in Redis."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    model: str
    priority: Priority = Priority.normal
    status: JobStatus = JobStatus.pending
    max_tokens: int = 1024
    temperature: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class JobResponse(BaseModel):
    """API response for a single job."""
    id: str
    status: JobStatus
    priority: Priority
    model: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None
    result: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class QueueStats(BaseModel):
    """Queue health metrics."""
    total_depth: int = 0
    by_priority: dict[str, int] = Field(default_factory=dict)
    running: int = 0
    completed_last_hour: int = 0
    failed_last_hour: int = 0
    dlq_size: int = 0
    workers_active: int = 0
    backpressure_active: bool = False


class SubmitResponse(BaseModel):
    """Response after job submission."""
    job_id: str
    status: JobStatus
    position: int
