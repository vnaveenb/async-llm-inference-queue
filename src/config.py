from __future__ import annotations

import os
import yaml
from pathlib import Path
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8300
    reload: bool = False


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    prefix: str = "aiq:"


class QueueConfig(BaseModel):
    max_depth: int = 1000
    backpressure_threshold: int = 800
    priority_levels: list[str] = ["critical", "high", "normal", "low"]
    default_priority: str = "normal"
    job_ttl_hours: int = 24


class WorkerConfig(BaseModel):
    concurrency: int = 4
    poll_interval_ms: int = 100
    max_retries: int = 3
    base_delay: float = 1.0
    jitter: bool = True


class InferenceConfig(BaseModel):
    mock: bool = False
    default_model: str = "gemini/gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: int = 60


class TrackerConfig(BaseModel):
    enabled: bool = True
    project_id: str = "async-inference-queue"


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    redis: RedisConfig = RedisConfig()
    queue: QueueConfig = QueueConfig()
    worker: WorkerConfig = WorkerConfig()
    inference: InferenceConfig = InferenceConfig()
    tracker: TrackerConfig = TrackerConfig()


_config: AppConfig | None = None


def get_config(path: str = "config.yaml") -> AppConfig:
    global _config
    if _config is None:
        cfg_path = Path(path)
        if cfg_path.exists():
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            _config = AppConfig(**raw)
        else:
            _config = AppConfig()
        # Environment variable overrides
        if os.environ.get("REDIS_URL"):
            _config.redis.url = os.environ["REDIS_URL"]
    return _config


def reload_config(path: str = "config.yaml") -> AppConfig:
    """Force reload from disk — used by POST /reload-config."""
    global _config
    _config = None
    return get_config(path)
