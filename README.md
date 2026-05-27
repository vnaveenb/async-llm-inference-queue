# 07 — Async Inference Queue

> Redis-backed priority job queue for LLM inference with configurable workers, backpressure, dead-letter queue, and real-time monitoring dashboard.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![Redis](https://img.shields.io/badge/Redis-7+-DC382D)
![LiteLLM](https://img.shields.io/badge/LiteLLM-1.67+-orange)

---

## Why This Project

Production LLM systems need asynchronous processing — you can't hold an HTTP connection open for 30+ seconds while a model generates. This project demonstrates:

- **Job lifecycle management** — submit → queue → process → return results
- **Priority scheduling** — critical requests skip the line
- **Backpressure** — graceful degradation under load (429 responses)
- **Dead Letter Queue** — failed jobs don't disappear; they're inspectable and retryable
- **Worker concurrency control** — bounded parallelism via configurable pool
- **Dual-mode inference** — real LiteLLM calls or mock mode for demos

---

## Quick Start

```bash
# Clone and enter project
cd projects/07-async-inference-queue

# Option A: Docker (recommended)
docker compose up --build

# Option B: Local dev
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Start Redis (WSL or Docker): redis-server
uvicorn src.api:app --reload --port 8300
```

```bash
# Submit a job
curl -X POST http://localhost:8310/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain async queues", "priority": "high"}'

# Check status
curl http://localhost:8310/jobs/{job_id}

# View queue stats
curl http://localhost:8310/queue/stats

# Dashboard
open http://localhost:8310/
```

---

## Architecture

### System Overview

```
┌─────────────┐         ┌──────────────────────────────────────────────────────┐
│   Client    │         │              Async Inference Queue                    │
│  (curl/UI)  │         │                                                      │
└─────┬───────┘         │  ┌─────────┐    ┌───────────┐    ┌───────────────┐  │
      │                 │  │ FastAPI  │───▶│   Redis   │───▶│  Worker Pool  │  │
      │  POST /jobs     │  │  + CORS  │    │  Sorted   │    │  (N workers)  │  │
      ├────────────────▶│  │  + BP    │    │   Sets    │    │               │  │
      │                 │  └────┬─────┘    └─────┬─────┘    └───────┬───────┘  │
      │  GET /jobs/{id} │       │                │                  │          │
      ├────────────────▶│       │                │ DLQ              ▼          │
      │                 │       │                │           ┌─────────────┐   │
      │  GET /queue/... │       │                └──────────▶│  LiteLLM /  │   │
      ├────────────────▶│       ▼                            │  Mock Mode  │   │
      │                 │  ┌─────────┐                       └──────┬──────┘   │
      │  GET /          │  │Dashboard│                              │          │
      ├────────────────▶│  │ (HTML)  │                              ▼          │
      │                 │  └─────────┘                       ┌─────────────┐   │
      │                 │                                    │ Project 06   │   │
      │                 │                                    │Cost Tracker  │   │
      │                 │                                    └─────────────┘   │
      │                 └──────────────────────────────────────────────────────┘
      │
```

### Job Lifecycle

```
  SUBMIT                    WORKER PICKS UP              INFERENCE
    │                            │                          │
    ▼                            ▼                          ▼
┌────────┐  enqueue()  ┌────────┐  dequeue()  ┌─────────┐     ┌───────────┐
│PENDING │────────────▶│ QUEUED │────────────▶│ RUNNING │────▶│ COMPLETED │
└────────┘             └────┬───┘             └────┬────┘     └───────────┘
                            │                      │
                      cancel_job()            on failure
                            │                      │
                            ▼                      ▼
                     ┌───────────┐          ┌──────────┐  retries < max
                     │ CANCELLED │          │  FAILED  │────────────────▶ QUEUED
                     └───────────┘          └────┬─────┘
                                                 │ retries exhausted
                                                 ▼
                                           ┌──────────┐  manual retry
                                           │   DEAD   │────────────────▶ QUEUED
                                           │  (DLQ)   │
                                           └──────────┘
```

### Priority Queue Ordering

```
Redis Sorted Set scores:  priority_level × 1e12 + unix_timestamp

  Score 0×10¹² + ts  →  Critical  (processed first)
  Score 1×10¹² + ts  →  High
  Score 2×10¹² + ts  →  Normal
  Score 3×10¹² + ts  →  Low       (processed last)

Within same priority: FIFO by submission time
```

### Backpressure Flow

```
  POST /jobs
      │
      ▼
  ┌──────────────────────┐
  │ BackpressureMiddleware│
  │                      │
  │ queue_depth >= 800?  │──── YES ────▶ 429 Too Many Requests
  │                      │              {"error": "Queue backpressure active"}
  └──────────┬───────────┘
             │ NO
             ▼
       Normal processing
```

---

## API Endpoints

### Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit inference request → returns `job_id` |
| `GET` | `/jobs/{id}` | Get job status + result |
| `GET` | `/jobs` | List jobs (filters: `?status=`, `?priority=`, `?limit=`) |
| `POST` | `/jobs/{id}/cancel` | Cancel a pending/queued job |
| `POST` | `/jobs/{id}/retry` | Retry a failed/dead job |

### Queue

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/queue/stats` | Queue depth, running count, DLQ size, worker status |
| `GET` | `/queue/dlq` | List all dead-letter jobs |
| `POST` | `/queue/dlq/{id}/retry` | Move DLQ job back to queue |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service + Redis connectivity check |
| `POST` | `/reload-config` | Hot-reload `config.yaml` |
| `GET` | `/` | Monitoring dashboard |

---

## Project Structure

```
07-async-inference-queue/
├── config.yaml              # Queue, worker, inference, and server settings
├── docker-compose.yml       # queue-api + Redis services
├── Dockerfile               # python:3.12-slim, non-root user
├── requirements.txt         # Pinned dependencies
├── .env.example             # Required environment variables
├── src/
│   ├── __init__.py
│   ├── config.py            # Pydantic config models + get_config()/reload_config()
│   ├── models.py            # Job, status, priority, queue stats models
│   ├── queue.py             # Redis queue operations (enqueue/dequeue/DLQ/stats)
│   ├── worker.py            # Async worker pool with graceful shutdown
│   ├── inference.py         # LiteLLM real + mock inference executor
│   ├── backpressure.py      # 429 middleware when queue threshold exceeded
│   ├── api.py               # FastAPI app with all endpoints
│   └── static/
│       └── index.html       # Real-time monitoring dashboard (Chart.js)
├── tests/
│   ├── __init__.py
│   ├── test_queue.py        # 14 tests — queue ops, priority, DLQ, cancel/retry
│   ├── test_worker.py       # 5 tests — job processing, retry, worker lifecycle
│   ├── test_api.py          # 12 tests — endpoints, backpressure, CRUD
│   └── test_inference.py    # 5 tests — mock mode, real mode, tracker integration
├── examples/
│   ├── submit_jobs.py       # Batch submission with varying priorities
│   └── monitor_queue.py     # CLI queue monitor with live polling
└── data/                    # Mounted volume (tracker DB, if enabled)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Redis sorted sets for priority queue | O(log N) insert/pop, native score-based ordering, industry standard |
| No Celery | Raw redis + asyncio demonstrates deeper understanding than framework wrapper |
| asyncio worker pool (not threads) | LLM calls are I/O-bound; async avoids GIL and thread overhead |
| Backpressure via middleware | Centralized, transparent — clients get clear 429 + queue depth info |
| fakeredis for tests | No external Redis dependency in CI; tests run fully isolated |
| DLQ as separate list | Dead jobs are inspectable, retryable, and don't clog the main queue |
| Dual mock/real mode | `inference.mock: true` allows demos without API keys |
| Redis AOF persistence | `--appendonly yes` ensures job durability across container restarts |

---

## Configuration

All settings in `config.yaml`:

```yaml
queue:
  max_depth: 1000              # Hard limit on total queued jobs
  backpressure_threshold: 800  # 429 starts at this depth
  job_ttl_hours: 24            # Auto-expire completed job data

worker:
  concurrency: 4               # Number of parallel workers
  max_retries: 3               # Attempts before DLQ
  base_delay: 1.0              # Exponential backoff base (seconds)

inference:
  mock: false                  # true = no API key needed
  default_model: gemini/gemini-2.5-flash
```

---

## Integration with Project 06

When deployed alongside the Token + Cost Budget Dashboard (Project 06), inference costs are automatically tracked:

```yaml
# docker-compose.yml shares the tracker_data volume
volumes:
  tracker_data:
    external: true
```

Set `TRACKER_DB_PATH=/app/data/usage.db` and the queue will log every inference call to the shared SQLite database.

---

## Test Coverage

**36 tests** across 4 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_queue.py` | 14 | Queue operations, priority ordering, FIFO, DLQ, cancel/retry |
| `test_api.py` | 12 | All endpoints, status codes, filters, backpressure |
| `test_worker.py` | 5 | Job processing, failure→retry, failure→DLQ, worker lifecycle |
| `test_inference.py` | 5 | Mock mode, real mode, latency, tracker metadata |

```bash
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Create shared volume (one-time, shared with Project 06)
docker volume create tracker_data

# Build and run
docker compose up --build -d

# Verify
curl http://localhost:8310/health
```

**Ports:**
- `8310` → queue-api (external)
- `8300` → queue-api (internal)
- Redis: internal only (no host port conflict)

---

## Upgrade Path

- **Project 8 (Batch Inference Worker)** — different pattern: file-based batch processing with progress tracking
- **Project 9 (Model Fallback Router)** — can sit in front of this queue for model selection + failover
