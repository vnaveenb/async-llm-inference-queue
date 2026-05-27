"""Submit a batch of inference jobs with varying priorities.

Usage:
    python -m examples.submit_jobs
    python -m examples.submit_jobs --count 20 --base-url http://localhost:8310
"""
from __future__ import annotations

import argparse
import random
import httpx
import time


PROMPTS = [
    "Explain quantum computing in simple terms",
    "What are the key principles of distributed systems?",
    "Write a haiku about machine learning",
    "Compare REST and GraphQL APIs",
    "What is the CAP theorem?",
    "Explain the difference between concurrency and parallelism",
    "What are the SOLID principles in software engineering?",
    "Describe how a neural network learns",
    "What is eventual consistency?",
    "Explain the actor model for concurrent computation",
]

PRIORITIES = ["critical", "high", "normal", "normal", "normal", "low"]


def main():
    parser = argparse.ArgumentParser(description="Submit batch inference jobs")
    parser.add_argument("--count", type=int, default=10, help="Number of jobs to submit")
    parser.add_argument("--base-url", default="http://localhost:8310", help="API base URL")
    args = parser.parse_args()

    print(f"Submitting {args.count} jobs to {args.base_url}...")
    print("-" * 60)

    submitted = []
    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        for i in range(args.count):
            prompt = random.choice(PROMPTS)
            priority = random.choice(PRIORITIES)

            resp = client.post("/jobs", json={
                "prompt": prompt,
                "priority": priority,
                "metadata": {"batch_id": f"example-{int(time.time())}"},
            })

            if resp.status_code == 200:
                data = resp.json()
                submitted.append(data["job_id"])
                print(f"  [{i+1:3d}] {priority:8s} | pos={data['position']:3d} | {data['job_id'][:8]}… | {prompt[:40]}…")
            elif resp.status_code == 429:
                print(f"  [{i+1:3d}] BACKPRESSURE — queue full, stopping")
                break
            else:
                print(f"  [{i+1:3d}] ERROR {resp.status_code}: {resp.text}")

    print("-" * 60)
    print(f"Submitted {len(submitted)} jobs")

    # Poll stats
    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        resp = client.get("/queue/stats")
        if resp.status_code == 200:
            stats = resp.json()
            print(f"\nQueue stats:")
            print(f"  Depth:    {stats['total_depth']}")
            print(f"  Running:  {stats['running']}")
            print(f"  Workers:  {stats['workers_active']}")


if __name__ == "__main__":
    main()
