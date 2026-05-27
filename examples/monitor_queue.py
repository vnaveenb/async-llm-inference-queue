"""Monitor queue stats with live polling.

Usage:
    python -m examples.monitor_queue
    python -m examples.monitor_queue --interval 2 --base-url http://localhost:8310
"""
from __future__ import annotations

import argparse
import time
import httpx
import os


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    parser = argparse.ArgumentParser(description="Monitor inference queue")
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds")
    parser.add_argument("--base-url", default="http://localhost:8310", help="API base URL")
    args = parser.parse_args()

    print(f"Monitoring {args.base_url} (Ctrl+C to stop)\n")

    with httpx.Client(base_url=args.base_url, timeout=5) as client:
        while True:
            try:
                resp = client.get("/queue/stats")
                if resp.status_code != 200:
                    print(f"Error: {resp.status_code}")
                    time.sleep(args.interval)
                    continue

                stats = resp.json()
                clear()
                print("╔══════════════════════════════════════════╗")
                print("║    ASYNC INFERENCE QUEUE — MONITOR       ║")
                print("╠══════════════════════════════════════════╣")
                print(f"║  Queue Depth:      {stats['total_depth']:>6}               ║")
                print(f"║  Running:          {stats['running']:>6}               ║")
                print(f"║  Workers Active:   {stats['workers_active']:>6}               ║")
                print(f"║  Completed (1h):   {stats['completed_last_hour']:>6}               ║")
                print(f"║  Failed (1h):      {stats['failed_last_hour']:>6}               ║")
                print(f"║  DLQ Size:         {stats['dlq_size']:>6}               ║")
                print("╠══════════════════════════════════════════╣")
                print("║  Priority Breakdown:                     ║")
                bp = stats.get("by_priority", {})
                print(f"║    Critical: {bp.get('critical', 0):>4}  High: {bp.get('high', 0):>4}            ║")
                print(f"║    Normal:   {bp.get('normal', 0):>4}  Low:  {bp.get('low', 0):>4}            ║")
                print("╠══════════════════════════════════════════╣")
                bp_status = "🔴 ACTIVE" if stats["backpressure_active"] else "🟢 OK"
                print(f"║  Backpressure: {bp_status:<25} ║")
                print("╚══════════════════════════════════════════╝")
                print(f"\n  Polling every {args.interval}s | Ctrl+C to stop")

            except httpx.ConnectError:
                print("⚠ Connection failed — is the server running?")
            except KeyboardInterrupt:
                print("\nStopped.")
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
