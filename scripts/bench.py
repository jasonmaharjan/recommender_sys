"""Benchmark /recommend latency: cache-miss pass + cache-hit pass + percentiles."""

from __future__ import annotations

import random
import statistics
import sys
import time
import urllib.request

import pandas as pd

URL = "http://127.0.0.1:8000/recommend/{user_id}?k=10"
N_REQUESTS = 200


def hit(user_id: int) -> float:
    t0 = time.perf_counter()
    with urllib.request.urlopen(URL.format(user_id=user_id), timeout=30) as r:  # noqa: S310
        r.read()
    return (time.perf_counter() - t0) * 1000


def percentiles(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    return {
        "p50": s[int(0.50 * len(s))],
        "p95": s[int(0.95 * len(s))],
        "p99": s[int(0.99 * len(s))],
        "mean": statistics.mean(s),
        "max": s[-1],
        "n": len(s),
    }


def main() -> int:
    df = pd.read_parquet("data/processed/train.parquet", columns=["user_id"])
    user_ids = df["user_id"].drop_duplicates().to_numpy()
    random.seed(42)
    sample = random.sample(list(user_ids), N_REQUESTS)

    print(f"warming up...")
    hit(sample[0])

    print(f"cache-miss pass ({N_REQUESTS} requests, distinct users)...")
    miss = [hit(int(u)) for u in sample]

    print(f"cache-hit pass ({N_REQUESTS} requests, same users)...")
    hits = [hit(int(u)) for u in sample]

    print()
    print(f"{'metric':<10}{'cache_miss (ms)':>20}{'cache_hit (ms)':>20}")
    m, h = percentiles(miss), percentiles(hits)
    for key in ("p50", "p95", "p99", "mean", "max"):
        print(f"{key:<10}{m[key]:>20.2f}{h[key]:>20.2f}")
    print(f"\nthroughput cache-miss: {1000 / m['mean']:.1f} req/s/worker")
    print(f"throughput cache-hit:  {1000 / h['mean']:.1f} req/s/worker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
