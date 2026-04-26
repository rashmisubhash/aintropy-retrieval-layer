"""Hit the running /query endpoint with the first 50 MS MARCO queries and
report per-stage p50/p95/p99/mean latencies.

Usage:
    # (server must already be running)
    make serve  &
    python scripts/quick_bench.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QUERIES_PATH  # noqa: E402
from src.timing import TimingAggregator, TimingResult  # noqa: E402

N_QUERIES = 50
SERVER_URL = "http://127.0.0.1:8000/query"


def main() -> None:
    with open(QUERIES_PATH) as f:
        queries = json.load(f)[:N_QUERIES]

    agg = TimingAggregator()

    with httpx.Client(timeout=30.0) as client:
        for i, q in enumerate(queries):
            resp = client.post(SERVER_URL, json={"query": q["query_text"]})
            resp.raise_for_status()
            data = resp.json()
            for t in data["timings"]:
                agg.add(TimingResult(**t))
            agg.add(
                TimingResult(
                    stage_name="end_to_end",
                    duration_ms=resp.elapsed.total_seconds() * 1000,
                )
            )
            print(
                f"[{i + 1:>2}/{len(queries)}] {q['query_id']} "
                f"e2e={resp.elapsed.total_seconds() * 1000:.1f}ms"
            )

    print()
    header = f"{'stage':<16} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'mean_ms':>10} {'n':>6}"
    print(header)
    print("-" * len(header))
    for stage in ["embed", "vector_search", "rerank", "end_to_end"]:
        s = agg.stats(stage)
        if s.count == 0:
            continue
        print(
            f"{s.stage_name:<16} "
            f"{s.p50_ms:>10.2f} {s.p95_ms:>10.2f} {s.p99_ms:>10.2f} "
            f"{s.mean_ms:>10.2f} {s.count:>6d}"
        )


if __name__ == "__main__":
    main()
