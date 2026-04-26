"""Instrumentation helpers for per-stage latency measurement."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TimingResult:
    stage_name: str
    duration_ms: float


class StageTimer:
    """Context manager that records wall-clock duration of a named stage.

    Usage:
        with StageTimer("embed") as t:
            ...
        print(t.result.duration_ms)
    """

    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        self._start_ns: int | None = None
        self.result: TimingResult | None = None

    def __enter__(self) -> "StageTimer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end_ns = time.perf_counter_ns()
        assert self._start_ns is not None
        duration_ms = (end_ns - self._start_ns) / 1_000_000
        self.result = TimingResult(stage_name=self.stage_name, duration_ms=duration_ms)


@dataclass
class StageStats:
    stage_name: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class TimingAggregator:
    """Collects TimingResults across many queries and computes per-stage stats."""

    def __init__(self) -> None:
        self._by_stage: dict[str, list[float]] = defaultdict(list)

    def add(self, result: TimingResult) -> None:
        self._by_stage[result.stage_name].append(result.duration_ms)

    def add_many(self, results: Iterable[TimingResult]) -> None:
        for r in results:
            self.add(r)

    def stages(self) -> list[str]:
        return list(self._by_stage.keys())

    def stats(self, stage: str) -> StageStats:
        vals = self._by_stage.get(stage, [])
        if not vals:
            return StageStats(stage, 0, 0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(vals, dtype=float)
        return StageStats(
            stage_name=stage,
            count=len(vals),
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
        )

    def summary(self) -> dict[str, StageStats]:
        return {s: self.stats(s) for s in self._by_stage}
