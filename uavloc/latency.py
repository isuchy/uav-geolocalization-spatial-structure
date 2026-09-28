"""Lightweight latency profiling."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class LatencyRecorder:
    samples_ms: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.samples_ms.setdefault(name, []).append(dt)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.samples_ms.items():
            arr = np.array(v, dtype=float) if v else np.array([float("nan")])
            out[k] = {
                "n": int(len(v)),
                "mean_ms": float(np.mean(arr)),
                "median_ms": float(np.median(arr)),
                "p90_ms": float(np.percentile(arr, 90)) if v else float("nan"),
                "p99_ms": float(np.percentile(arr, 99)) if v else float("nan"),
            }
        return out
