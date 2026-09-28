"""Evaluation metrics for tile retrieval."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def euclid_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


@dataclass
class RetrievalEvalResult:
    n: int = 0
    recall_at_k: dict[int, float] = field(default_factory=dict)
    errors_m_rank1: list[float] = field(default_factory=list)
    median_error_m: float = float("nan")
    mean_error_m: float = float("nan")
    p90_error_m: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "recall_at_k": self.recall_at_k,
            "median_error_m": self.median_error_m,
            "mean_error_m": self.mean_error_m,
            "p90_error_m": self.p90_error_m,
        }


def evaluate_retrieval(
    gt_xy: Sequence[tuple[float, float]],
    candidate_xy_per_query: Sequence[Sequence[tuple[float, float]]],
    k_values: Sequence[int] = (1, 5, 10),
    tolerance_m: float = 30.0,
) -> RetrievalEvalResult:
    """
    Generic retrieval evaluation.

    For each query:
      gt_xy[i] = (x, y) ground-truth position in EPSG:5514.
      candidate_xy_per_query[i] = list of (x, y) for the retrieved candidates,
      sorted by similarity score descending.

    'recall_at_k' counts a query as a hit at k if any of the top-k candidate
    centres lies within tolerance_m of the ground-truth position.
    """
    n = len(gt_xy)
    if n == 0:
        return RetrievalEvalResult()
    if len(candidate_xy_per_query) != n:
        raise ValueError("Length mismatch between gt and candidates.")

    k_max = max(k_values)
    rank1_errors: list[float] = []
    hits_at_k = {k: 0 for k in k_values}

    for gt, cands in zip(gt_xy, candidate_xy_per_query):
        if not cands:
            rank1_errors.append(float("nan"))
            continue
        # rank1
        d_rank1 = euclid_xy(gt[0], gt[1], cands[0][0], cands[0][1])
        rank1_errors.append(d_rank1)
        # any in top-k within tolerance
        for k in k_values:
            for cand in cands[:k]:
                if euclid_xy(gt[0], gt[1], cand[0], cand[1]) <= tolerance_m:
                    hits_at_k[k] += 1
                    break

    finite = [e for e in rank1_errors if not math.isnan(e)]
    arr = np.array(finite) if finite else np.array([float("nan")])

    res = RetrievalEvalResult(n=n)
    res.errors_m_rank1 = rank1_errors
    res.median_error_m = float(np.median(arr))
    res.mean_error_m = float(np.mean(arr))
    res.p90_error_m = float(np.percentile(arr, 90)) if finite else float("nan")
    res.recall_at_k = {k: hits_at_k[k] / n for k in k_values}
    return res
