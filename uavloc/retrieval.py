"""
HierarchicalRetriever — two-stage page retrieval used by smoke test (05),
synthetic eval (07), real-drone eval (08) and the descriptor ablation (09).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np

from .page_cache import PageCache


@dataclass
class Candidate:
    tile_id: int
    score: float
    page_id: str


class HierarchicalRetriever:
    """
    Coarse -> per-page two-stage search. Optionally consults a PageCache:
      - if a query position (x, y) is provided, the page that contains it
        is forced into the candidate set (always-on local search),
      - neighbours can be prefetched via the cache's note_query_position
        + maybe_prefetch combination.

    The retriever expects L2-normalised query vectors and uses inner product
    as similarity (== cosine for unit vectors).
    """

    def __init__(
        self,
        index_dir: Path,
        cache: Optional[PageCache] = None,
    ):
        self.index_dir = Path(index_dir)
        self._coarse = faiss.read_index(str(self.index_dir / "coarse_pages.faiss"))
        self._page_ids: list[str] = json.loads(
            (self.index_dir / "coarse_page_ids.json").read_text(encoding="utf-8")
        )
        # Cache is optional. If not given, instantiate a small default one.
        self.cache = cache or PageCache(self.index_dir, max_pages_in_ram=8)

    def _coarse_page_candidates(self, q: np.ndarray, top_pages: int) -> list[str]:
        scores, idxs = self._coarse.search(q.astype("float32"), top_pages)
        out = []
        for idx in idxs[0]:
            if idx < 0:
                continue
            out.append(self._page_ids[int(idx)])
        return out

    def search(
        self,
        q: np.ndarray,
        top_pages: int = 5,
        top_k: int = 10,
        query_xy: Optional[tuple[float, float]] = None,
    ) -> list[Candidate]:
        if q.ndim == 1:
            q = q.reshape(1, -1)
        # L2-normalise defensively.
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = (q / np.maximum(norms, 1e-12)).astype("float32")

        candidate_pages = self._coarse_page_candidates(q, top_pages)
        if query_xy is not None:
            local = self.cache.note_query_position(query_xy[0], query_xy[1])
            if local is not None and local not in candidate_pages:
                candidate_pages.append(local)
            self.cache.maybe_prefetch()

        results: list[Candidate] = []
        for page_id in candidate_pages:
            page = self.cache.get_page(page_id)
            scores, local_idx = page.index.search(q, top_k)
            for s, li in zip(scores[0], local_idx[0]):
                if li < 0:
                    continue
                results.append(
                    Candidate(
                        tile_id=int(page.tile_ids[int(li)]),
                        score=float(s),
                        page_id=page_id,
                    )
                )

        results.sort(key=lambda c: c.score, reverse=True)
        # Deduplicate by tile_id, keep best score.
        seen: set[int] = set()
        deduped: list[Candidate] = []
        for c in results:
            if c.tile_id in seen:
                continue
            seen.add(c.tile_id)
            deduped.append(c)
            if len(deduped) >= top_k:
                break
        return deduped

    def search_multirotation(
        self,
        q_rot: np.ndarray,
        top_pages: int = 5,
        top_k: int = 10,
        query_xy: Optional[tuple[float, float]] = None,
    ) -> list[Candidate]:
        """
        Multi-rotation query-time search.
        q_rot: [R, D] descriptors (one per rotation).
        For each rotation, runs the full hierarchical search; for the same
        DB tile, the BEST score across rotations is kept.
        """
        merged: dict[int, Candidate] = {}
        for r in range(q_rot.shape[0]):
            cands = self.search(q_rot[r:r + 1], top_pages=top_pages, top_k=top_k, query_xy=query_xy)
            for c in cands:
                prev = merged.get(c.tile_id)
                if (prev is None) or (c.score > prev.score):
                    merged[c.tile_id] = c
        out = sorted(merged.values(), key=lambda c: c.score, reverse=True)[:top_k]
        return out
