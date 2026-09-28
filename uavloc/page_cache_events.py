"""Instrumented page cache that emits an auditable event log.

Subclasses PageCache and records one event per page access: demand_load,
cache_hit, prefetch and eviction, each with load time, the query index that
caused it, and the resident page and byte counts afterwards. Hits, misses and
evictions are derived from the difference in cache keys before and after the
delegated load, so the loading logic itself is not duplicated.

Peak resident pages and bytes are tracked continuously. prefetch_pages() is
the explicit interface for predicted-state prefetch; maybe_prefetch() remains
and delegates to it.

The "used_by_future_query" flag is an approximation: it records reuse before
eviction, not true future use.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from uavloc.page_cache import PageCache, PageEntry


class CacheEventRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._seq = 0

    def add(self, event_type: str, **kwargs) -> dict:
        self._seq += 1
        row = {
            "event_seq": self._seq,
            "timestamp_ns": time.time_ns(),
            "event_type": event_type,
            **kwargs,
        }
        self.rows.append(row)
        return row

    def save(self, path: Path) -> None:
        df = pd.DataFrame(self.rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


class PageCacheWithEvents(PageCache):
    """PageCache + event logging. Base LRU semantics untouched (super() calls)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events = CacheEventRecorder()
        self._current_query_key = None
        self._current_query_index = None
        self._pending_prefetches: dict[str, dict] = {}
        self._page_bytes_cache: dict[str, float] = {}
        self.peak_resident_pages = 0
        self.peak_resident_bytes = 0.0

    def set_current_query(self, *, query_key: str, query_index: int) -> None:
        self._current_query_key = query_key
        self._current_query_index = query_index

    def _page_bytes(self, page_id: str) -> float:
        if page_id in self._page_bytes_cache:
            return self._page_bytes_cache[page_id]
        meta = self._page_meta.get(page_id)
        if meta is None:
            return float("nan")
        try:
            b = float(
                (self.index_dir / meta["faiss_path"]).stat().st_size
                + (self.index_dir / meta["tile_ids_path"]).stat().st_size
            )
        except OSError:
            b = float("nan")
        self._page_bytes_cache[page_id] = b
        return b

    def _resident_bytes(self) -> float:
        return float(sum(self._page_bytes(pid) for pid in self._lru.keys()))

    def _snapshot(self) -> dict:
        pages = len(self._lru)
        rbytes = self._resident_bytes()
        self.peak_resident_pages = max(self.peak_resident_pages, pages)
        self.peak_resident_bytes = max(self.peak_resident_bytes, rbytes)
        return {"resident_pages_after": pages, "resident_bytes_after": rbytes}

    def _log_evictions(self, evicted: set) -> None:
        for evicted_id in evicted:
            pending = self._pending_prefetches.pop(evicted_id, None)
            if pending is not None:
                pending["used_by_future_query"] = False
                pending["evicted_before_use"] = True
                pending["eviction_query_index"] = self._current_query_index
            self.events.add(
                "eviction",
                query_index=self._current_query_index,
                query_key=self._current_query_key,
                source_page_id=None,
                target_page_id=evicted_id,
                page_bytes=self._page_bytes(evicted_id),
                load_ms=None,
                used_by_future_query=None,
                **self._snapshot(),
            )

    def _load_page(self, page_id: str) -> PageEntry:
        was_hit = page_id in self._lru
        before_keys = set(self._lru.keys())
        started = time.perf_counter()
        entry = super()._load_page(page_id)
        load_ms = (time.perf_counter() - started) * 1000.0
        evicted = before_keys - set(self._lru.keys())

        if was_hit:
            pending = self._pending_prefetches.pop(page_id, None)
            if pending is not None:
                if (
                    self._current_query_index is not None
                    and pending.get("prefetch_query_index") is not None
                    and self._current_query_index > pending["prefetch_query_index"]
                ):
                    pending["used_by_future_query"] = True
                    pending["use_query_index"] = self._current_query_index
                else:
                    pending["used_by_future_query"] = False
                    pending["use_query_index"] = self._current_query_index
            self.events.add(
                "cache_hit",
                query_index=self._current_query_index,
                query_key=self._current_query_key,
                source_page_id=self._last_page,
                target_page_id=page_id,
                page_bytes=self._page_bytes(page_id),
                load_ms=0.0,
                used_by_future_query=None,
                **self._snapshot(),
            )
        else:
            self.events.add(
                "demand_load",
                query_index=self._current_query_index,
                query_key=self._current_query_key,
                source_page_id=self._last_page,
                target_page_id=page_id,
                page_bytes=self._page_bytes(page_id),
                load_ms=load_ms,
                used_by_future_query=None,
                **self._snapshot(),
            )
        self._log_evictions(evicted)
        return entry

    def prefetch_pages(self, page_ids) -> list:
        """Prefetch an explicit, ordered list of page_ids. Unknown or already
        resident pages are skipped. Returns pages actually loaded."""
        touched = []
        for pid in page_ids:
            if pid is None or pid not in self._page_meta or pid in self._lru:
                continue
            self.events.add(
                "prefetch_start",
                query_index=self._current_query_index,
                query_key=self._current_query_key,
                source_page_id=self._last_page,
                target_page_id=pid,
                page_bytes=None,
                load_ms=None,
                used_by_future_query=None,
                prefetch_query_index=self._current_query_index,
                **self._snapshot(),
            )
            before_keys = set(self._lru.keys())
            started = time.perf_counter()
            super()._load_page(pid)
            load_ms = (time.perf_counter() - started) * 1000.0
            evicted = before_keys - set(self._lru.keys())
            row = self.events.add(
                "prefetch_complete",
                query_index=self._current_query_index,
                query_key=self._current_query_key,
                source_page_id=self._last_page,
                target_page_id=pid,
                page_bytes=self._page_bytes(pid),
                load_ms=load_ms,
                used_by_future_query=False,
                prefetch_query_index=self._current_query_index,
                use_query_index=None,
                eviction_query_index=None,
                evicted_before_use=False,
                **self._snapshot(),
            )
            self._pending_prefetches[pid] = row
            self._log_evictions(evicted)
            touched.append(pid)
        return touched

    def maybe_prefetch(self, motion=None) -> list:
        if not self.prefetch_neighbours or self._last_page is None:
            return []
        return self.prefetch_pages(self._neighbour_ids(self._last_page))
