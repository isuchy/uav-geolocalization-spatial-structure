"""Page-level least-recently-used cache with optional neighbour prefetch.

Pages are loaded on demand and evicted least-recently-used first. If a query
position is supplied, the containing page is resolved from the page bounding
boxes and its eight-connected neighbours may be prefetched; prefetching is
disabled in all evaluations reported in the paper.

Neighbour generation assumes the 'pRRRR_CCCC' page identifier convention
written by build_index.py. Under any other convention the cache refuses to
prefetch rather than behaving incorrectly.
"""
from __future__ import annotations

import re
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np


_PAGE_ID_RE = re.compile(r"^p(\d{4})_(\d{4})$")


class PageEntry:
    __slots__ = ("page_id", "index", "tile_ids")

    def __init__(self, page_id: str, index: faiss.Index, tile_ids: np.ndarray):
        self.page_id = page_id
        self.index = index
        self.tile_ids = tile_ids


class PageCache:
    """
    LRU page cache backed by SQLite metadata. Loads page-level FAISS index
    files and tile_id arrays on demand. Supports optional predictive
    prefetch of immediate neighbours (up to 8 surrounding pages).
    """

    def __init__(
        self,
        index_dir: Path,
        max_pages_in_ram: int = 12,
        prefetch_neighbours: bool = False,
    ):
        self.index_dir = Path(index_dir)
        self.max_pages_in_ram = int(max_pages_in_ram)
        self.prefetch_neighbours = bool(prefetch_neighbours)
        self._lru: "OrderedDict[str, PageEntry]" = OrderedDict()

        sqlite_path = self.index_dir / "tiles.sqlite"
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing SQLite at {sqlite_path}")
        self._con = sqlite3.connect(str(sqlite_path))
        # Cache of page metadata: page_id -> (faiss_path, tile_ids_path, bbox)
        self._page_meta: dict[str, dict] = {}
        for row in self._con.execute(
            "SELECT page_id, faiss_path, tile_ids_path, minx, miny, maxx, maxy FROM pages"
        ):
            self._page_meta[row[0]] = {
                "faiss_path": row[1],
                "tile_ids_path": row[2],
                "minx": row[3], "miny": row[4], "maxx": row[5], "maxy": row[6],
            }
        if not self._page_meta:
            raise RuntimeError("pages table is empty.")
        # Sanity-check convention.
        sample = next(iter(self._page_meta))
        if not _PAGE_ID_RE.match(sample):
            raise RuntimeError(
                f"page_id convention mismatch in SQLite: got '{sample}', "
                "expected 'pRRRR_CCCC'. PageCache neighbour generation will not work."
            )

        self._last_page: Optional[str] = None

    # ---- page IO ----

    def _load_page(self, page_id: str) -> PageEntry:
        if page_id in self._lru:
            self._lru.move_to_end(page_id)
            return self._lru[page_id]
        meta = self._page_meta.get(page_id)
        if meta is None:
            raise KeyError(page_id)
        faiss_path = self.index_dir / meta["faiss_path"]
        ids_path = self.index_dir / meta["tile_ids_path"]
        index = faiss.read_index(str(faiss_path))
        tile_ids = np.load(ids_path)
        entry = PageEntry(page_id, index, tile_ids)
        self._lru[page_id] = entry
        while len(self._lru) > self.max_pages_in_ram:
            self._lru.popitem(last=False)
        return entry

    def get_page(self, page_id: str) -> PageEntry:
        return self._load_page(page_id)

    def get_pages(self, page_ids: Iterable[str]) -> list[PageEntry]:
        return [self._load_page(p) for p in page_ids]

    # ---- spatial / page_id helpers ----

    def page_id_for_position(self, x: float, y: float) -> Optional[str]:
        """
        Return the page_id whose bbox contains (x, y), or None if none does.
        Uses SQLite for an exact match; for tens of thousands of pages this is
        still fast (R*Tree available, but a plain SELECT works too).
        """
        cur = self._con.execute(
            "SELECT page_id FROM pages WHERE minx <= ? AND maxx >= ? AND miny <= ? AND maxy >= ? LIMIT 1",
            (x, x, y, y),
        )
        row = cur.fetchone()
        return row[0] if row else None

    @staticmethod
    def _parse_page_id(page_id: str) -> tuple[int, int]:
        m = _PAGE_ID_RE.match(page_id)
        if not m:
            raise ValueError(f"Bad page_id: {page_id!r}")
        return int(m.group(1)), int(m.group(2))

    def _neighbour_ids(self, page_id: str) -> list[str]:
        row, col = self._parse_page_id(page_id)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nid = f"p{row + dr:04d}_{col + dc:04d}"
                if nid in self._page_meta:
                    out.append(nid)
        return out

    # ---- public API used by eval scripts ----

    def note_query_position(
        self,
        x: float,
        y: float,
        page_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record the current query position. If page_id is not supplied it is
        resolved from (x, y) using the page bounding-box table. Returns the
        resolved page_id, or None if the position lies outside every page.
        """
        if page_id is None:
            page_id = self.page_id_for_position(x, y)
        if page_id is None:
            self._last_page = None
            return None
        self._last_page = page_id
        return page_id

    def maybe_prefetch(self, motion: Optional[str] = None) -> list[str]:
        """
        Prefetch immediate neighbours of the last known page.
        `motion` is accepted for forward compatibility but not used today;
        future versions can prioritise neighbours in the direction of motion.

        Returns the list of page_ids that were touched (pulled into LRU).
        """
        if not self.prefetch_neighbours or self._last_page is None:
            return []
        touched = []
        for nid in self._neighbour_ids(self._last_page):
            self._load_page(nid)
            touched.append(nid)
        return touched

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass
