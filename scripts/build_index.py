#!/usr/bin/env python3
"""Build the exact retrieval index.

Produces one coarse FAISS index over per-page descriptors, one fine FAISS
index per page, and an SQLite database with tile and page tables and R-tree
indexes over their bounding boxes. Page geometry is taken from the catalogue,
so the page size is fixed when the catalogue is built."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype("float32")


def build_index_ip(vectors: np.ndarray) -> faiss.Index:
    """Inner-product FAISS index on L2-normalized vectors == cosine similarity."""
    vectors = np.ascontiguousarray(vectors.astype("float32"))
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def create_sqlite(db_path: Path, catalog: pd.DataFrame, page_records: list[dict]) -> None:
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("""
        CREATE TABLE tiles (
            tile_id INTEGER PRIMARY KEY,
            page_id TEXT NOT NULL,
            page_int_id INTEGER NOT NULL,
            config_name TEXT,
            tile_px INTEGER,
            stride_px INTEGER,
            gsd_m REAL,
            tile_m REAL,
            stride_m REAL,
            center_x REAL,
            center_y REAL,
            window_col_off INTEGER,
            window_row_off INTEGER,
            window_width INTEGER,
            window_height INTEGER,
            minx REAL,
            miny REAL,
            maxx REAL,
            maxy REAL
        );
    """)
    cur.execute("CREATE VIRTUAL TABLE tile_rtree USING rtree(tile_id, minx, maxx, miny, maxy);")

    cur.execute("""
        CREATE TABLE pages (
            page_int_id INTEGER PRIMARY KEY,
            page_id TEXT UNIQUE NOT NULL,
            page_row INTEGER,
            page_col INTEGER,
            minx REAL,
            miny REAL,
            maxx REAL,
            maxy REAL,
            faiss_path TEXT,
            tile_ids_path TEXT,
            tile_count INTEGER
        );
    """)
    cur.execute("CREATE VIRTUAL TABLE page_rtree USING rtree(page_int_id, minx, maxx, miny, maxy);")

    page_lookup = {rec["page_id"]: int(rec["page_int_id"]) for rec in page_records}

    tile_rows = []
    rtree_rows = []
    for _, r in catalog.iterrows():
        page_int_id = page_lookup[str(r["page_id"])]
        tile_rows.append((
            int(r["tile_id"]), str(r["page_id"]), page_int_id, str(r.get("config_name", "")),
            int(r["tile_px"]), int(r["stride_px"]), float(r["gsd_m"]),
            float(r["tile_m"]), float(r["stride_m"]),
            float(r["center_x"]), float(r["center_y"]),
            int(r["window_col_off"]), int(r["window_row_off"]),
            int(r["window_width"]), int(r["window_height"]),
            float(r["minx"]), float(r["miny"]), float(r["maxx"]), float(r["maxy"])
        ))
        rtree_rows.append((
            int(r["tile_id"]),
            float(r["minx"]), float(r["maxx"]), float(r["miny"]), float(r["maxy"]),
        ))

    cur.executemany(
        "INSERT INTO tiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",
        tile_rows,
    )
    cur.executemany("INSERT INTO tile_rtree VALUES (?,?,?,?,?);", rtree_rows)

    page_rows = []
    page_rtree_rows = []
    for rec in page_records:
        page_rows.append((
            int(rec["page_int_id"]), rec["page_id"], int(rec["page_row"]), int(rec["page_col"]),
            float(rec["minx"]), float(rec["miny"]), float(rec["maxx"]), float(rec["maxy"]),
            rec["faiss_path"], rec["tile_ids_path"], int(rec["tile_count"])
        ))
        page_rtree_rows.append((
            int(rec["page_int_id"]),
            float(rec["minx"]), float(rec["maxx"]), float(rec["miny"]), float(rec["maxy"]),
        ))

    cur.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?,?,?);", page_rows)
    cur.executemany("INSERT INTO page_rtree VALUES (?,?,?,?,?);", page_rtree_rows)

    cur.execute("CREATE INDEX idx_tiles_page ON tiles(page_id);")
    cur.execute("CREATE INDEX idx_tiles_center ON tiles(center_x, center_y);")
    # New: explicit index on pages.page_id; UNIQUE constraint creates one too,
    # but having an explicit index documents the access pattern used by the cache.
    cur.execute("CREATE INDEX idx_pages_page_id ON pages(page_id);")

    con.commit()
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--tile-ids", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog_csv)
    embeddings = np.load(args.embeddings).astype("float32")
    tile_ids = np.load(args.tile_ids).astype(np.int64)

    # Derive config_name from the catalog. Script 02 persists it on every row.
    config_name = (
        str(catalog["config_name"].iloc[0])
        if "config_name" in catalog.columns and not catalog.empty
        else None
    )

    if len(tile_ids) != embeddings.shape[0]:
        raise ValueError("tile_ids and embeddings length mismatch.")

    emb_df = pd.DataFrame({"tile_id": tile_ids, "emb_pos": np.arange(len(tile_ids), dtype=np.int64)})
    catalog = catalog.merge(emb_df, on="tile_id", how="inner")
    if len(catalog) != len(tile_ids):
        raise ValueError("Catalog and embedding IDs do not fully match.")

    embeddings = normalize(embeddings)

    # Read paging-grid origin from the catalog (persisted by script 02).
    if not {"raster_left", "raster_top", "page_size_m"}.issubset(catalog.columns):
        raise ValueError(
            "Catalog is missing raster_left/raster_top/page_size_m columns. "
            "Rerun 02_build_tile_catalog_from_qgis_csv.py from this version."
        )
    raster_left = float(catalog["raster_left"].iloc[0])
    raster_top = float(catalog["raster_top"].iloc[0])
    page_size_m = float(catalog["page_size_m"].iloc[0])

    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_records = []
    coarse_vectors = []
    coarse_page_ids = []

    for page_int_id, (page_id, group) in enumerate(
        sorted(catalog.groupby("page_id"), key=lambda kv: kv[0])
    ):
        positions = group["emb_pos"].to_numpy(dtype=np.int64)
        page_vectors = embeddings[positions]
        page_tile_ids = group["tile_id"].to_numpy(dtype=np.int64)

        page_index = build_index_ip(page_vectors)
        faiss_rel = f"pages/{page_id}.faiss"
        ids_rel = f"pages/{page_id}_tile_ids.npy"
        faiss.write_index(page_index, str(out_dir / faiss_rel))
        np.save(out_dir / ids_rel, page_tile_ids)

        # Coarse page descriptor: mean of L2-normalized tile descriptors,
        # re-normalized. This is the same simple choice the paper text says.
        page_desc = normalize(page_vectors.mean(axis=0, keepdims=True))[0]
        coarse_vectors.append(page_desc)
        coarse_page_ids.append(page_id)

        # Exact geometric page bbox in EPSG:5514, derived from raster origin
        # and page_size_m. NOT from tile footprint union.
        page_row_i = int(group["page_row"].iloc[0])
        page_col_i = int(group["page_col"].iloc[0])
        page_minx = raster_left + page_col_i * page_size_m
        page_maxx = page_minx + page_size_m
        page_maxy = raster_top - page_row_i * page_size_m
        page_miny = page_maxy - page_size_m

        page_records.append({
            "page_int_id": page_int_id,
            "page_id": page_id,
            "page_row": page_row_i,
            "page_col": page_col_i,
            "minx": page_minx,
            "miny": page_miny,
            "maxx": page_maxx,
            "maxy": page_maxy,
            "faiss_path": faiss_rel,
            "tile_ids_path": ids_rel,
            "tile_count": int(len(group)),
        })

    coarse_vectors_np = normalize(np.vstack(coarse_vectors).astype("float32"))
    coarse_index = build_index_ip(coarse_vectors_np)
    faiss.write_index(coarse_index, str(out_dir / "coarse_pages.faiss"))
    (out_dir / "coarse_page_ids.json").write_text(
        json.dumps(coarse_page_ids, indent=2), encoding="utf-8"
    )

    create_sqlite(out_dir / "tiles.sqlite", catalog, page_records)

    manifest = {
        "index_type": "hierarchical page index",
        "config_name": config_name,
        "metric": "inner_product_on_l2_normalized_vectors",
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_count": int(embeddings.shape[0]),
        "tile_count": int(len(catalog)),
        "page_count": int(len(page_records)),
        "page_size_m": page_size_m,
        "raster_left": raster_left,
        "raster_top": raster_top,
        "coarse_descriptor": "mean_of_tile_descriptors_then_l2",
        "page_id_convention": "pRRRR_CCCC",
        "catalog_csv": str(args.catalog_csv),
        "embeddings_path": str(args.embeddings),
        "tile_ids_path": str(args.tile_ids),
        "files": {
            "sqlite": "tiles.sqlite",
            "coarse_faiss": "coarse_pages.faiss",
            "coarse_page_ids": "coarse_page_ids.json",
            "page_indexes_dir": "pages/",
        },
    }
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
