#!/usr/bin/env python3
"""Build the tile catalogue from a QGIS-exported point grid.

Reads tile centres in EPSG:5514 (accepted field names: center_x/center_y,
X/Y, x/y or Easting/Northing), resolves each centre to a raster window of
--tile-px pixels, rejects windows that fall outside the raster, assigns each
tile to a page of --page-size-m metres by the page containing its centre, and
writes one CSV row per tile. The paging-grid origin (raster_left, raster_top)
and page_size_m are persisted in every row, so the index builder needs no
further geometry arguments."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise ValueError(f"Could not find any of these columns: {candidates}. Existing: {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--tile-points-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--tile-px", type=int, required=True)
    parser.add_argument("--stride-px", type=int, required=True)
    parser.add_argument("--gsd", type=float, default=0.15)
    parser.add_argument("--page-size-m", type=float, default=2048.0)
    parser.add_argument("--config-name", required=True)
    args = parser.parse_args()

    raster_path = Path(args.raster)
    points_csv = Path(args.tile_points_csv)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(points_csv)
    x_col = find_column(pts, ["center_x", "x", "X", "Easting", "east", "POINT_X"])
    y_col = find_column(pts, ["center_y", "y", "Y", "Northing", "north", "POINT_Y"])

    if args.tile_px % 2 != 0:
        raise ValueError("tile_px must be even.")
    half_px = args.tile_px // 2
    tile_m = args.tile_px * args.gsd
    stride_m = args.stride_px * args.gsd

    rows = []
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        for idx, rec in pts.iterrows():
            cx = float(rec[x_col])
            cy = float(rec[y_col])
            row, col = src.index(cx, cy)
            row_off = int(row - half_px)
            col_off = int(col - half_px)
            if row_off < 0 or col_off < 0:
                continue
            if row_off + args.tile_px > src.height or col_off + args.tile_px > src.width:
                continue

            win = Window(col_off, row_off, args.tile_px, args.tile_px)
            minx, miny, maxx, maxy = window_bounds(win, src.transform)

            # Paging-grid assignment in EPSG:5514, relative to raster origin.
            # Convention: page_id = f"p{row:04d}_{col:04d}". This MUST match
            # uavloc/page_cache.py neighbour generator and downstream
            # FAISS filenames pages/{page_id}.faiss
            page_col = int(math.floor((cx - left) / args.page_size_m))
            page_row = int(math.floor((top - cy) / args.page_size_m))
            page_id = f"p{page_row:04d}_{page_col:04d}"

            rows.append({
                "tile_id": len(rows),
                "source_point_id": int(rec["tile_id"]) if "tile_id" in pts.columns else int(idx),
                "config_name": args.config_name,
                "tile_px": args.tile_px,
                "stride_px": args.stride_px,
                "gsd_m": args.gsd,
                "tile_m": tile_m,
                "stride_m": stride_m,
                "center_x": cx,
                "center_y": cy,
                "window_col_off": col_off,
                "window_row_off": row_off,
                "window_width": args.tile_px,
                "window_height": args.tile_px,
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy,
                "page_id": page_id,
                "page_row": page_row,
                "page_col": page_col,
                # Paging-grid origin persisted so script 04 can build exact
                # page bboxes (left + page_col * P, top - (page_row+1) * P,
                # left + (page_col+1) * P, top - page_row * P).
                "raster_left": left,
                "raster_top": top,
                "page_size_m": args.page_size_m,
            })

    if not rows:
        raise RuntimeError(
            "No valid tiles were produced. Check that QGIS CSV center coordinates "
            "are in EPSG:5514 and lie inside the raster extent."
        )

    catalog = pd.DataFrame(rows)
    catalog.to_csv(out_csv, index=False)

    print(f"Saved tile catalog: {out_csv}")
    print(f"Valid tiles: {len(catalog)}")
    print(f"Tile footprint: {tile_m:.3f} m x {tile_m:.3f} m")
    print(f"Stride: {stride_m:.3f} m")
    print(f"Pages: {catalog['page_id'].nunique()}")
    print(f"Raster origin (left, top) persisted: ({left:.3f}, {top:.3f})")
    print(f"Page size persisted: {args.page_size_m} m")


if __name__ == "__main__":
    main()
