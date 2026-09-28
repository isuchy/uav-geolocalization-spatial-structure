#!/usr/bin/env python3
"""Generate the synthetic query set from the orthomosaic.

Each query is a 512x512 PNG with controlled rotation, scale and brightness
perturbation relative to a clean window of the map. Centres are drawn
uniformly from the area of interest shrunk by --aoi-margin-m, and each query
is rendered from an oversampled window and centre-cropped, so no out-of-frame
region can enter. Writes synthetic_queries.csv with the ground-truth (x, y) in
EPSG:5514 together with the per-query rotation, scale and brightness."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uavloc.synthetic import SyntheticConfig, generate_synthetic_queries  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raster", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-queries", type=int, default=200)
    p.add_argument("--aoi-margin-m", type=float, default=200.0)
    p.add_argument("--out-size", type=int, default=512)
    p.add_argument("--db-tile-m", type=float, default=153.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--aoi-bounds",
        default=None,
        help="Optional 'minx,miny,maxx,maxy' to constrain query positions.",
    )
    args = p.parse_args()

    aoi = None
    if args.aoi_bounds:
        try:
            aoi = tuple(float(x) for x in args.aoi_bounds.split(","))
            if len(aoi) != 4:
                raise ValueError
        except Exception:
            raise SystemExit("--aoi-bounds must be 'minx,miny,maxx,maxy'")

    cfg = SyntheticConfig(
        num_queries=args.num_queries,
        aoi_margin_m=args.aoi_margin_m,
        out_size=args.out_size,
        db_tile_m=args.db_tile_m,
        seed=args.seed,
    )
    manifest = generate_synthetic_queries(
        raster_path=Path(args.raster),
        out_dir=Path(args.out_dir),
        config=cfg,
        aoi_bounds=aoi,
    )
    csv_path = Path(args.out_dir) / "synthetic_queries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "gt_x", "gt_y", "rotation_deg", "scale", "brightness"])
        for q in manifest.queries:
            w.writerow([q["frame_path"], q["gt_x"], q["gt_y"], q["rotation_deg"], q["scale"], q["brightness"]])
    print(f"Generated {len(manifest.queries)} synthetic queries -> {csv_path}")


if __name__ == "__main__":
    main()
