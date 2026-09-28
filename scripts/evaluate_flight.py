#!/usr/bin/env python3
"""Evaluate the index against real UAV flight frames.

Ground-truth positions come from --gt-csv with columns
filename,lat,lon,alt_m,yaw_deg, or from EXIF GPS as a fallback. Coordinates
are transformed from WGS-84 to EPSG:5514 before evaluation. Writes
summary.json and per_query.csv. Query ground-truth positions are never passed
to the retriever."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uavloc.descriptors import DescriptorExtractor, multirotation_query  # noqa: E402
from uavloc.exif_utils import read_gps_from_image  # noqa: E402
from uavloc.geo import latlon_to_epsg5514  # noqa: E402
from uavloc.latency import LatencyRecorder  # noqa: E402
from uavloc.metrics import evaluate_retrieval  # noqa: E402
from uavloc.retrieval import HierarchicalRetriever  # noqa: E402
from uavloc.page_cache import PageCache  # noqa: E402


def load_tile_centres(sqlite_path: Path) -> dict[int, tuple[float, float]]:
    con = sqlite3.connect(str(sqlite_path))
    out: dict[int, tuple[float, float]] = {}
    for tid, cx, cy in con.execute("SELECT tile_id, center_x, center_y FROM tiles"):
        out[int(tid)] = (float(cx), float(cy))
    con.close()
    return out


def load_gt(gt_csv: Path | None, images_dir: Path) -> list[dict]:
    """Return list of {filename, x, y} in EPSG:5514."""
    if gt_csv is not None and gt_csv.exists():
        import pandas as pd
        df = pd.read_csv(gt_csv)
        out = []
        for _, r in df.iterrows():
            x, y = latlon_to_epsg5514(float(r["lat"]), float(r["lon"]))
            out.append({"filename": str(r["filename"]), "x": x, "y": y})
        return out
    out = []
    for p in sorted(images_dir.glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        gps = read_gps_from_image(p)
        if not gps:
            continue
        lat, lon, _ = gps
        x, y = latlon_to_epsg5514(lat, lon)
        out.append({"filename": p.name, "x": x, "y": y})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drone-images-dir", required=True)
    p.add_argument("--gt-csv", default=None)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--radial-bins", type=int, default=8)
    p.add_argument("--strategy", choices=("cls", "mean_patch", "radial", "multirot"), default="radial")
    p.add_argument("--multirot-base", choices=("cls", "mean_patch", "radial"), default="radial")
    p.add_argument("--multirot-angles", default="0,90,180,270")
    p.add_argument("--top-pages", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tolerance-m", type=float, default=30.0)
    p.add_argument("--max-pages-in-ram", type=int, default=12)
    p.add_argument("--prefetch-neighbours", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token for gated DINOv3 models. "
            "If omitted, falls back to the HF_TOKEN environment variable "
            "or to ~/.cache/huggingface/token written by `huggingface-cli login`."
        ),
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.drone_images_dir)

    gt = load_gt(Path(args.gt_csv) if args.gt_csv else None, images_dir)
    if not gt:
        raise SystemExit("No ground-truth positions resolved. Provide --gt-csv or EXIF GPS.")
    centres = load_tile_centres(Path(args.index_dir) / "tiles.sqlite")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(args.model_name, token=args.hf_token)
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=dtype, token=args.hf_token).to(device).eval()
    base_strategy = args.multirot_base if args.strategy == "multirot" else args.strategy
    extractor = DescriptorExtractor(model=model, strategy=base_strategy, radial_bins=args.radial_bins)

    cache = PageCache(
        index_dir=Path(args.index_dir),
        max_pages_in_ram=args.max_pages_in_ram,
        prefetch_neighbours=args.prefetch_neighbours,
    )
    retriever = HierarchicalRetriever(index_dir=Path(args.index_dir), cache=cache)
    angles = [int(a) for a in args.multirot_angles.split(",")] if args.strategy == "multirot" else None

    rec = LatencyRecorder()
    gt_xy: list[tuple[float, float]] = []
    cand_xy: list[list[tuple[float, float]]] = []
    per_query_rows: list[dict] = []

    for g in tqdm(gt, desc=f"real-eval {args.strategy}"):
        img = Image.open(images_dir / g["filename"]).convert("RGB")
        with rec.measure("preprocess"):
            inputs = processor(
                images=[img],
                return_tensors="pt",
                do_resize=True,
                size={"height": args.input_size, "width": args.input_size},
            )
            pv = inputs["pixel_values"].to(device)
            if dtype == torch.float16:
                pv = pv.to(dtype)

        with rec.measure("inference"), torch.inference_mode():
            if args.strategy == "multirot":
                desc = multirotation_query(pv, extractor, angles_deg=angles)
                desc_np = desc.detach().float().cpu().numpy()[0]
            else:
                desc = extractor.extract(pv)
                desc_np = desc.detach().float().cpu().numpy()

        with rec.measure("search"):
            if args.strategy == "multirot":
                cands = retriever.search_multirotation(
                    desc_np, top_pages=args.top_pages, top_k=args.top_k
                )
            else:
                cands = retriever.search(
                    desc_np, top_pages=args.top_pages, top_k=args.top_k
                )

        gt_xy.append((g["x"], g["y"]))
        cand_xy.append([centres[c.tile_id] for c in cands if c.tile_id in centres])
        per_query_rows.append({
            "filename": g["filename"],
            "gt_x": g["x"],
            "gt_y": g["y"],
            "top1_tile_id": cands[0].tile_id if cands else -1,
            "top1_score": cands[0].score if cands else float("nan"),
            "top1_page_id": cands[0].page_id if cands else "",
        })

    result = evaluate_retrieval(gt_xy, cand_xy, k_values=(1, 5, 10), tolerance_m=args.tolerance_m)
    (out_dir / "summary.json").write_text(
        json.dumps({
            "strategy": args.strategy,
            "n": result.n,
            "median_error_m": result.median_error_m,
            "mean_error_m": result.mean_error_m,
            "p90_error_m": result.p90_error_m,
            "recall_at_k": result.recall_at_k,
            "latency_ms": rec.summary(),
            "tolerance_m": args.tolerance_m,
            "prefetch_neighbours": bool(args.prefetch_neighbours),
        }, indent=2),
        encoding="utf-8",
    )
    with open(out_dir / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()) if per_query_rows else [])
        w.writeheader()
        for row in per_query_rows:
            w.writerow(row)
    print(json.dumps({"median_error_m": result.median_error_m}, indent=2))


if __name__ == "__main__":
    main()
