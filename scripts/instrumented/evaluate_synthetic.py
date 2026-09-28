#!/usr/bin/env python3
"""Instrumented variant of evaluate_synthetic.py for edge profiling.

Differs from the workstation script only in measurement: image decoding is
inside the timed region, per-stage timings are written per query, and the
page cache emits a load event on every demand load. Retrieval behaviour and
outputs are otherwise identical."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uavloc.descriptors import DescriptorExtractor, multirotation_query  # noqa: E402
from uavloc.latency import LatencyRecorder  # noqa: E402
from uavloc.metrics import evaluate_retrieval  # noqa: E402
from uavloc.retrieval import HierarchicalRetriever  # noqa: E402
# PageCacheWithEvents replaces PageCache here: it records load_ms per page and
# per query index, which is the source of the page_load_ms latency component.
from uavloc.page_cache_events import PageCacheWithEvents as PageCache  # noqa: E402


def load_tile_centres(sqlite_path: Path) -> dict[int, tuple[float, float]]:
    con = sqlite3.connect(str(sqlite_path))
    out: dict[int, tuple[float, float]] = {}
    for tid, cx, cy in con.execute("SELECT tile_id, center_x, center_y FROM tiles"):
        out[int(tid)] = (float(cx), float(cy))
    con.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries-dir", required=True)
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

    queries_csv = Path(args.queries_dir) / "synthetic_queries.csv"
    queries = pd.read_csv(queries_csv)
    centres = load_tile_centres(Path(args.index_dir) / "tiles.sqlite")

    # A plain `args.device == "cuda"` test silently falls back to CPU for valid
    # values such as "cuda:0", which would corrupt a profiling run.
    if args.device != "cuda" and args.device.startswith("cuda"):
        print(f"VAROVANIE: --device {args.device!r} nie je presne 'cuda' - "
              f"the run will fall back to CPU. Pass --device cuda.", file=sys.stderr)
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

    for qi, (_, q) in enumerate(tqdm(queries.iterrows(), total=len(queries), desc=f"eval {args.strategy}")):
        # Bind the query index to the cache before search(), so that the
        # demand_load and eviction events raised by this query carry it.
        cache.set_current_query(query_key=str(q["frame_path"]), query_index=qi)
        with rec.measure("decode"):
            img = Image.open(Path(args.queries_dir) / q["frame_path"]).convert("RGB")
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
                desc = multirotation_query(pv, extractor, angles_deg=angles)  # [1,R,D]
                desc_np = desc.detach().float().cpu().numpy()[0]  # [R,D]
            else:
                desc = extractor.extract(pv)  # [1,D]
                desc_np = desc.detach().float().cpu().numpy()  # [1,D]

        gx, gy = float(q["gt_x"]), float(q["gt_y"])
        with rec.measure("search"):
            if args.strategy == "multirot":
                cands = retriever.search_multirotation(
                    desc_np, top_pages=args.top_pages, top_k=args.top_k
                )
            else:
                cands = retriever.search(
                    desc_np, top_pages=args.top_pages, top_k=args.top_k
                )

        gt_xy.append((gx, gy))
        cand_xy.append([centres[c.tile_id] for c in cands if c.tile_id in centres])
        per_query_rows.append({
            "frame_path": q["frame_path"],
            "gt_x": gx,
            "gt_y": gy,
            "top1_tile_id": cands[0].tile_id if cands else -1,
            "top1_score": cands[0].score if cands else float("nan"),
            "top1_page_id": cands[0].page_id if cands else "",
        })

    result = evaluate_retrieval(gt_xy, cand_xy, k_values=(1, 5, 10), tolerance_m=args.tolerance_m)

    # Per-query timing breakdown. LatencyRecorder.samples_ms is filled in loop
    # order, so index i of rec.samples_ms[stage] corresponds to
    # per_query_rows[i]. Nothing extra is measured here; the values already
    # collected are attributed back to individual queries.
    _stage_to_column = {
        "decode": "decode_ms",
        "preprocess": "preprocess_ms",
        "inference": "inference_ms",
        "search": "search_ms",
    }
    for stage, column in _stage_to_column.items():
        values = rec.samples_ms.get(stage, [])
        if len(values) == len(per_query_rows):
            for row, v in zip(per_query_rows, values):
                row[column] = round(float(v), 4)
    for row in per_query_rows:
        row["total_ms"] = round(sum(
            row.get(c, 0.0) for c in _stage_to_column.values()
        ), 4)

    # Cache event log: demand_load, cache_hit and eviction records with load_ms
    # and query index. This is the source of the page_load_ms component.
    cache.events.save(out_dir / "cache_events.csv")

    # Persist outputs.
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
            "top_pages": args.top_pages,
            "top_k": args.top_k,
            "prefetch_neighbours": bool(args.prefetch_neighbours),
        }, indent=2),
        encoding="utf-8",
    )
    with open(out_dir / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()) if per_query_rows else [])
        w.writeheader()
        for row in per_query_rows:
            w.writerow(row)

    print(json.dumps({
        "strategy": args.strategy,
        "median_error_m": result.median_error_m,
        "mean_error_m": result.mean_error_m,
    }, indent=2))


if __name__ == "__main__":
    main()
