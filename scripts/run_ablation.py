#!/usr/bin/env python3
"""Run the descriptor ablation end to end.

For each strategy in --strategies (cls, mean_patch, radial) this builds a
separate database, calling extract_descriptors.py and build_index.py as
subprocesses, and then evaluates it with evaluate_synthetic.py. The
'multirot' strategy is query-time only and reuses the database of
--multirot-base. Results for all strategies are collected under --work-root."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Subprocess failed with code {res.returncode}: {cmd}")


def build_db(
    raster: Path,
    catalog: Path,
    out_emb: Path,
    out_idx: Path,
    strategy: str,
    model: str,
    input_size: int,
    radial_bins: int,
    device: str,
    fp16: bool,
    hf_token: str | None = None,
) -> None:
    out_emb.mkdir(parents=True, exist_ok=True)
    out_idx.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_SCRIPTS / "extract_descriptors.py"),
        "--raster", str(raster),
        "--catalog-csv", str(catalog),
        "--out-dir", str(out_emb),
        "--model-name", model,
        "--input-size", str(input_size),
        "--radial-bins", str(radial_bins),
        "--strategy", strategy,
        "--device", device,
    ]
    if fp16:
        cmd.append("--fp16")
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    _run(cmd)
    _run([
        sys.executable, str(_SCRIPTS / "build_index.py"),
        "--catalog-csv", str(catalog),
        "--embeddings", str(out_emb / "embeddings.npy"),
        "--tile-ids", str(out_emb / "tile_ids.npy"),
        "--out-dir", str(out_idx),
    ])


def eval_strategy(
    queries_dir: Path,
    index_dir: Path,
    out_dir: Path,
    strategy: str,
    multirot_base: str,
    multirot_angles: str,
    model: str,
    input_size: int,
    radial_bins: int,
    device: str,
    fp16: bool,
    top_pages: int,
    top_k: int,
    tolerance_m: float,
    max_pages_in_ram: int,
    hf_token: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_SCRIPTS / "evaluate_synthetic.py"),
        "--queries-dir", str(queries_dir),
        "--index-dir", str(index_dir),
        "--out-dir", str(out_dir),
        "--model-name", model,
        "--input-size", str(input_size),
        "--radial-bins", str(radial_bins),
        "--strategy", strategy,
        "--multirot-base", multirot_base,
        "--multirot-angles", multirot_angles,
        "--top-pages", str(top_pages),
        "--top-k", str(top_k),
        "--tolerance-m", str(tolerance_m),
        "--max-pages-in-ram", str(max_pages_in_ram),
        "--device", device,
    ]
    if fp16:
        cmd.append("--fp16")
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    _run(cmd)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return summary



def _recall_at(summary: dict, k: int):
    """Read recall@k from a summary dict, tolerating string or int JSON keys."""
    recall = summary.get("recall_at_k") or {}
    value = recall.get(str(k))
    if value is None:
        value = recall.get(k)
    return value


def _fmt(value) -> str:
    """Format a recall value for the Markdown table; '-' when unavailable."""
    return "-" if value is None else f"{float(value):.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raster", required=True)
    p.add_argument("--catalog-csv", required=True)
    p.add_argument("--queries-dir", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--strategies", default="cls,radial,multirot")
    p.add_argument("--multirot-base", default="radial", choices=("cls", "mean_patch", "radial"))
    p.add_argument("--multirot-angles", default="0,90,180,270")
    p.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--radial-bins", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--top-pages", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tolerance-m", type=float, default=30.0)
    p.add_argument("--max-pages-in-ram", type=int, default=12)
    p.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token forwarded to subprocess calls of "
            "scripts 03 and 07. Falls back to HF_TOKEN env var if omitted."
        ),
    )
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    raster = Path(args.raster)
    catalog = Path(args.catalog_csv)

    # Build DBs for the unique 'storage' strategies (cls, mean_patch, radial).
    base_strategies = []
    for s in strategies:
        if s == "multirot":
            base_strategies.append(args.multirot_base)
        elif s in ("cls", "mean_patch", "radial"):
            base_strategies.append(s)
        else:
            raise SystemExit(f"Unknown strategy: {s}")
    unique_bases = sorted(set(base_strategies))

    db_paths: dict[str, tuple[Path, Path]] = {}
    for s in unique_bases:
        emb_dir = work_root / f"db_{s}" / "embeddings"
        idx_dir = work_root / f"db_{s}" / "index"
        if not (idx_dir / "tiles.sqlite").exists():
            print(f"[ablation] Building DB for strategy={s}")
            build_db(
                raster=raster,
                catalog=catalog,
                out_emb=emb_dir,
                out_idx=idx_dir,
                strategy=s,
                model=args.model_name,
                input_size=args.input_size,
                radial_bins=args.radial_bins,
                device=args.device,
                fp16=args.fp16,
                hf_token=args.hf_token,
            )
        else:
            print(f"[ablation] Re-using existing DB for strategy={s}")
        db_paths[s] = (emb_dir, idx_dir)

    # Run evaluations.
    summary_rows = []
    for s in strategies:
        base = args.multirot_base if s == "multirot" else s
        _, idx_dir = db_paths[base]
        out_dir = work_root / f"eval_{s}"
        if s == "multirot":
            print(f"[ablation] eval multirot (DB shared with base={base})")
        else:
            print(f"[ablation] eval {s}")
        summary = eval_strategy(
            queries_dir=Path(args.queries_dir),
            index_dir=idx_dir,
            out_dir=out_dir,
            strategy=s,
            multirot_base=args.multirot_base,
            multirot_angles=args.multirot_angles,
            model=args.model_name,
            input_size=args.input_size,
            radial_bins=args.radial_bins,
            device=args.device,
            fp16=args.fp16,
            top_pages=args.top_pages,
            top_k=args.top_k,
            tolerance_m=args.tolerance_m,
            max_pages_in_ram=args.max_pages_in_ram,
            hf_token=args.hf_token,
        )
        summary_rows.append({
            "strategy": s,
            "base_db": base,
            "n": summary.get("n"),
            "median_error_m": summary.get("median_error_m"),
            "mean_error_m": summary.get("mean_error_m"),
            "recall_at_1": _recall_at(summary, 1),
            "recall_at_5": _recall_at(summary, 5),
            "recall_at_10": _recall_at(summary, 10),
        })

    # Write the aggregate summary as CSV, plus a Markdown rendering of the
    # same rows for quick inspection.
    csv_path = work_root / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    md_path = work_root / "ablation_summary.md"
    lines = [
        "# Descriptor ablation summary",
        "",
        "| strategy | base_db | n | median_err_m | mean_err_m | R@1 | R@5 | R@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['strategy']} | {r['base_db']} | {r['n']} | {r['median_error_m']:.2f} | "
            f"{r['mean_error_m']:.2f} | {_fmt(r['recall_at_1'])} | {_fmt(r['recall_at_5'])} | "
            f"{_fmt(r['recall_at_10'])} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
