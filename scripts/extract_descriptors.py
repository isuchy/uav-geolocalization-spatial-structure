#!/usr/bin/env python3
"""Extract DINOv3 descriptors for every tile in the catalogue.

Reads the raster window of each tile directly from the orthomosaic, so tiles
are never written to disk. --strategy selects the aggregation: cls,
mean_patch or radial. Writes embeddings.npy, tile_ids.npy and
embedding_manifest.json."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, AutoModel

# Make the uavloc package importable when running from repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uavloc.descriptors import DescriptorExtractor  # noqa: E402


def read_rgb_window(src: rasterio.DatasetReader, rec: pd.Series) -> Image.Image:
    win = Window(
        int(rec["window_col_off"]),
        int(rec["window_row_off"]),
        int(rec["window_width"]),
        int(rec["window_height"]),
    )
    arr = src.read([1, 2, 3], window=win)  # [C,H,W]
    arr = np.moveaxis(arr, 0, -1)  # [H,W,C]
    if arr.dtype != np.uint8:
        # Ortofoto SR is byte; if anything else arrives we keep going but warn.
        warnings.warn(
            f"Source dtype {arr.dtype} clipped to uint8; expected uint8 ortho.",
            stacklevel=2,
        )
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--catalog-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--radial-bins", type=int, default=8)
    parser.add_argument(
        "--strategy",
        choices=("cls", "mean_patch", "radial"),
        default="radial",
        help="Descriptor strategy. 'radial' is the main configuration.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token for gated DINOv3 models. "
            "If omitted, falls back to the HF_TOKEN environment variable "
            "or to ~/.cache/huggingface/token written by `huggingface-cli login`."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(args.catalog_csv)
    if args.limit:
        catalog = catalog.head(args.limit).copy()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    processor = AutoImageProcessor.from_pretrained(args.model_name, token=args.hf_token)
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=dtype, token=args.hf_token).to(device)
    model.eval()

    extractor = DescriptorExtractor(
        model=model,
        strategy=args.strategy,
        radial_bins=args.radial_bins,
    )

    all_embeddings: list[np.ndarray] = []
    all_tile_ids: list[int] = []

    with rasterio.open(args.raster) as src:
        if src.count < 3:
            raise ValueError(
                f"Expected RGB raster with at least 3 bands, got {src.count}. "
                "Use the *_rgb_*.tif export, not *_rgbn_*.tif."
            )
        for start in tqdm(range(0, len(catalog), args.batch_size), desc=f"DINOv3 {args.strategy}"):
            batch_df = catalog.iloc[start:start + args.batch_size]
            images = [read_rgb_window(src, rec) for _, rec in batch_df.iterrows()]

            inputs = processor(
                images=images,
                return_tensors="pt",
                do_resize=True,
                size={"height": args.input_size, "width": args.input_size},
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # CRITICAL FIX: cast pixel_values to model dtype, otherwise fp16
            # inference fails with 'Input type and weight type should be the same'.
            if dtype == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

            with torch.inference_mode():
                desc = extractor.extract(inputs["pixel_values"])

            all_embeddings.append(desc.detach().float().cpu().numpy())
            all_tile_ids.extend([int(v) for v in batch_df["tile_id"].tolist()])

    embeddings = np.vstack(all_embeddings).astype("float32")
    # Defensive L2 norm; DescriptorExtractor already L2-normalizes, this is a no-op
    # but guards against numerical drift in fp16->fp32 conversion.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / np.maximum(norms, 1e-12)).astype("float32")
    tile_ids = np.array(all_tile_ids, dtype=np.int64)

    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "tile_ids.npy", tile_ids)

    descriptor_text = {
        "cls": "DINOv3 CLS token + L2 normalization",
        "mean_patch": "DINOv3 patch tokens + global mean pooling + L2 normalization",
        "radial": (
            f"DINOv3 patch tokens + radial token pooling "
            f"(equal-radius bins, mean within bin) + L2 normalization "
            f"(radial_bins={args.radial_bins})"
        ),
    }[args.strategy]

    config_name = catalog["config_name"].iloc[0] if "config_name" in catalog.columns else None

    manifest = {
        "model_name": args.model_name,
        "config_name": config_name,
        "input_size": args.input_size,
        "radial_bins": args.radial_bins,
        "strategy": args.strategy,
        "embedding_shape": list(embeddings.shape),
        "tile_count": int(len(tile_ids)),
        "dtype": "float32",
        "descriptor": descriptor_text,
        "fp16_inference": bool(args.fp16 and device.type == "cuda"),
    }
    (out_dir / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved embeddings: {out_dir / 'embeddings.npy'} shape={embeddings.shape}")
    print(f"Saved tile IDs: {out_dir / 'tile_ids.npy'}")


if __name__ == "__main__":
    main()
