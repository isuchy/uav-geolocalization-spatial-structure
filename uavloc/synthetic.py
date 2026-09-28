"""
Synthetic UAV query generation. For each query we:
  - sample a random (x, y) inside the AOI (with margin),
  - cut a tile from the orthomosaic at that position,
  - apply random affine perturbations (rotation, slight scale, brightness),
  - save as a PNG.

The point is to evaluate retrieval with controlled disturbances that
approximate the gap between a clean map tile and a UAV view of the same
location.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


@dataclass
class SyntheticConfig:
    num_queries: int = 200
    aoi_margin_m: float = 200.0
    out_size: int = 512
    db_tile_m: float = 153.6
    oversample: float = 1.6
    rotation_deg_range: tuple[float, float] = (-30.0, 30.0)
    scale_range: tuple[float, float] = (0.95, 1.05)
    brightness_range: tuple[float, float] = (0.85, 1.15)
    seed: int = 42


@dataclass
class SyntheticManifest:
    config: dict = field(default_factory=dict)
    queries: list[dict] = field(default_factory=list)


def _read_window_rgb(src: rasterio.DatasetReader, cx: float, cy: float, tile_m: float, out_px: int) -> Image.Image:
    """Read a square ground-window centred at (cx, cy) and resize to out_px."""
    half = tile_m / 2.0
    minx, miny, maxx, maxy = cx - half, cy - half, cx + half, cy + half
    inv = ~src.transform
    col_min, row_max = inv * (minx, miny)
    col_max, row_min = inv * (maxx, maxy)
    col_off = int(round(min(col_min, col_max)))
    row_off = int(round(min(row_min, row_max)))
    width = int(round(abs(col_max - col_min)))
    height = int(round(abs(row_max - row_min)))
    win = Window(col_off, row_off, width, height)
    arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img.resize((out_px, out_px), Image.BILINEAR)


def generate_synthetic_queries(
    raster_path: Path,
    out_dir: Path,
    config: SyntheticConfig,
    aoi_bounds: Optional[tuple[float, float, float, float]] = None,
) -> SyntheticManifest:
    rng = random.Random(config.seed)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest = SyntheticManifest()
    manifest.config = {
        **config.__dict__,
        "raster": str(raster_path),
        "aoi_bounds": list(aoi_bounds) if aoi_bounds else None,
    }

    with rasterio.open(raster_path) as src:
        if aoi_bounds is None:
            left, bottom, right, top = src.bounds
        else:
            left, bottom, right, top = aoi_bounds
        m = config.aoi_margin_m
        x_lo, x_hi = left + m, right - m
        y_lo, y_hi = bottom + m, top - m

        th = math.radians(max(abs(a) for a in config.rotation_deg_range))
        need = (math.cos(th) + math.sin(th)) / min(config.scale_range)
        if config.oversample < need:
            raise ValueError(
                f"oversample={config.oversample} is too small, at least {need:.4f} is required"
            )

        for i in range(config.num_queries):
            cx = rng.uniform(x_lo, x_hi)
            cy = rng.uniform(y_lo, y_hi)
            '''
            base = _read_window_rgb(src, cx, cy, config.db_tile_m, config.out_size)

            angle = rng.uniform(*config.rotation_deg_range)
            scale = rng.uniform(*config.scale_range)
            brightness = rng.uniform(*config.brightness_range)

            tile = base.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            if scale != 1.0:
                new_size = int(round(config.out_size * scale))
                tile = tile.resize((new_size, new_size), Image.BILINEAR)
                if new_size >= config.out_size:
                    pad = (new_size - config.out_size) // 2
                    tile = tile.crop((pad, pad, pad + config.out_size, pad + config.out_size))
                else:
                    pad = (config.out_size - new_size) // 2
                    canvas = Image.new("RGB", (config.out_size, config.out_size))
                    canvas.paste(tile, (pad, pad))
                    tile = canvas
            '''
            angle = rng.uniform(*config.rotation_deg_range)
            scale = rng.uniform(*config.scale_range)
            brightness = rng.uniform(*config.brightness_range)

            # Read an oversampled ground window, rotate and scale it, and only
            # then centre-crop, so that no padding can enter the crop.
            over_m = config.db_tile_m * config.oversample
            over_px = int(round(config.out_size * config.oversample))
            base = _read_window_rgb(src, cx, cy, over_m, over_px)

            tile = base.rotate(angle, resample=Image.BILINEAR)   # bez fillcolor
            if scale != 1.0:
                new_size = int(round(over_px * scale))
                tile = tile.resize((new_size, new_size), Image.BILINEAR)

            c = tile.size[0] // 2
            h = config.out_size // 2
            tile = tile.crop((c - h, c - h, c + h, c + h))

            arr = np.asarray(tile, dtype=np.float32) * brightness
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            tile = Image.fromarray(arr, mode="RGB")

            fname = f"q_{i:06d}.png"
            tile.save(frames_dir / fname)
            manifest.queries.append({
                "frame_path": f"frames/{fname}",
                "gt_x": cx,
                "gt_y": cy,
                "rotation_deg": angle,
                "scale": scale,
                "brightness": brightness,
            })

    (out_dir / "synthetic_queries_manifest.json").write_text(
        json.dumps(manifest.__dict__, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)),
        encoding="utf-8",
    )
    return manifest
