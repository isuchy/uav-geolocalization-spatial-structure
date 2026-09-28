# UAV visual geolocalization — spatial structure evaluation

Code and metadata for the paper *Spatial Structure in UAV Visual
Geolocalization: Controlled Evaluation of Descriptor Pooling and Geographic
Index Partitioning*.

The pipeline localizes a UAV against a georeferenced orthophoto by retrieving
the most similar map tile, using a frozen DINOv3 ViT-S/16 backbone, exact FAISS
inner-product indices and an SQLite tile database. The paper compares four
descriptor aggregation strategies under exhaustive and geographically
partitioned retrieval, and profiles the pipeline on an NVIDIA Jetson Orin Nano.

The map is tiled at 1024 px with a 512 px stride, which at 0.15 m ground
sampling distance is a 153.6 m footprint every 76.8 m. The catalogues in
`metadata/` hold the geometry of all 3213 tiles. **Tiles are never written to
disk**: each is a raster window read from the orthophoto at run time.

## Data

The imagery is archived separately:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23021010.svg)](https://doi.org/10.5281/zenodo.23021010)

Unzip the three archives so the tree looks like this:

```
.
├── scripts/
├── uavloc/
├── metadata/
└── data/
    ├── orthophoto_epsg5514_gsd015m.tif
    ├── synthetic_queries/
    │   ├── synthetic_queries.csv
    │   └── frames/                     200 images
    └── flights/
        ├── flight_080m_01/{video.mp4, telemetry.srt, frames/}   359 frames
        ├── flight_100m_01/{video.mp4, telemetry.srt, frames/}   380 frames
        └── flight_120m_01/{video.mp4, telemetry.srt, frames/}   365 frames
```

`metadata/synthetic_queries.csv` is the same file as the one inside the
synthetic-query archive; either copy works, because `evaluate_synthetic.py`
reads it from `--queries-dir`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_workstation.txt
```

Run every script from the repository root. Each one adds the repository root to
`sys.path` itself, so the `uavloc` package is found without setting
`PYTHONPATH`.

Use `requirements_jetson.txt` on a Jetson. The DINOv3 checkpoint is gated on
Hugging Face; run `huggingface-cli login` before the first extraction, or pass
`--hf-token`.

## Reproduce

### 1. Tile catalogue

Already provided in `metadata/` for all three page sizes. To rebuild one:

```bash
python scripts/build_tile_catalog.py \
  --raster data/orthophoto_epsg5514_gsd015m.tif \
  --tile-points-csv metadata/tile_centres.csv \
  --output-csv metadata/tile_catalog_p2048.csv \
  --tile-px 1024 --stride-px 512 --gsd 0.15 \
  --page-size-m 2048 --config-name L0_1024_s512
```

The page size is fixed here, not at index time, which is why three catalogues
exist: `p2048`, `p1000` and `p600` give 6, 20 and 63 pages over the same 3213
tiles.

### 2. Descriptors

One run per strategy. `--strategy` is one of `cls`, `mean_patch`, `radial`.

```bash
python scripts/extract_descriptors.py \
  --raster data/orthophoto_epsg5514_gsd015m.tif \
  --catalog-csv metadata/tile_catalog_p2048.csv \
  --out-dir results/cls/embeddings \
  --model-name facebook/dinov3-vits16-pretrain-lvd1689m \
  --input-size 512 --batch-size 2 --radial-bins 8 \
  --strategy cls --device cpu
```

Writes `embeddings.npy`, `tile_ids.npy` and `embedding_manifest.json`.

### 3. Index

```bash
python scripts/build_index.py \
  --catalog-csv metadata/tile_catalog_p2048.csv \
  --embeddings results/cls/embeddings/embeddings.npy \
  --tile-ids   results/cls/embeddings/tile_ids.npy \
  --out-dir    results/cls/index_p2048
```

### 4. Synthetic query set

The generated set is on Zenodo. To regenerate it with the same seed:

```bash
python scripts/generate_synthetic_queries.py \
  --raster data/orthophoto_epsg5514_gsd015m.tif \
  --out-dir data/synthetic_queries \
  --num-queries 200 --aoi-margin-m 200 \
  --out-size 512 --db-tile-m 153.6 --seed 42
```

### 5. Evaluate synthetic queries

```bash
python scripts/evaluate_synthetic.py \
  --queries-dir data/synthetic_queries \
  --index-dir   results/cls/index_p2048 \
  --out-dir     results/cls/eval_synthetic \
  --model-name facebook/dinov3-vits16-pretrain-lvd1689m \
  --input-size 512 --radial-bins 8 \
  --strategy cls --top-pages 3 --top-k 10 \
  --tolerance-m 30 --max-pages-in-ram 12 --device cpu
```

### 6. Real flights

```bash
python scripts/evaluate_flight.py \
  --drone-images-dir data/flights/flight_120m_01/frames \
  --gt-csv metadata/ground_truth_flight_120m_01.csv \
  --index-dir results/cls/index_p2048 \
  --out-dir   results/cls/eval_flight_120m \
  --strategy cls --top-pages 3 --top-k 10 \
  --tolerance-m 30 --max-pages-in-ram 12 --device cpu
```

### 7. Full descriptor ablation

Builds a separate database per strategy and evaluates all of them, which is
how Table I was produced:

```bash
python scripts/run_ablation.py \
  --raster data/orthophoto_epsg5514_gsd015m.tif \
  --catalog-csv metadata/tile_catalog_p2048.csv \
  --queries-dir data/synthetic_queries \
  --work-root results/ablation \
  --strategies cls,mean_patch,radial,multirot \
  --multirot-base radial --multirot-angles 0,90,180,270 \
  --input-size 512 --radial-bins 8 \
  --top-pages 3 --top-k 10 --tolerance-m 30 \
  --max-pages-in-ram 12 --device cpu
```

### 8. Edge profiling

On the Jetson, the instrumented variants take the same arguments and
additionally record per-stage timings and page-cache load events:

```bash
python scripts/log_device_telemetry.py --out results/telemetry.csv &

python scripts/instrumented/evaluate_flight.py \
  --drone-images-dir data/flights/flight_120m_01/frames \
  --gt-csv metadata/ground_truth_flight_120m_01.csv \
  --index-dir results/cls/index_p2048 \
  --out-dir   results/cls/edge_flight_120m \
  --strategy cls --top-pages 3 --top-k 10 \
  --tolerance-m 30 --max-pages-in-ram 5 \
  --device cuda --fp16
```

In addition to `summary.json` and `per_query.csv`, these variants write
per-stage timings per query and a page-cache event log, which together give the
latency decomposition reported in the paper.

## Outputs

Each evaluation writes `summary.json` and `per_query.csv`. Every number
reported in the paper is a function of these per-query records.

Query ground-truth positions are never passed to the retriever, so reported
accuracy is independent of GNSS at query time.

## Scope of this release

This repository contains the pipeline that produces the per-query results
reported in the paper. The tooling that aggregates those records into the
tables and figures is not part of it: every aggregate is a function of
`per_query.csv`, so it can be recomputed directly from the outputs above.

## Licence

Code is released under the MIT Licence; see `LICENSE`. Data are released under
the terms stated in the Zenodo record. The orthophoto derives from *Ortofotomozaika SR, Stred 2024 (RGB)*, a public product
of GKÚ Bratislava and NLC Zvolen, published by ÚGKK SR.

The DINOv3 checkpoint is distributed by Meta under its own licence and is not
redistributed here.

## Citation

```bibtex
@unpublished{suchy2026spatial,
  author = {Such\'{y}, Ivan and Tur\v{c}an\'{i}k, Michal},
  title  = {Spatial Structure in {UAV} Visual Geolocalization: Controlled
            Evaluation of Descriptor Pooling and Geographic Index Partitioning},
  note   = {Submitted for review},
  year   = {2026}
}
```

Replace this entry with the published reference once the paper appears. Please
cite the Zenodo record as well when you use the data.
