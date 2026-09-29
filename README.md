# Tobacco barn-roof bounding boxes — pilot

## Source
Parent folder `barn-coordinates/` holds the raw assets (not in this git repo):
- `TRB_Tobacco_Barn_Image_Patches-0000000000-0000037888.tif` (29MB pilot strip, 354 x 31105 px RGB)
- `TRB_Tobacco_Barn_Image_Patches-0000000000-0000000000.tif` (3.2GB full mosaic, 37888 x 31105 px — scaled later)
- `cleaned_tobacco_barn_coordinates.csv` (6235 GPS points, for validation)

## Pipeline (AI-assisted → manual correct, YOLO + Roboflow)
1. `python scripts/inspect_pilot.py` — reads GeoTIFF tags via tifffile, writes `pilot_meta.json`.
2. `python scripts/tile_pilot.py` — tiles the strip into 640x640 JPGs with overlap, writes `tile_index.csv`.
3. `python scripts/predict_pilot.py` — YOLOv8m pre-labels (low conf, building-like classes) → `datasets/pilot/labels/*.txt`.
4. Upload `tiles/` + `labels/` to Roboflow Object Detection project, correct boxes, export YOLOv8.
5. `python scripts/qc_boxes.py` — draws QC preview + flags bad boxes.
6. `python scripts/pixel_to_geo.py` — pixel boxes → `geo_boxes.geojson` using TIF transform (if present).

## Label rule (summary)
Single class `0 = barn_roof`. Tight box on roof drip-edge only. See `LABEL_GUIDE.md`.

## Pilot results (2026-09-29)
- Strip: 354 x 31105 px, EPSG:4326, ~10m/px (`pilot_meta.json`).
- 58 native tiles 354x640, 15% overlap (`datasets/pilot/tile_index.csv`).
- YOLOv8m-COCO at conf 0.25: **0 detections** (expected — nadir 10m satellite
  texture is out-of-domain for ground-photo pretraining).
- Classical bright-roof candidates (`scripts/roof_candidates.py`): 305 raw.
- GEE script (`gee_script.js`, pasted 2026-09-29): Sentinel-2 2024 median,
  scale 10, per-point 50m buffer. `scripts/gee_seeds.py` projected all 6235
  CSV points → **3 in pilot**, each a 100m seed box; fused 308 → **272 merged**
  (`merged_boxes.json`), all QC-clean, georeferenced to
  `datasets/pilot/geo_boxes.geojson`. All 305 detector hits are >50m from the
  3 GEE points in-strip (flagged `far_from_point` in sidecars) — review
  accordingly.
- **Next human step:** upload `tiles/` + `labels/` to Roboflow per
  `ROBOFLOW_UPLOAD.md`; confirm/adjust seeds, delete river/sand false
  positives, export YOLOv8 — corrections become training data for an
  in-domain detector to scale to the full 37888-wide mosaic (6232 points).
