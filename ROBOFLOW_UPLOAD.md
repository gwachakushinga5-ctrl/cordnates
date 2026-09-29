# Roboflow upload — pilot (GEE-anchored II)

## What's in `datasets/pilot/` (ready to upload)
- `tiles/tile_*.jpg` — 58 native tiles (354x640, 15% overlap) + `tile_index.csv`.
- `labels/tile_*.txt` — standard **5-col YOLO** (`0 cx cy w h`), 272 fused boxes
  after NMS: 270 brightness candidates + 3 GEE seed boxes (100m windows from
  `gee_script.js` buffer logic; 1 seed merged into a detector box).
- `labels/tile_*.sidecar.json` — local-only, row-aligned: `{conf, record_id,
  far_from_point}`. Do NOT upload; kept for re-linking after export.
- `seed_audit.csv` — all 6235 CSV points projected to strip pixels; 3 in pilot.

## Steps
1. In Roboflow: create **Object Detection** project, class `barn_roof`.
2. Upload `tiles/*.jpg`, then matching `labels/*.txt` stems.
3. Correct per `LABEL_GUIDE.md`: tighten to drip-edge, split double barns,
   delete houses/sheds/kilns/river-sand. Note: all 305 detector candidates are
   >50m from the 3 GEE points in this strip (`far_from_point`) — treat lone
   bright hits with suspicion, but keep clear rectangles near tracks/compounds.
4. Generate version: **train/val 80/20**, no auto-augment for pilot review.
5. Export **YOLOv8** into `datasets/pilot/roboflow_export/`. Re-link exported
   boxes to `record_id`s by IoU match against `merged_boxes.json` + sidecars
   (positions are stable), then run `pixel_to_geo.py` for `geo_boxes.geojson`.
