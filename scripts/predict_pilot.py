"""AI pre-labels: YOLOv8m over pilot tiles -> YOLO txt + strip-level merged boxes.

COCO has no 'barn' class, so by default every detection above --conf is kept
as a class-0 (barn_roof) CANDIDATE for human review in Roboflow. Humans delete
false positives (houses/sheds) and tighten boxes.

Usage:
    python scripts/predict_pilot.py --conf 0.25
    python scripts/predict_pilot.py --merge-only   # re-run NMS from existing labels
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TILES = REPO / "datasets" / "pilot" / "tiles"
LABELS = REPO / "datasets" / "pilot" / "labels"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def nms(boxes, iou_thr=0.5):
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(b)
    return keep


def merge_from_labels(iou_thr=0.5):
    meta = json.loads((TILES.parent / "tile_meta.json").read_text())
    pad_left, W_orig, H_orig = (meta.get("pad_left", 0), meta["W_orig"],
                                meta["H_orig"])
    Tw = meta.get("tile_w", meta.get("tile", 640))
    Th = meta.get("tile_h", meta.get("tile", 640))
    rows = list(csv.DictReader(open(TILES.parent / "tile_index.csv")))
    y0_of = {r["tile_id"]: int(r["y0"]) for r in rows}
    strip = []
    for lab in sorted(LABELS.glob("*.txt")):
        tid, y0 = lab.stem, y0_of[lab.stem]
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            x1t, y1t = (cx - w / 2) * Tw, (cy - h / 2) * Th
            x2t, y2t = (cx + w / 2) * Tw, (cy + h / 2) * Th
            x1 = min(max(x1t - pad_left, 0), W_orig)
            x2 = min(max(x2t - pad_left, 0), W_orig)
            y1 = min(max(y1t + y0, 0), H_orig)
            y2 = min(max(y2t + y0, 0), H_orig)
            if x2 > x1 and y2 > y1:
                strip.append({"tile": tid, "conf": conf,
                              "xyxy": [x1, y1, x2, y2]})
    merged = nms(strip, iou_thr)
    out = TILES.parent / "merged_boxes.json"
    out.write_text(json.dumps({"n_raw": len(strip), "n_merged": len(merged),
                               "boxes": merged}, indent=2))
    print(f"Merge: {len(strip)} raw -> {len(merged)} merged -> {out.name}")
    return merged


def main(conf=0.25, iou_thr=0.5, model="yolov8m.pt", merge_only=False,
         max_per_tile=50):
    LABELS.mkdir(parents=True, exist_ok=True)
    if merge_only:
        merge_from_labels(iou_thr)
        return
    from ultralytics import YOLO

    tiles = sorted(TILES.glob("*.jpg"))
    if not tiles:
        raise SystemExit(f"No tiles in {TILES} — run tile_pilot.py first.")
    m = YOLO(model)
    meta = json.loads((TILES.parent / "tile_meta.json").read_text())
    Tw = meta.get("tile_w", meta.get("tile", 640))
    Th = meta.get("tile_h", meta.get("tile", 640))
    for t in tiles:
        r = m.predict(t, conf=conf, iou=0.5, verbose=False)[0]
        lines = []
        if r.boxes is not None:
            dets = sorted(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()),
                          key=lambda z: z[1], reverse=True)[:max_per_tile]
            for (x1, y1, x2, y2), c in dets:
                cx, cy = ((x1 + x2) / 2 / Tw), ((y1 + y2) / 2 / Th)
                w, h = ((x2 - x1) / Tw), ((y2 - y1) / Th)
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.3f}")
        (LABELS / f"{t.stem}.txt").write_text("\n".join(lines))
        print(f"{t.name}: {len(lines)} candidates")
    merge_from_labels(iou_thr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--merge-only", action="store_true")
    a = p.parse_args()
    main(a.conf, a.iou, a.model, a.merge_only)
