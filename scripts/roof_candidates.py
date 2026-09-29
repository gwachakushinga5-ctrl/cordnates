"""Classical bright-roof candidate detector (works at 10m/px where COCO-YOLO fires nothing).

Metallic barn roofs are among the brightest pixels in these scenes. This finds
bright, compact, roughly rectangular blobs and writes YOLO txt pre-labels
(class 0 + score) for human review in Roboflow.

Method per tile:
  1. brightness = min(R,G,B)  (neutral/white = high in ALL bands; red soil is not)
  2. threshold at high percentile (default 99.5) + absolute floor
  3. connected components; keep area in [min_area, max_area]
  4. score by fill-ratio inside min-area rect (rectangularity) + squareness-agnostic
  5. keep top_k per tile; dilate box by --grow px for context

Usage:
    python scripts/roof_candidates.py [--pct 99.5] [--min-area 2] [--max-area 500]
    python scripts/predict_pilot.py --merge-only   # then merge to strip level
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "datasets" / "pilot"
TILES = PILOT / "tiles"
LABELS = PILOT / "labels"


def candidates_from_array(img_bgr, pct=99.5, floor=150, min_area=2,
                          max_area=500, top_k=40, grow=3):
    import cv2
    import numpy as np

    b, g, r = cv2.split(img_bgr)
    bright = np.minimum(np.minimum(r, g), b).astype(np.float32)
    thr = max(float(np.percentile(bright, pct)), float(floor))
    mask = ((bright >= thr) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    outs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > 6:  # tracks/roads are long and thin; barns are compact
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        rect_area = w * h
        fill = area / max(1, rect_area)
        if fill < 0.35:
            continue
        score = 0.5 * fill + 0.5 * min(1.0, area / 25.0)
        x1 = max(0, x - grow)
        y1 = max(0, y - grow)
        x2 = min(W - 1, x + w + grow)
        y2 = min(H - 1, y + h + grow)
        outs.append((score, x1, y1, x2, y2))
    outs.sort(reverse=True)
    return outs[:top_k]


def candidates(tile_path, pct=99.5, floor=150, min_area=2, max_area=500,
               top_k=40, grow=3):
    import cv2

    img = cv2.imread(str(tile_path))
    if img is None:
        return []
    return candidates_from_array(img, pct, floor, min_area, max_area,
                                 top_k, grow)


def main(pct, floor, min_area, max_area, top_k, grow):
    tiles = sorted(TILES.glob("*.jpg"))
    if not tiles:
        raise SystemExit(f"No tiles in {TILES} — run tile_pilot.py first.")
    meta = json.loads((PILOT / "tile_meta.json").read_text())
    Tw, Th = meta["tile_w"], meta["tile_h"]
    LABELS.mkdir(parents=True, exist_ok=True)
    total = 0
    for t in tiles:
        dets = candidates(t, pct, floor, min_area, max_area, top_k, grow)
        lines = []
        for s, x1, y1, x2, y2 in dets:
            cx, cy = ((x1 + x2) / 2 / Tw), ((y1 + y2) / 2 / Th)
            w, h = ((x2 - x1) / Tw), ((y2 - y1) / Th)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.3f}")
        (LABELS / f"{t.stem}.txt").write_text("\n".join(lines))
        total += len(lines)
        print(f"{t.name}: {len(lines)} candidates")
    print(f"TOTAL {total} candidates over {len(tiles)} tiles. "
          f"Next: python scripts/predict_pilot.py --merge-only")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pct", type=float, default=99.5)
    p.add_argument("--floor", type=float, default=150)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--max-area", type=int, default=500)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--grow", type=int, default=3)
    a = p.parse_args()
    main(a.pct, a.floor, a.min_area, a.max_area, a.top_k, a.grow)
