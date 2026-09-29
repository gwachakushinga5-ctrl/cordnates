"""GEE-anchored seed boxes + fusion with brightness candidates.

For each CSV point (= GEE asset proxy): inverse-affine to image pixels and
emit the script's own 100m patch window (+-50m ~= +-5px) as a class-0 seed box
(conf 1.0, linked to record_id). Seeds are appended to the per-tile YOLO
labels; detector boxes >50m from every point are flagged in sidecars.

Writes standard 5-col YOLO txt (Roboflow-safe) + per-tile sidecar JSON with
{conf, record_id, far_from_point} aligned by row index.

Usage:
    python scripts/gee_seeds.py [--buffer-m 50]
    python scripts/gee_seeds.py --base datasets/full --lon0 29.372753834026465
    python scripts/predict_pilot.py --merge-only   # refresh merged_boxes.json
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV = REPO.parent / "cleaned_tobacco_barn_coordinates.csv"

S = 8.983152841195215e-05
M_PER_DEG = 111320.0
PX_M = S * M_PER_DEG  # ~= 10.0


def main(base, lon0, lat0, buffer_m=50):
    base = REPO / base if not Path(base).is_absolute() else Path(base)
    tiles_dir, labels_dir = base / "tiles", base / "labels"
    meta = json.loads((base / "tile_meta.json").read_text())
    Tw, Th, W, H = meta["tile_w"], meta["tile_h"], meta["W_orig"], meta["H_orig"]
    rows = list(csv.DictReader(open(base / "tile_index.csv")))
    half = buffer_m / PX_M
    in_key = "in_image"

    def lonlat_to_px(lon, lat):
        return (lon - lon0) / S, (lat0 - lat) / S

    def x0_of(r):
        return int(r.get("x0", r.get("x0_orig", 0)))

    import numpy as np
    tids = [t["tile_id"] for t in rows]
    X0 = np.array([int(t.get("x0", t.get("x0_orig", 0))) for t in rows])
    Y0 = np.array([int(t["y0"]) for t in rows])

    points = list(csv.DictReader(open(CSV)))
    audit, seeds = [], []
    for r in points:
        rid, la, lo = r["record_id"], float(r["latitude"]), float(r["longitude"])
        px, py = lonlat_to_px(lo, la)
        inside = 0 <= px < W and 0 <= py < H
        if inside:
            m = (X0 <= px) & (px < X0 + Tw) & (Y0 <= py) & (py < Y0 + Th)
            hit_tiles = [tids[i] for i in np.nonzero(m)[0]]
        else:
            hit_tiles = []
        audit.append({"record_id": rid, "latitude": la, "longitude": lo,
                      "px": round(px, 1), "py": round(py, 1),
                      "inside": inside, "tiles": ";".join(hit_tiles)})
        if inside:
            x1, y1 = max(0, px - half), max(0, py - half)
            x2, y2 = min(W - 1, px + half), min(H - 1, py + half)
            seeds.append({"record_id": rid, "xyxy": [x1, y1, x2, y2],
                          "tiles": hit_tiles})
    (base / "seed_audit.csv").write_text(
        "record_id,latitude,longitude,px,py,inside,tiles\n" +
        "\n".join(f"{a['record_id']},{a['latitude']},{a['longitude']},"
                  f"{a['px']},{a['py']},{a['inside']},{a['tiles']}"
                  for a in audit))

    # seed pixel grid for fast far_from_point flagging
    sxs = [(s["xyxy"][0] + s["xyxy"][2]) / 2 for s in seeds]
    sys_ = [(s["xyxy"][1] + s["xyxy"][3]) / 2 for s in seeds]
    cell = max(10.0, half * 2)
    seed_grid = {}
    for sx, sy in zip(sxs, sys_):
        seed_grid.setdefault((int(sx // cell), int(sy // cell)), []).append(
            (sx, sy))

    def is_far(cx, cy):
        gx, gy = int(cx // cell), int(cy // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sx, sy in seed_grid.get((gx + dx, gy + dy), ()):
                    if abs(cx - sx) <= half and abs(cy - sy) <= half:
                        return False
        return True

    per_tile = {}
    for s in seeds:
        for tid in s["tiles"]:
            per_tile.setdefault(tid, []).append(s)
    x0_of_map = {t["tile_id"]: x0_of(t) for t in rows}
    y0_of_map = {t["tile_id"]: int(t["y0"]) for t in rows}

    n_seed_rows = 0
    for lab in sorted(labels_dir.glob("*.txt")):
        tid = lab.stem
        if tid not in y0_of_map:
            continue
        x0, y0 = x0_of_map[tid], y0_of_map[tid]
        tml = [l for l in lab.read_text().splitlines() if l.strip()]
        # flag detector boxes by image distance to nearest seed
        side = []
        for l in tml:
            p = l.split()
            c = float(p[5]) if len(p) > 5 else 1.0
            cx_s = float(p[1]) * Tw + x0
            cy_s = float(p[2]) * Th + y0
            side.append({"conf": c, "record_id": None,
                         "far_from_point": is_far(cx_s, cy_s)})
        out_lines, new_side = [], list(side)
        for s in per_tile.get(tid, []):
            x1, y1, x2, y2 = s["xyxy"]
            cx = ((x1 + x2) / 2 - x0) / Tw
            cy = ((y1 + y2) / 2 - y0) / Th
            w, h = (x2 - x1) / Tw, (y2 - y1) / Th
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            out_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            new_side.append({"conf": 1.0, "record_id": s["record_id"],
                             "far_from_point": False})
            n_seed_rows += 1
        # normalize old rows to 5-col
        base_lines = [" ".join(l.split()[:5]) for l in tml]
        lab.write_text("\n".join(base_lines + out_lines))
        (labels_dir / f"{tid}.sidecar.json").write_text(json.dumps(new_side))
    print(f"points={len(points)} inside={sum(a['inside'] for a in audit)} "
          f"seeds={len(seeds)} seed_rows_written={n_seed_rows}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="datasets/pilot")
    p.add_argument("--lon0", type=float, default=32.776290782498506)
    p.add_argument("--lat0", type=float, default=-16.443032455108956)
    p.add_argument("--buffer-m", type=float, default=50)
    a = p.parse_args()
    main(a.base, a.lon0, a.lat0, a.buffer_m)
