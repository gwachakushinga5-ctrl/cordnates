"""GEE-anchored seed boxes + fusion with brightness candidates.

For each CSV point (= GEE asset proxy): inverse-affine to strip pixels and
emit the script's own 100m patch window (+-50m ~= +-5px) as a class-0 seed box
(conf 1.0, linked to record_id). Seeds are appended to the per-tile YOLO
labels; detector boxes >50m from every point are flagged in sidecars.

Writes standard 5-col YOLO txt (Roboflow-safe) + per-tile sidecar JSON with
{conf, record_id, far_from_point} aligned by row index.

Usage:
    python scripts/gee_seeds.py [--buffer-m 50]
    python scripts/predict_pilot.py --merge-only   # refresh merged_boxes.json
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "datasets" / "pilot"
TILES = PILOT / "tiles"
LABELS = PILOT / "labels"
CSV = REPO.parent / "cleaned_tobacco_barn_coordinates.csv"

LON0, LAT0 = 32.776290782498506, -16.443032455108956
S = 8.983152841195215e-05
M_PER_DEG = 111320.0
PX_M = S * M_PER_DEG  # ~= 10.0


def lonlat_to_px(lon, lat):
    return (lon - LON0) / S, (LAT0 - lat) / S


def main(buffer_m=50):
    meta = json.loads((PILOT / "tile_meta.json").read_text())
    Tw, Th, W, H = meta["tile_w"], meta["tile_h"], meta["W_orig"], meta["H_orig"]
    rows = list(csv.DictReader(open(PILOT / "tile_index.csv")))
    half = buffer_m / PX_M

    points = list(csv.DictReader(open(CSV)))
    audit, seeds = [], []
    for r in points:
        rid, la, lo = r["record_id"], float(r["latitude"]), float(r["longitude"])
        px, py = lonlat_to_px(lo, la)
        inside = 0 <= px < W and 0 <= py < H
        hit_tiles = [t["tile_id"] for t in rows
                     if int(t["y0"]) <= py < int(t["y0"]) + Th] if inside else []
        audit.append({"record_id": rid, "latitude": la, "longitude": lo,
                      "px": round(px, 1), "py": round(py, 1),
                      "in_pilot": inside, "tiles": ";".join(hit_tiles)})
        if inside:
            x1, y1 = max(0, px - half), max(0, py - half)
            x2, y2 = min(W - 1, px + half), min(H - 1, py + half)
            seeds.append({"record_id": rid, "xyxy": [x1, y1, x2, y2],
                          "tiles": hit_tiles})
    (PILOT / "seed_audit.csv").write_text(
        "record_id,latitude,longitude,px,py,in_pilot,tiles\n" +
        "\n".join(f"{a['record_id']},{a['latitude']},{a['longitude']},"
                  f"{a['px']},{a['py']},{a['in_pilot']},{a['tiles']}"
                  for a in audit))

    # seed strip-pixel list for far_from_point flagging
    sxs = [ (s["xyxy"][0] + s["xyxy"][2]) / 2 for s in seeds]
    sys_ = [(s["xyxy"][1] + s["xyxy"][3]) / 2 for s in seeds]

    def near_seed(cx, cy, r_px=half):
        return any(abs(cx - sx) <= r_px and abs(cy - sy) <= r_px
                   for sx, sy in zip(sxs, sys_))

    per_tile = {}
    for s in seeds:
        for tid in s["tiles"]:
            per_tile.setdefault(tid, []).append(s)

    n_seed_rows = 0
    for lab in sorted(LABELS.glob("*.txt")):
        tid = lab.stem
        y0 = next(int(t["y0"]) for t in rows if t["tile_id"] == tid)
        tml = [l for l in lab.read_text().splitlines() if l.strip()]
        # flag detector boxes by strip distance to nearest seed
        side = []
        for l in tml:
            p = l.split()
            c = float(p[5]) if len(p) > 5 else 1.0
            cx_s = float(p[1]) * Tw  # tile x == strip x (x0_orig=0)
            cy_s = float(p[2]) * Th + y0
            side.append({"conf": c, "record_id": None,
                         "far_from_point": not any(
                             abs(cx_s - sx) <= half and abs(cy_s - sy) <= half
                             for sx, sy in zip(sxs, sys_))})
        out_lines, new_side = [], list(side)
        for s in per_tile.get(tid, []):
            x1, y1, x2, y2 = s["xyxy"]
            cx = ((x1 + x2) / 2) / Tw
            cy = ((y1 + y2) / 2 - y0) / Th
            w, h = (x2 - x1) / Tw, (y2 - y1) / Th
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            out_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            new_side.append({"conf": 1.0, "record_id": s["record_id"],
                             "far_from_point": False})
            n_seed_rows += 1
        # normalize old rows to 5-col
        base = [" ".join(l.split()[:5]) for l in tml]
        lab.write_text("\n".join(base + out_lines))
        (LABELS / f"{tid}.sidecar.json").write_text(json.dumps(new_side))
    print(f"points={len(points)} in_pilot={sum(a['in_pilot'] for a in audit)} "
          f"seeds={len(seeds)} seed_rows_written={n_seed_rows}")
    print("Next: python scripts/predict_pilot.py --merge-only")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-m", type=float, default=50)
    a = p.parse_args()
    main(a.buffer_m)
