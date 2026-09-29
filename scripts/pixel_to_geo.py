"""Pixel boxes -> GeoJSON using the pilot TIF geotransform (from pilot_meta.json).

Supports ModelTransformationTag (6 or 12-elem) or Scale+Tiepoint pair.
If no geo tags exist, writes pixel_boxes.geojson so work is not blocked.

Usage: python scripts/pixel_to_geo.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "datasets" / "pilot"


def transform_fn(meta):
    t = meta.get("ModelTransformationTag")
    if t and len(t) >= 6:
        # [a, b, c, d, X0, Y0] affine-ish (first 6 of 16/12-elem form)
        a, b, X0, d, e, Y0 = t[0], t[1], t[3], t[4], t[5], t[7]
        return lambda px, py: (a * px + b * py + X0, d * px + e * py + Y0)
    s, tp = meta.get("ModelPixelScaleTag"), meta.get("ModelTiepointTag")
    if s and tp and len(tp) >= 6:
        sx, sy = s[0], s[1]
        X0, Y0 = tp[3], tp[4]
        return lambda px, py: (X0 + px * sx, Y0 - py * sy)
    return None


def main():
    meta = json.loads((REPO / "pilot_meta.json").read_text())
    mb = PILOT / "merged_boxes.json"
    if not mb.exists():
        raise SystemExit("Run predict_pilot.py first (no merged_boxes.json).")
    boxes = json.loads(mb.read_text())["boxes"]
    fn = transform_fn(meta)
    feats = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["xyxy"]
        if fn:
            X1, Y1 = fn(x1, y1)
            X2, Y2 = fn(x2, y2)
            ring = [[X1, Y1], [X2, Y1], [X2, Y2], [X1, Y2], [X1, Y1]]
        else:
            ring = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
        feats.append({"type": "Feature",
                      "properties": {"id": i, "conf": b.get("conf"),
                                     "tile": b.get("tile")},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    name = "geo_boxes.geojson" if fn else "pixel_boxes.geojson"
    out = PILOT / name
    out.write_text(json.dumps({"type": "FeatureCollection",
                               "features": feats}, indent=2))
    print(f"Wrote {len(feats)} features -> {out.name} "
          f"({'georeferenced' if fn else 'PIXEL-ONLY: no geo tags in TIF'})")


if __name__ == "__main__":
    main()
