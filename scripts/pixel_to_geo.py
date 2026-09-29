"""Pixel boxes -> GeoJSON using image geotransform.

Supports ModelTransformationTag (6+-elem) or Scale+Tiepoint pair.
If no geo tags exist, writes pixel_boxes.geojson so work is not blocked.

Usage:
    python scripts/pixel_to_geo.py [--base datasets/pilot]
    python scripts/pixel_to_geo.py --base datasets/full --lon0 29.372753834026465
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
S_DEFAULT = 8.983152841195215e-05


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


def main(base="datasets/pilot", lon0=None, lat0=None):
    base = REPO / base if not Path(base).is_absolute() else Path(base)
    meta_path = REPO / "pilot_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    mb = base / "merged_boxes.json"
    if not mb.exists():
        raise SystemExit(f"No merged_boxes.json in {base}.")
    boxes = json.loads(mb.read_text())["boxes"]
    tile_meta = json.loads((base / "tile_meta.json").read_text())
    lon0 = tile_meta.get("origin_lon", lon0)
    lat0 = tile_meta.get("origin_lat", lat0)
    if lon0 is not None and lat0 is not None:
        # explicit per-dataset affine wins (pilot_meta.json is pilot-only)
        s = tile_meta.get("pixel_scale_deg", S_DEFAULT)
        fn = lambda px, py: (lon0 + px * s, lat0 - py * s)  # noqa: E731
    else:
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
    out = base / name
    out.write_text(json.dumps({"type": "FeatureCollection",
                               "features": feats}))
    print(f"[{base.name}] {len(feats)} features -> {out.name} "
          f"({'georeferenced' if fn else 'PIXEL-ONLY'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="datasets/pilot")
    p.add_argument("--lon0", type=float, default=None)
    p.add_argument("--lat0", type=float, default=None)
    a = p.parse_args()
    main(a.base, a.lon0, a.lat0)
