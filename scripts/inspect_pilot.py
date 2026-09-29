"""Read pilot GeoTIFF tags with tifffile (no GDAL needed) -> pilot_meta.json."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TIF = REPO.parent / "TRB_Tobacco_Barn_Image_Patches-0000000000-0000037888.tif"


def main(tif_path: str = str(DEFAULT_TIF), out: str = "pilot_meta.json"):
    import tifffile

    tif_path = Path(tif_path)
    if not tif_path.exists():
        sys.exit(f"TIFF not found: {tif_path}")
    out_path = REPO / out if not Path(out).is_absolute() else Path(out)

    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        tags = {t.name: t.value for t in page.tags.values()}
        meta = {
            "file": tif_path.name,
            "shape": list(page.shape),
            "dtype": str(page.dtype),
            "axes": getattr(page, "axes", ""),
            "description": page.description,
            "tags_present": sorted(tags.keys()),
            "ModelPixelScaleTag": tags.get("ModelPixelScaleTag"),
            "ModelTiepointTag": tags.get("ModelTiepointTag"),
            "ModelTransformationTag": tags.get("ModelTransformationTag"),
            "GeoKeyDirectoryTag": tags.get("GeoKeyDirectoryTag"),
            "GeoAsciiParamsTag": tags.get("GeoAsciiParamsTag"),
        }
        # JSON-safe conversion (tuples -> lists)
        meta = json.loads(json.dumps(meta, default=str))

    out_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps({k: meta[k] for k in ("shape", "dtype", "tags_present")}, indent=2))
    gt = meta.get("ModelTransformationTag") or (
        meta.get("ModelPixelScaleTag"), meta.get("ModelTiepointTag"))
    print("Geotransform-ish:", gt)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_TIF))
    p.add_argument("--out", default="pilot_meta.json")
    a = p.parse_args()
    main(a.src, a.out)
