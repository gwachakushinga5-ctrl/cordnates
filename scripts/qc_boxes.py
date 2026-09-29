"""QC: flag odd boxes + draw downscaled preview with merged boxes.

Pilot: builds preview from the source PNG.
Full: uses the 1/16 overview saved by tile_full.py (--preview).

Usage:
    python scripts/qc_boxes.py [--base datasets/pilot] [--down 8]
    python scripts/qc_boxes.py --base datasets/full --preview preview_16x.jpg --down 1
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(base="datasets/pilot", down=8, preview=None):
    from PIL import Image, ImageDraw

    base = REPO / base if not Path(base).is_absolute() else Path(base)
    meta = json.loads((base / "tile_meta.json").read_text())
    W, H = meta["W_orig"], meta["H_orig"]
    mb = base / "merged_boxes.json"
    boxes = json.loads(mb.read_text())["boxes"] if mb.exists() else []
    if preview:
        img = Image.open(base / preview).convert("RGB")
    else:
        src_png = REPO.parent / "roboflow_png_images" / meta["src"]
        img = Image.open(src_png).convert("RGB")
        img.thumbnail((max(1, W // down), max(1, H // down)))
    sx, sy = img.size[0] / W, img.size[1] / H
    d = ImageDraw.Draw(img)
    flags = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        w, h = x2 - x1, y2 - y1
        ar = max(w, h) / max(1, min(w, h))
        # 10m/px imagery: a barn is ~2-6px; thresholds tuned for that
        if w < 2 or h < 2:
            flags.append(f"subpixel {b}")
        if w > 60 or h > 60:  # >600m cannot be a single barn
            flags.append(f"huge {b}")
        if ar > 6:
            flags.append(f"skinny ar={ar:.1f} {b}")
        d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline="red",
                    width=1)
    out = REPO / "runs" / f"qc_preview_{base.name}.jpg"
    out.parent.mkdir(exist_ok=True)
    img.save(out, quality=88)
    (REPO / "runs" / f"qc_report_{base.name}.txt").write_text(
        f"boxes={len(boxes)} flags={len(flags)}\n" + "\n".join(flags[:500]))
    print(f"[{base.name}] boxes={len(boxes)} flags={len(flags)} preview={out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="datasets/pilot")
    p.add_argument("--down", type=int, default=8)
    p.add_argument("--preview", default=None)
    a = p.parse_args()
    main(a.base, a.down, a.preview)
