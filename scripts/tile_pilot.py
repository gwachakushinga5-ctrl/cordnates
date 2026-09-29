"""Tile the narrow pilot strip (354 x 31105) into NATIVE-WIDTH 354x640 JPGs.

No width padding (padding 1-col edge data 143px sideways smears streak
artefacts that hurt detectors and labelers). YOLO letterboxes rectangular
inputs internally, so native tiles are fine.

Usage:
    python scripts/tile_pilot.py [--src ...] [--tile-h 640] [--overlap 0.15]
"""
import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO.parent / "roboflow_png_images" / \
    "TRB_Tobacco_Barn_Image_Patches-0000000000-0000037888.png"


def tile(src: Path, out_dir: Path, tile_h: int = 640, overlap: float = 0.15,
         quality: int = 92):
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale tiles from earlier padded runs
    for old in out_dir.glob("tile_*.jpg"):
        old.unlink()
    img = Image.open(src).convert("RGB")
    W, H = img.size
    print(f"Source {src.name}: {W}x{H}")
    tile_w = min(640, W)

    step = int(tile_h * (1 - overlap))
    n = max(1, math.ceil((H - tile_h) / step) + 1)
    needed = (n - 1) * step + tile_h
    pad_bottom = max(0, needed - H)
    if pad_bottom:  # edge-replicate last rows (small, bottom tile only)
        canvas = Image.new("RGB", (W, needed), (0, 0, 0))
        canvas.paste(img, (0, 0))
        edge = img.crop((0, H - 1, W, H)).resize((W, pad_bottom))
        canvas.paste(edge, (0, H))
        img = canvas

    index = []
    for i in range(n):
        y0 = i * step
        chip = img.crop((0, y0, tile_w, y0 + tile_h))
        tid = f"tile_{i:04d}"
        chip.save(out_dir / f"{tid}.jpg", quality=quality)
        index.append({"tile_id": tid, "y0": y0, "x0_orig": 0,
                      "tile_w": tile_w, "tile_h": tile_h,
                      "W_orig": W, "H_orig": H})

    with open(out_dir.parent / "tile_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    (out_dir.parent / "tile_meta.json").write_text(json.dumps(
        {"W_orig": W, "H_orig": H, "tile_w": tile_w, "tile_h": tile_h,
         "tile": tile_h, "pad_left": 0, "pad_right": 0,
         "pad_bottom": pad_bottom, "overlap": overlap, "step": step,
         "n_tiles": n, "src": Path(src).name}, indent=2))
    print(f"Wrote {n} native tiles ({tile_w}x{tile_h}) -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(REPO / "datasets" / "pilot" / "tiles"))
    p.add_argument("--tile-h", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.15)
    a = p.parse_args()
    tile(Path(a.src), Path(a.out), a.tile_h, a.overlap)
