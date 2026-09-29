"""Tile the full 37888x31105 mosaic with windowed reads (low RAM) + candidates.

Single pass per window: read 640x640 (edge-padded at borders) -> save JPG ->
run bright-roof candidates -> write 5-col YOLO txt + sidecar. Also accumulates
a 1/16 overview preview for QC (full image can never be loaded at once).

~70 cols x 57 rows = ~3990 tiles. Expect 30-90 min on CPU.

Usage:
    python scripts/tile_full.py [--tile 640] [--overlap 0.15] [--quality 82]
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
FULL = REPO / "datasets" / "full"
TILES = FULL / "tiles"
LABELS = FULL / "labels"
SRC = REPO.parent / "TRB_Tobacco_Barn_Image_Patches-0000000000-0000000000.tif"


def main(tile=640, overlap=0.15, quality=82):
    import cv2
    import numpy as np
    import tifffile
    import zarr
    from roof_candidates import candidates_from_array

    TILES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    tif = tifffile.TiffFile(SRC)
    z = zarr.open(tif.aszarr(), mode="r")
    H, W = int(z.shape[0]), int(z.shape[1])
    print(f"Mosaic {W}x{H}")

    step = int(tile * (1 - overlap))
    ncols = math.ceil((W - tile) / step) + 1
    nrows = math.ceil((H - tile) / step) + 1
    print(f"Grid {ncols} cols x {nrows} rows = {ncols * nrows} tiles")

    # 1/16 overview canvas (2368x1944 RGB, ~13MB)
    pv_scale = 16
    preview = np.zeros((math.ceil(H / pv_scale), math.ceil(W / pv_scale), 3),
                       np.uint8)

    index, total_cand, done = [], 0, 0
    for rr in range(nrows):
        y0 = min(rr * step, H - tile)
        for cc in range(ncols):
            x0 = min(cc * step, W - tile)
            win = z[y0:y0 + tile, x0:x0 + tile, :]
            win = np.asarray(win)
            if win.shape[:2] != (tile, tile):  # border: edge-pad
                win = cv2.copyMakeBorder(
                    win, 0, tile - win.shape[0], 0, tile - win.shape[1],
                    cv2.BORDER_REPLICATE)
            rgb = win[:, :, ::-1].copy()  # RGB -> BGR for cv2
            tid = f"r{rr:03d}_c{cc:03d}"
            cv2.imwrite(str(TILES / f"{tid}.jpg"), rgb,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            dets = candidates_from_array(rgb)
            lines, side = [], []
            for s, x1, y1, x2, y2 in dets:
                cx, cy = ((x1 + x2) / 2 / tile), ((y1 + y2) / 2 / tile)
                w, h = ((x2 - x1) / tile), ((y2 - y1) / tile)
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                side.append({"conf": round(float(s), 3), "record_id": None,
                             "far_from_point": None})
            (LABELS / f"{tid}.txt").write_text("\n".join(lines))
            (LABELS / f"{tid}.sidecar.json").write_text(json.dumps(side))
            total_cand += len(lines)
            index.append({"tile_id": tid, "x0": x0, "y0": y0,
                          "tile_w": tile, "tile_h": tile,
                          "W_orig": W, "H_orig": H})
            # overview: downsample window into canvas (overwrite overlaps)
            small = cv2.resize(win, (tile // pv_scale, tile // pv_scale),
                               interpolation=cv2.INTER_AREA)
            py, px = y0 // pv_scale, x0 // pv_scale
            preview[py:py + small.shape[0], px:px + small.shape[1]] = \
                small[:preview.shape[0] - py, :preview.shape[1] - px]
            done += 1
            if done % 200 == 0:
                free = shutil.disk_usage(REPO).free // 1024**2
                print(f"  {done}/{ncols*nrows} tiles, "
                      f"{total_cand} candidates, disk free {free}MB")
                if free < 300:
                    raise SystemExit("Disk <300MB — aborting safely.")
    with open(FULL / "tile_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    (FULL / "tile_meta.json").write_text(json.dumps(
        {"W_orig": W, "H_orig": H, "tile_w": tile, "tile_h": tile,
         "tile": tile, "pad_left": 0, "pad_right": 0, "pad_bottom": 0,
         "overlap": overlap, "step": step, "ncols": ncols, "nrows": nrows,
         "n_tiles": len(index),
         "src": Path(SRC).name,
         "origin_lon": 29.372753834026465,
         "origin_lat": -16.443032455108956,
         "pixel_scale_deg": 8.983152841195215e-05}, indent=2))
    cv2.imwrite(str(FULL / "preview_16x.jpg"), preview[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Done: {len(index)} tiles, {total_cand} candidates.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.15)
    p.add_argument("--quality", type=int, default=82)
    a = p.parse_args()
    main(a.tile, a.overlap, a.quality)
