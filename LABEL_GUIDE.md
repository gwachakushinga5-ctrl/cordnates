# Label guide — tobacco barn roofs

## Class
- `0 = barn_roof` (only class for pilot). Do NOT box curing kilns, houses, sheds, or shadows alone.

## What a tobacco barn roof looks like
- Distinctive **rectangle**, typically 2:1 to 4:1 aspect ratio (long axis either direction).
- **Metallic**: bright white/light-grey, sharp edges, high contrast vs soil/vegetation.
- **Thatched**: dull brown/grey-tan, lower contrast, matte texture — still rectangular.
- Context cues: often near **cured kilns** (small square block adjacent), **farm tracks** (thin pale lines), cleared compounds. Use as hints, not as box targets.

## Box rules
1. Tight to the **roof drip-edge** — include full roof, exclude shadow skirt and surrounding ground.
2. One box per roof. Split adjoining/double barns into two boxes with a shared edge.
3. If roof is cut by a tile edge, box the visible part; overlap merging (`predict_pilot.py --merge`) rejoins it on the full strip.
4. Occlusion (tree over <20% of roof): still box full inferred rectangle. If >50% hidden → flag `occluded` in Roboflow comment, keep box.
5. Small outbuildings that are square (< ~8px on tile) → skip unless clearly barn-like rectangle + kiln/track context.
6. When unsure between house vs barn: barn = isolated in field/compound + kiln nearby + no garden/fence pattern. Mark unsure in Roboflow, don't delete.

## YOLO format
`class cx cy w h` normalized 0–1 per tile, e.g. `0 0.512 0.498 0.210 0.132`.
