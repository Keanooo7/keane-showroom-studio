# Agent A — Structure-from-Motion & Pose Calibration

## Registration
- **5/5 images registered** in one pycolmap 4.2.1 model (CPU): 6,490 points, mean reprojection error **0.91 px** at 0.75× working scale (~1.2 px full-res), mean track length 2.26.
- Points per image: 02 → 3938, 03 → 3786, 04 → 2940, 05 → 2774, 01 → 1228.
- SIFT with domain-size pooling + affine shape, 10k features, 2268×3024 inputs; exhaustive + guided matching; relaxed incremental mapper. Verified inliers per pair: 271 (01–05) to 2155 (02–05).
- Cameras: 02–05 share one SIMPLE_RADIAL (f = 3002.4 px full-res, prior 3028; k1 = 0.0197). 01 has its own (f = 3024.4, k1 = 0.0073) — not cropped, but portrait-mode blur limits it to near-sink geometry.
- Full-resolution matching (≈50k features/image) ran out of memory twice; a 0.5× run (`work_v1`) also registered 5/5 and is kept for reference.

## Frame & scale
- Z up; **W1** (sink wall) = plane y = 0, **W2** (right wall) = plane x = 0, both at the backsplash **tile face** (drywall ≈ 0.4 in behind); origin = inside corner at the floor. Axes from RANSAC planes, refined with vanishing directions from ~2,500 line segments (orthogonal within 0.1°, residual tilt ≈ 0.5°).
- **Scale from the sink**: the price tag reads **Nantucket Sinks FCFS3018W-OSD** (30 × 18 × 9-7/8 in). The four apron edges were fitted as 3D lines from 3 views each; apron width = 30.0 in ⇒ 10.326 in per model unit, **±1.24 %** (the left outer edge is ambiguous between 29.65 and 30.35 in).
- Cross-checks (reported, not forced): apron height 9.92 in vs 9.875 spec; ten cabinet widths vs standard sizes, mean ratio 1.0002; upper box heights 41.96 / 29.76 in vs 42 / 30; countertop height measures **36.75 in** (not 36).

## Measurements (inches, ±1σ including scale)
| Item | Value |
|---|---|
| Countertop top | 36.75 ± 0.48 |
| Countertop thickness L / R | 1.25 ± 0.13 / 1.59 ± 0.11 |
| Counter front, L run / R run | y = 26.54 (1.74 overhang) / x = 25.29 (0.60 overhang) |
| Upper bottoms (42 in uppers) | 55.12 ± 0.69 |
| Short upper over sink, bottom | 67.38 ± 0.84 (12.20 above the others) |
| Upper box tops (crown underside) | 97.14 / 97.05 ± 1.2 |
| Crown top L run / R run | 100.60 / 98.77 ± 1.25 |
| Pantry height incl. cap | 99.13 ± 1.24 (cap underside 96.58) |
| Toe-kick top | 4.70 ± 0.16 |
| Sink apron | x 42.10 → 72.10, z 25.82 → 35.74, front y = 26.86 (2.06 in proud of doors) |

**Right run** (doors at x = 24.70): filler 24.81–28.34 · drawer-over-door base 28.34–50.88 (22.55) · 4-drawer 50.88–72.04 (21.15) · counter end 73.14. Uppers (x = 12.65): 12.19–23.12 · 23.83–47.48 · 47.97–71.55.

**Left run** (doors at y = 24.81): corner filler 24.70–26.87 · 3-drawer 26.87–41.77 (14.90) · sink base –72.46 (30.69) · drawer base –90.18 (17.72) · pantry 90.45–111.43 (20.98). Uppers (y = 12.40): blind corner 0–23.87 · 23.87–42.26 · 42.26–72.50 (30 in over sink) · 72.50–90.45.

## Coverage
Sparse points per region: countertop 1641 · W1/W2 backsplash 939/321 · base fronts 341/276 · upper fronts 175/178 · crown 60 · counter front edges 119 · counter end 17 · sink apron/bowl 14/22 · upper undersides 20 · **upper tops 4** · toe kicks 25 · floor 88 · pantry front 16 · **pantry side 1** · visible side panels 7.

Never seen: drawer undersides, cabinet interiors, blind corner boxes, the sink's right end, corner-facing cabinet sides (every camera stood on the room/pantry side). Behind the sample tiles: 1–10 points per 4-in strip vs 58–80 on the tiles.

**For a proper capture:** 30–60+ frames in two height bands with obliques from both ends of each run, a ChArUco board or tape for scale, texture targets on doors, step-ladder shots of cabinet tops, one shot under the uppers and one into a toe kick, no portrait mode, tape checks of counter height, both crown heights and the pantry cap.

## Caveats
Scale dominates large absolute heights (≈ ±1.2 in at 97 in); widths within one image carry only 0.1–0.4 in geometric error. Counter front edges, R-run crown, pantry cap and several edge positions are single-view (ray ∩ fitted plane).

## Files
`measurements.json` (all values in in + mm with σ, method, cross-checks) · `cameras.json` (K, k1, R/t, cam→world, centres; inch frame) · `stats.json` · `coverage.json` · `sparse_points.ply` (coloured, inch frame) · `sparse_native/`, `sparse_aligned_inches/` (bin + txt) · `qa/` (fitted lines over all photos, top view, wall elevations) · `scripts/` (all picks in `features.py`) · `work_s075/` (adopted model), `work_v1/` (reference).
