# cad_clean test results (actual run, 2026-10-07)

**Environment:** container with 2 CPUs and 7 GB RAM, Python 3.13, numpy 2.5.3, scipy 1.18.1, trimesh 5.1.1, manifold3d, shapely 2.2. No GPU, no open3d dependency. (`pip install open3d` succeeded on 3.13, but the code doesn't use it.)

**Command:** `python3 tests/test_cad_clean.py` (seed 7) exited 0 with **ALL PASS**. Wall time 20.4 s, of which cad_clean itself took about 19 s.

## Synthetic input (`tests/make_synthetic.py`)
* **Geometry:** L-shaped kitchen built as an exact union of boxes, plus 15 bar pulls (cylinders).
  * Floor slab and two walls.
  * Base runs: plinth / toe kick, carcass, 19 mm proud door slabs, and a 1.5 in countertop at **36 in** with a 1 in overhang.
  * 12 in uppers at 54–84 in.
* **Tessellation:** 20 mm edges, giving **452,635 faces**.
* **Corruption:**
  * 1 Laplacian iteration (soft creases)
  * 3 mm low-frequency ripples (wavelength 0.5–0.9 m)
  * σ = 1 mm Gaussian noise
  * 10 free-space floaters (blobs and thin slivers) plus 3 one-triangle islands
  * 1 bar pull detached 2 mm from a door, which must be kept
* **Frame:** random 8° rotation, a translation, and a non-metric scale of 1/1234.5, like arbitrary COLMAP units.
* **Ground truth:** all box faces as planes in mm. The test maps every output plane back to truth using the 4×4 transform in the tool's report.

## Checks (seed 7)

| # | Check | Result | Measured |
|---|---|---|---|
| 1 | Floaters removed | **PASS** | 15 components in, 2 kept (room + detached pull). Removed 3 tiny, 3 thin, 7 free-space. Nearest output vertex to any floater centre: 384.5 mm. |
| 1b | Detached pull kept | **PASS** | Output within 1.18 mm of its centre; labelled hardware. |
| 2 | Plane normals within 0.5° of truth | **PASS** | 26/26 planes matched a true surface, with no spurious planes. **Max error 0.0142°**, mean 0.0122°. |
| 3 | Orthogonal planes at 90 ± 0.5° | **PASS** | 223 pairs, max deviation 0.0° (exact, because normals are snapped). |
| 4 | Inlier RMS < 1 mm after snapping | **PASS** | RMS of each plane's output vertices to the **true** plane: **max 0.847 mm** over the 23 planes ≥ 0.05 m². Area-weighted RMS over all 26 planes: **0.337 mm**. Before snapping, the PCA fit RMS was 1.55 mm median, i.e. the noise plus ripple. RMS to the plane's own snapped plane: **0.0 mm**, so every patch is exactly flat. |
| 5 | Metric distances within 2 mm | **PASS** | 77 parallel plane pairs (floor to counter, wall to door front, and so on): max error 1.30 mm, mean 0.355 mm. Scale came from `counter_height=914.4` and was then refined from the fitted planes. |
| 6 | Face count reduced ≥ 80 % | **PASS** | 449,432 to **28,880** faces (**−93.6 %**). All 28 planar patches were re-triangulated, with 0 fallbacks. |
| 7 | Topology | **PASS** | Watertight, consistent winding, 0 non-manifold edges, 0 boundary edges, 0 degenerate faces, 0 duplicate faces. 2 components (room + detached pull). |

Pipeline log (seed 7): Manhattan axes explain 88.6 % of the area. Up axis was chosen by the floor heuristic, with no hint. Frame refinement from plane normals rotated a further 0.0175°.

## Per-plane detail (seed 7)

| id | truth surface | area m² | snapped | angle err ° | RMS fit before snap mm | RMS to truth after snap+projection mm | RMS to own plane mm | layer |
|---|---|---|---|---|---|---|---|---|
| 0 | floor_slab-z | 14.755 | -Z | 0.0118 | 1.58 | 0.328 | 0 | cabinet_carcass |
| 1 | floor_slab+z | 11.511 | +Z | 0.0118 | 1.59 | 0.436 | 0 | floor |
| 2 | floor_slab-y | 10.042 | +X | 0.0142 | 1.58 | 0.308 | 0 | walls |
| 3 | floor_slab-x | 8.817 | -Y | 0.0097 | 1.59 | 0.383 | 0 | walls |
| 4 | left_wall+x | 5.967 | +Y | 0.0097 | 1.57 | 0.134 | 0 | walls |
| 5 | back_wall+y | 4.566 | -X | 0.0142 | 1.55 | 0.242 | 0 | walls |
| 6 | upper_doors+y | 2.130 | -X | 0.0142 | 1.53 | 0.239 | 0 | cabinet_fronts |
| 7 | back_counter+z | 3.058 | +Z | 0.0118 | 1.46 | 0.405 | 0 | countertop |
| 8 | back_doors+y | 1.739 | -X | 0.0142 | 1.45 | 0.153 | 0 | cabinet_fronts |
| 9 | left_doors+x | 1.332 | +Y | 0.0097 | 1.44 | 0.092 | 0 | cabinet_fronts |
| 10 | uppers-z | 0.911 | -Z | 0.0118 | 1.46 | 0.284 | 0 | cabinet_carcass |
| 11 | uppers+z | 0.912 | +Z | 0.0118 | 1.44 | 0.276 | 0 | cabinet_carcass |
| 12 | back_plinth+x | 0.732 | +Y | 0.0097 | 1.50 | 0.060 | 0 | cabinet_fronts |
| 13 | back_wall+z | 0.757 | +Z | 0.0118 | 1.62 | 0.371 | 0 | cabinet_carcass |
| 14 | left_plinth+y | 0.503 | -X | 0.0142 | 1.66 | 0.108 | 0 | cabinet_fronts |
| 15 | floor_slab+y | 0.445 | -X | 0.0142 | 1.57 | 0.653 | 0 | walls |
| 16 | floor_slab+x | 0.418 | +Y | 0.0097 | 1.74 | 0.157 | 0 | walls |
| 17 | back_plinth+y | 0.253 | -X | 0.0142 | 1.66 | 0.771 | 0 | cabinet_fronts |
| 18 | back_counter-z | 0.235 | -Z | 0.0118 | 1.55 | 0.242 | 0 | countertop |
| 19 | back_carcass-z | 0.330 | -Z | 0.0118 | 1.46 | 0.164 | 0 | cabinet_carcass |
| 20 | left_plinth+x | 0.192 | +Y | 0.0097 | 1.46 | 0.847 | 0 | cabinet_fronts |
| 21 | back_counter+y | 0.092 | -X | 0.0142 | 1.74 | 0.245 | 0 | countertop |
| 22 | left_counter+x | 0.069 | +Y | 0.0097 | 1.51 | 0.241 | 0 | countertop |
| 23 | back_doors+z | 0.040 | +Z | 0.0118 | 1.39 | 0.352 | 0 | cabinet_carcass |
| 24 | back_carcass+y | 0.028 | -X | 0.0142 | 1.24 | 0.830 | 0 | cabinet_fronts |
| 25 | left_counter+y | 0.024 | -X | 0.0142 | 1.64 | 1.465 | 0 | countertop |

Planes 23–25 are below the 0.05 m² assertion threshold. A face only 38 mm tall can't average out a 3 mm ripple whose wavelength is 0.5–0.9 m. That limit comes from the data, not from snapping, so these are reported but not asserted. The "floor_slab-z / -x / -y" surfaces are the outside of the synthetic slab and walls. A real scan wouldn't contain them.

## Extra validation
* **Independent scale path.** Running `--scale-ref points=<raw floor point>;<raw point 914.4 mm above>;914.4 --up <rough up, about 10° off>` measured the countertop at **914.47 mm** (truth 914.4). The result was watertight, with 0 non-manifold edges and 0 degenerate faces, at 30,232 faces.
* **Other seeds**, same assertions:

| seed | normals max err | RMS max (≥ 0.05 m²) / area-weighted | metric max err | reduction | topology | verdict |
|---|---|---|---|---|---|---|
| 11 | 0.0061° | 0.818 / 0.222 mm | 1.95 mm | 93.8 % | watertight, 0 degenerate | **ALL PASS** |
| 3 | 0.0141° | **1.547** / 0.421 mm | 1.92 mm | 92.8 % | watertight, but **2 degenerate faces** | **2 checks fail** |

  On seed 3, a 0.15 m² toe kick (102 mm tall) is offset 1.55 mm by the ripple. Two zero-area slivers also survive in residual geometry; the mesh is still watertight and manifold.

## Known limitations (honest list)
* **Ripple limit.** Narrow faces (under about 100 mm in one direction) inherit ripple and blur bias, typically 1–1.5 mm.
* **Degenerate slivers.** On some inputs a few zero-area slivers remain, because there is no edge-flip pass yet. They don't break watertightness.
* **Residual labelling.** The `hardware` and `unclassified` layers come from size heuristics. They contain the real pulls plus leftover rounded-crease fragments: 10,292 and 8,275 faces on seed 7. Planar layers are reliable. Treat the non-planar ones as review layers.
* **Thin reveals.** Steps between parallel planes thinner than the mesh resolution (e.g. the 6 mm door-to-counter reveal) are left as-is rather than snapped.
