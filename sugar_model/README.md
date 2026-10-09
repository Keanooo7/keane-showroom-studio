# SuGaR kitchen kit: video to metric, segmented CAD mesh

This kit takes a phone video of a kitchen and produces a watertight, Z-up, metric, plane-snapped mesh. The mesh is split into layers: floor, walls, countertop, cabinet fronts, carcass and hardware. It also writes a JSON file listing every plane and crease edge.

```
iPhone video ──ffmpeg──► 150-300 frames ──COLMAP (convert.py)──► camera poses
        ──SuGaR train_full_pipeline.py──► raw mesh (≈1M verts, wavy, floaters)
        ──postprocess/cad_clean.py (CPU)──► clean CAD mesh + layers + plane/edge report
```

| File | What it is |
|---|---|
| `CAPTURE_PROTOCOL.md` | One-page checklist for installers and salespeople. **Read this first**, because capture quality decides everything. |
| `setup_gpu.sh` | One-time setup on an Ubuntu + NVIDIA box: conda env, SuGaR, rasterizer, nvdiffrast, COLMAP. |
| `run_pipeline.sh` | Runs the whole job: frames, COLMAP, SuGaR, cad_clean. Logs go to `jobs/<name>/logs/`. |
| `postprocess/` | Pure numpy/scipy/trimesh/manifold3d CPU tools. No GPU and no open3d needed. |
| `tests/` | Builds a synthetic kitchen and checks the post-processor end to end. Measured numbers are in `tests/RESULTS.md`. |

The current 5-photo capture is **far too few frames** for SuGaR or COLMAP. Recapture following `CAPTURE_PROTOCOL.md` (one 3–5 min video).

## 1. Hardware, time and cost

* **GPU:** NVIDIA with CUDA and **24 GB of VRAM recommended**: RTX 3090/4090, A5000, L4 or A10G. Use 40–48 GB (A6000, L40S, A100-40G) for 4K frames or very large rooms. The SuGaR README states no VRAM figure. It only warns that its 3DGS wrapper "may not be the most optimal one for memory usage". The 24 GB number comes from the original 3D Gaussian Splatting repo's paper-quality guidance. On 12–16 GB cards, use `--poly low`, about 150 frames, and frames downscaled to around 1600 px.
* **Software:** Ubuntu 20.04/22.04, NVIDIA driver ≥ 520, conda, and the **CUDA toolkit 11.8** (the README requires 11.8). `setup_gpu.sh` puts 11.8 inside the env if the host has a different version.
* **Disk:** about 20 GB for the env, plus about 3 GB per job (the refined Gaussian `.ply` is "+/- 500MB").
* **Runtime per kitchen on one RTX 4090.** These are estimates. Only the mesh-extraction figure is from the README.

| Step | Time |
|---|---|
| ffmpeg, 200 frames | < 1 min |
| COLMAP, exhaustive matching, 200 frames (GPU SIFT) | 10–25 min. CPU SIFT is about 2–3× slower. |
| Vanilla 3DGS, 7k iterations | 5–10 min |
| SuGaR regularized optimization + mesh extraction | about 30 min. README: "extracts a mesh in 30 minutes on average on a single GPU". |
| Refinement | short (2k iterations): a few minutes. medium (7k): about 10–20 min. long (15k): 30+ min. |
| Textured OBJ with nvdiffrast | seconds |
| `cad_clean.py` on CPU, 1M-vertex mesh | about 1–3 min |
| **Total** | **about 1–1.5 h** (medium refinement) |

* **Rental cost (rough, 2026 marketplace rates):** RTX 4090 at about $0.35–0.70/h, A6000 or L40S at about $0.6–1.0/h, A100 at about $1.2–2/h. That works out to **about $1–3 per kitchen** on a 4090, including setup. The env install takes about 20 min once, so keep a persistent volume and reuse it.

## 2. Step by step

```bash
# on the GPU box (RunPod/Vast/Lambda: pick an "Ubuntu 22.04 + CUDA" image, 24 GB GPU, 50 GB disk)
git clone <this kit> kit && cd kit
bash setup_gpu.sh                 # ~20 min, once
# upload the video, then:
bash run_pipeline.sh --video smith.mov --name smith --frames 200 --refine medium --counter-in 36
# results: jobs/smith/cad/  (download this folder)
```

What `run_pipeline.sh` runs. Every command and flag below comes from the SuGaR README or its scripts (fetched 2026-10-07):

```bash
# frames (README "Tips" section 1, verbatim)
ffmpeg -i <video> -qscale:v 1 -qmin 1 -vf fps=<FPS> %04d.jpg     # FPS chosen so frames = 100-300
# poses (README "Tips" section 2): images in <location>/input, then
python gaussian_splatting/convert.py -s <location>                 # [--no_gpu] if COLMAP lacks CUDA
#   if COLMAP made several sub-models: keep the largest in <location>/distorted/sparse/0, then
python gaussian_splatting/convert.py -s <location> --skip_matching
# SuGaR (README "Quick Start"; flags from train_full_pipeline.py argparse)
python train_full_pipeline.py -s <location> -r dn_consistency --high_poly True \
       --refinement_time medium --export_obj True --export_ply True
# CAD clean (this kit)
python postprocess/cad_clean.py <refined_mesh.obj> jobs/<name>/cad --units mm \
       --scale-ref counter_height=914.4 --colmap-images <location>/sparse/0/images.bin
```

SuGaR flags as documented: `-s/--scene_path`, `-r/--regularization_type` (`"dn_consistency"` recommended, `"density"`, `"sdf"`), `--gs_output_dir` (reuse a vanilla 3DGS checkpoint and skip its 7k iterations), `--high_poly True` (1,000,000 verts, 1 Gaussian per triangle), `--low_poly True` (200,000 verts, 6 Gaussians per triangle), `--refinement_time short|medium|long` (2k/7k/15k iterations, default long), `-t/--export_obj` (default True), `--export_ply` (default True), `--square_size` (UV texels per triangle pair, default 8), `--eval` (default True), `--white_background`, `--gpu`. The script also accepts `-l/--surface_level` (0.3), `-v/--n_vertices_in_mesh`, `-b/--bboxmin`, `-B/--bboxmax` and `--postprocess_mesh`.

SuGaR outputs, written relative to the SuGaR repo:
* `output/coarse_mesh/<scene>/sugarmesh_*.ply`: Poisson mesh, geometry only.
* `output/refined_mesh/<scene>/*.obj` plus a texture PNG: the refined mesh. **This is what `cad_clean` uses by default.** It welds UV-seam duplicate vertices on load.
* `output/refined_ply/<scene>/*.ply`: refined Gaussians, for 3DGS viewers.

`<scene>` is the last folder name of `-s`, so `run_pipeline.sh` names the dataset folder after `--name`.

## 3. The post-processor (`postprocess/cad_clean.py`)

```bash
python postprocess/cad_clean.py in.ply|in.obj out_dir --units mm \
   [--scale-ref counter_height=914.4 | "points=x1,y1,z1;x2,y2,z2;D" | factor=s] \
   [--up 0,-1,0 | --colmap-images sparse/0/images.bin] [--plane-tol 8] [--snap-deg 5] \
   [--min-plane-area 5000] [--conf-thresh 0.2] [--no-decimate]
```

| Stage | Module | What it does |
|---|---|---|
| a | `floaters.py` | Optional confidence filter on per-vertex `opacity`, `confidence`, `quality` or `density`. Keeps the room component. Drops tiny islands, thin slivers and blobs in free space. **Keeps** small parts within about 0.5 % of the scene size from the surface, such as a pull that Poisson detached, as hardware candidates. |
| e | `scale.py` | Metric scale, chosen in this order: `points=` (two points picked on the raw mesh, e.g. ArUco board corners or tape-measure ends, plus their true distance), `counter_height=` (floor-to-countertop planes, measured robustly from thousands of points; default 36 in, with a warning), or `factor=`. |
| b | `manhattan.py` | Finds the three dominant orthogonal directions from area-weighted face normals and refines them with Kabsch. Up comes from `--up`, then COLMAP camera orientation, then a heuristic (the floor is the largest extreme plane facing into the scene). Y points from the largest wall into the room, and the frame stays right-handed. |
| c | `planes.py` | Iterative 1-point-plus-normal RANSAC on area-weighted face centroids. Inliers are split into connected patches, then refit with PCA. The global frame is refined with Kabsch using the precise plane normals. Normals within 5° of an axis are **snapped**, so orthogonal planes meet at exactly 90°. Offsets are re-solved from interior vertices. Rejected as fake planes: crease strips, near-axis (5–20°) step-bridging fits, and bridging planes. Rounded crease strips and noise islands are absorbed. Every vertex is projected to its plane, edge line or corner point. |
| d | `decimate.py` | For each planar patch: trace boundary loops, drop collinear crease vertices that both neighboring patches share, and re-triangulate with manifold3d's constrained triangulator, which adds no Steiner points and keeps edges. Self-intersecting boundary rings are peeled, and the patch is retried up to 6×. The result stays **watertight with no T-junctions**. Non-planar detail (handles, hinges, faucet) is left untouched. |
| f | `labels.py` | Assigns layers: `floor` (up-facing at z≈0), `countertop` (up-facing at 34–38.5 in plus its edge band and underside), `walls` (vertical, ≥ 1.5 m tall), `ceiling`, `cabinet_fronts` (vertical, facing the same way as a wall), `cabinet_carcass`, `hardware` (non-planar groups under 30 cm, small separate parts) and `unclassified`. |
| g | `topology.py` | Reports watertightness, boundary and non-manifold edges, winding, degenerate and duplicate faces, Euler number and volume, before and after. Collapses only true zero-length edges, and only when the link condition holds. |

**Outputs in `out_dir`:**
* `*_clean.ply`, `*_clean.stl`: the full mesh, in output units, Z-up, floor at z = 0, back wall at y = 0.
* `*_layers.obj`: one object per layer. Import it into Fusion 360, SketchUp, Rhino or Blender.
* `*_<layer>.stl`: one file per layer.
* `*_face_plane_ids.npy`, `*_face_layers.npy`: the plane id and layer of each face.
* `*_report.json` contains:
  * `transform_raw_to_output` (4×4, including scale)
  * `planes`: normal, offset (`n·x = d`), area, snapped axis and angle, inlier RMS before snap, after snap, and final
  * `edges`: plane–plane crease segments with endpoints, length and dihedral angle
  * `patch_loops`: CAD face outlines
  * `topology_before` and `topology_after`, plus the floater and decimation stats

**ArUco scale (recommended for field use; documented here, not automated):**
1. Detect the board in 3 or more undistorted frames with `cv2.aruco.ArucoDetector`.
2. Triangulate each corner with the COLMAP poses in `sparse/0` (DLT across views).
3. Pass two triangulated corners and the printed edge length as `--scale-ref points=...;...;D`.

The counter-height scale is a good fallback, but it assumes the installer measured the height correctly.

## 4. Known failure modes in kitchens, and fixes

| Problem | Symptom | Fix |
|---|---|---|
| **White, textureless doors** (painted shaker, white gloss) | COLMAP can't match frames, the model splits into sub-models, or doors come out as holes or blobs | Lock exposure. Add blue painter's-tape dots. Include countertop and floor texture in every frame. Use `-r dn_consistency`. Add more frames from the middle pass. |
| **Glossy quartz or granite, lacquer** | View-dependent highlights become floaters or dents in the counter | Close blinds. Avoid point lights in frame. Film at an angle. `cad_clean` snaps the counter plane and absorbs dents up to 2.5× plane-tol (20 mm by default). |
| **Stainless appliances, faucet, mirrors, glass doors** | Ghost geometry "behind" the surface, mirrored cabinets | Film these briefly and at an angle. Free-space floaters are pruned. Ghost geometry fused to the surface stays `unclassified`, so mask it out or model the appliance from catalog dimensions. |
| **Thin handles, edge-pulls, hinges** | Broken or detached bars, or bars fused into the door | Do the close detail pass from 2 angles. Detached bars are kept as `hardware` because they are near the surface. Treat hardware as a position marker and use catalog CAD for the real part. |
| **Reveals and gaps under 1 cm** (door-to-counter gap, door seams) | Steps thinner than the mesh resolution | `cad_clean` leaves these vertices unprojected (it reports "parallel-plane conflicts") instead of collapsing them into zero-thickness sheets. Get reveal dimensions from shop standards, not from the scan. |
| **Holes** (inside corners, behind the fridge, toe kicks) | Open borders and a non-watertight raw mesh | Do the low pass and corner arcs. In SuGaR, lower `vertices_density_quantile` (`sugar_extractors/coarse_mesh.py` line 43). `cad_clean` keeps open borders and reports them. |
| **Ellipsoid bumps on the surface** | Lumpy flat panels | README fix: lower `poisson_depth` (line 42 of `coarse_mesh.py`) to 7–8. `cad_clean` then flattens the planar regions. |
| **Few or blurry frames** (like the current 5 photos) | COLMAP registers too few images, and SuGaR diverges | Recapture. 150–300 sharp frames from 3 height passes. |
| **Wrong up axis** (e.g. island-only scan, little floor visible) | Output lies on its side | Pass `--colmap-images` (done by `run_pipeline.sh`) or `--up x,y,z` in raw coordinates. |

## 5. How the outputs feed the shop

The `planes` list in `*_report.json`, together with the `edges` list, describes the room as exact boxes. Countertop height, wall-to-front depth, run lengths and upper heights come straight from the plane offsets. They are accurate to about 1 mm on synthetic data; real scans are typically ±3–5 mm. Always verify one tape measurement per run. The `*_layers.obj` file goes into CAD as reference geometry for cabinet layout and fabrication drawings.

## 6. Tests

```bash
python3 tests/test_cad_clean.py            # ~25 s on 2 CPUs; also: pytest tests/
python3 tests/test_cad_clean.py --seed 11 --keep-dir /tmp/cc   # keep outputs
```
See `tests/RESULTS.md` for the measured numbers.
