# Keane Showroom L-Corner — CAD Reconstruction

Five iPhone photos of the showroom vignette → metric camera solve → fitted parametric layout → **81 watertight, modular CAD parts** with pivots, OBBs, UVs, PBR material slots and an assembly manifest.

## Results

| Check | Result |
|---|---|
| Photos registered (SfM) | 5 / 5, 0.91 px mean reprojection error |
| Metric scale | from the sink's price tag (Nantucket FCFS3018W-OSD, 30 in apron), ±1.24 % |
| Parts | 81 (78 fabrication + 3 reference walls/floor) |
| Watertight / manifold / consistent winding | 81 / 81 |
| Interpenetration between parts | 0 mm³ (158 pairs tested) |
| Door/drawer reveals | 3.175 mm (1/8 in) on all 26 adjacent pairs, 0 shared vertices |
| UV overlap | 0 px |
| Model → photo fit (median edge distance) | 3.8 – 7.6 px full-res on the 4 sharp photos (≈ ¼–½ in) |
| STEP | 81 named B-rep solids, re-import clean |

## Why it isn't a raw SuGaR mesh
SuGaR needs a CUDA GPU and roughly 100–300 overlapping frames. This job had 5 photos and no GPU, so a splat-derived mesh would have been mostly holes and floaters. The pipeline used the photos for what they *can* support — exact camera poses and metric measurements — then fit clean parametric geometry to them and verified it by projecting the model back into every photo. The SuGaR route is packaged and ready (`sugar_model/`) for a proper video capture.

## Agents
- **A · SfM & pose calibration**: pycolmap solve, Manhattan alignment, sink-based scale, measurements, coverage report → `sfm/`
- **B · SuGaR kit**: GPU setup + run scripts with flags checked against the SuGaR repo, an iPhone capture protocol, and a CPU post-processor (floater pruning, RANSAC planes snapped to 90°, planar decimation, layer labels) that passes its synthetic tests → `sugar_model/`
- **C · Layout fit & snapping** (orchestrator): measurements snapped to 1/8 in and standard cabinet sizes, 90° corners → `geometry/layout_spec.json`
- **D · Mesh extraction & validation**: deterministic builder, segmentation, pivots, OBBs, UVs, materials, manifest, STEP, validation → `cad_output/`, `export/`

## Folder map
```
cad_output/
  assembly.json            master manifest: transform, pivot, OBB (mm), parent/children, material slots, swing/slide data
  components/*.obj|.stl    one standalone part per file, local mm, pivot at origin (+ materials.mtl)
  components/reference/    walls + floor (not for fabrication)
export/
  kitchen_world.obj/.mtl   all parts, world mm, Z-up, grouped layer/id
  cabinets|drawers|countertops|hardware|fixtures|backsplash|reference.stl
  kitchen_world.ply/.glb   vertex-colour PLY; glTF with PBR + parent nesting
  kitchen.step             B-rep solids
  kitchen_union_watertight.stl   single welded solid (scale models / printing)
  validation_report.json
sfm/          cameras.json, measurements.json, sparse_points.ply, REPORT.md, qa/
geometry/     layout_spec.json (the editable source of truth)
verify/       model-over-photo overlays, alignment scores, albedo estimate
sugar_model/  GPU kit, CAPTURE_PROTOCOL.md, postprocess/, tests/
tools/        make_spec.py → build_cad.py → overlay.py / albedo.py / build_viewer.py
viewer/       keane_showroom_twin.html (self-contained 3D viewer)
```

## Conventions
- **World**: millimetres, Z up. Origin = inside corner at the floor. Sink wall (W1) tile face = plane y = 0; right wall (W2) tile face = plane x = 0.
- **Component local frame** (OBJ/glTF style, right-handed): +X = width to the viewer's right, +Y = up, +Z = out of the front.
- **Pivots**: cabinet boxes, slabs and trim at the rear-bottom-left corner. Doors on the vertical hinge line (front face, hinge edge). Drawers at the slide-axis centre of the back face. Pulls at the mounting footprint centre.
- **Materials**: `mat_cabinet_primary`, `mat_countertop_surface`, `mat_hardware_finish` (+ fireclay, tile, faucet, hose, reference). The geometry has no baked lighting. Base colours were sampled from the photos and white-balanced against the white sink. Countertop, tile and pull colours got a brightness correction because shadows bias them dark.

## Rebuild after an edit
The scripts use the build workspace's absolute paths (`/home/claude/kitchen/...`, with `geometry/`, `export/`, `sfm/` under `artifacts/`). Change the path constants at the top of each script to run them elsewhere.
```
python3 tools/make_spec.py      # edit dimensions here
python3 tools/build_cad.py --spec artifacts/geometry/layout_spec.json   # ~12 s, regenerates everything
python3 tools/overlay.py        # re-check against the photos
```

## Photoreal studio viewer
`viewer/keane_showroom_studio.html` is a material configurator modelled on how car configurators (e.g. Porsche) present a build. It uses the same GLB. The CAD files and `keane_showroom_twin.html` are unchanged.
```
python3 tools/build_studio.py   # repo-relative paths; embeds GLB + Countertops/*.png -> viewer/keane_showroom_studio.html
```
- **Geometry, rebuilt at load inside each part's CAD envelope**: Shaker doors and drawer fronts (2-1/4 in stiles and rails, 1/4 in recessed panel, 1.6 mm edge radius). Quartz slabs re-extruded from the CAD top-face outline with a 1.5 mm eased edge, sink notch included. Bar pulls on square posts with radiused edges. Carcass and trim boxes get 1.5 mm rounded edges. Pivots, door swings and drawer travel are untouched.
- **Materials**: slab photos mapped at true 126 × 63 in slab scale. Each run is cut from a different area of the slab, and edge faces are unfolded so veins continue over the edge. Cabinet paint is a two-layer satin: a pigment base plus a smoother top coat. The film is visually smooth. Only faint grain shows, and only in raking light: vertical on stiles and panels, horizontal on rails and drawer fronts. Most of the realism comes from light on the profiles: highlight lines on the edges and shadow in the recesses. An earlier blotchy "orange-peel" bump map was removed because it read as hammered leather. Backsplash is procedural glazed 3 × 12 in tile (W1 vertical stack, W2 running bond) with colour, normal and roughness maps. Fireclay and quartz use clear-coat.
- **Lighting**: a procedural HDR "light room" (Showroom / Whiteroom / Evening) lights both modes. The live view adds a soft-shadow key light, GTAO ambient occlusion and Khronos PBR Neutral tone mapping. **Photo render** path-traces (three-gpu-pathtracer 0.0.23) with an area key light and an edge-preserving denoise that fades as samples converge. It reaches about 4 samples/s at 1080 × 578 on an M1 Pro and caps at 400 samples. Its shader compiles in the background about 2.5 s after load, because the first compile takes about 30 s on Chrome/ANGLE-Metal.
- **Finish library** (`materials/`, sources and names in `materials/manifest.json`, downloaded 2026-10-09):
  - *Countertops*: 34 Vadara quartz designs (vadara.uk/designs) are mapped as full 126 × 63 in slabs; 4 of them come from `Countertops/*.png`. 48 MSI granites (msisurfaces.com/granite-countertops) are tiled from the maker's close-up. **Its 24 × 24 in scale is an assumption.** The stone tiles are made seamless by offset-and-blend, not mirroring, to avoid symmetry lines.
  - *Backsplash*: glazed tile laid vertically (3 × 12 in stack) or horizontally (12 × 3 in running bond), or a full-height slab cut from the same texture as the countertop.
  - *Cabinets*: showroom paint plus 38 Decore DRS131 finishes (23 solid, 16 woodgrain). DRS131 is a one-piece RTF door, so Decore woodgrain runs vertically on doors and horizontally on drawer fronts with no rail joints. Embossed foils (Taction, Deep Grain, Groovz, Portuna) get a relief map taken from the printed grain. The chip scale (~10 in) is assumed. Woodgrain chips are mirrored 2 × 2, which reads like book-matched veneer.
  - Textures are written to `viewer/textures/<id>.js` (28 MB) and load only when picked, so **keep the `textures/` folder next to the HTML**. The page still opens directly from disk.
- **Interface** (modelled on the Porsche Car Configurator):
  - *App bar*: Share build, Summary, and Photo render as the primary action.
  - *Stage*: pinned while the options scroll. A glass control pill holds Adjust scene, Open fronts, download and fullscreen, and a chip shows Live or Photo progress.
  - *Views*: a strip of picture thumbnails of every camera, re-rendered whenever a finish changes.
  - *Options*: accordion cards for Countertop, Backsplash, Cabinets and Hardware. Brands are sub-dropdowns and families are tiers (Vadara collections, MSI tone bands, Decore sheen families). The selected name shows under its tier and in the card header.
  - *Swatches*: rounded frames with a gloss sheen scaled to the finish, a ring on the selected one, and a tooltip with name, code, brand and family.
  - *Backsplash*: two picture option cards.
  - *Search*: covers every finish.
  - *Camera*: follows the section being edited (Countertop → high view into the L-corner, Backsplash → straight-on sink wall, Cabinets and Hardware → drawers).
  - *Summary dialog*: a Change link per row.
  - *Build link*: the whole build lives in the URL hash, so a copied link reopens it exactly.
- Camera presets include the five solved photo cameras, so renders can be compared directly against `verify/` photos.

## Assumptions to confirm with a tape measure
- **Countertop height measured 36.75 in**, not 36. The cabinet widths support the scale, so check this one with a tape.
- **R-run drawer-over-door base measured 22-5/8 in** with a 3-1/2 in filler, which is not a stock width. A 24 in base with a 2-1/4 in filler fits the run length equally well.
- **Counter overhang** measured 1-3/4 in on the sink run and 1/2 in on the right run.
- **Hidden areas**: the blind-corner base box was never seen, so it is assumed. Carcasses are solid boxes, not panel-level.
- **Narrow right-run upper**: no pull is visible, so its hinge side is assumed.
- **18 in base**: the lower front has a horizontal pull, so it is modelled as a pull-out.
- **Excluded as merchandise**: tile samples, brochures, the binder and the disposal display.
- **Door swing limits** (collision sweep): door_01 stops at 85° (its pull hits the sink apron), door_11 at 80° (inside corner), door_15 at 90° (pantry). All other doors open to 110°.

## Next step for scan-grade geometry
Shoot a 2–3 minute 4K video following `sugar_model/CAPTURE_PROTOCOL.md`, run `sugar_model/run_pipeline.sh` on a rented 24 GB GPU (about $1–3 per kitchen), and compare the cleaned mesh against this parametric model.
