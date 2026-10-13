# Top Knobs hardware library

Source: https://www.topknobs.com/products/knobs-pulls.html (every product in the category, scraped 2026-10-13).

| Step | Script | Output |
|---|---|---|
| Scrape | `tools/fetch_topknobs.py` | `catalog.json` (raw: products, SKUs, spec tables, photo URLs), `pages/` cache |
| Normalise + photos | `tools/topknobs_prep.py` | `products.json`, `img/ref/`, `img/thumb/`, `img/finish/` |
| Pull geometry | `tools/hw_pulls.py` | `geom/<slug>.json` for two-screw products |
| Knob geometry | `tools/hw_knobs.py` | `geom/<slug>.json` for one-screw products |
| Finishes | `tools/hw_finishes.py` | `finishes.json` |
| QA renders | `tools/hw_preview.py` | `qa/<slug>.png`, contact sheets |
| Studio | `tools/build_studio.py` | `viewer/hardware/<slug>.js`, `viewer/hardware/thumbs/`, catalog embedded in the page |
| Studio check | `tools/hw_sweep.mjs` | `qa/sweep.json` (every size on every mount vs spec, finishes, doors / drawers), `qa/shots/` (`--shots`) |

## products.json

`products[]`: `slug`, `name`, `url`, `collection`, `kind_hint` (`pull` when the SKUs list a center-to-center, else `knob`),
`finishes` (labels exactly as Top Knobs prints them), `thumb`, and `sizes[]`:
`key`, `label`, `c2c_mm`, `length_mm` (overall length), `width_mm`, `proj_mm` (projection), `diam_mm`, `base` (raw "Base
Diameter" text), `metal`, `specs` (the full raw spec table), `ref_sku`, `ref_finish`, `ref_img` (local paths of every photo
of the reference SKU, relative to this folder), `skus` (`{finish: sku}`).

Photo `_0` of a pull is normally a side elevation (length across, projection up, door surface at the bottom). Photo `_0`
of a knob is normally a 3/4 view from slightly above. Later photos are often room scenes. All are 490 × 490 on white.

## Geometry: `geom/<slug>.json`

```json
{ "slug": "brixton_pull", "type": "bar_pull",
  "sizes": { "<size key>": { "shape": SHAPE, "fit": { "iou": 0.95, "notes": "" } } },
  "notes": "" }
```

`type` is one of `round_knob`, `square_knob`, `oblong_knob`, `t_knob`, `ring_pull`, `drop_pull`, `bar_pull`, `cup_pull`,
`appliance_pull`, `tab_pull`, `finger_pull`, `other`.

**Local frame, millimetres.** Origin = the mounting point on the cabinet front surface (for two-screw pulls, the
midpoint between the screw holes). +X runs along the long (center-to-center) axis, +Y comes out of the door (projection),
+Z runs across. The door surface is the plane Y = 0 and nothing is at Y < 0. Dimensions must match the spec table:
overall length along X, projection along Y, width along Z (knobs: diameter or length × width in X and Z).

Pieces drawn from a front photo (cup, drop and plate pulls) take +Z as *down the door* for a horizontally mounted pull
(right-handed, X right, Y out) and say `+Z = down the door` in their notes; the studio keeps them horizontal on doors too.
Two deliberate exceptions to "dimensions along X / Y / Z": edge tabs (notes `edge tab`) have their origin on the middle
of the front's top edge and hang their Projection down along +Z with Width as the plate thickness along Y; hanging rings
(notes `ring hangs towards +Z`) put their spec length along Z and the ring width along X.

SHAPE is one of:

- `{"kind": "extrude", "outline": [[x, y], ...], "holes": [[[x, y], ...]], "depth": mm, "bevel": mm}`
  The side elevation (X × Y) as one closed polygon, counter-clockwise, extruded symmetrically about Z = 0 through `depth`,
  with every edge rounded by `bevel` (0 ≤ bevel ≤ depth / 2; bevel ≈ depth / 2 reads as a round bar). The open gap
  between the bar and the door is a concavity of the outline, not a hole. `holes` is optional.
- `{"kind": "lathe", "profile": [[r, y], ...], "section": "round" | "square" | "hexagon" | "octagon", "stretch": [sx, sz], "rotate": deg}`
  A profile revolved about the Y axis, from the base on the door (y = 0) to the crown (r = 0 at the top). For `square`,
  `hexagon` and `octagon`, r is the half-width across flats; flats face Z (and X, except the hexagon, whose corners sit
  on X). `stretch` (default [1, 1]) scales X and Z after revolving, for oblong knobs. A number `rotate` (default 0) turns
  the result about Y; a list `[rx, ry, rz]` is the part rotation of a group part, as below.
- `{"kind": "group", "parts": [SHAPE + {"offset": [x, y, z], "rotate": [rx, ry, rz]}, ...]}`
  For hardware that is several solids (ring pull = backplate + ring, cup pull = plate + cup, knob with a separate rosette).
  Rotations are degrees, XYZ order, applied before the offset.

Any SHAPE may carry `"mat": "<id>"` to use a secondary material from `finishes.json` (`glass`, `crystal`, `ceramic`,
`leather`, `wood`, ...), or `"mat": "accent"` for the second colour of a two-tone finish. Without it the part takes the
selected metal finish.

**Overrides: `geom/overrides/<slug>.json`.** Hand corrections that survive a re-run of the generators:
`{"type": ..., "params": {...}, "sizes": {"<key>": {"shape": SHAPE, "spec_fix": {...}, "notes": ""}}, "notes": "why"}`. A full
`shape` for a size wins outright; its `spec_fix` (e.g. `{"proj_mm": 34.92}`) records dims the shape uses instead of a
catalogue value judged wrong and is copied into `fit.spec_fix`, as the generators do for their own repairs, so QA compares
against it. `params` (product-wide, or per size under `sizes.<key>.params`) are generator-specific knobs (photo index, section, bevel, crop, ...) documented in each generator's
docstring. Both generators accept `--only slug[,slug...]`.

## Finishes: `finishes.json`

```json
{ "finishes": { "<label exactly as in products.json>": {
      "id": "tk_<slug>", "label": "...", "family": "Polished | Brushed | Satin | Matte | Bronze | Brass | Black | Specialty",
      "color": [r, g, b], "roughness": 0.3, "metalness": 1, "clearcoat": 0, "anisotropy": 0,
      "swatch": "#rrggbb", "samples": 8, "notes": "" } },
  "secondary": { "<id>": { "label": "...", "color": [r, g, b], "roughness": 0.1, "metalness": 0,
      "transmission": 0, "ior": 1.5, "thickness": 0 } } }
```

Two-tone labels ("Flat Black and Honey Bronze") get `"two_tone": ["Flat Black", "Honey Bronze"]`: the first is the body,
the second is what `"mat": "accent"` parts take. `color` is linear RGB in 0..1 (three.js `color.setRGB(..., LinearSRGBColorSpace)`), `swatch` is sRGB for UI chips.

## Studio

`viewer/hardware/<slug>.js` holds `window.__hwData("<slug>", {"type": ..., "sizes": {"<key>": SHAPE}})` and loads only
when the product is picked. Thumbnails are `viewer/hardware/thumbs/<slug>.jpg`. The page embeds the catalog
(products, sizes, SKUs, finishes) as `CAT.hardware`. Keep `hardware/` next to the HTML, as with `textures/`.
