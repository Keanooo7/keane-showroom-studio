"""Agent C: layout fit -> explicit, fully-resolved part list (layout_spec.json).

Frame (inches, from Agent A's metric SfM): Z up, floor z=0.
  W1 = sink wall, backsplash tile face at y=0, room y>0. Left run runs along +x, fronts face +y.
  W2 = right wall, tile face at x=0, room x>0. Right run runs along +y, fronts face +x.
  Drywall is 0.375 in behind the tile face (x=-0.375 / y=-0.375).
Every value is either MEASURED (Agent A SfM, +/- sigma) or SNAPPED to a standard size within that sigma.
"""
import json, math

T = 0.375          # tile + setting bed -> drywall plane at -T
FT = 0.75          # door / drawer front thickness
REV = 0.125        # reveal between adjacent fronts (1/8")
TK_H, TK_D = 4.625, 3.0          # toe kick (measured 4.70 +/- 0.16)
CT_TOP, CT_TH = 36.75, 1.25      # countertop top (measured 36.75 +/- 0.48), thickness (1.25 +/- 0.13)
CT_BOT = CT_TOP - CT_TH          # 35.5 = base carcass top
FR_BOT, FR_TOP = TK_H, CT_BOT - REV   # base fronts span
UP_BOT, UP_SHORT_BOT, UP_TOP = 55.125, 67.125, 97.125   # measured 55.12 / 67.38 / 97.1
BASE_FACE, UP_FACE_L, UP_FACE_R, PANTRY_FACE = 24.75, 12.40, 12.65, 24.5
PULL = dict(len=6.75, bar_w=0.5, bar_t=0.375, standoff=1.25, post=0.375)

parts = []
def add(**p):
    parts.append(p); return p["id"]

counters = {}
def nid(prefix):
    counters[prefix] = counters.get(prefix, 0) + 1
    return f"{prefix}_{counters[prefix]:02d}"

def box(x0, x1, y0, y1, z0, z1):
    return [[x0, y0, z0], [x1, y1, z1]]

# ---------- helpers that build fronts in world coords for a run ----------
# run 'L' : lateral axis x, fronts at +y.  viewer's LEFT = +x.
# run 'R' : lateral axis y, fronts at +x.  viewer's LEFT = -y.
def front_box(run, a0, a1, z0, z1, face):
    if run == "L":
        return box(a0, a1, face - FT, face, z0, z1)
    return box(face - FT, face, a0, a1, z0, z1)

def hinge_world_side(run, side_lateral_min):
    """Return viewer-relative hinge side given whether hinge is at the lateral-min edge."""
    if run == "L":   # lateral min (small x) is viewer's RIGHT
        return "right" if side_lateral_min else "left"
    return "left" if side_lateral_min else "right"   # run R: small y is viewer's LEFT

def pull(parent, run, orient, center_lat, center_z, face, length=None, label=""):
    L = length or PULL["len"]
    return add(id=nid("hardware_handle"), type="bar_pull", layer="hardware", material="mat_hardware_finish",
               parent=parent, run=run, orientation=orient,       # 'vertical' | 'horizontal'
               mount_center=[center_lat, center_z], mount_face=face, length=L,
               bar_w=PULL["bar_w"], bar_t=PULL["bar_t"], standoff=PULL["standoff"], post=PULL["post"],
               label=label or f"{L:.2f}in bar pull")

def door(parent, run, a0, a1, z0, z1, face, hinge_at_min, pull_spec, label, frame=2.25):
    did = add(id=nid("door"), type="shaker_front", role="door", layer="cabinets", material="mat_cabinet_primary",
              parent=parent, run=run, aabb=front_box(run, a0, a1, z0, z1, face), frame_width=frame,
              panel_recess=0.25, hinge_side=hinge_world_side(run, hinge_at_min),
              hinge_lateral=a0 if hinge_at_min else a1, label=label)
    if pull_spec:
        orient, where = pull_spec   # where: 'bottom' (uppers), 'top' (bases), ('z', zc)
        lat = (a1 - 1.75) if hinge_at_min else (a0 + 1.75)
        L = PULL["len"] if not isinstance(where, tuple) or len(where) < 3 else where[2]
        if where == "bottom": zc = z0 + 2.5 + L / 2
        elif where == "top": zc = z1 - 2.5 - L / 2
        else: zc = where[1]
        pull(did, run, orient, lat, zc, face, L)
    return did

def drawer(parent, run, a0, a1, z0, z1, face, label, pull_at="center", frame=None):
    h = z1 - z0
    fw = frame if frame else (1.75 if h < 8 else 2.25)
    did = add(id=nid("drawer_front"), type="shaker_front", role="drawer" if h < 16 else "pullout",
              layer="drawers", material="mat_cabinet_primary", parent=parent, run=run,
              aabb=front_box(run, a0, a1, z0, z1, face), frame_width=fw, panel_recess=0.25,
              slide_axis="+y" if run == "L" else "+x", label=label)
    zc = (z0 + z1) / 2 if pull_at == "center" else z1 - 2.5 - PULL["bar_w"] / 2
    pull(did, run, "horizontal", (a0 + a1) / 2, zc, face)
    return did

def stack(heights):
    """heights of fronts from TOP down (last may be None = remainder). Returns list of (z0,z1)."""
    total = FR_TOP - FR_BOT
    fixed = sum(h for h in heights if h) + REV * (len(heights) - 1)
    hs = [h if h else total - fixed for h in heights]
    out, z = [], FR_TOP
    for h in hs:
        out.append((z - h, z)); z -= h + REV
    return out

def base_carcass(run, a0, a1, label, depth_face=BASE_FACE, top=CT_BOT, notch=True, prefix="base_cabinet", **kw):
    front = depth_face - FT
    aabb = box(a0, a1, -T, front, 0, top) if run == "L" else box(-T, front, a0, a1, 0, top)
    return add(id=nid(prefix), type="carcass", layer="cabinets", material="mat_cabinet_primary", run=run,
               aabb=aabb, toe_kick=[TK_H, TK_D] if notch else None, front_normal="+y" if run == "L" else "+x",
               label=label, **kw)

def upper_carcass(run, a0, a1, z0, label, face):
    front = face - FT
    aabb = box(a0, a1, -T, front, z0, UP_TOP) if run == "L" else box(-T, front, a0, a1, z0, UP_TOP)
    return add(id=nid("upper_cabinet"), type="carcass", layer="cabinets", material="mat_cabinet_primary", run=run,
               aabb=aabb, toe_kick=None, front_normal="+y" if run == "L" else "+x", label=label)

h = REV / 2
# ======================= RIGHT RUN (W2, along +y) =======================
# measured: filler 24.81-28.34, door base 28.34-50.88, 4-drawer 50.88-72.04, counter end 73.14
c = base_carcass("R", 28.25, 50.875, "R-run drawer-over-door base 22-5/8in (measured 22.55)")
zz = stack([6.25, None])
drawer(c, "R", 28.25 + h, 50.875 - h, *zz[0], BASE_FACE, "top drawer")
door(c, "R", 28.25 + h, 50.875 - h, *zz[1], BASE_FACE, hinge_at_min=True, pull_spec=("vertical", "top"), label="door, hinge at corner side")
c = base_carcass("R", 50.875, 72.0, "R-run 4-drawer base 21in (measured 21.15)")
for i, (z0, z1) in enumerate(stack([6.25, 6.25, 6.25, None])):
    drawer(c, "R", 50.875 + h, 72.0 - h, z0, z1, BASE_FACE, f"drawer {i+1} of 4")
# uppers R: door faces x=12.65 ; measured 12.19-23.12 | 23.83-47.48 | 47.97-71.55
c = upper_carcass("R", 12.5, 24.0, UP_BOT, "R-run narrow upper 11-1/2in x 42in (no visible pull)", UP_FACE_R)
door(c, "R", 12.5 + h, 24.0 - h, UP_BOT + h, UP_TOP - h, UP_FACE_R, hinge_at_min=False, pull_spec=None, label="narrow door, no pull visible (push-latch?); hinged away from corner so it clears door at inside corner")
for a0, a1 in [(24.0, 48.0), (48.0, 72.0)]:
    c = upper_carcass("R", a0, a1, UP_BOT, f"R-run 24in x 42in double-door upper", UP_FACE_R)
    m = (a0 + a1) / 2
    door(c, "R", a0 + h, m - h, UP_BOT + h, UP_TOP - h, UP_FACE_R, True, ("vertical", "bottom"), "left leaf")
    door(c, "R", m + h, a1 - h, UP_BOT + h, UP_TOP - h, UP_FACE_R, False, ("vertical", "bottom"), "right leaf")

# ======================= LEFT RUN (W1, along +x) =======================
# corner: dead/blind corner box owned by the left run, hidden behind fillers
add(id=nid("base_cabinet"), type="carcass", layer="cabinets", material="mat_cabinet_primary", run="L",
    aabb=box(-T, 26.875, -T, BASE_FACE - FT, 0, CT_BOT), toe_kick=[TK_H, TK_D], front_normal="+y",
    label="blind/dead corner base box (not visible in photos; assumed)")
# measured: corner filler 24.70-26.87, 3-drawer 26.87-41.77, sink base -72.46, drawer base -90.18, pantry 90.45-111.43
c = base_carcass("L", 26.875, 41.875, "L-run 3-drawer base 15in (measured 14.90)")
for i, (z0, z1) in enumerate(stack([7.25, 9.25, None])):
    drawer(c, "L", 26.875 + h, 41.875 - h, z0, z1, BASE_FACE, f"drawer {i+1} of 3")
SINK = dict(x0=42.125, x1=72.125, y0=26.875 - 18.0, y1=26.875, z0=CT_BOT - 9.875, z1=CT_BOT)
c = base_carcass("L", 41.875, 72.375, "L-run farmhouse sink base 30-1/2in (measured 30.69)",
                 subtract_box=box(SINK["x0"], SINK["x1"], SINK["y0"], BASE_FACE + 1, SINK["z0"], CT_BOT + 1))
m = (41.875 + 72.375) / 2
door(c, "L", 41.875 + h, m - h, FR_BOT, SINK["z0"] - REV, BASE_FACE, True, ("vertical", "top"), "sink door (viewer's right leaf)")
door(c, "L", m + h, 72.375 - h, FR_BOT, SINK["z0"] - REV, BASE_FACE, False, ("vertical", "top"), "sink door (viewer's left leaf)")
c = base_carcass("L", 72.375, 90.375, "L-run drawer + pull-out base 18in (measured 17.72)")
zz = stack([6.25, None])
drawer(c, "L", 72.375 + h, 90.375 - h, *zz[0], BASE_FACE, "top drawer")
drawer(c, "L", 72.375 + h, 90.375 - h, *zz[1], BASE_FACE, "pull-out front (horizontal pull => pull-out, likely waste bin)", pull_at="top")
# pantry (tall)
c = add(id=nid("tall_cabinet"), type="carcass", layer="cabinets", material="mat_cabinet_primary", run="L",
        aabb=box(90.375, 111.375, -T, PANTRY_FACE - FT, 0, 96.5), toe_kick=[TK_H, TK_D], front_normal="+y",
        label="L-run pantry 21in x 96-1/2in (measured 20.98 wide, 99.13 incl. cap)")
PSPLIT = 64.625
door(c, "L", 90.375 + h, 111.375 - h, FR_BOT, PSPLIT, PANTRY_FACE, False, ("vertical", ("z", 48.25, 10.0)), "pantry lower door")
door(c, "L", 90.375 + h, 111.375 - h, PSPLIT + REV, 96.5 - h, PANTRY_FACE, False, ("vertical", "bottom"), "pantry upper door")
# uppers L: faces y=12.40 ; measured corner 0-23.87 (visible from 12.91) | 23.87-42.26 | 42.26-72.50 | 72.50-90.45
c = upper_carcass("L", -T, 24.0, UP_BOT, "L-run 24in blind-corner upper x 42in (+3/8in scribe to W2 drywall)", UP_FACE_L)
door(c, "L", 12.875, 24.0 - h, UP_BOT + h, UP_TOP - h, UP_FACE_L, True, ("vertical", "bottom"), "blind-corner door (visible portion)")
c = upper_carcass("L", 24.0, 42.0, UP_BOT, "L-run 18in x 42in upper", UP_FACE_L)
door(c, "L", 24.0 + h, 42.0 - h, UP_BOT + h, UP_TOP - h, UP_FACE_L, True, ("vertical", "bottom"), "single door")
c = upper_carcass("L", 42.0, 72.0, UP_SHORT_BOT, "L-run 30in x 30in double-door upper over sink", UP_FACE_L)
door(c, "L", 42.0 + h, 57.0 - h, UP_SHORT_BOT + h, UP_TOP - h, UP_FACE_L, True, ("vertical", "bottom"), "leaf (viewer's right)")
door(c, "L", 57.0 + h, 72.0 - h, UP_SHORT_BOT + h, UP_TOP - h, UP_FACE_L, False, ("vertical", "bottom"), "leaf (viewer's left)")
c = upper_carcass("L", 72.0, 90.0, UP_BOT, "L-run 18in x 42in upper beside pantry", UP_FACE_L)
door(c, "L", 72.0 + h, 90.0 - h, UP_BOT + h, UP_TOP - h, UP_FACE_L, False, ("vertical", "bottom"), "single door")

# ======================= TRIM =======================
add(id=nid("trim_filler"), type="box", layer="cabinets", material="mat_cabinet_primary", run="L",
    aabb=box(BASE_FACE - FT, 26.875, BASE_FACE - FT, BASE_FACE, TK_H, CT_BOT), label="L-run corner filler 2-1/8in (measured 2.17)")
add(id=nid("trim_filler"), type="box", layer="cabinets", material="mat_cabinet_primary", run="R",
    aabb=box(BASE_FACE - FT, BASE_FACE, BASE_FACE, 28.25, TK_H, CT_BOT), label="R-run corner filler 3-1/2in (measured 3.53)")
add(id=nid("trim_filler"), type="box", layer="cabinets", material="mat_cabinet_primary", run="R",
    aabb=box(BASE_FACE - FT - TK_D, BASE_FACE - FT, BASE_FACE, 28.25, TK_H, CT_BOT), label="hidden filler backer block (attaches filler_02 to base_cabinet_01 side)")
add(id=nid("trim_toekick"), type="box", layer="cabinets", material="mat_cabinet_primary", run="L",
    aabb=box(BASE_FACE - FT - TK_D, 90.375, BASE_FACE - FT - TK_D, BASE_FACE - FT - TK_D + 0.5, 0, TK_H), label="L-run toe kick board")
add(id=nid("trim_toekick"), type="box", layer="cabinets", material="mat_cabinet_primary", run="L",
    aabb=box(90.375, 111.375, PANTRY_FACE - FT - TK_D, PANTRY_FACE - FT - TK_D + 0.5, 0, TK_H), label="pantry toe kick board")
add(id=nid("trim_toekick"), type="box", layer="cabinets", material="mat_cabinet_primary", run="R",
    aabb=box(BASE_FACE - FT - TK_D, BASE_FACE - FT - TK_D + 0.5, BASE_FACE - FT - TK_D + 0.5, 72.0, 0, TK_H), label="R-run toe kick board")
add(id=nid("trim_crown"), type="box", layer="cabinets", material="mat_cabinet_primary", run="L",
    aabb=box(-T, 90.375, -T, UP_FACE_L + 0.75, UP_TOP, 100.625), label="L-run flat crown 3-1/2in (measured 97.14-100.60)")
add(id=nid("trim_crown"), type="box", layer="cabinets", material="mat_cabinet_primary", run="R",
    aabb=box(-T, UP_FACE_R + 0.75, UP_FACE_L + 0.75, 72.75, UP_TOP, 98.75), label="R-run flat crown 1-5/8in (measured 97.05-98.77)")
add(id=nid("trim_cap"), type="box", layer="cabinets", material="mat_cabinet_primary", run="L",
    aabb=box(90.375, 112.125, -T, PANTRY_FACE + 0.75, 96.5, 99.125), label="pantry cap (measured 96.58-99.13)")

# ======================= COUNTERTOPS =======================
add(id=nid("countertop_slab"), type="slab", layer="countertops", material="mat_countertop_surface", run="R",
    aabb=box(-T, 25.25, -T, 73.125, CT_BOT, CT_TOP), cutouts=[],
    label="R-run + corner slab, 3cm quartz (Vadara), front x=25.25 (measured 25.29), end y=73.125 (measured 73.14)")
add(id=nid("countertop_slab"), type="slab", layer="countertops", material="mat_countertop_surface", run="L",
    aabb=box(25.25, 90.375, -T, 26.5, CT_BOT, CT_TOP),
    cutouts=[box(SINK["x0"] + 1.0, SINK["x1"] - 1.0, SINK["y0"] + 1.25, 26.5 + 1, CT_BOT - 1, CT_TOP + 1)],
    label="L-run slab with farmhouse sink notch, front y=26.5 (measured 26.54), 1/4in overhang over sink back wall")

# ======================= FIXTURES =======================
add(id=nid("fixture_sink"), type="farmhouse_sink", layer="fixtures", material="mat_fixture_fireclay", run="L",
    aabb=box(SINK["x0"], SINK["x1"], SINK["y0"], SINK["y1"], SINK["z0"], SINK["z1"]),
    wall=1.0, bowl_depth=8.875, drain=dict(center_xy=[49.625, 17.875], diameter=3.5),
    label="Nantucket Sinks FCFS3018W-OSD 30x18x9-7/8 fireclay apron sink (read from price tag); apron 2.1in proud of doors (measured 2.06)")
add(id=nid("fixture_faucet"), type="faucet", layer="fixtures", material="mat_faucet_finish", run="L",
    base=[57.3, 6.6, CT_TOP], body_height=10.75, body_d_bottom=2.0, body_d_top=1.4,
    arm=dict(z=46.0, dir_xy=[-0.21, 0.98], length=9.75, d=0.9),
    head=dict(z0=44.5, z1=49.0, d=1.3), hose=dict(apex_z=57.3, d=1.0, material="mat_faucet_hose"),
    label="pull-down faucet, brushed nickel body with black hose (base triangulated at x57.3 y6.6, z36.6 vs counter 36.75)")

# ======================= BACKSPLASH =======================
add(id=nid("backsplash"), type="union_boxes", layer="backsplash", material="mat_backsplash_tile", run="L",
    boxes=[box(0.0, 90.375, -T, 0.0, CT_TOP, UP_BOT), box(42.0, 72.0, -T, 0.0, UP_BOT, UP_SHORT_BOT)],
    label="W1 glazed subway tile field (vertical stack behind sink)")
add(id=nid("backsplash"), type="union_boxes", layer="backsplash", material="mat_backsplash_tile", run="R",
    boxes=[box(-T, 0.0, -T, 72.0, CT_TOP, UP_BOT)], label="W2 glazed subway tile field")

# ======================= REFERENCE (non-fabrication) =======================
add(id=nid("reference_wall"), type="box", layer="reference", material="mat_reference", run="L",
    aabb=box(-T - 4.0, 120.0, -T - 4.0, -T, 0, 120.0), label="W1 wall (drywall face at y=-0.375)")
add(id=nid("reference_wall"), type="box", layer="reference", material="mat_reference", run="R",
    aabb=box(-T - 4.0, -T, -T, 84.0, 0, 120.0), label="W2 wall (drywall face at x=-0.375)")
add(id=nid("reference_floor"), type="box", layer="reference", material="mat_reference", run=None,
    aabb=box(-T - 4.0, 120.0, -T - 4.0, 84.0, -1.0, 0.0), label="floor")

# baseColor = linear albedo. Photo-derived (tools/albedo.py: median linear RGB over 4 views, white-balanced so the
# fireclay sink = 0.85 neutral). Cabinet value used as-is (well-sampled vertical faces). Countertop / tile / hardware
# keep the photo chromaticity but get a luminance prior, because shadowing under uppers + gloss biases them dark.
def lumfix(rgb, target):
    l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return [round(min(1.0, c / l * target), 3) for c in rgb]
ALB = json.load(open("/home/claude/kitchen/artifacts/verify/albedo_estimate.json"))["albedo_linear"]
materials = {
    "mat_cabinet_primary":    dict(baseColor=ALB["mat_cabinet_primary"], metallic=0.0, roughness=0.55, photo_albedo=ALB["mat_cabinet_primary"], note="greige satin lacquer; photo-derived albedo"),
    "mat_countertop_surface": dict(baseColor=lumfix(ALB["mat_countertop_surface"], 0.78), metallic=0.0, roughness=0.20, photo_albedo=ALB["mat_countertop_surface"], note="white quartz w/ gray veining (Vadara); photo chroma, luminance prior 0.78; vein texture slot"),
    "mat_hardware_finish":    dict(baseColor=lumfix(ALB["mat_hardware_finish"], 0.62), metallic=1.0, roughness=0.35, photo_albedo=ALB["mat_hardware_finish"], note="champagne bronze bar pulls; photo chroma, metal F0 luminance prior"),
    "mat_faucet_finish":      dict(baseColor=[0.66, 0.64, 0.60], metallic=1.0, roughness=0.40, note="brushed nickel faucet body"),
    "mat_fixture_fireclay":   dict(baseColor=[0.85, 0.85, 0.85], metallic=0.0, roughness=0.12, note="glossy white fireclay; white-balance reference"),
    "mat_backsplash_tile":    dict(baseColor=lumfix(ALB["mat_backsplash_tile"], 0.22), metallic=0.0, roughness=0.15, photo_albedo=ALB["mat_backsplash_tile"], note="glazed taupe/gray subway; photo chroma, luminance prior 0.22"),
    "mat_faucet_hose":        dict(baseColor=[0.04, 0.04, 0.04], metallic=0.0, roughness=0.6, note="black silicone hose"),
    "mat_reference":          dict(baseColor=[0.55, 0.55, 0.55], metallic=0.0, roughness=0.9, note="reference walls/floor, not for fabrication"),
}

spec = dict(
    schema="keane-kitchens/layout-spec/v1", units="in", export_units="mm", inch_to_mm=25.4,
    frame=dict(up="+z", origin="inside corner at floor, at backsplash tile faces",
               W1="sink wall, tile face y=0, room y>0; left run along +x; fronts +y",
               W2="right wall, tile face x=0, room x>0; right run along +y; fronts +x",
               drywall_offset=T, viewer_left={"L": "+x", "R": "-y"}),
    globals=dict(front_thickness=FT, reveal=REV, toe_kick=[TK_H, TK_D], countertop_top=CT_TOP, countertop_thickness=CT_TH,
                 upper_bottom=UP_BOT, short_upper_bottom=UP_SHORT_BOT, upper_top=UP_TOP, pull=PULL),
    provenance="Agent A metric SfM (pycolmap, 5/5 images, 0.91px reproj, scale from 30in sink apron +/-1.24%) "
               "+ Agent C snapping to 1/8in / standard sizes. Excluded as merchandise: tile samples, brochures, binder, disposal display.",
    materials=materials, parts=parts)
json.dump(spec, open("/home/claude/kitchen/artifacts/geometry/layout_spec.json", "w"), indent=1)
from collections import Counter
print(len(parts), "parts", Counter(p["id"].rsplit("_", 1)[0] for p in parts))
