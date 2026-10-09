"""Final metric measurements in the kitchen frame (inches; mm also reported).

Frame: z up; W1 (sink wall) = plane y=0, room y>0; W2 (right wall) = plane x=0, room x>0;
origin = inside corner at floor level.  Wall planes are taken at the BACKSPLASH TILE FACE
(drywall is ~3/8-1/2" behind it).

Aligned (unscaled) coords Xa = R_vp @ X_model.  inches = s * (Xa - o).
"""
import sys, json, numpy as np
sys.path.insert(0, "/home/claude/kitchen/artifacts/sfm/scripts")
from manhattan import MFrame

P = "/home/claude/kitchen/artifacts/sfm/work_s075/sparse/0"
R = np.load(P + "/../R_align_vp.npy")
F = MFrame(P, R)
L = json.load(open("/home/claude/kitchen/artifacts/sfm/work_s075/line_fits.json"))

# ---------------- frame constants (aligned units) ----------------
W1_y, W1_sig = -6.295, 0.03     # backsplash face, sink wall
W2_x, W2_sig = -9.925, 0.03     # backsplash face, right wall
FL_z, FL_sig = -6.362, 0.012    # floor (sparse floor points)
CT_z, CT_sig = -2.803, 0.008    # countertop top surface (RANSAC/median of 1650 pts)
o = np.array([W2_x, W1_y, FL_z])

# ---------------- scale from sink apron ----------------
xl_50 = L["sink_apron_left_edge"]["p"][0]        # 50% shading transition of rounded front-left edge
xl_gap = L["sink_side_back_gap"]["p"][0]         # back boundary of exposed sink side (gap line)
x_left = 0.5 * (xl_50 + xl_gap)                  # best estimate of the apron's outer left plane
x_left_sig = 0.5 * abs(xl_gap - xl_50)
x_right = L["sink_apron_right_edge"]["p"][0]
w_units = x_right - x_left
w_units = abs(w_units)
w_sig = np.hypot(x_left_sig, L["sink_apron_right_edge"]["sig"][0] + 0.005)
s = 30.0 / w_units
s_sig = s * w_sig / w_units
h_units = L["sink_apron_top_lip"]["p"][1] - L["sink_apron_bottom"]["p"][1]
s_h = 9.875 / h_units
SCALE = dict(in_per_unit=s, rel_sigma=s_sig / s, apron_width_units=w_units, apron_height_units=h_units,
             s_from_apron_height=s_h, apron_height_in_at_s=h_units * s,
             apron_width_alternatives_in={"to_50pct_left_edge": abs(x_right - xl_50) * s, "to_gap_line": abs(x_right - xl_gap) * s})
print("SCALE", SCALE)


def to_in(v, axis):
    return s * (v - o[axis])


def ray_plane(k, uv, axis, value):
    return F.ray_plane(k, uv, axis, value)


def rp_meas(k, uv, axis_plane, plane_val, plane_sig, out_axis, px=2.0):
    """value on out_axis (aligned units) of the ray-plane intersection + sigma from px noise and plane sigma."""
    X0 = ray_plane(k, uv, axis_plane, plane_val)[out_axis]
    d = []
    for du in [(px, 0), (0, px)]:
        X1 = ray_plane(k, (uv[0] + du[0], uv[1] + du[1]), axis_plane, plane_val)[out_axis]
        d.append(X1 - X0)
    Xp = ray_plane(k, uv, axis_plane, plane_val + plane_sig)[out_axis]
    sig = np.sqrt(d[0] ** 2 + d[1] ** 2 + (Xp - X0) ** 2)
    return X0, sig


M = {}  # name -> dict


def put(name, val_in, sig_in, method, extra=None):
    sc = abs(float(val_in)) * s_sig / s
    tot = float(np.hypot(sig_in, sc))
    d = dict(in_=round(float(val_in), 2), mm=round(float(val_in) * 25.4, 1),
             sigma_in=round(tot, 2), sigma_mm=round(tot * 25.4, 1),
             sigma_geom_in=round(float(sig_in), 2), sigma_scale_in=round(sc, 2), method=method)
    if extra:
        d.update(extra)
    M[name] = d
    print(f"{name:62s} {val_in:8.2f} in  +/- {tot:5.2f} (geom {sig_in:4.2f})  [{method[:70]}]")


def comb(vals, sigs):
    """combine independent estimates; uncertainty = max(weighted sigma, half-spread)"""
    vals = np.array(vals, float); sigs = np.array(sigs, float)
    w = 1 / sigs ** 2
    m = (w * vals).sum() / w.sum()
    sw = np.sqrt(1 / w.sum())
    spread = (vals.max() - vals.min()) / 2 if len(vals) > 1 else 0
    return m, max(sw, spread)


def abs_sig(val_units_from_origin_sig, val_in):
    """geometric sigma (in) for an absolute coordinate (scale sigma is added in put())"""
    return s * val_units_from_origin_sig


# ======================= (a) countertop =======================
zc_in = to_in(CT_z, 2)
put("countertop_top_height_above_floor", zc_in, abs_sig(np.hypot(CT_sig, FL_sig), zc_in),
    "z of RANSAC/median countertop plane (1650 sparse pts) minus floor z (17 sparse floor pts)")

# left-run counter front edge (top arris) from 05: intersect with top plane
yv, ys = [], []
for uv in [(1100, 2631), (1240, 2626), (2550, 2527), (2700, 2521)]:
    v, sg = rp_meas("05", uv, 2, CT_z, CT_sig, 1); yv.append(v); ys.append(sg)
y_ctfL, y_ctfL_sig = comb(yv, ys)
put("left_run_countertop_front_edge_y", to_in(y_ctfL, 1), abs_sig(np.hypot(y_ctfL_sig, W1_sig), to_in(y_ctfL, 1)),
    "05 top-arris pixels (4 pts, both sides of sink) ray-intersected with countertop top plane")
zb, zbs = [], []
for uv in [(1000, 2679.8), (1400, 2648.0), (2550, 2567.7), (2700, 2557.3)]:
    v, sg = rp_meas("05", uv, 1, y_ctfL, y_ctfL_sig, 2); zb.append(v); zbs.append(sg)
zbL, zbL_sig = comb(zb, zbs)
put("left_run_countertop_thickness", (CT_z - zbL) * s, s * np.hypot(zbL_sig, CT_sig),
    "05 front-edge bottom pixels ray-intersected with the counter front plane; top plane minus bottom")

# right-run counter front edge from 04
xv, xs = [], []
for uv in [(1500, 2724), (2600, 2740)]:
    v, sg = rp_meas("04", uv, 2, CT_z, CT_sig, 0); xv.append(v); xs.append(sg)
x_ctfR, x_ctfR_sig = comb(xv, xs)
put("right_run_countertop_front_edge_x", to_in(x_ctfR, 0), abs_sig(np.hypot(x_ctfR_sig, W2_sig), to_in(x_ctfR, 0)),
    "04 top-arris pixels ray-intersected with countertop top plane")
zb, zbs = [], []
for uv in [(1500, 2784), (2600, 2797)]:
    v, sg = rp_meas("04", uv, 0, x_ctfR, x_ctfR_sig, 2); zb.append(v); zbs.append(sg)
zbR, zbR_sig = comb(zb, zbs)
put("right_run_countertop_thickness", (CT_z - zbR) * s, s * np.hypot(zbR_sig, CT_sig),
    "04 front-edge bottom pixels ray-intersected with the counter front plane")

# countertop right end: front-top-right corner triangulated in 04 + 03, and ray-plane cross-check
Xc, Cc, errc = F.point({"04": (2815, 2748), "03": (2988, 2805)})
y_end_rp, y_end_rp_sig = rp_meas("04", (2815, 2748), 2, CT_z, CT_sig, 1)
y_end_rp3, y_end_rp3_sig = rp_meas("03", (2988, 2805), 2, CT_z, CT_sig, 1)
y_end, y_end_sig = comb([Xc[1], y_end_rp, y_end_rp3], [np.sqrt(Cc[1, 1]), y_end_rp_sig, y_end_rp3_sig])
put("right_run_countertop_end_y (run length from W2... along W1 normal)", to_in(y_end, 1),
    abs_sig(np.hypot(y_end_sig, W1_sig), to_in(y_end, 1)),
    f"front-top-right counter corner: 2-view triangulation (04,03; reproj {np.round(list(errc.values()),1)} px) + 2 ray/top-plane estimates",
    extra=dict(triangulated_corner_in=[round(float(to_in(Xc[i], i)), 2) for i in range(3)]))

# ======================= (b,c) uppers =======================
y_uf = np.mean([L["L_upper_short_left"]["p"][1], L["L_upper_short_right"]["p"][1], L["L_upper_short_bottom"]["p"][0]])
y_uf_sig = 0.02
x_uf = L["R_upper_bottom"]["p"][0]; x_uf_sig = max(L["R_upper_bottom"]["sig"][0], 0.025)
put("left_run_upper_door_face_y (depth from W1 backsplash)", to_in(y_uf, 1), abs_sig(np.hypot(y_uf_sig, W1_sig), to_in(y_uf, 1)),
    "mean y of 3 multi-view Manhattan line fits on the 30in upper over sink")
put("right_run_upper_door_face_x (depth from W2 backsplash)", to_in(x_uf, 0), abs_sig(np.hypot(x_uf_sig, W2_sig), to_in(x_uf, 0)),
    "x of 3-view Manhattan line fit of right-run upper bottom front edge")

r = L["R_upper_bottom"]
z = to_in(r["p"][1], 2)
put("right_run_upper_bottom_height", z, abs_sig(np.hypot(r["sig"][1], FL_sig), z),
    f"3-view (04,03,02) Manhattan y-line fit, rms {r['rms']:.1f}px; loo z spread {np.ptp([v[1] for v in r['loo'].values()]):.3f} units")
r = L["L_upper_short_bottom"]
z = to_in(r["p"][1], 2)
put("left_run_short_upper_over_sink_bottom_height", z, abs_sig(np.hypot(r["sig"][1], FL_sig), z),
    f"3-view (05,02,03) Manhattan x-line fit, rms {r['rms']:.1f}px")
# tall neighbours: ray-plane at upper face plane
vals, sigs = [], []
for k, uv in [("05", (2300, 1958.6)), ("05", (2650, 1951.9)), ("02", (1100, 1761.1)), ("02", (1350, 1747.6))]:
    v, sg = rp_meas(k, uv, 1, y_uf, y_uf_sig, 2); vals.append(v); sigs.append(sg)
zt, zts = comb(vals, sigs)
put("left_run_tall_upper_right_of_sink_bottom_height", to_in(zt, 2), abs_sig(np.hypot(zts, FL_sig), to_in(zt, 2)),
    "rays from 05 (2 px) and 02 (2 px) intersected with left upper face plane", extra=dict(per_view_in=[round(float(to_in(v, 2)), 2) for v in vals]))
vals, sigs = [], []
for k, uv in [("05", (1000, 1981.7)), ("05", (1350, 1975.3))]:
    v, sg = rp_meas(k, uv, 1, y_uf, y_uf_sig, 2); vals.append(v); sigs.append(sg)
zt2, zt2s = comb(vals, sigs)
put("left_run_tall_upper_left_of_sink_bottom_height", to_in(zt2, 2), abs_sig(np.hypot(zt2s, FL_sig), to_in(zt2, 2)),
    "05 rays intersected with left upper face plane (single view)")
vals, sigs = [], []
for k, uv in [("05", (2800, 1949.8)), ("05", (2900, 1950.4))]:
    v, sg = rp_meas(k, uv, 1, y_uf, y_uf_sig, 2); vals.append(v); sigs.append(sg)
zt3, zt3s = comb(vals, sigs)
put("left_run_corner_upper_bottom_height", to_in(zt3, 2), abs_sig(np.hypot(zt3s, FL_sig), to_in(zt3, 2)),
    "05 rays intersected with left upper face plane (single view)")
put("short_upper_bottom_minus_tall_upper_bottom", (L["L_upper_short_bottom"]["p"][1] - zt) * s, s * np.hypot(L["L_upper_short_bottom"]["sig"][1], zts),
    "difference of the two above (scale-dependent only)")

# crowns: left run = mean of 2-view (05,03) and 3-view (05,03,02) fits; right run = multi-view fits (04,03[,02])
CROWN2 = {"L_crown_top": (-4.9793, 3.3614), "L_crown_bottom": (-5.0111, 3.0339)}   # 2-view fits (05,03), recorded earlier
for nm, key in [("left_run_crown_underside_height (top of upper boxes)", "L_crown_bottom"), ("left_run_crown_top_height", "L_crown_top")]:
    z3 = L[key]["p"][1]; z2 = CROWN2[key][1]
    zm = 0.5 * (z2 + z3); zs = max(0.5 * abs(z3 - z2), L[key]["sig"][1])
    put(nm, to_in(zm, 2), abs_sig(np.hypot(zs, FL_sig), to_in(zm, 2)),
        f"mean of 2-view (05,03) and 3-view (05,03,02) Manhattan x-line fits ({to_in(z2,2):.1f} / {to_in(z3,2):.1f} in)")
for nm, key in [("pantry_cap_top_height", "L_pantry_cap_top"), ("pantry_cap_underside_height", "L_pantry_cap_bottom")]:
    r = L[key]
    z = to_in(r["p"][1], 2)
    put(nm, z, abs_sig(np.hypot(max(r["sig"][1], 0.015), FL_sig), z), f"2-view (05,03) Manhattan x-line fit, rms {r['rms']:.1f}px; edge y={to_in(r['p'][0],1):.1f}in from W1 (cap projects past pantry face)")
for nm, key in [("right_run_crown_top_height", "R_crown_top"), ("right_run_crown_underside_height (top of upper boxes)", "R_crown_bottom")]:
    r = L[key]
    z = to_in(r["p"][1], 2)
    put(nm, z, abs_sig(np.hypot(max(r["sig"][1], 0.015), FL_sig), z), f"Manhattan y-line fit views {r['views']}, rms {r['rms']:.1f}px")

# ======================= (d) right-run widths =======================
x_bf = L["R_base_door_left"]["p"][0]; x_bf_sig = 0.02
put("right_run_base_door_face_x (depth from W2 backsplash)", to_in(x_bf, 0), abs_sig(np.hypot(x_bf_sig, W2_sig), to_in(x_bf, 0)),
    "3-view Manhattan vertical-line fit of right-run door-base left edge")


def ypos(k, uvs, xplane, xs):
    vals, sigs = [], []
    for uv in uvs:
        v, sg = rp_meas(k, uv, 0, xplane, xs, 1); vals.append(v); sigs.append(sg)
    return comb(vals, sigs)

yb_doorL = L["R_base_door_left"]["p"][1]; yb_doorL_s = max(L["R_base_door_left"]["sig"][1], 0.01)
yb_seam, yb_seam_s = ypos("04", [(1948, 3300), (1950, 3600)], x_bf, x_bf_sig)
yb_end, yb_end_s = ypos("04", [(2758, 3300), (2744, 3600)], x_bf, x_bf_sig)
for nm, v, sg, meth in [("right_run_door_base_left_edge_y", yb_doorL, yb_doorL_s, "3-view vertical line fit"),
                        ("right_run_door_base_to_drawer_stack_seam_y", yb_seam, yb_seam_s, "04 rays at seam gap centre on base face plane"),
                        ("right_run_drawer_stack_right_end_y", yb_end, yb_end_s, "04 rays at stack right edge on base face plane")]:
    put(nm, to_in(v, 1), abs_sig(np.hypot(sg, W1_sig), to_in(v, 1)), meth)
put("right_run_door_base_width", (yb_seam - yb_doorL) * s, s * np.hypot(yb_seam_s, yb_doorL_s), "difference of edge positions")
put("right_run_4drawer_stack_width", (yb_end - yb_seam) * s, s * np.hypot(yb_end_s, yb_seam_s), "difference of edge positions")

# right-run uppers (04 at row 1300 on x=x_uf)
up_edges = {"cab1_left": 628, "cab1_right": 982, "cab2_left": 1005, "cab2_door_meet": 1397, "cab2_right": 1794,
            "cab3_left": 1811, "cab3_door_meet": 2219, "cab3_right_end": 2636}
upy = {}
for nm, u in up_edges.items():
    v, sg = rp_meas("04", (u, 1300), 0, x_uf, x_uf_sig, 1)
    upy[nm] = (v, sg)
    put(f"right_run_upper_{nm}_y", to_in(v, 1), abs_sig(np.hypot(sg, W1_sig), to_in(v, 1)), "04 ray (row 1300) on right upper face plane")
for nm, a, b in [("cab1_single_door", "cab1_left", "cab1_right"), ("cab2_double_door", "cab2_left", "cab2_right"), ("cab3_double_door", "cab3_left", "cab3_right_end")]:
    put(f"right_run_upper_{nm}_width", (upy[b][0] - upy[a][0]) * s, s * np.hypot(upy[a][1], upy[b][1]), "difference of 04 edge rays on face plane")

# ======================= (e) left run =======================
y_bf = np.mean([L["L_base_sink_left"]["p"][1], L["L_base_3dr_left"]["p"][1]]); y_bf_sig = 0.015
put("left_run_base_door_face_y (depth from W1 backsplash)", to_in(y_bf, 1), abs_sig(np.hypot(y_bf_sig, W1_sig), to_in(y_bf, 1)),
    "mean y of 2 multi-view vertical-line fits (sink-base left gap, 3-drawer left gap)")


def xpos(k, uvs, yplane, ys):
    vals, sigs = [], []
    for uv in uvs:
        v, sg = rp_meas(k, uv, 1, yplane, ys, 0); vals.append(v); sigs.append(sg)
    return comb(vals, sigs)

y_pf = L["L_pantry_right_edge"]["p"][1]
xe = {}
xe["pantry_left_edge"] = xpos("05", [(118.2, 1000), (156.9, 2000)], y_pf, 0.03)
xe["pantry_right_edge"] = comb([L["L_pantry_right_edge"]["p"][0], xpos("05", [(931.2, 2000), (930.7, 3000)], y_bf, y_bf_sig)[0]], [0.01, 0.01])
xe["drawer_over_door_base_left_edge"] = xpos("05", [(949, 3200), (944, 2800)], y_bf, y_bf_sig)
xe["drawer_over_door_base_right_edge = sink_base_left"] = (L["L_base_sink_left"]["p"][0], 0.01)
xe["sink_base_doors_meet"] = (L["L_base_sink_doors_mid"]["p"][0], L["L_base_sink_doors_mid"]["sig"][0])
xe["sink_base_right = 3drawer_left"] = comb([L["L_base_3dr_left"]["p"][0], xpos("05", [(2361, 3200)], y_bf, y_bf_sig)[0]], [0.01, 0.01])
xe["3drawer_base_right_edge"] = xpos("05", [(2751, 2700), (2733, 3100)], y_bf, y_bf_sig)
xe["sink_apron_left_outer"] = (x_left, x_left_sig)
xe["sink_apron_right_outer"] = (x_right, 0.01)
xe["upper_tall_left_right_edge = short_upper_left"] = (L["L_upper_short_left"]["p"][0], 0.01)
xe["upper_short_right = tall_right_left"] = (L["L_upper_short_right"]["p"][0], 0.01)
xe["upper_tall_right_right_edge"] = xpos("05", [(2670, 1400), (2652, 1800)], y_uf, y_uf_sig)
xe["upper_corner_visible_end (meets right-run upper face)"] = xpos("05", [(2918, 1400), (2943, 1000)], y_uf, y_uf_sig)
for nm, (v, sg) in xe.items():
    put(f"left_run_{nm}_x", to_in(v, 0), abs_sig(np.hypot(sg, W2_sig), to_in(v, 0)), "05 rays on face plane / multi-view line fits")
xs_ = {k: v[0] for k, v in xe.items()}
def wid(a, b):
    return abs(xs_[a] - xs_[b]) * s
put("left_run_pantry_width", wid("pantry_left_edge", "pantry_right_edge"), 0.4, "difference of pantry edge x")
put("left_run_drawer_over_door_base_width", wid("drawer_over_door_base_left_edge", "drawer_over_door_base_right_edge = sink_base_left"), 0.3, "difference")
put("left_run_sink_base_width", wid("drawer_over_door_base_right_edge = sink_base_left", "sink_base_right = 3drawer_left"), 0.3, "difference (multi-view fits both sides)")
put("left_run_3drawer_base_width", wid("sink_base_right = 3drawer_left", "3drawer_base_right_edge"), 0.4, "difference")
put("left_run_upper_short_width", wid("upper_tall_left_right_edge = short_upper_left", "upper_short_right = tall_right_left"), 0.3, "difference (multi-view fits both sides)")
put("left_run_upper_tall_right_width", wid("upper_short_right = tall_right_left", "upper_tall_right_right_edge"), 0.4, "difference")
put("left_run_upper_tall_left_width (pantry edge to short upper)", wid("pantry_right_edge", "upper_tall_left_right_edge = short_upper_left"), 0.4, "difference (left edge hidden behind pantry face; uses pantry edge)")
pz = L["L_pantry_cap_top"]["p"][1]
put("pantry_height_floor_to_cap_top", to_in(pz, 2), abs_sig(np.hypot(L["L_pantry_cap_top"]["sig"][1], FL_sig), to_in(pz, 2)), "cap-top line fit z minus floor")
put("pantry_face_y (depth from W1 backsplash)", to_in(y_pf, 1), abs_sig(np.hypot(0.02, W1_sig), to_in(y_pf, 1)), "2-view vertical line fit of pantry right edge (rms 7.7px: weak)")

# base door bottoms / toe kick (left run, 05)
zz, zzs = [], []
for uv in [(2100, 3457), (2550, 3390)]:
    v, sg = rp_meas("05", uv, 1, y_bf, y_bf_sig, 2); zz.append(v); zzs.append(sg)
zt_, zts_ = comb(zz, zzs)
put("left_run_base_door_bottom_height (toe-kick top)", to_in(zt_, 2), abs_sig(np.hypot(zts_, FL_sig), to_in(zt_, 2)), "05 rays at door/drawer bottom edge on base face plane (single view)")

# ======================= (f) sink apron =======================
y_ap = np.mean([L["sink_apron_left_edge"]["p"][1], L["sink_apron_right_edge"]["p"][1], L["sink_apron_top_lip"]["p"][0], L["sink_apron_bottom"]["p"][0]])
y_ap_sig = np.std([L["sink_apron_left_edge"]["p"][1], L["sink_apron_right_edge"]["p"][1], L["sink_apron_top_lip"]["p"][0], L["sink_apron_bottom"]["p"][0]])
put("sink_apron_front_plane_y", to_in(y_ap, 1), abs_sig(np.hypot(y_ap_sig, W1_sig), to_in(y_ap, 1)), "mean of 4 multi-view apron edge line fits")
put("sink_apron_proud_of_base_door_face", (y_ap - y_bf) * s, s * np.hypot(y_ap_sig, y_bf_sig), "apron plane minus base door-face plane")
put("sink_apron_top_height", to_in(L["sink_apron_top_lip"]["p"][1], 2), abs_sig(0.006, to_in(L["sink_apron_top_lip"]["p"][1], 2)), "3-view x-line fit of apron front-top lip highlight")
put("sink_apron_bottom_height", to_in(L["sink_apron_bottom"]["p"][1], 2), abs_sig(0.008, to_in(L["sink_apron_bottom"]["p"][1], 2)), "3-view x-line fit of apron bottom edge")
put("sink_apron_height_measured", h_units * s, s * 0.012, "top lip minus bottom (spec 9.875in)")
put("countertop_front_overhang_past_left_base_face", (y_ctfL - y_bf) * s, s * np.hypot(y_ctfL_sig, y_bf_sig), "difference")
put("countertop_front_overhang_past_right_base_face", (x_ctfR - x_bf) * s, s * np.hypot(x_ctfR_sig, x_bf_sig), "difference")

# ======================= corner details =======================
put("right_run_door_base_left_edge_minus_left_base_face (corner filler)", (yb_doorL - y_bf) * s, s * np.hypot(yb_doorL_s, y_bf_sig), "difference")
put("left_run_3drawer_right_edge_minus_right_base_face (corner filler)", (xs_["3drawer_base_right_edge"] - x_bf) * s, s * 0.03, "difference")

json.dump(dict(frame=dict(description="z up; W1 (sink wall, backsplash tile face) is y=0, room y>0; W2 (right wall, backsplash tile face) is x=0, room x>0; origin = inside corner at floor level. Units inches (mm also given). Drywall is ~0.4in behind the tile face.",
                          aligned_to_frame=dict(R_vp_model_to_aligned=R.tolist(), origin_aligned_units=o.tolist(), inches_per_unit=s,
                                                formula="X_in = s * (R_vp @ X_model - o)")),
               scale=SCALE, values=M), open("/home/claude/kitchen/artifacts/sfm/scripts/measurements_raw.json", "w"), indent=1)
