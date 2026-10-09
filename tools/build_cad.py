#!/usr/bin/env python3
"""Agent D -- mesh extraction, semantic decomposition & topological validation.

Turns the orchestrator's fully-resolved parametric layout (layout_spec.json: inches, Z-up world)
into clean, watertight, modular CAD components in millimetres with pivots, OBBs, UV atlases,
PBR material slots, an assembly manifest, layered world-space exports and a validation report.

    python3 tools/build_cad.py --spec artifacts/geometry/layout_spec.json [--no-step] [--no-preview]

Everything is regenerated deterministically from the spec (output folders are wiped first).
All booleans/unions use manifold3d (guaranteed-manifold output, float64 Mesh64 path).

Frames
------
World (spec): Z up, inches -> exported in mm (x25.4).  W1 tile face y=0 (left run "L" along +x,
fronts +y); W2 tile face x=0 (right run "R" along +y, fronts +x).
Component-local (every component file): right-handed, Y-up, OBJ/glTF style.  +X = viewer's right
when facing the part's front, +Y = up, +Z = out of the front.
    run L : X = -x_w, Y = +z_w, Z = +y_w        run R : X = +y_w, Y = +z_w, Z = +x_w
    run None (floor) : X = +x_w, Y = +z_w, Z = -y_w
world_mm = R @ local_mm + pivot_mm   (R columns = local X,Y,Z expressed in world).
"""
import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from collections import OrderedDict

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh64, OpType
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------------------------
# 0. Constants
# ----------------------------------------------------------------------------------------------
IN = 25.4                 # inch -> mm
EPS = 1.0                 # mm overshoot for boolean cutters (cutters never share a plane with kept faces)
LAT_AX = {"L": 0, "R": 1}     # world axis index of a run's lateral direction
FRONT_AX = {"L": 1, "R": 0}   # world axis index of a run's front normal (+)
RUN_ROT = {               # local->world rotation; columns are local X, Y, Z in world coordinates
    "L": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
    "R": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float),
    None: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
}
CATEGORY_PREFIX = [("base_cabinet", "base_cabinet"), ("upper_cabinet", "upper_cabinet"),
                   ("tall_cabinet", "tall_cabinet"), ("door", "door"), ("drawer_front", "drawer_front"),
                   ("countertop_slab", "countertop_slab"), ("hardware_handle", "hardware_handle"),
                   ("trim_", "trim"), ("fixture_", "fixture"), ("backsplash", "backsplash"),
                   ("reference_", "reference")]
LAYER_STL = OrderedDict([("cabinets", "cabinets.stl"), ("drawers", "drawers.stl"),
                         ("countertops", "countertops.stl"), ("hardware", "hardware.stl"),
                         ("fixtures", "fixtures.stl"), ("backsplash", "backsplash.stl"),
                         ("reference", "reference.stl")])
SEG = dict(body=48, head=40, arm=24, drain=48, hose_ring=24, hose_path=96)  # tessellation of round parts
HOSE_EMBED_IN = 0.3       # hose tube continues this far straight down into body / head (robust union)
FAUCET_SIMPLIFY_MM = 1e-3 # manifold3d simplify tolerance for the faucet union (removes boolean slivers)
UNION_WELD_MM = 0.01      # outward offset per part for the position-space-watertight union STL
SMOOTH_DEG = 40.0         # corner-normal smoothing threshold (box edges are 90 deg -> stay flat)
UV_RES = 1024             # rasterisation resolution for the UV-overlap proof
ISO_MAX_EDGE_MM = 120.0   # preview-only subdivision for the matplotlib painter's sort
DOOR_OPEN_DEG = 110.0     # nominal open angle; swing sweep caps max_open_deg here
SWING_STEP_DEG = 5        # swing-sweep resolution for max_open_deg
STEP_COLORS = False       # True adds per-part colours to kitchen.step but makes the file byte-nondeterministic
PULL_MOUNT = "contact_surface"   # catalog pull is seated on the real surface under its posts
                                 # ("spec_face" would seat it on mount_face even over a recess)
FRONT_GAP_SEARCH_MM = 12.7       # front-plane pairs closer than this get a clearance measurement
PAIR_AABB_TOL_MM = 0.02 * IN     # pairwise interference candidates: AABBs within 0.02 in
INTERSECT_FLAG_MM3 = 1.0


def mm(a):
    return np.asarray(a, float) * IN


def category_of(pid):
    for pre, cat in CATEGORY_PREFIX:
        if pid.startswith(pre):
            return cat
    raise ValueError(f"unknown part id prefix: {pid}")


def fmt_in(x):
    return f"{x:.4f}".rstrip("0").rstrip(".")


def frac_in(x, den=16):
    """22.625 -> '22-5/8' (only when exact on a 1/16 grid; else decimal)."""
    n = round(x * den)
    if abs(n / den - x) > 1e-4:
        return fmt_in(x)
    whole, rem = divmod(n, den)
    if rem == 0:
        return str(whole)
    g = math.gcd(rem, den)
    return f"{whole}-{rem // g}/{den // g}" if whole else f"{rem // g}/{den // g}"


def T44(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def r6(a):
    """Round for JSON (deterministic, compact)."""
    return np.round(np.asarray(a, float), 6).tolist()


# ----------------------------------------------------------------------------------------------
# 1. Primitive geometry helpers (manifold3d, world mm)
# ----------------------------------------------------------------------------------------------
def m_box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def m_from_arrays(V, F):
    return Manifold(Mesh64(vert_properties=np.ascontiguousarray(V, np.float64),
                           tri_verts=np.ascontiguousarray(F, np.uint64)))


def m_to_arrays(man):
    """Manifold -> (V float64, F int64, per-triangle originalID)."""
    mesh = man.to_mesh64()
    V = np.asarray(mesh.vert_properties, float)[:, :3].copy()
    F = np.asarray(mesh.tri_verts).astype(np.int64)
    ri = np.asarray(mesh.run_index).astype(np.int64) // 3
    ro = np.asarray(mesh.run_original_id).astype(np.int64)
    tri_orig = np.zeros(len(F), np.int64)
    for i, oid in enumerate(ro):
        tri_orig[ri[i]:ri[i + 1]] = oid
    return V, F, tri_orig


def frame_from_dir(w):
    """Orthonormal right-handed basis (u, v, w) with w given."""
    w = w / np.linalg.norm(w)
    ref = np.array([0, 0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(w, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(w, u), w


def m_cylinder_between(p0, p1, r, segs):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u, v, w = frame_from_dir(p1 - p0)
    M = np.column_stack([u, v, w, p0])
    return Manifold.cylinder(float(np.linalg.norm(p1 - p0)), r, -1.0, segs).transform(M)


def m_tube(path, r, nring):
    """Closed swept tube (rotation-minimising frames, capped ends) along a polyline path."""
    P = np.asarray(path, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.zeros_like(P)
    N[0] = frame_from_dir(T[0])[0]
    for i in range(1, n):                      # parallel transport of the normal
        v = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        N[i] = v / np.linalg.norm(v)
    B = np.cross(T, N)
    phi = 2 * np.pi * np.arange(nring) / nring
    ring = (P[:, None, :] + r * (np.cos(phi)[None, :, None] * N[:, None, :]
                                 + np.sin(phi)[None, :, None] * B[:, None, :]))
    V = np.vstack([ring.reshape(-1, 3), P[0], P[-1]])
    c0, c1 = n * nring, n * nring + 1
    F = []
    for i in range(n - 1):
        for j in range(nring):
            a, b = i * nring + j, i * nring + (j + 1) % nring
            c, d = b + nring, a + nring
            F += [(a, b, c), (a, c, d)]
    for j in range(nring):
        F.append((c0, (j + 1) % nring, j))
        F.append((c1, (n - 1) * nring + j, (n - 1) * nring + (j + 1) % nring))
    F = np.array(F, np.int64)
    if signed_volume(V, F) < 0:
        F = F[:, ::-1]
    m = m_from_arrays(V, F)
    assert str(m.status()) == "Error.NoError", m.status()
    return m


def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ----------------------------------------------------------------------------------------------
# 2. Part builders (spec inches -> world mm Manifold).  Each returns (manifold, {originalID: mat})
# ----------------------------------------------------------------------------------------------
def build_carcass(p, notes):
    lo, hi = mm(p["aabb"][0]), mm(p["aabb"][1])
    solid = m_box(lo, hi)
    run = p["run"]
    if p.get("toe_kick"):                      # notch along the full width at the front-bottom
        th, td = mm(p["toe_kick"])
        fr = FRONT_AX[run]
        nlo, nhi = lo - EPS, hi + EPS
        nlo[fr] = hi[fr] - td
        nhi[2] = lo[2] + th
        solid = solid - m_box(nlo, nhi)
    if p.get("subtract_box"):
        solid = solid - m_box(*mm(p["subtract_box"]))
    return solid, {}


def front_geometry(p):
    """Lateral/z outline + recessed panel rectangle (inches) for a shaker front."""
    run = p["run"]
    la, fr = LAT_AX[run], FRONT_AX[run]
    lo, hi = np.asarray(p["aabb"][0], float), np.asarray(p["aabb"][1], float)
    fw = p["frame_width"]
    pw, ph = (hi[la] - lo[la]) - 2 * fw, (hi[2] - lo[2]) - 2 * fw
    panel = None
    if pw >= 1.5 and ph >= 1.5:
        panel = (lo[la] + fw, hi[la] - fw, lo[2] + fw, hi[2] - fw)
    return dict(run=run, outline=(lo[la], hi[la], lo[2], hi[2]), face=hi[fr], back=lo[fr],
                panel=panel, recess=p.get("panel_recess", 0.25), panel_size=(pw, ph))


def build_shaker(p, notes):
    lo, hi = mm(p["aabb"][0]), mm(p["aabb"][1])
    solid = m_box(lo, hi)
    g = front_geometry(p)
    if g["panel"] is not None:                 # 0.25in pocket on the FRONT face
        la, fr = LAT_AX[p["run"]], FRONT_AX[p["run"]]
        plo, phi = lo.copy(), hi.copy()
        plo[la], phi[la], plo[2], phi[2] = mm(g["panel"])
        plo[fr], phi[fr] = hi[fr] - g["recess"] * IN, hi[fr] + EPS
        solid = solid - m_box(plo, phi)
    else:
        notes.append(f"{p['id']}: panel {g['panel_size']} < 1.5in -> plain slab")
    return solid, {}


def build_box(p, notes):
    return m_box(*mm(p["aabb"])), {}


def build_slab(p, notes):
    solid = m_box(*mm(p["aabb"]))
    for c in p.get("cutouts", []):
        solid = solid - m_box(*mm(c))
    return solid, {}


def build_union_boxes(p, notes):
    return Manifold.batch_boolean([m_box(*mm(b)) for b in p["boxes"]], OpType.Add), {}


def build_sink(p, notes):
    lo, hi = mm(p["aabb"][0]), mm(p["aabb"][1])
    w = p["wall"] * IN
    bowl = m_box(lo + [w, w, w], [hi[0] - w, hi[1] - w, hi[2] + EPS])
    dr = p["drain"]
    drain = Manifold.cylinder(w + 2 * EPS, dr["diameter"] * IN / 2, -1.0, SEG["drain"]).translate(
        (dr["center_xy"][0] * IN, dr["center_xy"][1] * IN, lo[2] - EPS))
    if abs((p["aabb"][1][2] - p["aabb"][0][2] - p["wall"]) - p["bowl_depth"]) > 1e-6:
        notes.append(f"{p['id']}: bowl_depth {p['bowl_depth']} != height - wall")
    return Manifold.batch_boolean([m_box(lo, hi), bowl, drain], OpType.Subtract), {}


def hose_path_in(p):
    """Half-ellipse hose path (inches) from body-top centre to head-top centre, apex at hose.apex_z,
    with short straight embeds into body/head so the union is robust (no coplanar caps)."""
    bx, by, bz = p["base"]
    d = np.array(p["arm"]["dir_xy"], float)
    d /= np.linalg.norm(d)
    hx, hy = np.array([bx, by]) + d * p["arm"]["length"]
    P0 = np.array([bx, by, bz + p["body_height"]])
    P1 = np.array([hx, hy, p["head"]["z1"]])
    zmid, half_dz = (P0[2] + P1[2]) / 2, (P1[2] - P0[2]) / 2
    A = math.sqrt((p["hose"]["apex_z"] - zmid) ** 2 - half_dz ** 2)   # exact apex: zmid+sqrt(hdz^2+A^2)
    t = np.linspace(0, np.pi, SEG["hose_path"])
    s = (1 - np.cos(t)) / 2
    xy = P0[:2][None] + (P1[:2] - P0[:2])[None] * s[:, None]
    z = zmid - half_dz * np.cos(t) + A * np.sin(t)
    arc = np.column_stack([xy, z])
    e = np.array([0, 0, HOSE_EMBED_IN])
    return np.vstack([P0 - e, arc, P1 - e]), arc


def build_faucet(p, notes):
    bx, by, bz = mm(p["base"])
    H = p["body_height"] * IN
    body = Manifold.cylinder(H, p["body_d_bottom"] * IN / 2, p["body_d_top"] * IN / 2,
                             SEG["body"]).translate((bx, by, bz)).as_original()
    d = np.array(p["arm"]["dir_xy"], float)
    d /= np.linalg.norm(d)
    end = np.array([bx, by]) + d * p["arm"]["length"] * IN
    az = p["arm"]["z"] * IN
    arm = m_cylinder_between((bx, by, az), (end[0], end[1], az), p["arm"]["d"] * IN / 2,
                             SEG["arm"]).as_original()
    hz0, hz1 = p["head"]["z0"] * IN, p["head"]["z1"] * IN
    head = Manifold.cylinder(hz1 - hz0, p["head"]["d"] * IN / 2, -1.0, SEG["head"]).translate(
        (end[0], end[1], hz0)).as_original()
    path, arc = hose_path_in(p)
    hose = m_tube(path * IN, p["hose"]["d"] * IN / 2, SEG["hose_ring"]).as_original()
    # simplify(1 um) collapses the one ~1e-6 mm^2 sliver where the hose ring meets the head's top rim;
    # originalIDs (-> hose material) survive simplification.
    solid = Manifold.batch_boolean([body, arm, head, hose], OpType.Add).simplify(FAUCET_SIMPLIFY_MM)
    notes.append(f"{p['id']}: hose apex z = {arc[:, 2].max():.4f} in (spec {p['hose']['apex_z']})")
    return solid, {hose.original_id(): p["hose"].get("material", "mat_faucet_hose")}


def pull_mount(p, finfo):
    """Where the catalog pull actually seats: classify each post footprint against the parent
    front's recessed panel.  Returns (contact plane coordinate along the front normal [in], info)."""
    L, post = p["length"], p["post"]
    lat, zc = p["mount_center"]
    off = L / 2 - post / 2
    centres = ([(lat, zc - off), (lat, zc + off)] if p["orientation"] == "vertical"
               else [(lat - off, zc), (lat + off, zc)])
    half = post / 2
    status = []
    for (cl, cz) in centres:
        r = (cl - half, cl + half, cz - half, cz + half)
        P = finfo["panel"]
        if P is not None and r[0] >= P[0] - 1e-9 and r[1] <= P[1] + 1e-9 and r[2] >= P[2] - 1e-9 and r[3] <= P[3] + 1e-9:
            status.append("panel")
        elif P is None or not (r[0] < P[1] and r[1] > P[0] and r[2] < P[3] and r[3] > P[2]):
            status.append("frame")
        else:
            status.append("straddle")
    O = finfo["outline"]
    bw = p["bar_w"]
    bar = ((lat - bw / 2, lat + bw / 2, zc - L / 2, zc + L / 2) if p["orientation"] == "vertical"
           else (lat - L / 2, lat + L / 2, zc - bw / 2, zc + bw / 2))
    inside = bar[0] >= O[0] and bar[1] <= O[1] and bar[2] >= O[2] and bar[3] <= O[3]
    surf = p["mount_face"]
    if PULL_MOUNT == "contact_surface" and all(s == "panel" for s in status):
        surf = p["mount_face"] - finfo["recess"]
    return surf, dict(post_status=status, bar_within_front=bool(inside),
                      spec_mount_face_in=p["mount_face"], contact_surface_in=surf,
                      surface="recessed_panel" if surf != p["mount_face"] else "frame_face",
                      offset_from_spec_mm=round((surf - p["mount_face"]) * IN, 6))


def build_pull(p, finfo, notes):
    """Bar pull in its own local frame (inches, pivot = centre of mounting footprint, +Z out),
    then placed in the world.  Posts overlap into the bar by bar_t/2 (identical outer shape)."""
    L, bw, bt, so, post = p["length"], p["bar_w"], p["bar_t"], p["standoff"], p["post"]
    ax, ac = (1, 0) if p["orientation"] == "vertical" else (0, 1)   # bar axis / across axis (local)
    off = L / 2 - post / 2
    pieces = []
    for c in (-off, off):
        lo, hi = np.zeros(3), np.zeros(3)
        lo[ax], hi[ax] = c - post / 2, c + post / 2
        lo[ac], hi[ac] = -post / 2, post / 2
        lo[2], hi[2] = 0.0, so + bt / 2
        pieces.append(m_box(lo * IN, hi * IN))
    lo, hi = np.zeros(3), np.zeros(3)
    lo[ax], hi[ax] = -L / 2, L / 2
    lo[ac], hi[ac] = -bw / 2, bw / 2
    lo[2], hi[2] = so, so + bt
    pieces.append(m_box(lo * IN, hi * IN))
    local = Manifold.batch_boolean(pieces, OpType.Add)
    surf, info = pull_mount(p, finfo)
    run = p["run"]
    piv = np.zeros(3)
    piv[LAT_AX[run]], piv[2], piv[FRONT_AX[run]] = p["mount_center"][0], p["mount_center"][1], surf
    piv *= IN
    R = RUN_ROT[run]
    if any(s == "straddle" for s in info["post_status"]) or len(set(info["post_status"])) > 1:
        notes.append(f"{p['id']}: posts straddle frame/panel {info['post_status']}")
    if not info["bar_within_front"]:
        notes.append(f"{p['id']}: bar footprint extends beyond parent front outline")
    return local.transform(np.column_stack([R, piv])), {}, piv, info


BUILDERS = dict(carcass=build_carcass, shaker_front=build_shaker, box=build_box, slab=build_slab,
                union_boxes=build_union_boxes, farmhouse_sink=build_sink, faucet=build_faucet)


# ----------------------------------------------------------------------------------------------
# 3. Part container, pivots, local frames
# ----------------------------------------------------------------------------------------------
class Part:
    def __init__(self, spec, materials):
        self.s = spec
        self.id = spec["id"]
        self.cat = category_of(self.id)
        self.layer = spec["layer"]
        self.run = spec.get("run")
        self.R = RUN_ROT[self.run]
        self.mats = [spec["material"]]
        self.extra = {}
        self.ref = self.layer == "reference"

    def set_geometry(self, man, matmap):
        V, F, orig = m_to_arrays(man)
        for mat in matmap.values():
            if mat not in self.mats:
                self.mats.append(mat)
        fmat = np.array([self.mats.index(matmap.get(o, self.mats[0])) for o in orig], np.int64)
        order = np.argsort(fmat, kind="stable")          # group faces by material (file order)
        self.Vw, self.F, self.fmat = V, F[order], fmat[order]

    def compute_pivot(self, finfo):
        s, run = self.s, self.run
        if self.cat == "door":
            la, fr = LAT_AX[run], FRONT_AX[run]
            piv = np.zeros(3)
            piv[la], piv[2], piv[fr] = s["hinge_lateral"], s["aabb"][0][2], s["aabb"][1][fr]   # front face
            self.pivot, self.pivot_type = piv * IN, "hinge_axis_front_face"
        elif self.cat == "drawer_front":
            la, fr = LAT_AX[run], FRONT_AX[run]
            lo, hi = np.asarray(s["aabb"][0], float), np.asarray(s["aabb"][1], float)
            piv = (lo + hi) / 2
            piv[fr] = lo[fr]
            self.pivot, self.pivot_type = piv * IN, "slide_axis_center_of_back_face"
        elif self.cat == "hardware_handle":
            self.pivot_type = "mounting_footprint_center_on_parent_front"   # set by build_pull
        elif s["type"] == "faucet":
            self.pivot, self.pivot_type = mm(s["base"]), "base_center_at_counter_surface"
        else:                                    # rear-bottom-left corner in the local frame
            m = (self.Vw @ self.R).min(axis=0)
            self.pivot, self.pivot_type = self.R @ m, "rear_bottom_left_corner"
        self.T = T44(self.R, self.pivot)
        self.Vl = (self.Vw - self.pivot) @ self.R      # world -> local (R orthonormal)


# ----------------------------------------------------------------------------------------------
# 4. Normals + UV atlas (per part, local frame)
# ----------------------------------------------------------------------------------------------
def corner_normals(V, F, thr_deg=SMOOTH_DEG):
    """Area-weighted normals per face corner, smoothing only across edges below thr_deg."""
    tm = trimesh.Trimesh(V, F, process=False)
    fn, fa, vf = tm.face_normals, tm.area_faces, tm.vertex_faces
    cos_t = math.cos(math.radians(thr_deg))
    out = np.zeros((len(F), 3, 3))
    for k in range(3):
        inc = vf[F[:, k]]
        valid = inc >= 0
        incn = fn[inc.clip(0)]
        w = valid & (np.einsum("fdk,fk->fd", incn, fn) > cos_t)
        acc = np.einsum("fd,fdk->fk", w * fa[inc.clip(0)], incn)
        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        out[:, k] = np.where(nrm > 1e-12, acc / np.maximum(nrm, 1e-12), fn)
    flat = np.round(out.reshape(-1, 3), 6) + 0.0
    vn, inv = np.unique(flat, axis=0, return_inverse=True)
    return vn, inv.reshape(-1, 3)


def project_axis(label, P):
    """Unmirrored planar projection (mm) for dominant-normal label 0..5 = +X,-X,+Y,-Y,+Z,-Z."""
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    conds = [label == i for i in range(6)]
    u = np.select(conds, [-Z, Z, X, X, X, -X])
    v = np.select(conds, [Y, Y, -Z, Z, Y, Y])
    return np.column_stack([u, v])


def shelf_pack(sizes, margin):
    """Shelf packer (no rotation, keeps texture grain along local Y); searches the shelf width
    that minimises the square atlas side.  Returns offsets (mm) and atlas side S (mm)."""
    n = len(sizes)
    order = sorted(range(n), key=lambda i: (-sizes[i][1], -sizes[i][0], i))
    maxw = max(w for w, _ in sizes)
    lo_w, hi_w = maxw + 2 * margin, sum(w + margin for w, _ in sizes) + margin
    cands = sorted(set(np.round(np.geomspace(lo_w, max(hi_w, lo_w), 64), 6).tolist()))
    best = None
    for W in cands:
        x = y = margin
        shelf = 0.0
        pos = [None] * n
        for i in order:
            w, h = sizes[i]
            if x + w + margin > W + 1e-9 and x > margin:
                y += shelf + margin
                x, shelf = margin, 0.0
            pos[i] = (x, y)
            x += w + margin
            shelf = max(shelf, h)
        H = y + shelf + margin
        S = max(W, H)
        if best is None or S < best[0] - 1e-9:
            best = (S, pos)
    return np.array(best[1]), best[0]


def uv_atlas(V, F):
    """Charts = edge-connected faces sharing a dominant normal axis; planar projection in mm;
    shelf-packed into [0,1]^2 with one uniform scale.  UV seams never split positions (vt index)."""
    tm = trimesh.Trimesh(V, F, process=False)
    fn = tm.face_normals
    ax = np.abs(fn).argmax(axis=1)
    label = ax * 2 + (fn[np.arange(len(F)), ax] < 0)
    adj = tm.face_adjacency
    a = adj[label[adj[:, 0]] == label[adj[:, 1]]]
    nf = len(F)
    G = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(nf, nf))
    nc, comp = connected_components(G, directed=False)
    chart_label = np.zeros(nc, np.int64)
    chart_label[comp] = label
    nv = len(V)
    keys = comp[:, None] * nv + F
    ukeys, inv = np.unique(keys.ravel(), return_inverse=True)
    vt_chart, vt_vert = ukeys // nv, ukeys % nv
    uvmm = project_axis(chart_label[vt_chart], V[vt_vert])
    cmin = np.full((nc, 2), np.inf)
    cmax = np.full((nc, 2), -np.inf)
    np.minimum.at(cmin, vt_chart, uvmm)
    np.maximum.at(cmax, vt_chart, uvmm)
    sizes = [tuple(x) for x in (cmax - cmin)]
    area = sum((w + 1) * (h + 1) for w, h in sizes)
    margin = max(1.5, 0.006 * math.sqrt(area))
    off, S = shelf_pack(sizes, margin)
    if margin < 0.004 * S:                    # keep >= ~4 px gutters at 1024^2
        margin = 0.004 * S
        off, S = shelf_pack(sizes, margin)
    uv = (uvmm - cmin[vt_chart] + off[vt_chart]) / S
    return dict(uv=uv, vt_idx=inv.reshape(-1, 3), chart_of_face=comp, n_charts=int(nc),
                mm_per_unit=float(S), margin_mm=float(margin))


def raster_uv_overlap(uv, vt_idx, res=UV_RES):
    """Rasterise UV triangles (strict-interior test at pixel centres) and count pixels covered by
    more than one triangle.  Shared edges are never double counted; any true overlap is."""
    cnt = np.zeros((res, res), np.int32)
    tri = uv[vt_idx] * res
    for p in tri:
        a2 = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])
        if abs(a2) < 1e-12:
            continue
        x0, x1 = max(int(math.floor(p[:, 0].min() - 0.5)), 0), min(int(math.ceil(p[:, 0].max())), res - 1)
        y0, y1 = max(int(math.floor(p[:, 1].min() - 0.5)), 0), min(int(math.ceil(p[:, 1].max())), res - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        sg = 1.0 if a2 > 0 else -1.0
        tol = 1e-9 * abs(a2)
        inside = np.ones(xs.shape, bool)
        for i in range(3):
            A, B = p[i], p[(i + 1) % 3]
            e = (B[0] - A[0]) * (ys - A[1]) - (B[1] - A[1]) * (xs - A[0])
            inside &= sg * e > tol
        cnt[y0:y1 + 1, x0:x1 + 1] += inside
    oob = int(((uv < -1e-9) | (uv > 1 + 1e-9)).any(axis=1).sum())
    return int((cnt >= 2).sum()), int((cnt >= 1).sum()), oob


# ----------------------------------------------------------------------------------------------
# 5. Writers
# ----------------------------------------------------------------------------------------------
def mtl_text(materials, title):
    out = [f"# {title}", "# PBR extension keys: Pr = roughness, Pm = metallic (baseColor used as-is)"]
    for mid, m in materials.items():
        c = np.asarray(m["baseColor"], float)
        ks = 0.04 * (1 - m["metallic"]) + c * m["metallic"]
        r = max(m["roughness"], 0.01)
        ns = min(max(2.0 / r ** 4 - 2.0, 1.0), 1000.0)      # Blinn-Phong exponent from roughness
        out += ["", f"newmtl {mid}", f"# {m.get('note', '')}",
                "Ka 0.000000 0.000000 0.000000", "Kd {:.6f} {:.6f} {:.6f}".format(*c),
                "Ks {:.6f} {:.6f} {:.6f}".format(*ks), f"Ns {ns:.4f}", "d 1.000000", "illum 2",
                f"Pr {m['roughness']:.6f}", f"Pm {m['metallic']:.6f}"]
    return "\n".join(out) + "\n"


def obj_block(name, V, uv, vn, F, vt_idx, vn_idx, fmat, mats, base=(0, 0, 0)):
    """One OBJ object block; indices offset by base=(nv, nvt, nvn) already written."""
    bv, bt, bn = base
    L = [f"o {name}", f"g {name}"]
    L += ["v %.6f %.6f %.6f" % tuple(x) for x in V]
    L += ["vt %.6f %.6f" % tuple(x) for x in uv]
    L += ["vn %.6f %.6f %.6f" % tuple(x) for x in vn]
    cur = None
    fi, ti, ni = F + 1 + bv, vt_idx + 1 + bt, vn_idx + 1 + bn
    for f in range(len(F)):
        if fmat[f] != cur:
            cur = fmat[f]
            L.append(f"usemtl {mats[cur]}")
        L.append("f %d/%d/%d %d/%d/%d %d/%d/%d" % (fi[f, 0], ti[f, 0], ni[f, 0], fi[f, 1], ti[f, 1],
                                                   ni[f, 1], fi[f, 2], ti[f, 2], ni[f, 2]))
    return L


def write_stl(path, V, F, header="keane-kitchens build_cad.py binary STL, units mm"):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    arr = np.zeros(len(F), dt)
    arr["n"], arr["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(header.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(F)))
        fh.write(arr.tobytes())


def read_obj_positions(path):
    """Minimal OBJ reader: positions + triangle position indices (no merging, no processing)."""
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V, float), np.array(F, np.int64)


def write_glb(path, parts, materials):
    """glTF 2.0 binary.  Root node 'kitchen' converts world Z-up mm -> glTF Y-up metres; one node per
    part (name = id) holding the part's LOCAL mesh, nested by parent (doors under carcasses, pulls
    under fronts) with relative matrices, so a viewer can swing doors about the node's local +Y."""
    mat_ids = list(materials)
    gl_mats = [dict(name=m, pbrMetallicRoughness=dict(
        baseColorFactor=list(materials[m]["baseColor"]) + [1.0], metallicFactor=materials[m]["metallic"],
        roughnessFactor=materials[m]["roughness"])) for m in mat_ids]
    blob = bytearray()
    views, accs = [], []

    def add(arr, typ, comp, target, minmax=False):
        while len(blob) % 4:
            blob.append(0)
        b = arr.tobytes()
        views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(b), target=target))
        blob.extend(b)
        acc = dict(bufferView=len(views) - 1, componentType=comp, count=int(len(arr)), type=typ)
        if minmax:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accs.append(acc)
        return len(accs) - 1

    meshes, nodes = [], []
    index = {p.id: i + 1 for i, p in enumerate(parts)}           # node 0 = root
    byid = {p.id: p for p in parts}
    P = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    root = np.diag([0.001, 0.001, 0.001, 1.0]) @ P
    nodes.append(dict(name="kitchen", matrix=root.T.ravel().tolist(), children=[],
                      extras=dict(note="world Z-up mm -> glTF Y-up m")))
    for p in parts:
        corners = np.stack([p.F, p.uvd["vt_idx"], p.vn_idx], axis=-1).reshape(-1, 3)
        uc, inv = np.unique(corners, axis=0, return_inverse=True)
        pos = p.Vl[uc[:, 0]].astype(np.float32)
        nrm = p.vn[uc[:, 2]].astype(np.float32)
        tex = p.uvd["uv"][uc[:, 1]].copy()
        tex[:, 1] = 1.0 - tex[:, 1]                                 # glTF UV origin is top-left
        tex = tex.astype(np.float32)
        ia = add(pos, "VEC3", 5126, 34962, True)
        ina = add(nrm, "VEC3", 5126, 34962)
        ita = add(tex, "VEC2", 5126, 34962)
        idx = inv.reshape(-1, 3).astype(np.uint32)
        prims = []
        for mi, mat in enumerate(p.mats):
            sel = idx[p.fmat == mi].ravel()
            if len(sel):
                prims.append(dict(attributes=dict(POSITION=ia, NORMAL=ina, TEXCOORD_0=ita),
                                  indices=add(sel, "SCALAR", 5125, 34963), material=mat_ids.index(mat), mode=4))
        meshes.append(dict(name=p.id, primitives=prims))
        par = p.s.get("parent")
        rel = np.linalg.inv(byid[par].T) @ p.T if par in byid else p.T
        nodes.append(dict(name=p.id, mesh=len(meshes) - 1, matrix=np.round(rel.T.ravel(), 9).tolist(),
                          extras=dict(layer=p.layer, category=p.cat, materials=p.mats, pivot=p.pivot_type)))
    for p in parts:
        par = p.s.get("parent")
        holder = nodes[index[par]] if par in byid else nodes[0]
        holder.setdefault("children", []).append(index[p.id])
    gltf = dict(asset=dict(version="2.0", generator="keane-kitchens tools/build_cad.py"), scene=0,
                scenes=[dict(name="kitchen", nodes=[0])], nodes=nodes, meshes=meshes, materials=gl_mats,
                accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(blob)))
        fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))


# ----------------------------------------------------------------------------------------------
# 6. STEP (optional, cadquery / OCCT B-rep)
# ----------------------------------------------------------------------------------------------
def build_step(parts, materials, path, finfo):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    def box(lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return cq.Solid.makeBox(*(hi - lo), pnt=cq.Vector(*lo))

    def trsf(T):                       # rigid 4x4 -> gp_Trsf (cq.Matrix would build a general GTrsf)
        t = gp_Trsf()
        t.SetValues(*[float(x) for x in T[:3].ravel()])
        return t

    def moved(sol, T):                 # bake a rigid transform into the B-rep geometry
        return cq.Shape.cast(BRepBuilderAPI_Transform(sol.wrapped, trsf(T), True).Shape())

    def solid_world(p):
        s, t = p.s, p.s["type"]
        if t == "carcass":
            lo, hi = mm(s["aabb"][0]), mm(s["aabb"][1])
            sol = box(lo, hi)
            if s.get("toe_kick"):
                th, td = mm(s["toe_kick"])
                fr = FRONT_AX[s["run"]]
                nlo, nhi = lo - EPS, hi + EPS
                nlo[fr], nhi[2] = hi[fr] - td, lo[2] + th
                sol = sol.cut(box(nlo, nhi))
            if s.get("subtract_box"):
                sol = sol.cut(box(*mm(s["subtract_box"])))
            return sol
        if t == "shaker_front":
            lo, hi = mm(s["aabb"][0]), mm(s["aabb"][1])
            sol = box(lo, hi)
            g = front_geometry(s)
            if g["panel"] is not None:
                la, fr = LAT_AX[s["run"]], FRONT_AX[s["run"]]
                plo, phi = lo.copy(), hi.copy()
                plo[la], phi[la], plo[2], phi[2] = mm(g["panel"])
                plo[fr], phi[fr] = hi[fr] - g["recess"] * IN, hi[fr] + EPS
                sol = sol.cut(box(plo, phi))
            return sol
        if t in ("box",):
            return box(*mm(s["aabb"]))
        if t == "slab":
            sol = box(*mm(s["aabb"]))
            for c in s.get("cutouts", []):
                sol = sol.cut(box(*mm(c)))
            return sol
        if t == "union_boxes":
            sols = [box(*mm(b)) for b in s["boxes"]]
            sol = sols[0]
            for o in sols[1:]:
                sol = sol.fuse(o)
            return sol
        if t == "farmhouse_sink":
            lo, hi = mm(s["aabb"][0]), mm(s["aabb"][1])
            w = s["wall"] * IN
            dr = s["drain"]
            cyl = cq.Solid.makeCylinder(dr["diameter"] * IN / 2, w + 2 * EPS,
                                        cq.Vector(dr["center_xy"][0] * IN, dr["center_xy"][1] * IN, lo[2] - EPS))
            return box(lo, hi).cut(box(lo + [w, w, w], [hi[0] - w, hi[1] - w, hi[2] + EPS])).cut(cyl)
        if t == "bar_pull":
            L, bw, bt, so, post = (s[k] * IN for k in ("length", "bar_w", "bar_t", "standoff", "post"))
            ax, ac = (1, 0) if s["orientation"] == "vertical" else (0, 1)
            sols = []
            for c in (-(L / 2 - post / 2), L / 2 - post / 2):
                lo, hi = np.zeros(3), np.zeros(3)
                lo[ax], hi[ax], lo[ac], hi[ac], hi[2] = c - post / 2, c + post / 2, -post / 2, post / 2, so + bt / 2
                sols.append(box(lo, hi))
            lo, hi = np.zeros(3), np.zeros(3)
            lo[ax], hi[ax], lo[ac], hi[ac], lo[2], hi[2] = -L / 2, L / 2, -bw / 2, bw / 2, so, so + bt
            sol = box(lo, hi).fuse(sols[0]).fuse(sols[1])
            return moved(sol, p.T)
        if t == "faucet":
            bx, by, bz = mm(s["base"])
            body = cq.Solid.makeCone(s["body_d_bottom"] * IN / 2, s["body_d_top"] * IN / 2,
                                     s["body_height"] * IN, cq.Vector(bx, by, bz))
            d = np.array(s["arm"]["dir_xy"], float)
            d /= np.linalg.norm(d)
            end = np.array([bx, by]) + d * s["arm"]["length"] * IN
            az = s["arm"]["z"] * IN
            arm = cq.Solid.makeCylinder(s["arm"]["d"] * IN / 2, s["arm"]["length"] * IN,
                                        cq.Vector(bx, by, az), cq.Vector(d[0], d[1], 0))
            head = cq.Solid.makeCylinder(s["head"]["d"] * IN / 2, (s["head"]["z1"] - s["head"]["z0"]) * IN,
                                         cq.Vector(end[0], end[1], s["head"]["z0"] * IN))
            path, _ = hose_path_in(s)
            pts = [cq.Vector(*q) for q in path[1:-1] * IN]
            up, down = cq.Vector(0, 0, 1), cq.Vector(0, 0, -1)
            spine = cq.Wire.assembleEdges([cq.Edge.makeLine(cq.Vector(*(path[0] * IN)), pts[0]),
                                           cq.Edge.makeSpline(pts, tangents=[up, down]),
                                           cq.Edge.makeLine(pts[-1], cq.Vector(*(path[-1] * IN)))])
            prof = cq.Wire.makeCircle(s["hose"]["d"] * IN / 2, cq.Vector(*(path[0] * IN)), up)
            hose = cq.Solid.sweep(prof, [], spine, makeSolid=True, isFrenet=False, transitionMode="right")
            return body.fuse(arm).fuse(head).fuse(hose).clean()
        return None

    assy = cq.Assembly(name="keane_kitchen")
    vols, skipped = {}, {}
    for p in parts:
        try:
            sol = solid_world(p)
        except Exception as e:                       # noqa: BLE001 - record and skip the part
            skipped[p.id] = f"{type(e).__name__}: {e}"
            continue
        if sol is None or not sol.isValid():
            skipped[p.id] = "invalid or unsupported B-rep"
            continue
        sol = sol.clean() if p.s["type"] != "faucet" else sol
        vols[p.id] = sol.Volume()
        local = moved(sol, np.linalg.inv(p.T))          # solids stored in the component's LOCAL frame
        kw = {}
        if STEP_COLORS:                               # OCCT writes colour styles in hash order (non-deterministic)
            kw["color"] = cq.Color(*materials[p.mats[0]]["baseColor"], 1.0)
        assy.add(local, name=p.id, loc=cq.Location(trsf(p.T)), **kw)
    assy.export(path, "STEP")
    txt = open(path).read()                          # pin the timestamp -> byte-deterministic file
    txt = re.sub(r"(FILE_NAME\('[^']*',)'[^']*'", r"\1'2026-01-01T00:00:00'", txt, count=1)
    open(path, "w").write(txt)
    back = cq.importers.importStep(path).solids().vals()      # re-import proof
    reimport = dict(solids=len(back), total_volume_mm3=round(sum(s.Volume() for s in back), 1),
                    all_valid=bool(all(s.isValid() for s in back)))
    return vols, skipped, reimport


# ----------------------------------------------------------------------------------------------
# 7. Previews
# ----------------------------------------------------------------------------------------------
def shade_colors(tris, base, light):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    s = 0.35 + 0.65 * np.abs(n @ light)
    return np.clip(base * s[:, None], 0, 1)


def preview_iso(parts, materials, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tris, cols = [], []
    for p in parts:
        if p.ref and not p.id.startswith("reference_floor"):
            continue                                   # walls would hide the run in a painter's sort
        # mplot3d sorts whole polygons by centroid depth: split long faces so the sort is reliable
        V, F, idx = trimesh.remesh.subdivide_to_size(p.Vw, p.F, max_edge=ISO_MAX_EDGE_MM, return_index=True)
        tris.append(V[F])
        cols.append(np.array([materials[p.mats[i]]["baseColor"] for i in p.fmat[idx]]))
    tris, cols = np.vstack(tris), np.vstack(cols)
    light = np.array([0.45, 0.65, 0.62])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(12, 10), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    fc = shade_colors(tris, cols, light)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.15))  # hide AA seams
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=22, azim=40)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_title("Keane Kitchens CAD reconstruction - world mm, Z-up (walls hidden)")
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def preview_camera(parts, materials, cams_path, img_path, out_path, scale=0.25):
    """Z-buffer render from SfM camera 02 (SIMPLE_RADIAL) + overlay on the photo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    cams = json.load(open(cams_path))["images"]
    key = [k for k in cams if k.startswith("02")][0]
    cam = cams[key]
    W, H = int(cam["width"] * scale), int(cam["height"] * scale)
    f, cx, cy, k1 = cam["f_px"] * scale, cam["K"][0][2] * scale, cam["K"][1][2] * scale, cam["distortion"]["k1"]
    Rc, tc = np.array(cam["R_world_to_cam"]), np.array(cam["t_world_to_cam"]) * IN
    fwd = Rc[2]                                    # camera optical axis in world (constant -> flat faces)

    def render(sel):
        tris, cols = [], []
        for p in sel:
            V, F = p.Vw, p.F
            fm = p.fmat
            if p.ref:
                V, F, idx = trimesh.remesh.subdivide_to_size(V, F, max_edge=150.0, return_index=True)
                fm = p.fmat[idx]
            t = V[F]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            base = np.array([materials[p.mats[i]]["baseColor"] for i in fm])
            tris.append(t)
            cols.append(np.clip(base * (0.3 + 0.7 * np.abs(n @ fwd))[:, None], 0, 1))
        tris, cols = np.vstack(tris), np.vstack(cols)
        Pc = tris.reshape(-1, 3) @ Rc.T + tc
        Pc = Pc.reshape(-1, 3, 3)
        keep = (Pc[..., 2] > 50.0).all(1)
        Pc, cols = Pc[keep], cols[keep]
        x, y = Pc[..., 0] / Pc[..., 2], Pc[..., 1] / Pc[..., 2]
        d = 1 + k1 * (x * x + y * y)
        u, v, z = f * x * d + cx, f * y * d + cy, Pc[..., 2]
        img = np.ones((H, W, 3)) * 0.97
        zb = np.full((H, W), np.inf)
        mask = np.zeros((H, W), bool)
        for i in range(len(u)):
            pu, pv, pz = u[i], v[i], z[i]
            x0, x1 = max(int(pu.min()), 0), min(int(math.ceil(pu.max())), W - 1)
            y0, y1 = max(int(pv.min()), 0), min(int(math.ceil(pv.max())), H - 1)
            if x1 < x0 or y1 < y0:
                continue
            a2 = (pu[1] - pu[0]) * (pv[2] - pv[0]) - (pv[1] - pv[0]) * (pu[2] - pu[0])
            if abs(a2) < 1e-9:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((pu[1] - xs) * (pv[2] - ys) - (pv[1] - ys) * (pu[2] - xs)) / a2
            w1 = ((pu[2] - xs) * (pv[0] - ys) - (pv[2] - ys) * (pu[0] - xs)) / a2
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not ins.any():
                continue
            depth = 1.0 / (w0 / pz[0] + w1 / pz[1] + w2 / pz[2])
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            m = ins & (depth < sub)
            sub[m] = depth[m]
            img[y0:y1 + 1, x0:x1 + 1][m] = cols[i]
            mask[y0:y1 + 1, x0:x1 + 1][m] = True
        return img, mask

    full, _ = render(parts)
    fab, fmask = render([p for p in parts if not p.ref])
    photo = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    if photo.size != (cam["width"], cam["height"]):
        raise RuntimeError(f"photo size {photo.size} != camera {cam['width']}x{cam['height']}")
    photo = np.asarray(photo.resize((W, H), Image.BILINEAR), float) / 255.0
    blend = photo.copy()
    blend[fmask] = 0.45 * photo[fmask] + 0.55 * fab[fmask]
    fig, axs = plt.subplots(1, 2, figsize=(12, 8.2), dpi=120)
    for a, im, t in zip(axs, [full, blend], ["model rendered from SfM camera 02", "fabrication parts over photo 02"]):
        a.imshow(im)
        a.set_title(t)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, metadata={"Software": None})
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# 7b. Door swing sweep
# ----------------------------------------------------------------------------------------------
def door_swing_sweep(parts, byid, children):
    """Rotate each door (+ its pulls) about its pivot (front-face hinge line, local +Y) in
    SWING_STEP_DEG steps toward its open direction; max_open_deg = largest angle with no
    intersection > INTERSECT_FLAG_MM3 against any other fabrication part (cap DOOR_OPEN_DEG)."""
    statics = OrderedDict((p.id, m_from_arrays(p.Vw, p.F)) for p in parts if not p.ref)
    sbb = {k: np.array(m.bounding_box()).reshape(2, 3) for k, m in statics.items()}
    out = OrderedDict()
    for d in [p for p in parts if p.cat == "door"]:
        kids = children[d.id]
        local = Manifold.batch_boolean([m_from_arrays(d.Vl, d.F)] +
                                       [m_from_arrays((byid[k].Vw - d.pivot) @ d.R, byid[k].F) for k in kids],
                                       OpType.Add)
        sgn = -1 if d.s["hinge_side"] == "left" else 1
        max_open, first = int(DOOR_OPEN_DEG), None
        for a in range(SWING_STEP_DEG, int(DOOR_OPEN_DEG) + 1, SWING_STEP_DEG):
            th = math.radians(sgn * a)
            Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
            moving = local.transform(np.column_stack([d.R @ Ry, d.pivot]))
            mb = np.array(moving.bounding_box()).reshape(2, 3)
            hits = {}
            for sid, m in statics.items():
                if sid == d.id or sid in kids or np.any(mb[0] > sbb[sid][1]) or np.any(sbb[sid][0] > mb[1]):
                    continue
                v = float((moving ^ m).volume())
                if v > INTERSECT_FLAG_MM3:
                    hits[sid] = round(v, 3)
            if hits:
                max_open = a - SWING_STEP_DEG
                first = dict(angle_deg=a, parts=sorted(hits), intersection_mm3=hits)
                break
        out[d.id] = OrderedDict(hinge_side=d.s["hinge_side"], max_open_deg=max_open,
                                first_collision=first, swept_with=kids)
    return out


# ----------------------------------------------------------------------------------------------
# 8. Main pipeline
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default="artifacts/geometry/layout_spec.json")
    ap.add_argument("--no-step", action="store_true", help="skip the optional STEP export")
    ap.add_argument("--no-preview", action="store_true", help="skip preview PNGs")
    args = ap.parse_args()
    t_start = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = args.spec if os.path.isabs(args.spec) else os.path.join(root, args.spec)
    spec = json.load(open(spec_path))
    assert spec.get("units") == "in", "spec units must be inches"
    materials = spec["materials"]
    cad = os.path.join(root, "cad_output")
    comp_dir, ref_dir = os.path.join(cad, "components"), os.path.join(cad, "components", "reference")
    exp = os.path.join(root, "artifacts", "export")
    for d in (comp_dir, exp):                   # deterministic regeneration: wipe our own outputs
        shutil.rmtree(d, ignore_errors=True)
    for d in (comp_dir, ref_dir, exp):
        os.makedirs(d, exist_ok=True)
    notes = []                                 # automatic spec observations
    timings = OrderedDict()

    # ---- 8.1 geometry (world mm) ------------------------------------------------------------
    t0 = time.time()
    parts, byid, finfo = [], {}, {}
    for ps in spec["parts"]:
        p = Part(ps, materials)
        parts.append(p)
        byid[p.id] = p
        if ps["type"] == "shaker_front":
            finfo[p.id] = front_geometry(ps)
    for p in parts:
        t = p.s["type"]
        if t == "bar_pull":
            man, matmap, piv, info = build_pull(p.s, finfo[p.s["parent"]], notes)
            p.pivot, p.extra["mount"] = piv, info
        else:
            man, matmap = BUILDERS[t](p.s, notes)
        assert str(man.status()) == "Error.NoError", (p.id, man.status())
        p.set_geometry(man, matmap)
        p.compute_pivot(finfo)
    timings["geometry_s"] = time.time() - t0

    # ---- 8.2 spec consistency checks that need geometry ---------------------------------------
    for p in parts:
        if p.cat == "door":
            la = LAT_AX[p.run]
            lo, hi = p.s["aabb"][0][la], p.s["aabb"][1][la]
            hl = p.s["hinge_lateral"]
            at_min = abs(hl - lo) < 1e-9
            if not (at_min or abs(hl - hi) < 1e-9):
                notes.append(f"{p.id}: hinge_lateral {hl} not on a lateral edge")
            expect = ("right" if at_min else "left") if p.run == "L" else ("left" if at_min else "right")
            if expect != p.s["hinge_side"]:
                notes.append(f"{p.id}: hinge_side {p.s['hinge_side']} disagrees with hinge_lateral (expected {expect})")
            xs = p.Vl[:, 0]
            ok = (abs(xs.min()) < 1e-6) if p.s["hinge_side"] == "left" else (abs(xs.max()) < 1e-6)
            if not ok:
                notes.append(f"{p.id}: local X range [{xs.min():.3f},{xs.max():.3f}] violates hinge convention")
            if abs(p.Vl[:, 2].max()) > 1e-6 or p.Vl[:, 2].min() > -1e-6:
                notes.append(f"{p.id}: local Z range violates front-face pivot (expected [-t, 0])")
        if p.cat == "hardware_handle" and p.extra["mount"]["surface"] == "recessed_panel":
            pass   # summarised below
    recessed = [p.id for p in parts if p.cat == "hardware_handle" and p.extra["mount"]["surface"] == "recessed_panel"]
    if recessed:
        notes.append(f"{len(recessed)} pulls have mount_face = frame face but both posts land on the 0.25in-recessed "
                     f"panel -> seated on the panel surface (PULL_MOUNT='{PULL_MOUNT}'): {', '.join(recessed)}")

    # ---- 8.3 normals, UVs, component files ----------------------------------------------------
    t0 = time.time()
    mtl = mtl_text(materials, "Keane Kitchens material slots (swappable) - generated by tools/build_cad.py")
    for d in (comp_dir, ref_dir):
        open(os.path.join(d, "materials.mtl"), "w").write(mtl)
    for p in parts:
        p.vn, p.vn_idx = corner_normals(p.Vl, p.F)
        p.uvd = uv_atlas(p.Vl, p.F)
        d = ref_dir if p.ref else comp_dir
        p.rel_obj = os.path.relpath(os.path.join(d, p.id + ".obj"), cad)
        p.rel_stl = os.path.relpath(os.path.join(d, p.id + ".stl"), cad)
        hdr = [f"# {p.id} - {p.s.get('label', '')}", "# units mm, LOCAL frame (Y-up, +Z out of front), pivot at origin",
               f"# uv_mm_per_unit {p.uvd['mm_per_unit']:.6f}", "mtllib materials.mtl"]
        body = obj_block(p.id, p.Vl, p.uvd["uv"], p.vn, p.F, p.uvd["vt_idx"], p.vn_idx, p.fmat, p.mats)
        open(os.path.join(cad, p.rel_obj), "w").write("\n".join(hdr + body) + "\n")
        write_stl(os.path.join(cad, p.rel_stl), p.Vl, p.F)
    timings["components_s"] = time.time() - t0

    # ---- 8.4 OBBs, swing / slide data, assembly manifest --------------------------------------
    t0 = time.time()
    children = {p.id: [] for p in parts}
    for p in parts:
        if p.s.get("parent") in children:
            children[p.s["parent"]].append(p.id)
    swing = door_swing_sweep(parts, byid, children)
    comps = []
    obb_dev = 0.0
    for p in parts:
        lo, hi = p.Vl.min(0), p.Vl.max(0)
        ext = hi - lo
        tm = trimesh.Trimesh(p.Vw, p.F, process=False)
        ob = tm.bounding_box_oriented
        ob_ext = np.asarray(ob.primitive.extents, float)
        if np.prod(ext) <= np.prod(ob_ext) * (1 + 1e-6):
            axes, dims = p.R.copy(), ext
            cen = p.R @ ((lo + hi) / 2) + p.pivot
            method = "local_aabb (trimesh OBB volume equal within 1e-6)"
            obb_dev = max(obb_dev, float(np.abs(np.sort(ob_ext) - np.sort(ext)).max()))
        else:
            OT = np.asarray(ob.primitive.transform, float)
            A = OT[:3, :3]
            used, axes, dims = [], np.zeros((3, 3)), np.zeros(3)
            for i in range(3):                   # match OBB axes to local X, Y, Z
                dots = [abs(A[:, j] @ p.R[:, i]) if j not in used else -1 for j in range(3)]
                j = int(np.argmax(dots))
                used.append(j)
                axes[:, i] = A[:, j] * np.sign(A[:, j] @ p.R[:, i] or 1.0)
                dims[i] = ob_ext[j]
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1
            cen, method = OT[:3, 3], "trimesh_bounding_box_oriented"
        p.obb = dict(width_mm=float(dims[0]), height_mm=float(dims[1]), depth_mm=float(dims[2]),
                     center_world_mm=r6(cen), axes=r6(axes.T), method=method,
                     transform=r6(T44(axes, cen)))
        win = ext / IN
        c = OrderedDict(id=p.id, label=p.s.get("label", ""), category=p.cat, layer=p.layer, run=p.run,
                        type=p.s["type"], parent=p.s.get("parent"), children=children[p.id],
                        files=dict(obj=p.rel_obj, stl=p.rel_stl), material_slots=p.mats,
                        material_default=p.mats[0],
                        pivot=dict(type=p.pivot_type, world_position_mm=r6(p.pivot)),
                        transform_local_to_world=r6(p.T),
                        local_bounds_mm=dict(min=r6(lo), max=r6(hi)), obb=p.obb,
                        nominal_size_in=f"{fmt_in(win[0])}W x {fmt_in(win[1])}H x {fmt_in(win[2])}D",
                        nominal_size_frac=f"{frac_in(win[0])}W x {frac_in(win[1])}H x {frac_in(win[2])}D",
                        uv=dict(charts=p.uvd["n_charts"], uv_mm_per_unit=round(p.uvd["mm_per_unit"], 6),
                                margin_mm=round(p.uvd["margin_mm"], 4)))
        if p.cat == "door":
            ang = -DOOR_OPEN_DEG if p.s["hinge_side"] == "left" else DOOR_OPEN_DEG

            def free_edge_z(a):
                th = math.radians(a)
                Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
                xs = p.Vl[:, 0]
                edge = p.Vl[np.abs(np.abs(xs) - np.abs(xs).max()) < 1e-6]
                return float((edge @ Ry.T)[:, 2].mean())
            zf, zr = free_edge_z(ang), free_edge_z(-ang)
            c["door_swing"] = dict(hinge_side=p.s["hinge_side"], axis_local=[0, 1, 0], axis_world=r6(p.R[:, 1]),
                                   axis_point_world_mm=r6(p.pivot), open_angle_deg=ang,
                                   convention="right-hand rotation about local +Y through the pivot "
                                              "(hinge line on the front face)",
                                   max_open_deg=swing[p.id]["max_open_deg"],
                                   first_collision=swing[p.id]["first_collision"],
                                   sweep=f"door + pulls vs all other fabrication parts, {SWING_STEP_DEG} deg steps, "
                                         f"cap {DOOR_OPEN_DEG:g}, flag > {INTERSECT_FLAG_MM3} mm3",
                                   verified=dict(free_edge_mean_z_mm_at_open=round(zf, 3),
                                                 free_edge_mean_z_mm_opposite_sign=round(zr, 3),
                                                 opens_toward_plus_z=bool(zf > 0 > zr)))
            if not zf > 0 > zr:
                notes.append(f"{p.id}: swing sign check FAILED")
        if p.cat == "drawer_front":
            par = byid[p.s["parent"]]
            fr = FRONT_AX[p.run]
            depth_in = par.s["aabb"][1][fr] - par.s["aabb"][0][fr]
            c["drawer_slide"] = dict(role=p.s["role"], axis_local=[0, 0, 1], axis_world=r6(p.R[:, 2]),
                                     parent_carcass_depth_in=depth_in, travel_in=depth_in - 3.0,
                                     travel_mm=round((depth_in - 3.0) * IN, 4), rule="travel = carcass depth - 3 in")
        if p.cat == "hardware_handle":
            c["mount"] = p.extra["mount"]
        if p.s["type"] == "faucet":
            c["material_by_feature"] = {"body/arm/head": p.mats[0], "hose": p.mats[-1]}
        tmw = trimesh.Trimesh(p.Vw, p.F, process=False)
        c["mesh"] = dict(vertices=int(len(p.Vw)), faces=int(len(p.F)), volume_mm3=round(float(tmw.volume), 4),
                         watertight=bool(tmw.is_watertight))
        comps.append(c)
    layers = OrderedDict()
    for p in parts:
        layers.setdefault(p.layer, dict(fabrication=not p.ref, components=[]))["components"].append(p.id)
    assembly = OrderedDict(
        schema="keane-kitchens/assembly/v1", generator="tools/build_cad.py", source_spec=os.path.relpath(spec_path, root),
        units="mm", inch_to_mm=IN,
        world_frame=dict(up="+Z", handedness="right", origin="inside corner at floor, at backsplash tile faces",
                         W1="sink wall: tile face y=0, room y>0; run L along +x, fronts +y",
                         W2="right wall: tile face x=0, room x>0; run R along +y, fronts +x"),
        local_frame=dict(convention="right-handed, Y-up, OBJ/glTF style: +X viewer's right facing the front, +Y up, "
                                    "+Z out of the front; component files are in local mm with the pivot at the origin",
                         run_L="X=-x_world, Y=+z_world, Z=+y_world", run_R="X=+y_world, Y=+z_world, Z=+x_world",
                         run_none="X=+x_world, Y=+z_world, Z=-y_world",
                         transform="world_mm = transform_local_to_world (4x4 row-major) @ [local_mm, 1]",
                         pivots=dict(cabinet_slab_trim_backsplash_sink_reference="rear-bottom-left corner; geometry in "
                                     "X[0,w] Y[0,h] Z[0,d]",
                                     door="(hinge edge, bottom edge, FRONT face); left-hinged X[0,w], right-hinged "
                                          "X[-w,0]; Z[-t,0]; swing about local +Y, -110 deg left / +110 deg right "
                                          "(limited by door_swing.max_open_deg)",
                                     drawer_front="centre of back face; X[-w/2,w/2] Y[-h/2,h/2] Z[0,t]; slide local +Z",
                                     hardware_handle="centre of mounting footprint on the parent's actual front surface",
                                     faucet="base centre at counter surface")),
        materials=OrderedDict((k, dict(baseColor=v["baseColor"], metallic=v["metallic"], roughness=v["roughness"],
                                       photo_albedo=v.get("photo_albedo"), note=v.get("note", "")))
                              for k, v in materials.items()),
        material_file="components/materials.mtl", layers=layers,
        glb_note="artifacts/export/kitchen_world.glb: root node 'kitchen' maps world Z-up mm -> glTF Y-up m; "
                 "part nodes hold local meshes, nested by parent, so door/drawer nodes animate about their origin.",
        components=comps)
    json.dump(assembly, open(os.path.join(cad, "assembly.json"), "w"), indent=1)
    timings["assembly_s"] = time.time() - t0

    # ---- 8.5 world-space layered exports ----------------------------------------------------------
    t0 = time.time()
    open(os.path.join(exp, "kitchen_world.mtl"), "w").write(mtl)
    L = ["# Keane Kitchens world model - units mm, Z-up world frame; one o/g per part named layer/id",
         "mtllib kitchen_world.mtl"]
    base = [0, 0, 0]
    layer_mesh = OrderedDict((k, ([], [])) for k in LAYER_STL)
    ply_v, ply_f, ply_c = [], [], []
    nply = 0
    layer_rank = {k: i for i, k in enumerate(LAYER_STL)}
    for p in sorted(parts, key=lambda q: (layer_rank.get(q.layer, 99), parts.index(q))):   # layers contiguous
        vnw = p.vn @ p.R.T
        L += obj_block(f"{p.layer}/{p.id}", p.Vw, p.uvd["uv"], vnw, p.F, p.uvd["vt_idx"], p.vn_idx, p.fmat, p.mats,
                       tuple(base))
        base = [base[0] + len(p.Vw), base[1] + len(p.uvd["uv"]), base[2] + len(vnw)]
        vs, fs = layer_mesh.setdefault(p.layer, ([], []))
        off = sum(len(v) for v in vs)
        vs.append(p.Vw)
        fs.append(p.F + off)
        for mi, mat in enumerate(p.mats):          # PLY: split per material so colours don't bleed
            Fm = p.F[p.fmat == mi]
            if not len(Fm):
                continue
            used, inv = np.unique(Fm, return_inverse=True)
            ply_v.append(p.Vw[used])
            ply_f.append(inv.reshape(-1, 3) + nply)
            ply_c.append(np.tile(np.r_[np.round(np.asarray(materials[mat]["baseColor"]) * 255), 255], (len(used), 1)))
            nply += len(used)
    open(os.path.join(exp, "kitchen_world.obj"), "w").write("\n".join(L) + "\n")
    for layer, (vs, fs) in layer_mesh.items():
        if vs:
            write_stl(os.path.join(exp, LAYER_STL.get(layer, layer + ".stl")), np.vstack(vs), np.vstack(fs))
    ply = trimesh.Trimesh(np.vstack(ply_v), np.vstack(ply_f), vertex_colors=np.vstack(ply_c).astype(np.uint8),
                          process=False)
    open(os.path.join(exp, "kitchen_world.ply"), "wb").write(ply.export(file_type="ply", encoding="binary"))
    write_glb(os.path.join(exp, "kitchen_world.glb"), parts, materials)
    timings["world_exports_s"] = time.time() - t0

    # ---- 8.6 validation (on the files actually written) -------------------------------------------
    t0 = time.time()
    asm = json.load(open(os.path.join(cad, "assembly.json")))
    per_part, mans = [], {}
    rt_obj = rt_stl = 0.0
    uv_tot = dict(parts=0, overlap_px_total=0, parts_with_overlap=[], oob_vertices=0)
    for p, c in zip(parts, asm["components"]):
        Vl, Fl = read_obj_positions(os.path.join(cad, c["files"]["obj"]))
        T = np.array(c["transform_local_to_world"])
        Vw = Vl @ T[:3, :3].T + T[:3, 3]
        rt_obj = max(rt_obj, float(np.abs(Vw - p.Vw).max()))
        st = trimesh.load(os.path.join(cad, c["files"]["stl"]))      # merges vertices on load
        stw = st.vertices @ T[:3, :3].T + T[:3, 3]
        rt_stl = max(rt_stl, float(cKDTree(p.Vw).query(stw)[0].max()))
        tm = trimesh.Trimesh(Vw, Fl, process=False)
        tri = Vw[Fl]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        dup = len(Fl) - len(np.unique(np.sort(Fl, axis=1), axis=0))
        m = m_from_arrays(Vw, Fl)
        mans[p.id] = m
        ov, cov, oob = raster_uv_overlap(p.uvd["uv"], p.uvd["vt_idx"])
        planar = bool(np.all(np.abs(np.abs(tm.face_normals).max(1) - 1) < 1e-9))
        uv_tot["parts"] += 1
        uv_tot["overlap_px_total"] += ov
        uv_tot["oob_vertices"] += oob
        if ov:
            uv_tot["parts_with_overlap"].append(dict(id=p.id, overlap_px=ov, axis_aligned_planar=planar))
        bodies = int(connected_components(coo_matrix(
            (np.ones(len(tm.face_adjacency)), (tm.face_adjacency[:, 0], tm.face_adjacency[:, 1])),
            shape=(len(Fl), len(Fl))), directed=False)[0])
        per_part.append(OrderedDict(
            id=p.id, watertight=bool(tm.is_watertight), winding_consistent=bool(tm.is_winding_consistent),
            volume_mm3=round(float(tm.volume), 4), volume_positive=bool(tm.volume > 0),
            euler_number=int(tm.euler_number), shells=bodies,
            genus=int(round((2 * bodies - tm.euler_number) / 2)),
            degenerate_faces=int((area < 1e-6).sum()), min_face_area_mm2=float(area.min()),
            duplicate_faces=int(dup), manifold3d_status=str(m.status()).replace("Error.", ""),
            manifold3d_tri_count_matches=bool(m.num_tri() == len(Fl)),
            stl_watertight_after_reload=bool(st.is_watertight),
            uv=dict(charts=p.uvd["n_charts"], overlap_px=ov, covered_px=cov, out_of_bounds_vt=oob,
                    raster=f"{UV_RES}x{UV_RES}", axis_aligned_planar=planar)))
    # pairwise interference (AABBs within 0.02 in)
    ids = [p.id for p in parts]
    bb = np.array([[mans[i].bounding_box()[:3], mans[i].bounding_box()[3:]] for i in ids])
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            gap = np.maximum(bb[i, 0] - bb[j, 1], bb[j, 0] - bb[i, 1])
            if np.all(gap <= PAIR_AABB_TOL_MM):
                vol = float((mans[ids[i]] ^ mans[ids[j]]).volume())
                pairs.append(dict(a=ids[i], b=ids[j], aabb_gap_mm=round(float(max(gap.max(), 0.0)), 4),
                                  intersection_mm3=round(vol, 6), flagged=bool(vol > INTERSECT_FLAG_MM3)))
    # clearances between front-plane parts (doors, drawer fronts, fillers)
    fronts = [p for p in parts if p.cat in ("door", "drawer_front")]
    fplane = fronts + [p for p in parts if p.id.startswith("trim_filler")]
    clear = []
    for i in range(len(fplane)):
        for j in range(i + 1, len(fplane)):
            a, b = fplane[i], fplane[j]
            ia, ib = ids.index(a.id), ids.index(b.id)
            gap = np.maximum(bb[ia, 0] - bb[ib, 1], bb[ib, 0] - bb[ia, 1])
            g = float(np.linalg.norm(np.maximum(gap, 0)))
            if g < FRONT_GAP_SEARCH_MM:
                mg = float(mans[a.id].min_gap(mans[b.id], 2 * FRONT_GAP_SEARCH_MM))
                kind = "front-front" if b.cat in ("door", "drawer_front") else "front-filler"
                if a.id.startswith("trim") and b.id.startswith("trim"):
                    kind = "filler-filler"
                n_ax = int((gap > 1e-6).sum())       # 1 = edge-adjacent reveal, 2+ = diagonal / inside corner
                clear.append(dict(a=a.id, b=b.id, kind=kind, adjacency="edge" if n_ax <= 1 else "diagonal",
                                  aabb_gap_mm=round(g, 4), min_gap_mm=round(mg, 4),
                                  is_reveal_3_175=bool(abs(mg - 3.175) < 0.01)))
    ff = [c for c in clear if c["kind"] == "front-front"]
    ffe = [c for c in ff if c["adjacency"] == "edge"]
    # coincident vertices across different fronts
    fv = np.vstack([p.Vw for p in fronts])
    owner = np.concatenate([[k] * len(p.Vw) for k, p in enumerate(fronts)])
    cpairs = cKDTree(fv).query_pairs(1e-3, output_type="ndarray")
    coincident = int((owner[cpairs[:, 0]] != owner[cpairs[:, 1]]).sum()) if len(cpairs) else 0
    # union of all fabrication parts.  The exact union is a valid manifold3d solid, but where parts
    # touch only along a line (or manifold3d's symbolic perturbation separates coincident faces) it
    # keeps topologically distinct vertices at the same position, which an STL reader merges into
    # non-manifold edges.  The deliverable STL therefore uses a weld: every part offset outward by
    # UNION_WELD_MM (scale about its AABB centre) so touching faces overlap -> watertight on reload.
    def union_stats(man, path):
        Vu, Fu, _ = m_to_arrays(man)
        write_stl(path, Vu, Fu)
        st = trimesh.load(path)
        e = np.sort(st.edges, axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        bad = ue[cnt != 2]
        return dict(file=os.path.relpath(path, root), status=str(man.status()).replace("Error.", ""),
                    volume_mm3=round(float(man.volume()), 3), genus=int(man.genus()),
                    components=len(man.decompose()), triangles=int(man.num_tri()),
                    stl_reload_watertight=bool(st.is_watertight), stl_reload_nonmanifold_edges=int(len(bad)),
                    nonmanifold_edges_in=[r6(np.round(st.vertices[b] / IN, 4)) for b in bad])

    def weld(m, dlt):
        b = np.array(m.bounding_box()).reshape(2, 3)
        c = b.mean(0)
        return m.translate(tuple(-c)).scale(tuple(1 + 2 * dlt / (b[1] - b[0]))).translate(tuple(c))
    fab = [mans[p.id] for p in parts if not p.ref]
    exact = union_stats(Manifold.batch_boolean(fab, OpType.Add), os.path.join(exp, "kitchen_union_exact.stl"))
    # simplify(1 um) removes near-coincident vertices that float32 STL rounding would otherwise merge
    welded = union_stats(Manifold.batch_boolean([weld(m, UNION_WELD_MM) for m in fab], OpType.Add)
                         .simplify(FAUCET_SIMPLIFY_MM), os.path.join(exp, "kitchen_union_watertight.stl"))
    union_info = dict(sum_part_volumes_mm3=round(float(sum(m.volume() for m in fab)), 3),
                      watertight_weld_mm=UNION_WELD_MM, welded=welded, exact=exact,
                      weld_volume_delta_pct=round(100 * (welded["volume_mm3"] / exact["volume_mm3"] - 1), 5))
    # GLB round trip (trimesh loader, glTF Y-up metres -> world Z-up mm)
    glb_err = None
    try:
        sc = trimesh.load(os.path.join(exp, "kitchen_world.glb"), force="scene")
        g = sc.to_geometry() if hasattr(sc, "to_geometry") else sc.dump(concatenate=True)
        G = np.asarray(g.vertices) * 1000.0
        Gw = np.column_stack([G[:, 0], -G[:, 2], G[:, 1]])
        allw = np.vstack([p.Vw for p in parts])
        glb_err = float(max(cKDTree(allw).query(Gw)[0].max(), cKDTree(Gw).query(allw)[0].max()))
    except Exception as e:                      # noqa: BLE001
        glb_err = f"GLB reload failed: {e}"
    timings["validation_s"] = time.time() - t0

    # ---- 8.7 STEP (optional) ----------------------------------------------------------------------
    t0 = time.time()
    step = dict(status="skipped (--no-step)")
    if not args.no_step:
        try:
            vols, skipped, reimport = build_step(parts, materials, os.path.join(exp, "kitchen.step"), finfo)
            dv = {k: abs(v - mans[k].volume()) / mans[k].volume() for k, v in vols.items()}
            step = dict(status="ok", file="artifacts/export/kitchen.step", solids=len(vols), skipped=skipped,
                        reimport=reimport, brep_total_volume_mm3=round(sum(vols.values()), 1),
                        max_rel_volume_diff_vs_mesh_box_parts=float(max(v for k, v in dv.items()
                                                                        if not k.startswith("fixture_faucet"))),
                        faucet_rel_volume_diff_vs_mesh=(round(dv[[k for k in dv if k.startswith("fixture_faucet")][0]], 5)
                                                        if any(k.startswith("fixture_faucet") for k in dv) else None))
        except ImportError as e:
            step = dict(status=f"not available: {e}")
        except Exception as e:                  # noqa: BLE001
            step = dict(status=f"failed: {type(e).__name__}: {e}")
    timings["step_s"] = time.time() - t0

    # ---- 8.8 previews -------------------------------------------------------------------------------
    t0 = time.time()
    previews = []
    if not args.no_preview:
        preview_iso(parts, materials, os.path.join(exp, "preview_iso.png"))
        previews.append("preview_iso.png")
        try:
            preview_camera(parts, materials, os.path.join(root, "artifacts", "sfm", "cameras.json"),
                           os.path.join(root, "images", "02_corner_wide.jpg"), os.path.join(exp, "preview_cam02.png"))
            previews.append("preview_cam02.png")
        except Exception as e:                  # noqa: BLE001
            notes.append(f"camera-02 preview skipped: {e}")
    timings["preview_s"] = time.time() - t0

    # ---- 8.9 report -------------------------------------------------------------------------------------
    fails = []
    for r in per_part:
        bad = [k for k in ("watertight", "winding_consistent", "volume_positive", "stl_watertight_after_reload",
                           "manifold3d_tri_count_matches") if not r[k]]
        if r["degenerate_faces"] or r["duplicate_faces"] or r["manifold3d_status"] != "NoError":
            bad.append("degenerate/duplicate/manifold3d")
        if bad:
            fails.append(f"{r['id']}: {bad}")
    flagged = [x for x in pairs if x["flagged"]]
    nonstd = [c for c in ffe if not c["is_reveal_3_175"]]
    diag = [(c["a"], c["b"], c["min_gap_mm"]) for c in ff if c["adjacency"] == "diagonal"]
    summary = OrderedDict(
        parts=len(parts), fabrication_parts=sum(not p.ref for p in parts), reference_parts=sum(p.ref for p in parts),
        part_failures=fails,
        all_watertight=all(r["watertight"] for r in per_part),
        all_winding_consistent=all(r["winding_consistent"] for r in per_part),
        all_volume_positive=all(r["volume_positive"] for r in per_part),
        all_manifold3d_ok=all(r["manifold3d_status"] == "NoError" for r in per_part),
        degenerate_faces_total=sum(r["degenerate_faces"] for r in per_part),
        duplicate_faces_total=sum(r["duplicate_faces"] for r in per_part),
        pairs_checked=len(pairs), pairs_flagged_gt_1mm3=len(flagged),
        max_intersection_mm3=max((x["intersection_mm3"] for x in pairs), default=0.0),
        front_front_edge_adjacent_pairs=len(ffe),
        front_front_edge_min_gap_mm=min((c["min_gap_mm"] for c in ffe), default=None),
        front_front_edge_max_gap_mm=max((c["min_gap_mm"] for c in ffe), default=None),
        front_front_edge_non_3_175=[(c["a"], c["b"], c["min_gap_mm"]) for c in nonstd],
        front_front_diagonal=diag,
        front_filler_gaps_mm=sorted({c["min_gap_mm"] for c in clear if c["kind"] == "front-filler"}),
        coincident_vertex_pairs_across_fronts=coincident,
        uv_overlap_px_total=uv_tot["overlap_px_total"], uv_parts_with_overlap=uv_tot["parts_with_overlap"],
        uv_out_of_bounds_vt=uv_tot["oob_vertices"],
        assembly_roundtrip_max_err_mm=dict(obj=rt_obj, stl=rt_stl, glb=glb_err),
        obb_vs_local_aabb_max_dev_mm=obb_dev, union=union_info, step=step,
        door_max_open_deg=OrderedDict((k, v["max_open_deg"]) for k, v in swing.items()),
        doors_limited_below_open_angle=OrderedDict((k, dict(max_open_deg=v["max_open_deg"],
                                                            first_collision_parts=v["first_collision"]["parts"]))
                                                   for k, v in swing.items() if v["first_collision"]),
        spec_notes=notes, timings_s={k: round(v, 2) for k, v in timings.items()},
        total_runtime_s=round(time.time() - t_start, 2))
    report = OrderedDict(schema="keane-kitchens/validation/v1", units="mm", summary=summary, parts=per_part,
                         pairwise_interference=dict(criterion=f"AABBs within {PAIR_AABB_TOL_MM:.3f} mm; flag > "
                                                    f"{INTERSECT_FLAG_MM3} mm3 (manifold3d intersection)", pairs=pairs),
                         front_clearances=dict(expected_reveal_mm=0.125 * IN, pairs=clear),
                         door_swing_sweep=dict(note=f"door + its pulls rotated about the front-face hinge line in "
                                               f"{SWING_STEP_DEG} deg steps (cap {DOOR_OPEN_DEG:g}); max_open_deg = "
                                               f"largest angle with no intersection > {INTERSECT_FLAG_MM3} mm3 vs any "
                                               f"other fabrication part", doors=swing),
                         uv=uv_tot, previews=previews)
    json.dump(report, open(os.path.join(exp, "validation_report.json"), "w"), indent=1)
    write_readme(cad, summary)

    # compact console summary
    print("=" * 78)
    print(f"build_cad: {len(parts)} parts ({summary['fabrication_parts']} fabrication + {summary['reference_parts']} "
          f"reference) in {summary['total_runtime_s']} s")
    print(f"  per-part: watertight {summary['all_watertight']}, winding {summary['all_winding_consistent']}, "
          f"vol>0 {summary['all_volume_positive']}, manifold3d {summary['all_manifold3d_ok']}, "
          f"degenerate {summary['degenerate_faces_total']}, duplicate {summary['duplicate_faces_total']}")
    print(f"  failures: {fails or 'none'}")
    print(f"  interference: {len(pairs)} pairs, flagged {len(flagged)}, max {summary['max_intersection_mm3']:.6f} mm3")
    print(f"  front reveals (edge-adjacent): {len(ffe)} pairs, min {summary['front_front_edge_min_gap_mm']} / max "
          f"{summary['front_front_edge_max_gap_mm']} mm; non-3.175: {summary['front_front_edge_non_3_175']}")
    print(f"  front gaps (diagonal / inside corner): {diag}")
    print(f"  front-filler gaps: {summary['front_filler_gaps_mm']} mm; coincident front vertices: {coincident}")
    print(f"  UV overlap px: {uv_tot['overlap_px_total']} ({len(uv_tot['parts_with_overlap'])} parts); OOB vt: "
          f"{uv_tot['oob_vertices']}")
    print(f"  round trip (mm): obj {rt_obj:.2e}, stl {rt_stl:.2e}, glb {glb_err}")
    for k in ("welded", "exact"):
        u = union_info[k]
        print(f"  union[{k}]: {u['status']}, vol {u['volume_mm3']:.0f} mm3, genus {u['genus']}, components "
              f"{u['components']}, STL reload watertight {u['stl_reload_watertight']} "
              f"(non-manifold edges {u['stl_reload_nonmanifold_edges']})")
    print(f"  union weld {UNION_WELD_MM} mm volume delta {union_info['weld_volume_delta_pct']}%; "
          f"sum of parts {union_info['sum_part_volumes_mm3']:.0f} mm3")
    print(f"  STEP: {step}")
    print("  door max_open_deg: " + ", ".join(f"{k}={v['max_open_deg']}" +
                                             (f" ({','.join(v['first_collision']['parts'])})" if v["first_collision"] else "")
                                             for k, v in swing.items()))
    for n in notes:
        print(f"  note: {n}")
    print(f"  timings: {summary['timings_s']}")


def write_readme(cad, summary):
    txt = f"""Keane Kitchens - CAD components (generated by tools/build_cad.py; do not hand-edit)
==================================================================================
Regenerate:  python3 tools/build_cad.py --spec artifacts/geometry/layout_spec.json

components/<id>.obj|.stl   one closed manifold per part, LOCAL frame, millimetres, pivot at origin
components/reference/      walls/floor (layer "reference", NOT for fabrication)
components/materials.mtl   swappable material slots (Kd/Ks/Ns + PBR keys Pr=roughness, Pm=metallic)
assembly.json              manifest: per-part transform_local_to_world (4x4 row-major, mm), pivot, OBB,
                           parent/children, material slots, door swing / drawer slide, mesh stats
../artifacts/export/       world-space exports (OBJ/MTL, per-layer STL, PLY, GLB, union STL, STEP, PNG,
                           validation_report.json)

Local frame: right-handed, Y-up. +X = viewer's right facing the part's front, +Y = up, +Z = out of front.
  run L (sink wall, fronts +y):  X=-x  Y=+z  Z=+y       run R (right wall, fronts +x):  X=+y  Y=+z  Z=+x
  run None (floor):              X=+x  Y=+z  Z=-y
Pivots: boxes/slabs/trim/backsplash/sink = rear-bottom-left corner (X[0,w] Y[0,h] Z[0,d]);
  doors = hinge edge/bottom/FRONT face (left-hinged X[0,w], right-hinged X[-w,0], Z[-t,0]; open -110 / +110 deg
  about local +Y, limited to door_swing.max_open_deg); drawer fronts = centre of back face (slide +Z,
  travel = carcass depth - 3in);
  pulls = centre of mounting footprint on the parent's actual surface; faucet = base centre on counter.
OBJ uses v/vt/vn indexing: UV seams do not split positions, so meshes stay watertight in position space.
UVs: one non-overlapping atlas per part in [0,1]^2, uniform scale; 1 UV unit = uv_mm_per_unit mm
  (see assembly.json) so real-world-scale textures can be applied.
GLB: root node 'kitchen' converts world Z-up mm -> glTF Y-up metres; part nodes (name = id) carry local
  meshes nested by parent, so rotating a door node about its local +Y swings the door with its pull.

Union: kitchen_union_watertight.stl = union of all fabrication parts with a {UNION_WELD_MM} mm per-part weld
  (watertight after an STL reader merges vertices); kitchen_union_exact.stl = exact manifold3d union (valid
  manifold topology, but position-coincident edges where parts touch only along a line).

Last build: {summary['parts']} parts, all watertight={summary['all_watertight']}, interference pairs flagged=
{summary['pairs_flagged_gt_1mm3']}, UV overlap px={summary['uv_overlap_px_total']}, STEP={summary['step'].get('status')}.
"""
    open(os.path.join(cad, "README.txt"), "w").write(txt)


if __name__ == "__main__":
    main()
