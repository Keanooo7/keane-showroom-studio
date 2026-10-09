#!/usr/bin/env python3
"""cad_clean.py - turn a raw SuGaR mesh into clean, metric, segmented CAD geometry.

Usage:
  python postprocess/cad_clean.py in.ply out_dir --units mm \
      [--scale-ref counter_height=914.4 | points=x1,y1,z1;x2,y2,z2;D | factor=s] \
      [--up 0,-1,0 | --colmap-images path/to/sparse/0/images.bin] \
      [--plane-tol 8] [--snap-deg 5] [--min-plane-area 5000]

Steps (see module docstrings for details):
  a floaters.py  prune floaters / tiny islands (optional confidence filter)
  e scale.py     metric scale (known distance, countertop height, or factor)
  b manhattan.py Manhattan frame -> Z-up, axis-aligned, right-handed
  c planes.py    iterative RANSAC, Kabsch frame refinement, 5 deg axis snapping,
                 crease-strip growth, vertex projection onto planes/edges/corners
  d decimate.py  conforming re-triangulation of planar patches
  f labels.py    floor / walls / countertop / fronts / carcass / hardware layers
  g topology.py  watertight / manifold / winding / degenerate checks + export

Outputs in out_dir:
  <stem>_clean.ply / .stl        full cleaned mesh (output units, Z-up, floor z=0)
  <stem>_layers.obj              one object per layer (import into CAD/Blender)
  <stem>_<layer>.stl             per-layer meshes
  <stem>_face_plane_ids.npy      plane id per output face (-1 = non-planar)
  <stem>_face_layers.npy         layer index per output face
  <stem>_report.json             transform, planes, crease edges, patch loops, topology
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocess.common import UNIT_PER_MM, face_adjacency, face_normals_areas, log, setup_logging  # noqa: E402
from postprocess import floaters, manhattan, planes as pl, decimate, scale as sc, labels, topology  # noqa: E402


class Frame:
    """Tracks V and the cumulative 4x4 transform raw -> output."""

    def __init__(self, V):
        self.V = V.astype(np.float64)
        self.M = np.eye(4)

    def apply(self, M):
        self.V = self.V @ M[:3, :3].T + M[:3, 3]
        self.M = M @ self.M

    def scale(self, s):
        self.apply(np.diag([s, s, s, 1.0]))

    def rotate(self, R):
        M = np.eye(4); M[:3, :3] = R; self.apply(M)

    def translate(self, t):
        M = np.eye(4); M[:3, 3] = t; self.apply(M)


def load_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    attrs = {}
    raw = m.metadata.get("_ply_raw", {}) if hasattr(m, "metadata") else {}
    try:
        data = raw["vertex"]["data"]
        for k in data.dtype.names:
            attrs[k] = np.asarray(data[k])
    except Exception:
        pass
    V0 = len(m.vertices)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if len(m.vertices) != V0:
        attrs = {}   # attributes no longer aligned after welding UV seams
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), attrs


def crease_edges(V, F, patch, patch_plane, planes, upm):
    he = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    hf = np.repeat(np.arange(len(F)), 3)
    key = np.sort(he, 1)
    k1 = key[:, 0] * len(V) + key[:, 1]
    order = np.argsort(k1)
    k1s = k1[order]
    same = np.flatnonzero(k1s[1:] == k1s[:-1])
    fa, fb = hf[order[same]], hf[order[same + 1]]
    pa, pb = patch[fa], patch[fb]
    m = (pa >= 0) & (pb >= 0) & (pa != pb)
    pa, pb, ek = np.minimum(pa[m], pb[m]), np.maximum(pa[m], pb[m]), key[order[same]][m]
    out = []
    pairs = np.stack([pa, pb], 1)
    if len(pairs) == 0:
        return out
    up, inv = np.unique(pairs, axis=0, return_inverse=True)
    for i, (a, b) in enumerate(up):
        na, nb = planes[patch_plane[a]].n, planes[patch_plane[b]].n
        dirv = np.cross(na, nb)
        if np.linalg.norm(dirv) < 1e-6:
            continue
        dirv /= np.linalg.norm(dirv)
        vid = np.unique(ek[inv.ravel() == i])
        t = V[vid] @ dirv
        p0, p1 = V[vid[np.argmin(t)]], V[vid[np.argmax(t)]]
        out.append({"patches": [int(a), int(b)],
                    "planes": [int(patch_plane[a]), int(patch_plane[b])],
                    "dihedral_deg": round(float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1)))), 4),
                    "p0": np.round(p0, 3).tolist(), "p1": np.round(p1, 3).tolist(),
                    "length": round(float(np.linalg.norm(p1 - p0)), 3)})
    return out


def patch_loops(V, F, patch, nP):
    he = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    hf = np.repeat(np.arange(len(F)), 3)
    twin = decimate._twin_info(he, len(V))
    g = patch[hf]
    tg = np.where(twin >= 0, g[np.maximum(twin, 0)], -9)
    bnd = (g >= 0) & (tg != g)
    loops = {}
    for p in range(nP):
        sel = np.flatnonzero(bnd & (g == p))
        L = decimate._loops_for_patch(sel.tolist(), he, twin, bnd)
        if L and sum(len(x) for x in L) <= 400:
            loops[int(p)] = [np.round(V[x], 3).tolist() for x in L]
    return loops


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--units", default="mm", choices=list(UNIT_PER_MM))
    ap.add_argument("--scale-ref", default=None,
                    help="counter_height=H | points=x1,y1,z1;x2,y2,z2;D | factor=s "
                         "(distances in --units; default counter_height=36in)")
    ap.add_argument("--up", default=None, help="rough up vector in RAW mesh coords, e.g. 0,-1,0")
    ap.add_argument("--colmap-images", default=None, help="COLMAP images.txt/.bin to derive up")
    ap.add_argument("--plane-tol", type=float, default=8.0, help="plane inlier distance, mm")
    ap.add_argument("--snap-deg", type=float, default=5.0, help="snap normals to axes within this angle")
    ap.add_argument("--min-plane-area", type=float, default=5000.0, help="mm^2 (default 50 cm^2)")
    ap.add_argument("--oblique-min-width", type=float, default=60.0,
                    help="non-axis planes narrower than this (mm) are treated as rounded creases")
    ap.add_argument("--max-island-area", type=float, default=20000.0,
                    help="mm^2; flat residual islands smaller than this inside one plane are absorbed")
    ap.add_argument("--conf-thresh", type=float, default=None, help="drop faces with vertex opacity/confidence below")
    ap.add_argument("--keep-frac", type=float, default=0.02, help="components >= frac of largest always kept")
    ap.add_argument("--no-decimate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    setup_logging(not a.quiet)
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.input))[0]
    upm = UNIT_PER_MM[a.units]
    report = {"input": os.path.abspath(a.input), "units": a.units, "warnings": []}

    # ---- load + (a) floaters --------------------------------------------
    V, F, attrs = load_mesh(a.input)
    log.info("loaded %s: %d vertices, %d faces", a.input, len(V), len(F))
    report["faces_input"] = int(len(F))
    F, conf_key = floaters.confidence_filter(V, F, attrs, a.conf_thresh)
    F, _small, finfo = floaters.prune_floaters(V, F, keep_frac=a.keep_frac)
    V, F, _ = topology.compact(V, F)
    report["floaters"] = finfo
    report["faces_after_floaters"] = int(len(F))
    tm = trimesh.Trimesh(V, F, process=False)
    trimesh.repair.fix_winding(tm)
    F = np.asarray(tm.faces, np.int64)
    report["topology_before"] = topology.topology_report(V, F, deg_area=0.0)
    fr = Frame(V)

    # ---- (e) scale part 1 ------------------------------------------------
    kind, val, val_units, is_default = sc.parse_scale_ref(a.scale_ref)
    if is_default:
        report["warnings"].append("no --scale-ref given: assumed standard 36 in countertop height")
    if kind == "points":
        s = sc.scale_from_points(*val)
        fr.scale(s)
        log.info("scale from points: x%.6f", s)
    elif kind == "factor":
        fr.scale(val)

    # ---- (b) Manhattan frame --------------------------------------------
    Nf, A, C = face_normals_areas(fr.V, F)
    R_m, explained = manhattan.estimate_manhattan(Nf, A, seed=a.seed)
    up_hint = None
    if a.up:
        up_hint = fr.M[:3, :3] @ np.array(list(map(float, a.up.split(","))))
    elif a.colmap_images:
        up_hint = fr.M[:3, :3] @ manhattan.colmap_up(a.colmap_images)
    R, up_how = manhattan.choose_axes(R_m, fr.V, Nf, A, C, up_hint)
    fr.rotate(R)
    report["manhattan"] = {"area_explained_by_3_axes": round(explained, 4), "up_from": up_how}
    log.info("Manhattan frame explains %.1f%% of surface area", 100 * explained)

    # ---- (e) scale part 2: countertop height ----------------------------
    H = None
    if kind == "counter_height":
        H = val * (25.4 * upm if val_units == "in" else 1.0)
        Nf, A, C = face_normals_areas(fr.V, F)
        fz, cz = sc.measure_counter_height(fr.V, Nf, A, C)
        fr.scale(H / (cz - fz))
        fz, cz = fz * H / (cz - fz), cz * H / (cz - fz)

    # ---- (c) planes -----------------------------------------------------
    tol = a.plane_tol * upm
    min_area = a.min_plane_area * upm ** 2
    adj = face_adjacency(F)
    planes = pl.ransac_planes(fr.V, F, adj, tol, min_area, seed=a.seed,
                              oblique_min_width=a.oblique_min_width * upm, snap_deg=a.snap_deg)
    Vr, R2 = pl.refine_frame_and_snap(fr.V, F, planes, a.snap_deg)
    fr.rotate(R2)
    planes = pl.drop_bridging_planes(planes, tol)
    planes = pl.merge_planes(fr.V, F, planes, tol)

    # exact scale + origin from the fitted planes
    zp = [p for p in planes if p.snapped_axis == "+Z"]
    Nf, A, C = face_normals_areas(fr.V, F)
    big = lambda p: A[p.faces].sum()
    if H is not None and zp:
        pf = min(zp, key=lambda p: abs(p.d - fz))
        pc = min(zp, key=lambda p: abs(p.d - cz))
        corr = H / (pc.d - pf.d)
        fr.scale(corr)
        for p in planes:
            p.d *= corr
        report["scale_correction_from_planes"] = corr
    floor_planes = [p for p in planes if p.snapped_axis == "+Z" and big(p) > 0.5e6 * upm ** 2]
    t = fr.V.min(0).copy()
    if floor_planes:
        t[2] = min(floor_planes, key=lambda p: p.d).d
    for ax, name in ((1, "+Y"), (0, "+X")):
        cand = [p for p in planes if p.snapped_axis == name and big(p) > 0.5e6 * upm ** 2
                and p.d < fr.V[:, ax].min() + 0.1 * np.ptp(fr.V[:, ax])]
        if cand:
            t[ax] = max(cand, key=big).d
    fr.translate(-t)
    for p in planes:
        p.d -= float(p.n @ t)
    V = fr.V

    lab = pl.face_labels(len(F), planes)
    lab = pl.grow_regions(V, F, adj, lab, planes, tol)
    lab = pl.smooth_labels(V, F, adj, lab, planes, tol)
    lab = pl.grow_regions(V, F, adj, lab, planes, tol)
    lab = pl.absorb_islands(V, F, adj, lab, planes, tol, max_area=a.max_island_area * upm ** 2)
    lab = pl.absorb_on_plane_faces(F, lab, len(planes))
    for i, p in enumerate(planes):
        p.faces = np.flatnonzero(lab == i)
    patch, patch_plane = pl.split_patches(F, adj, lab, planes, A, min_area)
    V, conflict = pl.project_vertices(V, F, patch, patch_plane, planes, max_move=3 * tol)
    # faces touching unresolved (thin step/slot) vertices, or not lying on their
    # plane after projection, become residual geometry so every patch is exactly flat
    nrm_ = np.array([p.n for p in planes]); off_ = np.array([p.d for p in planes])
    lp = np.where(patch >= 0, patch_plane[np.maximum(patch, 0)], 0)
    offp = np.abs(np.einsum("fkj,fj->fk", V[F], nrm_[lp]) - off_[lp][:, None]).max(1) > 1e-6 * max(1.0, upm)
    demote = (patch >= 0) & (conflict[F].any(1) | offp)
    patch[demote] = -1
    log.info("demoted %d faces with unresolved/off-plane vertices to residual", demote.sum())
    F2, ok, nmerged = topology.merge_close_vertices(V, F, 1e-2 * upm)
    F, patch = F2[ok], patch[ok]
    log.info("collapsed %d zero-length edges, dropped %d collapsed faces", nmerged, (~ok).sum())

    # ---- (d) decimation -------------------------------------------------
    if a.no_decimate:
        dstats = {"skipped": True}
    else:
        F, patch, dstats = decimate.decimate_planar(V, F, patch, patch_plane, planes,
                                                    max_fold_area=(0.5 * upm) ** 2)
    report["decimation"] = dstats
    V, F, _ = topology.compact(V, F)

    # ---- (f) layers -----------------------------------------------------
    layer, pinfo, ctop = labels.label_layers(V, F, patch, patch_plane, planes, upm, tol=tol)
    report["countertop_height"] = ctop

    # ---- (g) topology + export ------------------------------------------
    deg = 1e-6 * upm ** 2
    report["topology_after"] = topology.topology_report(V, F, deg_area=deg)
    out = trimesh.Trimesh(V, F, process=False)
    out.export(os.path.join(a.out_dir, f"{stem}_clean.ply"))
    out.export(os.path.join(a.out_dir, f"{stem}_clean.stl"))
    report["layers"] = topology.export_layers(V, F, layer, labels.LAYERS, a.out_dir, stem)
    face_plane = np.where(patch >= 0, patch_plane[np.maximum(patch, 0)], -1)
    np.save(os.path.join(a.out_dir, f"{stem}_face_plane_ids.npy"), face_plane)
    np.save(os.path.join(a.out_dir, f"{stem}_face_layers.npy"), layer)

    Nf, A, _ = face_normals_areas(V, F)
    plist = []
    for i, p in enumerate(planes):
        fs = face_plane == i
        if not fs.any():
            continue
        vid = np.unique(F[fs])
        plist.append({
            "id": i, "normal": np.round(p.n, 9).tolist(), "offset": round(p.d, 4),
            "normal_fit_before_snap": np.round(p.n_fit, 6).tolist(),
            "snapped_axis": p.snapped_axis, "snap_angle_deg": round(p.snap_angle_deg, 4),
            "area": round(float(A[fs].sum()), 2),
            "inlier_rms_fit_before_snap": round(p.rms_fit, 4),
            "inlier_rms_after_snap_before_projection": round(p.extra.get("rms_after_snap_before_projection", float("nan")), 4),
            "inlier_rms_final": float(np.sqrt(np.mean((V[vid] @ p.n - p.d) ** 2))),
            "patches": sorted({int(q["patch"]) for q in pinfo if q["plane"] == i}),
            "layers": sorted({q["layer"] for q in pinfo if q["plane"] == i}),
        })
    report["planes"] = plist
    report["patches"] = pinfo
    report["edges"] = crease_edges(V, F, patch, patch_plane, planes, upm)
    report["patch_loops"] = patch_loops(V, F, patch, len(patch_plane))
    report["transform_raw_to_output"] = fr.M.tolist()
    report["scale_raw_to_output"] = float(np.cbrt(np.linalg.det(fr.M[:3, :3])))
    report["faces_output"] = int(len(F))
    report["runtime_s"] = round(time.time() - t0, 2)
    with open(os.path.join(a.out_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    ta = report["topology_after"]
    log.info("done in %.1fs: %d -> %d faces, watertight=%s, winding=%s, non-manifold=%d, degenerate=%d",
             report["runtime_s"], report["faces_after_floaters"], len(F), ta["watertight"],
             ta["winding_consistent"], ta["non_manifold_edges"], ta["degenerate_faces"])
    return report


if __name__ == "__main__":
    main()
