"""(f) Heuristic layer labelling for a Z-up, metric, floor-at-z=0 kitchen mesh.

Layers (per face):
  floor           up-facing horizontal patch at z ~ 0
  countertop      up-facing horizontal patch with top at 34-38.5 in, plus its
                  vertical edge band and overhang underside (within 2.5 in below)
  walls           vertical patch spanning >= 1.5 m of height, or one of the two
                  largest vertical planes touching the scene's outer bounds
  ceiling         down-facing horizontal patch above 2 m with large area
  cabinet_fronts  other vertical patches facing the same way as a wall normal
                  (doors/drawer fronts/face frames)
  cabinet_carcass all remaining planar patches (sides, tops/bottoms of uppers, toe kicks)
  hardware        non-planar residual groups smaller than 30 cm, and small
                  separate components kept by the floater filter
  (non-planar faces that hug a planar patch - rounded creases - inherit that
   patch's layer)
  unclassified    other non-planar residual geometry (sink, faucet, appliances)
"""
from __future__ import annotations

import numpy as np

from .common import components, face_adjacency, face_normals_areas

LAYERS = ["floor", "countertop", "walls", "ceiling", "cabinet_fronts",
          "cabinet_carcass", "hardware", "unclassified"]


def label_layers(V, F, patch, patch_plane, planes, upm, small_frac=0.02, tol=None):
    """upm = output units per millimetre.  Returns (face_layer_idx, patch_info)."""
    IN = 25.4 * upm
    tol = 8.0 * upm if tol is None else tol
    _, A, C = face_normals_areas(V, F)
    nP = len(patch_plane)
    layer = np.full(len(F), LAYERS.index("unclassified"))
    info = []
    zmin_s, zmax_s = V[:, 2].min(), V[:, 2].max()
    xy_lo, xy_hi = V[:, :2].min(0), V[:, :2].max(0)
    # per patch geometry
    for p in range(nP):
        fs = np.flatnonzero(patch == p)
        vz = V[np.unique(F[fs]), 2] if len(fs) else np.zeros(1)
        info.append({"patch": p, "plane": int(patch_plane[p]), "area": float(A[fs].sum()),
                     "zmin": float(vz.min()), "zmax": float(vz.max()), "faces": fs})
    # countertop height
    tops = [q for q in info if planes[q["plane"]].n[2] > 0.99
            and 34 * IN <= planes[q["plane"]].d <= 38.5 * IN and q["area"] > 0.05e6 * upm ** 2]
    ctop = max((planes[q["plane"]].d for q in tops), default=None)
    vert_areas = sorted(((q["area"], q["patch"]) for q in info
                         if abs(planes[q["plane"]].n[2]) < 0.05), reverse=True)
    wall_dirs = []
    names = {}
    for q in info:
        pl = planes[q["plane"]]
        n, d = pl.n, pl.d
        if n[2] > 0.99:
            if abs(d) < 30 * upm:
                nm = "floor"
            elif ctop is not None and abs(d - ctop) < 1 * upm:
                nm = "countertop"
            else:
                nm = "cabinet_carcass"
        elif n[2] < -0.99:
            if ctop is not None and ctop - 2.5 * IN <= -d <= ctop:
                nm = "countertop"
            elif -d > 2000 * upm and q["area"] > 1e6 * upm ** 2:
                nm = "ceiling"
            else:
                nm = "cabinet_carcass"
        elif abs(n[2]) < 0.05:
            height = q["zmax"] - q["zmin"]
            if ctop is not None and q["zmin"] >= ctop - 2.5 * IN and q["zmax"] <= ctop + 1 * upm:
                nm = "countertop"
            elif height >= 1500 * upm:
                nm = "walls"
            else:
                nm = None   # decided after walls are known
        else:
            nm = "cabinet_carcass"
        names[q["patch"]] = nm
        if nm == "walls":
            wall_dirs.append(n)
    for q in info:
        if names[q["patch"]] is None:
            n = planes[q["plane"]].n
            names[q["patch"]] = ("cabinet_fronts" if any(n @ w > 0.99 for w in wall_dirs)
                                 else "cabinet_carcass")
        layer[q["faces"]] = LAYERS.index(names[q["patch"]])
        q["layer"] = names[q["patch"]]
        del q["faces"]
    q_layer = {q["patch"]: q["layer"] for q in info}
    # residual faces hugging a planar patch (rounded creases, peeled rings)
    # inherit that patch's layer: inside the patch bbox (+3 tol), all vertices
    # within tol of its plane and normal within 60 deg
    res = patch < 0
    Nf, _, _ = face_normals_areas(V, F)
    fr = np.flatnonzero(res)
    best = np.full(len(fr), np.inf); best_p = np.full(len(fr), -1)
    for q in info:
        p = q["patch"]
        vv = V[np.unique(F[patch == p])]
        lo, hi = vv.min(0) - 3 * tol, vv.max(0) + 3 * tol
        c = C[fr]
        inside = np.all((c >= lo) & (c <= hi), axis=1)
        if not inside.any():
            continue
        pl_ = planes[q["plane"]]
        idx = np.flatnonzero(inside)
        dist = np.abs(V[F[fr[idx]]] @ pl_.n - pl_.d).max(1)
        ok = (dist < tol) & (Nf[fr[idx]] @ pl_.n > 0.5) & (dist < best[idx])
        best[idx[ok]] = dist[ok]; best_p[idx[ok]] = p
    hug = best_p >= 0
    layer[fr[hug]] = [LAYERS.index(q_layer[pp]) for pp in best_p[hug]] if hug.any() else []
    res[fr[hug]] = False
    # remaining residual groups -> hardware / unclassified
    if res.any():
        lab = components(len(F), face_adjacency(F), res)
        for g in range(lab.max() + 1):
            fs = np.flatnonzero(lab == g)
            vv = V[np.unique(F[fs])]
            diag = np.linalg.norm(vv.max(0) - vv.min(0))
            if diag < 300 * upm:
                layer[fs] = LAYERS.index("hardware")
    # small separate components kept by the floater filter (detached pulls)
    comp = components(len(F), face_adjacency(F))
    ca = np.bincount(comp, weights=A)
    small = ca < small_frac * ca.max()
    layer[small[comp]] = LAYERS.index("hardware")
    return layer, info, ctop
