"""(g) Topology checks, light repair and export helpers."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .common import face_normals_areas, log


def merge_close_vertices(V, F, tol):
    """Collapse mesh edges shorter than tol (e.g. two vertices projected onto the
    same corner point).  Only true edges are collapsed and only when the link
    condition holds, so the mesh stays manifold.  Returns (F, keep_mask, n)."""
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    e = np.unique(e, axis=0)
    short = e[np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1) < tol]
    parent = np.arange(len(V))
    F2 = F.copy()
    n = 0
    if len(short):
        from collections import defaultdict
        nbr = defaultdict(set)
        for a, b in e:
            nbr[a].add(b); nbr[b].add(a)
        vf = defaultdict(list)
        for fi, f in enumerate(F):
            for v in f:
                vf[v].append(fi)
        for a, b in short:
            a, b = parent[a], parent[b]
            if a == b:
                continue
            opp = set()
            for fi in vf[a]:
                if b in F2[fi]:
                    opp |= set(F2[fi]) - {a, b}
            if (nbr[a] & nbr[b]) != opp or len(opp) != 2:
                continue                       # link condition fails
            for fi in vf[b]:
                F2[fi][F2[fi] == b] = a
            vf[a] += vf[b]; nbr[a] |= nbr[b]; nbr[a] -= {a, b}
            for w in nbr[b]:
                nbr[w].discard(b)
                if w != a:
                    nbr[w].add(a)
            parent[parent == b] = a
            n += 1
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return F2, ok, n


def compact(V, F):
    used, inv = np.unique(F, return_inverse=True)
    return V[used], inv.reshape(F.shape), used


def topology_report(V, F, deg_area):
    m = trimesh.Trimesh(V, F, process=False)
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    _, A, _ = face_normals_areas(V, F)
    sf = np.sort(F, axis=1)
    rep = {
        "vertices": int(len(V)),
        "faces": int(len(F)),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "boundary_edges": int((cnt == 1).sum()),
        "non_manifold_edges": int((cnt > 2).sum()),
        "degenerate_faces": int((A < deg_area).sum()),
        "duplicate_faces": int(len(sf) - len(np.unique(sf, axis=0))),
        "euler_number": int(m.euler_number),
        "connected_components": int(len(m.split(only_watertight=False))) if len(F) < 2_000_000 else None,
    }
    if rep["watertight"] and rep["winding_consistent"]:
        rep["volume"] = float(m.volume)
    return rep


def export_layers(V, F, layer, layer_names, out_dir, stem):
    """OBJ with one group/object per layer, plus one STL per layer."""
    scene = trimesh.Scene()
    written = {}
    for i, nm in enumerate(layer_names):
        fs = layer == i
        if not fs.any():
            continue
        v, f, _ = compact(V, F[fs])
        sub = trimesh.Trimesh(v, f, process=False)
        scene.add_geometry(sub, node_name=nm, geom_name=nm)
        p = f"{out_dir}/{stem}_{nm}.stl"
        sub.export(p)
        written[nm] = {"faces": int(fs.sum()), "stl": p}
    scene.export(f"{out_dir}/{stem}_layers.obj", include_normals=False)
    return written
