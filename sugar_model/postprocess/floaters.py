"""(a) Floater / noise pruning.

Gaussian-splat reconstructions leave "floaters": blobs of geometry in free
space (in front of cameras, from reflections or bad depth), plus tiny islands
of a few triangles.  Real hardware (a pull that Poisson separated from its
door by a millimetre) also shows up as a small separate component, so size
alone is not enough.  Rule used here:

* keep the largest component (the room) and anything with area >= `keep_frac`
  of it;
* remove components with fewer than `min_faces` faces;
* remove thin/needle components (smallest PCA extent < `thin_ratio` x largest
  extent) that are not touching the main surface;
* remove other small components only if they are farther than `near_tol`
  from the main surface (free-space floaters); small components hugging the
  surface are kept and later labelled as hardware candidates.

Optional confidence filter: if the PLY carries a per-vertex `opacity`,
`confidence`, `quality` or `density` property, faces whose vertices fall
below `conf_thresh` are dropped first.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .common import components, face_adjacency, face_normals_areas, log

CONF_KEYS = ("opacity", "confidence", "quality", "density")


def confidence_filter(V, F, vertex_attrs: dict, thresh: float | None):
    if thresh is None or not vertex_attrs:
        return F, None
    for k in CONF_KEYS:
        if k in vertex_attrs:
            c = np.asarray(vertex_attrs[k], float).reshape(-1)
            if len(c) != len(V):
                continue
            keep = (c[F] >= thresh).all(axis=1)
            log.info("confidence filter on '%s' >= %.3f: dropped %d faces", k, thresh, (~keep).sum())
            return F[keep], k
    return F, None


def prune_floaters(V, F, keep_frac=0.02, min_faces=20, near_tol_rel=0.005,
                   thin_ratio=0.05):
    """Return (F_kept, info).  `near_tol_rel` is relative to the main bbox diagonal."""
    _, A, C = face_normals_areas(V, F)
    lab = components(len(F), face_adjacency(F))
    ncomp = lab.max() + 1
    comp_area = np.bincount(lab, weights=A, minlength=ncomp)
    comp_nf = np.bincount(lab, minlength=ncomp)
    main = int(np.argmax(comp_area))
    mainV = np.unique(F[lab == main])
    diag = float(np.linalg.norm(V[mainV].max(0) - V[mainV].min(0)))
    near_tol = near_tol_rel * diag
    tree = cKDTree(V[mainV])

    keep_comp = np.zeros(ncomp, bool)
    reasons = {"tiny": 0, "thin": 0, "free_space": 0}
    kept_small = []
    for c in range(ncomp):
        if c == main or comp_area[c] >= keep_frac * comp_area[main]:
            keep_comp[c] = True
            continue
        if comp_nf[c] < min_faces:
            reasons["tiny"] += 1
            continue
        verts = np.unique(F[lab == c])
        dmin = float(tree.query(V[verts], k=1)[0].min())
        P = V[verts] - V[verts].mean(0)
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(P.T @ P / len(P)), 0))
        if dmin > near_tol:
            if ev[0] < thin_ratio * ev[2]:
                reasons["thin"] += 1
            else:
                reasons["free_space"] += 1
            continue
        keep_comp[c] = True
        kept_small.append(c)
    keep = keep_comp[lab]
    info = {
        "components_in": int(ncomp),
        "components_kept": int(keep_comp.sum()),
        "removed": reasons,
        "faces_removed": int((~keep).sum()),
        "small_components_kept_near_surface": len(kept_small),
        "near_tol_input_units": near_tol,
    }
    log.info("floaters: %d components -> kept %d (removed %s, %d faces)",
             ncomp, keep_comp.sum(), reasons, (~keep).sum())
    # per-face flag: face belongs to a small kept component (hardware candidate)
    small_flag = np.isin(lab, kept_small)[keep]
    return F[keep], small_flag, info
