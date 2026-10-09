"""(c) Iterative RANSAC planar segmentation, Manhattan snapping, projection.

Pipeline (all distances in output units, mesh already Manhattan-aligned):

1. `ransac_planes` - iterative 1-point-with-normal RANSAC on face centroids,
   area weighted: a hypothesis is (face normal, face centroid).  Inliers must
   be within `dist_tol` of the plane AND have a face normal within `ang_tol`
   (signed, so the two sides of a thin panel are different planes).  Inlier
   sets are split into connected patches and patches smaller than
   `min_area` are returned to the pool (prevents an infinite plane from
   stealing coplanar faces elsewhere).  Each accepted plane is re-fit by
   area-weighted PCA (twice, re-gathering inliers).
2. `refine_frame_and_snap` - Kabsch-refines the global rotation using the
   precise PCA plane normals (much better than raw face normals), then snaps
   every normal within `snap_deg` of a Manhattan axis exactly onto it and
   re-solves its offset.  Orthogonal surfaces therefore meet at exactly 90 deg.
3. `merge_planes`, `grow_regions` - merge duplicates; absorb the rounded
   crease strips (faces whose vertices are all within `dist_tol` of an
   adjacent patch's plane).
4. `project_vertices` - each vertex is moved to the closest point satisfying
   all planes of its incident labelled faces: 1 plane -> projection,
   2 planes -> intersection line, 3 planes -> corner point.  This flattens
   micro-ripples and makes creases/corners exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .common import components, face_normals_areas, kabsch, log


@dataclass
class Plane:
    n: np.ndarray            # unit normal (oriented like the faces)
    d: float                 # n . x = d
    faces: np.ndarray        # inlier face indices
    rms_fit: float = 0.0     # RMS of inlier vertices to the un-snapped fit
    snapped_axis: str | None = None
    snap_angle_deg: float = 0.0
    n_fit: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _fit_plane(V, F, faces, A, Nf):
    """Area-weighted PCA fit through vertices of the given faces."""
    vid, inv = np.unique(F[faces], return_inverse=True)
    w = np.bincount(inv.ravel(), weights=np.repeat(A[faces], 3), minlength=len(vid))
    P = V[vid]
    c = (P * w[:, None]).sum(0) / w.sum()
    Q = P - c
    M = (Q * w[:, None]).T @ Q
    n = np.linalg.eigh(M)[1][:, 0]
    if n @ (Nf[faces] * A[faces, None]).sum(0) < 0:
        n = -n
    d = float(n @ c)
    rms = float(np.sqrt(np.average((P @ n - d) ** 2, weights=w)))
    return n, d, rms


def _minor_width(V, F, faces, n):
    """5-95 percentile width of a patch along its in-plane minor axis."""
    P = V[np.unique(F[faces])]
    P = P - P.mean(0)
    P = P - np.outer(P @ n, n)
    w, U = np.linalg.eigh(P.T @ P)
    t = P @ U[:, 1]          # eigvals ascending: [~0 (normal), minor, major]
    lo, hi = np.percentile(t, [5, 95])
    return float(hi - lo)


def ransac_planes(V, F, adj, dist_tol, min_area, ang_tol_deg=20.0, n_hyp=64,
                  n_eval=30000, max_planes=300, max_fail=40, seed=0,
                  oblique_min_width=None, manhattan_deg=20.0, snap_deg=5.0,
                  reject_near_axis_deg=20.0):
    rng = np.random.default_rng(seed)
    Nf, A, C = face_normals_areas(V, F)
    nF = len(F)
    pool = np.ones(nF, bool)
    cos_t = np.cos(np.radians(ang_tol_deg))
    planes: list[Plane] = []
    fails = 0
    n_strips = 0
    while len(planes) < max_planes and fails < max_fail:
        idx = np.flatnonzero(pool)
        if A[idx].sum() < min_area:
            break
        p = A[idx] / A[idx].sum()
        seeds = idx[rng.choice(len(idx), size=n_hyp, p=p)]
        ev = idx[rng.choice(len(idx), size=min(n_eval, len(idx)), replace=False)]
        n_h = Nf[seeds]; d_h = np.einsum("ij,ij->i", n_h, C[seeds])
        inl = (np.abs(C[ev] @ n_h.T - d_h) < dist_tol) & (Nf[ev] @ n_h.T > cos_t)
        best = int(np.argmax(A[ev] @ inl))
        n, d = n_h[best], d_h[best]
        accepted = None
        for _ in range(3):   # gather -> connected patches -> refit
            m = pool & (np.abs(C @ n - d) < dist_tol) & (Nf @ n > cos_t)
            lab = components(nF, adj, m)
            if lab.max() < 0:
                break
            pa = np.bincount(lab[m], weights=A[m])
            good = np.flatnonzero(pa >= min_area)
            if len(good) == 0:
                accepted = None
                break
            faces = np.flatnonzero(m & np.isin(lab, good))
            n, d, rms = _fit_plane(V, F, faces, A, Nf)
            accepted = (faces, n, d, rms)
        if accepted is None:
            fails += 1
            continue
        faces, n, d, rms = accepted
        pool[faces] = False
        off_axis = float(np.degrees(np.arccos(min(1.0, np.abs(n).max()))))
        oblique = off_axis > manhattan_deg
        if snap_deg < off_axis <= reject_near_axis_deg:
            # 5-20 deg off an axis: a fit bridging a small step (e.g. door slab vs
            # carcass), not a real sloped surface in a kitchen -> leave residual
            n_strips += 1
            continue
        if oblique and oblique_min_width and _minor_width(V, F, faces, n) < oblique_min_width:
            # rounded crease strip (Poisson blur), not a real surface: leave as
            # residual so the adjacent planes can absorb it and meet sharply
            n_strips += 1
            continue
        planes.append(Plane(n=n, d=d, faces=faces, rms_fit=rms, n_fit=n.copy()))
        fails = 0
    log.info("RANSAC: %d planes cover %.1f%% of area (%d crease strips rejected, stop: %s)",
             len(planes), 100 * (1 - A[pool].sum() / A.sum()), n_strips,
             "max_fail" if fails >= max_fail else "pool exhausted/max_planes")
    return planes


AXES = np.eye(3)
AXIS_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def _nearest_axis(n):
    k = int(np.argmax(np.abs(n)))
    s = np.sign(n[k])
    return k, s, float(np.degrees(np.arccos(min(1.0, abs(n[k])))))


def refine_frame_and_snap(V, F, planes, snap_deg=5.0):
    """Kabsch-refine the frame with plane normals, then snap.  Returns (V, R)."""
    _, A, _ = face_normals_areas(V, F)
    P, Q, W = [], [], []
    for pl in planes:
        k, s, ang = _nearest_axis(pl.n)
        if ang < snap_deg:
            P.append(pl.n); Q.append(s * AXES[k]); W.append(A[pl.faces].sum())
    R = np.eye(3)
    if len(P) >= 2:
        R = kabsch(np.array(P), np.array(Q), np.array(W))
        V = V @ R.T
        for pl in planes:
            pl.n = R @ pl.n
            pl.n_fit = R @ pl.n_fit
    log.info("frame refinement rotation: %.4f deg",
             np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    Nf, A, C = face_normals_areas(V, F)
    vdeg = np.bincount(F.ravel(), minlength=len(V))      # faces per vertex
    for pl in planes:
        k, s, ang = _nearest_axis(pl.n)
        if ang < snap_deg:
            pl.snapped_axis = AXIS_NAMES[2 * k + (0 if s > 0 else 1)]
            pl.snap_angle_deg = ang
            pl.n = s * AXES[k]
        # re-solve offset with the (possibly snapped) normal.  Use interior
        # vertices only (all incident faces are inliers): crease vertices are
        # pulled inward by Poisson/Gaussian blur and would bias narrow faces.
        vid, inv = np.unique(F[pl.faces], return_inverse=True)
        w = np.bincount(inv.ravel(), weights=np.repeat(A[pl.faces], 3), minlength=len(vid))
        deg_in = np.bincount(inv.ravel(), minlength=len(vid))
        interior = deg_in == vdeg[vid]
        sel = interior if interior.sum() >= 10 else np.ones(len(vid), bool)
        pl.d = float(np.average(V[vid[sel]] @ pl.n, weights=w[sel]))
        pl.extra["rms_after_snap_before_projection"] = float(
            np.sqrt(np.average((V[vid] @ pl.n - pl.d) ** 2, weights=w)))
    return V, R


def drop_bridging_planes(planes, dist_tol, max_rms_frac=0.5):
    """After snapping, a plane whose inliers sit > max_rms_frac*tol (RMS) from it
    was bridging two offset surfaces; drop it (faces return to residual)."""
    keep = [p for p in planes
            if p.extra.get("rms_after_snap_before_projection", 0.0) <= max_rms_frac * dist_tol]
    from .common import log as _log
    _log.info("dropped %d bridging planes after snapping", len(planes) - len(keep))
    return keep


def merge_planes(V, F, planes, dist_tol, ang_deg=2.0):
    """Merge planes with (nearly) equal normal and offset."""
    _, A, _ = face_normals_areas(V, F)
    planes = sorted(planes, key=lambda p: -A[p.faces].sum())
    out: list[Plane] = []
    for pl in planes:
        for q in out:
            if q.n @ pl.n > np.cos(np.radians(ang_deg)) and abs(q.d - pl.d) < dist_tol:
                wq, wp = A[q.faces].sum(), A[pl.faces].sum()
                q.d = (q.d * wq + pl.d * wp) / (wq + wp)
                q.faces = np.concatenate([q.faces, pl.faces])
                break
        else:
            out.append(pl)
    return out


def face_labels(nF, planes):
    lab = -np.ones(nF, np.int64)
    for i, pl in enumerate(planes):
        lab[pl.faces] = i
    return lab


def grow_regions(V, F, adj, lab, planes, dist_tol, max_rings=6, gate_deg=50.0):
    """Absorb residual faces adjacent to a plane if all 3 vertices are within tol
    and the face normal is within gate_deg of the plane normal."""
    nrm = np.array([p.n for p in planes]); off = np.array([p.d for p in planes])
    Nf, _, _ = face_normals_areas(V, F)
    cos_gate = np.cos(np.radians(gate_deg))
    for ring in range(max_rings):
        a, b = adj[:, 0], adj[:, 1]
        pairs = np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)])
        pairs = pairs[(lab[pairs[:, 0]] < 0) & (lab[pairs[:, 1]] >= 0)]
        if len(pairs) == 0:
            break
        f, g = pairs[:, 0], pairs[:, 1]
        pid = lab[g]
        dist = np.abs(np.einsum("fkj,fj->fk", V[F[f]], nrm[pid]) - off[pid][:, None]).max(1)
        # normal gate: faces on the steep part of a rounded crease belong to the
        # other plane; absorbing them here would fold when projected
        ok = (dist < dist_tol) & (np.einsum("ij,ij->i", Nf[f], nrm[pid]) > cos_gate)
        f, pid, dist = f[ok], pid[ok], dist[ok]
        if len(f) == 0:
            break
        order = np.lexsort((dist, f))
        f, pid = f[order], pid[order]
        first = np.r_[True, f[1:] != f[:-1]]
        lab[f[first]] = pid[first]
    for i, pl in enumerate(planes):
        pl.faces = np.flatnonzero(lab == i)
    return lab


def _face_neighbors(nF, adj):
    """(nF,3) neighbour faces (-1 padded; extra non-manifold neighbours ignored)."""
    pairs = np.concatenate([adj, adj[:, ::-1]])
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    first = np.searchsorted(pairs[:, 0], pairs[:, 0], "left")
    slot = np.arange(len(pairs)) - first
    ok = slot < 3
    nb = -np.ones((nF, 3), np.int64)
    nb[pairs[ok, 0], slot[ok]] = pairs[ok, 1]
    return nb


def absorb_on_plane_faces(F, lab, n_planes, iters=5):
    """A residual face whose 3 vertices all touch faces of plane P will be
    flattened onto P by the projection anyway; make it part of P so it does
    not become a folded/degenerate sliver."""
    import scipy.sparse as sp
    nV = int(F.max()) + 1
    for _ in range(iters):
        m = lab >= 0
        M = sp.csr_matrix((np.ones(m.sum() * 3), (F[m].ravel(), np.repeat(lab[m], 3))),
                          shape=(nV, n_planes))
        M.data[:] = 1.0
        res = np.flatnonzero(~m)
        if len(res) == 0:
            break
        C = M[F[res, 0]].multiply(M[F[res, 1]]).multiply(M[F[res, 2]]).tocsr()
        has = np.diff(C.indptr) > 0
        if not has.any():
            break
        rows = np.flatnonzero(has)
        lab[res[rows]] = C.indices[C.indptr[rows]]
    return lab


def absorb_islands(V, F, adj, lab, planes, dist_tol, max_area, max_dist_factor=2.5):
    """Residual islands (noise spikes, splat bumps) entirely surrounded by ONE
    plane, small and within max_dist_factor*tol of it, join that plane.
    Hardware (pulls stick out 25+ mm) fails the distance test and is kept."""
    nrm = np.array([p.n for p in planes]); off = np.array([p.d for p in planes])
    _, A, _ = face_normals_areas(V, F)
    res = lab < 0
    if not res.any():
        return lab
    comp = components(len(F), adj, res)
    nc = comp.max() + 1
    a, b = adj[:, 0], adj[:, 1]
    pairs = np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)])
    pairs = pairs[res[pairs[:, 0]] & ~res[pairs[:, 1]]]
    cp = np.unique(np.stack([comp[pairs[:, 0]], lab[pairs[:, 1]]], 1), axis=0)
    nplanes = np.bincount(cp[:, 0], minlength=nc)
    plane_of = np.full(nc, -1); plane_of[cp[:, 0]] = cp[:, 1]
    area = np.bincount(comp[res], weights=A[res], minlength=nc)
    fr = np.flatnonzero(res)
    pc = plane_of[comp[fr]]
    okf = pc >= 0
    dist = np.full(len(fr), np.inf)
    dist[okf] = np.abs(np.einsum("fkj,fj->fk", V[F[fr[okf]]], nrm[pc[okf]]) - off[pc[okf]][:, None]).max(1)
    cmax = np.zeros(nc); np.maximum.at(cmax, comp[fr], dist)
    good = (nplanes == 1) & (area < max_area) & (cmax < max_dist_factor * dist_tol)
    sel = good[comp[fr]]
    lab[fr[sel]] = plane_of[comp[fr[sel]]]
    from .common import log as _log
    _log.info("absorbed %d residual islands (%d faces) into their surrounding plane", good.sum(), sel.sum())
    return lab


def smooth_labels(V, F, adj, lab, planes, dist_tol, iters=3, gate_deg=50.0):
    """Majority filter: a face with >= 2 neighbours on plane q (and all its
    vertices within tol of q) joins q.  Removes zig-zag/pinch boundaries."""
    nrm = np.array([p.n for p in planes]); off = np.array([p.d for p in planes])
    nb = _face_neighbors(len(F), adj)
    Nf, _, _ = face_normals_areas(V, F)
    cos_gate = np.cos(np.radians(gate_deg))
    for _ in range(iters):
        L = np.where(nb >= 0, lab[np.maximum(nb, 0)], -2)
        changed = 0
        for a, b in ((0, 1), (0, 2), (1, 2)):
            q = L[:, a]
            cand = (q == L[:, b]) & (q >= 0) & (q != lab)
            f = np.flatnonzero(cand)
            if len(f) == 0:
                continue
            qq = q[f]
            dist = np.abs(np.einsum("fkj,fj->fk", V[F[f]], nrm[qq]) - off[qq][:, None]).max(1)
            ok = (dist < dist_tol) & (np.einsum("ij,ij->i", Nf[f], nrm[qq]) > cos_gate)
            lab[f[ok]] = qq[ok]
            changed += ok.sum()
        if changed == 0:
            break
    for i, pl in enumerate(planes):
        pl.faces = np.flatnonzero(lab == i)
    return lab


def split_patches(F, adj, lab, planes, A, min_patch_area):
    """Connected patches per plane.  Tiny patches go back to residual (-1)."""
    patch = -np.ones(len(F), np.int64)
    patch_plane = []
    m = lab >= 0
    # only link faces of the same plane
    same = lab[adj[:, 0]] == lab[adj[:, 1]]
    comp = components(len(F), adj[same], m)
    if comp.max() < 0:
        return patch, np.zeros(0, int)
    ca = np.bincount(comp[m], weights=A[m])
    cplane = np.zeros(len(ca), np.int64)
    cplane[comp[m]] = lab[m]
    keep = ca >= min_patch_area
    newid = -np.ones(len(ca), np.int64)
    newid[keep] = np.arange(keep.sum())
    patch[m] = newid[comp[m]]
    patch_plane = cplane[keep]
    return patch, patch_plane


def project_vertices(V, F, patch, patch_plane, planes, max_move, par_deg=20.0):
    """Constrained projection of every vertex onto the planes of its labelled faces."""
    nrm = np.array([p.n for p in planes]); off = np.array([p.d for p in planes])
    _, A, _ = face_normals_areas(V, F)
    lab = np.where(patch >= 0, patch_plane[np.maximum(patch, 0)], -1)
    vf = F.reshape(-1)
    fl = np.repeat(lab, 3)
    fa = np.repeat(A, 3)
    sel = fl >= 0
    vf, fl, fa = vf[sel], fl[sel], fa[sel]
    # unique (vertex, plane) with summed incident area
    key = vf.astype(np.int64) * (len(planes) + 1) + fl
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=fa)
    uv, up = uk // (len(planes) + 1), uk % (len(planes) + 1)
    order = np.lexsort((-w, uv))
    uv, up, w = uv[order], up[order], w[order]
    starts = np.flatnonzero(np.r_[True, uv[1:] != uv[:-1]])
    ends = np.r_[starts[1:], len(uv)]
    Vn = V.copy()
    cospar = np.cos(np.radians(par_deg))
    nplanes_per_v = np.zeros(len(V), np.int64)
    conflict = np.zeros(len(V), bool)
    capped = 0
    # fast path: vertices touching exactly one plane
    single = (ends - starts) == 1
    sv, sp_ = uv[starts[single]], up[starts[single]]
    Vn[sv] = V[sv] - nrm[sp_] * (np.einsum("ij,ij->i", V[sv], nrm[sp_]) - off[sp_])[:, None]
    nplanes_per_v[sv] = 1
    for s, e in zip(starts[~single], ends[~single]):
        v = uv[s]
        chosen = []
        par_conflict = False
        for j in range(s, e):
            p = up[j]
            if any(abs(nrm[p] @ nrm[q]) >= cospar and abs(off[p] - (nrm[p] @ nrm[q]) * off[q]) > 1e-9
                   for q in chosen):
                par_conflict = True          # two parallel planes claim this vertex
                break
            if all(abs(nrm[p] @ nrm[q]) < cospar for q in chosen):
                if len(chosen) == 2:
                    N2 = nrm[chosen]
                    if abs(np.linalg.det(np.vstack([N2, nrm[p]]))) < 0.2:
                        continue
                chosen.append(p)
                if len(chosen) == 3:
                    break
        if par_conflict:
            # a step/slot thinner than the mesh resolution: flattening onto either
            # plane would collapse it into a zero-thickness sheet, so leave the
            # vertex alone and let the caller mark its faces as residual
            conflict[v] = True
            continue
        N = nrm[chosen]; dd = off[chosen]
        x = V[v]
        r = N @ x - dd
        y = x - N.T @ np.linalg.solve(N @ N.T, r)
        if np.linalg.norm(y - x) > max_move:
            p = chosen[0]
            y = x - nrm[p] * (nrm[p] @ x - off[p])
            capped += 1
            chosen = chosen[:1]
        Vn[v] = y
        nplanes_per_v[v] = len(chosen)
    log.info("projection: %d vertices on 1 plane, %d on edges, %d at corners (%d capped, %d parallel-plane conflicts left as is)",
             (nplanes_per_v == 1).sum(), (nplanes_per_v == 2).sum(),
             (nplanes_per_v == 3).sum(), capped, conflict.sum())
    return Vn, conflict
