"""(d) Planar-region decimation that preserves watertightness.

A SuGaR mesh spends ~1M triangles; a cabinet door needs two.  For every
planar patch we:

1. extract its boundary loops from the half-edge structure (outer loop CCW,
   holes CW when seen from the patch normal);
2. drop "removable" loop vertices: vertices that lie on a straight crease
   shared by exactly two planar patches (non-parallel planes), touch no
   residual (non-planar) faces and no open border.  Both patches drop the
   same vertices, so the shared edge stays conforming (no T-junctions);
3. re-triangulate the simplified polygon-with-holes in the plane's 2-D frame
   with manifold3d's constrained triangulator (uses only loop vertices, no
   Steiner points, preserves the input edges).

Non-planar residual geometry (handles, hinges, faucets, rounded strips we
could not explain) is left untouched, so detail is preserved exactly where
curvature is high.  Any patch whose loops are not simple (pinch vertices,
non-manifold edges) or whose triangulation does not reproduce the polygon
area keeps its original triangles; the removable set is then recomputed so
its neighbours stay conforming.
"""
from __future__ import annotations

import numpy as np
import manifold3d as m3d

from .common import face_normals_areas, log


def _halfedges(F):
    he = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    hf = np.repeat(np.arange(len(F)), 3)
    return he, hf


def _twin_info(he, nV):
    """For each half-edge: index of its twin (-1 if none, -2 if non-manifold)."""
    key = he[:, 0].astype(np.int64) * nV + he[:, 1]
    tkey = he[:, 1].astype(np.int64) * nV + he[:, 0]
    order = np.argsort(key)
    sk = key[order]
    lo = np.searchsorted(sk, tkey, "left")
    hi = np.searchsorted(sk, tkey, "right")
    twin = np.full(len(he), -1, np.int64)
    one = (hi - lo) == 1
    twin[one] = order[lo[one]]
    twin[(hi - lo) > 1] = -2
    # duplicate directed edges are also non-manifold
    dup = np.zeros(len(he), bool)
    same = sk[1:] == sk[:-1]
    dup[order[1:][same]] = True; dup[order[:-1][same]] = True
    twin[dup] = -2
    return twin


def _removable(F, nV, face_group, retri, patch_plane, nrm, par_cos):
    """Vertices that can be dropped from both adjacent loops."""
    he, hf = _halfedges(F)
    twin = _twin_info(he, nV)
    g = face_group[hf]
    okF = (face_group >= 0) & retri[np.maximum(face_group, 0)]
    bad = np.zeros(nV, bool)
    bad[F[~okF].ravel()] = True                              # touches residual/fallback
    bad[he[twin < 0].ravel()] = True                         # open border / non-manifold
    # crease half-edges (twin in a different patch)
    tg = np.where(twin >= 0, g[np.maximum(twin, 0)], -9)
    crease = (twin >= 0) & (tg != g)
    ncrease = np.bincount(he[crease, 0], minlength=nV)       # outgoing crease half-edges
    # distinct patches per vertex: min & max of incident patch ids must differ by
    # exactly the two patches; count distinct via unique pairs
    vp = np.unique(np.stack([F.ravel(), np.repeat(face_group, 3)], 1), axis=0)
    cnt = np.bincount(vp[:, 0], minlength=nV)
    pmin = np.full(nV, np.iinfo(np.int64).max); pmax = np.full(nV, -1)
    np.minimum.at(pmin, vp[:, 0], vp[:, 1]); np.maximum.at(pmax, vp[:, 0], vp[:, 1])
    rem = (~bad) & (cnt == 2) & (ncrease == 2)
    idx = np.flatnonzero(rem)
    na = nrm[patch_plane[pmin[idx]]]; nb = nrm[patch_plane[pmax[idx]]]
    rem[idx[np.abs(np.einsum("ij,ij->i", na, nb)) > par_cos]] = False
    return rem


def _loops_for_patch(he_idx, he, twin, bmask):
    """Chain a patch's boundary half-edges into closed loops.

    At each loop vertex the next boundary half-edge is found by walking around
    the vertex *through the patch's own faces* (next-in-face, then twin), so
    pinch vertices where a patch touches itself are handled correctly.
    Returns a list of vertex loops, or None if the walk fails (non-manifold)."""
    loops, used = [], set()
    for h0 in he_idx:
        if h0 in used:
            continue
        loop, h = [], h0
        for _ in range(len(he_idx) + 1):
            used.add(h)
            loop.append(int(he[h, 0]))
            j = 3 * (h // 3) + (h % 3 + 1) % 3          # next half-edge in face
            guard = 0
            while not bmask[j]:
                t = twin[j]
                if t < 0 or guard > 64:
                    return None
                j = 3 * (t // 3) + (t % 3 + 1) % 3
                guard += 1
            h = j
            if h == h0:
                break
        else:
            return None
        if h != h0:
            return None
        loops.append(loop)
    return loops


def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _crossing_vertices(polys, ids):
    """Vertex ids of loop segments that cross other (non-adjacent) segments."""
    import shapely
    segs, sid = [], []
    k = 0
    for P in polys:
        n = len(P)
        idx = np.arange(n)
        segs.append(np.stack([P[idx], P[(idx + 1) % n]], 1))
        sid.append(np.stack([np.asarray(ids[k:k + n])[idx], np.asarray(ids[k:k + n])[(idx + 1) % n]], 1))
        k += n
    S = np.concatenate(segs); I = np.concatenate(sid)
    geoms = shapely.linestrings(S)
    tree = shapely.STRtree(geoms)
    a, b = tree.query(geoms, predicate="intersects")
    m = a < b
    a, b = a[m], b[m]
    share = (I[a, 0] == I[b, 0]) | (I[a, 0] == I[b, 1]) | (I[a, 1] == I[b, 0]) | (I[a, 1] == I[b, 1])
    a, b = a[~share], b[~share]
    return np.unique(np.concatenate([I[a].ravel(), I[b].ravel()]))


def decimate_planar(V, F, patch, patch_plane, planes, par_deg=20.0, max_rounds=5,
                    max_fold_area=1.0, max_fold_frac=1e-5, max_peels=6):
    """Returns (F_new, patch_new, stats)."""
    nV = len(V)
    nP = len(patch_plane)
    nrm = np.array([p.n for p in planes]) if planes else np.zeros((0, 3))
    Nf, A, _ = face_normals_areas(V, F)
    par_cos = np.cos(np.radians(par_deg))
    retri = np.ones(max(nP, 1), bool)
    he, hf = _halfedges(F)
    twin = _twin_info(he, nV)
    patch = patch.copy()
    # orientation per patch (face winding vs plane normal)
    orient = np.ones(nP)
    for p in range(nP):
        fs = patch == p
        if (Nf[fs] * A[fs, None]).sum(0) @ nrm[patch_plane[p]] < 0:
            orient[p] = -1.0
    g0 = patch[hf]
    retri[np.unique(g0[(twin == -2) & (g0 >= 0)])] = False   # non-manifold patches
    peels = np.zeros(max(nP, 1), int)
    vf_idx = None

    new_tris = {}
    reasons = {}

    def fail(p, why):
        failed.append(p)
        reasons[int(p)] = why

    for rnd in range(max_rounds + max_peels):
        g = patch[hf]
        tg = np.where(twin >= 0, g[np.maximum(twin, 0)], -9)
        bnd = (g >= 0) & (tg != g)
        bidx = np.flatnonzero(bnd)
        bidx = bidx[np.argsort(g[bidx], kind="stable")]
        bg = g[bidx]
        starts = np.searchsorted(bg, np.arange(nP), "left")
        ends = np.searchsorted(bg, np.arange(nP), "right")
        rem = _removable(F, nV, patch, retri, patch_plane, nrm, par_cos)
        failed = []
        peel_v = {}
        new_tris = {}
        for p in np.flatnonzero(retri[:nP]):
            loops = _loops_for_patch(bidx[starts[p]:ends[p]].tolist(), he, twin, bnd)
            if not loops:
                fail(p, "loop walk failed"); continue
            n = nrm[patch_plane[p]] * orient[p]
            u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
            w = np.cross(n, u)
            polys, ids = [], []
            okp = True
            for L in loops:
                L = [v for v in L if not rem[v]]
                if len(L) < 3:
                    okp = False; break
                P2 = np.stack([V[L] @ u, V[L] @ w], 1)
                polys.append(P2); ids.extend(L)
            if not okp:
                fail(p, "loop collapsed (<3 vertices)"); continue
            areas = [_signed_area(P2) for P2 in polys]
            outer = int(np.argmax(areas))
            inv_holes = [i for i, ar in enumerate(areas) if i != outer and ar > 0]
            if inv_holes:      # folded residual faces inside the patch -> peel them
                if peels[p] < max_peels:
                    k0 = np.cumsum([0] + [len(P2) for P2 in polys])
                    peel_v[p] = np.unique(np.concatenate([np.asarray(ids[k0[i]:k0[i + 1]]) for i in inv_holes]))
                    continue
                fail(p, "inverted hole loop"); continue
            try:
                tri = np.asarray(m3d.triangulate(polys), np.int64)
            except Exception as ex:
                fail(p, f"triangulator error {ex}"); continue
            allP = np.vstack(polys)
            target = sum(_signed_area(P2) for P2 in polys)
            if len(tri) == 0 or target <= 0:
                fail(p, f"empty triangulation / non-positive area {target:.3g}"); continue
            a, b, c = allP[tri[:, 0]], allP[tri[:, 1]], allP[tri[:, 2]]
            ta = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            # allow only microscopic folds (noise-level slivers along a boundary)
            neg = -ta[ta < 0].sum()
            if (neg > max_fold_frac * target or ta.min() < -max_fold_area
                    or abs(ta.sum() - target) > 1e-6 * abs(target) + 1e-9):
                bad_v = _crossing_vertices(polys, ids)
                if len(bad_v) and peels[p] < max_peels:
                    peel_v[p] = bad_v          # retry after peeling a ring
                    continue
                fail(p, f"area check: tri {ta.sum():.6g} vs polygon {target:.6g}, min tri {ta.min():.3g}"); continue
            new_tris[p] = np.asarray(ids, np.int64)[tri]
        if not failed and not peel_v:
            break
        retri[failed] = False
        for p, bv in peel_v.items():
            # turn the 1-ring of the self-intersecting loop vertices into residual
            # (they stay projected/flat; only the polygon boundary moves outward)
            ring = np.isin(F, bv).any(1) & (patch == p)
            patch[ring] = -1
            peels[p] += 1
        if peel_v:
            log.info("decimation round %d: peeled self-intersecting boundary rings on %d patches", rnd, len(peel_v))
        log.info("decimation round %d: %d patches fall back to original triangles", rnd, len(failed))

    keep_orig = (patch < 0) | ~retri[np.maximum(patch, 0)]
    F_parts = [F[keep_orig]]
    P_parts = [patch[keep_orig]]
    for p, t in new_tris.items():
        F_parts.append(t); P_parts.append(np.full(len(t), p))
    F_new = np.vstack(F_parts)
    patch_new = np.concatenate(P_parts)
    stats = {"patches": int(nP), "retriangulated": int(len(new_tris)),
             "fallback": int(nP - len(new_tris)), "fallback_reasons": reasons,
             "faces_in": int(len(F)),
             "faces_out": int(len(F_new))}
    log.info("decimation: %d -> %d faces (%d/%d patches re-triangulated)",
             len(F), len(F_new), len(new_tris), nP)
    return F_new, patch_new, stats
