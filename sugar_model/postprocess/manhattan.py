"""(b) Manhattan frame estimation and rotation to a Z-up, axis-aligned frame.

Kitchens are "Manhattan worlds": almost all area lies on three mutually
orthogonal plane families (floor/counters, two wall directions).  We find the
three dominant directions from area-weighted face normals:

1. d1 = densest normal direction (antipodal-symmetric vote, then eigen-refine)
2. d2 = densest direction among normals roughly perpendicular to d1
3. d3 = d1 x d2, then joint Kabsch refinement: every face normal within 15 deg
   of an axis votes for that axis, and we solve for the rotation that best
   aligns them (iterated).

Which axis is "up" is not knowable from normals alone.  Priority:
  --up hint  >  COLMAP camera poses (cameras are held upright)  >  heuristic.
Heuristic: the floor is the largest plane that sits at the extreme low end of
the scene with its normal pointing into the scene; we score all 6 signed axis
directions that way and take the best.  The Y axis is chosen the same way
among the horizontal directions (largest wall = back wall, Y points into the
room), X = Y x Z so the frame is always right-handed.
"""
from __future__ import annotations

import struct
import numpy as np

from .common import kabsch, log


def _dominant_direction(n, w, rng, n_cand=300, n_eval=20000, cos_tol=np.cos(np.radians(10))):
    p = w / w.sum()
    cand = n[rng.choice(len(n), size=min(n_cand, len(n)), p=p)]
    ev = rng.choice(len(n), size=min(n_eval, len(n)), p=p, replace=len(n) < n_eval)
    score = (np.abs(n[ev] @ cand.T) > cos_tol).sum(0)
    d = cand[np.argmax(score)]
    for tol in (10, 6, 4):
        inl = np.abs(n @ d) > np.cos(np.radians(tol))
        M = (n[inl] * w[inl, None]).T @ n[inl]
        d = np.linalg.eigh(M)[1][:, -1]
    return d / np.linalg.norm(d)


def estimate_manhattan(normals, areas, seed=0, iters=4):
    """Return R (3x3) whose rows are the three Manhattan axes (in input frame)
    and the fraction of area explained by them."""
    rng = np.random.default_rng(seed)
    d1 = _dominant_direction(normals, areas, rng)
    perp = np.abs(normals @ d1) < np.sin(np.radians(10))
    q = normals[perp] - np.outer(normals[perp] @ d1, d1)
    q /= np.maximum(np.linalg.norm(q, axis=1), 1e-12)[:, None]
    d2 = _dominant_direction(q, areas[perp], rng)
    d2 -= d1 * (d2 @ d1); d2 /= np.linalg.norm(d2)
    R = np.stack([d1, d2, np.cross(d1, d2)])
    for _ in range(iters):
        m = normals @ R.T                       # normals in Manhattan coords
        k = np.argmax(np.abs(m), axis=1)
        mk = m[np.arange(len(m)), k]
        ok = np.abs(mk) > np.cos(np.radians(15))
        tgt = np.zeros_like(m[ok]); tgt[np.arange(ok.sum()), k[ok]] = np.sign(mk[ok])
        Q = kabsch(m[ok], tgt, areas[ok])
        R = Q @ R
    m = normals @ R.T
    explained = float(areas[np.abs(m).max(1) > np.cos(np.radians(10))].sum() / areas.sum())
    return R, explained


def _extreme_plane_score(V, normals, areas, centroids, u, band=0.10):
    """Area of faces facing +u that lie within `band` of the low end along u."""
    h = V @ u
    lo, ext = h.min(), np.ptp(h)
    hc = centroids @ u
    sel = (normals @ u > np.cos(np.radians(10))) & (hc < lo + band * ext)
    return float(areas[sel].sum())


# ---------------------------------------------------------------- COLMAP up
def _qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def colmap_up(path: str) -> np.ndarray:
    """Average world-space 'up' of COLMAP cameras (images.txt or images.bin).

    COLMAP camera +Y points down in the image, so world up = -R^T [0,1,0].
    Phones are held upright while filming a kitchen, so this is a good
    gravity estimate (used only to pick which Manhattan axis is Z)."""
    rots = []
    if path.endswith(".txt"):
        # two lines per image; the POINTS2D line may be empty, so keep blanks
        lines = [l for l in open(path) if not l.startswith("#")]
        for l in lines[::2]:
            e = l.split()
            if len(e) >= 10:
                rots.append(_qvec2rot(np.array(list(map(float, e[1:5])))))
    else:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                f.read(4)
                q = struct.unpack("<4d", f.read(32)); f.read(24); f.read(4)
                while f.read(1) != b"\x00":
                    pass
                npts = struct.unpack("<Q", f.read(8))[0]
                f.read(24 * npts)
                rots.append(_qvec2rot(np.array(q)))
    ups = np.array([-R.T @ np.array([0, 1.0, 0]) for R in rots])
    u = ups.mean(0)
    return u / np.linalg.norm(u)


def choose_axes(R_m, V, normals, areas, centroids, up_hint=None):
    """Order/sign the Manhattan axes into a proper rotation (rows X, Y, Z)."""
    axes = [s * R_m[i] for i in range(3) for s in (1, -1)]
    if up_hint is not None:
        up_hint = np.asarray(up_hint, float); up_hint /= np.linalg.norm(up_hint)
        z = max(axes, key=lambda a: a @ up_hint)
        how = "hint"
    else:
        scores = [_extreme_plane_score(V, normals, areas, centroids, a) for a in axes]
        z = axes[int(np.argmax(scores))]
        how = "floor-heuristic"
    horiz = [a for a in axes if abs(a @ z) < 0.5]
    sc = [_extreme_plane_score(V, normals, areas, centroids, a) for a in horiz]
    y = horiz[int(np.argmax(sc))]
    x = np.cross(y, z)
    R = np.stack([x, y, z])
    log.info("up axis chosen by %s", how)
    return R, how
