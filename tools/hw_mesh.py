"""Turn a hardware SHAPE (materials/topknobs/README.md, "Geometry") into a triangle mesh, built the way three.js builds it.

    from hw_mesh import mesh, parts, mesh_bounds
    V, F = mesh(shape)            # float (n, 3) mm in the local frame, int (m, 3) counter-clockwise from outside
    for V, F, mat in parts(shape) # one entry per solid, mat = the part's "mat" (None = the metal finish)
    lo, hi = mesh_bounds(shape)   # or mesh_bounds((V, F))

extrude  ExtrudeGeometry(shape, {depth: depth - 2 bevel, bevelThickness: bevel, bevelSize: bevel, bevelOffset: -bevel,
         bevelSegments: 3}) centred on Z = 0, so the outer size is exactly the outline x depth. Caps reuse the
         triangulation of the original outline on the inset contour, and the bevel vectors follow getBevelVec, so a
         spike or a self-overlap three.js would show is shown here too.
lathe    LatheGeometry(profile, segments): 48 for round, 4 square, 6 hexagon, 8 octagon, started at pi / segments so the flats
         face Z (and X, except the hexagon, whose corners sit on X) and scaled by 1 / cos(pi / segments) so r is the half-width
         across flats. Then stretch X/Z, then rotate about Y.
group    each part's own rotate [rx, ry, rz] (degrees, Euler XYZ, R = Rx Ry Rz as three.js) then offset, recursively.

Raises ValueError on anything malformed. Run directly for a smoke test:  python3 tools/hw_mesh.py
"""
import math
import numpy as np

SEGMENTS = {'round': 48, 'square': 4, 'hexagon': 6, 'octagon': 8}
BEVEL_SEGMENTS = 3


def _pts(a, name, n=3):
    try:
        p = np.asarray(a, float)
    except (TypeError, ValueError):
        raise ValueError(f'{name} is not a list of [x, y] pairs')
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < n or not np.isfinite(p).all():
        raise ValueError(f'{name} needs >= {n} finite [x, y] pairs')
    return p


def _clean(p):
    """Drop repeated points and the closing duplicate, which three.js also strips before triangulating."""
    keep = np.r_[True, np.any(np.abs(np.diff(p, axis=0)) > 1e-9, axis=1)]
    p = p[keep]
    if len(p) > 1 and np.allclose(p[0], p[-1], atol=1e-9):
        p = p[:-1]
    return p


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _seg_hits(p, q, A, B):
    """Which segments A[i]-B[i] properly cross segment p-q."""
    d1, d2 = _cross(A, B, p[None]), _cross(A, B, q[None])
    d3, d4 = _cross(p[None], q[None], A), _cross(p[None], q[None], B)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _bridge(P, ring, hole):
    """Splice a hole (index list) into ring (index list) through a segment that crosses no edge, as earcut does."""
    m = hole[int(np.argmax(P[hole, 0]))]
    hole = hole[hole.index(m):] + hole[:hole.index(m)]
    r = np.array(ring)
    edges = np.array([(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
                     + [(hole[i], hole[(i + 1) % len(hole)]) for i in range(len(hole))])
    d = np.hypot(*(P[r] - P[m]).T)
    right = P[r, 0] >= P[m, 0] - 1e-9
    order = list(np.argsort(np.where(right, d, d + 1e12)))
    for j in order:
        v = ring[j]
        ok = (edges[:, 0] != v) & (edges[:, 1] != v) & (edges[:, 0] != m) & (edges[:, 1] != m)
        if not _seg_hits(P[m], P[v], P[edges[ok, 0]], P[edges[ok, 1]]).any():
            return ring[:j + 1] + hole + [m] + ring[j:]
    return ring[:order[0] + 1] + hole + [m] + ring[order[0]:]


def triangulate(outer, holes=()):
    """Ear-clip a CCW outline with CW holes; returns int (k, 3) into np.vstack([outer, *holes])."""
    P = np.vstack([outer, *holes]) if len(holes) else np.asarray(outer)
    ring, start = list(range(len(outer))), len(outer)
    hl = []
    for h in holes:
        hl.append(list(range(start, start + len(h)))); start += len(h)
    for h in sorted(hl, key=lambda h: -P[h, 0].max()):
        ring = _bridge(P, ring, h)
    eps = 1e-12 * max(1.0, float(np.ptp(P, axis=0).max()) ** 2)
    tris, i, stall = [], 0, 0
    while len(ring) > 3:
        n = len(ring)
        i %= n
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        cr = _cross(P[a], P[b], P[c])
        level = stall // n           # 0: strict ears, 1: also flat ones, 2+: clip whatever is next
        ear = cr > eps if level == 0 else cr > -eps or level > 1
        if ear and level < 2:
            idx = np.array(ring)
            idx = idx[(idx != a) & (idx != b) & (idx != c)]
            Q = P[idx]
            # a point on the triangle's edge blocks the ear (it may be a reflex vertex the diagonal runs through),
            # unless it is a bridge copy of a, b or c
            inside = (_cross(P[a], P[b], Q) >= -eps) & (_cross(P[b], P[c], Q) >= -eps) & (_cross(P[c], P[a], Q) >= -eps)
            same = np.all(np.isclose(Q[:, None], P[[a, b, c]][None]), axis=2).any(1)
            ear = not (inside & ~same).any()
        if ear:
            if abs(cr) > eps or level > 0:
                tris.append((a, b, c))
            del ring[i]
            i -= 1; stall = 0
        else:
            i += 1; stall += 1
    if len(ring) == 3 and abs(_cross(*P[ring])) > eps:
        tris.append(tuple(ring))
    return np.array(tris, int).reshape(-1, 3)


def _bevel_vec(p):
    """three.js getBevelVec for every vertex of a ring, shifted to the right of the walk (outside the solid when the
    outline is CCW and holes are CW): the miter point one unit off both edges, clamped to length sqrt(2)."""
    pt, nx, pv = p, np.roll(p, -1, 0), np.roll(p, 1, 0)
    a, b = pt - nx, pv - pt                     # three.js v_prev, v_next on the reversed walk
    col = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    la, lb = np.hypot(*a.T), np.hypot(*b.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        sa = nx + np.c_[-a[:, 1], a[:, 0]] / la[:, None]
        sb = pv + np.c_[-b[:, 1], b[:, 0]] / lb[:, None]
        sf = ((sb[:, 0] - sa[:, 0]) * b[:, 1] - (sb[:, 1] - sa[:, 1]) * b[:, 0]) / col
        v = sa + a * sf[:, None] - pt
        l2 = (v ** 2).sum(1)
        v = np.where((l2 <= 2)[:, None], v, v / np.sqrt(l2 / 2)[:, None])
        same = np.where(np.abs(a[:, 0]) > 1e-12, np.sign(a[:, 0]) == np.sign(b[:, 0]), np.sign(a[:, 1]) == np.sign(b[:, 1]))
        vc = np.where(same[:, None], np.c_[-a[:, 1], a[:, 0]] / la[:, None], a / np.sqrt(la ** 2 / 2)[:, None])
    v = np.where((np.abs(col) > 1e-16)[:, None], v, vc)
    return np.nan_to_num(v)


def extrude(s):
    outer = _clean(_pts(s.get('outline'), 'outline'))
    holes = [_clean(_pts(h, 'hole')) for h in (s.get('holes') or [])]
    if len(outer) < 3 or any(len(h) < 3 for h in holes):
        raise ValueError('outline or hole collapses to < 3 points')
    D = float(s.get('depth', 0))
    if not D > 0:
        raise ValueError(f'depth must be > 0, got {s.get("depth")!r}')
    bev = min(max(float(s.get('bevel') or 0), 0.0), D / 2)
    if _area(outer) < 0:
        outer = outer[::-1]
    holes = [h[::-1] if _area(h) > 0 else h for h in holes]
    rings = [outer] + holes
    tri = triangulate(outer, holes)
    P = np.vstack(rings)
    vec = np.vstack([_bevel_vec(r) for r in rings])
    if bev > 0:   # (z, offset) per layer, back cap to front cap, as ExtrudeGeometry orders them
        ts = [b / BEVEL_SEGMENTS for b in range(BEVEL_SEGMENTS)]
        back = [(-bev * math.cos(t * math.pi / 2), bev * math.sin(t * math.pi / 2) - bev) for t in ts]
        layers = back + [(0.0, 0.0), (D - 2 * bev, 0.0)] + [(D - 2 * bev - z, o) for z, o in back[::-1]]
        layers = [(z + bev - D / 2, o) for z, o in layers]
    else:
        layers = [(-D / 2, 0.0), (D / 2, 0.0)]
    n = len(P)
    V = np.vstack([np.c_[P + vec * o, np.full(n, z)] for z, o in layers])
    F = [tri[:, ::-1], tri + n * (len(layers) - 1)]
    start = 0
    for r in rings:   # side walls between consecutive layers, edge i -> i+1 of each ring
        i = np.arange(len(r)) + start
        j = np.roll(i, -1)
        for k in range(len(layers) - 1):
            a, b, c, d = i + k * n, j + k * n, j + (k + 1) * n, i + (k + 1) * n
            F += [np.c_[a, b, c], np.c_[a, c, d]]
        start += len(r)
    return V, np.vstack(F)


def lathe(s):
    prof = _pts(s.get('profile'), 'profile', 2)
    if (prof[:, 0] < -1e-9).any():
        raise ValueError('profile radius < 0')
    sec = s.get('section') or 'round'
    if sec not in SEGMENTS:
        raise ValueError(f'unknown section {sec!r}')
    seg = SEGMENTS[sec]
    phi0 = 0.0 if sec == 'round' else math.pi / seg   # three.js phiStart: flats face Z; square / octagon flats face X too, hexagon corners sit on X
    k = 1 / math.cos(math.pi / seg) if sec != 'round' else 1.0
    phi = phi0 + 2 * math.pi * np.arange(seg) / seg
    r = np.maximum(prof[:, 0], 0) * k
    V = np.stack([r[:, None] * np.sin(phi)[None], np.repeat(prof[:, 1:], seg, 1), r[:, None] * np.cos(phi)[None]], -1).reshape(-1, 3)
    st = s.get('stretch') or [1, 1]
    if len(st) != 2 or not all(isinstance(v, (int, float)) and v > 0 for v in st):
        raise ValueError(f'bad stretch {st!r}')
    V[:, 0] *= st[0]; V[:, 2] *= st[1]
    rot = s.get('rotate')
    if isinstance(rot, (int, float)):   # a list here is the part's Euler rotate, applied in parts()
        V = V @ _rot([0, float(rot), 0]).T
    j, i = np.meshgrid(np.arange(len(prof) - 1), np.arange(seg), indexing='ij')
    a, b = (j * seg + i).ravel(), (j * seg + (i + 1) % seg).ravel()
    c, d = b + seg, a + seg
    return V, np.vstack([np.c_[a, b, c], np.c_[a, c, d]])


def _rot(deg):
    x, y, z = np.radians(np.asarray(deg, float))
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _vec3(v, name):
    if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(a, (int, float)) and math.isfinite(a) for a in v)):
        raise ValueError(f'{name} must be [x, y, z], got {v!r}')
    return [float(a) for a in v]


def parts(s, mat=None, depth=0):
    """[(V, F, mat)] for every solid in a SHAPE, transforms applied."""
    if not isinstance(s, dict):
        raise ValueError(f'SHAPE must be an object, got {type(s).__name__}')
    if depth > 16:
        raise ValueError('groups nested too deep')
    mat = s.get('mat', mat)
    kind = s.get('kind')
    if kind == 'group':
        ps = s.get('parts')
        if not isinstance(ps, list) or not ps:
            raise ValueError('group needs a non-empty parts list')
        out = [q for p in ps for q in parts(p, mat, depth + 1)]
    elif kind == 'extrude':
        out = [(*extrude(s), mat)]
    elif kind == 'lathe':
        out = [(*lathe(s), mat)]
    else:
        raise ValueError(f'unknown kind {kind!r}')
    # a part's own rotate/offset; a lathe's scalar "rotate" is its turn about Y and was applied above
    rot = s.get('rotate')
    R = np.eye(3) if rot is None or (kind == 'lathe' and isinstance(rot, (int, float))) else _rot(_vec3(rot, 'rotate'))
    off = np.asarray(_vec3(s['offset'], 'offset')) if s.get('offset') is not None else np.zeros(3)
    return [(V @ R.T + off, F, m) for V, F, m in out]


def mesh(s):
    ps = parts(s)
    V = np.vstack([p[0] for p in ps])
    starts = np.cumsum([0] + [len(p[0]) for p in ps[:-1]])
    return V, np.vstack([p[1] + o for p, o in zip(ps, starts)])


def mesh_bounds(s):
    """(lo, hi) xyz in mm, of a SHAPE or of a (V, F) pair."""
    V = s[0] if isinstance(s, tuple) else mesh(s)[0]
    return V.min(0), V.max(0)


if __name__ == '__main__':
    bar = {'kind': 'extrude', 'depth': 10, 'bevel': 5,
           'outline': [[-60, 0], [-45, 0], [-45, 25], [45, 25], [45, 0], [60, 0], [60, 35], [-60, 35]]}
    knob = {'kind': 'lathe', 'section': 'square', 'profile': [[6, 0], [6, 8], [12, 10], [12, 28], [0, 30]]}
    ring = {'kind': 'group', 'parts': [
        {'kind': 'lathe', 'profile': [[10, 0], [10, 3], [0, 3]]},
        {'kind': 'extrude', 'depth': 4, 'bevel': 2, 'outline': [[-15, 0], [15, 0], [15, 30], [-15, 30]],
         'holes': [[[-8, 8], [8, 8], [8, 22], [-8, 22]]], 'offset': [0, 3, 0], 'rotate': [0, 90, 0]}]}
    for name, s in [('bar', bar), ('knob', knob), ('ring', ring)]:
        V, F = mesh(s)
        lo, hi = mesh_bounds((V, F))
        print(f'{name:5s} {len(V):5d} v {len(F):5d} f  lo {np.round(lo, 2)}  hi {np.round(hi, 2)}')
