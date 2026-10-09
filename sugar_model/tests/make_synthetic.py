"""Synthetic "SuGaR-like" kitchen mesh generator (ground truth for cad_clean tests).

Builds an L-shaped kitchen as an exact boolean union of boxes and bar-pull
handles (manifold3d), tessellates it densely like a Poisson/SuGaR mesh, then
corrupts it the way a real reconstruction is corrupted:

* edge rounding      - 1 Laplacian iteration (~3 mm soft creases, like Poisson)
* low-freq ripples   - ~3 mm amplitude, 0.5-0.9 m wavelength, along normals
* Gaussian noise     - sigma = 1 mm, isotropic, per vertex
* floaters           - blobs/slivers in free space + a few 1-3 triangle islands
* detached hardware  - one bar pull floating 2 mm off a door (must be KEPT)
* arbitrary frame    - random rotation, translation and a non-metric scale
                       (COLMAP/SuGaR units are arbitrary)

Ground-truth units: millimetres, Z-up, floor top at z=0, back wall face y=0,
left wall face x=0.  Everything the test needs is returned in a dict.
"""
from __future__ import annotations

import numpy as np
import trimesh
import manifold3d as m3d

IN = 25.4  # mm per inch

# (name, min corner, max corner) in mm.  Faces of these boxes define the
# ground-truth planes.
BOXES = [
    ("floor_slab",        (-100, -100, -50), (4000, 3500, 0)),
    ("back_wall",         (-100, -100, 0), (4000, 0, 2400)),
    ("left_wall",         (-100, -100, 0), (0, 3500, 2400)),
    # back run of base cabinets: plinth (toe kick) + carcass + door slab
    ("back_plinth",       (0, 0, 0), (3000, 584 - 76, 4 * IN)),
    ("back_carcass",      (0, 0, 4 * IN), (3000, 584, 34.5 * IN)),
    ("back_doors",        (620, 584, 4.5 * IN), (2990, 603, 34.25 * IN)),
    ("back_counter",      (0, 0, 34.5 * IN), (3025, 635, 36 * IN)),
    # left run (L-shape)
    ("left_plinth",       (0, 0, 0), (584 - 76, 2400, 4 * IN)),
    ("left_carcass",      (0, 0, 4 * IN), (584, 2400, 34.5 * IN)),
    ("left_doors",        (584, 603, 4.5 * IN), (603, 2390, 34.25 * IN)),
    ("left_counter",      (0, 0, 34.5 * IN), (635, 2425, 36 * IN)),
    # upper cabinets on the back wall
    ("uppers",            (0, 0, 54 * IN), (3000, 12 * IN, 84 * IN)),
    ("upper_doors",       (10, 12 * IN, 54.5 * IN), (2990, 12 * IN + 19, 83.5 * IN)),
]


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3d.Manifold.cube(list(hi - lo)).translate(list(lo))


def _cyl(p0, p1, r, segs=12):
    """Cylinder between two points (manifold3d), used for bar pulls."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    c = m3d.Manifold.cylinder(L, r, r, segs)          # along +Z from origin
    z = (p1 - p0) / L
    a = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(a) < 1e-9:
        R = np.eye(3) if z[2] > 0 else np.diag([1, -1, -1.0])
    else:
        R = trimesh.transformations.rotation_matrix(
            np.arccos(np.clip(z[2], -1, 1)), a)[:3, :3]
    M = np.hstack([R, p0[:, None]])
    return c.transform(M.tolist())


def _bar_pull(base, along, out, length=128.0, standoff=25.0):
    """Bar pull: one bar parallel to `along`, two posts going `out` of a door."""
    base, along, out = (np.asarray(v, float) for v in (base, along, out))
    a = base - along * length / 2
    b = base + along * length / 2
    parts = [_cyl(a + out * standoff, b + out * standoff, 6.0),
             _cyl(a - out * 1.0, a + out * standoff, 5.0),
             _cyl(b - out * 1.0, b + out * standoff, 5.0)]
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def truth_planes():
    """All box faces as (normal, offset) with n.x = d, outward normals."""
    planes = []
    for name, lo, hi in BOXES:
        for k in range(3):
            n = np.zeros(3); n[k] = 1.0
            planes.append((n.copy(), float(hi[k]), f"{name}+{'xyz'[k]}"))
            planes.append((-n, -float(lo[k]), f"{name}-{'xyz'[k]}"))
    return planes


def handles():
    hs = []
    # base back run: horizontal pulls near top of doors, every 600 mm
    for x in np.arange(300, 2900, 600):
        hs.append(_bar_pull((x, 603, 31 * IN), (1, 0, 0), (0, 1, 0)))
    # base left run
    for y in np.arange(900, 2300, 600):
        hs.append(_bar_pull((603, y, 31 * IN), (0, 1, 0), (1, 0, 0)))
    # uppers: vertical pulls near bottom
    for x in np.arange(450, 2900, 600):
        hs.append(_bar_pull((x, 12 * IN + 19, 58 * IN), (0, 0, 1), (0, 1, 0)))
    return hs


def make_synthetic(seed: int = 7, edge_len: float = 20.0, noise_sigma: float = 1.0,
                   ripple_amp: float = 3.0, rot_deg: float = 8.0,
                   raw_scale: float = 1.0 / 1234.5):
    rng = np.random.default_rng(seed)

    # ---- exact CAD truth ---------------------------------------------------
    solids = [_box(lo, hi) for _, lo, hi in BOXES] + handles()
    kitchen = m3d.Manifold.batch_boolean(solids, m3d.OpType.Add)
    kitchen = kitchen.refine_to_length(edge_len)
    mm = kitchen.to_mesh()
    V = np.asarray(mm.vert_properties, float)[:, :3]
    F = np.asarray(mm.tri_verts, np.int64)
    main = trimesh.Trimesh(V, F, process=True)
    assert main.is_watertight

    # ---- SuGaR-like corruption -------------------------------------------
    trimesh.smoothing.filter_laplacian(main, lamb=0.5, iterations=1,
                                       volume_constraint=False)
    Vn = main.vertex_normals
    P = main.vertices
    ripple = np.zeros(len(P))
    for _ in range(3):
        k = rng.normal(size=3); k /= np.linalg.norm(k)
        lam = rng.uniform(500, 900)
        ripple += np.sin(2 * np.pi * (P @ k) / lam + rng.uniform(0, 2 * np.pi))
    ripple *= ripple_amp / 3.0 * 1.0
    P = P + Vn * ripple[:, None] + rng.normal(scale=noise_sigma, size=P.shape)
    main = trimesh.Trimesh(P, main.faces, process=False)
    n_main_faces = len(main.faces)

    # detached bar pull, 2 mm off the left-run door at y=600 (kept as hardware)
    det = _bar_pull((603 + 2.0, 600, 31 * IN), (0, 1, 0), (1, 0, 0)).refine_to_length(edge_len)
    dm = det.to_mesh()
    det_mesh = trimesh.Trimesh(np.asarray(dm.vert_properties)[:, :3], dm.tri_verts)
    det_mesh.vertices += rng.normal(scale=noise_sigma * 0.3, size=det_mesh.vertices.shape)
    detached_center = det_mesh.vertices.mean(0)

    # floaters in free space (room interior, away from all surfaces)
    floaters, floater_info = [], []
    centers = [(1800, 1800, 1200), (2500, 1500, 600), (1200, 2600, 1900),
               (3200, 2200, 300), (2000, 1100, 2000), (900, 1600, 1500),
               (3500, 900, 1200), (1500, 3000, 800), (2800, 2800, 1600),
               (2200, 2000, 250)]
    for i, c in enumerate(centers):
        r = rng.uniform(15, 60)
        if i % 3 == 2:   # thin sliver floater
            f = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            f.vertices *= np.array([r * 2.5, r * 0.6, r * 0.08])
        else:
            f = trimesh.creation.icosphere(subdivisions=2, radius=r)
        f.vertices += np.asarray(c, float) + rng.normal(scale=0.5, size=f.vertices.shape)
        floaters.append(f)
        floater_info.append({"center": list(map(float, c)), "radius": float(r * 2.5)})
    # tiny 1-3 triangle islands
    for c in [(1000, 1000, 1000), (3000, 3000, 2000), (2500, 600, 1800)]:
        t = rng.normal(scale=15, size=(3, 3)) + np.asarray(c, float)
        floaters.append(trimesh.Trimesh(t, [[0, 1, 2]], process=False))
        floater_info.append({"center": list(map(float, c)), "radius": 60.0})

    raw_mm = trimesh.util.concatenate([main, det_mesh] + floaters)

    # ---- arbitrary SfM frame: rotation + translation + non-metric scale ---
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    R = trimesh.transformations.rotation_matrix(np.radians(rot_deg), axis)[:3, :3]
    t = np.array([0.31, -0.22, 1.13])
    T = np.eye(4); T[:3, :3] = raw_scale * R; T[:3, 3] = t      # truth(mm) -> raw
    raw = raw_mm.copy(); raw.apply_transform(T)

    return {
        "raw": raw,
        "T_truth_to_raw": T,
        "truth_planes": truth_planes(),
        "floaters": floater_info,
        "detached_handle_center": detached_center,
        "n_main_faces": n_main_faces,
        "counter_height_mm": 36 * IN,
        "noise": {"sigma_mm": noise_sigma, "ripple_amp_mm": ripple_amp,
                  "rot_deg": rot_deg, "raw_scale": raw_scale, "edge_len_mm": edge_len},
    }


if __name__ == "__main__":
    import sys
    d = make_synthetic()
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_kitchen_raw.ply"
    d["raw"].export(out)
    print(f"wrote {out}: {len(d['raw'].faces)} faces")
