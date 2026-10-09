import sys, numpy as np, pycolmap
path = sys.argv[1]
r = pycolmap.Reconstruction(path)
ids = np.array(list(r.points3D.keys()))
X = np.array([r.points3D[i].xyz for i in ids])
err = np.array([r.points3D[i].error for i in ids])
tl = np.array([r.points3D[i].track.length() for i in ids])
print("npts", len(X), "median err", np.median(err))
med = np.median(X, 0)
scale = np.median(np.linalg.norm(X - med, axis=1))
rng = np.random.default_rng(0)


def ransac_plane(P, thr, iters=4000):
    best = None
    n = len(P)
    for _ in range(iters):
        s = P[rng.choice(n, 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(nv) < 1e-9:
            continue
        nv /= np.linalg.norm(nv)
        d = -nv @ s[0]
        inl = np.abs(P @ nv + d) < thr
        if best is None or inl.sum() > best[2].sum():
            best = (nv, d, inl)
    nv, d, inl = best
    # refine
    Q = P[inl]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    nv = Vt[2]
    d = -nv @ c
    inl = np.abs(P @ nv + d) < thr
    return nv, d, inl


rem = np.ones(len(X), bool)
planes = []
thr = 0.01 * scale
for k in range(8):
    idx = np.where(rem)[0]
    if len(idx) < 50:
        break
    nv, d, inl = ransac_plane(X[idx], thr)
    planes.append((nv, d, idx[inl]))
    rem[idx[inl]] = False
    print(f"plane {k}: n={np.round(nv,3)} d={d:.3f} inliers={inl.sum()}")
N = np.array([p[0] for p in planes])
A = np.degrees(np.arccos(np.clip(np.abs(N @ N.T), 0, 1)))
np.set_printoptions(precision=1, suppress=True, linewidth=150)
print(A)
np.save(path + "/../planes_tmp.npy", np.array([np.r_[p[0], p[1]] for p in planes]))
