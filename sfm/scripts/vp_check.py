"""Estimate vanishing directions from line segments in each image and express in aligned frame.
Checks the up-vector & wall directions derived from point-cloud planes."""
import sys, numpy as np, cv2
sys.path.insert(0, "/home/claude/kitchen/artifacts/sfm/scripts")
from geom import Model
path = sys.argv[1]
m = Model(path)
R = np.load(path + "/../R_align_tmp.npy")
rng = np.random.default_rng(0)


def segs(key):
    im = cv2.imread(f"/home/claude/kitchen/images/{m.ims[key]['name'].split('/')[-1]}", cv2.IMREAD_GRAYSCALE)
    sc = 0.5
    ims = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    lsd = cv2.createLineSegmentDetector()
    L = lsd.detect(ims)[0].reshape(-1, 4) / sc
    ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    return L[ln > 80]


def seg_planes(key, L):
    """each segment -> interpretation-plane normal in WORLD aligned frame"""
    a = m.normalized(key, L[:, :2] + 0.5)
    b = m.normalized(key, L[:, 2:] + 0.5)
    A = np.c_[a, np.ones(len(a))]; B = np.c_[b, np.ones(len(b))]
    n = np.cross(A, B)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    Rw = m.ims[key]["R"].T  # cam->world
    nw = (n @ Rw.T) @ R.T
    w = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    return nw, w


allN, allW = [], []
for k in ["02", "03", "04", "05", "01"]:
    L = segs(k)
    n, w = seg_planes(k, L)
    allN.append(n); allW.append(w)
    print(k, "segments", len(L))
N = np.vstack(allN); Wt = np.concatenate(allW)
# for each axis direction d (aligned frame), segments parallel to d have n.d ~ 0.
for ax, nm in enumerate("xyz"):
    d0 = np.eye(3)[ax]
    sel = np.abs(N @ d0) < np.sin(np.radians(3))
    # refine: d = smallest singular vector of weighted N[sel]
    d = d0
    for it in range(5):
        sel = np.abs(N @ d) < np.sin(np.radians(2.0 if it else 3.0))
        M = N[sel] * np.sqrt(Wt[sel])[:, None]
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        d = Vt[-1] * np.sign(Vt[-1][ax])
    ang = np.degrees(np.arccos(abs(d[ax])))
    print(f"axis {nm}: VP-direction (aligned) = {np.round(d,4)}  deviation from axis = {ang:.2f} deg  (n segs {sel.sum()})")

# build refined rotation from VP directions
D = []
for ax in range(3):
    d0 = np.eye(3)[ax]; d = d0
    for it in range(5):
        sel = np.abs(N @ d) < np.sin(np.radians(2.0 if it else 3.0))
        M = N[sel] * np.sqrt(Wt[sel])[:, None]
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        d = Vt[-1] * np.sign(Vt[-1][ax])
    D.append(d)
D = np.array(D)
U, S, Vt = np.linalg.svd(D)
Rvp = U @ Vt   # rows ~ VP directions in old aligned frame
print("Rvp\n", np.round(Rvp, 4))
Rnew = Rvp @ R
np.save(path + "/../R_align_vp.npy", Rnew)
