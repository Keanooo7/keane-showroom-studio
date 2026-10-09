import sys, numpy as np, pycolmap
from scipy.optimize import least_squares
path = sys.argv[1]
r = pycolmap.Reconstruction(path)
ids = np.array(list(r.points3D.keys()))
X = np.array([r.points3D[i].xyz for i in ids])
C = np.array([r.points3D[i].color for i in ids])
err = np.array([r.points3D[i].error for i in ids])
P = np.load(path + "/../planes_tmp.npy")
N = P[:, :3]
up = N[0] / np.linalg.norm(N[0])
# families
fa, fb, fh = [], [], []
for i, n in enumerate(N):
    a = np.degrees(np.arccos(abs(n @ up)))
    if a < 5:
        fh.append(i)
for i, n in enumerate(N):
    if i in fh:
        continue
    # wall-ish
    if abs(n @ up) < 0.1:
        fa.append(i)
print("horiz", fh, "walls", fa)
walls = N[fa] * np.sign(N[fa][:, [0]])  # orient +x cam
# cluster into 2 groups by z sign
gA = [w for w in walls if w[2] < 0]  # (0.77,..,-0.63)
gB = [w for w in walls if w[2] > 0]  # (0.64,..,0.76) -> flip later
print("gA", np.round(gA, 3))
print("gB", np.round(gB, 3))


def robust_dirs(planes_idx_list):
    pass

# Use inlier-weighted refit: orthogonal Procrustes of the 3 directions
nA = np.mean(gA, 0); nA /= np.linalg.norm(nA)       # W1 normal (into room) -> +y
nB = -np.mean(gB, 0); nB /= np.linalg.norm(nB)      # W2 normal (into room) -> +x
M = np.stack([nB, nA, up])  # desired rows x,y,z
U, S, Vt = np.linalg.svd(M)
R = U @ Vt
print("R det", np.linalg.det(R))
print("residual angles deg:", [np.degrees(np.arccos(np.clip(R[i] @ M[i], -1, 1))) for i in range(3)])
Y = X @ R.T
np.save(path + "/../R_align_tmp.npy", R)
for ax, nm in enumerate("xyz"):
    h, e = np.histogram(Y[:, ax], bins=120)
    top = np.argsort(h)[::-1][:8]
    print(nm, "range", np.percentile(Y[:, ax], [1, 99]), "peaks", [(round((e[t] + e[t + 1]) / 2, 3), h[t]) for t in sorted(top)])
