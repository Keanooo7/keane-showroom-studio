"""Probe wall/floor/counter positions in the rotation-aligned (unscaled) frame."""
import sys, numpy as np, pycolmap
path = sys.argv[1]
R = np.load(path + "/../R_align_tmp.npy")
r = pycolmap.Reconstruction(path)
ids = np.array(list(r.points3D.keys()))
X = np.array([r.points3D[i].xyz for i in ids]) @ R.T
C = np.array([r.points3D[i].color for i in ids]).astype(float)
TL = np.array([r.points3D[i].track.length() for i in ids])


def band(sel, ax, lo, hi, nb=40):
    v = X[sel, ax]
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    return "\n".join(f"   {(e[i]+e[i+1])/2:7.3f} {h[i]:5d} " + "#" * min(h[i] // 3, 60) for i in range(nb))

# W1 backsplash: points with z between counter (-2.8) and -1.2, x in (-9.8,-3)
sel = (X[:, 2] > -2.7) & (X[:, 2] < -1.3) & (X[:, 0] > -9.7) & (X[:, 0] < -2.5)
print("y-hist for backsplash-height points (W1 region):")
print(band(sel & (X[:, 1] < -5), 1, -6.6, -5.6))
sel2 = (X[:, 2] > -2.7) & (X[:, 2] < -1.3) & (X[:, 1] > -5.8) & (X[:, 1] < 0)
print("x-hist for backsplash-height points (W2 region):")
print(band(sel2 & (X[:, 0] < -9.3), 0, -10.6, -9.4))
print("z-hist all points near walls:")
print(band(np.ones(len(X), bool), 2, -7, 5.5, 50))
