"""Manhattan-constrained line / point triangulation in the VP-aligned (unscaled) frame.

Axis-parallel 3D line along axis a: free coords = the other two axes.
Observations: per view, a list of full-res pixel points lying on the edge's image.
Residual: perpendicular distance (in px, using focal) of each observed point to the projected line,
computed in undistorted normalized coords.
"""
import numpy as np
from scipy.optimize import least_squares
from geom import Model


class MFrame:
    def __init__(self, model_path, R):
        self.m = Model(model_path)
        self.R = R  # world(model) -> aligned (rows = axes)
        self.cam = {}
        for k, d in self.m.ims.items():
            Rw = d["R"] @ R.T  # aligned -> cam rotation
            C = R @ self.m.center(k)
            f = d["cam"].params[0] / d["s"]  # focal in full-res px
            self.cam[k] = (Rw, C, f)

    def norm(self, k, uv):
        return self.m.normalized(k, np.asarray(uv, float).reshape(-1, 2) + 0.5)

    def proj_norm(self, k, X):
        Rw, C, f = self.cam[k]
        Xc = (np.asarray(X).reshape(-1, 3) - C) @ Rw.T
        return Xc[:, :2] / Xc[:, 2:3], Xc[:, 2]

    def line(self, axis, obs, sigma_px=2.0, x0=None):
        free = [i for i in range(3) if i != axis]
        pts = {k: self.norm(k, v) for k, v in obs.items()}

        def build(p, t):
            X = np.zeros((len(t), 3)); X[:, axis] = t; X[:, free[0]] = p[0]; X[:, free[1]] = p[1]
            return X

        # parameter range along axis for projection: use a range from initial guess
        def res(p):
            r = []
            for k, n in pts.items():
                f = self.cam[k][2]
                X = build(p, np.array([-50.0, 50.0]))
                # clip segment to be in front of camera: use points near camera depth range
                Rw, C, _ = self.cam[k]
                # choose two points on the 3D line in front of camera
                d = np.zeros(3); d[axis] = 1
                P0 = build(p, np.array([C[axis]]))[0]
                a = P0 + d * 1.0; b = P0 - d * 1.0
                A = (a - C) @ Rw.T; B = (b - C) @ Rw.T
                # homogeneous image line through projections
                l = np.cross(A, B)  # line in normalized homogeneous coords
                l = l / np.linalg.norm(l[:2])
                for q in n:
                    r.append((l[0] * q[0] + l[1] * q[1] + l[2]) * f)
            return np.array(r)

        if x0 is None:
            A = []; bb = []
            for k, n in pts.items():
                if len(n) < 2:
                    continue
                Rw, C, f = self.cam[k]
                q1 = np.r_[n[0], 1.0]; q2 = np.r_[n[-1], 1.0]
                nw = Rw.T @ np.cross(q1, q2)
                nf = nw[free]
                nf = nf / np.linalg.norm(nf)
                A.append(nf); bb.append(nf @ C[free])
            x0 = np.linalg.lstsq(np.array(A), np.array(bb), rcond=None)[0]
        sol = least_squares(res, x0, method="lm")
        J = sol.jac
        r = sol.fun
        dof = max(len(r) - 2, 1)
        try:
            C = np.linalg.inv(J.T @ J) * sigma_px ** 2
        except np.linalg.LinAlgError:
            C = np.full((2, 2), np.nan)
        out = dict(axis="xyz"[axis], free=["xyz"[i] for i in free], p=sol.x, cov=C, sig=np.sqrt(np.diag(C)),
                   rms=float(np.sqrt(np.mean(r ** 2))), maxres=float(np.max(np.abs(r))), nres=len(r))
        return out

    def _closest_on(self, k2, pts2, k1, uv1):
        """pick the point in view k2's list closest to the epipolar line of uv1 (cheap: just return first)."""
        # use epipolar geometry to find the point on polyline pts2 nearest epipolar line
        c1, v1 = self.m.ray(k1, np.array(uv1, float) + 0.5)
        best = None
        P = np.asarray(pts2, float)
        # sample along polyline
        cand = []
        for i in range(len(P) - 1 if len(P) > 1 else 1):
            a = P[i]; b = P[min(i + 1, len(P) - 1)]
            for t in np.linspace(0, 1, 50):
                cand.append(a + t * (b - a))
        cand = np.array(cand)
        dmin = None
        for q in cand:
            c2, v2 = self.m.ray(k2, q + 0.5)
            w = np.cross(v1, v2)
            if np.linalg.norm(w) < 1e-9:
                continue
            dist = abs((c2 - c1) @ w) / np.linalg.norm(w)
            if dmin is None or dist < dmin:
                dmin, best = dist, q
        return best

    def point(self, obs, sigma_px=2.0):
        X, err = self.m.triangulate(obs)
        _, C = self.m.tri_cov(obs, sigma_px)
        return self.R @ X, self.R @ C @ self.R.T, err

    def ray_plane(self, k, uv, axis, value):
        Rw, C, f = self.cam[k]
        n = self.norm(k, uv)[0]
        v = Rw.T @ np.array([n[0], n[1], 1.0])
        t = (value - C[axis]) / v[axis]
        return C + t * v
