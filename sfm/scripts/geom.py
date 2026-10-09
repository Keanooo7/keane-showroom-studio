"""Geometry helpers: load model, project / triangulate full-res pixel picks."""
import numpy as np
import pycolmap
from scipy.optimize import least_squares

FULL_W, FULL_H = 3024, 4032


class Model:
    def __init__(self, path):
        self.rec = pycolmap.Reconstruction(path)
        self.ims = {}
        for im in self.rec.images.values():
            key = im.name.split("/")[-1][:2]
            cam = self.rec.cameras[im.camera_id]
            s = cam.width / FULL_W  # model px per full-res px
            T = im.cam_from_world().matrix()  # 3x4
            self.ims[key] = dict(im=im, cam=cam, s=s, R=T[:, :3], t=T[:, 3], name=im.name)

    def normalized(self, key, uv_full):
        """full-res continuous pixel coords (N,2) -> undistorted normalized coords (N,2)"""
        d = self.ims[key]
        uv = (np.asarray(uv_full, float).reshape(-1, 2)) * d["s"]
        return np.asarray(d["cam"].cam_from_img(uv))

    def project(self, key, X):
        """world points (N,3) -> full-res pixel coords (N,2)"""
        d = self.ims[key]
        X = np.asarray(X, float).reshape(-1, 3)
        Xc = X @ d["R"].T + d["t"]
        uv = np.asarray(d["cam"].img_from_cam(Xc))
        return uv / d["s"], Xc[:, 2]

    def center(self, key):
        d = self.ims[key]
        return -d["R"].T @ d["t"]

    def ray(self, key, uv_full):
        n = self.normalized(key, uv_full)[0]
        d = self.ims[key]
        v = d["R"].T @ np.array([n[0], n[1], 1.0])
        return self.center(key), v / np.linalg.norm(v)

    def triangulate(self, obs):
        """obs: dict key -> (u,v) full-res (pixel index; +0.5 applied here).
        Returns X (3,), per-view reprojection error (full-res px)."""
        keys = list(obs)
        A = []
        for k in keys:
            n = self.normalized(k, np.array(obs[k], float) + 0.5)[0]
            d = self.ims[k]
            P = np.hstack([d["R"], d["t"][:, None]])
            A.append(n[0] * P[2] - P[0])
            A.append(n[1] * P[2] - P[1])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1][:3] / Vt[-1][3]

        def res(X):
            r = []
            for k in keys:
                uv, _ = self.project(k, X)
                r.extend(uv[0] - (np.array(obs[k], float) + 0.5))
            return np.array(r)

        if len(keys) >= 2:
            X = least_squares(res, X, method="lm").x
        r = res(X).reshape(-1, 2)
        return X, {k: float(np.linalg.norm(r[i])) for i, k in enumerate(keys)}

    def tri_cov(self, obs, sigma_px=2.0):
        """First-order covariance of triangulated point given pixel noise sigma (full-res px)."""
        X, _ = self.triangulate(obs)
        J = []
        eps = 1e-6
        for k in obs:
            uv0, _ = self.project(k, X)
            Jk = np.zeros((2, 3))
            for j in range(3):
                dX = np.zeros(3); dX[j] = eps * max(1.0, np.linalg.norm(X))
                uv1, _ = self.project(k, X + dX)
                Jk[:, j] = (uv1[0] - uv0[0]) / dX[j]
            J.append(Jk)
        J = np.vstack(J)
        C = np.linalg.inv(J.T @ J) * sigma_px ** 2
        return X, C
