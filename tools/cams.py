"""Camera helpers on Agent A's cameras.json (inch frame, full-res pixels)."""
import json, numpy as np
C = json.load(open('/home/claude/kitchen/artifacts/sfm/cameras.json'))['images']
def key(n):
    for k in C:
        if k.startswith(n[:2]) or n in k: return k
    raise KeyError(n)
def project(n, X):
    c = C[key(n)]; X = np.atleast_2d(np.asarray(X, float))
    R = np.array(c['R_world_to_cam']); t = np.array(c['t_world_to_cam'])
    Xc = X @ R.T + t; z = Xc[:, 2]
    x, y = Xc[:, 0]/z, Xc[:, 1]/z; r2 = x*x+y*y; d = 1+c['distortion']['k1']*r2
    K = np.array(c['K']); u = K[0,0]*x*d + K[0,2]; v = K[1,1]*y*d + K[1,2]
    return np.stack([u, v], 1), z
def ray(n, uv):
    c = C[key(n)]; K = np.array(c['K']); k1 = c['distortion']['k1']
    x = (uv[0]-K[0,2])/K[0,0]; y = (uv[1]-K[1,2])/K[1,1]
    for _ in range(10):  # undistort
        r2 = x*x+y*y; d = 1+k1*r2; x, y = (uv[0]-K[0,2])/K[0,0]/d, (uv[1]-K[1,2])/K[1,1]/d
    R = np.array(c['R_world_to_cam']); o = np.array(c['camera_center'])
    dvec = R.T @ np.array([x, y, 1.0]); return o, dvec/np.linalg.norm(dvec)
def triangulate(obs):
    """obs: list of (image_name, (u,v)) full-res. Least-squares ray intersection."""
    A = np.zeros((3,3)); b = np.zeros(3)
    for n, uv in obs:
        o, d = ray(n, uv); P = np.eye(3)-np.outer(d, d); A += P; b += P @ o
    X = np.linalg.solve(A, b)
    err = [float(np.linalg.norm(project(n, X)[0][0]-np.array(uv))) for n, uv in obs]
    return X, err
