import sys, numpy as np, pycolmap, cv2
path = sys.argv[1]
out = sys.argv[2]
r = pycolmap.Reconstruction(path)
ids = np.array(list(r.points3D.keys()))
X = np.array([r.points3D[i].xyz for i in ids])
C = np.array([r.points3D[i].color for i in ids])
R = np.load(path + "/../R_align_tmp.npy")
Y = X @ R.T
lo = np.percentile(Y, 2, 0); hi = np.percentile(Y, 98, 0)
pad = (hi - lo) * 0.15
lo -= pad; hi += pad
cams = {im.name.split('/')[-1][:2]: R @ im.projection_center() for im in r.images.values()}


def view(a, b, flipb=True, size=900):
    img = np.full((size, size, 3), 255, np.uint8)
    s = (size - 40) / max(hi[a] - lo[a], hi[b] - lo[b])
    def px(p):
        u = int(20 + (p[a] - lo[a]) * s)
        v = int(20 + (hi[b] - p[b]) * s) if flipb else int(20 + (p[b] - lo[b]) * s)
        return u, v
    for p, c in zip(Y, C):
        u, v = px(p)
        if 0 <= u < size and 0 <= v < size:
            cv2.circle(img, (u, v), 1, (int(c[2]) // 2, int(c[1]) // 2, int(c[0]) // 2), -1)
    for k, c in cams.items():
        u, v = px(c)
        cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(img, k, (u + 5, v), 0, 0.5, (0, 0, 255), 1)
    # ticks every 1 unit
    for t in np.arange(np.ceil(lo[a]), hi[a], 1.0):
        u, _ = px(np.where(np.arange(3) == a, t, lo))
        cv2.line(img, (u, 0), (u, 10), (255, 0, 0), 1); cv2.putText(img, f"{t:.0f}", (u, 22), 0, 0.35, (255, 0, 0), 1)
    for t in np.arange(np.ceil(lo[b]), hi[b], 1.0):
        q = lo.copy(); q[b] = t
        _, v = px(q)
        cv2.line(img, (0, v), (10, v), (255, 0, 0), 1); cv2.putText(img, f"{t:.0f}", (12, v), 0, 0.35, (255, 0, 0), 1)
    cv2.putText(img, f"horiz={'xyz'[a]} vert={'xyz'[b]}", (300, size - 10), 0, 0.6, (0, 0, 0), 1)
    return img

cv2.imwrite(out + "_top.png", view(0, 1))
cv2.imwrite(out + "_W1front.png", view(0, 2))
cv2.imwrite(out + "_W2front.png", view(1, 2))
