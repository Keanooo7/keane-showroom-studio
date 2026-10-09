"""Verification: z-buffer render the world model into each SfM camera and overlay part boundaries on the photo.
Also scores alignment: distance from model silhouette/part boundaries to the nearest photo edge (Canny).
Usage: python3 tools/overlay.py [scale=0.25]
"""
import sys, json, numpy as np, cv2, trimesh
sys.path.insert(0, '/home/claude/kitchen/tools'); import cams
S = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
OUT = '/home/claude/kitchen/artifacts/verify'; import os; os.makedirs(OUT, exist_ok=True)
LAYER_COL = {'cabinets': (0, 220, 255), 'drawers': (0, 255, 0), 'countertops': (255, 0, 255), 'hardware': (0, 0, 255),
             'fixtures': (255, 255, 0), 'backsplash': (255, 128, 0)}

def load_obj_parts(path):
    """Minimal OBJ reader keeping each `o` block as its own part (trimesh renames split objects)."""
    V, objs, cur = [], {}, None
    for line in open(path):
        if line.startswith('v '): V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('o '): cur = line[2:].strip(); objs[cur] = []
        elif line.startswith('f ') and cur:
            objs[cur].append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    V = np.array(V); out = []
    for name, F in objs.items():
        F = np.array(F); used, inv = np.unique(F, return_inverse=True)
        out.append((name, V[used], inv.reshape(F.shape)))
    return out
parts = []
for name, Vm, F in load_obj_parts('/home/claude/kitchen/artifacts/export/kitchen_world.obj'):
    layer = name.split('/')[0]
    if layer == 'reference' or layer not in LAYER_COL: continue
    parts.append((name, layer, Vm / 25.4, F))
print(len(parts), 'parts')

def cam_space(n, X):
    c = cams.C[cams.key(n)]; R = np.array(c['R_world_to_cam']); t = np.array(c['t_world_to_cam'])
    return X @ R.T + t, c

def to_px(Xc, c):
    z = Xc[:, 2]; x, y = Xc[:, 0] / z, Xc[:, 1] / z; d = 1 + c['distortion']['k1'] * (x * x + y * y)
    K = np.array(c['K']); return np.stack([(K[0, 0] * x * d + K[0, 2]) * S, (K[1, 1] * y * d + K[1, 2]) * S], 1)

def clip_near(tri, near=1.0):
    out = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        ia, ib = a[2] >= near, b[2] >= near
        if ia: out.append(a)
        if ia != ib:
            t = (near - a[2]) / (b[2] - a[2]); out.append(a + t * (b - a))
    return [np.array([out[0], out[k], out[k + 1]]) for k in range(1, len(out) - 1)] if len(out) >= 3 else []

def render(n):
    c = cams.C[cams.key(n)]; W, H = int(c['width'] * S), int(c['height'] * S)
    zb = np.full((H, W), np.inf); idb = np.full((H, W), -1, int)
    for pi, (name, layer, V, F) in enumerate(parts):
        Xc, _ = cam_space(n, V)
        for f in F:
            tri = Xc[f]
            if (tri[:, 2] < 1.0).all(): continue
            for t3 in (clip_near(tri) if (tri[:, 2] < 1.0).any() else [tri]):
                p = to_px(t3, c); iz = 1.0 / t3[:, 2]
                x0, x1 = int(max(0, np.floor(p[:, 0].min()))), int(min(W - 1, np.ceil(p[:, 0].max())))
                y0, y1 = int(max(0, np.floor(p[:, 1].min()))), int(min(H - 1, np.ceil(p[:, 1].max())))
                if x0 > x1 or y0 > y1: continue
                xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                (ax, ay), (bx, by), (cx, cy) = p
                den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if abs(den) < 1e-12: continue
                w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
                w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
                w2 = 1 - w0 - w1
                m = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
                if not m.any(): continue
                z = 1.0 / (w0 * iz[0] + w1 * iz[1] + w2 * iz[2])
                sub = zb[y0:y1 + 1, x0:x1 + 1]; sid = idb[y0:y1 + 1, x0:x1 + 1]
                upd = m & (z < sub); sub[upd] = z[upd]; sid[upd] = pi
    return idb

summary = {}
for n in ['01', '02', '03', '04', '05']:
    k = cams.key(n); img = cv2.imread('/home/claude/kitchen/images/' + k)
    img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    idb = render(n)
    # boundaries between different part ids (incl. background)
    b = np.zeros(idb.shape, bool)
    b[:, 1:] |= idb[:, 1:] != idb[:, :-1]; b[1:, :] |= idb[1:, :] != idb[:-1, :]
    ov = (img * 0.85).astype(np.uint8)
    ys, xs = np.nonzero(b)
    for y, x in zip(ys, xs):
        pid = max(idb[y, x], idb[max(0, y - 1), x], idb[y, max(0, x - 1)])
        ov[y, x] = LAYER_COL[parts[pid][1]] if pid >= 0 else (255, 255, 255)
    cv2.imwrite(f'{OUT}/overlay_{k}', ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    # alignment score: model boundary pixel -> nearest photo edge
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    e = cv2.Canny(g, 30, 90)
    dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)
    inner = b.copy(); inner[:3] = inner[-3:] = False; inner[:, :3] = inner[:, -3:] = False
    d = dt[inner] / S  # in full-res px
    summary[k] = dict(boundary_px=int(inner.sum()), median_px_fullres=float(np.median(d)),
                      within_8px=float((d <= 8).mean()), within_16px=float((d <= 16).mean()))
    print(k, summary[k])
json.dump(summary, open(f'{OUT}/alignment_scores.json', 'w'), indent=1)
