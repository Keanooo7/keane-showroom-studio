"""Fit all Manhattan line features, save JSON, and draw QA overlays (projected fitted lines + picks)."""
import sys, json, numpy as np, cv2, importlib
sys.path.insert(0, "/home/claude/kitchen/artifacts/sfm/scripts")
from manhattan import MFrame
import features
importlib.reload(features)
P = "/home/claude/kitchen/artifacts/sfm/work_s075/sparse/0"
R = np.load(P + "/../R_align_vp.npy")
F = MFrame(P, R)
OUT = "/home/claude/kitchen/artifacts/sfm/work_s075/line_fits.json"
names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(features.LINES)
try:
    allres = json.load(open(OUT))
except Exception:
    allres = {}
for nm in names:
    ax, obs = features.LINES[nm]
    r = F.line(ax, obs)
    loo = {}
    for k in obs:
        sub = {kk: v for kk, v in obs.items() if kk != k}
        if len(sub) >= 2:
            loo[k] = F.line(ax, sub)["p"].tolist()
    allres[nm] = dict(axis=r["axis"], free=r["free"], p=r["p"].tolist(), sig=r["sig"].tolist(), rms=r["rms"], maxres=r["maxres"], loo=loo, views=list(obs))
    print(f"{nm:26s} {r['axis']}-line  {r['free'][0]}={r['p'][0]:8.4f} {r['free'][1]}={r['p'][1]:8.4f}  sig={np.round(r['sig'],4)} rms={r['rms']:.2f}px max={r['maxres']:.2f}  loo={ {k: np.round(v,3).tolist() for k,v in loo.items()} }")
json.dump(allres, open(OUT, "w"), indent=1)

# overlays
if "--no-overlay" not in sys.argv:
    imgs = {}
    for k in F.m.ims:
        imgs[k] = cv2.imread(f"/home/claude/kitchen/images/{F.m.ims[k]['name'].split('/')[-1]}")
    rng = np.random.default_rng(1)
    for nm, res in allres.items():
        ax = "xyz".index(res["axis"]); fr = [i for i in range(3) if i != ax]
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        for k in imgs:
            ts = np.linspace(-15, 15, 600)
            X = np.zeros((len(ts), 3)); X[:, ax] = ts; X[:, fr[0]] = res["p"][0]; X[:, fr[1]] = res["p"][1]
            Xm = X @ R  # aligned -> model
            uv, z = F.m.project(k, Xm)
            ok = (z > 0) & (uv[:, 0] > -100) & (uv[:, 0] < 3124) & (uv[:, 1] > -100) & (uv[:, 1] < 4132)
            pts = uv[ok].astype(int)
            for i in range(len(pts) - 1):
                cv2.line(imgs[k], tuple(pts[i]), tuple(pts[i + 1]), col, 2)
            if nm in features.LINES and k in features.LINES[nm][1]:
                for (u, v) in features.LINES[nm][1][k]:
                    cv2.circle(imgs[k], (int(u), int(v)), 9, (0, 0, 255), 2)
            if len(pts):
                cv2.putText(imgs[k], nm, tuple(pts[len(pts) // 2] + 5), 0, 1.0, col, 2)
    for k, im in imgs.items():
        cv2.imwrite(f"/home/claude/kitchen/artifacts/sfm/scripts/view/overlay_{k}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
