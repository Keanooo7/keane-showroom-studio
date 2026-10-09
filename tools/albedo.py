"""De-lit albedo estimate per material: median linear RGB of each material's visible pixels across the 4 sharp
SfM views, white-balanced + exposure-normalised against the fireclay sink (assumed neutral albedo 0.85)."""
import sys, json, numpy as np, cv2
sys.argv = [sys.argv[0], '0.25']
sys.path.insert(0, '/home/claude/kitchen/tools')
import importlib.util
spec = importlib.util.spec_from_file_location('ov', '/home/claude/kitchen/tools/overlay.py')
src = open('/home/claude/kitchen/tools/overlay.py').read().split('summary = {}')[0]   # reuse loader+renderer only
ns = {}; exec(compile(src, 'overlay_head', 'exec'), ns)
A = json.load(open('/home/claude/kitchen/cad_output/assembly.json'))
mat_of = {c['id']: c['material_default'] for c in A['components']}
def lin(c): c = c / 255.0; return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
acc = {}
for n in ['02', '03', '04', '05']:
    k = ns['cams'].key(n); img = cv2.imread('/home/claude/kitchen/images/' + k)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(float)
    idb = ns['render'](n)
    er = cv2.erode((idb >= 0).astype(np.uint8), np.ones((5, 5), np.uint8))  # avoid boundaries
    for pi, (name, layer, V, F) in enumerate(ns['parts']):
        pid = name.split('/')[-1]; m = mat_of.get(pid) or mat_of.get(pid.rsplit('_', 1)[0], None)
        if m is None: continue
        msk = (idb == pi)
        msk = cv2.erode(msk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & er.astype(bool)
        if msk.sum(): acc.setdefault(m, []).append(lin(img[msk]))
med = {m: np.median(np.concatenate(v), 0) for m, v in acc.items()}
cnt = {m: int(sum(len(x) for x in v)) for m, v in acc.items()}
ref = med['mat_fixture_fireclay']; gain = 0.85 / ref
out = {m: np.clip(med[m] * gain, 0, 1).round(3).tolist() for m in med}
for m in out: print(f'{m:26s} n={cnt[m]:7d} raw_lin={np.round(med[m],3)} albedo={out[m]}')
json.dump(dict(method=__doc__, reference='mat_fixture_fireclay=0.85 neutral', gain=gain.round(4).tolist(),
               albedo_linear=out, pixel_counts=cnt), open('/home/claude/kitchen/artifacts/verify/albedo_estimate.json', 'w'), indent=1)
