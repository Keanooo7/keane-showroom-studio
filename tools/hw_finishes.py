"""Top Knobs finish labels -> materials/topknobs/finishes.json (PBR parameters for MeshPhysicalMaterial) + QA sheets.

Per finish label (exact strings from products.json) the base colour is measured from the product photos in
img/finish/<slug>/ and solved by inverse rendering: the part is segmented from the white sweep, a robust statistic is
taken (mean linear RGB of the 55th-80th luminance percentile of part pixels, clipped pixels dropped), and the colour is
iterated until a sphere rendered with the family's roughness/metalness under a product-studio environment gives the same
statistic. The studio exposure is anchored on Polished Chrome (linear F0 ~0.55). Roughness, metalness, clearcoat and
anisotropy are set per family below (anisotropy runs along the pull's long axis, +X in the geometry frame).

Secondary (non-metal) materials are the ones the catalog actually has (crystal colours, clear glass, black glazed
ceramic, travertine), with their colours measured from the reference photos of the products that use them (crystal
tints relative to clear crystal on the same sweep), plus generic fallbacks (wood, leather, acrylic) for geometry that
needs one. Antiqued finishes also get "wear": the base colour of their brightest relief, for edge/cavity tinting.

Finishes with fewer than 4 usable photos get more from catalog.json (other SKUs, then other photos of the same SKUs when
they are on white). Photos of crystal/glass/ceramic/stone products are skipped for metal colour: the metal is only the base.

    python3 tools/hw_finishes.py            # writes finishes.json and qa/finishes/sheet_*.jpg
    python3 tools/hw_finishes.py --no-fetch # use only photos already on disk
"""
import json, re, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

R = Path(__file__).resolve().parent.parent
TK = R / 'materials/topknobs'; IMG = TK / 'img'; QA = TK / 'qa/finishes'; QA.mkdir(parents=True, exist_ok=True)
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'
slug = lambda s: re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')
NONMETAL = re.compile(r'crystal|glass|ceramic|travertine', re.I)
BAND = (55, 80)            # luminance percentile band of part pixels used as the colour statistic
CHROME_F0 = 0.55           # exposure anchor
# Pulls shot side-on mostly mirror the overhead box, which reads brighter than the alloy is. Cap the solved luminance at
# the measured reflectance of the base metal (stainless ~0.6, brass ~0.8) so polished stainless does not go white.
CAP = {'Polished Stainless Steel': 0.62, 'Stainless Steel': 0.60, 'Polished Brass': 0.80}

# family, roughness, metalness, clearcoat, anisotropy, notes. Colour is measured.
SPEC = {
    'Polished Chrome':          ('Polished', 0.06, 1, 0, 0, 'mirror chrome; exposure anchor'),
    'Polished Nickel':          ('Polished', 0.08, 1, 0, 0, 'warmer than chrome'),
    'Polished Stainless Steel': ('Polished', 0.10, 1, 0, 0, ''),
    'Polished Brass':           ('Polished', 0.08, 1, 0, 0, 'lacquered bright brass'),
    'Black Nickel':             ('Polished', 0.10, 1, 0, 0, 'smoked mirror finish'),
    'Brushed Satin Nickel':     ('Brushed', 0.34, 1, 0, 0.7, ''),
    'Stainless Steel':          ('Brushed', 0.30, 1, 0, 0.75, 'brushed stainless'),
    'Brushed Bronze':           ('Brushed', 0.35, 1, 0, 0.7, 'brushed warm gold'),
    'Honey Bronze':             ('Satin', 0.28, 1, 0, 0.5, 'satin champagne gold'),
    'Ash Gray':                 ('Satin', 0.30, 1, 0, 0.4, 'satin gunmetal'),
    'Aluminum':                 ('Satin', 0.40, 1, 0, 0.3, 'bead-blast/anodised satin'),
    'Oil Rubbed Bronze':        ('Bronze', 0.45, 1, 0, 0, 'very dark brown, copper glints'),
    'Tuscan Bronze':            ('Bronze', 0.35, 1, 0, 0, 'near-black body, copper wear on edges (see wear)'),
    'Umbrio':                   ('Bronze', 0.35, 1, 0, 0, 'near-black body, copper wear on edges (see wear)'),
    'Sable':                    ('Bronze', 0.42, 0.8, 0, 0, 'dark chocolate satin'),
    'German Bronze':            ('Bronze', 0.50, 1, 0, 0, 'living cast-bronze patina, mottled'),
    'Light Bronze':             ('Bronze', 0.45, 1, 0, 0, 'living cast bronze'),
    'Mahogany Bronze':          ('Bronze', 0.48, 1, 0, 0, 'living cast bronze, red-brown'),
    'Medium Bronze':            ('Bronze', 0.50, 1, 0, 0, 'living cast bronze, dark brown'),
    'Silicon Bronze Light':     ('Bronze', 0.48, 1, 0, 0, 'living cast bronze, grey'),
    'Dark Antique Brass':       ('Brass', 0.50, 1, 0, 0, 'black body, brass on raised detail (see wear)'),
    'Brass Antique':            ('Brass', 0.50, 1, 0, 0, 'dark recesses, brass on raised hammered detail (see wear)'),
    'Flat Black':               ('Black', 0.62, 0.3, 0, 0, 'powder coat'),
    'Coal Black':               ('Black', 0.68, 0.3, 0, 0, 'textured powder coat'),
    'Patina Black':             ('Black', 0.50, 0.9, 0, 0, 'hand-forged steel, black patina'),
    'Black Iron':               ('Black', 0.55, 0.8, 0, 0, 'mottled dark pewter-like iron (see wear)'),
    'Cast Iron':                ('Specialty', 0.60, 0.8, 0, 0, 'antiqued grey cast iron, mottled (see wear)'),
    'Rust':                     ('Specialty', 0.75, 0.3, 0, 0, 'dark rust-brown cast iron'),
    'Antique Copper':           ('Specialty', 0.45, 1, 0, 0, 'brushed-antiqued copper'),
    'Old English Copper':       ('Specialty', 0.50, 1, 0, 0, 'antiqued copper, darker recesses (see wear)'),
    'Patina Rouge':             ('Specialty', 0.55, 0.9, 0, 0, 'red-brown forged patina'),
    'Pewter Antique':           ('Specialty', 0.40, 1, 0, 0, 'warm antiqued pewter'),
    'Pewter Light':             ('Specialty', 0.48, 1, 0, 0, 'light matte pewter'),
    'Pewter':                   ('Specialty', 0.42, 1, 0, 0, 'forged pewter'),
}
# Labels without photos of the metal itself: same parameters as another entry.
ALIAS = {'Chrome': ('Polished Chrome', 'only used for the chrome bases of the two travertine knobs; photos show the stone'),
         'Default': ('Stainless Steel', 'unnamed finish of SS46 (brushed stainless, use this) and SS47 (polished stainless: '
                                        'use "Polished Stainless Steel" for that product)')}
WEAR = {'Tuscan Bronze', 'Umbrio', 'Dark Antique Brass', 'Brass Antique', 'Black Iron', 'Cast Iron', 'Old English Copper',
        'German Bronze', 'Oil Rubbed Bronze', 'Patina Rouge'}
# These bodies photograph as a cold near-black (studio reflections); the copper relief is what the eye reads as
# the finish, and the studio geometry has no worn edges, so part of the wear hue goes into the body (luminance kept).
WARM = {'Tuscan Bronze': 0.6, 'Umbrio': 0.6}

# Secondary materials: id -> (label, name regex of the products that use it (colour sampled from their photos), ior)
SECONDARY = {
    'crystal':            ('Clear crystal',      r'^(Clear|Clear Octagon|Clear Melon|Clear Crystal Center|Wentworth|Crystal Emerald) ', 1.55),
    'crystal_amber':      ('Amber crystal',      r'^Amber Crystal', 1.55),
    'crystal_wine':       ('Wine crystal',       r'^Wine', 1.55),
    'crystal_light_blue': ('Light blue crystal', r'^Light Blue Crystal', 1.55),
    'crystal_blue':       ('Blue crystal',       r'^Blue Crystal', 1.55),
    'crystal_green':      ('Green crystal',      r'^Green Crystal', 1.55),
    'crystal_pink':       ('Pink crystal',       r'^Pink Crystal', 1.55),
    'crystal_black':      ('Black crystal',      r'^Black (Octagon )?Crystal', 1.55),
    'glass':              ('Clear glass',        r'^Clarity Clear Glass', 1.5),
    'ceramic':            ('Black glazed ceramic', r'^Ceramic Large', 1.5),
    'travertine':         ('Tumbled travertine', r'^Tumbled Travertine', 1.5),
    'travertine_red':     ('Honed red travertine', r'^Honed Red Travertine', 1.5),
}


def get(url, dest):
    if dest.exists() and dest.stat().st_size: return dest
    for i in range(3):
        try:
            dest.write_bytes(urlopen(Request(url, headers={'User-Agent': UA}), timeout=60).read()); return dest
        except Exception:
            time.sleep(2 + 2 * i)


def lin(a): return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
def srgb(a): a = np.clip(a, 0, 1); return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)
def hexc(c): return '#' + ''.join(f'{int(round(v * 255)):02x}' for v in srgb(np.asarray(c)))
def luma(a): return a @ np.array([0.2126, 0.7152, 0.0722])


def segment(a):
    """sRGB float image on white -> part mask (antialiased rim eroded away)."""
    m = (a.min(-1) < 0.93) | (a.max(-1) - a.min(-1) > 0.06)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=1))
    lab, n = ndi.label(m)
    if n > 1:  # drop specks, keep every sizeable piece (pulls photograph as one blob, two-tone posts too)
        sz = ndi.sum(m, lab, range(1, n + 1)); m = np.isin(lab, 1 + np.flatnonzero(sz > 0.02 * sz.max()))
    return ndi.binary_erosion(m, iterations=2)


def on_white(a):
    """product shot on the white sweep (room scenes are not): the four corners are white"""
    b = np.concatenate([a[:40, :40], a[:40, -40:], a[-40:, :40], a[-40:, -40:]]).reshape(-1, 3)
    return (b.min(-1) > 0.95).mean() > 0.97


def band_stat(a, m, band=BAND):
    """mean linear RGB of the part pixels between two luminance percentiles, clipped pixels dropped"""
    px = lin(a[m]); y = luma(px)
    lo, hi = np.percentile(y, band)
    sel = (y >= lo) & (y <= hi) & (a[m].max(-1) < 0.985)
    return px[sel].mean(0) if sel.sum() > 20 else px.mean(0)


# ---------------------------------------------------------------- studio environment + split-sum shading
def env(d):
    """Product-photo studio (camera on +Z, Y up): dark room on the camera side, white sweep behind and below,
    overhead softbox, two tall strip boxes front-left and front-right."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    L = 0.06 + 0.9 * np.clip((0.15 - z) / 0.4, 0, 1)                      # sweep behind
    L = L + 0.6 * np.clip((-y - 0.35) / 0.25, 0, 1)                       # white table below
    L = L + 2.6 * np.clip((y - 0.62) / 0.2, 0, 1)                         # overhead box
    for sx in (-1, 1):                                                    # strip boxes: narrow in azimuth, tall
        az = np.arctan2(x, z) - sx * 0.85
        L = L + 2.2 * np.clip((0.22 - np.abs(az)) / 0.06, 0, 1) * np.clip((0.75 - np.abs(y)) / 0.1, 0, 1) * (z > -0.2)
    return L[..., None] * np.ones(3)


_th, _ph = np.meshgrid((np.arange(48) + 0.5) / 48 * np.pi, (np.arange(96) + 0.5) / 96 * 2 * np.pi, indexing='ij')
DIRS = np.stack([np.sin(_th) * np.cos(_ph), np.cos(_th), np.sin(_th) * np.sin(_ph)], -1).reshape(-1, 3).astype(np.float32)
SA = (np.sin(_th) * (np.pi / 48) * (2 * np.pi / 96)).reshape(-1).astype(np.float32)
_cache = {}


def lobe(dirs, k, scale, tag):
    """env averaged over a lobe exp(k (cos - 1)) around each direction; k=0 -> irradiance / pi.
    Cached per (pixel set, lobe, exposure): colour iterations then cost nothing."""
    key = (tag, k, scale)
    if key in _cache: return _cache[key]
    E = (env(DIRS) * scale).astype(np.float32); out = np.zeros((len(dirs), 3), np.float32)
    for i in range(0, len(dirs), 4096):
        c = dirs[i:i + 4096].astype(np.float32) @ DIRS.T
        if k == 0: out[i:i + 4096] = (np.maximum(c, 0) * SA) @ E / np.pi
        else: w = np.exp(k * (c - c.max(1, keepdims=True))) * SA; out[i:i + 4096] = (w @ E) / w.sum(1, keepdims=True)
    _cache[key] = out; return out


def env_brdf(r, nv):  # Karis' analytic fit of the split-sum DFG terms
    c = r * np.array([-1, -0.0275, -0.572, 0.022]) + np.array([1, 0.0425, 1.04, -0.04])
    a = np.minimum(c[0] ** 2, 2 ** (-9.28 * nv)) * c[0] + c[1]
    return (-1.04 * a + c[2])[:, None], (1.04 * a + c[3])[:, None]


def kexp(r): a = max(r, 0.04) ** 2; return min(2 / a ** 2 - 2, 3000)


def shade(n, m, scale, tag, view=np.array([0, 0, 1.0])):
    """normals (N,3) + material dict -> linear radiance (N,3)"""
    c = np.asarray(m['color'], float); r = m['roughness'] + 0.08 * m.get('anisotropy', 0)
    nv = np.clip(n @ view, 1e-3, 1); refl = 2 * nv[:, None] * n - view
    A, B = env_brdf(r, nv)
    f0 = 0.04 * (1 - m['metalness']) + c * m['metalness']
    out = lobe(refl, kexp(r), scale, tag + 'r') * (f0 * A + B)
    if m.get('transmission'):  # a solid ball acts as a lens: the scene behind it seen inverted, tinted by colour
        t = np.stack([-n[:, 0], -n[:, 1], -np.ones(len(n))], 1); t /= np.linalg.norm(t, axis=1, keepdims=True)
        out = out + (1 - (0.04 * A + B)) * m['transmission'] * c * lobe(t, kexp(r), scale, tag + 't')
    else:
        out = out + (1 - m['metalness']) * c * (1 - (0.04 * A + B)) * lobe(n, 0, scale, tag + 'n')
    if m.get('clearcoat'):
        A2, B2 = env_brdf(0.05, nv); cf = m['clearcoat'] * (0.04 * A2 + B2)
        out = out * (1 - cf) + cf * lobe(refl, kexp(0.05), scale, tag + 'r')
    return out


S = 120
_yy, _xx = (np.mgrid[0:S, 0:S] + 0.5) / S * 2 - 1
SPHERE = _xx ** 2 + _yy ** 2 < 0.97
SPH_N = np.stack([_xx, -_yy, np.sqrt(np.clip(1 - _xx ** 2 - _yy ** 2, 0, 1))], -1)[SPHERE]
CYL = np.abs(_yy) < 0.3
_cz = np.sqrt(np.clip(1 - (_yy / 0.3) ** 2, 0, 1))
CYL_N = np.stack([np.zeros_like(_yy), -_yy / 0.3, _cz], -1)[CYL]


def render(m, scale, shape='sphere'):
    mask, n = (SPHERE, SPH_N) if shape == 'sphere' else (CYL, CYL_N)
    img = np.ones((S, S, 3)); img[mask] = srgb(shade(n, m, scale, shape)); return img, mask


def solve(target, spec, scale, iters=8, band=BAND):
    """colour so that the rendered sphere's band statistic equals the photo's"""
    c = np.clip(target * 2, 0.01, 0.95)
    for _ in range(iters):
        img, mk = render(dict(spec, color=c), scale)
        got = band_stat(img, ndi.binary_erosion(mk, iterations=2), band)
        c = np.clip(c * (target / np.maximum(got, 1e-4)) ** 0.9, 0.003, 0.98)
    return c


# ---------------------------------------------------------------- photos
P = json.load(open(TK / 'products.json'))['products']
CAT = json.load(open(TK / 'catalog.json'))['products']
skip_sku = {sku for p in P if NONMETAL.search(p['name']) or any(s.get('metal') in ('Stone', 'Ceramic') for s in p['sizes'])
            for s in p['sizes'] for sku in s['skus'].values()}
labels = sorted({f for p in P for f in p['finishes']})


def photos(label, fetch):
    d = IMG / 'finish' / slug(label); d.mkdir(parents=True, exist_ok=True)
    have = lambda: sorted(f for f in d.glob('*.jpg') if f.stem.split('_')[0] not in skip_sku)
    if fetch and len(have()) < 4 and label not in ALIAS:
        vs = [v for p in CAT for v in (p['variants'] or []) if (v['finish'] or 'Default') == label and v['sku'] not in skip_sku]
        for v in vs:  # other SKUs first, then further photos of the same SKUs if they are on white
            if len(have()) >= 8: break
            if v['images'] and get(v['images'][0], d / f'{v["sku"]}.jpg'): pass
        for v in vs:
            for i, u in enumerate(v['images'][1:4], 1):
                if len(have()) >= 6: break
                f = d / f'{v["sku"]}_{i}.jpg'
                if f.exists() or f.with_suffix('.scene').exists() or not get(u, f): continue
                if not on_white(np.asarray(Image.open(f).convert('RGB'), float) / 255): f.rename(f.with_suffix('.scene'))  # kept so reruns skip it
    return have()


def crop(f, m):
    ys, xs = np.nonzero(m); h, w = m.shape
    return Image.open(f).convert('RGB').crop((max(xs.min() - 4, 0), max(ys.min() - 4, 0), min(xs.max() + 5, w), min(ys.max() + 5, h)))


def measure(files):
    """per photo: band statistic, relief-highlight statistic (92-99th pct), crop of the part"""
    stats, wear, crops = [], [], []
    for f in files:
        a = np.asarray(Image.open(f).convert('RGB'), float) / 255
        if not on_white(a): continue
        m = segment(a)
        if m.sum() < 400: continue
        stats.append(band_stat(a, m)); wear.append(band_stat(a, m, (92, 99)))
        crops.append(crop(f, m))
    return np.array(stats), np.array(wear), crops


def measure_secondary(rx, band=(20, 60)):
    """colour sample of the non-metal part: the upper (crown) half of the part in the reference photo"""
    stats, crops = [], []
    for p in P:
        if not re.search(rx, p['name']): continue
        for s in p['sizes'][:2]:
            f = TK / s['ref_img'][0]
            a = np.asarray(Image.open(f).convert('RGB'), float) / 255; m = segment(a)
            ys, xs = np.nonzero(m); top = ys.min() + 0.5 * (ys.max() - ys.min())
            if re.search(r'Center|Wentworth|Emerald', p['name']):  # insert in the face: middle of the top
                mm = m & (np.abs(np.arange(m.shape[0])[:, None] - (ys.min() + 0.25 * (ys.max() - ys.min()))) < 0.2 * (ys.max() - ys.min()))
            else:
                mm = m & (np.arange(m.shape[0])[:, None] < top)
            stats.append(band_stat(a, mm, band))
            crops.append(crop(f, m))
    return np.array(stats), crops


def main():
    fetch = '--no-fetch' not in sys.argv
    meas = {L: measure(photos(L, fetch)) for L in labels if L not in ALIAS and ' and ' not in L}
    # exposure: scale the studio so the chrome solve lands on CHROME_F0
    scale = 1.0
    for _ in range(3):
        st = np.median(meas['Polished Chrome'][0], 0)
        c = solve(st, dict(zip(('roughness', 'metalness'), SPEC['Polished Chrome'][1:3])), scale)
        scale *= c.mean() / CHROME_F0
    out, rows = {}, []
    for L in labels:
        if L in ALIAS or ' and ' in L: continue
        fam, r, mt, cc, an, note = SPEC[L]
        st, wr, crops = meas[L]
        spec = dict(roughness=r, metalness=mt, clearcoat=cc, anisotropy=an)
        c = solve(np.median(st, 0), spec, scale)
        if L in CAP: c = c * min(1, CAP[L] / luma(c))
        w = solve(np.median(wr, 0), spec, scale, band=(92, 99)) if L in WEAR else None
        if L in WARM: c = luma(c) * ((1 - WARM[L]) * c / luma(c) + WARM[L] * w / luma(w))
        e = dict(id=f'tk_{slug(L)}', label=L, family=fam, color=[round(float(v), 4) for v in c], roughness=r, metalness=mt,
                 clearcoat=cc, anisotropy=an, swatch=hexc(np.median(st, 0)), samples=len(st), notes=note)
        if cc: e['clearcoatRoughness'] = 0.08
        if w is not None: e['wear'] = [round(float(v), 4) for v in w]
        out[L] = e; rows.append((L, crops, e))
    for L in labels:
        if L in ALIAS:
            src, why = ALIAS[L]; out[L] = dict(out[src], id=f'tk_{slug(L)}', label=L, samples=0, notes=why)
    for L in labels:
        if ' and ' in L:
            body, acc = L.split(' and ')
            st, _, crops = measure(photos(L, fetch))
            out[L] = dict(out[body], id=f'tk_{slug(L)}', label=L, two_tone=[body, acc], samples=len(st),
                          notes=f'body {body}, accent ("mat": "accent") {acc}')
            rows.append((L, crops, out[L]))
    sec, srows = {}, []
    glassy = lambda k: k.startswith(('crystal', 'glass'))
    clear = np.median(measure_secondary(SECONDARY['crystal'][1])[0], 0)  # clear crystal on the sweep: the untinted level
    for k, (lab, rx, ior) in SECONDARY.items():
        st, crops = measure_secondary(rx)
        used = sorted({p['slug'] for p in P if re.search(rx, p['name'])})
        c = np.median(st, 0)
        if k == 'crystal_black':
            e = dict(label=lab, color=[0.006, 0.006, 0.007], roughness=0.03, metalness=0, transmission=0, ior=ior, thickness=0)
        elif glassy(k):
            t = c / clear  # transmission tint relative to clear crystal; hue kept, brightness capped below 1
            tint = np.full(3, 0.97) if k in ('crystal', 'glass') else np.clip(t / t.max() * min(0.97, t.max()), 0.02, 0.97)
            e = dict(label=lab, color=[round(float(v), 4) for v in tint], roughness=0.02, metalness=0, transmission=1, ior=ior,
                     thickness=20)
        else:
            r = 0.06 if k == 'ceramic' else 0.55  # glazed vs honed/tumbled stone
            a = solve(c, dict(roughness=r, metalness=0), scale)
            # a near-white stone saturates the solve (overexposed sweep): keep the photographed hue, albedo at most ~0.8
            a = np.clip(c / c.max() * 0.8 if a.max() > 0.95 else a * min(1, 0.8 / a.max()), 0.005, 1)
            e = dict(label=lab, color=[round(float(v), 4) for v in a], roughness=r, metalness=0, transmission=0, ior=ior, thickness=0)
        e.update(swatch=hexc(c), samples=len(st), used_by=used)
        sec[k] = e; srows.append((k, crops, e))
    for k, lab, c, r in (('wood', 'Wood (fallback, not in catalog)', [0.30, 0.17, 0.08], 0.55),
                         ('leather', 'Leather (fallback, not in catalog)', [0.12, 0.06, 0.03], 0.6),
                         ('acrylic', 'Clear acrylic (fallback, not in catalog)', [0.95, 0.95, 0.95], 0.05)):
        sec[k] = dict(label=lab, color=c, roughness=r, metalness=0, transmission=1 if k == 'acrylic' else 0,
                      ior=1.49 if k == 'acrylic' else 1.5, thickness=10 if k == 'acrylic' else 0, swatch=hexc(c), samples=0, used_by=[])
    (TK / 'finishes.json').write_text(json.dumps(dict(
        note='color is linear sRGB (0..1), the base colour / F0; swatch is sRGB, the photographed appearance (for UI chips). '
             'anisotropy runs along the long (+X) axis. thickness in mm (geometry frame). "wear" (optional) is the '
             'relief-highlight base colour of antiqued finishes, for edge/cavity tinting.',
        exposure=round(scale, 4), finishes=out, secondary=sec), indent=1))
    sheets(rows, srows, scale)
    for L, e in out.items():
        print(f'{L:30s} {e["family"]:9s} {e["swatch"]} r{e["roughness"]:.2f} m{e["metalness"]} n={e["samples"]}')


def sheets(rows, srows, scale):
    """per finish: label + numbers + swatch, rendered sphere (accent for two-tone) and bar, then the photo crops"""
    W = 96
    allrows = rows + srows
    for k in range(0, len(allrows), 12):
        grp = allrows[k:k + 12]
        sh = Image.new('RGB', (230 + W * 8, (W + 6) * len(grp)), 'white'); d = ImageDraw.Draw(sh)
        for i, (L, crops, e) in enumerate(grp):
            y = i * (W + 6)
            d.text((4, y + 6), L[:34], fill='black')
            d.text((4, y + 22), f'{e["swatch"]} r{e["roughness"]} m{e["metalness"]}', fill='black')
            d.text((4, y + 38), f'n={e["samples"]} ' + ' '.join(f'{v:.3f}' for v in e['color']), fill='black')
            d.rectangle((4, y + 56, 60, y + 90), fill=e['swatch'])
            mats = [e] if 'two_tone' not in e else [e, next(r[2] for r in rows if r[0] == e['two_tone'][1])]
            sp, _ = render(mats[-1], scale); cy, _ = render(mats[0], scale, 'cyl')
            sh.paste(Image.fromarray((sp * 255).astype(np.uint8)).resize((W, W)), (230, y))
            sh.paste(Image.fromarray((cy * 255).astype(np.uint8)).resize((W, W)), (230 + W, y))
            for j, c in enumerate(crops[:6]):
                c = c.copy(); c.thumbnail((W, W)); sh.paste(c, (230 + W * (2 + j) + (W - c.width) // 2, y + (W - c.height) // 2))
        sh.save(QA / f'sheet_{k // 12}.jpg', quality=90)


if __name__ == '__main__':
    main()
