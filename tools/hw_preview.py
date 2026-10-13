"""QA renders of Top Knobs hardware geometry next to the product photos -> materials/topknobs/qa/<slug>.png.

Each size is a row: the spec dims against the mesh bounds (red when off by more than 3 %), the reference photo with the
rendered silhouette edge drawn over it (magenta, both at the photo's mm/px taken from the spec length), three grey
renders of the mesh and the next product photo. Pulls render side (looking along -Z, X right, Y up, door at the
bottom), 3/4 from above and front (looking at the door, X right, -Z up); knobs render 3/4 from 20 deg above, side and
front. The overlay uses whichever view's silhouette aspect best fits the photo, since _0 is not always the side view.
The side view also draws the spec box (blue) and the screw holes (red ticks at +-c2c/2).

    python3 tools/hw_preview.py kent_pull,brixton_rimmed_knob    # qa/kent_pull.png, qa/brixton_rimmed_knob.png
    python3 tools/hw_preview.py --all --sheet qa/sheet.png       # every product, plus contact sheets of the ref sizes
    python3 tools/hw_preview.py --all --kind pull --sheet qa/pulls.png --sheet-only
    python3 tools/hw_preview.py brixton_pull --geom qa/preview/geom --out qa/preview

--ref renders only the reference size (the first one with geometry). --json PATH writes every size's bounds, deltas,
flags and photo IoU. Paths not starting with / are relative to materials/topknobs. A missing or malformed geom file,
or a shape hw_mesh rejects, renders as a red tile; it never stops the run.
"""
import argparse, json, math, os, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw_mesh

TK = Path(__file__).resolve().parent.parent / 'materials/topknobs'
TOL = 0.03
VIEWS = {'pull': [('side', 0, 0), ('3/4', -30, 35), ('front', 0, 90)],
         'knob': [('3/4', 0, 20), ('side', 0, 0), ('front', 0, 90)]}
MAT = {None: (0.72, 0.72, 0.74), 'accent': (0.80, 0.62, 0.38), 'glass': (0.70, 0.84, 0.95), 'crystal': (0.75, 0.88, 1.0),
       'wood': (0.62, 0.45, 0.30), 'leather': (0.50, 0.33, 0.22), 'ceramic': (0.12, 0.12, 0.13), 'acrylic': (0.85, 0.88, 0.9),
       'crystal_black': (0.15, 0.15, 0.17), 'crystal_amber': (0.95, 0.7, 0.25), 'crystal_wine': (0.6, 0.3, 0.2),
       'crystal_blue': (0.35, 0.5, 0.95), 'crystal_light_blue': (0.65, 0.82, 1.0), 'crystal_green': (0.45, 0.8, 0.5),
       'crystal_pink': (0.95, 0.65, 0.75), 'travertine': (0.88, 0.82, 0.7), 'travertine_red': (0.75, 0.45, 0.32)}
mat_rgb = lambda m: MAT.get(m) or MAT['crystal' if (m or '').startswith('crystal') else None]
RED, BLUE, MAG, INK, DIM = (215, 25, 25), (60, 120, 230), (235, 0, 200), (25, 25, 25), (120, 120, 120)


def font(size, bold=False):
    for f in (['/System/Library/Fonts/Menlo.ttc'] if not bold else ['/System/Library/Fonts/HelveticaNeue.ttc', '/System/Library/Fonts/Helvetica.ttc']):
        try:
            return ImageFont.truetype(f, size, index=1 if bold and 'Neue' in f else 0)
        except OSError:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- rasteriser

def camera(az, el):
    """Orthographic camera: (right, up, toward-camera) unit vectors for azimuth about Y and elevation towards +Y."""
    a, e = math.radians(az), math.radians(el)
    r = np.array([math.cos(a), 0, -math.sin(a)])
    c = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    return np.stack([r, np.cross(c, r), c])


def raster(xy, z, F, W, H, chunk=4_000_000):
    """Face id per pixel (-1 = empty) by z-buffer; xy in pixels (y down), larger z is nearer. Vectorised scanline:
    each triangle expands to its rows, each row to the pixels between its edges (pixel centres), so long thin
    triangles cost their area, not their bounding box."""
    fid = np.full(W * H, -1, np.int64)
    zb = np.full(W * H, -np.inf)
    if not len(F):
        return fid.reshape(H, W)
    T, Z = xy[F], z[F]
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    ok = np.abs(area) > 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):   # depth plane z = zA + dzdx (x - xA) + dzdy (y - yA)
        dzdx = ((Z[:, 1] - Z[:, 0]) * (C[:, 1] - A[:, 1]) - (Z[:, 2] - Z[:, 0]) * (B[:, 1] - A[:, 1])) / area
        dzdy = ((B[:, 0] - A[:, 0]) * (Z[:, 2] - Z[:, 0]) - (C[:, 0] - A[:, 0]) * (Z[:, 1] - Z[:, 0])) / area
    y0 = np.clip(np.ceil(T[..., 1].min(1) - 0.5), 0, H).astype(np.int64)
    y1 = np.clip(np.floor(T[..., 1].max(1) - 0.5), -1, H - 1).astype(np.int64)
    nr = np.where(ok & (T[..., 0].max(1) >= 0) & (T[..., 0].min(1) <= W), np.maximum(y1 - y0 + 1, 0), 0)
    t = np.nonzero(nr)[0]
    ti = np.repeat(t, nr[t])
    cy = y0[ti] + np.arange(len(ti)) - np.repeat(np.cumsum(nr[t]) - nr[t], nr[t]) + 0.5
    xl, xr = np.full(len(ti), np.inf), np.full(len(ti), -np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        pa, pb = T[ti, i], T[ti, j]
        hit = (cy - pa[:, 1]) * (cy - pb[:, 1]) <= 0
        with np.errstate(divide='ignore', invalid='ignore'):
            x = pa[:, 0] + (cy - pa[:, 1]) / (pb[:, 1] - pa[:, 1]) * (pb[:, 0] - pa[:, 0])
        x = np.where(hit & np.isfinite(x), x, np.nan)
        xl, xr = np.fmin(xl, x), np.fmax(xr, x)
    x0 = np.clip(np.ceil(xl - 0.5), 0, W).astype(np.int64)
    n = np.maximum(np.clip(np.floor(xr - 0.5), -1, W - 1).astype(np.int64) - x0 + 1, 0)
    cum = np.cumsum(n)
    lo = 0
    while lo < len(n):    # rows in chunks so the pixel arrays stay bounded
        hi = max(lo + 1, int(np.searchsorted(cum, cum[lo] - n[lo] + chunk, 'right')))
        r = slice(lo, hi)
        m = n[r]
        k = np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)
        tp = np.repeat(ti[r], m)
        px, py = np.repeat(x0[r], m) + k, np.repeat(cy[r] - 0.5, m).astype(np.int64)
        zz = Z[tp, 0] + dzdx[tp] * (px + 0.5 - A[tp, 0]) + dzdy[tp] * (py + 0.5 - A[tp, 1])
        pix = py * W + px
        o = np.lexsort((-zz, pix))
        o = o[np.r_[True, pix[o][1:] != pix[o][:-1]]]
        win = zz[o] > zb[pix[o]]
        zb[pix[o][win]], fid[pix[o][win]] = zz[o][win], tp[o][win]
        lo = hi
    return fid.reshape(H, W)


def face_colors(V, F, mats, view):
    """Flat Lambert + Blinn spec, two-sided, key light upper left; per face RGB 0..1."""
    P = V @ view.T
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.where(n[:, 2:] < 0, -1, 1)
    L = np.array([-0.45, 0.65, 0.62]); L /= np.linalg.norm(L)
    Hv = L + [0, 0, 1]; Hv /= np.linalg.norm(Hv)
    fill = np.array([0.7, -0.25, 0.5]); fill /= np.linalg.norm(fill)
    d = np.clip(n @ L, 0, 1)[:, None]
    s = (np.clip(n @ Hv, 0, 1) ** 40)[:, None]
    f = np.clip(n @ fill, 0, 1)[:, None]
    base = np.array([mat_rgb(m) for m in mats])
    return np.clip(base * (0.2 + 0.68 * d + 0.18 * f) + 0.3 * s, 0, 1)


class Model:
    """A SHAPE's mesh with per-face material, projected on demand."""

    def __init__(self, shape):
        ps = hw_mesh.parts(shape)
        starts = np.cumsum([0] + [len(p[0]) for p in ps[:-1]])
        self.V = np.vstack([p[0] for p in ps])
        self.F = np.vstack([p[1] + o for p, o in zip(ps, starts)])
        self.mats = [p[2] for p in ps for _ in range(len(p[1]))]
        self.lo, self.hi = self.V.min(0), self.V.max(0)

    def extent(self, view):
        P = self.V @ view[:2].T
        return P.min(0), P.max(0)

    def render(self, view, s, x0, y1, W, H, ss=2, mask_only=False):
        """Pixel (0, 0) is screen point (x0, y1) in mm; s mm per output pixel."""
        P = self.V @ view.T
        xy = np.c_[(P[:, 0] - x0) / s, (y1 - P[:, 1]) / s] * (1 if mask_only else ss)
        k = 1 if mask_only else ss
        fid = raster(xy, P[:, 2], self.F, W * k, H * k)
        if mask_only:
            return fid >= 0
        col = np.vstack([face_colors(self.V, self.F, self.mats, view), [[1, 1, 1]]])
        img = Image.fromarray((col[fid] * 255).astype(np.uint8))
        return img.resize((W, H), Image.Resampling.BOX) if k > 1 else img


# ---------------------------------------------------------------- photos

def photo(rel):
    """(RGB array, on-white?, foreground mask, bbox x0 y0 x1 y1) or None."""
    try:
        a = np.asarray(Image.open(TK / rel).convert('RGB'))
    except Exception:
        return None
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    white = (border.min(1) > 232).mean() > 0.85
    m = ndimage.binary_opening(a.min(2) < 228, iterations=1)
    lab, n = ndimage.label(m)
    if n:
        sz = np.bincount(lab.ravel())[1:]
        m = np.isin(lab, 1 + np.nonzero(sz >= 0.02 * sz.max())[0])
    ys, xs = np.nonzero(m)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None
    return a, white and box is not None, m, box


def _place(model, v, name, L, box, k, P):
    """The view's silhouette at the photo's mm/px (spec length across the photo bbox), aligned on the bbox: centred
    across, bottom on the door line (centred for the front view). (P x P mask) or None."""
    bx0, by0, bx1, by1 = (c * k for c in box)
    lo, hi = model.extent(v)
    s = (L or (hi[0] - lo[0])) / (bx1 - bx0)
    W, H = int(math.ceil((hi[0] - lo[0]) / s)) + 4, int(math.ceil((hi[1] - lo[1]) / s)) + 4
    if W * H > 4 * P * P:
        return None
    mk = model.render(v, s, lo[0] - 2 * s, hi[1] + 2 * s, W, H, mask_only=True)
    ys, xs = np.nonzero(mk)
    if not len(xs):
        return None
    dx = round((bx0 + bx1) / 2 - (xs.min() + xs.max() + 1) / 2)
    dy = round((by0 + by1) / 2 - (ys.min() + ys.max() + 1) / 2) if name == 'front' else round(by1 - (ys.max() + 1))
    place = np.zeros((P, P), bool)
    y0, y1, x0, x1 = max(dy, 0), min(dy + H, P), max(dx, 0), min(dx + W, P)
    if y1 > y0 and x1 > x0:
        place[y0:y1, x0:x1] = mk[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return place


def overlay(model, ph, views, L, P):
    """Photo panel (P px) with the silhouette edge of the view that fits it best; returns (img, view name, IoU).
    Views whose silhouette aspect is near the photo's are all tried and the best IoU wins, since a pull whose width
    equals its projection looks the same in side and front views."""
    a, _, m, box = ph
    k = P / a.shape[1]
    img = Image.fromarray(a).resize((P, P), Image.Resampling.LANCZOS)
    pa = (box[3] - box[1]) / (box[2] - box[0])
    cands = []
    for i, (name, az, el) in enumerate(views):
        if name == '3/4' and views[0][0] != '3/4':
            continue   # a pull's 3/4 photo is in perspective; never scale against it
        v = camera(az, el)
        lo, hi = model.extent(v)
        w, h = hi - lo
        if w > 0 and h > 0:
            cands.append((abs(math.log(pa / (h / w))) + 0.08 * (i > 0), name, v))
    cands.sort(key=lambda c: c[0])
    pm = np.asarray(Image.fromarray(m).resize((P, P), Image.Resampling.NEAREST))
    best = (None, None, -1)
    for sc, name, v in cands:
        if sc > cands[0][0] + 0.25:
            break
        place = _place(model, v, name, L, box, k, P)
        if place is not None:
            iou = (place & pm).sum() / max((place | pm).sum(), 1)
            if iou > best[2]:
                best = (place, name, float(iou))
    place, name, iou = best
    if place is None:
        return img, name, None
    arr = np.asarray(img).copy()
    arr[place & ~ndimage.binary_erosion(place)] = MAG
    return Image.fromarray(arr), name, iou


# ---------------------------------------------------------------- per product

def frame_of(geom):
    """The generators' documented departures from the plain frame: edge tabs hang their Projection along +Z (Width is
    the plate thickness along Y); hanging rings put their spec length along Z and the ring width along X."""
    n = (geom or {}).get('notes', '') + ' '.join((v.get('fit') or {}).get('notes', '') for v in (geom or {}).get('sizes', {}).values())
    if 'edge tab' in n or 'mode tab' in n:
        return {'W': 1, 'P': 2}
    if 'ring hangs towards +Z' in n:
        return {'L': 2, 'W': 0}
    return {}


def check(sz, lo, hi, fit=None, frame=None):
    """[(axis, spec, mesh, rel diff)] and flag strings for one size. Spec values the generator replaced on purpose
    (fit.spec_fix) are compared instead of the catalogue's."""
    ext = hi - lo
    rows, flags = [], []
    sz = dict(sz, **((fit or {}).get('spec_fix') or {}))
    for ax, key, i in (('L', 'length_mm', 0), ('W', 'width_mm', 2), ('P', 'proj_mm', 1)):
        i = (frame or {}).get(ax, i)
        spec = sz.get(key) or (sz.get('diam_mm') if ax in 'LW' else None)
        if ax == 'L' and not sz.get('c2c_mm') and sz.get('diam_mm'):   # a knob's X is its diameter
            spec = sz['diam_mm']
        d = (ext[i] - spec) / spec if spec else None
        rows.append((ax, spec, float(ext[i]), d))
        if d is not None and abs(d) > TOL:
            flags.append(f'{ax} {d * 100:+.1f}%')
    if lo[1] < -0.05:
        flags.append(f'Y<0 ({lo[1]:.1f})')
    return rows, flags


def load_geom(path):
    try:
        g = json.load(open(path))
        if not isinstance(g, dict) or not isinstance(g.get('sizes'), dict):
            raise ValueError('no "sizes" object')
        return g, None
    except FileNotFoundError:
        return None, 'missing ' + path.name
    except Exception as e:
        return None, f'invalid {path.name}: {e}'[:200]


def size_model(geom, key):
    if geom is None:
        return None, None, None
    e = geom['sizes'].get(key)
    if e is None:
        return None, None, 'no shape for this size'
    shape = e.get('shape', e) if isinstance(e, dict) else e
    try:
        m = Model(shape)
        if not 0 < np.ptp(m.V, 0).max() < 2000:   # keeps a runaway shape from eating the rasteriser's memory
            raise ValueError(f'extent {np.ptp(m.V, 0).round(1).tolist()} mm is not hardware-sized')
        return m, (e.get('fit') if isinstance(e, dict) else None), None
    except Exception as ex:
        return None, None, f'invalid shape: {ex}'[:200]


def panel_views(model, views, sz, P, pad=10):
    """Grey renders at one shared mm/px, spec box and screw ticks drawn on the side view."""
    cams = [(n, camera(az, el)) for n, az, el in views]
    ext = [model.extent(v) for _, v in cams]
    s = max(max((hi[0] - lo[0]) / (P - 2 * pad), (hi[1] - lo[1]) / (P - 2 * pad)) for lo, hi in ext) or 1
    out = []
    for (name, v), (lo, hi) in zip(cams, ext):
        cx, cy = (lo + hi) / 2
        x0, y1 = cx - P / 2 * s, cy + P / 2 * s
        img = model.render(v, s, x0, y1, P, P)
        d = ImageDraw.Draw(img)
        if name == 'side':
            Lm = sz.get('length_mm') or sz.get('diam_mm'); Pm = sz.get('proj_mm'); c2c = sz.get('c2c_mm')
            px = lambda x, y: ((x - x0) / s, (y1 - y) / s)
            if Lm and Pm:
                d.rectangle([px(-Lm / 2, Pm), px(Lm / 2, 0)], outline=BLUE)
            d.line([px(-1e3, 0), px(1e3, 0)], fill=(170, 170, 170))
            for x in ([-c2c / 2, c2c / 2] if c2c else [0]):
                u, w = px(x, 0)
                d.line([(u, w), (u, w + 6)], fill=RED, width=2)
        d.text((4, P - 15), name, fill=DIM, font=font(11))
        out.append(img)
    return out, s


def text_block(d, xy, lines, f):
    x, y = xy
    for ln in lines:
        segs = ln if isinstance(ln, list) else [(ln, INK)]
        u = x
        for t, c in segs:
            d.text((u, y), t, fill=c, font=f)
            u += d.textlength(t, font=f)
        y += f.size + 4
    return y


def fmt(v):
    return '   -  ' if v is None else f'{v:6.1f}'


def process(p, opt):
    """Render one product; returns (contact tile array, summary dict). Never raises."""
    try:
        return _process(p, opt)
    except Exception as e:
        traceback.print_exc()
        t = Image.new('RGB', (opt.tile_w, 120), (255, 235, 235))
        ImageDraw.Draw(t).text((8, 8), f'{p["slug"]}\npreview crashed: {e}'[:300], fill=RED, font=font(12))
        return np.asarray(t), {'slug': p['slug'], 'error': f'preview crashed: {e}'}


def _process(p, opt):
    kind = 'pull' if p['kind_hint'] == 'pull' else 'knob'
    geom, gerr = load_geom(opt.geom / f'{p["slug"]}.json')
    views = VIEWS[kind]
    if geom and geom.get('type') in ('round_knob', 'square_knob', 'oblong_knob', 't_knob') and kind == 'pull':
        views = VIEWS['knob']
    elif geom and '+Z = down' in geom.get('notes', ''):   # cups, drops, plates: see the hood from above, not from below
        views = [(n, az + 180 if n == '3/4' else az, el) for n, az, el in views]
    sizes = p['sizes']
    keys = [s['key'] for s in sizes]
    ref = next((k for k in keys if geom and k in geom['sizes']), keys[0] if keys else None)
    summary = {'slug': p['slug'], 'kind': kind, 'type': geom and geom.get('type'), 'error': gerr, 'sizes': {}}
    rows = []
    for sz in sizes:
        model, fit, err = size_model(geom, sz['key'])
        info = {'error': err or gerr}
        if model is not None:
            r, flags = check(sz, model.lo, model.hi, fit, frame_of(geom))
            info.update(bounds=[model.lo.round(2).tolist(), model.hi.round(2).tolist()], dims=r, flags=flags, fit=fit)
        summary['sizes'][sz['key']] = info
        rows.append((sz, model, info))
    flagged = [(sz['key'], i['flags']) for sz, _, i in rows if i.get('flags')]
    broken = [sz['key'] for sz, m, i in rows if m is None]

    def size_panels(sz, model, info, P):
        phs = []
        for r in sz.get('ref_img') or []:
            q = photo(r)
            if q and q[1]:
                phs.append(q)
            if len(phs) == 2:
                break
        if phs and model is not None:
            th = (info.get('fit') or {}).get('theta_deg')   # knob photos: overlay at the generator's fitted elevation
            ov = [(n, az, th if n == '3/4' and th is not None else el) for n, az, el in views]
            pimg, vname, iou = overlay(model, phs[0], ov, sz.get('length_mm') or sz.get('diam_mm'), P)
            info.update(overlay_view=vname, photo_iou=iou and round(iou, 3))
        elif phs:
            pimg = Image.fromarray(phs[0][0]).resize((P, P), Image.Resampling.LANCZOS)
        else:
            pimg = Image.new('RGB', (P, P), (245, 245, 245))
            ImageDraw.Draw(pimg).text((8, P // 2), 'no photo on white', fill=DIM, font=font(12))
        if model is not None:
            rend, s = panel_views(model, views, sz, P)
            info['mm_per_px'] = round(s, 4)
        else:
            bad = Image.new('RGB', (P, P), (255, 228, 228))
            dd = ImageDraw.Draw(bad)
            dd.rectangle([0, 0, P - 1, P - 1], outline=RED, width=3)
            msg = (info['error'] or 'missing/invalid')
            dd.text((10, 10), 'missing/invalid', fill=RED, font=font(14, True))
            text_block(dd, (10, 34), [msg[i:i + P // 8] for i in range(0, min(len(msg), 8 * (P // 8)), P // 8)], font(11))
            rend = [bad] * 3
        extra = Image.fromarray(phs[1][0]).resize((P, P), Image.Resampling.LANCZOS) if len(phs) > 1 else None
        return pimg, rend, extra

    def dims_lines(sz, info):
        lines = [f'spec {" ".join(a + fmt(sz.get(k) or (sz.get("diam_mm") if a in "LW" else None)) for a, k in (("L", "length_mm"), ("W", "width_mm"), ("P", "proj_mm")))}'
                 + (f'  c2c {sz["c2c_mm"]:.1f}' if sz.get('c2c_mm') else '')]
        if info.get('dims'):
            segs = [('mesh ', INK)]
            for ax, spec, got, dlt in info['dims']:
                segs.append((f'{ax}{got:6.1f}', RED if dlt is not None and abs(dlt) > TOL else INK))
                segs.append((' ', INK))
            lines.append(segs)
            lines.append([('diff ', INK)] + [(('   -   ' if dlt is None else f'{ax}{dlt * 100:+5.1f}% '), RED if dlt is not None and abs(dlt) > TOL else DIM)
                                            for ax, _, _, dlt in info['dims']])
        return lines

    # ---- per-product sheet
    if not opt.sheet_only:
        P, TW = opt.panel, 330
        show = [r for r in rows if not opt.ref or r[0]['key'] == ref]
        W = TW + 5 * P
        img = Image.new('RGB', (W, 64 + len(show) * (P + 8)), 'white')
        d = ImageDraw.Draw(img)
        d.text((10, 8), f'{p["name"]}', fill=INK, font=font(20, True))
        notes = (geom or {}).get('notes') or ''
        hdr = f'{p["slug"]}   kind_hint={p["kind_hint"]}   type={summary["type"]}   {len(sizes)} sizes   {p.get("collection") or ""}'
        d.text((10, 36), hdr + (f'   notes: {notes}'[:160] if notes else ''), fill=DIM, font=font(12))
        if gerr:
            d.text((W // 2, 10), gerr, fill=RED, font=font(14, True))
        y = 64
        for sz, model, info in show:
            pimg, rend, extra = size_panels(sz, model, info, P)
            for i, im in enumerate([pimg] + rend + ([extra] if extra else [])):
                img.paste(im, (TW + i * P, y))
            f = font(12)
            lines = [[(sz['label'], INK)], [(f'{sz["key"]}  {sz.get("ref_sku") or ""}', DIM)]] + dims_lines(sz, info)
            if info.get('fit'):
                lines.append([(f'fit {json.dumps(info["fit"])[:36]}', DIM)])
            if info.get('photo_iou') is not None:
                lines.append([(f'photo IoU {info["photo_iou"]:.2f} vs {info["overlay_view"]}', DIM)])
            if info.get('mm_per_px'):
                lines.append([(f'{info["mm_per_px"]:.3f} mm/px in renders', DIM)])
            for fl in info.get('flags') or []:
                lines.append([('! ' + fl + ' vs spec', RED)])
            if model is None:
                lines.append([(info['error'] or 'missing/invalid', RED)])
            text_block(d, (10, y + 4), lines, f)
            d.line([(0, y + P + 3), (W, y + P + 3)], fill=(225, 225, 225))
            y += P + 8
        img.save(opt.out / f'{p["slug"]}.png', compress_level=3)

    # ---- contact tile (reference size)
    tile = None
    if opt.sheet:
        P, TWd = opt.tile_p, opt.tile_w
        sz, model, info = next(r for r in rows if r[0]['key'] == ref) if rows else (None, None, {'error': 'no sizes'})
        t = Image.new('RGB', (TWd, P + 132), 'white')
        d = ImageDraw.Draw(t)
        bad = model is None or flagged or broken
        d.text((6, 4), p['name'][:44], fill=RED if model is None else INK, font=font(14, True))
        d.text((6, 22), f'{p["slug"]}  {summary["type"] or "-"}  {sz["label"] if sz else ""}'[:84], fill=DIM, font=font(11))
        if sz:
            pimg, rend, _ = size_panels(sz, model, info, P)
            for i, im in enumerate([pimg] + rend[:2]):
                t.paste(im, (6 + i * (P + 4), 40))
            lines = dims_lines(sz, info)
            if info.get('photo_iou') is not None:
                lines.append([(f'photo IoU {info["photo_iou"]:.2f} vs {info["overlay_view"]}', DIM)])
            if flagged:
                msg = f'{len(flagged)}/{len(rows)} sizes off: ' + '; '.join(f'{k} {",".join(v)}' for k, v in flagged)
                lines += [[(msg[i:i + 84], RED)] for i in range(0, min(len(msg), 168), 84)]
            if broken:
                lines.append([(f'{len(broken)}/{len(rows)} sizes missing/invalid: ' + (info.get('error') or gerr or broken[0])[:60], RED)])
            text_block(d, (6, 44 + P), lines, font(11))
        d.rectangle([0, 0, TWd - 1, t.height - 1], outline=RED if bad else (215, 215, 215), width=2 if bad else 1)
        tile = np.asarray(t)
    summary['flagged'] = len(flagged)
    summary['broken'] = len(broken)
    return tile, summary


def sheets(tiles, path, cols):
    """Contact sheets, cols tiles per row, opt.per_page tiles per file: path, or path_01.png, path_02.png..."""
    per = cols * 10
    pages = [tiles[i:i + per] for i in range(0, len(tiles), per)] or [[]]
    out = []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for n, pg in enumerate(pages):
        h = max((t.shape[0] for t in pg), default=100)
        w = max((t.shape[1] for t in pg), default=100)
        rows = math.ceil(len(pg) / cols) or 1
        img = Image.new('RGB', (cols * w + (cols + 1) * 6, rows * (h + 6) + 6), (240, 240, 240))
        for i, t in enumerate(pg):
            img.paste(Image.fromarray(t), (6 + (i % cols) * (w + 6), 6 + (i // cols) * (h + 6)))
        f = path if len(pages) == 1 else path.with_name(f'{path.stem}_{n + 1:02d}{path.suffix}')
        img.save(f)
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('slugs', nargs='?', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--kind', choices=['pull', 'knob'])
    ap.add_argument('--geom', default='geom')
    ap.add_argument('--out', default='qa')
    ap.add_argument('--sheet')
    ap.add_argument('--sheet-only', action='store_true')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--ref', action='store_true', help='only the reference size in per-product images')
    ap.add_argument('--panel', type=int, default=240)
    ap.add_argument('--json')
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count())
    opt = ap.parse_args()
    rel = lambda s: Path(s) if s.startswith('/') else TK / s
    opt.geom, opt.out = rel(opt.geom), rel(opt.out)
    opt.sheet = opt.sheet and rel(opt.sheet)
    opt.tile_p = 190
    opt.tile_w = 3 * opt.tile_p + 20
    opt.out.mkdir(parents=True, exist_ok=True)
    prods = json.load(open(TK / 'products.json'))['products']
    if opt.all:
        sel = prods
    else:
        want = [s for s in opt.slugs.split(',') if s]
        by = {p['slug']: p for p in prods}
        miss = [s for s in want if s not in by]
        if miss:
            print('not in products.json:', ', '.join(miss))
        sel = [by[s] for s in want if s in by]
    if opt.kind:
        sel = [p for p in sel if p['kind_hint'] == opt.kind]
    if not sel:
        ap.error('nothing to render: give slug[,slug...] or --all')
    if opt.sheet_only and not opt.sheet:
        ap.error('--sheet-only needs --sheet PATH')
    if len(sel) > 1 and opt.jobs > 1:
        with ProcessPoolExecutor(min(opt.jobs, len(sel))) as ex:
            res = list(ex.map(process, sel, [opt] * len(sel), chunksize=2))
    else:
        res = [process(p, opt) for p in sel]
    for _, s in res:
        st = s['error'] or ', '.join([f'{s["flagged"]} sizes off spec'] * bool(s.get('flagged'))
                                     + [f'{s["broken"]} sizes missing/invalid'] * bool(s.get('broken'))) or 'ok'
        print(f'{s["slug"]:40s} {st}')
    if opt.sheet:
        for f in sheets([t for t, _ in res], opt.sheet, opt.cols):
            print('sheet', f)
    if opt.json:
        json.dump([s for _, s in res], open(rel(opt.json), 'w'), indent=1)
    if not opt.sheet_only:
        print(f'{len(res)} images in {opt.out}')


if __name__ == '__main__':
    main()
