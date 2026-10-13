"""One-screw Top Knobs hardware (products.json kind_hint == "knob") -> materials/topknobs/geom/<slug>.json.

Every size gets a SHAPE in the README contract's local frame (mm, door at Y = 0, X the long axis) whose bounds are the
spec's diameter / length x width x projection, and whose silhouette is fitted to the reference photo:

  lathe    round, square and octagonal knobs, eggs, potatoes, ovals. Photo _0 is a side view from a small elevation
           theta, so a circle of radius r at height y projects to an ellipse centred at y cos(theta) with vertical
           semi-axis r sin(theta) (x the stretch for oblong heads). r(y) at 48 knots and theta are fitted so that
           envelope reproduces the photo's per-row half-widths; x is scaled so max diameter = spec, y so the crown =
           spec projection, which is what pins theta. Only when no elevation reconciles the spec projection with the
           photo (rms > 6 % of the diameter, and the photo's own height halves it) does the photo's projection win;
           fit.notes says so. Oblong knobs whose base spec is round keep a round stem and stretch only the head.
  tee      T-knobs and bar-on-post knobs: a lathe stem plus a bar read column by column from the photo, either a
           lathe turned onto X (round / football bars, stretched to the spec width) or an extrude with a bevel.
  extrude  flat one-piece shapes (tabs, levers, finger pulls): the photo outline extruded through the spec width.
  special  ring pulls, latches, backplate knobs and the other pieces no single photo silhouette describes, built from
           measured photo proportions (see SPECIAL). Rings hang towards +Z, so their spec length runs along Z.

Crystal, glass, ceramic and stone bodies are separate group parts with "mat" (crystal, crystal_<colour>, glass, ceramic,
travertine, travertine_red: ids of finishes.json "secondary"); their metal base keeps the finish and is kept widest at the
door (the photo's foot ellipse otherwise reads as a bulge that rolls under).
fit.iou is the IoU of the final SHAPE (meshed by hw_mesh exactly as three.js builds it, rendered at the fitted theta)
against the photo mask. Contact sheets (photo | fitted silhouette in red over it) go to materials/topknobs/qa/knobs/.

Overrides, geom/overrides/<slug>.json: {"type": ..., "params": {...}, "sizes": {"<key>": {"shape": SHAPE}}, "notes"}.
A full shape wins outright (fit.iou is then its IoU at params.theta, else at the generator's fitted elevation; a "spec_fix"
next to it, {"proj_mm": ...}, records dims it uses instead of a catalogue value judged wrong and goes into fit). The
top-level "type" names the geom type. params (all optional, merged over the built-in table KN below; per size under
"sizes": {"<key>": {"params": {...}}}):
  mode      lathe | tee | extrude | special          photo  index into ref_img (default 0)
  section   round | square | hexagon | octagon        rotate lathe turn about Y in degrees
  theta     fixed elevation in degrees (default: fitted)          smooth  profile smoothness weight (default 0.02)
  stem      round | same  (oblong lathe: unstretched stem or stretch the whole knob; default from the base spec)
  split     height fraction where head / crystal body starts (default: the narrowest neck)
  mat       secondary material of the head above split (crystal, glass, ceramic, ...)
  inset     crystal inset in the crown: fraction of the crown radius (crystal "center" knobs);
            inset_of: max measures it from the widest radius instead
  bar       round | flat (tee)       bevel  fraction of half the width used as bevel (tee flat bar / extrude)
  trim      drop rows below this height fraction, or 'auto' (a bolt or post below the base in the photo)
  free_P    false: always keep the spec projection; 'photo': always take it from the photo
  thr, close  photo mask threshold (default 14) and closing (default 2); lower / wider for clear and white bodies
  stem_section, bar_section, z_from_photo, inset_of, origin, holes: see KN entries
  dims      {"diam_mm"|"length_mm"|"width_mm"|"proj_mm": value} spec corrections, with a note saying why
  swap      true: the photo's long axis is the spec width (spec lists length < width)

  python3 tools/hw_knobs.py                         # every knob product
  python3 tools/hw_knobs.py --only kent_knob,riverside_t_knob --sheet-only
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi, optimize

R = Path(__file__).resolve().parent.parent
T = R / 'materials/topknobs'
G = T / 'geom'; OV = G / 'overrides'; QA = T / 'qa/knobs'
sys.path.insert(0, str(R / 'tools'))
import hw_mesh

# ------------------------------------------------------------------ per-product table (by eye from the photos)
SQ, OCT = dict(section='square'), dict(section='octagon')
TEE_R, TEE_F = dict(mode='tee', bar='round'), dict(mode='tee', bar='flat')
KN = {
    # square / octagonal heads
    **{s: SQ for s in ['channing_knob', 'ascendra_knob', 'podium_knob', 'juliet_knob', 'tapered_square_knob',
                       'linear_square_knob', 'aspen_square_knob', 'aspen_ii_square_knob', 'modern_metro_slot_knob',
                       'trevi_square_knob', 'cobblestone_square_knob', 'square_iron_knob_dimpled', 'square_inset_knob',
                       'britannia_square_knob', 'great_wall_flare_knob', 'glacier_knob', 'quilted_knob',
                       'cobblestone_rectangle_knob', 'tower_bridge_knob', 'telfair_knob']},
    **{s: OCT for s in ['davenport_knob', 'spectrum_knob', 'emerald_knob', 'lydia_knob', 'chalet_knob']},
    'dustin_knob': dict(section='hexagon', rotate=30),   # a corner faces the camera: flats on X (spec diameter), corners on Z
    **{s: dict(section='octagon', stem_section='round') for s in ['stanmore_knob', 'devon_knob']},
    'crystal_emerald_knob': dict(section='octagon', split=0.84, mat='crystal'),   # the faceted top cap is the crystal
    'wine_octagon_crystal_knob': dict(section='octagon', mat='crystal_wine', trim='auto'),
    'clear_octagon_crystal_knob': dict(section='octagon', mat='crystal', trim='auto', thr=6, close=4),
    'black_octagon_crystal_knob': dict(section='octagon', mat='crystal_black', trim='auto'),
    # crystal / glass / ceramic / stone heads on a metal base
    'clear_crystal_knob': dict(mat='crystal', thr=6, close=4), 'clear_melon_crystal_knob': dict(mat='crystal', thr=6, close=4),
    'amber_crystal_knob': dict(mat='crystal_amber'),
    'light_blue_crystal_knob': dict(mat='crystal_light_blue', thr=8, close=3), 'blue_crystal_knob': dict(mat='crystal_blue'),
    'green_crystal_knob': dict(mat='crystal_green'), 'pink_crystal_knob': dict(mat='crystal_pink', thr=8, close=3),
    'wine_crystal_knob': dict(mat='crystal_wine'), 'black_crystal_knob': dict(mat='crystal_black'),
    'clarity_clear_glass_knob': dict(mat='glass', thr=6, close=4), 'ceramic_large_knob': dict(mat='ceramic'),
    'tumbled_travertine_knob': dict(mat='travertine', thr=5, close=4),
    'honed_red_travertine_knob': dict(mat='travertine_red'),
    'wentworth_crystal_knob': dict(inset=0.86, mat='crystal'),
    'clear_crystal_center_knob': dict(inset=0.8, mat='crystal'), 'black_crystal_center_knob': dict(inset=0.8, mat='crystal_black'),
    'blue_crystal_center_knob': dict(inset=0.8, mat='crystal_blue'), 'wine_crystal_center_knob': dict(inset=0.8, mat='crystal_wine'),
    # T-knobs and bars on a post
    'clarence_t_knob': dict(mode='tee', bar='round', photo=1),
    **{s: TEE_R for s in ['lydia_t_shape_knob', 'hopewell_t_handle', 'solid_t_handle', 'dempsey_t_knob',
                          'burnham_t_knob', 'prestwick_t_knob', 'luxor_t_handle', 'tuscany_t_handle',
                          'chareaur_t_handle', 'shrewsbury_small_t_handle', 'ribbon_reed_t_pull',
                          't_shaped_twist_knob']},
    'square_twist_t_handle': dict(mode='tee', bar='round', bar_section='square'),
    'newburn_t_knob': dict(mode='tee', bar='round', stem_section='square'),
    'versailles_t_handle': dict(mode='tee', bar='round', z_from_photo=True),
    **{s: TEE_F for s in ['riverside_t_knob', 'kingsbridge_knob', 'stainless_t_knob', 'inset_rail_knob', 'flared_knob',
                          'tower_bridge_t_handle']},
    'cobblestone_oval_knob': dict(stem='round', stem_section='square'),
    # flat one-piece silhouettes
    'arched_knob': dict(mode='extrude', type='t_knob', bevel=0.3),
    'neo_knob_pull': dict(mode='extrude', type='other', bevel=0.25, origin='stem'),
    'thumb_knob': dict(mode='extrude', type='oblong_knob', bevel=0.35),
    'tango_finger_pull': dict(mode='extrude', type='finger_pull', bevel=0.2, holes=True, depth='base',
                              note='extruded through the 7/16" base depth; the listed width repeats the length'),
    # specials
    **{s: dict(mode='special') for s in ['newton_ring', 'tudor_ring_vertical', 'canterbury_knob', 'cabinet_latch',
                                         'transcend_cabinet_latch', 's_shaped_knob', 'slanted_a_knob', 'slanted_b_knob']},
    # spec corrections
    'shrewsbury_knob': dict(dims_by_size={'1_5_16_inch': {'diam_mm': 33.34, 'length_mm': 33.34}},
                            note='TK385 spec lists Diameter/Length 1 15/16" against the 1 5/16" size and Width; used 1 5/16"'),
    'flat_oval_knob': dict(swap=True, note='spec lists length 3/4" and width 1 1/2"; the long axis is put on X'),
    **{s: dict(free_P='photo', note='spec projection 2" repeats the length; projection taken from the photo')
       for s in ['aspen_large_potato_knob', 'aspen_ii_large_potato_knob']},
    'ss46': dict(dims={'diam_mm': 19.05, 'length_mm': 19.05, 'width_mm': 19.05},
                 note='no spec table; 3/4" diameter from the product name, projection from the photo at the fitted elevation'),
    'ss47': dict(dims={'diam_mm': 19.05, 'length_mm': 19.05, 'width_mm': 19.05},
                 note='no spec table; 3/4" diameter from the product name, projection from the photo at the fitted elevation'),
}
OBLONG_TYPES = ('oval', 'egg', 'potato', 'oblong')


# ------------------------------------------------------------------ photo -> mask -> rows

def photo_mask(path, thr=14, close=2, fill=True):
    """Foreground of a product shot on white: anything visibly off-white, holes filled, specks dropped. Clear
    crystal and glass need a lower threshold and a wider closing to keep their faint rims."""
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    m = (255 - a).max(2) > thr
    m = ndi.binary_closing(m, iterations=close, border_value=0) | m
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_fill_holes(m) if fill else m
    lab, n = ndi.label(m)
    if n > 1:
        sz = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, np.flatnonzero(sz > 0.03 * sz.max()) + 1)
    return m


def row_extents(m):
    """Left / right foreground column per row (bottom row first), -1 where a row is empty."""
    has = m.any(1)
    l = np.where(has, m.argmax(1), -1)
    r = np.where(has, m.shape[1] - 1 - m[:, ::-1].argmax(1), -1)
    ys = np.flatnonzero(has)
    lo, hi = ys.max(), ys.min()
    return l[hi:lo + 1][::-1], r[hi:lo + 1][::-1], lo


# ------------------------------------------------------------------ forward model for a lathe seen from elevation theta

def envelope(r, a, sec, y, th, v):
    """Silhouette half-width at heights v of a lathe r(y) (dense samples) seen from elevation th (radians).
    a: per-sample Z/X stretch (vertical semi-axis of each section is r a sin th); sec: per-sample 0 round,
    1 square, 2 octagon (flats facing the camera)."""
    b = np.maximum(r * a * math.sin(th), 1e-6)[:, None]
    t = np.abs(v[None] - (y * math.cos(th))[:, None]) / b
    sec = np.broadcast_to(np.asarray(sec), r.shape)[:, None]
    f = np.where(sec == 0, np.sqrt(np.clip(1 - t * t, 0, 1)), np.where(sec == 1, 1.0, np.clip(math.sqrt(2) - t, 0, 1)))
    return (np.where(t <= 1, f, 0) * r[:, None]).max(0)


def v_range(r, a, y, th):
    b = r * a * math.sin(th); c = y * math.cos(th)
    return (c - b).min(), (c + b).max()


SEC = {'round': 0, 'square': 1, 'octagon': 2, 'hexagon': 2}   # a hexagon is fitted with the octagon's envelope


def fit_lathe(h, s, P, sec_fn, aspect_fn, theta=None, smooth=0.02, K=48, mono=0.0, keep=0.72):
    """Fit knot radii rk (at yk = linspace(0, P, K)) and theta to observed half-widths h (px, bottom row first) at
    s px/mm. aspect_fn(y) / sec_fn(y) give the Z/X stretch and section code per height. Returns (yk, rk, theta, rms_mm).
    For a given theta the radii come from a fixed-point pass (each knot is what the photo shows at the height its
    section projects to, corrected by the model's own error there); theta is then a 1-D search, and a final least
    squares with a banded Jacobian polishes the radii (a row only sees the few knots whose ellipses reach it).
    Below height mono every radius stays at least keep x the widest one there (keep None: the radius may not grow
    upwards at all): a base sits flat on the door (a rounded edge is fine), but its own bottom ellipse can be explained
    as a rim rolling under to a point, which misleads theta."""
    u = np.arange(len(h)) / s
    hm = h / s
    u = np.r_[u, u[-1] + np.arange(1, 1 + len(h) // 12) / s]; hm = np.r_[hm, np.zeros(len(u) - len(hm))]
    sel = np.linspace(0, len(u) - 1, min(len(u), 240)).astype(int)
    u, hm = u[sel], hm[sel]
    yk = np.linspace(0, P, K)
    nm = int(np.searchsorted(yk, mono, side='right'))
    floor = lambda r: np.maximum.accumulate(r[::-1])[::-1] if keep is None else np.full_like(r, keep * r.max(initial=0))
    flat = lambda rk: np.r_[np.maximum(rk[:nm], floor(rk[:nm])), rk[nm:]] if nm > 1 else rk
    yd = np.linspace(0, P, 320)
    ad, ak = aspect_fn(yd), aspect_fn(yk)
    sc = sec_fn(yd)
    hi = hm.max() * 1.3 + 0.5

    def model(rk, th):
        rd = np.interp(yd, yk, rk)
        vmin, _ = v_range(rd, ad, yd, th)
        return envelope(rd, ad, sc, yd, th, u + vmin)

    def solve(th, iters=40):
        rk = np.interp(yk * math.cos(th), u, hm)
        best = (np.inf, rk)
        for _ in range(iters):
            rd = np.interp(yd, yk, rk)
            vmin, _ = v_range(rd, ad, yd, th)
            uk = yk * math.cos(th) - vmin
            mk = envelope(rd, ad, sc, yd, th, uk + vmin)
            c = float(np.mean((model(rk, th) - hm) ** 2))
            if c < best[0]:
                best = (c, rk)
            rk = flat(np.clip(rk + 0.5 * (np.interp(uk, u, hm) - mk), 0, hi))   # damped: square tops oscillate at 0.8
        return best[1], best[0]

    def polish(th, rk):
        yv = u / max(math.cos(th), 0.3)
        band = np.abs(yv[:, None] - yk[None]) <= (P / K) * 4 + hm.max() * math.tan(th) * 1.5 + 0.5
        sp = np.vstack([band, (np.eye(K) + np.eye(K, k=1) + np.eye(K, k=-1))[1:-1] > 0,
                        np.tile(np.arange(K) < nm, (nm, 1))]).astype(int)

        def res(rk):
            d2 = rk[:-2] - 2 * rk[1:-1] + rk[2:]
            return np.r_[model(rk, th) - hm, smooth * d2 * K, 5 * np.maximum(floor(rk[:nm]) - rk[:nm], 0)]
        rk = flat(optimize.least_squares(res, flat(rk), bounds=(0, hi), jac_sparsity=sp, max_nfev=150, diff_step=1e-3).x)
        return rk, float(np.sqrt(np.mean((model(rk, th) - hm) ** 2)))

    if theta is not None:
        cands = [math.radians(theta)]
    else:   # the inner solve is not smooth in theta: polish the best few grid points and their refinements
        grid = np.radians(np.arange(0.5, 48, 2.0))
        cost = np.array([solve(t, 30)[1] for t in grid])
        cands = []
        for i in np.argsort(cost)[:3]:
            o = optimize.minimize_scalar(lambda t: solve(t, 30)[1], bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]),
                                         method='bounded', options=dict(xatol=2e-3))
            cands.append(float(o.x) if o.fun <= cost[i] else float(grid[i]))
    best = None
    for th in cands:
        rk, rms = polish(th, solve(th, 60)[0])
        if best is None or rms < best[2]:
            best = (th, rk, rms)
    th, rk, rms = best
    return yk, rk, th, rms


# ------------------------------------------------------------------ helpers

def inch(t):
    """'1 3/16"' -> mm (first number only)."""
    v = 0.0
    for w in (t or '').replace('"', '').split()[:2]:
        if '/' in w:
            a, b = w.split('/')[:2]
            v += float(a) / float(b) if b.strip('0123456789') == '' and float(b) else 0
        elif w.replace('.', '', 1).isdigit():
            v += float(w)
        else:
            break
    return v * 25.4


def base_dims(t):
    """Base spec '5/8" x 3/8"' -> (a, b) mm; b is None for a single number."""
    if not t:
        return None, None
    p = t.split('x')
    return inch(p[0]) or None, (inch(p[1]) or None) if len(p) > 1 else None


def simplify(pts, tol):
    """Douglas-Peucker on an open polyline."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    d = b - a
    n = np.hypot(*d)
    dist = np.abs(np.cross(d, pts - a)) / n if n > 1e-12 else np.hypot(*(pts - a).T)
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(pts[:i + 1], tol)[:-1], simplify(pts[i:], tol)])


def rnd(a, n=2):
    return [[round(float(x), n) for x in p] for p in np.asarray(a)]


def lathe(profile, section='round', stretch=None, **kw):
    s = {'kind': 'lathe', 'profile': rnd(profile), 'section': section}
    if stretch is not None and any(abs(v - 1) > 1e-3 for v in stretch):
        s['stretch'] = [round(float(v), 4) for v in stretch]
    s.update({k: v for k, v in kw.items() if v is not None})
    return s


def closed_profile(y, r, y0=None, y1=None, tol=0.03):
    """Lathe profile from the door (or y0) to the crown: axis point, the radii, back to the axis."""
    pr = simplify(np.c_[r, y], tol)
    y0 = pr[0, 1] if y0 is None else y0
    y1 = pr[-1, 1] if y1 is None else y1
    q = np.vstack([[0, y0], pr, [0, y1]])
    return q[np.r_[True, np.abs(np.diff(q, axis=0)).max(1) > 1e-6]]


# ------------------------------------------------------------------ silhouette of a SHAPE, IoU against the photo

def project(shape, th):
    """Mesh a SHAPE and project it as the photo camera sees it: x right, v up (mm)."""
    V, F = hw_mesh.mesh(shape)
    return np.c_[V[:, 0], V[:, 1] * math.cos(th) - V[:, 2] * math.sin(th)], F


def silhouette(shape, th, s, size, col0, row0):
    """Binary mask (size) of the SHAPE seen at elevation th, s px/mm, model x = 0 at column col0 and the lowest
    projected point on row row0."""
    P, F = project(shape, th)
    px = np.c_[col0 + P[:, 0] * s, row0 - (P[:, 1] - P[:, 1].min()) * s]
    im = Image.new('L', size, 0)
    d = ImageDraw.Draw(im)
    for t in px[F]:
        d.polygon([tuple(p) for p in t], fill=255)
    return ndi.binary_fill_holes(np.asarray(im) > 0)   # the photo mask has its holes filled too


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


# ------------------------------------------------------------------ size dims

def size_dims(p, sz, prm):
    d = {k: sz.get(k) for k in ('diam_mm', 'length_mm', 'width_mm', 'proj_mm')}
    d.update(prm.get('dims') or {})
    d.update((prm.get('dims_by_size') or {}).get(sz['key'], {}))
    X = d['diam_mm'] or d['length_mm']
    Z = d['width_mm'] or X
    if prm.get('swap') and Z > X:
        X, Z = Z, X
    return X, Z, d['proj_mm']


def crown(yk, rk, zero=0.03):
    """The fit can reach the spec projection through trailing knots of ~zero radius, which cost nothing in
    silhouette but would be a spike in 3D. Drop that tail and stretch only the dome above half radius up to P."""
    P, rmax = yk[-1], rk.max()
    i = int(np.flatnonzero(rk >= zero * rmax).max())
    if P - yk[i] < 0.01 * P:
        return yk, rk
    j = int(np.flatnonzero(rk[:i + 1] >= 0.5 * rmax).max())
    y, r = yk[:i + 1].copy(), rk[:i + 1].copy()
    if i > j:
        y[j:] = y[j] + (y[j:] - y[j]) * (P - y[j]) / (y[i] - y[j])
    else:
        y = y * P / y[-1]
    return np.r_[y, P], np.r_[r, 0.0]


def neck(yk, rk, lo=0.08, hi=0.7):
    """Height of the top of the narrowest stretch between lo and hi of the projection (stem -> head)."""
    P = yk[-1]
    i = np.flatnonzero((yk >= lo * P) & (yk <= hi * P))
    rmin = rk[i].min()
    j = i[rk[i] <= rmin * 1.04 + 0.05]
    k = j.max()
    # walk up to where the head really starts to widen
    while k + 1 < len(rk) and rk[k + 1] < rmin + 0.12 * (rk.max() - rmin):
        k += 1
    return float(yk[k])


# ------------------------------------------------------------------ builders

def build_lathe(p, sz, prm, m, X, Z, P):
    l, r, bot = row_extents(m)
    trim = prm.get('trim')
    h = np.where(l >= 0, (r - l + 1) / 2, 0).astype(float)
    if trim == 'auto':   # a bolt below the base: narrow rows at the bottom, then a step out
        big = h > 0.4 * h.max()
        k = int(np.argmax(big))
        if k > 3 and (~big[:k]).all():
            l, r, h, bot = l[k:], r[k:], h[k:], bot - k
    elif trim:
        k = int(trim * len(h)); l, r, h, bot = l[k:], r[k:], h[k:], bot - k
    col0 = float(np.median(((l + r) / 2)[l >= 0]))
    sec = prm.get('section', 'round')
    a_head = Z / X
    ba, bb = base_dims(sz.get('base'))
    stem = prm.get('stem')
    if stem is None:
        stem = 'same'
        if a_head < 0.93 and ba and bb and abs(bb / ba - a_head) > 0.2 and abs(bb / ba - 1) < 0.1:
            stem = 'round'
    stem_sec = prm.get('stem_section', 'round' if stem == 'round' else sec)
    # spec X belongs to the head; a wider rosette at the bottom gets its own base diameter
    Xmax = X
    if ba and ba > X * 1.03 and (not bb or bb == ba) and np.argmax(h) < 0.35 * len(h):
        Xmax = ba
    s = 2 * h.max() / Xmax
    split = prm.get('split')
    zoned = stem_sec != sec or stem == 'round' or prm.get('mat') and not prm.get('inset')
    theta = prm.get('theta')
    notes = []

    def run(Pfit, ys):
        sec_fn = lambda y: np.where(y < ys, SEC[stem_sec], SEC[sec])
        asp_fn = lambda y: np.where(y < ys, 1.0 if stem == 'round' else a_head, a_head)
        # foot zone that may not roll under: the whole metal base under a crystal / stone body (split at the neck, not
        # a given split such as a crystal cap on a metal head), else the bottom 12 %
        base = prm.get('mat') and not split and ys > 0   # a metal base under a crystal / stone body is widest at the door
        return fit_lathe(h, s, Pfit, sec_fn, asp_fn, theta=theta, smooth=prm.get('smooth', 0.02),
                         mono=ys if base else 0.12 * Pfit, keep=None if base else 0.72)

    ys = -1.0
    yk, rk, th, rms = run(P, ys)
    if zoned:
        ys = split * P if split else neck(yk, rk)
        yk, rk, th, rms = run(P, ys)
    free = prm.get('_free_P') or prm.get('free_P') == 'photo'
    # the photo overrides the spec projection only when the spec cannot be made to fit at any elevation
    if free or prm.get('free_P', True) and (rms > 0.06 * Xmax or math.degrees(th) > 46):
        best = None
        for k in (np.linspace(0.6, 2.5, 39) if free else np.linspace(0.8, 1.5, 29)):
            f = run(P * k, -1.0)        # one zone while searching: a split found at the wrong height misleads
            # a trial projection the photo cannot fill ends in a needle of ~zero radius that costs nothing: skip it
            if f[0][np.flatnonzero(f[1] >= 0.05 * f[1].max()).max()] < 0.93 * P * k:
                continue
            if best is None or f[3] < best[1][3]:
                best = (k, f)
        k, f = best if best else (1.0, run(P, -1.0))
        ys2 = ys
        if zoned:
            ys2 = split * P * k if split else neck(f[0], f[1])
            f = run(P * k, ys2)
        if free or f[3] < rms * 0.5 and abs(k - 1) > 0.04:
            notes.append(f'projection {P * k:.2f} mm from the photo' + ('' if free else
                         f' (the spec {P:.2f} mm leaves {rms:.1f} mm rms at every elevation)'))
            P, (yk, rk, th, rms), ys = P * k, f, ys2
    rk = rk * (Xmax / 2) / rk.max()
    # an elevated view hides the neck under the head and the fit can pinch it to a needle: keep the neck at least half
    # the base spec or 12 % of the head
    ih = int(np.argmax(rk))
    rk[:ih] = np.maximum(rk[:ih], min(max(0.12 * rk[ih], 0.5 * (ba or 0) / 2), 0.45 * rk[ih]))
    inset = prm.get('inset')
    if not inset:
        yk, rk = crown(yk, rk)
    stretch = [1, a_head]
    if zoned:
        i = np.searchsorted(yk, ys)
        r_s = float(np.interp(ys, yk, rk))
        ya, ra = np.r_[yk[:i], ys], np.r_[rk[:i], r_s]
        yb, rb = np.r_[ys, yk[i:]] - ys, np.r_[r_s, rk[i:]]
        A = lathe(closed_profile(ya, ra, 0), stem_sec, None if stem == 'round' else stretch)
        B = lathe(closed_profile(yb, rb, 0), sec, stretch, offset=[0, round(ys, 2), 0], mat=prm.get('mat'))
        shape = {'kind': 'group', 'parts': [A, B]}
        notes.append(f'{"metal base" if prm.get("mat") else "stem"} below {ys:.1f} mm' + (f', {prm["mat"]} above' if prm.get('mat') else ''))
    else:
        prof = closed_profile(yk, rk, 0)
        if inset:   # flat crown at the rim radius, the stone set into it
            dep = round(min(2.0, 0.08 * P), 2)
            side = prof[1:-1][prof[1:-1, 1] <= P - 0.5]
            rt = float(rk.max() if prm.get('inset_of') == 'max' else rk[yk >= P - 1.5].max()); rc = inset * rt
            body = np.vstack([[0, 0], side, [rt, P], [rc, P], [rc, P - dep], [0, P - dep]])
            shape = {'kind': 'group', 'parts': [
                lathe(body, sec, stretch),
                lathe([[0, 0], [rc * 0.995, 0], [rc * 0.995, dep], [0, dep]], sec, stretch, offset=[0, P - dep, 0],
                      mat=prm.get('mat'))]}
            notes.append(f'{prm.get("mat")} inset {2 * rc:.1f} mm across in the crown')
        else:
            shape = lathe(prof, sec, stretch)
    if prm.get('rotate') is not None:
        shape['rotate'] = prm['rotate'] if shape['kind'] == 'lathe' else [0, prm['rotate'], 0]
    if Xmax != X:
        notes.append(f'base rosette {Xmax:.1f} mm is wider than the {X:.1f} mm head')
    return shape, dict(theta=th, s=s, col0=col0, bot=bot, rms=rms, P=P, notes=notes, Xmax=Xmax, Z=Z * Xmax / X,
                       profile=(yk, rk))


def column_edges(m, rows):
    """Per column: first foreground row from the top within rows, and the end of that first run."""
    sub = m[rows[0]:rows[1]]
    has = sub.any(0)
    top = np.where(has, sub.argmax(0), -1)
    run_end = np.full(m.shape[1], -1)
    for c in np.flatnonzero(has):
        col = sub[top[c]:, c]
        e = np.argmin(col) if not col.all() else len(col)
        run_end[c] = top[c] + e - 1
    return top + rows[0], np.where(has, run_end + rows[0], -1), has


def build_tee(p, sz, prm, m, X, Z, P):
    ys_, xs_ = np.nonzero(m)
    top_row, bot = ys_.min(), ys_.max()
    xl, xr = xs_.min(), xs_.max()
    s = (xr - xl + 1) / X
    l, r, _ = row_extents(m)
    w = np.where(l >= 0, r - l + 1, 0).astype(float)
    us = int(np.argmax(w > 0.5 * w.max()))           # first row (from the bottom) that belongs to the bar
    stem_rows = slice(0, max(us, 3))
    sw = float(np.median(w[stem_rows]))
    us_stem = max(3, int(np.argmax(np.r_[w[:us], np.inf] > 1.6 * sw)))   # stem proper, before it flares into the bar
    col0 = float(np.median(((l + r) / 2)[:us_stem]))
    rb = float(np.median(w[:max(3, us_stem // 3)])) / 2 / s
    H = (bot - top_row + 1) / s
    rnd_bar = prm.get('bar', 'round') == 'round'
    # bar edges per column (mm above the silhouette's lowest point), x relative to the stem axis
    tops, ends, has = column_edges(m, (top_row, bot + 1))
    cols = np.arange(m.shape[1])
    vt, vb = (bot - tops) / s, (bot - ends) / s
    xs = (cols - col0) / s
    stem_w = sw / s
    ok = has & (vb > us / s * 0.5) & ~(np.abs(xs) < stem_w * 0.75)
    inb = has & (cols >= xl) & (cols <= xr)
    xb, vt_b = xs[inb], vt[inb]
    vb_b = np.interp(xb, xs[ok], vb[ok]) if ok.any() else np.full(len(xb), us / s)
    ta = np.maximum(vt_b - vb_b, 0.1)
    ra_max = float(np.percentile(ta, 90)) / 2

    def rho_of(th):   # true radius of a round bar whose Z half-width is Z / 2, from its apparent half-thickness
        return math.sqrt(max(ra_max ** 2 - (Z / 2 * math.sin(th)) ** 2, (0.25 * ra_max) ** 2)) / math.cos(th)

    def height(th, Pp):
        if rnd_bar:
            return (Pp - rho_of(th)) * math.cos(th) + ra_max + rb * math.sin(th)
        return Pp * math.cos(th) + (Z / 2 + rb) * math.sin(th)
    notes = []
    ths = np.radians(np.linspace(0, 45, 451))
    if prm.get('theta') is not None:
        th = math.radians(prm['theta'])
    else:
        # side shots of T-knobs are often dead level: a mild prior keeps a 3 % height gap from becoming 40 deg
        th = float(ths[np.argmin([abs(height(t, P) - H) / H + 0.001 * math.degrees(t) for t in ths])])
    if abs(height(th, P) - H) > 0.1 * H:   # no elevation reconciles photo and spec: trust the photo at 15 deg
        th = math.radians(prm.get('theta', 15))
        Pp = P + (H - height(th, P)) / math.cos(th)
        notes.append(f'spec projection {P:.2f} mm does not fit the photo at any elevation; used {Pp:.2f} mm')
        P = Pp
    elif abs(height(th, P) - H) > 0.03 * H:
        notes.append(f'photo height {H:.1f} mm vs {height(th, P):.1f} expected; y rescaled to spec')
    c, sn = math.cos(th), math.sin(th)
    v0 = rb * sn                                       # the base's near edge sits this far below y = 0
    vt_b, vb_b = vt_b - v0, vb_b - v0
    sm = lambda a: ndi.gaussian_filter1d(ndi.median_filter(a, 5, mode='nearest'), 1.5, mode='nearest')
    if rnd_bar:
        ra = sm(np.maximum((vt_b - vb_b) / 2, 0.05))
        ca = (vt_b + vb_b) / 2
        q = 1.0
        for _ in range(4):
            rho = ra / math.sqrt(c * c + q * q * sn * sn)
            q = 1.0 if prm.get('z_from_photo') else (Z / 2) / rho.max()
        if prm.get('z_from_photo'):
            notes.append(f'spec width {Z:.2f} mm contradicts the photo; bar taken as round, {2 * rho.max():.1f} mm across')
        yc = float(np.median(ca[np.abs(xb) < 0.3 * X])) / c
        top = yc + rho.max()
        x0 = float(xb[0])
        prof = closed_profile(xb - x0, rho, tol=0.04)

        def make(fy):
            bar = lathe(prof, prm.get('bar_section', 'round'), [fy, q], rotate=[0, 0, -90],
                        offset=[round(x0, 2), round(yc * fy, 2), 0])
            return bar, yc * fy
        notes.append(f'{prm.get("bar_section", "round")} bar on a lathe turned onto X, {2 * rho.max():.1f} mm thick')
    else:
        yt = sm((vt_b - Z / 2 * sn) / c)
        yb = sm((vb_b + Z / 2 * sn) / c)
        tmed = float(np.median(yt - yb))
        yb = np.minimum(yb, yt - 0.6 * tmed)           # square ends: a taper to a point spikes under the bevel
        top = yt.max()
        bev = round(prm.get('bevel', 0.3) * Z / 2, 2)
        mid = np.abs(xb) < stem_w

        step = max(bev, 0.6)                           # no edge shorter than the bevel, or three.js bevels spike
        k = np.unique(np.r_[np.searchsorted(xb, np.arange(xb[0], xb[-1], step)), len(xb) - 1])
        xb, yt, yb = xb[k], yt[k], yb[k]
        mid = np.abs(xb) < stem_w

        def make(fy):
            out = np.vstack([np.c_[xb, yb * fy], np.c_[xb, yt * fy][::-1]])
            out = simplify(np.vstack([out, out[:1]]), 0.1)[:-1]
            jn = float(np.median((yt + yb)[mid]) / 2 * fy) if mid.any() else float(yb.min() * fy)
            return {'kind': 'extrude', 'outline': rnd(out), 'depth': round(Z, 2), 'bevel': bev}, jn
        notes.append(f'flat bar extruded {Z:.1f} mm, bevel {bev} mm')
    fy = P / top
    uu = np.arange(us_stem) / s - v0
    keep = uu > 0
    zcap = 2 * float(rho.max()) if rnd_bar and prm.get('z_from_photo') else Z   # a wrong spec width must not pinch the stem
    hr = np.minimum(w[:us_stem] / 2 / s, 0.49 * zcap if prm.get('stem_section', 'round') == 'round' else 1e9)[keep]
    hy = (uu / c)[keep]
    if not len(hy):
        hy, hr = np.array([0.0]), np.array([stem_w / 2])
    hr = ndi.median_filter(hr, 3, mode='nearest')
    sst = prm.get('stem_section', 'round')
    stretch = None
    if sst == 'square':
        ba, bb = base_dims(sz.get('base'))
        stretch = [1, (bb / ba) if ba and bb else 1]

    def shape(f):
        bar, yj = make(f)
        st = closed_profile(np.r_[0, hy * f, yj], np.r_[hr[0], hr, hr[-1]], 0, yj)
        return {'kind': 'group', 'parts': [lathe(st, sst, stretch), bar]}
    info = dict(theta=th, s=s, col0=col0, bot=bot, rms=0.0, P=P, notes=notes, iou_shape=shape(1.0))
    if rnd_bar and prm.get('z_from_photo'):
        info['Z'] = 2 * float(rho.max())
    return shape(fy), info


def outline_of(m, holes=False):
    """Outer contour (and optionally holes) of a mask as (col, row) float polylines."""
    import contourpy
    z = np.pad(m.astype(float), 2)
    lines = contourpy.contour_generator(z=ndi.gaussian_filter(z, 0.7)).lines(0.5)
    lines = [ln - 2 for ln in lines if len(ln) > 8]
    area = lambda q: 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(np.roll(q[:, 0], -1), q[:, 1]))
    lines.sort(key=area, reverse=True)
    return lines[0], [q for q in lines[1:] if area(q) > 0.01 * area(lines[0])] if holes else []


def ccw(q, want=True):
    a = 0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(np.roll(q[:, 0], -1), q[:, 1]))
    return q if (a > 0) == want else q[::-1]


def build_extrude(p, sz, prm, m, X, Z, P):
    if prm.get('depth') == 'base':   # the footprint's short side, when the listed width just repeats the length
        Z = base_dims(sz.get('base'))[1] or Z
    ys_, xs_ = np.nonzero(m)
    bot, top_row, xl, xr = ys_.max(), ys_.min(), xs_.min(), xs_.max()
    s = (xr - xl + 1) / X
    fy = P / ((bot - top_row + 1) / s)
    if prm.get('origin') == 'stem':   # the screw sits under the part touching the door
        low = m[bot - max(3, int(0.06 * (bot - top_row))):bot + 1]
        col0 = float(np.mean(np.nonzero(low)[1]))
    else:
        col0 = (xl + xr) / 2
    raw = photo_mask(prm['_photo'], prm.get('thr', 14), prm.get('close', 2), fill=False) if prm.get('holes') else m
    outer, hs = outline_of(raw, prm.get('holes'))
    tomm = lambda q: np.c_[(q[:, 0] - col0) / s, (bot + 0.5 - q[:, 1]) / s * fy]
    out = ccw(simplify(np.vstack([tomm(outer), tomm(outer)[:1]]), 0.08)[:-1])
    out[:, 1] = np.clip(out[:, 1], 0, None)
    sh = {'kind': 'extrude', 'outline': rnd(out), 'depth': round(Z, 2), 'bevel': round(prm.get('bevel', 0.3) * Z / 2, 2)}
    if hs:
        sh['holes'] = [rnd(ccw(simplify(np.vstack([tomm(q), tomm(q)[:1]]), 0.08)[:-1], False)) for q in hs]
    th = 0.0
    return sh, dict(theta=th, s=s, col0=col0, bot=bot, rms=0.0, P=P, fy=fy, Z=Z,
                    notes=[f'photo outline extruded {Z:.1f} mm' + (f', y scaled x{fy:.2f} to the spec projection' if abs(fy - 1) > 0.05 else '')])


# ------------------------------------------------------------------ specials: one builder per product
# Hanging rings hang towards +Z (down in a front view, where -Z is up), so their spec "length" runs along Z.

def torus(R, a, n=20):
    """Lathe profile of a ring of wire radius a around a circle of radius R, lying on the door (y = 0 .. 2a)."""
    t = np.linspace(0, 2 * np.pi, n + 1)
    return np.c_[R + a * np.cos(t), a + a * np.sin(t)]


def box(x0, x1, y0, y1, depth, bevel=0.0, **kw):
    s = {'kind': 'extrude', 'outline': rnd([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), 'depth': round(depth, 2),
         'bevel': round(min(bevel, depth / 2), 2)}
    s.update(kw)
    return s


def plate(outline_xz, t, bevel=0.4):
    """A plate on the door from its outline in door coordinates (x, z down), t thick."""
    return {'kind': 'extrude', 'outline': rnd(ccw(np.asarray(outline_xz) * [1, -1])), 'depth': round(t, 2),
            'bevel': round(bevel, 2), 'rotate': [-90, 0, 0], 'offset': [0, round(t / 2, 2), 0]}


def region_outline(m, rows, s, col0, row0, tol=0.15):
    """Outline (mm, x right, z down from row0) of the mask restricted to rows[0]:rows[1]."""
    sub = np.zeros_like(m)
    sub[rows[0]:rows[1]] = m[rows[0]:rows[1]]
    q, _ = outline_of(sub)
    q = np.c_[(q[:, 0] - col0) / s, (q[:, 1] - row0) / s]
    return simplify(np.vstack([q, q[:1]]), tol)[:-1]


def sp_newton_ring(p, sz, prm, m, X, Z, P):
    W, L = Z, sz['length_mm']            # ring 1 1/2" across, 1 5/8" from bail to ring bottom
    a = 2.4                               # wire radius, ~1/8 of the ring width in the photo
    R = W / 2 - a
    zt = -10.0                            # ring top above the rosette centre (photo: rosette sits inside the ring top)
    rb = (base_dims(sz.get('base'))[0] or 12.7) / 2
    ros = lathe([[0, 0], [rb + 0.8, 0], [rb + 0.8, 1.2], [rb, 2.5], [rb * 0.8, 6], [rb * 0.55, P - 1.5], [rb * 0.35, P], [0, P]])
    bail = box(-3.6, 3.6, 0, 2 * a + 1.2, 6.5, 1.2, rotate=[0, 90, 0], offset=[0, 0, round(zt - (L - W) + 3.25, 2)])
    ring = lathe(torus(R, a), offset=[0, 0, round(zt + W / 2, 2)])
    sh = {'kind': 'group', 'parts': [ros, bail, ring]}
    info = front_info(m, sh, ['ring hangs towards +Z (down); spec length 1 5/8" lies along Z, ring width along X',
                              'wire 4.8 mm, rosette inside the ring top as in the photo (a view from ~55 deg)'], P, th=55)
    info.update(Xmax=W, Z=L)
    return sh, info


def front_info(m, sh, notes, P, th=90.0):
    """Align a front-view (or any) photo by bounding boxes: scale from the model's projected width."""
    Q, _ = project(sh, math.radians(th))
    ys_, xs_ = np.nonzero(m)
    s = (xs_.max() - xs_.min() + 1) / (Q[:, 0].max() - Q[:, 0].min())
    col0 = xs_.min() - Q[:, 0].min() * s
    return dict(theta=math.radians(th), s=s, col0=col0, bot=ys_.max(), rms=0.0, P=P, notes=notes)


def sp_tudor_ring_vertical(p, sz, prm, m, X, Z, P):
    W = Z                                 # ring 1 1/2" across; the backplate is 1 1/4" wide (base spec)
    ba, bb = base_dims(sz.get('base'))
    ys_, xs_ = np.nonzero(m)
    l, r, bot = row_extents(m)
    w = np.where(l >= 0, r - l + 1, 0)
    ring_px = w.max()
    s = ring_px / W
    top = ys_.min()
    ring_top = bot - int(round(ring_px * 1.03))
    col0 = float(np.median(((l + r) / 2)[l >= 0]))
    zr = (ring_top - top) / s             # plate top to ring top
    out = region_outline(m, (top, ring_top + int(0.12 * ring_px)), s, col0, top)
    sc = (bb or 31.75) / (out[:, 0].max() - out[:, 0].min())
    out = out * sc
    piv = zr * sc - 2.0                   # the ring hangs from a boss just above its top
    t, a = 3.0, 3.2
    sh = {'kind': 'group', 'parts': [
        plate(out - [0, piv], t),
        lathe([[0, 0], [5, 0], [5, t + 2], [4, P - 2], [2.5, P], [0, P]]),
        lathe(torus(W / 2 - a, a), offset=[0, t, round(2.0 + W / 2 - 1.0, 2)])]}
    info = front_info(m, sh, [f'ring {W:.1f} mm and backplate {bb or 31.75:.1f} mm wide as specified; at that scale the photo '
                              f'puts the overall length near {zr * sc + W:.0f} mm, not the listed {sz["length_mm"]:.1f}',
                              'ring hangs towards +Z (down), twist not modelled'], P)
    lo, hi = hw_mesh.mesh_bounds(sh)
    info.update(Xmax=W, Z=float(hi[2] - lo[2]))
    return sh, info


def sp_canterbury_knob(p, sz, prm, m, X, Z, P):
    """Knob on an ornate backplate, photo _0 a 3/4 view from about 35 deg."""
    th = math.radians(35)
    l, r, bot = row_extents(m)
    w = np.where(l >= 0, r - l + 1, 0).astype(float)
    n = len(w)
    # the plate is the wide band at the bottom; the knob the narrower part above it
    wide = w > 0.6 * w.max()
    pl_top = n - 1 - int(np.argmax(wide[::-1]))           # highest wide row (from the bottom)
    knob_w = np.percentile(w[pl_top + 1:], 95) if pl_top + 1 < n else w.max() / 3
    s = knob_w / X
    col0 = float(np.median(((l + r) / 2)[pl_top + 1:])) if pl_top + 1 < n else float(np.median((l + r) / 2))
    top_row = bot - (n - 1)
    rows = (bot - pl_top, bot + 1)
    out = region_outline(m, rows, s, col0, bot - pl_top)
    zc = (out[:, 1].min() + out[:, 1].max()) / 2
    out = np.c_[out[:, 0], (out[:, 1] - zc) / math.sin(th) * 0.92]
    t = 2.5
    hk = w[pl_top + 1:].astype(float) / 2
    Pk = P - t
    yk, rk, _, _ = fit_lathe(hk, s, Pk, lambda y: np.zeros_like(y), lambda y: np.ones_like(y), theta=35)
    rk = rk * (X / 2) / rk.max()
    sh = {'kind': 'group', 'parts': [plate(out, t), lathe(closed_profile(yk, rk, 0), offset=[0, t, 0])]}
    info = dict(theta=th, s=s, col0=col0, bot=bot, rms=0.0, P=P,
                notes=[f'backplate {np.ptp(out[:, 0]):.0f} x {np.ptp(out[:, 1]):.0f} mm from the photo (not in the spec), '
                       f'{t} mm thick; knob {X:.1f} mm on top'], Xmax=float(np.ptp(out[:, 0])), Z=float(np.ptp(out[:, 1])))
    return sh, info


def sp_cabinet_latch(p, sz, prm, m, X, Z, P):
    """Box latch: base flange + body + knob, keeper on the right. Proportions from photos _0/_1."""
    bw = 0.66 * X; kx = X - bw - 1.0      # body and keeper widths along X
    x0 = -bw / 2
    knob = lathe([[0, 0], [3.2, 0], [3.2, 3], [2.4, 6], [3.4, 8.5], [6.0, 11.5], [6.6, 14.0], [5.5, 16.6], [3.0, 17.9], [0, 18.0]])
    hb = P - 18.0                         # body height: the knob sits on top of it
    parts = [box(x0, x0 + bw, 0, 1.6, Z, 0.3),
             box(x0 + 1.5, x0 + bw - 1.5, 0, hb, Z * 0.62, 0.8),
             dict(knob, offset=[0, round(hb, 2), 0]),
             box(x0 + bw + 1.0, x0 + bw + 1.0 + kx, 0, 1.6, Z * 0.9, 0.3),
             box(x0 + bw + 2.0, x0 + bw + kx, 0, hb * 0.85, Z * 0.5, 0.6),
             box(x0 + bw - 2.0, x0 + bw + 3.0, hb * 0.35, hb * 0.65, Z * 0.2, 0.3)]
    sh = {'kind': 'group', 'parts': parts}
    info = front_info(m, sh, ['built from boxes: flange + body + bolt + keeper, knob on the body; proportions by eye'], P, th=25)
    return sh, info


def sp_transcend_cabinet_latch(p, sz, prm, m, X, Z, P):
    """Lever latch: the photo-_0 side outline (lever, body, keeper) extruded over a base flange."""
    sh, info = build_extrude(p, sz, dict(prm, bevel=0.3), m, X, 0.45 * Z, P)
    xs = np.asarray(sh['outline'])[:, 0]
    sh = {'kind': 'group', 'parts': [box(xs.min(), xs.max(), 0, 1.6, Z, 0.3), sh]}
    info['notes'] = [f'side outline extruded {0.45 * Z:.1f} mm on a {X:.1f} x {Z:.1f} mm flange (the base spec)']
    info['Z'] = Z
    return sh, info


def sp_s_shaped_knob(p, sz, prm, m, X, Z, P):
    """Figure-of-eight loop on a post: two stretched wire rings meeting over the stem."""
    a = 3.0
    Rx = X / 4
    zl = 0.58 * Z                          # each loop is a narrow slot; the S offsets them across Z
    q = (zl / 2) / Rx
    yc = P - 2 * a
    stem = lathe([[0, 0], [5.5, 0], [5.5, 1.5], [3.6, 3], [3.2, yc * 0.5], [3.8, yc], [3.8, yc + a], [0, yc + a]])
    ring = lambda x, z: lathe(torus(Rx - a, a), stretch=[1, q], offset=[round(x, 2), round(yc, 2), round(z, 2)])
    sh = {'kind': 'group', 'parts': [stem, ring(-Rx, (Z - zl) / 2), ring(Rx, -(Z - zl) / 2)]}
    return sh, front_info(m, sh, ['figure-of-eight as two 6 mm wire loops, offset across Z into an S to the spec width'], P, th=35)


def sp_slanted(p, sz, prm, m, X, Z, P):
    """A thin bar along X crossing a flat tab that stands on the door; the tab is the Z width (1 1/4")."""
    ys_, xs_ = np.nonzero(m)
    bot, top_row, xl, xr = ys_.max(), ys_.min(), xs_.min(), xs_.max()
    s = (xr - xl + 1) / X
    fy = P / ((bot - top_row + 1) / s)
    low = m[bot - max(3, int(0.08 * (bot - top_row))):bot + 1]
    tc = np.nonzero(low.any(0))[0]
    ta, tb = tc.min(), tc.max()
    col0 = (ta + tb) / 2
    tops, ends, has = column_edges(m, (top_row, bot + 1))
    cols = np.arange(m.shape[1])
    bar = has & ((cols < ta - 2) | (cols > tb + 2))
    x = (cols - col0) / s
    yt, yb = (bot - tops) / s * fy, (bot - ends) / s * fy
    xb = x[has]
    yt_b = np.interp(xb, x[bar], yt[bar]); yb_b = np.interp(xb, x[bar], yb[bar])
    k = np.unique(np.r_[np.searchsorted(xb, np.arange(xb[0], xb[-1], 1.0)), len(xb) - 1])
    out = np.vstack([np.c_[xb[k], yb_b[k]], np.c_[xb[k], yt_b[k]][::-1]])
    tbar = float(np.median(yt_b - yb_b))
    tw = (tb - ta + 1) / s
    tab_top = float(yt[(cols >= ta) & (cols <= tb)].max())
    sh = {'kind': 'group', 'parts': [
        {'kind': 'extrude', 'outline': rnd(out), 'depth': round(tbar, 2), 'bevel': round(0.15 * tbar, 2)},
        box(-tw / 2, tw / 2, 0, tab_top, Z, 0.4)]}
    info = dict(theta=0.0, s=s, col0=col0, bot=bot, rms=0.0, P=P, fy=fy,
                notes=[f'bar {tbar:.1f} mm square crossing a {tw:.1f} mm tab that is the {Z:.1f} mm spec width'])
    return sh, info


SPECIAL = {'newton_ring': sp_newton_ring, 'tudor_ring_vertical': sp_tudor_ring_vertical, 'canterbury_knob': sp_canterbury_knob,
           'cabinet_latch': sp_cabinet_latch, 'transcend_cabinet_latch': sp_transcend_cabinet_latch,
           's_shaped_knob': sp_s_shaped_knob, 'slanted_a_knob': sp_slanted, 'slanted_b_knob': sp_slanted}


# ------------------------------------------------------------------ driver

def kind_of(p, prm, X, Z):
    if prm.get('type'):
        return prm['type']
    if prm.get('mode') == 'special':
        return {'newton_ring': 'ring_pull', 'tudor_ring_vertical': 'ring_pull', 'canterbury_knob': 'round_knob',
                's_shaped_knob': 'other', 'slanted_a_knob': 't_knob', 'slanted_b_knob': 't_knob'}.get(p['slug'], 'other')
    if prm.get('mode') == 'tee':
        return 't_knob'
    if prm.get('section') in ('square', 'hexagon', 'octagon'):
        return 'oblong_knob' if X / Z > 1.15 else 'square_knob'
    if X / Z > 1.08 or any(w in p['name'].lower() for w in OBLONG_TYPES):
        return 'oblong_knob'
    return 'round_knob'


def edge(m):
    return m & ~ndi.binary_erosion(m)


def qa_tile(ph, m, sil, txt, cs=260):
    a = np.asarray(ph.convert('RGB')).copy()
    e = ndi.binary_dilation(edge(sil), iterations=1)
    b = a.copy()
    b[e] = (230, 20, 20)
    b[edge(m) & ~e] = (40, 90, 230)
    t = Image.new('RGB', (2 * cs, cs + 34), 'white')
    t.paste(ph.convert('RGB').resize((cs, cs)), (0, 0))
    t.paste(Image.fromarray(b).resize((cs, cs)), (cs, 0))
    d = ImageDraw.Draw(t)
    for i, ln in enumerate(txt[:2]):
        d.text((4, cs + 3 + 15 * i), ln, fill=(20, 20, 20))
    return t


def bounds_check(shape, X, Z, P):
    lo, hi = hw_mesh.mesh_bounds(shape)
    ext = hi - lo
    got = dict(x=ext[0], y=hi[1], z=ext[2])
    want = dict(x=X, y=P, z=Z)
    bad = {k: round(float(got[k]), 2) for k in got if want[k] and abs(got[k] / want[k] - 1) > 0.03}
    return lo, hi, bad


def process(p, only_sheet=False):
    ov_path = OV / f'{p["slug"]}.json'
    ov = json.load(open(ov_path)) if ov_path.exists() else {}
    prm0 = {**KN.get(p['slug'], {}), **(ov.get('params') or {})}
    out = {'slug': p['slug'], 'type': None, 'sizes': {}, 'notes': ' '.join(x for x in [prm0.get('note', ''), ov.get('notes', '')] if x)}
    tiles = []
    for sz in p['sizes']:
        so = (ov.get('sizes') or {}).get(sz['key'], {})
        prm = {**prm0, **(so.get('params') or {})}
        X, Z, P = size_dims(p, sz, prm)
        P_spec = P
        mode = prm.get('mode', 'lathe')
        ph_path = T / sz['ref_img'][min(prm.get('photo', 0), len(sz['ref_img']) - 1)]
        ph = Image.open(ph_path)
        m = photo_mask(ph_path, prm.get('thr', 14), prm.get('close', 2))
        prm = {**prm, '_photo': ph_path}
        if P is None:   # no spec projection: fit the photo with a free projection, seeded from the diameter
            P = X
            prm = {**prm, '_free_P': True}
        fixed = so.get('shape')
        if mode == 'special':
            shape, info = SPECIAL[p['slug']](p, sz, prm, m, X, Z, P)
        elif mode == 'tee':
            shape, info = build_tee(p, sz, prm, m, X, Z, P)
        elif mode == 'extrude':
            shape, info = build_extrude(p, sz, prm, m, X, Z, P)
        else:
            shape, info = build_lathe(p, sz, prm, m, X, Z, P)
        if fixed:   # the generator ran only to align the photo: its fitted projection, notes and shape do not apply
            shape, info['notes'] = fixed, ['shape from geom/overrides']
            info.pop('iou_shape', None); info.pop('fy', None)
            info['P'] = (so.get('spec_fix') or {}).get('proj_mm') or P_spec or float(hw_mesh.mesh_bounds(fixed)[1][1])
            if prm.get('theta') is not None:
                info['theta'] = math.radians(prm['theta'])
        P = info.get('P', P)
        th = info['theta']
        try:
            sil = silhouette(info.get('iou_shape', shape), th, info['s'], m.shape[::-1], info['col0'], info['bot'])
            if info.get('fy') and abs(info['fy'] - 1) > 0.02:   # extrudes squeezed to the spec: compare unsqueezed
                sil = silhouette(scale_y(shape, 1 / info['fy']), th, info['s'], m.shape[::-1], info['col0'], info['bot'])
            score = iou(sil, m)
        except Exception as e:   # pragma: no cover - reported, never fatal
            sil, score = np.zeros_like(m), 0.0
            info['notes'].append(f'render failed: {e}')
        lo, hi, bad = bounds_check(shape, info.get('Xmax', X), info.get('Z', Z), P)
        if lo[1] < -1e-6:
            info['notes'].append(f'geometry below the door ({lo[1]:.2f} mm)')
        if bad:
            info['notes'].append(f'bounds off spec by > 3%: {bad}')
        ext = hi - lo
        out['sizes'][sz['key']] = {'shape': shape, 'fit': {
            'iou': round(score, 3), 'theta_deg': round(math.degrees(th), 1),
            'photo': ph_path.name, 'notes': '; '.join(info['notes'])}}
        # spec_fix: dims used instead of the catalogue's on purpose (typo repairs, photo projections, a photo bar width,
        # a hexagon's corners, a base footprint as the depth); notes say why. Rings use their own frame and are left out.
        fix = dict(so.get('spec_fix') or {}) if fixed else {}
        if not fixed and mode != 'special':
            used = dict(x=info.get('Xmax', X), z=hi[2] - lo[2] if '"hexagon"' in json.dumps(shape) else info.get('Z', Z), y=P)
            keys = dict(x='diam_mm' if sz.get('diam_mm') else 'length_mm', z='width_mm', y='proj_mm')
            for a, k in keys.items():
                if used[a] and abs(used[a] - (sz.get(k) or 0)) > 0.01 * used[a]:
                    fix[k] = round(float(used[a]), 2)
        if fix:
            out['sizes'][sz['key']]['fit']['spec_fix'] = fix
        tiles.append(qa_tile(ph, m, sil, [f'{p["slug"]} {sz["key"]}  iou {score:.3f}  el {math.degrees(th):.0f}',
                                          f'{ext[0]:.1f} x {ext[2]:.1f} x {hi[1]:.1f}  spec {X:.1f} x {Z:.1f} x {P:.1f}']))
    out['type'] = ov.get('type') or kind_of(p, prm0, X, Z)
    if not only_sheet:
        G.mkdir(parents=True, exist_ok=True)
        (G / f'{p["slug"]}.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
    return out, tiles


def scale_y(shape, f):
    """The same SHAPE squeezed along Y (extrude outlines and lathe profiles; offsets follow)."""
    s = json.loads(json.dumps(shape))
    if s['kind'] == 'extrude':
        s['outline'] = [[x, y * f] for x, y in s['outline']]
        s['holes'] = [[[x, y * f] for x, y in h] for h in s.get('holes', [])]
    elif s['kind'] == 'lathe':
        s['profile'] = [[r, y * f] for r, y in s['profile']]
    else:
        s['parts'] = [scale_y(q, f) for q in s['parts']]
    if s.get('offset'):
        s['offset'][1] *= f
    return s


def sheets(tiles, cols=3, rows=8):
    QA.mkdir(parents=True, exist_ok=True)
    for old in QA.glob('sheet_*.jpg'):
        old.unlink()
    per = cols * rows
    for n in range(0, len(tiles), per):
        chunk = tiles[n:n + per]
        w, h = chunk[0].size
        sh = Image.new('RGB', (cols * w, ((len(chunk) + cols - 1) // cols) * h), 'white')
        for i, t in enumerate(chunk):
            sh.paste(t, ((i % cols) * w, (i // cols) * h))
        sh.save(QA / f'sheet_{n // per:02d}.jpg', quality=82)


def _safe(p, sheet_only):
    try:
        return (*process(p, sheet_only), None)
    except Exception as e:
        import traceback
        return None, [], f'{type(e).__name__}: {e} @ {traceback.extract_tb(e.__traceback__)[-1].lineno}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='slug[,slug...]')
    ap.add_argument('--sheet-only', action='store_true', help='render QA sheets without writing geom/')
    a = ap.parse_args()
    prods = [p for p in json.load(open(T / 'products.json'))['products'] if p['kind_hint'] == 'knob']
    if a.only:
        want = a.only.split(',')
        prods = [p for p in prods if p['slug'] in want]
    tiles, report = [], []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_safe, prods, [a.sheet_only] * len(prods)))
    for p, (out, t, err) in zip(prods, res):
        if err:
            print(f"{p['slug'][:32]:32} FAILED {err}")
            report.append((p['slug'], 'FAILED', err))
            continue
        tiles += t
        for k, v in out['sizes'].items():
            report.append((p['slug'], k, v['fit']['iou'], v['fit']['theta_deg'], v['fit']['notes']))
            print(f"{p['slug'][:32]:32} {k[:12]:12} {out['type']:12} iou {v['fit']['iou']:.3f} el {v['fit']['theta_deg']:5.1f}  {v['fit']['notes'][:110]}")
    if tiles:
        sheets(tiles)
    (QA / 'report.json').write_text(json.dumps(report, indent=0))


if __name__ == '__main__':
    main()
