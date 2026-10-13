"""Two-screw Top Knobs hardware (products.json kind_hint == 'pull') -> materials/topknobs/geom/<slug>.json.

Every size is fitted to its own reference photo (materials/topknobs/img/ref/<sku>_<n>.jpg, 490 x 490 on white) and scaled
to the spec table, so overall length / projection / width match it exactly. See materials/topknobs/README.md for the
local frame and the SHAPE kinds. On top of the README frame this generator uses +Z = down the door for a horizontally
mounted pull (right-handed: X right, Y out of the door), which only matters for cup, drop and edge-tab pulls.

Modes (picked from the name, overridable with params.mode):
  side      bar / D / appliance / tab / finger pulls. The side elevation photo is segmented, traced, simplified (0.2 mm),
            scaled X to Overall Length and Y to Projection, centred on the screw posts, and extruded through Width.
  lathe     two-screw items that are really knob-shaped (Contour Knob, Victoria Falls Knob, Oval Pull): the side
            silhouette's half-width per height becomes a lathe profile, stretched to Width.
  cup       cup / bin pulls, photographed from the front (X x Z): the front silhouette as a thin plate on the door, and
            over it a hood shell sliced along X, each slice a quarter superellipse from the silhouette's top edge at the
            door to its bottom edge at the lip, hollow underneath for the fingers. The projection falls off where the
            silhouette is short (screw ears, pointed ends).
  drop      bail / drop pulls, photographed from the front: rosette posts at the screws up to Projection, the bail as a
            round-edged slab lying on the door.
  standoff  a flat shape held off the door by two posts (Half Circle Open, Squiggly): front silhouette as a slab at
            the top of the posts.
  tab       edge tabs that hang over the top edge of the drawer (Europa, Tab Pull): the visible front plate, Width thick.
            Its Projection is how far it hangs down, so it runs along +Z from the origin (the drawer's top edge).
  plate     Mission Plate Pull: backplate + bail ring + two posts.
  dpull     no usable elevation (Kentfield: perspective shots only): parametric D pull from L / P / c2c; when Width is
            much larger than the bar it becomes an offset handle (posts and grip shifted apart in Z by Width).
  edge      Crestview: a plate on the drawer's top edge overhanging the front by Projection, Width-diameter barrel under
            the lip; origin = middle of the top edge at the front.
Side-mode scale: if the feet sit on c2c at the length scale, X uses Overall Length and Y Projection (a projection off by
2x against such a photo is a catalogue typo and is replaced by the photo's, see fit.spec_fix); if the photo is far
shorter than the spec aspect (one stock shot for every length) the posts keep the photo's own proportions, sit on c2c,
and the bar is lengthened between them. Two-tone products (ACCENT) give their posts / collars "mat": "accent".

Photo choice: every photo of the size is segmented; room scenes (part touching the frame / filling it) are dropped and
the one whose silhouette aspect best matches the spec for the mode's view (L/P side, L/W front) wins. A size without a
usable photo borrows the nearest size's silhouette (bars are lengthened between the posts, not stretched).
fit.iou is the IoU of the final outline rasterised back into that photo's pixel frame against the photo mask (null
for parametric shapes). fit.spec_fix lists dims used instead of a spec value judged wrong; fit.notes says why.

Overrides: materials/topknobs/geom/overrides/<slug>.json, {"type", "params", "sizes": {key: {"shape": SHAPE}}, "notes"}.
A full shape for a size wins outright; that size may also carry "spec_fix" ({"proj_mm": ..}: dims the shape uses instead of
a catalogue value judged wrong) and "notes", both copied into its fit. params (whole product, or per size under "sizes": {key: {"params": {...}}}):
  mode     side | lathe | cup | drop | standoff | tab | plate | dpull | edge
  type     geom type written to the file (bar_pull, cup_pull, ...)
  photo    index n of img/ref/<sku>_<n>.jpg to use instead of the automatic choice
  view     side | front: which spec aspect the photo is matched against (default follows mode)
  flip     true: the photo has the door at the top (mirror vertically before tracing)
  crop     [x0, y0, x1, y1] pixel box of the photo to keep (drops screws, labels, a second part)
  thr      ink threshold 0..255 for 'not background' (default 10; lower for chrome / white parts)
  section  round | flat: bar cross-section (default: name keywords, then the FLAT list judged from the 3/4 photos)
  bevel    mm, overrides the bevel derived from section
  depth    mm, overrides the extrusion depth (default Width) when the spec Width is wrong
  levels   cup: number of hood slices along X (default 96)

--sheets writes materials/topknobs/qa/pulls/sheet_NN.png (photo + fitted outline in red, the SHAPE at true proportions
in blue, accent parts in orange; first and last size per product, every size with --only) and report.txt (worst IoU first).

    python3 tools/hw_pulls.py [--only slug[,slug...]] [--sheets]
"""
import argparse, json, math, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import contourpy
from matplotlib.path import Path as MPath

R = Path(__file__).resolve().parent.parent
TK = R / 'materials/topknobs'; GEOM = TK / 'geom'; OVR = GEOM / 'overrides'; QA = TK / 'qa/pulls'

# rectangular / flat bar sections, judged by eye from the 3/4 photos (_1); everything else defaults to round
FLAT = set('''riverside hillmont channing minetta_tab ascendra channing_cup harrison haddonfield morris florham lawrence torbay
modern_metro glacier edgewater exeter holland quilted juliet lydia modern_metro_tab kingsbridge tango_cut_out podium
emerald langston neo tinley twist infinity_bar flat_rail somerdale aspen_flat_sided aspen_ii_flat_sided boro arched hidra
square_bar amwell_bar crescent_flair crest linear tapered tapered_bar nouveau_bow modern_metro_notch_pull_a newport griggs
modern_metro_slot lida oval_long_slot curved_tidal inset_rail modern_metro_notch_pull_b flared u rue euro_arched
modern_metro_asymmetrical sloped princetonian_bar ring slanted roselle alton sibley latham iola ashmore lansing hull
crescent_u princetonian_bar_pull_3_posts half_circle_open crescent verona harmony sydney_flair sydney_thin trevi
tower_bridge cobblestone great_wall trevi_crest wedge channel newburn tango_u_finger square_knob aspen_flat_sided_knob
aspen_ii_flat_sided_knob tapered_knob cressing_cup saddle'''.split())
FLAT_WORDS = ('flat', 'square', 'rail', 'metro', 'tab', 'plate', 'linear', 'slot', 'notch', 'channel')
DROP = {'salisbury_pull', 'eton_pull', 'oxford_pull', 'versailles_pull'}
MODE = {'contour_knob': 'lathe', 'victoria_falls_knob': 'lathe', 'oval_pull': 'lathe', 'half_circle_open_pull': 'standoff',
        'squiggly_pull': 'standoff', 'europa_tab_pull': 'tab', 'europa_cut_out_tab_pull': 'tab', 'tab_pull': 'tab',
        'crestview_tab_pull': 'edge', 'mission_plate_pull': 'plate', 'kentfield_pull': 'dpull',
        # box / scoop cups shot from the side: a side extrusion reads better than a dome
        'channing_cup_pull': 'side', 'cressing_cup_pull': 'side', 'sydney_flair_knob': 'side'}
FRONT = ('cup', 'drop', 'standoff', 'tab', 'plate')
TYPE = {'contour_knob': 'square_knob', 'square_knob': 'square_knob', 'aspen_flat_sided_knob': 'square_knob',
        'aspen_ii_flat_sided_knob': 'square_knob', 'victoria_falls_knob': 'oblong_knob', 'oval_pull': 'oblong_knob',
        'crestview_tab_pull': 'tab_pull', 'tapered_knob': 'other', 'sydney_flair_knob': 'finger_pull', 'half_circle_open_pull': 'finger_pull', 'tango_u_finger_pull': 'finger_pull'}
ACCENT = {'bit_pull'}  # two-tone: label 'A and B' = bar in A, posts and collars in B (checked on every size's photo)
r2 = lambda v: round(float(v), 2)


def mode_of(p):
    s, n = p['slug'], p['name'].lower()
    if s in MODE: return MODE[s]
    if 'cup' in n: return 'cup'
    if 'drop' in n or s in DROP: return 'drop'
    return 'side'


def type_of(p, mode, sec):
    s, n = p['slug'], p['name'].lower()
    if s in TYPE: return TYPE[s]
    if mode == 'cup' or 'cup' in n: return 'cup_pull'
    if mode == 'drop': return 'drop_pull'
    if mode == 'tab' or 'tab' in n: return 'tab_pull'
    if 'finger' in n: return 'finger_pull'
    if s in ('ring_pull',): return 'ring_pull'
    if 'knob' in n: return 'square_knob' if sec == 'flat' or 'square' in n else ('round_knob' if mode == 'lathe' else 'oblong_knob')
    if 'latch' in n or mode in ('standoff', 'plate'): return 'other'
    if min(z['length_mm'] or 0 for z in p['sizes']) > 300 or all('app' in z['key'] for z in p['sizes']): return 'appliance_pull'
    return 'bar_pull'


def section_of(p, prm):
    if prm.get('section'): return prm['section']
    s, n = p['slug'], p['name'].lower()
    if TYPE.get(s) == 'square_knob': return 'flat'   # lathe mode turns a 'flat' section square
    base = re.sub(r'_(pull|knob|bar)$', '', s)
    if s in FLAT or base in FLAT or any(w in n for w in FLAT_WORDS): return 'flat'
    return 'round'


# ---------------------------------------------------------------- photo -> mask -> loops

def segment(path, thr=10, crop=None, flip=False):
    """Part mask: everything not connected to the border through near-white pixels. Enclosed white pockets become
    holes only when they are big and compact (a ring, a cut-out); thin ones are chrome highlights and stay solid."""
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    if flip: im = im[::-1]
    ink = ndi.gaussian_filter(255 - im.min(2), 0.8)
    if crop:
        x0, y0, x1, y1 = crop; keep = np.zeros(ink.shape, bool); keep[y0:y1, x0:x1] = True; ink[~keep] = 0
    white = ink < thr
    lab, n = ndi.label(white)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, edge[edge > 0])
    # close: a specular streak on a chrome bar can reach the silhouette edge and let the background flood in
    part = ndi.binary_closing(ndi.binary_opening(~bg, iterations=1), structure=np.ones((3, 3)), iterations=3, border_value=0)
    lab, n = ndi.label(part)
    if not n: return None, {}
    area = ndi.sum(part, lab, range(1, n + 1))
    part = np.isin(lab, 1 + np.flatnonzero(area >= 0.01 * area.max()))
    hl, hn = ndi.label(ndi.binary_opening(white & part, iterations=2))
    light = ink[part].mean() < 110  # chrome / nickel: highlights can be big white blobs, so only large pockets count
    for i, sl in enumerate(ndi.find_objects(hl)):
        reg = hl[sl] == i + 1; h, w = reg.shape
        if reg.sum() >= max(80, (0.05 if light else 0.01) * part.sum()) and min(h, w) >= 8 and reg.sum() >= 0.35 * h * w:
            part[sl] &= ~ndi.binary_dilation(reg, iterations=1)
    border = np.r_[part[0], part[-1], part[:, 0], part[:, -1]].mean()
    return part, dict(border=float(border), frac=float(part.mean()))


def loops_of(mask):
    """Closed sub-pixel boundary loops [(N,2) arrays in (col,row)], each tagged outer or hole by nesting."""
    z = ndi.gaussian_filter(np.pad(mask.astype(float), 2), 0.7)
    gen = contourpy.contour_generator(z=z, line_type='Separate')
    ls = [l - 2 for l in gen.lines(0.5) if len(l) > 8]
    out = []
    for i, l in enumerate(ls):
        depth = sum(MPath(o).contains_point(l[0]) for j, o in enumerate(ls) if j != i)
        out.append((l, depth % 2 == 1))
    return out


def area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rdp(pts, eps):
    """Douglas-Peucker on a closed polygon (split at the two farthest-apart vertices)."""
    def simp(a):
        keep = np.zeros(len(a), bool); keep[0] = keep[-1] = True; st = [(0, len(a) - 1)]
        while st:
            i, j = st.pop()
            if j <= i + 1: continue
            p, q = a[i], a[j]; d = q - p; L = math.hypot(*d)
            seg = a[i + 1:j] - p
            dist = np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / L if L > 1e-9 else np.hypot(seg[:, 0], seg[:, 1])
            k = int(np.argmax(dist))
            if dist[k] > eps: keep[i + 1 + k] = True; st += [(i, i + 1 + k), (i + 1 + k, j)]
        return a[keep]
    if np.allclose(pts[0], pts[-1]): pts = pts[:-1]
    i0 = 0; i1 = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    a = simp(np.r_[pts[i0:i1 + 1]]); b = simp(np.r_[pts[i1:], pts[:1]])
    return np.r_[a[:-1], b[:-1]]


def ccw(pts, want=True):
    return pts if (area(pts) > 0) == want else pts[::-1]


def runs(v):
    """[(start, end)] of True runs in a 1-D bool array (end exclusive)."""
    d = np.diff(np.r_[0, v.astype(int), 0]); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def raster(polys, holes, shape):
    im = Image.new('L', shape[::-1], 0); dr = ImageDraw.Draw(im)
    for p in polys: dr.polygon([tuple(q) for q in p], fill=1)
    for h in holes: dr.polygon([tuple(q) for q in h], fill=0)
    return np.asarray(im, bool)


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


# ---------------------------------------------------------------- photo choice

def candidates(sz, prm, want):
    """Segment every photo of the size; score by how well the silhouette aspect (width / height) matches `want`."""
    out = []
    for i, rel in enumerate(sz['ref_img']):
        n = int(re.search(r'_(\d+)\.jpg$', rel).group(1))
        if 'photo' in prm and n != prm['photo']: continue
        m, st = segment(TK / rel, prm.get('thr', 10), prm.get('crop'), prm.get('flip', False))
        if m is None or st['border'] > 0.4 or not 0.003 < st['frac'] < 0.85: continue
        ys, xs = np.nonzero(m); asp = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
        out.append((abs(math.log(asp / want)) + (0 if n == 0 else 0.15), n, rel, m))  # _0 is the studio elevation: prefer it
    return sorted(out, key=lambda c: c[0])


# ---------------------------------------------------------------- builders

def build_side(m, L, P, W, c2c, sec, prm, notes, fix, accent=None):
    lps = loops_of(m)
    outer = [l for l, h in lps if not h]; hole = [l for l, h in lps if h]
    allp = np.vstack(outer); xmin, ymin = allp.min(0); xmax, ymax = allp.max(0)
    sx, sy = L / (xmax - xmin), P / (ymax - ymin)
    rows = np.flatnonzero(m.any(1)); bot = rows[-1]; k = max(3, int(0.04 * (bot - rows[0])))
    feet = [f for f in runs(m[bot - k:bot + 1].any(0)) if f[1] - f[0] >= 2]
    x0 = (xmin + xmax) / 2; src = None
    if len(feet) >= 2 and c2c:
        cl, cr = np.mean(feet[0]) - 0.5, np.mean(feet[-1]) - 0.5
        # a centre post: c2c may be post to post (Princetonian 3 Posts) or end to end (long Solid Bar)
        err = min(abs((cr - cl) * sx - c2c * k) / (c2c * k) for k in ((1, 2) if len(feet) == 3 else (1,)))
        # the feet only prove the length scale when the bar overhangs them; posts at the very ends fit any length
        if err < 0.06 and (L - c2c) / L > 0.08:
            x0 = (cl + cr) / 2; Pp = (ymax - ymin) * sx
            if not 0.5 < Pp / P < 2 and ymax - ymin > 40:  # posts confirm the length scale, spec projection off 2x: a typo
                notes.append(f'spec Projection {P} mm contradicts the photo ({Pp:.1f} mm at the c2c-confirmed scale): photo value used')
                P = round(float(Pp), 2); sy = P / (ymax - ymin); fix['proj_mm'] = P
        elif sx / sy > 1.25 and len(feet) == 2 and (cr - cl) * sy < c2c:
            # photo much shorter than the spec aspect: one stock shot serves every length. Keep the posts' own
            # proportions (photo scale = projection scale), put them on the screws and lengthen the bar between them
            x0 = (cl + cr) / 2; pw = min(min(f[1] - f[0] for f in (feet[0], feet[-1])) / 2, (cr - cl) / 4)
            ovl, ovr = cl - xmin, xmax - cr; ov = L - c2c
            src = np.array([xmin, cl - pw, cl + pw, cr - pw, cr + pw, xmax])
            dl = ov * ovl / (ovl + ovr) if ovl + ovr > 0 else ov / 2
            dst = np.array([-c2c / 2 - dl, -c2c / 2 - pw * sy, -c2c / 2 + pw * sy, c2c / 2 - pw * sy, c2c / 2 + pw * sy, c2c / 2 + ov - dl])
            if np.any(np.diff(dst) <= 0) or np.any(np.diff(src) <= 0):
                # posts flush with the bar ends: keep each end (post included) rigid at the projection scale instead
                src = np.array([xmin, cl + pw, cr - pw, xmax])
                dst = np.array([-L / 2, -L / 2 + (cl + pw - xmin) * sy, L / 2 - (xmax - cr + pw) * sy, L / 2])
                if np.any(np.diff(dst) <= 0) or np.any(np.diff(src) <= 0): src = None
            if src is not None: notes.append(f'photo is not to length scale (posts {(cr - cl) * sx:.0f} mm apart at the length scale): posts kept at the projection scale and set on c2c, bar lengthened between them')
        elif err < 0.06: x0 = (cl + cr) / 2
        else: notes.append(f'outer feet {(cr - cl) * sx:.1f} mm apart vs c2c {c2c}: centred on the overall length')
    if src is None and abs(sx / sy - 1) > 0.08:
        notes.append(f'x/y photo scales disagree by {abs(sx / sy - 1) * 100:.0f}% (perspective or not a pure side view); spec length and projection used')
    fx = (lambda x: np.interp(x, src, dst)) if src is not None else (lambda x: (x - x0) * sx)
    ix = (lambda X: np.interp(X, dst, src)) if src is not None else (lambda X: X / sx + x0)
    tf = lambda l: np.c_[fx(l[:, 0]), np.maximum(0, (ymax - l[:, 1]) * sy)]
    def shapes_of(lp):
        out_, hole_ = [l for l, h in lp if not h], [l for l, h in lp if h]
        return [(ccw(rdp(tf(o), 0.2)), [ccw(rdp(tf(h), 0.2), False) for h in hole_ if MPath(o).contains_point(h[0])])
                for o in sorted(out_, key=lambda l: -abs(area(l)))]
    shapes = shapes_of(lps)
    mid = (min(o[:, 0].min() for o, _ in shapes) + max(o[:, 0].max() for o, _ in shapes)) / 2
    if src is None and abs(mid) > 0.08 * L:
        # the feet found were not the screw posts (a shadow, a knot on a twig): centre on the overall length instead
        notes.append(f'feet gave an origin {mid:.1f} mm off the middle: centred on the overall length')
        x0 += mid / sx; shapes = shapes_of(lps)
    # bar thickness: top run of the column(s) at the middle
    xc = int(round(x0)); ts = []
    for c in range(xc - 3, xc + 4):
        r = runs(m[:, c])
        if r: ts.append((r[0][1] - r[0][0]) * sy)
    t = float(np.median(ts)) if ts else P / 3
    depth = prm.get('depth') or W or round(t, 2)
    if prm.get('bevel') is not None: bevel = prm['bevel']
    elif sec == 'round': bevel = 0.45 * min(depth, t)
    else: bevel = min(0.6, 0.25 * depth)
    bevel = min(bevel, depth / 2)
    ext = lambda o, hs, dp, bv, **kw: dict(kind='extrude', outline=[[r2(x), r2(y)] for x, y in o], depth=r2(dp), bevel=r2(min(bv, dp / 2)),
                                           **({'holes': [[[r2(x), r2(y)] for x, y in h] for h in hs]} if hs else {}), **kw)
    if accent is not None and accent.any():
        # two-tone: posts / collars carry the accent finish at full Width, the bar runs through them at its own thickness
        db = min(depth, t)
        parts = [ext(o, hs, db, 0.45 * db if sec == 'round' else bevel) for o, hs in shapes_of(loops_of(m & ~accent))]
        parts += [ext(o, hs, depth, bevel, mat='accent') for o, hs in shapes_of(loops_of(accent))]
    else:
        parts = [ext(o, hs, depth, bevel) for o, hs in shapes]
    shape = parts[0] if len(parts) == 1 else dict(kind='group', parts=[dict(p, offset=[0, 0, 0], rotate=[0, 0, 0]) for p in parts])
    inv = lambda q: np.c_[ix(q[:, 0]), ymax - q[:, 1] / sy]
    fit = raster([inv(o) for o, _ in shapes], [inv(h) for _, hs in shapes for h in hs], m.shape)
    return shape, iou(fit, m), dict(px=[inv(o) for o, _ in shapes] + [inv(h) for _, hs in shapes for h in hs]), t


def build_lathe(m, L, P, W, sec, notes):
    ys, xs = np.nonzero(m); top, bot = ys.min(), ys.max(); xc = (xs.min() + xs.max()) / 2
    rows = np.linspace(bot, top, 40).round().astype(int)
    hw = np.array([max(abs(np.flatnonzero(m[r]) - xc).max() + 0.5, 0.5) if m[r].any() else 0.5 for r in rows])
    s = (L / 2) / hw.max(); sy = P / (bot - top + 1)
    prof = [[r2(h * s), r2((bot - r) * sy)] for h, r in zip(hw, rows)] + [[0, r2(P)]]
    shape = dict(kind='lathe', profile=prof, section='square' if sec == 'flat' else 'round', stretch=[1, r2(W / L)], rotate=0)
    poly = np.r_[np.c_[xc + hw, rows], np.c_[xc - hw, rows][::-1]]
    fit = raster([poly], [], m.shape)
    return shape, iou(fit, m), dict(px=[poly])


def slab(region, sx, sz, x0, z0, y0, y1, bevel, mat=None):
    """Extrusions of a front-view (X x Z) pixel region spanning Y in [y0, y1]."""
    parts, px = [], []
    for o, h in loops_of(region):
        if h: continue
        hs = [l for l, hh in loops_of(region) if hh and MPath(o).contains_point(l[0])]
        tf = lambda l: np.c_[(l[:, 0] - x0) * sx, (l[:, 1] - z0) * sz]
        oo = ccw(rdp(tf(o), 0.2))
        if len(oo) < 3: continue
        d = y1 - y0
        p = dict(kind='extrude', outline=[[r2(x), r2(z)] for x, z in oo], depth=r2(d), bevel=r2(min(bevel, d / 2)),
                 offset=[0, r2((y0 + y1) / 2), 0], rotate=[90, 0, 0])
        if hs: p['holes'] = [[[r2(x), r2(z)] for x, z in ccw(rdp(tf(l), 0.2), False)] for l in hs]
        if mat: p['mat'] = mat
        parts.append(p); px.append(o); px += hs
    return parts, px


def hood(m, sx, sz, x0, z0, P, n, ex=2.5):
    """Cup hood over a front-view mask: n slices along X (denser at the ends), each an extruded quarter superellipse
    (exponent ex) from the column's top edge on the door (y = 0) to its bottom edge at the lip (y = projection), as a
    shell open underneath. A column's projection is P scaled by how tall it is against the tallest column."""
    has = m.any(0); cols = np.arange(m.shape[1])
    top = np.where(has, m.argmax(0), 0).astype(float); bot = np.where(has, m.shape[0] - m[::-1].argmax(0), 0).astype(float)
    h = np.where(has, bot - top, 0.0)
    run = max(runs(h >= 0.3 * h.max()), key=lambda r: r[1] - r[0])
    sm = lambda a: ndi.gaussian_filter1d(a, 2, mode='nearest')
    fill = lambda a: np.interp(cols, cols[has], a[has])   # empty columns would drag the smoothed edges to row 0
    top, bot, h = sm(fill(top)), sm(fill(bot)), sm(h); hm = h.max()
    xa, xb = (run[0] - x0) * sx, (run[1] - x0) * sx
    e = xa + (xb - xa) * (1 - np.cos(np.linspace(0, math.pi, n + 1))) / 2
    ph = np.linspace(0, math.pi / 2, 14); c, s_ = np.cos(ph) ** (2 / ex), np.sin(ph) ** (2 / ex)
    parts = []
    for a, b in zip(e[:-1], e[1:]):
        col = x0 + (a + b) / 2 / sx
        T, B = (np.interp(col, cols, top) - z0) * sz, (np.interp(col, cols, bot) - z0) * sz
        Pp = P * np.clip((np.interp(col, cols, h) / hm - 0.3) / 0.7, 0, 1) ** 0.5
        if B - T < 1 or Pp < 1: continue
        t = min(2.2, 0.3 * Pp, 0.3 * (B - T))
        outer = np.c_[B - (B - T) * c, Pp * s_]
        inner = np.c_[B - (B - T - t) * c, (Pp - t) * s_][::-1]
        poly = np.vstack([outer, inner]) * [-1, 1]   # slice outline in (-z, y): rotate [0, 90, 0] turns it back to +z
        parts.append(dict(kind='extrude', outline=[[r2(u), r2(v)] for u, v in ccw(poly)], depth=r2(b - a + 0.02), bevel=0,
                          offset=[r2((a + b) / 2), 0, 0], rotate=[0, 90, 0]))
    return parts


def build_front(mode, m, L, P, W, c2c, prm, notes):
    ys, xs = np.nonzero(m)
    Zx = P if mode == 'tab' else W
    if mode == 'tab' and not Zx: Zx = L * (np.ptp(ys) + 1) / (np.ptp(xs) + 1); notes.append('no Projection in the spec: tab height from the photo aspect')
    sx = L / (np.ptp(xs) + 1); sz = Zx / (np.ptp(ys) + 1)
    if abs(sx / sz - 1) > 0.1: notes.append(f'x/z photo scales disagree by {abs(sx / sz - 1) * 100:.0f}%; spec dims used')
    x0 = (xs.min() + xs.max() + 1) / 2 - 0.5
    sc = [int(round(x0 + s * (c2c or 0) / 2 / sx)) for s in (-1, 1)]
    zr = []
    for c in sc:
        r = runs(m[:, min(max(c, 0), m.shape[1] - 1)])
        if r: zr.append((r[0][0] + r[0][1] - 1) / 2)
    z0 = float(np.mean(zr)) if zr else (ys.min() + ys.max()) / 2
    if mode == 'tab': z0 = ys.min() - 0.5
    d = ndi.distance_transform_edt(m, sampling=(sz, sx))
    parts, px = [], []
    if mode == 'cup':
        pp, pl = slab(m, sx, sz, x0, z0, 0, 1.6, 0.5); parts += pp; px += pl
        parts += hood(m, sx, sz, x0, z0, P, prm.get('levels', 96))
        notes.append('cup: front silhouette as a 1.6 mm plate, hood shell sliced along X, open underneath')
    elif mode in ('drop', 'standoff', 'plate'):
        yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
        disc = np.zeros_like(m); rp = []
        for c in sc:
            r_mm = float(np.clip(d[int(round(z0)), min(max(c, 0), m.shape[1] - 1)], 3, 10)) if c2c else 4
            rp.append(r_mm); disc |= ((xx - c) * sx) ** 2 + ((yy - z0) * sz) ** 2 <= r_mm ** 2
        if mode == 'plate':
            t = 3.0
            pp, pl = slab(m, sx, sz, x0, z0, 0, t, 0.8); parts += pp; px += pl
            ring = (((xx - x0) * sx / (0.36 * L)) ** 2 + ((yy - z0) * sz / (0.62 * W)) ** 2 <= 1) & \
                   (((xx - x0) * sx / (0.36 * L - 4.5)) ** 2 + ((yy - z0) * sz / (0.62 * W - 4.5)) ** 2 > 1) & (yy >= z0)
            pp, _ = slab((ring | disc) & m, sx, sz, x0, z0, t, t + 4.5, 2.0); parts += pp
            pp, _ = slab(disc & m, sx, sz, x0, z0, t, P, 1.5); parts += pp
            notes.append('Mission plate: plate + bail ring + posts built parametrically over the front silhouette')
        else:
            body = m & ~disc
            tb = float(np.clip(2 * np.percentile(d[body], 90), 3, P * 0.5)) if body.any() else 3
            y0, y1 = (P - tb, P) if mode == 'standoff' else (0, tb)
            pp, pl = slab(body | (disc & m), sx, sz, x0, z0, y0, y1, 0.45 * tb); parts += pp; px += pl
            pp, pl = slab(disc, sx, sz, x0, z0, 0, P, 0.4 * min(rp)); parts += pp
            notes.append(f'{mode}: posts r~{min(rp):.1f} mm at the screws to Projection, body {tb:.1f} mm thick at Y {y0:.1f}..{y1:.1f}')
    else:  # tab
        t = W or 1.59
        pp, pl = slab(m, sx, sz, x0, z0, 0, t, min(0.6, t / 2)); parts += pp; px += pl
        notes.append('edge tab: front plate only; origin = middle of the drawer top edge, plate hangs along +Z by Projection, Width is the plate thickness')
    fit = np.zeros_like(m)
    for p in parts:
        if p.get('rotate') != [90, 0, 0]: continue   # hood slices stand across the face; the plate carries the silhouette
        o = np.array(p['outline']); fit |= raster([np.c_[o[:, 0] / sx + x0, o[:, 1] / sz + z0]],
                                                [np.c_[np.array(h)[:, 0] / sx + x0, np.array(h)[:, 1] / sz + z0] for h in p.get('holes', [])], m.shape)
    return dict(kind='group', parts=parts), iou(fit, m), dict(px=px)


# ---------------------------------------------------------------- per size / product

def stretch_side(shape, Lo, Po, L, P, W, depth_from_spec=True):
    """Lengthen a bar pull outline between its posts (ends keep their shape) and rescale the projection."""
    xs = [x for q in flat_polys(shape, False) for x in q[:, 0]]; a, b = min(xs), max(xs)
    dl = (L - (b - a)) / 2; c = 0.25 * (b - a)
    fx = (lambda x: float(np.interp(x, [a, -c, c, b], [a - dl, -(c + dl), c + dl, b + dl]))) if c + dl > 0.1 * c else (lambda x: x * L / (b - a))
    def f(s):
        s = dict(s)
        if s['kind'] == 'group': s['parts'] = [f(p) for p in s['parts']]; return s
        s['outline'] = [[r2(fx(x)), r2(y * P / Po)] for x, y in s['outline']]
        if 'holes' in s: s['holes'] = [[[r2(fx(x)), r2(y * P / Po)] for x, y in h] for h in s['holes']]
        if depth_from_spec: s['bevel'] = r2(min(s['bevel'] * W / s['depth'], W / 2)); s['depth'] = r2(W)
        return s
    return f(shape)


def scale_shape(shape, kx, ky, kz):
    def f(s):
        s = dict(s)
        if s['kind'] == 'group': s['parts'] = [f(p) for p in s['parts']]; return s
        if s['kind'] == 'lathe': s['profile'] = [[r2(r * kx), r2(y * ky)] for r, y in s['profile']]; return s
        rot = s.get('rotate', [0, 0, 0]) == [90, 0, 0]
        a, b, d = (kx, kz, ky) if rot else (kx, ky, kz)
        s['outline'] = [[r2(x * a), r2(y * b)] for x, y in s['outline']]
        if 'holes' in s: s['holes'] = [[[r2(x * a), r2(y * b)] for x, y in h] for h in s['holes']]
        s['depth'] = r2(s['depth'] * d); s['bevel'] = r2(min(s['bevel'] * min(a, b, d), s['depth'] / 2))
        if 'offset' in s: s['offset'] = [r2(s['offset'][0] * kx), r2(s['offset'][1] * ky), r2(s['offset'][2] * kz)]
        return s
    return f(shape)


def dims(sz):
    """Spec dims with the catalogue's swapped / copied values repaired."""
    L, P, W, c2c = sz['length_mm'], sz['proj_mm'], sz['width_mm'], sz['c2c_mm']; notes = []
    if L and c2c and L < c2c and W and W >= c2c:
        L, W = W, L; notes.append('spec Overall Length < c2c and Width > c2c: swapped length and width')
    return L, P, W, c2c, notes


def accent_mask(rel, m, finish):
    """Posts of a two-tone bar pull: split by brightness when the reference SKU is two-tone ('A and B': B is the
    posts), else take the columns of the posts' feet (a single-finish photo has nothing to split by colour)."""
    lum = np.asarray(Image.open(TK / rel).convert('L')).astype(float)
    if ' and ' in finish:
        v = lum[m]; t = np.mean(v)
        for _ in range(10): t = (v[v <= t].mean() + v[v > t].mean()) / 2  # 2-means threshold
        a = m & ((lum < t) if finish.split(' and ')[1] == 'Flat Black' else (lum >= t))
    else:
        rows = np.flatnonzero(m.any(1)); lo = rows[0] + int(0.55 * (rows[-1] - rows[0]))
        cols = np.zeros(m.shape[1], bool)
        for c0, c1 in runs(m[lo:].any(0)): cols[max(c0 - 2, 0):c1 + 2] = True
        a = m & cols[None, :]
    # colour only says where the posts are: take whole columns there (a collar wraps the bar; shading splits a post)
    cols = ndi.binary_closing(ndi.binary_opening(a, iterations=1).sum(0) >= 0.5 * np.maximum(m.sum(0), 1), structure=np.ones(7), border_value=0)
    lab, n = ndi.label(cols)
    if n:  # keep the post-sized bands, not specks
        w = ndi.sum(cols, lab, range(1, n + 1)); cols = np.isin(lab, 1 + np.flatnonzero(w >= 0.25 * w.max()))
    rows = np.flatnonzero(m.any(1)); lo = rows[0] + int(0.55 * (rows[-1] - rows[0]))
    for c0, c1 in runs(m[lo:].any(0)):  # a post's base is wider than its collar: widen each band to its foot
        if cols[c0:c1].any(): cols[max(c0 - 2, 0):c1 + 2] = True
    return m & cols[None, :]


def parametric(mode, L, P, W, c2c, sec, notes):
    """Shapes the photos cannot give (no usable elevation): a plain D pull, or a top-edge tab with a front barrel."""
    if mode == 'dpull':
        pw = max(L - c2c, 0.25 * P, 6); t = 0.3 * P; d = min(W, 12.7) if W else 12.7
        bv = r2(0.45 * min(d, t) if sec == 'round' else 0.6)
        rect = lambda x0, x1, y0, y1: [[r2(x0), r2(y0)], [r2(x1), r2(y0)], [r2(x1), r2(y1)], [r2(x0), r2(y1)]]
        if W and W > 1.6 * d:
            # Width much larger than the bar: an offset handle, posts on the screws (Z = 0) and the grip shifted down the
            # door (+Z) by Width - bar, as the room shots show
            zo = r2((W - d) / 2)
            parts = [dict(kind='extrude', outline=rect(sg * L / 2 - (pw if sg > 0 else 0), sg * L / 2 + (0 if sg > 0 else pw), 0, P - t),
                          depth=r2(d), bevel=bv, offset=[0, 0, 0], rotate=[0, 0, 0]) for sg in (-1, 1)]
            parts.append(dict(kind='extrude', outline=rect(-L / 2, L / 2, P - t, P), depth=r2(d), bevel=bv, offset=[0, 0, r2(2 * zo)], rotate=[0, 0, 0]))
            parts += [dict(kind='extrude', outline=rect(-W / 2, W / 2, P - t, P), depth=r2(pw), bevel=bv,
                           offset=[r2(sg * (L - pw) / 2), 0, zo], rotate=[0, 90, 0]) for sg in (-1, 1)]
            notes.append(f'parametric offset D pull (photos are perspective only): posts at Z 0, grip at Z {r2(2 * zo)} (+Z = down the door)')
            return dict(kind='group', parts=parts)
        o = [[-L / 2, 0], [-L / 2 + pw, 0], [-L / 2 + pw, P - t], [L / 2 - pw, P - t], [L / 2 - pw, 0], [L / 2, 0], [L / 2, P], [-L / 2, P]]
        notes.append(f'parametric D pull (photos are perspective only); depth {d} mm')
        return dict(kind='extrude', outline=[[r2(x), r2(y)] for x, y in o], depth=r2(d), bevel=bv)
    # edge: plate lying on the drawer's top edge, overhanging the front by Projection, with a barrel of Width under its lip
    r = W / 2; th = 1.6
    plate = dict(kind='extrude', outline=[[r2(-L / 2), 0], [r2(L / 2), 0], [r2(L / 2), r2(P)], [r2(-L / 2), r2(P)]], depth=th, bevel=0.5,
                 offset=[0, 0, -th / 2], rotate=[0, 0, 0])
    circ = [[r2(r * math.cos(a)), r2(r * math.sin(a))] for a in np.linspace(0, 2 * math.pi, 33)[:-1]]
    barrel = dict(kind='extrude', outline=circ, depth=r2(L * 0.94), bevel=0.3, offset=[0, r2(P - r), r2(r - th)], rotate=[0, 90, 0])
    notes.append('parametric edge tab: origin = middle of the drawer top edge at the front; plate overhangs along +Y, barrel below its lip')
    return dict(kind='group', parts=[plate, barrel])


def fit_size(p, sz, mode, sec, prm):
    L, P, W, c2c, notes = dims(sz); fix = {}
    if L != sz['length_mm']: fix.update(length_mm=L, width_mm=W)   # swapped in the catalogue
    if mode in ('dpull', 'edge'):
        dbg = dict(img=sz['ref_img'][0], n=0, front=False, px=[]) if sz['ref_img'] else None
        return dict(shape=parametric(mode, L, P, W, c2c, sec, notes), fit=dict(iou=None, photo=None, notes='; '.join(notes))), dbg
    if mode == 'tab': prm = dict(prm, photo=prm.get('photo', 0))  # _0 is the face; _1 a 3/4 view
    view = prm.get('view', 'front' if mode in FRONT else 'side')
    want = L / ((P if view == 'side' or mode == 'tab' else W) or L * 0.4) if L else 0
    cands = candidates(sz, prm, want) if L else []
    if not cands: return None
    err, n, rel, m = cands[0]
    err -= 0 if n == 0 else 0.15
    if err > 0.25: notes.append(f'best photo aspect is {math.exp(err) * 100 - 100:.0f}% off the spec {view} view')
    dbg = dict(img=rel, n=n, front=mode in FRONT)
    if mode == 'side':
        if W and W >= 0.9 * L:
            notes.append(f'spec Width {W} >= Overall Length looks wrong: depth taken from the bar thickness'); W = None
        acc = accent_mask(rel, m, sz['ref_finish']) if p['slug'] in ACCENT else None
        shape, q, d, t = build_side(m, L, P, W, c2c, sec, prm, notes, fix, acc)
    elif mode == 'lathe':
        shape, q, d = build_lathe(m, L, P, W or L, sec, notes)
    else:
        shape, q, d = build_front(mode, m, L, P, W, c2c, prm, notes)
    dbg.update(d)
    if mode == 'side' and (W is None or prm.get('depth')): fix['width_mm'] = r2(prm.get('depth') or t)
    fit = dict(iou=round(q, 3), photo=Path(rel).name, notes='; '.join(notes))
    if fix: fit['spec_fix'] = fix  # dims used instead of a spec value judged wrong (see notes)
    return dict(shape=shape, fit=fit), dbg


def euler(r):
    """three.js Euler 'XYZ' (degrees) -> rotation matrix Rx @ Ry @ Rz."""
    x, y, z = np.radians(r); cx, sx_, cy, sy_, cz, sz_ = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return (np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]]) @ np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
            @ np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]]))


def solids(shape, R=np.eye(3), o=np.zeros(3)):
    """[(points (N,3) of the solid's hull, mid-depth outline (M,3), mat)] in the pull's frame, ignoring bevel growth."""
    rt = shape.get('rotate', [0, 0, 0]); o = o + R @ np.array(shape.get('offset', [0, 0, 0]), float)
    R = R @ euler(rt if isinstance(rt, list) else [0, 0, 0])
    if shape['kind'] == 'group':
        return [q for part in shape['parts'] for q in solids(part, R, o)]
    if shape['kind'] == 'lathe':
        pr = np.array(shape['profile'], float); sxz = shape.get('stretch', [1, 1]); t = np.linspace(0, 2 * np.pi, 33)[:-1]
        ring = np.c_[np.cos(t), np.sin(t)]
        if shape.get('section') == 'square': ring = ring / np.abs(ring).max(1, keepdims=True)
        pts = np.array([[r * c * sxz[0], y, r * s_ * sxz[1]] for r, y in pr for c, s_ in ring])
        mid = np.r_[np.c_[pr[:, 0] * sxz[0], pr[:, 1], 0 * pr[:, 0]], np.c_[-pr[::-1, 0] * sxz[0], pr[::-1, 1], 0 * pr[:, 0]]]
        rot = euler([0, shape.get('rotate', 0), 0]) if not isinstance(shape.get('rotate', 0), list) else np.eye(3)
        return [((R @ rot @ pts.T).T + o, (R @ rot @ mid.T).T + o, shape.get('mat'))]
    ol = np.array(shape['outline'], float); d = shape['depth'] / 2
    pts = np.r_[np.c_[ol, np.full(len(ol), -d)], np.c_[ol, np.full(len(ol), d)]]
    mid = np.c_[ol, np.zeros(len(ol))]
    return [((R @ pts.T).T + o, (R @ mid.T).T + o, shape.get('mat'))]


def extents(shape):
    """(min xyz, max xyz) of a SHAPE."""
    a = np.vstack([q for q, _, _ in solids(shape)]); return a.min(0), a.max(0)


def product(p, ov):
    prm = dict(ov.get('params', {}))
    mode = prm.get('mode') or mode_of(p); sec = section_of(p, prm)
    out = dict(slug=p['slug'], type=ov.get('type') or prm.get('type') or type_of(p, mode, sec), sizes={}, notes='')
    dbg, pending = {}, []
    for sz in p['sizes']:
        so = ov.get('sizes', {}).get(sz['key'], {})
        if 'shape' in so:
            fit = dict(iou=None, notes='; '.join(filter(None, ['override shape', so.get('notes')])), **({'spec_fix': so['spec_fix']} if so.get('spec_fix') else {}))
            out['sizes'][sz['key']] = dict(shape=so['shape'], fit=fit); continue
        r = fit_size(p, sz, so.get('params', {}).get('mode', mode), sec, dict(prm, **so.get('params', {})))
        if r:
            out['sizes'][sz['key']] = r[0]
            if r[1]: dbg[sz['key']] = r[1]
        else: pending.append(sz)
    for sz in pending:  # borrow the nearest fitted size's silhouette
        L, P, W, c2c, _ = dims(sz)
        if not out['sizes'] or not L: out['sizes'][sz['key']] = None; continue
        near = lambda k: (lambda z: abs((z['proj_mm'] or 0) - (P or 0)) / (P or 1) + 0.3 * abs(z['length_mm'] - L) / L)(
            next(z for z in p['sizes'] if z['key'] == k))
        src = min((k for k in out['sizes'] if out['sizes'][k] and k in dbg), default=None, key=near)
        if src is None: continue
        zs = next(z for z in p['sizes'] if z['key'] == src); Lo, Po, Wo, _, _ = dims(zs)
        s = out['sizes'][src]['shape']
        if mode == 'side': s = stretch_side(s, Lo, Po, L, P or Po, W or Wo)
        else: s = scale_shape(s, L / Lo, (P or Po or 1) / (Po or P or 1), (W or Wo or 1) / (Wo or W or 1))
        out['sizes'][sz['key']] = dict(shape=s, fit=dict(iou=out['sizes'][src]['fit']['iou'], photo=None,
                                                          notes=f'no usable photo: silhouette borrowed from {src}'))
        dbg[sz['key']] = dict(dbg[src], borrowed=src)
    missing = [k for k, v in out['sizes'].items() if v is None]
    out['sizes'] = {k: v for k, v in out['sizes'].items() if v}
    # dims check against the (repaired) spec
    for sz in p['sizes']:
        v = out['sizes'].get(sz['key'])
        if not v or v['fit'].get('notes') == 'override shape': continue
        L, P, W, _, _ = dims(sz); lo, hi = extents(v['shape']); e = hi - lo
        fx = v['fit'].get('spec_fix', {}); P = fx.get('proj_mm', P); W = fx.get('width_mm', W)
        bad = []
        if mode != 'tab':
            for nm, got, want in (('length', e[0], L), ('projection', e[1], P), ('width', e[2], W)):
                if want and abs(got - want) / want > 0.02: bad.append(f'{nm} {got:.1f} vs {want}')
        if lo[1] < -0.01: bad.append(f'min Y {lo[1]:.2f}')
        if bad: v['fit']['notes'] = '; '.join(filter(None, [v['fit']['notes'], 'DIM CHECK: ' + ', '.join(bad)]))
    out['notes'] = '; '.join(filter(None, [ov.get('notes', ''), f'mode {mode}, section {sec}', '+Z = down the door' if mode in FRONT else '',
                                           f'no photo for {missing}' if missing else '']))
    return out, dbg


# ---------------------------------------------------------------- QA contact sheets

def flat_polys(shape, front, with_mat=False):
    """Outlines of a SHAPE projected on X x Y (side) or X x Z (front), in mm: a part seen edge-on draws as its box."""
    out, mats = [], []
    for pts, mid, mat in solids(shape):
        q = mid[:, [0, 2 if front else 1]]
        if abs(area(q)) < 1e-3 * max(1, np.ptp(pts[:, 0]) ** 2):
            p2 = pts[:, [0, 2 if front else 1]]; lo, hi = p2.min(0), p2.max(0)
            q = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        out.append(q); mats.append(mat)
    return (out, mats) if with_mat else out


def sheet(items, path, cols=4, sz=300):
    """items: (label, photo, [pixel polylines], shape, front) -> photo with the fitted outline in red, and under it the
    SHAPE drawn in mm at true proportions (side elevation, or the face for front-view modes) for a dims / distortion check."""
    rows = (len(items) + cols - 1) // cols; hh = sz + 26 + sz // 2
    W = Image.new('RGB', (cols * sz, rows * hh), 'white'); dr = ImageDraw.Draw(W)
    for i, (lab, img, polys, shape, front) in enumerate(items):
        x, y = (i % cols) * sz, (i // cols) * hh
        im = Image.open(TK / img).convert('RGB'); d = ImageDraw.Draw(im)
        for pl in polys: d.line([tuple(q) for q in np.r_[pl, pl[:1]]], fill=(255, 0, 0), width=2)
        W.paste(im.resize((sz, sz)), (x, y + 26))
        for j, t in enumerate(lab.split('\n')[:2]): dr.text((x + 3, y + 1 + 12 * j), t[:48], fill='black')
        ps, ms = flat_polys(shape, front, True); a = np.vstack(ps); lo, hi = a.min(0), a.max(0)
        k = min((sz - 20) / max(hi[0] - lo[0], 1), (sz // 2 - 20) / max(hi[1] - lo[1], 1))
        for pl, mt in zip(ps, ms):
            q = [(x + 10 + (u - lo[0]) * k, y + 26 + sz + 10 + ((hi[1] - v) if not front else (v - lo[1])) * k) for u, v in pl]
            dr.polygon(q, outline=(230, 120, 0) if mt else (0, 90, 200))
        dr.text((x + 3, y + hh - 12), f'{hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} mm', fill=(0, 90, 200))
    W.save(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only'); ap.add_argument('--sheets', action='store_true')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    P = [p for p in json.load(open(TK / 'products.json'))['products'] if p['kind_hint'] == 'pull' and (not only or p['slug'] in only)]
    GEOM.mkdir(exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
    report, items = [], []
    for p in P:
        ov = json.load(open(OVR / f"{p['slug']}.json")) if (OVR / f"{p['slug']}.json").exists() else {}
        g, dbg = product(p, ov)
        (GEOM / f"{p['slug']}.json").write_text(json.dumps(g, separators=(',', ':')))
        for k, v in g['sizes'].items():   # override and parametric shapes have no photo IoU: listed last, not counted as 1
            report.append((v['fit']['iou'] if v['fit']['iou'] is not None else math.inf, p['slug'], k, v['fit'].get('notes', '')))
        for k in list(dbg) if only else list(dbg)[:1] + list(dbg)[-1:] if len(dbg) > 1 else list(dbg):  # --only: every size
            v = g['sizes'][k]['fit']
            items.append((f"{p['slug']} {k}\niou {v['iou']} _{dbg[k]['n']} {g['type']} {g['notes'][5:14]}", dbg[k]['img'], dbg[k]['px'],
                          g['sizes'][k]['shape'], dbg[k]['front']))
    report.sort()
    fitted = [r[0] for r in report if r[0] != math.inf]
    print(f'{len(P)} products, {len(report)} sizes; iou median {np.median(fitted) if fitted else float("nan"):.3f} over {len(fitted)} photo-fitted sizes')
    for r in report[:40]: print(f'  {r[0]:.3f} {r[1]} {r[2]} {r[3][:150]}')
    (QA / 'report.txt').write_text('\n'.join(f'{"-" if r[0] == math.inf else f"{r[0]:.3f}"}\t{r[1]}\t{r[2]}\t{r[3]}' for r in report) + '\n')
    if a.sheets:
        for f in QA.glob('sheet_*.png'): f.unlink()
        for i in range(0, len(items), 24): sheet(items[i:i + 24], QA / f'sheet_{i // 24:02d}.png')


if __name__ == '__main__':
    main()
