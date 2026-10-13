"""MSI subway / backsplash tile -> studio backsplash variants.

Input   materials/msi_tile/source/msi_subway_tile_full.json (+ _skus.csv / _products.csv): scrape of
        msisurfaces.com/backsplash-tile/subway-tile/ (2026-10-13), 238 product pages, 546 SKU listings = 263 unique SKUs.
Output  materials/msi_tile/tiles.json            one entry per unique SKU (see ENTRY below)
        viewer/textures/msitile_<slug>.js          colour map, JSONP window.__texData(id, JPEG data URL) like build_studio
        viewer/textures/msitile_<slug>_n.js        tangent-space normal map (grout recess, bevels, handmade relief)
        materials/msi_tile/qa/sheet_NN.jpg         contact sheets, 12 SKUs each: photo + analysis overlay (red courses, blue
                                                   joints / red crop box) | texture tiled 3x3 at true scale + 6 in ruler |
                                                   flat lit swatch + swatch colour + UI thumb | 2x2 wrap-seam centre at 1:1
        materials/msi_tile/qa/rows/<SKU>.jpg       the same row per SKU;  qa/only_NN.jpg for an --only run
        materials/msi_tile/cache/ (not in git)     downloaded photos (img/) and product pages (pages/); refilled on demand

Run (python 3.10 with numpy, scipy, PIL):
        python3 tools/msi_tiles.py                 everything, ~10 min (downloads what the cache lacks: <= 4 requests at a
                                                   time, retries with backoff, 403/404/410 cached as <file>.miss)
        python3 tools/msi_tiles.py --only SKU[,SKU]   just these SKUs (tiles.json entries for other SKUs are kept)
        python3 tools/msi_tiles.py --fetch         only fill the download cache

How a texture is made
  photos    the SKU's own pictures, best first: sku image, product main image, the product page's gallery (variation /
            detail / colornames / skus pictures whose file name names this product - siblings' pictures, 3D-mix or
            bevel or trim variants are skipped), thumbnail; room scenes only as a last resort -> kind "scene", unusable.
            Each photo is tried in turn (<= 5) and the one whose grout pattern fits the nominal size best is kept.
  classify  trim (bullnose, quarter round, corner, pencil, liner, chair rail, molding) -> kind "trim", unusable.
            One piece on white -> "synth": field laid in running bond at the nominal size, 1/16 in warm grey grout,
            per-tile variation from different windows of the piece (rejected if the piece's aspect is not the nominal's).
            Rectangular tile -> "cells"; hexagon / penny / octagon / arabesque / picket / herringbone / scallop /
            teardrop / weave / checker / patterns -> "lattice".  kind "mosaic" = sold on a sheet, else "field".
  cells     reject 45 deg layouts (edge orientation); deskew (projection-profile sharpness); every clear peak of the
            row autocorrelation of the horizontal-edge energy (full and coarse scale, both orientations) is a candidate
            course pitch c; for each, joints per course (vertical edges spanning the course) give the joint pitch p
            (spacing cluster nearest the nominal length, or the 1/2-1/3-1/4 bond offsets between neighbouring courses,
            or c x nominal aspect when no joint is visible). Candidates are scored on autocorrelation strength, line
            continuity, p/c vs the nominal aspect, lines in between courses (ribs / multiples), joints the pitch
            predicts but the photo lacks, and a plausible photo width. Long tile axis -> texture X.
            mm/px = nominal (tile + grout) / measured pitch per axis; > 10 % disagreement noted, > 25 % unusable.
            Grout width: narrow dip at the course lines (bevels excluded); for clay / brick the mortar width solved
            from both axes, (L+g)/p = (S+g)/c. Every tile is a cell from grout centre to grout centre; whole tiles,
            end pieces joined across the picture edge (feathered, gain-matched, stretched <= 12 %, 60 % on plain
            glaze) and, for tiles longer than the photo, start + interior + end sections. The texture is n tiles x m
            courses (>= ~600 x 400 mm, <= 2048 px, >= 2 px/mm) laid in the measured bond (stack, 1/2, 1/3, 1/4 or the
            photo's own offsets; "stack" in the name wins), unique tiles first then a seeded shuffle of mirrored
            copies, so every wrap seam falls on a grout centre and the period is a whole number of tiles.
  lattice   chip lattice from the 2-D autocorrelation of the (grain-smoothed) edge map: reduced basis -> shortest
            axis-aligned repeat; the colour pattern's repeat is the smallest multiple that correlates (> 0.6 for
            two-colour patterns, else a random blend uses the chip repeat). Crop = whole repeats; the wrap seams are
            minimum-difference cuts (along grout) against the lattice-equivalent pixels one span away, with matching
            ends, so it tiles exactly. Scale: hexagon / penny / octagon / checker from the chip pitch and the nominal
            chip size; patterns MSI gives no size for from photos of the same series (collection, folder, pixel size)
            measured on known-size SKUs (span of the photo in inches; series spread <= 25 %) - "series_scale".
  colour    lighting gradient divided out (low-order fit of log luminance on tile pixels; per-tile variation kept),
            glossy glints and window reflections compressed per tile, exposure: white / cream glaze to 0.80 linear at
            the 90th percentile (matte / rustic whites 0.70), other colours only capped at 0.85.
  normal    height = grout recess + edge profile (pillowed, beveled, inverted bevel, convex, fluted) + fine relief
            from the photo's band-passed luminance; slopes in mm/mm, so relief 1.0 = physical.

materials/msi_tile/overrides/<SKU>.json (all keys optional):
  image       URL, or index into the candidate photo list (0 = first)
  crop        [x0, y0, x1, y1] source pixels (after the <= 1600 px resize) to analyse
  rotate      extra degrees (counter-clockwise) before analysis;  rotate90: true/false forces the course orientation
  mode        "cells" | "lattice" | "synth" | "picket" (picket layout built at the nominal size when one picket spans the
              photo: columns with nesting 45 deg points, faces from the photographed picket bodies)
  piece       synth from a laid-up photo: [x0, y0, x1, y1] of one whole tile (after rotate) instead of a piece on white
  kind        "field" | "mosaic" | "trim" | "scene"
  usable      true/false;  reason: text
  size        [L, S] tile face in inches (long, short) - replaces the parsed nominal size
  period_px   [px, py] repeat in photo px (cells: joint pitch, course pitch; lattice: Tx, Ty);  chip_px: lattice chip
              repeat [x, y];  grout_px: grout width in photo px
  bond        "running" | "stack" | "herringbone" | "mosaic" | "pattern";  offset: course offset fraction (0.5, 1/3 ..)
  grout_mm    grout width;  n / m: tiles per course / courses in the texture (lattice: repeats in x / y)
  exposure    multiplier on the final linear colour;  white: true/false (treat as white glaze for exposure)
  relief      normal-map strength;  profile: "flat" | "pillow" | "bevel" | "bigbevel" | "invbevel" | "convex" | "fluted"
  px_per_in   lattice scale override (photo pixels per inch)
  notes       appended to the entry's notes

ENTRY (tiles.json): id, sku, name, product, url, collection, color, material, finish, style, size_label,
  tile_mm [w, h] (nominal face, long side = w), kind, usable, reason, tex {id, period_mm [w, h] (= the texture's
  physical size), bond, grout_mm, long_axis "x", offset, course_offsets, normal, px}, rough, clearcoat, relief, swatch,
  thumb (84x56 JPEG data URL), scale_method, scale_check {mm_per_px_x/y, disagree, pitch_px, fit_score, ...}, photo,
  notes.
"""
import json, io, os, re, sys, time, html, math, csv, base64, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont, ImageOps

R = Path(__file__).resolve().parent.parent
MT = R / 'materials/msi_tile'
SRC = json.load(open(MT / 'source/msi_subway_tile_full.json'))
CACHE = MT / 'cache'; (CACHE / 'img').mkdir(parents=True, exist_ok=True); (CACHE / 'pages').mkdir(exist_ok=True)
TEX = R / 'viewer/textures'; QA = MT / 'qa'; OVR = MT / 'overrides'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36'
BASE = 'https://www.msisurfaces.com'
IN = 25.4
_sem = threading.Semaphore(4)

# ------------------------------------------------------------------ fetch (cached, polite)
def _get(url, fn, binary=True):
    """Cached GET with retries/backoff; returns bytes or None (a 404 is cached as an empty .miss file)."""
    if fn.exists(): return fn.read_bytes()
    miss = fn.with_suffix(fn.suffix + '.miss')
    if miss.exists(): return None
    for a in range(4):
        try:
            with _sem:
                r = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=40)
                b = r.read(); time.sleep(0.15)
            fn.write_bytes(b); return b
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410): miss.write_text(str(e.code)); return None
            time.sleep(2 * 2 ** a)
        except Exception:
            time.sleep(2 * 2 ** a)
    return None

def img_file(url):
    url = url.strip()
    return CACHE / 'img' / (re.sub(r'[^A-Za-z0-9.]+', '_', url.split('msisurfaces.com/')[-1])[-150:])

def fetch_img(url):
    url = html.unescape(url.strip())
    if not url or not url.lower().split('?')[0].endswith(('.jpg', '.jpeg', '.png', '.webp')): return None
    return _get(url.replace(' ', '%20').replace('&', '%26'), img_file(url))

def fetch_page(path):
    fn = CACHE / 'pages' / (re.sub(r'[^A-Za-z0-9]+', '_', path).strip('_')[:150] + '.html')
    b = _get(BASE + path, fn)
    return b.decode('utf-8', 'replace') if b else ''

def page_images(path):
    """Product-page pictures, best first: og:image, the colornames variation/main shots, mosaics/porcelain pictures;
    room scenes last (they make the SKU a "scene")."""
    h = fetch_page(path)
    urls = re.findall(r'<meta[^>]+og:image[^>]+content="([^"]+)"', h)
    urls += re.findall(r'https?://cdn\.msisurfaces\.com/images/[^"\' )>]+?\.(?:jpg|jpeg|png|webp)', h)
    urls = [html.unescape(u) for u in urls]
    bad = ('thumb', 'miscellaneous', 'roomvo', 'flyers', '/svg/', 'videos', '/iso/', '/edge/', 'vignette', 'privacy')
    def rank(u):
        if 'roomscenes' in u: return 9
        if '/variation/' in u: return 0
        return 1 if '/colornames/' in u else 2
    out = []
    for u in sorted(dict.fromkeys(urls), key=rank):
        if not any(b in u.lower() for b in bad): out.append(u)
    return out

# ------------------------------------------------------------------ catalogue
def catalogue():
    """One record per unique SKU: the SKU line, the product page it belongs to (its own page when it has one) and
    every product that lists it (for specs a sibling-only SKU lacks)."""
    prods = SRC['products']; byurl = {BASE + p['path']: p for p in prods}
    listed = {}
    for p in prods:
        for s in p['skus']: listed.setdefault(s['sku'], []).append((p, s))
    pcsv = {r['product_url']: r for r in csv.DictReader(open(MT / 'source/msi_subway_tile_products.csv'))}
    out = []
    for r in csv.DictReader(open(MT / 'source/msi_subway_tile_skus.csv')):
        sku = r['sku']; cands = listed.get(sku, [])
        own = byurl.get(r['product_url'])
        p, s = next(((p, s) for p, s in cands if p is own), cands[0])
        sp = dict(p['specs']); pc = pcsv.get(BASE + p['path'], {})
        for q, _ in cands:      # fill missing specs from siblings
            for k, v in q['specs'].items(): sp.setdefault(k, v)
        cols = [c for c in p.get('collections', []) if 'main grid' not in c]
        out.append(dict(sku=sku, item=s['item_name'], size=s['size'], finish=s['finish'] or sp.get('Available Finishes', ''),
                        product=p['name'], path=p['path'], url=BASE + p['path'], desc=p.get('description', ''),
                        collection=(cols[0] if cols else sp.get('Collection', '')) or '', specs=sp,
                        color=sp.get('Primary Color(s)', ''), material=sp.get('Material Type', '') or p.get('ld_material', ''),
                        style=sp.get('Style', ''), chips=pc.get('chips_per_sheet', ''), sheet=pc.get('sheet_size', ''),
                        images=[s['image'], p['main_image'], p['thumbnail']], own=(p is own)))
    return out

GENERIC = {'tile', 'tiles', 'subway', 'glossy', 'matte', 'polished', 'honed', 'satin', 'porcelain', 'ceramic', 'mosaic',
           'mosaics', 'glass', 'wall', 'brick', 'variation', 'detail', 'two', 'three', 'with', 'and', 'the', 'size', 'medium'}
DIST = {'mix', 'bevel', 'inverted', 'bull', 'quarter', 'picket', 'herringbone', 'chevron', 'basketweave', 'checkered', 'stack'}
DN = {'beveled': 'bevel', 'bullnose': 'bull', 'nose': 'bull', 'weave': 'basketweave', 'pickett': 'picket'}
def _stem(u): return re.sub(r'\.(jpe?g|png|webp)$', '', u.split('?')[0].rsplit('/', 1)[-1].lower())

def photo_urls(rec, ov):
    """Candidate photos, best first: sku image, product main image, the product page's own pictures (variation /
    detail / colornames; pictures of sibling products on the same page are skipped), thumbnail, own room scenes."""
    if isinstance(ov.get('image'), str): return [ov['image']]
    first = [html.unescape(u.strip()) for u in rec['images'][:2] if u and u.strip()]
    thumb = [html.unescape(u.strip()) for u in rec['images'][2:] if u and u.strip()]
    stems = {re.sub(r'-(variation|detail.*)$', '', _stem(u)) for u in first[:1]}
    m = re.search(r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)', rec['item'] + ' ' + rec['size'])
    size_tok = {f'{m.group(1)}x{m.group(2)}', f'{m.group(2)}x{m.group(1)}'} if m else set()
    size_tok |= {t.replace('.', '') for t in size_tok}             # '2.5x13' is spelled '25x13' in file names
    words = {w for w in re.findall(r'[a-z]{3,}', (rec['product'] + ' ' + rec['item']).lower()) if w not in GENERIC}
    coll = set(re.findall(r'[a-z]{3,}', (rec['collection'] or '').lower()))
    core = {DN.get(w, w) for w in re.findall(r'[a-z]{3,}', (rec['product'] + ' ' + rec['item']).lower())} & DIST
    def own(u):
        st = re.sub(r'-(variation|detail.*)$', '', _stem(u))
        if re.search(r'bull-?nose|quarter-?round|pencil|liner|chair-?rail|corner|-bn\b|molding', st): return False
        toks = set(re.findall(r'[a-z]{3,}', st)); sz = {t.replace('-', 'x') for t in re.findall(r'\d+(?:\.\d+)?[x-]\d+(?:\.\d+)?', st)}
        if {DN.get(t, t) for t in toks} & DIST != core: return False   # sibling variant (3d mix, beveled, bullnose ...)
        if any(st.startswith(x) for x in stems): return True
        wx = toks & words - coll                   # a word beyond the collection name (the colour) must match
        return bool(wx) and (bool(sz & size_tok) if size_tok and sz else len(toks & words) >= 2)
    page = [u for u in page_images(rec['path']) if own(u)]
    first = first[:1] + [u for u in first[1:] if own(u) or u == first[0]]     # the product main image may be a sibling's
    urls = first + [u for u in page if 'roomscenes' not in u] + thumb + [u for u in page if 'roomscenes' in u]
    urls = list(dict.fromkeys(urls))
    if isinstance(ov.get('image'), int): urls = urls[ov['image']:] + urls[:ov['image']]
    return urls

def load_photo(u):
    b = fetch_img(u)
    if not b: return None
    try: im = Image.open(io.BytesIO(b)).convert('RGB')
    except Exception: return None
    orig = im.size; im = trim_border(im)
    if max(im.size) > 1600: im = im.resize((round(im.width * 1600 / max(im.size)), round(im.height * 1600 / max(im.size))), Image.Resampling.LANCZOS)
    im.info['orig'] = orig
    return im

# ------------------------------------------------------------------ sizes & classification
NUM = r'(\d+\s+\d+/\d+|\d+/\d+|\d*\.\d+|\d+)'
def num(s):
    s = s.strip()
    if ' ' in s: a, b = s.split(None, 1); return float(a) + num(b)
    if '/' in s: a, b = s.split('/'); return float(a) / float(b)
    return float(s)

def parse_dims(*texts):
    """First 'A x B' in inches from the texts (fractions, decimals, quotes); ignores a trailing thickness."""
    for t in texts:
        t = (t or '').replace('”', '"').replace('″', '"')
        m = re.search(NUM + r'\s*(?:"|in\.?|inch)?\s*[xX×]\s*' + NUM + r'(?!\s*mm)', t)
        if m:
            a, b = num(m.group(1)), num(m.group(2))
            if 0.3 <= a <= 60 and 0.3 <= b <= 60: return max(a, b), min(a, b)
    return None

TRIM = re.compile(r'BULL\s*NOSE|\dBN\b|\bBN\b|QUARTER\s*ROUND|\bQTR\b|CORNER|PENCIL|LINER|CHAIR\s*RAIL|MOLDING|MOULDING|MLDG', re.I)
def shape_of(rec):
    t = ' '.join([rec['item'], rec['size'], rec['product']]).lower()
    if not re.search(r'\d\s*[xX]\s*\d', rec['item'] + ' ' + rec['size']): t += ' ' + (rec['style'] or '').lower()
    for k, ws in [('herringbone', ['herringbone']), ('hexagon', ['hexagon', 'hexo', ' hive', 'petal']),
                  ('penny', ['penny']), ('octagon', ['octagon']), ('arabesque', ['arabesque']), ('scallop', ['scallop']),
                  ('teardrop', ['teardrop']), ('picket', ['picket']), ('weave', ['weave']),
                  ('checker', ['checker', 'chex']), ('pattern', ['moderno'])]:
        if any(w in t for w in ws): return k
    return 'rect'

def nominal(rec, ov):
    """-> dict(shape, L, S (inches, tile face), how) or None when no size can be read."""
    sh = shape_of(rec); t = [rec['item'], rec['size'], rec['product'], rec['desc']]
    if ov.get('size'): return dict(shape=sh, L=max(ov['size']), S=min(ov['size']), how='override')
    if sh in ('hexagon', 'octagon'):
        m = re.search(NUM + r'\s*(?:"|in\b|inch|-inch|”)?\s*(?:x\s*\d+\s*)?(?:hexagon|hex|octagon|oct)', ' '.join(t).replace('”', '"'), re.I)
        d = num(m.group(1)) if m else None
        if d is None:
            m = re.search(r'(\d)\s*"?\s*x\s*\d\s*"?', rec['size'])   # Retro Hexo "2x2"
            d = float(m.group(1)) if m else None
        return dict(shape=sh, L=d, S=d, how='name') if d else dict(shape=sh, L=None, S=None, how='none')
    if sh == 'penny':
        return dict(shape=sh, L=0.75, S=0.75, how='penny round ~3/4 in')
    d = parse_dims(rec['item'], rec['size'] if re.search(r'\d', rec['size']) else '', rec['product'], rec['desc'])
    if d: return dict(shape=sh, L=d[0], S=d[1], how='name')
    m = re.search(r'(?<![\d.])(\d{1,2}(?:\.\d)?)X(\d{1,2}(?:\.\d)?)(?![\d.])', rec['sku'])     # last resort: the SKU code
    if m: return dict(shape=sh, L=max(float(m.group(1)), float(m.group(2))), S=min(float(m.group(1)), float(m.group(2))), how='sku code')
    return dict(shape=sh, L=None, S=None, how='none')

def finish_props(rec):
    f = (rec['item'] + ' ' + rec['finish']).lower(); mat = (rec['material'] or '').lower()   # the item name wins
    glass = 'glass' in mat or re.search(r'GLSST|GL-T', rec['sku'])
    if glass: return 0.06, 1.0, 'Glossy (glass)'
    hits = [(f.find(k), r, c, name) for k, r, c, name in [('gloss', 0.08, 1.0, 'Glossy'), ('polish', 0.10, 0.8, 'Polished'),
            ('satin', 0.30, 0.3, 'Satin'), ('honed', 0.45, 0.0, 'Honed'), ('matte', 0.55, 0.0, 'Matte'), (' hon', 0.45, 0.0, 'Honed'),
            ('gauged', 0.70, 0.0, 'Gauged'), ('natural', 0.85, 0.0, 'Natural')] if k in f]
    if hits: return min(hits)[1:]
    if 'clay' in mat or 'brick' in f: return 0.85, 0.0, 'Natural'
    if 'marble' in mat or 'travertine' in mat: return 0.45, 0.0, 'Honed'
    if 'slate' in mat: return 0.70, 0.0, 'Gauged'
    return 0.30, 0.3, rec['finish'] or 'unknown'

def edge_profile(rec, ov):
    if ov.get('profile'): return ov['profile']
    t = (rec['item'] + ' ' + rec['product'] + ' ' + (rec['style'] or '')).lower()
    if 'inverted' in t: return 'invbevel'
    if 'big bevel' in t: return 'bigbevel'
    if 'bevel' in t or re.search(r'\dHB\b|\dPB\b|BEV', rec['sku']): return 'bevel'
    if 'convex' in t: return 'convex'
    if 'flut' in t or 'ribbo' in t or 'slat' in t or 'gems' in t: return 'fluted'
    if any(k in t for k in ['handcraft', 'zellige', 'renzo', 'stella', 'flamenco', 'zest', 'marza', 'urbano', 'highland',
                            'antique', 'whisper', 'dove gray', 'portico', 'morning fog', 'bay blue', 'artisan', 'brick',
                            'village', 'clay']): return 'pillow'
    return 'flat'

# ------------------------------------------------------------------ image helpers
s2l = lambda a: np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
l2s = lambda a: np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 0.0031308, None), 1 / 2.4) - 0.055)
to_u8 = lambda a: Image.fromarray((np.clip(l2s(np.clip(a, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8))
lum = lambda a: a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722

def jpg_uri(im, q):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

def write_tex(tid, im, q=86):
    (TEX / f'{tid}.js').write_text(f'window.__texData({json.dumps(tid)},{json.dumps(jpg_uri(im, q))});\n')

def trim_border(im):
    """Drop flat (near-uniform) margins MSI sometimes leaves around a picture."""
    a = np.asarray(im.convert('L'), np.float32); h, w = a.shape
    rs, cs = a.std(1), a.std(0); y0, y1, x0, x1 = 0, h, 0, w
    while y0 < h // 4 and rs[y0] < 2: y0 += 1
    while y1 > 3 * h // 4 and rs[y1 - 1] < 2: y1 -= 1
    while x0 < w // 4 and cs[x0] < 2: x0 += 1
    while x1 > 3 * w // 4 and cs[x1 - 1] < 2: x1 -= 1
    return im.crop((x0, y0, x1, y1)) if (x0, y0, x1, y1) != (0, 0, w, h) else im

def on_white(im):
    """Fraction of the picture's border that is plain white background (a single piece shot on white)."""
    a = np.asarray(im.resize((200, 200)), np.float32)
    b = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    wb = (b.min(1) >= 251) & (b.max(1) - b.min(1) < 8)
    inner = a[60:140, 60:140]; inner_w = ((inner.min(2) >= 251) & (np.ptp(inner, 2) < 8)).mean()
    return float(wb.mean()), float(inner_w)

def edges(g, s=1.0):
    gs = ndi.gaussian_filter(g, s)
    return np.abs(ndi.sobel(gs, 1)), np.abs(ndi.sobel(gs, 0))     # gx (vertical lines), gy (horizontal lines)

def acf1(p):
    p = np.asarray(p, np.float64); n = len(p)
    p = p - ndi.gaussian_filter1d(p, max(3, n / 5)); p = p - p.mean()
    f = np.fft.rfft(p, 2 * n); a = np.fft.irfft(f * np.conj(f))[:n]
    a = a / np.maximum(n - np.arange(n), n * 0.3)
    return a / (a[0] + 1e-12)

def peak(a, lo, hi, rel=0.7):
    """First local maximum in [lo, hi) that is >= rel * the best one; sub-pixel. -> (lag, height) or (None, 0)."""
    lo, hi = max(2, int(lo)), min(int(hi), len(a) - 2)
    idx = [i for i in range(lo, hi) if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > 0]
    if not idx: return None, 0.0
    best = max(a[i] for i in idx); c = next(i for i in idx if a[i] >= rel * best)
    y0, y1, y2 = a[c - 1], a[c], a[c + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
    return c + float(np.clip(d, -0.5, 0.5)), float(a[c])

def near_peak(a, lo, hi, want, rel=0.45):
    """Among the clear local maxima in [lo, hi) the one closest (in log) to the expected lag."""
    lo, hi = max(2, int(lo)), min(int(hi), len(a) - 2)
    idx = [i for i in range(lo, hi) if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > 0]
    if not idx: return None, 0.0
    best = max(a[i] for i in idx); idx = [i for i in idx if a[i] >= rel * best]
    c = min(idx, key=lambda i: abs(math.log(i / want)))
    y0, y1, y2 = a[c - 1], a[c], a[c + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
    return c + float(np.clip(d, -0.5, 0.5)), float(a[c])

def harmonics(a, T):
    """Refine a period with the positions of its multiples in the autocorrelation."""
    if not T: return T
    ks, ps = [1], [T]
    for k in range(2, 12):
        c = k * T
        if c > 0.75 * len(a): break
        lo, hi = int(c - 0.12 * T), int(c + 0.12 * T) + 1
        if hi >= len(a) - 1: break
        i = lo + int(np.argmax(a[lo:hi]))
        if a[i] < 0.25 * a[int(round(T))] or not (lo < i < hi - 1): continue
        y0, y1, y2 = a[i - 1], a[i], a[i + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
        ks.append(k); ps.append(i + float(np.clip(d, -0.5, 0.5)))
    ks, ps = np.array(ks, float), np.array(ps)
    return float((ks * ps).sum() / (ks * ks).sum())

def rotate(a, deg):
    if abs(deg) < 1e-3: return a
    return ndi.rotate(a, deg, axes=(1, 0), reshape=False, order=3, mode='reflect', prefilter=True)

def deskew(g):
    """Angle (deg, ccw) that makes grout lines axis-aligned: maximise projection-profile sharpness."""
    f = min(1.0, 450 / max(g.shape)); s = ndi.zoom(ndi.gaussian_filter(g, 0.5 / f), f, order=1) if f < 1 else g
    def score(t, im=s):
        r = ndi.rotate(im, t, axes=(1, 0), reshape=False, order=1, mode='reflect') if abs(t) > 1e-3 else im
        h, w = r.shape; r = r[h // 6: -h // 6, w // 6: -w // 6]
        gx, gy = edges(r, 0.8)
        return gy.mean(1).std() / (gy.mean() + 1e-6) + gx.mean(0).std() / (gx.mean() + 1e-6)
    ts = np.arange(-4, 4.01, 0.5); sc = [score(t) for t in ts]; t0 = ts[int(np.argmax(sc))]
    ts = np.arange(t0 - 0.5, t0 + 0.51, 0.1); sc = [score(t) for t in ts]; t1 = ts[int(np.argmax(sc))]
    f2 = min(1.0, 900 / max(g.shape)); s2 = ndi.zoom(g, f2, order=1) if f2 < 1 else g    # fine step at higher resolution
    ts = np.arange(t1 - 0.1, t1 + 0.11, 0.025); sc = [score(t, s2) for t in ts]
    return float(ts[int(np.argmax(sc))])

# ------------------------------------------------------------------ cells: courses + joints
def lines_at(prof, T, sig):
    """Grout-line centres for a profile with period T: best comb phase, then each line refined locally."""
    p = ndi.gaussian_filter1d(prof, sig); n = len(p); xs = np.arange(n)
    best, ph = -1, 0
    for f in np.arange(0, T, 0.25):
        pos = np.arange(f, n - 1, T); v = np.interp(pos, xs, p).sum() / max(1, len(pos))
        if v > best: best, ph = v, f
    out = []
    for c in np.arange(ph, n, T):
        lo, hi = int(max(1, c - 0.12 * T)), int(min(n - 2, c + 0.12 * T))
        if hi - lo < 2: continue
        i = lo + int(np.argmax(p[lo:hi + 1]))
        if 0 < i < n - 1:
            y0, y1, y2 = p[i - 1], p[i], p[i + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
            out.append((i + float(np.clip(d, -0.5, 0.5)), float(p[i])))
    return out, p

def continuity(gy, ys, T):
    """How continuous the lines at ys are: fraction of their length with edge energy well above the tile interior."""
    if len(ys) < 2: return 0.0
    sm = ndi.gaussian_filter1d(gy, max(2, 0.03 * gy.shape[1]), axis=1)
    mids = [int((a + b) / 2) for a, b in zip(ys[:-1], ys[1:])]
    base = np.median(np.abs(sm[mids])) + 1e-6; fr = []
    for y in ys:
        y = int(round(y)); band = sm[max(0, y - 1): y + 2].max(0)
        fr.append(float((band > 2.5 * base).mean()))
    return float(np.median(fr))

def acf_peaks(a, lo, hi, k=6, rel=0.2):
    """Clear local maxima of an autocorrelation in [lo, hi): up to k (lag, height), strongest first."""
    lo, hi = max(2, int(lo)), min(int(hi), len(a) - 2)
    idx = [i for i in range(lo, hi) if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > 0]
    if not idx: return []
    best = max(a[i] for i in idx); idx = sorted([i for i in idx if a[i] >= rel * best], key=lambda i: -a[i])
    out = []
    for i in idx:
        y0, y1, y2 = a[i - 1], a[i], a[i + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
        T = harmonics(a, i + float(np.clip(d, -0.5, 0.5)))
        if all(abs(T / t - 1) > 0.06 for t, _ in out): out.append((T, float(a[i])))
        if len(out) >= k: break
    return out

def cells_at(g, gx, gy, c, nom, ov):
    """Courses at pitch c (horizontal grout lines) and the joints in each course. -> dict or None."""
    H, W = g.shape
    r = (nom['L'] * IN + 1.6) / (nom['S'] * IN + 1.6) if nom['L'] and nom['S'] else None
    ls, _ = lines_at(gy.mean(1), c, max(1.2, 0.012 * c)); ys = [y for y, _ in ls]
    ys2 = []
    for y in ys:                           # fill a missed line in a double-height gap
        if ys2 and 1.7 * c < y - ys2[-1] < 2.3 * c: ys2.append((ys2[-1] + y) / 2)
        ys2.append(y)
    ys = ys2
    courses = [(a, b) for a, b in zip(ys[:-1], ys[1:]) if 0.8 * c < b - a < 1.2 * c]
    if not courses: return None
    cont = continuity(gy, ys, c)
    pf = ndi.gaussian_filter1d(gy.mean(1), max(1.0, 0.01 * c)); base = np.median(pf)
    at = lambda y: pf[max(0, int(y - 0.04 * c)): int(y + 0.04 * c) + 1].max()
    sl = np.mean([at(y) for y in ys]) - base
    mid = 1.0 if sl <= 0 else 0.0     # ~1: lines between the courses as strong as the course lines (pitch a multiple,
    if len(ys) > 1 and sl > 0:         # ribs, flutes, striations) - real grout lines stand out from the tile face
        mid = float(np.median([pf[int(y0 + 0.25 * (y1 - y0)): int(y1 - 0.25 * (y1 - y0)) + 1].max() - base
                               for y0, y1 in zip(ys[:-1], ys[1:])]) / sl)
    # joints: columns whose vertical edge energy spans the course height (veins and glints rarely do)
    joints, spac, covs = [], [], []
    for a, b in courses:
        y0, y1 = int(a + 0.15 * (b - a)), int(b - 0.15 * (b - a)) + 1
        band = ndi.maximum_filter1d(gx[y0:y1], 3, axis=1)
        thr = max(4 * np.median(gx[y0:y1]), 0.2 * np.percentile(band, 99.7), 0.03)
        cov = ndi.gaussian_filter1d((band > thr).mean(0), 1.0); covs.append(cov)
        pr = ndi.gaussian_filter1d(gx[y0:y1].mean(0), 1.5)
        cand = [i for i in range(1, W - 1) if cov[i] >= 0.6 and cov[i] >= cov[i - 1] and cov[i] >= cov[i + 1]]
        kept = []
        for i in sorted(cand, key=lambda i: -cov[i]):
            if all(abs(i - k) > 0.3 * c for k in kept): kept.append(i)
        xs = []
        for i in sorted(kept):
            w = max(3, int(0.03 * c)); lo_, hi_ = max(1, i - w), min(W - 2, i + w)
            j = lo_ + int(np.argmax(pr[lo_:hi_ + 1]))
            y0_, y1_, y2_ = pr[j - 1], pr[j], pr[j + 1]
            x = j + float(np.clip(0.5 * (y0_ - y2_) / (y0_ - 2 * y1_ + y2_ - 1e-12), -0.5, 0.5))
            if 0.01 * c < x < W - 1 - 0.01 * c: xs.append(x)
        joints.append(xs); spac += list(np.diff(xs))
    want = r * c if r else None; how = 'measured'; spread = None
    if ov.get('period_px'): p = ov['period_px'][0]; how = 'override'
    else:
        sp = np.array([v for v in spac if v > 0.4 * c]); p = None
        if want: sp = sp[(sp > 0.4 * want) & (sp < 2.2 * want)]
        pw = np.array([b - a for js in joints for i, a in enumerate(js) for b in js[i + 1:]])     # all pairs in a course
        dense = bool(want and np.median([len(js) for js in joints]) > 2.5 * W / want + 1)   # flutes / ribs, not joints
        near = pw[np.abs(pw / want - 1) < 0.12] if want and len(pw) and not dense else np.array([])
        if len(near) >= 2 and len(near) >= 0.25 * len(sp):       # veins / decor add edges inside tiles: trust the
            p = float(np.median(near)); spread = float(np.std(near) / p)   # spacing cluster at the nominal length
        elif len(sp) >= 2 or (len(sp) == 1 and want and abs(sp[0] / want - 1) < 0.12):
            m0 = np.median(sp); half = sp[np.abs(sp - m0 / 2) < 0.12 * m0 / 2]
            if len(half) >= 0.25 * len(sp): m0 = m0 / 2
            cl = sp[np.abs(sp - m0) < 0.15 * m0]
            if len(cl): p = float(np.median(cl)); spread = float(np.std(cl) / p) if len(cl) > 1 else 0.0
        ds = []
        for ja, jb in zip(joints[:-1], joints[1:]):
            for x in ja:
                if jb: ds.append(min(jb, key=lambda y: abs(y - x)) - x)
        ds = np.array([d for d in ds if abs(d) > 0.1 * c])
        if p is None and want and len(ds) >= 2:   # few joints: neighbouring-course offsets that fit the nominal length
            for q in want * (1 + np.r_[0, np.ravel([[e, -e] for e in np.arange(0.01, 0.121, 0.01)])]):
                fr = (ds / q) % 1.0
                if any(np.all(np.minimum(np.abs(fr - f), np.abs(fr - (1 - f))) < 0.035) for f in (0.5, 1 / 3, 0.25)):
                    p, how = float(q), 'bond offsets'; break
        if p is None and want:     # one joint per course: a half bond puts neighbouring joints p/2 apart
            if len(ds) >= 2:
                grid = np.linspace(0.55 * want, 1.8 * want, 1200)
                sc = np.array([np.median(np.abs(np.abs(((np.array(ds) / q) % 1.0) - 0.5))) for q in grid])
                ok = np.where(sc < 0.03)[0]
                if len(ok): p, how = float(grid[min(ok, key=lambda i: abs(math.log(grid[i] / want)))]), 'half-bond offsets'
        if p is None:
            if not want: return None
            p, how = want, 'aspect'
    # the joints the pitch predicts next to each observed one must be there (and none in between)
    pred = miss = 0
    for js, cov in zip(joints, covs):
        for x in js[:2]:
            for q in (x - p, x + p, x - 2 * p, x + 2 * p):
                if 0.03 * W < q < 0.97 * W:
                    pred += 1; miss += cov[max(0, int(q - 0.03 * p)): int(q + 0.03 * p) + 1].max() < 0.4
    nj = sum(len(js) for js in joints)
    return dict(contradict=(round(miss / pred, 2) if pred else None), c=c, p=p, p_how=how, spread=spread,
                courses=courses, joints=joints, cont=cont, nj=nj, W=W, H=H, mid=mid)

def analyse_cells(g, nom, ov):
    """Courses and joints on the deskewed grey image. Every clear course-pitch candidate (row autocorrelation peaks
    of the horizontal-edge energy) in both orientations is tried; the best fit to the nominal tile wins.
    -> dict with orientation (transpose), courses, joints, score."""
    H, W = g.shape; gx, gy = edges(g, 1.0)
    lo, hi = 12, 0.55 * min(H, W)
    best = None; tried = []
    tr_opts = [bool(ov['rotate90'])] if ov.get('rotate90') is not None else [False, True]
    for tr in tr_opts:
        G, GX, GY = (g.T, gy.T, gx.T) if tr else (g, gx, gy)
        a = acf1(GY.mean(1))
        cands = [(ov['period_px'][1], 1.0)] if ov.get('period_px') else acf_peaks(a, lo, hi)
        f = 400 / max(H, W)
        if f < 0.6 and not ov.get('period_px'):     # coarse scale too: fine grain / flutes can swamp the full-res profile
            gs = ndi.zoom(ndi.gaussian_filter(G, 0.5 / f), f, order=1)
            for T, hh in acf_peaks(acf1(edges(gs, 1.0)[1].mean(1)), 8, 0.55 * min(gs.shape)):
                T /= f; i = int(round(T)); w = max(2, int(0.03 * T))
                if i + w < len(a) - 1:      # refine on the full-res autocorrelation when it has a peak there
                    j = i - w + int(np.argmax(a[i - w: i + w + 1]))
                    if i - w < j < i + w and a[j] > 0: T = float(j)
                if all(abs(T / t - 1) > 0.06 for t, _ in cands): cands.append((T, hh))
        for c, h in cands:
            an = cells_at(G, GX, GY, c, nom, ov)
            if not an: continue
            sc = 1.5 * h + 1.0 * an['cont']
            r = (nom['L'] * IN + 1.6) / (nom['S'] * IN + 1.6)
            if an['mid'] > 0.45: sc -= 2.0 * (an['mid'] - 0.3)
            if an['p_how'] in ('measured', 'half-bond offsets', 'bond offsets'):
                sc -= 4.0 * abs(math.log(an['p'] / (r * c))) + (0.1 if an['p_how'] == 'bond offsets' else 0)
                if an['spread']: sc -= 2.0 * max(0, an['spread'] - 0.04)
            else: sc -= 0.35
            if an['contradict'] is not None: sc -= 1.2 * an['contradict']
            if len(an['courses']) < 2: sc -= 0.6
            if an['nj'] == 0: sc -= 0.4
            if r * c > 1.05 * G.shape[1]: sc -= 0.8 * min(1.0, r * c / G.shape[1] - 1)   # tile longer than the photo
            span = max(G.shape) * (nom['S'] * IN + 1.6) / c          # photo width in mm this pitch implies
            if not 60 < span < 1500: sc -= 3.0                       # close-ups show a few inches to a few feet
            an.update(transpose=tr, score=round(sc, 3), h=h)
            tried.append((tr, round(c, 1), round(an['p'], 1), an['p_how'], round(sc, 2)))
            if best is None or sc > best['score']: best = an
    if best: best['tried'] = sorted(tried, key=lambda t: -t[-1])
    return best

def bond_of(an):
    """Course offsets as fractions of the joint pitch -> ('stack'|'running', fraction or None for random, phases).
    Uses every pair of courses that shows joints; with no evidence a half bond is assumed (subway default)."""
    p = an['p']; ph = [None if not js else (js[0] % p) / p for js in an['joints']]
    pairs = [(j - i, (ph[j] - ph[i]) % 1.0) for i in range(len(ph)) for j in range(i + 1, len(ph))
             if ph[i] is not None and ph[j] is not None]
    if not pairs: return 'running', 0.5, ph, False
    def fits(f): return all(min(abs((k * f) % 1.0 - d), 1 - abs((k * f) % 1.0 - d)) < 0.09 for k, d in pairs)
    ok = [f for f in (0.0, 0.5, 1 / 3, 2 / 3, 0.25, 0.75) if fits(f)]
    if ok:
        f = 0.5 if (0.5 in ok and not any(k % 2 for k, _ in pairs)) else ok[0]   # only even course gaps seen: half bond
        return ('stack' if f == 0 else 'running'), f, ph, True
    return 'running', None, ph, True

def sample(img, x0, x1, y0, y1, w, h, mode='mirror'):
    """Resample the source rectangle [x0,x1]x[y0,y1] (pixel-centre coordinates) to w x h."""
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w - 0.5; ys = y0 + (np.arange(h) + 0.5) * (y1 - y0) / h - 0.5
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    if img.ndim == 2: return ndi.map_coordinates(img, [Y, X], order=1, mode=mode)
    return np.stack([ndi.map_coordinates(img[..., k], [Y, X], order=1, mode=mode) for k in range(img.shape[2])], -1)

def refine_line(gy, y, x0, x1, c):
    """Local position of a horizontal grout line over the x-range of one cell."""
    lo, hi = int(max(1, y - 0.1 * c)), int(min(gy.shape[0] - 2, y + 0.1 * c))
    pr = ndi.gaussian_filter1d(gy[lo:hi + 1, int(max(0, x0)): int(min(gy.shape[1], x1))].mean(1), max(1.0, 0.012 * c))
    i = int(np.argmax(pr))
    if 0 < i < len(pr) - 1:
        y0, y1, y2 = pr[i - 1], pr[i], pr[i + 1]; return lo + i + float(np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12), -0.5, 0.5))
    return y

def grout_width(g, an, mmpp):
    """Grout width in px: the narrow feature at the line centre, measured at half depth against the profile
    R mm either side (R ~ 0.12 course, 2.5..6 mm), so bevels and pillowed edges next to the joint do not count."""
    c = an['c']; half = int(0.2 * c); prs = []
    for a, b in an['courses']:
        for y in (a, b):
            y = int(round(y))
            if half <= y < g.shape[0] - half: prs.append(g[y - half: y + half + 1].mean(1))
    if not prs: return None
    pr = ndi.zoom(np.median(prs, 0), 4, order=1); ctr = 4 * half    # 4x supersampled
    for Rmm in (1.5, 2.5, 4.0, 6.0):       # narrowest window first: a bevel / pillow next to a fine joint is not grout
        R = min(int(round(4 * Rmm / mmpp)), ctr - 1)
        if R < 6: continue
        base = 0.5 * (pr[ctr - R] + pr[ctr + R]); dev = pr - base
        win = dev[ctr - R // 2: ctr + R // 2 + 1]; k = int(np.argmax(np.abs(win))); m = win[k]; pk = ctr - R // 2 + k
        if abs(m) < 0.03: continue
        inside = np.sign(m) * dev > 0.5 * abs(m); l = r = pk
        while l > ctr - R and inside[l - 1]: l -= 1
        while r < ctr + R and inside[r + 1]: r += 1
        if r - l + 1 <= 1.2 * R: return (r - l + 1) / 4.0
    return None

# ------------------------------------------------------------------ lattice (mosaics / patterns)
def acf2c(f):
    """Full (signed-lag) normalised 2-D autocorrelation, centred: lag (dy, dx) at [h + dy, w + dx]."""
    f = f - f.mean(); h, w = f.shape
    F = np.fft.rfft2(f, (2 * h, 2 * w)); a = np.fft.fftshift(np.fft.irfft2(F * np.conj(F), (2 * h, 2 * w)))
    ny = h - np.abs(np.arange(2 * h) - h); nx = w - np.abs(np.arange(2 * w) - w)
    a = a / np.maximum(np.outer(ny, nx).astype(float), 0.25 * h * w)
    return a / (a[h, w] + 1e-12)

def _subpix(a, y, x):
    def par(m, c, p): d = m - 2 * c + p; return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (m - p) / d, -0.5, 0.5))
    return y + par(a[y - 1, x], a[y, x], a[y + 1, x]), x + par(a[y, x - 1], a[y, x], a[y, x + 1])

def lattice_basis(a, rmin=6, rmax=None):
    """Two shortest non-collinear strong peaks of a centred ACF -> reduced basis [(dx, dy), (dx, dy)], peak values."""
    H2, W2 = a.shape; h, w = H2 // 2, W2 // 2; rmax = rmax or 0.45 * min(h, w)
    mx = ndi.maximum_filter(a, 5)
    ys, xs = np.where((a == mx) & (a > 0.05))
    pk = []
    for y, x in zip(ys, xs):
        dy, dx = y - h, x - w; r = math.hypot(dx, dy)
        if r < rmin or r > rmax or dy < 0 or (dy == 0 and dx <= 0) or not (0 < y < H2 - 1 and 0 < x < W2 - 1): continue
        yy, xx = _subpix(a, y, x); pk.append((xx - w, yy - h, float(a[y, x]), r))
    if len(pk) < 2: return None
    top = max(p[2] for p in pk); strong = sorted([p for p in pk if p[2] >= 0.5 * top], key=lambda p: p[3])
    v1 = strong[0]
    v2 = next((p for p in strong[1:] if abs(v1[0] * p[1] - v1[1] * p[0]) / (v1[3] * p[3]) > 0.42), None)
    if v2 is None: return None
    b1, b2 = np.array(v1[:2]), np.array(v2[:2])
    for _ in range(10):                      # Lagrange-Gauss reduction
        if b1 @ b1 > b2 @ b2: b1, b2 = b2, b1
        mu = round((b1 @ b2) / (b1 @ b1))
        if mu == 0: break
        b2 = b2 - mu * b1
    return b1, b2, (v1[2], v2[2])

def axis_repeat(b1, b2, a, axis, tol=0.03):
    """Shortest lattice vector along x (axis 0) or y (axis 1), refined on the ACF -> length in px or None."""
    best = None
    for i in range(-10, 11):
        for j in range(-10, 11):
            v = i * b1 + j * b2; L = float(np.hypot(*v))
            if L < 4: continue
            along, across = (v[0], v[1]) if axis == 0 else (v[1], v[0])
            if along > 0 and abs(across) <= max(1.0, tol * L) and (best is None or along < best): best = along
    if best is None: return None
    H2, W2 = a.shape; h, w = H2 // 2, W2 // 2; k = int(round(best))
    if axis == 0:
        if w + k + 2 >= W2: return float(best)
        x = w + k - 2 + int(np.argmax(a[h, w + k - 2: w + k + 3])); return _subpix(a, h, x)[1] - w
    if h + k + 2 >= H2: return float(best)
    y = h + k - 2 + int(np.argmax(a[h + k - 2: h + k + 3, w])); return _subpix(a, y, w)[0] - h

def seam_path(cost, start=None, end=None):
    """Min-cost top-to-bottom path through cost (rows x cols), 8-connected; optional pinned start / end column."""
    h, w = cost.shape; M = cost.astype(np.float64).copy()
    if start is not None: M[0, :] = 1e12; M[0, start] = cost[0, start]
    back = np.zeros((h, w), np.int8)
    for y in range(1, h):
        prev = np.stack([np.r_[1e18, M[y - 1, :-1]], M[y - 1], np.r_[M[y - 1, 1:], 1e18]])
        k = np.argmin(prev, 0); back[y] = k - 1; M[y] += prev[k, np.arange(w)]
    x = end if end is not None else int(np.argmin(M[-1])); path = np.zeros(h, int)
    for y in range(h - 1, -1, -1):
        path[y] = x; x = int(np.clip(x + back[y, x], 0, w - 1))
    return path

def _diff(P, Q, band):
    d = np.abs(P[:, -band:] - Q[:, -band:]); d = d.sum(-1) if d.ndim == 3 else d
    return ndi.gaussian_filter(d, 1.0)

def _merge(P, Q, path, feather=1.5):
    h, w = P.shape[:2]
    t = np.clip((np.arange(w)[None, :] - path[:, None] + 0.5) / (2 * feather) + 0.5, 0, 1)
    if P.ndim == 3: t = t[..., None]
    return P * (1 - t) + Q * t

def lattice_crop(img, Tx, Ty, nx, ny, x0, y0, bx, by):
    """Exact-repeat crop of nx*Tx x ny*Ty source px at (x0, y0), at 1 output px per source px. Wrap seams: the
    lattice-equivalent pixels just before the crop (one span left / above) replace the last bx / by columns / rows
    along minimum-difference cuts (they follow grout, whole chips are kept). The cut ends that meet across the wrap
    are pinned to the same, cheapest, column / row so the texture tiles exactly."""
    Ws, Hs = nx * Tx, ny * Ty; w, h = int(round(Ws)), int(round(Hs))
    S = lambda dx, dy: sample(img, x0 + dx, x0 + dx + Ws, y0 + dy, y0 + dy + Hs, w, h)
    base, left, up, ul = S(0, 0), S(-Ws, 0), S(0, -Hs), S(-Ws, -Hs)
    dA, dB = _diff(base, left, bx), _diff(up, ul, bx)
    p = int(np.argmin(ndi.uniform_filter1d(dA[0] + dB[-1], 5)))
    A = _merge(base, left, w - bx + seam_path(dA, start=p)); B = _merge(up, ul, w - bx + seam_path(dB, end=p))
    At, Bt = np.swapaxes(A, 0, 1), np.swapaxes(B, 0, 1); dT = _diff(At, Bt, by)
    q = int(np.argmin(ndi.uniform_filter1d(dT[0] + dT[-1], 5)))
    return np.swapaxes(_merge(At, Bt, h - by + seam_path(dT, start=q, end=q)), 0, 1)

# ------------------------------------------------------------------ colour
def flatten(lin, mask, deg=2):
    """Divide out a smooth lighting gradient: low-order polynomial fit of log luminance on tile pixels."""
    L = lum(lin); h, w = L.shape; k = max(1, int(max(h, w) / 200))
    Ls, ms = L[::k, ::k], mask[::k, ::k]
    yy, xx = np.mgrid[0:Ls.shape[0], 0:Ls.shape[1]]; yy = yy / Ls.shape[0] - 0.5; xx = xx / Ls.shape[1] - 0.5
    terms = lambda x, y: [np.ones_like(x), x, y] + ([x * x, y * y, x * y] if deg >= 2 else [])
    sel = ms & (Ls > 0.005)
    if sel.sum() < 50: return lin
    v = np.log(Ls[sel] + 1e-4); Aa = np.stack([t[sel] for t in terms(xx, yy)], 1)
    keep = np.ones(len(v), bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(Aa[keep], v[keep], rcond=None); res = v - Aa @ c
        keep = np.abs(res - np.median(res[keep])) < 2.5 * res[keep].std() + 1e-6
    Y, X = np.mgrid[0:h, 0:w]; Y = Y / h - 0.5; X = X / w - 0.5
    fit = sum(ci * t for ci, t in zip(c, terms(X, Y))); fit -= np.median(fit[mask]) if mask.any() else fit.mean()
    return lin / np.exp(fit)[..., None]

def compress_glints(lin, cells, body, lowf=0.55, k=1.15, s=0.45):
    """Glossy photos bake reflections into the colour. Per tile: compress luminance above k x the tile median
    (glints) and, for glazed ceramics, pull the large-scale (> 1/3 tile) colour deviation from the tile median
    towards it by lowf (window / sky reflections), keeping the fine glaze mottling."""
    out = lin.copy()
    for (y0, y1, x0, x1) in cells:
        c = out[y0:y1, x0:x1]; bm = body[y0:y1, x0:x1]
        if c.size == 0 or bm.sum() < 20: continue
        lg = np.log(c + 1e-3); med = np.median(lg[bm], 0)
        if lowf > 0:
            sg = max(2, min(c.shape[:2]) / 3)
            lo = np.stack([ndi.gaussian_filter(lg[..., q], sg, mode='nearest') for q in range(3)], -1)
            lg = lg - lowf * (lo - med) * bm[..., None]
        c = np.exp(lg) - 1e-3
        Lc = lum(c) + 1e-6; m = np.exp(med @ np.array([0.2126, 0.7152, 0.0722])); t = k * m
        Ln = np.where(Lc > t, t + (Lc - t) * s, Lc); out[y0:y1, x0:x1] = c * (Ln / Lc)[..., None]
    return out

# ------------------------------------------------------------------ normal map
def height_cells(Wt, Ht, gpx, mmpp, prof, courses):
    """Height (mm) for a cells texture: grout recess, edge profile per tile. courses = [(y0, y1, [joint xs])]."""
    h = np.zeros((Ht, Wt), np.float32)
    for (y0, y1, js) in courses:
        ys = np.arange(y0, y1)[:, None] + 0.5
        dy = np.minimum(ys - y0, y1 - ys)                                   # to course grout centre
        xs = np.arange(Wt)[None, :] + 0.5
        dj = np.full((1, Wt), 1e9)
        for j in js:
            for jj in (j - Wt, j, j + Wt): dj = np.minimum(dj, np.abs(xs - jj))
        d = (np.minimum(dy, dj) - gpx / 2) * mmpp                          # mm from the tile edge (negative: grout)
        hh = tile_profile(d, prof, dy * mmpp - gpx / 2 * mmpp, (y1 - y0) * mmpp - gpx * mmpp)
        h[y0:y1] = hh
    return h

def tile_profile(d, prof, dshort, short_mm):
    grout = -1.6
    if prof == 'bevel': body = np.clip(d / 6.0, 0, 1) * 2.5
    elif prof == 'bigbevel': body = np.clip(d / 10.0, 0, 1) * 3.5
    elif prof == 'invbevel': body = 2.5 - np.clip(d / 6.0, 0, 1) * 1.8
    elif prof == 'convex':
        t = np.clip(dshort / max(1, short_mm / 2), 0, 1); body = 1.0 + 5.0 * np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))
    elif prof == 'pillow': t = np.clip(d / 4.0, 0, 1); body = 1.4 * np.sin(t * np.pi / 2)
    else: t = np.clip(d / 1.2, 0, 1); body = 1.0 * np.sin(t * np.pi / 2)
    return np.where(d < 0, grout, body).astype(np.float32)

def normal_map(h, mmpp, detail=None, ddepth=0.0):
    if detail is not None: h = h + ddepth * detail
    hx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * mmpp); hy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * mmpp)
    hx, hy = np.clip(hx, -2.5, 2.5), np.clip(hy, -2.5, 2.5)
    n = np.stack([-hx, hy, np.ones_like(hx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)  # image y down, uv v up
    return Image.fromarray((127.5 + 127.5 * n).round().clip(0, 255).astype(np.uint8))

def detail_height(lin, mmpp):
    """Fine relief from the photo: band-passed luminance (handmade waves, flutes, 3-D decors), ~unit amplitude."""
    L = np.log(lum(lin) + 0.02)
    b = ndi.gaussian_filter(L, max(0.7, 0.6 / mmpp), mode='wrap') - ndi.gaussian_filter(L, max(2, 8 / mmpp), mode='wrap')
    return (b / (np.percentile(np.abs(b), 95) + 1e-6)).astype(np.float32)

# ------------------------------------------------------------------ build
def load_ov(sku):
    f = OVR / (re.sub(r'[^A-Za-z0-9._-]+', '_', sku) + '.json')
    return json.load(open(f)) if f.exists() else {}

def slug(s): return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

class Fail(Exception): pass
GOOD_FIT, MIN_FIT = 1.0, 0.45      # cells fit score: stop trying photos at GOOD_FIT, reject below MIN_FIT

def build(rec, series_scale=None):
    """Analyse one SKU. -> (entry, artefacts for QA) ; artefacts: dict(photo, overlay, tex, nrm)."""
    ov = load_ov(rec['sku']); notes = []
    sid = 'msitile_' + slug(rec['sku'])
    rough, cc, fin = finish_props(rec)
    nom = nominal(rec, ov)
    e = dict(id=sid, sku=rec['sku'], name=rec['item'].strip(), product=rec['product'], url=rec['url'],
             collection=rec['collection'], color=rec['color'], material=rec['material'], finish=fin, style=rec['style'] or '',
             size_label=rec['size'], tile_mm=([round(nom['L'] * IN, 1), round(nom['S'] * IN, 1)] if nom['L'] else None),
             kind=None, usable=False, reason='', tex=None, rough=rough, clearcoat=cc, relief=None, swatch=None, thumb=None,
             scale_method=None, scale_check=None, photo=None, notes=notes)
    art = dict(photo=None, overlay=[], tex=None, nrm=None, title='')
    trim = bool(TRIM.search(rec['item'] + ' ' + rec['size'])) or bool(re.search(r'(QTR|BN[GP]?)(-N)?$', rec['sku']))
    urls = photo_urls(rec, ov)
    if trim or ov.get('kind') == 'trim':
        for u in urls:
            im = load_photo(u)
            if im is not None: e['photo'] = u; art['photo'] = im; break
        e.update(kind='trim', reason='trim piece (not field tile)'); return finish_entry(e, ov), art
    sheet = (rec['chips'].strip('"').isdigit() and int(rec['chips'].strip('"')) > 1) or re.search(r'MOSAIC|\d+MM\b|\dx\d+x\d+mm', rec['item'], re.I) or rec['size'] in ('Pattern', 'Misc')
    prof = edge_profile(rec, ov)
    # try the SKU's own photos in order; keep the one whose grout pattern fits the nominal size best
    best, fails, scene, ntry = None, [], None, 0
    for u in urls:
        if 'roomscenes' in u and not ov.get('mode'):
            if scene is None and best is None: scene = u
            continue
        if '/thumbnails/' in u and ntry: continue     # a thumbnail is only a smaller copy of the main photo
        im = load_photo(u)
        if im is None: continue
        if ov.get('crop'): im = im.crop(tuple(ov['crop']))
        ntry += 1
        bw, inner = on_white(im)
        mode = ov.get('mode') or ('synth' if bw > 0.55 else 'cells' if nom['shape'] == 'rect' else 'lattice')
        nts = []
        try:
            if mode == 'synth': r = make_synth(im, rec, nom, ov, nts); sc = 0.6
            elif mode == 'picket': r = make_picket(im, rec, nom, ov, nts); sc = 0.6
            elif mode == 'cells':
                r = make_cells(im, rec, nom, ov, nts, fin); sc = r['check']['fit_score']
                sc += 0.4 * min(1.0, r['check']['unique_tiles'] / 8)       # prefer photos with more whole tiles
            else: r = make_lattice(im, rec, nom, ov, nts, fin, series_scale); sc = 1.0
            if mode == 'cells' and sc < MIN_FIT and 'period_px' not in ov:
                raise Fail(f'grout pattern does not fit the nominal {rec["size"]} (fit {sc:.2f})')
        except Fail as ex:
            fails.append((u, im, str(ex))); continue
        if best is None or sc > best[0]: best = (sc, u, im, mode, r, nts)
        if sc >= GOOD_FIT + 0.4 or ntry >= 5 or 'image' in ov: break
    if best is None:
        if fails:
            u, im, why = fails[0]; e['photo'] = u; art['photo'] = im
            e['scale_check'] = dict(photo_px=list(im.size), photo_orig_px=list(im.info.get('orig', im.size)))
            e.update(kind=ov.get('kind') or ('mosaic' if nom['shape'] != 'rect' or sheet else 'field'),
                     reason=why + (f' ({len(fails)} photos tried)' if len(fails) > 1 else ''))
        elif scene:
            e['photo'] = scene; art['photo'] = load_photo(scene)
            e.update(kind='scene', reason='only room scenes are available (no close-up to measure)')
        else: e.update(kind='field', reason='no photo could be downloaded')
        return finish_entry(e, ov), art
    sc, url, im, mode, r, nts = best; notes += nts; e['photo'] = url; art['photo'] = im
    r['check']['photo_px'] = list(im.size); r['check']['photo_orig_px'] = list(im.info.get('orig', im.size))
    if len(fails) or urls.index(url) > 0: notes.append(f'photo #{urls.index(url)} of {len(urls)} used')
    e['kind'] = ov.get('kind') or ('mosaic' if (mode == 'lattice' or sheet) else 'field')
    lin, h_geo, mmpp, tex, check, method, overlay = r['lin'], r['h'], r['mmpp'], r['tex'], r['check'], r['method'], r['overlay']
    art['overlay'] = overlay
    # exposure
    body = r['body']; Lb = lum(lin)[body]
    white = ov.get('white', 'white' in (rec['color'] or '').lower() or (rec['color'] or '').lower() in ('cream',))
    if white:
        tgt = 0.80 if fin in ('Glossy', 'Polished', 'Satin', 'Glossy (glass)') else 0.70    # glaze vs matte / rustic body
        k = float(np.clip(tgt / (np.percentile(Lb, 90) + 1e-6), 0.6, 1.5)); notes.append(f'white exposure x{k:.2f} (to {tgt})')
    else:
        k = float(min(1.0, 0.85 / (np.percentile(Lb, 99) + 1e-6)))
        if k < 0.98: notes.append(f'exposure capped x{k:.2f}')
    lin = np.clip(lin * k * ov.get('exposure', 1.0), 0, 1)
    # normal map: geometry + photo detail
    det = detail_height(lin, mmpp)
    dd = {'pillow': 0.35, 'fluted': 0.9, 'convex': 0.3, 'flat': 0.12}.get(prof, 0.2)
    if re.search(r'3d mix|mix 4x12', rec['item'] + rec['product'], re.I): dd = 0.6
    if 'marble' in (rec['material'] or '').lower() or 'glass' in (rec['material'] or '').lower(): dd = min(dd, 0.05)
    nrm = normal_map(h_geo, mmpp, det * body, dd)
    col = to_u8(lin)
    mx = max(col.size)
    if mx > 2048:
        f = 2048 / mx; col = col.resize((round(col.width * f), round(col.height * f)), Image.Resampling.LANCZOS)
    nrm = nrm.resize(col.size, Image.Resampling.BILINEAR) if nrm.size != col.size else nrm
    if max(nrm.size) > 1024:
        f = 1024 / max(nrm.size); nrm = nrm.resize((round(nrm.width * f), round(nrm.height * f)), Image.Resampling.LANCZOS)
    sw = l2s(np.clip(lin[body].mean(0), 0, 1)); e['swatch'] = '#' + ''.join(f'{int(round(v * 255)):02x}' for v in sw)
    tex.update(id=sid, normal=sid + '_n', px=list(col.size), long_axis='x')
    e.update(tex=tex, usable=True, relief=ov.get('relief', {'bevel': 1.0, 'bigbevel': 1.0, 'invbevel': 1.0, 'convex': 1.0,
             'fluted': 1.0, 'pillow': 0.8}.get(prof, 0.6)), scale_method=method, scale_check=check)
    if check and check.get('disagree') is not None and abs(check['disagree']) > 0.10:
        notes.append(f"axes disagree {check['disagree'] * 100:+.0f}% (x vs y mm/px)")
        if abs(check['disagree']) > 0.25 and 'usable' not in ov:
            e.update(usable=False, reason=f"photo tile proportions do not match the nominal {rec['size']} "
                                           f"({check['disagree'] * 100:+.0f}% between axes)")
    if check and (check.get('joints_missing') or 0) > 0.34 and 'usable' not in ov:
        e.update(usable=False, reason=f"{check['joints_missing'] * 100:.0f}% of the joints the nominal {rec['size']} predicts are missing from the photo")
    e['thumb'] = jpg_uri(thumb_of(col, tex['period_mm']), 72)
    art.update(tex=col, nrm=nrm)
    return finish_entry(e, ov), art

def finish_entry(e, ov):
    for k in ('kind', 'usable', 'reason', 'bond'):
        if k in ov:
            if k == 'bond' and e['tex']: e['tex']['bond'] = ov[k]
            elif k != 'bond': e[k] = ov[k]
    if ov.get('notes'): e['notes'].append(ov['notes'])
    if not e['usable']:
        if e['kind'] == 'trim' and not e['reason']: e['reason'] = 'trim piece'
    return e

def thumb_of(col, period_mm, w=84, h=56, mm_w=14 * IN):
    """84 x 56 card picture at a common scale (84 px = 14 in), tiled from the texture."""
    pxmm = col.width / period_mm[0]; need_w = mm_w * pxmm; need_h = mm_w * h / w * pxmm
    reps = (int(math.ceil(need_w / col.width)) + 1, int(math.ceil(need_h / col.height)) + 1)
    big = Image.new('RGB', (col.width * reps[0], col.height * reps[1]))
    for i in range(reps[0]):
        for j in range(reps[1]): big.paste(col, (i * col.width, j * col.height))
    return big.crop((0, 0, int(need_w), int(need_h))).resize((w, h), Image.Resampling.LANCZOS)

def diagonal_share(g):
    """Share of strong-edge energy oriented within 12 deg of the diagonals (vs the axes): > ~0.5 = tiles laid at 45 deg."""
    gs = ndi.gaussian_filter(g, 1.5); gx, gy = ndi.sobel(gs, 1), ndi.sobel(gs, 0); m = np.hypot(gx, gy)
    sel = m > np.percentile(m, 90); a = np.degrees(np.arctan2(gy[sel], gx[sel])) % 90; w = m[sel]
    ax = w[(a < 12) | (a > 78)].sum(); dg = w[np.abs(a - 45) < 12].sum()
    return float(dg / (ax + dg + 1e-9))

def make_cells(im, rec, nom, ov, notes, fin):
    rgb = np.asarray(im, np.float32) / 255; lin = s2l(rgb); g = rgb.mean(2)
    dsh = diagonal_share(g)
    if dsh > 0.5 and not ov.get('rotate'): raise Fail(f'photo shows the tile laid diagonally / in a 45 deg pattern ({dsh:.0%} of edges)')
    if ov.get('rotate'): lin = np.clip(rotate(lin, ov['rotate']), 0, 1); g = rotate(g, ov['rotate'])
    ang = deskew(g)
    if abs(ang) > 0.05: lin = np.clip(rotate(lin, ang), 0, 1); g = rotate(g, ang); notes.append(f'deskew {ang:+.2f} deg')
    if not nom['L']: raise Fail('no nominal size')
    an = analyse_cells(g, nom, ov)
    if not an: raise Fail('no regular grout lines found')
    if an['transpose']: lin, g = lin.transpose(1, 0, 2), g.T; notes.append('rotated 90 deg (vertical courses in photo)')
    lin = np.ascontiguousarray(lin); g = np.ascontiguousarray(g); Hs, Ws = g.shape
    gx, gy = edges(g, 1.0)
    if an['p_how'] == 'aspect' and any(len(js) > Ws / an["p"] + 1 for js in an['joints']):
        an['joints'] = [[] if len(js) > Ws / an["p"] + 1 else js for js in an['joints']]   # more 'joints' than tiles: flutes / decor
        notes.append('edges at less than the tile length (flutes / decor) ignored as joints')
    bond, frac, ph, seen = bond_of(an)
    if not seen and 'bond' not in ov and 'offset' not in ov: notes.append('course offset not visible in the photo; half bond assumed')
    if 'bond' not in ov and re.search(r'\bstack\b', rec['item'] + ' ' + rec['product'], re.I) and bond != 'stack':
        bond, frac = 'stack', 0.0; notes.append('stack bond per the product name')
    if ov.get('bond') in ('stack', 'running'): bond = ov['bond']; frac = 0.0 if bond == 'stack' else ov.get('offset', 0.5)
    if 'offset' in ov: frac = ov['offset']
    c, p = an['c'], an['p']; K = len(an['courses'])
    L, S = nom['L'] * IN, nom['S'] * IN
    mmx0, mmy0 = (L + 1.6) / p, (S + 1.6) / c
    gpx = ov.get('grout_px') or grout_width(g, an, mmy0)
    gmm = ov.get('grout_mm') or (round(float(np.clip(gpx * (mmx0 + mmy0) / 2, 1.2, 10)), 1) if gpx else 1.6)
    if not gpx: notes.append('grout width not measurable; 1/16 in assumed')
    g_ax = (L * c - S * p) / (p - c) if an['p_how'] in ('measured', 'bond offsets') and p > 1.3 * c else None   # (L+g)/p = (S+g)/c
    mortar = re.search(r'clay|brick', (rec['item'] + ' ' + rec['product'] + ' ' + (rec['material'] or '')), re.I)
    if mortar and g_ax and gmm + 2 < g_ax < 14 and not ov.get('grout_mm'):
        notes.append(f'mortar joint {g_ax:.1f} mm from the two axes (profile read {gmm} mm)'); gmm = round(float(g_ax), 1)
    mmx, mmy = (L + gmm) / p, (S + gmm) / c
    # lighting: fit on tile pixels only
    mask = np.ones(g.shape, bool)
    for a, b in an['courses']:
        for y in (a, b): mask[max(0, int(y - 3)): int(y + 4)] = False
    nvis = (Ws / p) * (Hs / c)
    lin = np.clip(flatten(lin, mask, 2 if nvis >= 6 else 1), 0, 1.5)
    # cells: whole tiles between joints, near-whole tiles cut by the picture edge, pairs of end pieces
    full, lpart, rpart = [], [], []
    for k, ((a, b), js) in enumerate(zip(an['courses'], an['joints'])):
        js = sorted(js)
        for x0, x1 in zip(js[:-1], js[1:]):
            if 0.85 * p < x1 - x0 < 1.15 * p: full.append((k, x0, x1, a, b))
        if js:
            if js[0] >= 0.93 * p: full.append((k, js[0] - p, js[0], a, b))
            elif js[0] > 0.2 * p: lpart.append((k, 0.0, js[0], a, b))
            if Ws - js[-1] >= 0.93 * p: full.append((k, js[-1], js[-1] + p, a, b))
            elif Ws - js[-1] > 0.2 * p: rpart.append((k, js[-1], float(Ws), a, b))
    if not full and an['p_how'] == 'aspect' and Ws >= 0.97 * p:      # joints invisible (flutes, seamless glaze)
        for k, ((a, b), js) in enumerate(zip(an['courses'], an['joints'])):
            if js: continue
            nseg = max(1, int(Ws // p)); x = (Ws - nseg * p) / 2
            for j in range(nseg): full.append((k, max(0.0, x + j * p), min(float(Ws), x + (j + 1) * p), a, b))
        if full: notes.append('no joints visible in the photo; tiles cut from the courses at the nominal length')
    # end pieces cut by the picture edge: a tile = start piece (right of a joint) + end piece (left of a joint)
    # [+ an interior section of another course when the tile is longer than the photo], joined with a narrow feather
    gi = g[mask] if mask.any() else g
    plain = float(np.std((g - ndi.gaussian_filter(g, 0.15 * c))[mask]) / (gi.mean() + 1e-6)) < 0.012   # featureless glaze
    smax = 1.6 if plain else 1.12          # stretching pieces is invisible on a plain glaze
    seglen = lambda segs: sum(q[2] - q[1] for q in segs)
    pairs, lp_left = [], list(lpart)
    for rp in rpart:
        cand = sorted(lp_left, key=lambda q: (q[0] != rp[0], abs(q[0] - rp[0])))
        if cand: pairs.append(('chain', [rp, cand[0]])); lp_left.remove(cand[0])
    if full: pairs = [t for t in pairs if seglen(t[1]) >= p / min(smax, 1.15)]
    long_ = [t for t in pairs if seglen(t[1]) < p / smax]; pairs = [t for t in pairs if seglen(t[1]) >= p / smax]
    if not full and not pairs and long_:     # tile longer than the photo: add an interior section from another course
        inter = []
        for k, ((a_, b_), js) in enumerate(zip(an['courses'], an['joints'])):
            js = sorted(js); m_ = 0.06 * p
            edges_ = [0.0] + js + [float(Ws)]
            for x0, x1 in zip(edges_[:-1], edges_[1:]):
                x0 += m_ if x0 > 0 else 0; x1 -= m_ if x1 < Ws else 0
                if x1 - x0 > 0.15 * p: inter.append((k, x0, x1, a_, b_))
        inter.sort(key=lambda q: -(q[2] - q[1]))
        for i, t in enumerate(long_):
            ks = {q[0] for q in t[1]}; need = p - seglen(t[1]) + 2 * 0.08 * p
            mid = next((q for q in inter if q[0] not in ks), inter[i % len(inter)] if inter else None)
            if mid is None: continue
            if mid[2] - mid[1] > need: x0 = mid[1] + (mid[2] - mid[1] - need) / 2; mid = (mid[0], x0, x0 + need, mid[3], mid[4])
            segs = [t[1][0], mid, t[1][1]]
            if seglen(segs) >= p / smax: pairs.append(('chain', segs))
        if pairs: notes.append(f'tile longer than the photo: {len(pairs)} tiles composed of start + interior + end sections')
    if not full and not pairs: raise Fail('no complete tile visible')
    # texture size: >= 2 tiles x 4 courses, >= ~600 x 300 mm where the cap of 2048 px allows, whole bond repeats
    kb = 1 if (bond == 'stack' or frac is None) else next((d for d in (2, 3, 4) if abs(frac * d - round(frac * d)) < 0.02), 2)
    nu = len(full) + len(pairs)            # unique tiles available
    mmin = int(np.clip(math.ceil(400 / (S + gmm)), 4, 12))           # repeat >= ~600 x 400 mm: a small repeat of a
    m = ov.get('m') or -(-mmin // kb) * kb                                # varied blend reads as stripes on a wall
    if frac is None: m = ov.get('m') or int(np.clip(max(K, mmin), 4, 12))
    n = ov.get('n') or int(np.clip(math.ceil(600 / (L + gmm)), 1, 6))
    while n > 1 and 2048 / (n * (L + gmm)) < 2.0: n -= 1                  # keep >= 2 px/mm under the 2048 px cap
    while m > max(4, kb) and 2048 / (m * (S + gmm)) < 2.0: m -= kb
    ppmm = min(1 / mmx, 1 / mmy, 2048 / (n * (L + gmm)), 2048 / (m * (S + gmm)))
    pout, cout = int(round((L + gmm) * ppmm)), int(round((S + gmm) * ppmm))
    # assignment: unique tiles first (output course i <- photo course i % K keeps neighbours), then seeded shuffle
    # of the pool with mirrored copies, never the same tile as the left or upper neighbour
    pool = [('full', cl) for cl in full] + pairs; used = set(); tiles = []; reuse = 0
    flat_ = edge_profile(rec, ov) == 'flat'      # baked bevel / pillow shading must keep its light direction
    rng = np.random.default_rng(int.from_bytes(rec['sku'].encode()[:8].ljust(8, b'_'), 'little') % 2 ** 32)
    bycourse = {}
    for t in pool: bycourse.setdefault((t[1] if t[0] == 'full' else t[1][0])[0], []).append(t)
    for i in range(m):
        k = i % K; row = [t for t in bycourse.get(k, []) if id(t) not in used]; seq = []
        for j in range(n):
            t = row[j] if j < len(row) else next((q for q in pool if id(q) not in used), None)
            rot = 0
            if t is None:
                bad = {id(seq[-1][0])} if seq else set()
                if i: bad.add(id(tiles[-1][j][0]))
                cand = [q for q in pool if id(q) not in bad] or pool
                t = cand[int(rng.integers(len(cand)))]; reuse += 1
                rot = int(rng.integers(1, 4)) if flat_ else 2       # 180 deg / mirror x / mirror y (relief: x only)
            used.add(id(t)); seq.append((t, rot))
        tiles.append(seq)
    stretch = []
    def cell_img(t, rot):
        if t[0] == 'full':
            _, x0, x1, a, b = t[1]
            ya = refine_line(gy, a, max(0, x0), min(Ws, x1), c); yb = refine_line(gy, b, max(0, x0), min(Ws, x1), c)
            im_ = sample(lin, x0, x1, ya, yb, pout, cout)
        else:
            segs = t[1]; nj = len(segs) - 1; bl = max(3, int(0.025 * pout))
            ws = np.array([max(2.0, (q[2] - q[1]) / p * pout) for q in segs]); need = pout + nj * bl
            if ws.sum() < need: stretch.append(need / ws.sum()); ws *= need / ws.sum()
            ovl = (ws.sum() - pout) / max(1, nj)          # overlap per join (>= bl)
            pcs = [sample(lin, q[1], q[2], q[3], q[4], int(round(w)), cout) for q, w in zip(segs, ws)]
            mu = [pc[int(0.2 * cout): int(0.8 * cout)].reshape(-1, 3).mean(0) + 1e-4 for pc in pcs]; mall = np.mean(mu, 0)
            pcs = [pc * np.clip(mall / m_, 0.85, 1.15) for pc, m_ in zip(pcs, mu)]     # pieces lit differently: match
            im_ = pcs[0]
            for B in pcs[1:]:
                o = int(round(ovl)); Aw = im_.shape[1]; u = np.arange(Aw + B.shape[1] - o); S_ = Aw - o / 2
                t_ = np.clip((u - (S_ - bl / 2)) / bl, 0, 1)[None, :, None]
                im_ = im_[:, np.clip(u, 0, Aw - 1)] * (1 - t_) + B[:, np.clip(u - (Aw - o), 0, B.shape[1] - 1)] * t_
            im_ = sample(im_, 0, im_.shape[1], 0, cout, pout, cout) if im_.shape[1] != pout else im_
        return [im_, im_[::-1, ::-1], im_[:, ::-1], im_[::-1]][rot]
    Wt, Ht = n * pout, m * cout
    out = np.zeros((Ht, Wt, 3), np.float32); hcourses = []; cells_rect = []; offs = []
    k0 = ph[0] if ph and ph[0] is not None else 0
    for i, seq in enumerate(tiles):
        strip = np.concatenate([cell_img(t, rot) for t, rot in seq], 1)
        if frac is None: k = i % K; f = ((ph[k] if ph[k] is not None else 0) - k0) % 1.0
        else: f = (i * frac) % 1.0
        off = int(round(f * pout)) % Wt
        out[i * cout:(i + 1) * cout] = np.roll(strip, off, 1)
        js = [(off + j * pout) % Wt for j in range(n)]
        hcourses.append((i * cout, (i + 1) * cout, js)); offs.append(round(f, 3))
        for j in js:
            cells_rect.append((i * cout, (i + 1) * cout, j, min(Wt, j + pout)))
            if j + pout > Wt: cells_rect.append((i * cout, (i + 1) * cout, 0, j + pout - Wt))
    mmpp = 1 / ppmm; gpo = gmm * ppmm
    h = height_cells(Wt, Ht, gpo, mmpp, edge_profile(rec, ov), hcourses)
    body = h > -1.0
    if fin in ('Glossy', 'Polished', 'Glossy (glass)'):
        stone = re.search(r'marble|travertine|stone|glass', (rec['material'] or '') + rec['item'], re.I)
        out = compress_glints(out, cells_rect, body, 0.0 if stone else 0.75, 1.10, 0.35)
    if reuse: notes.append(f'{reuse} of {n * m} tiles reused (photo shows fewer whole tiles)')
    nsyn = len({id(t) for seq in tiles for t, _ in seq if t[0] == 'chain'})
    if nsyn: notes.append(f'{nsyn} tiles joined from end pieces' + (f', stretched up to {max(stretch) * 100 - 100:.0f}%' if stretch else ''))
    if frac is None: notes.append('random course offsets as photographed')
    if an['p_how'] == 'half-bond offsets': notes.append('joint pitch from the half-bond offset of neighbouring courses')
    if an['p_how'] == 'bond offsets': notes.append('joint pitch from the course-to-course joint offsets (1/2, 1/3 or 1/4 bond near the nominal length)')
    if an['spread'] and an['spread'] > 0.08: notes.append(f"tile lengths vary {an['spread'] * 100:.0f}% in the photo")
    if an['p_how'] == 'aspect': notes.append('joint pitch not measurable (fewer than 2 joints per course); from course pitch x nominal aspect')
    tex = dict(period_mm=[round(n * (L + gmm), 2), round(m * (S + gmm), 2)], bond=bond, grout_mm=gmm,
               offset=(None if frac is None else round(frac, 3)), course_offsets=offs if frac is None else None)
    marks = [('h', a) for a, b in an['courses']] + [('h', an['courses'][-1][1])]
    marks += [('v', x, a, b) for (a, b), js in zip(an['courses'], an['joints']) for x in js]
    return dict(lin=out, h=h, mmpp=mmpp, tex=tex, body=body, method='grout_period' if an['p_how'] != 'aspect' else 'course_pitch+aspect',
                check=dict(mm_per_px_x=round(mmx, 4), mm_per_px_y=round(mmy, 4),
                           disagree=(round(mmx / mmy - 1, 3) if an['p_how'] in ('measured', 'half-bond offsets', 'bond offsets') else None),
                           joints_missing=(an['contradict'] if an['p_how'] != 'measured' else None), grout_from_axes_mm=(round(g_ax, 1) if g_ax else None), fit_score=an['score'], length_spread=(round(an['spread'], 3) if an['spread'] is not None else None),
                           pitch_px=[round(p, 2), round(c, 2)], joint_pitch=an['p_how'], bond_detected=bond,
                           offset_detected=(None if frac is None else round(frac, 3)), courses=K, whole_tiles=len(full), unique_tiles=nu,
                           ),
                overlay=dict(img=to_u8(np.clip(lin, 0, 1)), marks=marks))

STRUCT = re.compile(r'&| and |hive|petal|floral|check|chex|moderno|tuxe|weave|teardrop|victorian|\bbw\b', re.I)

def make_lattice(im, rec, nom, ov, notes, fin, series_scale):
    """Mosaic sheets: chip lattice from the 2-D autocorrelation of the edge map, colour-pattern repeat as a multiple
    of its axis-aligned repeat, exact-repeat crop with grout-following wrap seams."""
    rgb = np.asarray(im, np.float32) / 255; lin = s2l(rgb); g = rgb.mean(2)
    if ov.get('rotate'): lin = np.clip(rotate(lin, ov['rotate']), 0, 1); g = rotate(g, ov['rotate'])
    ang = deskew(g)
    if abs(ang) > 0.05: lin = np.clip(rotate(lin, ang), 0, 1); g = rotate(g, ang); notes.append(f'deskew {ang:+.2f} deg')
    H, W = g.shape; sh = nom['shape']
    sg = max(1.5, min(H, W) / 250)          # wash out surface grain / JPEG blocks; chips are >= ~40 px
    e = np.hypot(*edges(g, 1.0)); e = ndi.gaussian_filter(e / (e.mean() + 1e-6), sg)
    a = acf2c(e); lb = lattice_basis(a, rmin=max(12, 5 * sg))
    if not lb: raise Fail('no regular chip lattice found')
    b1, b2, hv = lb
    Cx = axis_repeat(b1, b2, a, 0); Cy = axis_repeat(b1, b2, a, 1)
    if ov.get('chip_px'): Cx, Cy = ov['chip_px']
    if not Cx or not Cy: raise Fail('chip lattice has no axis-aligned repeat')
    v1 = float(np.hypot(*b1)); v2 = float(np.hypot(*b2)); angle = math.degrees(math.acos(abs(b1 @ b2) / (v1 * v2)))
    check = dict(chip_basis_px=[[round(float(x), 2) for x in b1], [round(float(x), 2) for x in b2]], chip_angle=round(angle, 1),
                 chip_repeat_px=[round(Cx, 2), round(Cy, 2)])
    # lighting: fit on chip interiors of the brighter colour (a two-colour sheet would otherwise fit its pattern)
    Ll = lum(lin); m_ = (e < np.percentile(e, 60)) & (Ll > np.percentile(Ll, 40))
    lin = np.clip(flatten(lin, m_, 2 if 6 * v1 < min(H, W) else 1), 0, 1.5)
    # colour pattern repeat: smallest multiples of the chip repeat where the colour map correlates
    col = np.stack([ndi.gaussian_filter(lin[..., k], max(1.0, 0.12 * v1)) for k in range(3)], -1)
    var = col.reshape(-1, 3).var(0); cv = float(np.sqrt(var.sum()) / (col.mean() + 1e-6))
    structured = bool(STRUCT.search(rec['item'] + ' ' + rec['product'])) or cv > 0.6
    kx = ky = 1
    if ov.get('period_px'): Tx, Ty = ov['period_px']
    else:
        if cv > 0.06:
            q_ = 2 if min(H, W) > 600 else 1; cq = col[::q_, ::q_]; Hq, Wq = cq.shape[:2]
            def corr(dx, dy):           # Pearson correlation of the colour map with itself shifted by (dx, dy)
                dx, dy = int(round(dx / q_)), int(round(dy / q_))
                if dx >= Wq - 4 or dy >= Hq - 4: return -1
                A_ = cq[dy:, dx:]; B_ = cq[:Hq - dy, :Wq - dx]
                A_ = A_ - A_.mean((0, 1)); B_ = B_ - B_.mean((0, 1))
                return float((A_ * B_).sum() / (np.sqrt((A_ * A_).sum() * (B_ * B_).sum()) + 1e-9))
            def best_k(C, n, dx):        # smallest multiple that correlates, refined over +-2 px (lattice drift)
                for k in range(1, 30):
                    if k * C + 0.6 * v1 > 0.99 * n: return None
                    cs = [(corr(k * C + d, 0) if dx else corr(0, k * C + d), d) for d in range(-2 * q_, 2 * q_ + 1, q_)]
                    c_, d = max(cs)
                    if c_ >= thr: return k, k * C + d
                return None
            thr = 0.6 if structured else 0.8
            rx = best_k(Cx, W, True); ry = best_k(Cy, H, False)
            check['colour_corr'] = [round(corr(Cx, 0), 2), round(corr(0, Cy), 2)]
            if rx and ry: (kx, Tx), (ky, Ty) = rx, ry
            elif structured: raise Fail('the colour pattern does not repeat within the photo (one repeat + seam band does not fit)')
            else: Tx, Ty = Cx, Cy; notes.append('random colour blend: chip repeat used, seams cut along grout')
        else: Tx, Ty = Cx, Cy
    check.update(colour_cv=round(cv, 3), pattern_mult=[kx, ky], structured=structured)
    # scale
    g_in = 1 / 16; method = 'chip_pitch'
    if ov.get('px_per_in'): ppi = ov['px_per_in']; method = 'override'
    elif sh in ('hexagon', 'penny', 'octagon', 'checker') and nom['L']:
        pitch = v1
        if sh == 'checker' and abs(abs(b1[0]) - abs(b1[1])) < 0.1 * v1: pitch = v1 / math.sqrt(2)   # diagonal checker lattice
        ppi = pitch / (nom['L'] + g_in)
        want = {'hexagon': 60, 'penny': 60, 'octagon': 90, 'checker': 90}[sh]
        if abs(angle - want) > 8: notes.append(f'chip lattice angle {angle:.0f} deg (expected {want}) - check scale')
        if sh == 'checker' and abs(angle - 90) > 8: raise Fail('checker photo does not show a square chip lattice')
    elif nom['L'] and nom['S']:              # pickets etc. with a stated chip size: course pitch = short side + grout
        gx_, gy_ = edges(g, 1.0); best_c = None
        for prof in (gy_.mean(1), gx_.mean(0)):
            for T, hh in acf_peaks(acf1(prof), 12, 0.6 * min(H, W), k=3):
                if best_c is None or hh > best_c[1]: best_c = (T, hh)
        if not best_c: raise Fail('no chip courses found')
        ppi = best_c[0] / (nom['S'] + g_in); method = 'course_pitch'
        check['course_px'] = round(best_c[0], 1)
        if (nom['L'] + g_in) * ppi > 0.5 * max(H, W): raise Fail(f'a {nom["L"]:g} in chip spans more than half the photo; no exact repeat fits')
        if abs(min(v1, best_c[0]) / max(v1, best_c[0]) - 1) > 0.15 and abs(v1 / best_c[0] - 2) > 0.3:
            raise Fail(f'chip lattice ({v1:.0f} px) does not match the course pitch ({best_c[0]:.0f} px)')
    elif series_scale:
        ppi = max(H, W) / series_scale[0]; method = 'series_scale'
        notes.append(f'scale from photo series {series_scale[1]}: photo spans {series_scale[0]:.1f} in ({ppi:.1f} px/in)')
    else: raise Fail('no size for this pattern and no photo-series scale')
    if series_scale and method == 'chip_pitch' and abs(ppi * series_scale[0] / max(H, W) - 1) > 0.25:
        notes.append(f'chip-pitch scale {ppi:.1f} px/in differs from the photo series ({max(H, W) / series_scale[0]:.1f})')
    mmpp = IN / ppi
    # crop: whole repeats plus a seam band in front of the crop
    ex = max(abs(b1[0]), abs(b2[0]), 0.6 * v1); ey = max(abs(b1[1]), abs(b2[1]), 0.6 * v1)   # chip extent per axis
    bx = int(np.clip(0.6 * ex, 8, 0.5 * Tx)); by = int(np.clip(0.6 * ey, 8, 0.5 * Ty))
    nx = ov.get('n') or int((W * 0.99 - bx) // Tx); ny = ov.get('m') or int((H * 0.99 - by) // Ty)
    nx, ny = max(1, min(nx, int(2048 // Tx))), max(1, min(ny, int(2048 // Ty)))
    if nx * Tx + bx > W or ny * Ty + by > H: raise Fail('pattern repeat larger than the photo allows for a seamless crop')
    x0 = bx + (W - bx - nx * Tx) / 2; y0 = by + (H - by - ny * Ty) / 2
    out = lattice_crop(lin, Tx, Ty, nx, ny, x0, y0, bx, by)
    # grout / chip mask from the edge ridge of the texture itself
    gl = lum(out); ee = ndi.gaussian_filter(np.hypot(*edges(gl, 0.8)), 1.0)
    ridge = ee / (np.percentile(ee, 97) + 1e-6)
    hgt = -1.2 * np.clip(ridge, 0, 1); body = ridge < 0.35
    if fin in ('Glossy', 'Polished', 'Glossy (glass)'):
        if not structured:                   # window reflections span several chips: pull bright large-scale excursions
            lg = np.log(lum(out) + 1e-3); med = np.median(lg[body])
            num = ndi.gaussian_filter(np.where(body, lg, 0), 0.7 * v1, mode='wrap')
            den = ndi.gaussian_filter(body.astype(np.float32), 0.7 * v1, mode='wrap') + 1e-6
            out = out * np.exp(-0.7 * np.clip(num / den - med, 0, None))[..., None]
        Lc = lum(out); med = np.median(Lc[body]); t = 1.15 * med
        Ln = np.where(Lc > t, t + (Lc - t) * 0.5, Lc); out = out * (Ln / (Lc + 1e-6))[..., None]
    bond = ov.get('bond') or {'herringbone': 'herringbone'}.get(sh, 'pattern' if structured or sh not in ('hexagon', 'penny') else 'mosaic')
    tex = dict(period_mm=[round(out.shape[1] * mmpp, 2), round(out.shape[0] * mmpp, 2)], bond=bond, grout_mm=round(g_in * IN, 1),
               offset=None, course_offsets=None)
    check.update(px_per_in=round(ppi, 2), repeats=[nx, ny], repeat_px=[round(Tx, 2), round(Ty, 2)])
    return dict(lin=out, h=hgt, mmpp=mmpp, tex=tex, body=body, method=method, check=check,
                overlay=dict(img=to_u8(np.clip(lin, 0, 1)), rect=(x0, y0, x0 + nx * Tx, y0 + ny * Ty), T=(Tx, Ty)))

def make_synth(im, rec, nom, ov, notes):
    """One piece photographed on white: lay it in running bond at the nominal size with 1/16 in grout."""
    if not nom['L']: raise Fail('single piece on white and no nominal size')
    a = np.asarray(im, np.float32); lin = s2l(a / 255)
    if ov.get('piece'):      # one whole tile picked by hand from a laid-up photo (after the optional rotate)
        if ov.get('rotate'): lin = np.clip(rotate(lin, ov['rotate']), 0, 1); notes.append(f'photo turned {ov["rotate"]:+g} deg')
        x0, y0, x1, y1 = ov['piece']; notes.append('single tile picked by hand from the laid-up photo')
    else:
        bg = (a.min(2) > 232) & (np.ptp(a, 2) < 14)
        fg = ndi.binary_opening(~bg, iterations=2); lab, nl = ndi.label(fg)
        if not nl: raise Fail('piece not found on the white background')
        big = np.argmax(ndi.sum(fg, lab, range(1, nl + 1))) + 1; ys, xs = np.where(lab == big)
        # straighten using the principal axis of the piece
        cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()])); ev, evec = np.linalg.eigh(cov)
        ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        if abs(ang) > 0.2: lin = np.clip(rotate(lin, ang), 0, 1); notes.append(f'piece straightened {ang:+.1f} deg')
        m2 = rotate((lab == big).astype(np.float32), ang) > 0.5
        ys, xs = np.where(m2)
        x0, x1 = np.percentile(xs, [1.5, 98.5]); y0, y1 = np.percentile(ys, [1.5, 98.5])
    ix = (x1 - x0) * 0.03; iy = (y1 - y0) * 0.06          # inset past the rounded / chamfered edge
    x0, x1, y0, y1 = x0 + ix, x1 - ix, y0 + iy, y1 - iy
    if (x1 - x0) < (y1 - y0): lin = lin.transpose(1, 0, 2); x0, x1, y0, y1 = y0, y1, x0, x1
    L, S = nom['L'] * IN, nom['S'] * IN
    meas = (x1 - x0) / (y1 - y0); asp = L / S
    if abs(meas / asp - 1) > 0.25: raise Fail(f'photographed piece has aspect {meas:.2f}, nominal {nom["L"]:g}x{nom["S"]:g} is {asp:.2f} (another product?)')
    thin_brick = 'clay' in (rec['item'] + rec['product']).lower() or 'brick' in rec['item'].lower()
    gmm = ov.get('grout_mm', 1.6)
    ppmm = min(4.0, (x1 - x0) / L)
    tw, th = int(round(L * ppmm)), int(round(S * ppmm)); gp = max(1, int(round(gmm * ppmm)))
    pw, ph = tw + gp, th + gp
    n, m = ov.get('n', 3), ov.get('m', 6)
    W, H = n * pw, m * ph
    grout = np.array(s2l(np.array([0.78, 0.76, 0.72])), np.float32)    # warm light grey
    out = np.empty((H, W, 3), np.float32); out[:] = grout
    tile = sample(lin, x0, x1, y0, y1, int((x1 - x0)), int((y1 - y0)))
    rng = np.random.default_rng(abs(hash(rec['sku'])) % 2 ** 32)
    th_, tw_ = tile.shape[:2]; cells = []
    for i in range(m):
        off = (pw // 2) * (i % 2)
        for j in range(n):
            # a different window of the photographed piece for each tile, mirrored / turned at random
            fw, fh = 0.86, 0.86
            sx = rng.uniform(0, 1 - fw) * tw_; sy = rng.uniform(0, 1 - fh) * th_
            t = sample(tile, sx, sx + fw * tw_, sy, sy + fh * th_, tw, th)
            if rng.random() < 0.5: t = t[:, ::-1]
            if rng.random() < 0.5: t = t[::-1]
            t = t * rng.uniform(0.96, 1.04)
            y = i * ph + gp // 2; x = (off + j * pw + gp // 2) % W
            xs_ = (np.arange(tw) + x) % W
            out[y:y + th][:, xs_] = t; cells.append((y, y + th, x, min(W, x + tw)))
    mmpp = 1 / ppmm
    hc = [(i * ph, (i + 1) * ph, [((pw // 2) * (i % 2) + j * pw) % W for j in range(n)]) for i in range(m)]
    # grout centres sit at multiples of the pitch, shifted by half a grout from the tile placement above
    out = np.roll(np.roll(out, -(gp // 2), 0), -(gp // 2), 1)
    h = height_cells(W, H, gp, mmpp, 'pillow', hc)
    body = h > -1.0
    notes.append('synthesised field from a single photographed piece (running bond, 1/16 in warm grey grout)')
    if thin_brick: notes.append('thin brick: real installs often use wider mortar joints; grout_mm override available')
    tex = dict(period_mm=[round(n * (L + gp * mmpp), 2), round(m * (S + gp * mmpp), 2)], bond='running', grout_mm=round(gp * mmpp, 1),
               offset=0.5, course_offsets=None)
    return dict(lin=out, h=h, mmpp=mmpp, tex=tex, body=body, method='nominal_single_piece',
                check=dict(piece_aspect=round(meas, 3), nominal_aspect=round(asp, 3)),
                overlay=dict(img=to_u8(np.clip(lin, 0, 1)), rect=(x0, y0, x1, y1)))

def make_picket(im, rec, nom, ov, notes):
    """Pickets (elongated hexagons, 45 deg points) whose photo shows less than one repeat: columns of pickets, the
    neighbouring column half a course lower so the points nest; each picket's face joins two photographed picket bodies
    (straight middle parts, true scale) cross-faded at mid-length. Grout colour is read off the photo."""
    if not (nom['L'] and nom['S']): raise Fail('picket without a nominal size')
    rgb = np.asarray(im, np.float32) / 255; lin = s2l(rgb); g = rgb.mean(2)
    ang = deskew(g)
    if abs(ang) > 0.05: lin = np.clip(rotate(lin, ang), 0, 1); g = rotate(g, ang); notes.append(f'deskew {ang:+.2f} deg')
    Hs, Ws = g.shape; xa, xb = int(0.2 * Ws), int(0.8 * Ws)            # the straight body of the middle picket
    gy = edges(g, 1.0)[1][:, xa:xb]; prof = gy.mean(1)
    pk = acf_peaks(acf1(prof), 0.08 * Hs, 0.45 * Hs, k=3)
    if not pk: raise Fail('no picket courses found')
    c = max(pk, key=lambda t: t[1])[0]
    ls, _ = lines_at(prof, c, max(1.2, 0.012 * c)); ys = [y for y, _ in ls]
    wins = [(a + 0.14 * c, b - 0.14 * c) for a, b in zip(ys[:-1], ys[1:]) if 0.8 * c < b - a < 1.2 * c]
    if not wins: raise Fail('no whole picket course found')
    gmask = np.zeros(g.shape, bool)
    for y in ys: gmask[max(0, int(y - 0.02 * c)): int(y + 0.02 * c) + 1, xa:xb] = True
    gcol = np.median(lin[gmask], 0) if gmask.any() else np.array(s2l(np.array([0.78, 0.76, 0.72])), np.float32)
    body0 = np.zeros(g.shape, bool)
    for a, b in wins: body0[int(a):int(b), xa:xb] = True
    lin = np.clip(flatten(lin, body0, 1), 0, 1.5)
    L, S = nom['L'] * IN, nom['S'] * IN; gmm = ov.get('grout_mm', 1.6); C = S + gmm
    P = L - C / 2 + gmm * math.sqrt(2)              # column pitch: 45 deg points nest with a g-wide joint
    ppm_photo = c / C; Lpx = L * ppm_photo; wlen = xb - xa
    m = ov.get('m', 6); ppmm = ov.get('ppmm', 2.0); mmpp = 1 / ppmm
    W, H = int(round(2 * P * ppmm)), int(round(m * C * ppmm))
    X, Y = np.meshgrid((np.arange(W) + 0.5) * 2 * P / W, (np.arange(H) + 0.5) * m * C / H)
    best = np.full((H, W), -1e9, np.float32); kid = np.zeros((H, W), int); U = np.zeros((H, W), np.float32); V = np.zeros_like(U)
    for k in (0, 1):                               # two columns per period; offsets wrap, so neighbours across a seam count
        for j in range(m):
            cx, cy = k * P, j * C + (C / 2 if k else 0)
            dx = (X - cx + P) % (2 * P) - P
            dy = (Y - cy + m * C / 2) % (m * C) - m * C / 2
            d = np.minimum(S / 2 - np.abs(dy), (L / 2 - np.abs(dx) - np.abs(dy)) / math.sqrt(2))
            upd = d > best; best = np.where(upd, d, best)
            kid = np.where(upd, k * m + j, kid); U = np.where(upd, dx / L + 0.5, U); V = np.where(upd, dy / S + 0.5, V)
    rng = np.random.default_rng(abs(hash(rec['sku'])) % 2 ** 32)
    out = np.empty((H, W, 3), np.float32); out[:] = gcol
    for t in range(2 * m):
        sel = kid == t
        if not sel.any(): continue
        # two halves from two photographed bodies at true length, cross-faded over the middle 16 %
        u = U[sel] * Lpx; v = V[sel]; v = 1 - v if rng.random() < 0.5 else v
        half = 0.58 * Lpx; smp = []
        for side in (0, 1):
            a, b = wins[rng.integers(len(wins))]
            o = rng.uniform(0, max(0.0, wlen - half)); xs_ = xa + o + (u if side == 0 else u - (Lpx - half)) * min(1.0, wlen / half)
            smp.append(np.stack([ndi.map_coordinates(lin[..., q], [a + v * (b - a), xs_], order=1, mode='mirror') for q in range(3)], -1))
        wgt = np.clip((u / Lpx - 0.42) / 0.16, 0, 1)[:, None]
        out[sel] = (smp[0] * (1 - wgt) + smp[1] * wgt) * rng.uniform(0.97, 1.03)
    out = np.where((best < 0)[..., None], gcol, out)       # best: mm inside the nominal outline; < 0 is grout
    h = tile_profile(best, edge_profile(rec, ov), best, S)
    notes.append(f'picket layout synthesised: columns {P:.1f} mm apart, neighbours half a course offset; faces from {len(wins)} photographed pickets, grout colour from the photo')
    tex = dict(period_mm=[round(2 * P, 2), round(m * C, 2)], bond='pattern', grout_mm=round(gmm, 1), offset=None, course_offsets=None)
    ov_img = to_u8(np.clip(lin, 0, 1))
    return dict(lin=out, h=h, mmpp=mmpp, tex=tex, body=best >= 0, method='nominal_picket',
                check=dict(course_px=round(float(c), 1), pickets_sampled=len(wins)),
                overlay=dict(img=ov_img, rect=(xa, wins[0][0], xb, wins[-1][1])))

# ------------------------------------------------------------------ QA contact sheets
def _font(sz):
    for f in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Helvetica.ttc', '/Library/Fonts/Arial.ttf'):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()

def qa_row(e, art, RH=300):
    F, Fs = _font(15), _font(12)
    row = Image.new('RGB', (1720, RH), (246, 246, 244)); d = ImageDraw.Draw(row)
    ov = art.get('overlay') or {}
    src = ov.get('img') if isinstance(ov, dict) and ov.get('img') is not None else art.get('photo')
    if src is not None:
        f = min((RH - 8) / src.height, 300 / src.width); ph = src.resize((max(1, int(src.width * f)), max(1, int(src.height * f))))
        dd = ImageDraw.Draw(ph)
        if isinstance(ov, dict):
            for mk in ov.get('marks', []):
                if mk[0] == 'h': dd.line([(0, mk[1] * f), (ph.width, mk[1] * f)], fill=(255, 40, 40), width=1)
                else: dd.line([(mk[1] * f, mk[2] * f), (mk[1] * f, mk[3] * f)], fill=(0, 200, 255), width=1)
            if ov.get('rect'):
                x0, y0, x1, y1 = [v * f for v in ov['rect']]; dd.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=2)
        row.paste(ph, (4, 4))
    if art.get('tex') is not None and e['tex']:
        col = art['tex']; pw, phh = e['tex']['period_mm']
        s = min(440 / (3 * pw), (RH - 30) / (3 * phh))           # px per mm on the sheet
        tw, th = max(1, int(round(pw * s))), max(1, int(round(phh * s)))
        t = col.resize((tw, th), Image.Resampling.LANCZOS)
        big = Image.new('RGB', (3 * tw, 3 * th))
        for i in range(3):
            for j in range(3): big.paste(t, (i * tw, j * th))
        row.paste(big, (312, 4))
        r6 = 6 * IN * s; y = 4 + 3 * th + 8; x = 312
        d.rectangle([x, y, x + r6, y + 6], fill=(30, 30, 30))
        for k in range(7): d.line([(x + k * r6 / 6, y - 3), (x + k * r6 / 6, y + 6)], fill=(30, 30, 30))
        d.text((x + r6 + 6, y - 5), '6 in', fill=(30, 30, 30), font=Fs)
        # flat lit swatch with the normal map
        n = art['nrm'].resize(col.size, Image.Resampling.BILINEAR)
        cw = min(col.width, int(round(250 / s * s * col.width / (pw * s)))) if False else col.width
        crop = min(col.width, col.height, int(col.width * min(1.0, 160 / pw)))
        cimg = np.asarray(col.crop((0, 0, crop, crop)), np.float32) / 255
        nn = np.asarray(n.crop((0, 0, crop, crop)), np.float32) / 127.5 - 1
        l = np.array([-0.45, 0.55, 0.70]); l /= np.linalg.norm(l)
        dif = np.clip((nn * l).sum(2), 0, 1); hv = l + np.array([0, 0, 1]); hv /= np.linalg.norm(hv)
        spec = np.clip((nn * hv).sum(2), 0, 1) ** (8 + 60 * (1 - e['rough'])) * (0.15 + 0.5 * e['clearcoat'])
        lit = np.clip(cimg * (0.45 + 0.6 * dif)[..., None] + spec[..., None], 0, 1)
        li = Image.fromarray((lit * 255).astype(np.uint8)).resize((200, 200), Image.Resampling.LANCZOS)
        row.paste(li, (770, 4))
        d.rectangle([770, 210, 830, 250], fill=e['swatch']); d.text((836, 222), e['swatch'], fill=(30, 30, 30), font=Fs)
        if e.get('thumb'):
            tb = Image.open(io.BytesIO(base64.b64decode(e['thumb'].split(',')[1]))); row.paste(tb, (880, 205))
        # seam check: the centre of a 2 x 2 tiling at native resolution (all four wrap seams meet there)
        W2, H2 = col.size; big = Image.new('RGB', (2 * W2, 2 * H2))
        for i in range(2):
            for j in range(2): big.paste(col, (i * W2, j * H2))
        q = min(140, W2, H2); sc_ = big.crop((W2 - q, H2 - q, W2 + q, H2 + q)).resize((240, 240), Image.Resampling.NEAREST if q < 120 else Image.Resampling.LANCZOS)
        row.paste(sc_, (982, 4)); d.text((982, 248), 'wrap seams (2x2 centre, 1:1)', fill=(90, 90, 90), font=Fs)
    x = 1232; y = 6
    lines = [f"{e['sku']}  [{e['kind']}{'' if e['usable'] else ' - UNUSABLE'}]", e['name'][:44], f"{e['collection']} | {e['color']} | {e['finish']}",
             f"size {e['size_label']}  tile_mm {e['tile_mm']}"]
    if e['tex']:
        t = e['tex']; lines += [f"bond {t['bond']} off {t.get('offset')}  grout {t['grout_mm']} mm", f"period {t['period_mm']} mm  px {t['px']}",
                                f"scale: {e['scale_method']}", f"rough {e['rough']} cc {e['clearcoat']} relief {e['relief']}"]
        sc = e.get('scale_check') or {}
        if sc.get('disagree') is not None: lines.append(f"mm/px x {sc['mm_per_px_x']} y {sc['mm_per_px_y']} ({sc['disagree'] * 100:+.0f}%)")
        if 'px_per_in' in sc: lines.append(f"{sc['px_per_in']} px/in  repeat {sc.get('repeat_px')}")
    if e['reason']: lines.append('reason: ' + e['reason'])
    for nt in e['notes']: lines.append('- ' + nt)
    for i, ln in enumerate(lines[:17]):
        d.text((x, y + i * 17), ln[:62], fill=(200, 0, 0) if ln.startswith('reason') or 'disagree' in ln or 'UNUSABLE' in ln else (25, 25, 25), font=F if i == 0 else Fs)
    return row

def qa_sheets(rows, prefix='sheet'):
    QA.mkdir(exist_ok=True)
    for f in QA.glob(prefix + '_*.jpg'): f.unlink()
    for k in range(0, len(rows), 12):
        part = rows[k:k + 12]; sh = Image.new('RGB', (1720, 304 * len(part)), 'white')
        for i, r in enumerate(part): sh.paste(r, (0, i * 304))
        sh.save(QA / f'{prefix}_{k // 12:02d}.jpg', quality=82)

def qa_rows_save(rows_by_sku):
    (QA / 'rows').mkdir(parents=True, exist_ok=True)
    for sku, r in rows_by_sku.items(): r.save(QA / 'rows' / (re.sub(r'[^A-Za-z0-9._-]+', '_', sku) + '.jpg'), quality=88)

# ------------------------------------------------------------------ main
def series_key(rec, e):
    """Photos shot the same way: same collection, same image folder, same original pixel size."""
    u = e.get('photo') or ''; sc = e.get('scale_check') or {}
    return f"{rec['collection'] or '?'}|{'/'.join(u.split('/')[4:-1])}|{max(sc.get('photo_orig_px') or [0])}"

def src_span(e):
    """Inches across the analysed photo (its long side) for an entry with a measured scale, else None."""
    sc = e.get('scale_check') or {}
    if not sc.get('photo_px'): return None
    if 'mm_per_px_x' in sc: ppi = IN / ((sc['mm_per_px_x'] + sc['mm_per_px_y']) / 2)
    elif 'px_per_in' in sc and e['scale_method'] not in ('series_scale', 'override'): ppi = sc['px_per_in']
    else: return None
    return max(sc['photo_px']) / ppi

def main():
    args = sys.argv[1:]
    cat = catalogue()
    if '--fetch' in args:
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda r: [u for u in photo_urls(r, load_ov(r['sku'])) if fetch_img(u)], cat))
        print(len(cat), 'skus', sum(bool(r) for r in res), 'with a photo'); return
    only = None
    if '--only' in args: only = set(args[args.index('--only') + 1].split(','))
    TEX.mkdir(exist_ok=True)
    old = {}
    if (MT / 'tiles.json').exists():
        old = {e['sku']: e for e in json.load(open(MT / 'tiles.json'))['tiles']}
    todo = [r for r in cat if only is None or r['sku'] in only]
    if only and len(todo) != len(only): print('unknown SKUs:', only - {r['sku'] for r in todo})
    ents, arts = dict(old), {}
    # pass 1: everything with its own scale; pass 2: patterns that need a photo-series scale
    later = []
    def run(rec, ss=None):
        try: e, a = build(rec, ss)
        except Exception as ex:
            import traceback; traceback.print_exc()
            e = dict(id='msitile_' + slug(rec['sku']), sku=rec['sku'], name=rec['item'], product=rec['product'], url=rec['url'],
                     collection=rec['collection'], color=rec['color'], material=rec['material'], finish=rec['finish'],
                     style=rec['style'], size_label=rec['size'], tile_mm=None, kind='field', usable=False,
                     reason=f'error: {ex}', tex=None, notes=[]); a = {}
        ents[rec['sku']] = e; arts[rec['sku']] = (rec, e, a)
        print(f"{rec['sku']:28s} {e['kind'] or '-':7s} {'ok ' if e['usable'] else 'NO '} {e.get('scale_method') or ''} "
              f"{(e['tex'] or {}).get('period_mm', '')} {e['reason']} {'; '.join(e['notes'])}", flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda r: run(r), todo))
    byser = {}
    for r in cat:
        e = ents.get(r['sku'])
        if e and e.get('usable') and src_span(e) and e['kind'] in ('mosaic',): byser.setdefault(series_key(r, e), []).append(src_span(e))
    for rec in todo:
        e = ents[rec['sku']]
        if not e['usable'] and 'photo-series scale' in (e.get('reason') or ''):
            v = byser.get(series_key(rec, e))
            if v and len(v) >= 2 and (np.percentile(v, 75) - np.percentile(v, 25)) / np.median(v) <= 0.25:
                run(rec, (float(np.median(v)), f"{series_key(rec, e)} n={len(v)}"))
    # write textures
    for sku, (rec, e, a) in arts.items():
        for f in TEX.glob(e['id'] + '.js'): f.unlink()
        for f in TEX.glob(e['id'] + '_n.js'): f.unlink()
        if e['usable'] and a.get('tex') is not None:
            write_tex(e['id'], a['tex'], 86); write_tex(e['id'] + '_n', a['nrm'], 90)
    if only is None:                        # full run: drop textures of earlier runs that no entry references
        # exact ids: a SKU ending in -N has a colour map named msitile_<x>_n.js, which is not a normal map
        keep = {i for e in ents.values() if e.get('usable') for i in (e['id'], e['id'] + '_n')}
        for f in TEX.glob('msitile_*.js'):
            if f.stem not in keep: f.unlink()
    order = [r['sku'] for r in cat]
    allents = [ents[s] for s in order if s in ents]
    for e in allents: e.pop('_art', None)
    out = dict(source='msisurfaces.com/backsplash-tile/subway-tile/ scrape 2026-10-13', generated_by='tools/msi_tiles.py',
               count=len(allents), tiles=allents)
    (MT / 'tiles.json').write_text(json.dumps(out, indent=1))
    rows = {s: qa_row(arts[s][1], arts[s][2]) for s in order if s in arts}
    qa_rows_save(rows); qa_sheets(list(rows.values()), 'sheet' if only is None else 'only')
    from collections import Counter
    print(Counter((e['kind'], e['usable']) for e in allents))

if __name__ == '__main__':
    main()
