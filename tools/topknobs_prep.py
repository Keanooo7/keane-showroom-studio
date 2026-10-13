"""materials/topknobs/catalog.json -> products.json + reference photos for the geometry and finish passes.

products.json, one entry per Top Knobs product (configurable parent):
  slug, name, url, collection, kind_hint ('pull' when the SKUs list a center-to-center, else 'knob'),
  finishes [labels], sizes [{key, label, c2c_mm, length_mm, width_mm, proj_mm, diam_mm, base, metal,
                             ref_sku, ref_img, skus {finish: sku}}], thumb (local list photo)
Dimensions are parsed from the spec table (fractions of an inch -> mm). Missing values are null.

Photos (materials/topknobs/img/):
  ref/<sku>_<n>.jpg      every photo of one reference SKU per size (darkest finish: best silhouette on white)
  thumb/<slug>.jpg       the listing photo
  finish/<fslug>/<sku>.jpg   up to 8 photos per finish, from different products, for colour calibration

    python3 tools/topknobs_prep.py
"""
import json, re, time, concurrent.futures as cf
from fractions import Fraction
from pathlib import Path
from urllib.request import Request, urlopen

R = Path(__file__).resolve().parent.parent
TK = R / 'materials/topknobs'; IMG = TK / 'img'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'
slug = lambda s: re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')
# darkest first: a dark part on the white sweep gives the cleanest silhouette
DARK = ['flat black', 'black', 'sable', 'umbrio', 'oil rubbed bronze', 'tuscan bronze', 'german bronze', 'patine', 'medium bronze',
        'venetian bronze', 'ash gray', 'honey bronze', 'antique bronze', 'brushed bronze', 'warm brass', 'aged brass']

def inches(s):
    """'1 1/4"', '18 in.', '1-1/4"', '33/64"', '7/8" x 7/8"' (first value) -> mm"""
    if not s: return None
    s = s.replace('″', '"').split('x')[0].strip()
    if re.search(r'\d\s*mm\b', s): return float(re.search(r'(\d+(?:\.\d+)?)\s*mm', s).group(1))
    m = (re.match(r'(\d+)[ -]+(\d+)/(\d+)', s) or re.match(r'()(\d+)/(\d+)', s) or re.match(r'(\d+(?:\.\d+)?)()()', s))
    if not m: return None
    v = Fraction(m.group(1) or 0)
    if m.group(2): v += Fraction(int(m.group(2)), int(m.group(3)))
    return round(float(v) * 25.4, 2) if v else None

def mm_field(sp, *keys):
    for k in keys:
        if sp.get(k): return inches(sp[k])
    return None

def get(url, dest):
    if dest.exists() and dest.stat().st_size > 0: return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        try:
            dest.write_bytes(urlopen(Request(url, headers={'User-Agent': UA}), timeout=60).read()); return dest
        except Exception:
            if i == 3: return None
            time.sleep(2 + 2 * i)

cat = json.load(open(TK / 'catalog.json'))
products, jobs, by_finish = [], [], {}
for p in cat['products']:
    if not p['variants']: continue
    ps = slug(p['url'].split('/')[-1].replace('.html', ''))
    sizes = {}
    for v in p['variants']:
        sizes.setdefault(v.get('size') or 'One size', []).append(v)
    out_sizes = []
    for label, vs in sizes.items():
        rank = lambda v: next((i for i, d in enumerate(DARK) if d in (v['finish'] or '').lower()), 99)
        ref = min(vs, key=rank); sp = ref['specs']
        out_sizes.append(dict(key=slug(label) or 'one_size', label=label,
                              c2c_mm=mm_field(sp, 'Center to Center'), length_mm=mm_field(sp, 'Overall Length', 'Length'),
                              width_mm=mm_field(sp, 'Width'), proj_mm=mm_field(sp, 'Projection'), diam_mm=mm_field(sp, 'Diameter'),
                              base=sp.get('Base Diameter'), metal=sp.get('Base Metal'), ref_sku=ref['sku'], ref_finish=ref['finish'],
                              ref_name=ref['name'], specs=sp,
                              ref_img=[f'img/ref/{ref["sku"]}_{i}.jpg' for i in range(len(ref['images']))],
                              skus={v['finish'] or 'Default': v['sku'] for v in vs}))
        for i, u in enumerate(ref['images']): jobs.append((u, IMG / 'ref' / f'{ref["sku"]}_{i}.jpg'))
        for v in vs:
            if v['images']: by_finish.setdefault(v['finish'] or 'Default', []).append((ps, v['sku'], v['images'][0]))
    out_sizes.sort(key=lambda s: (s['c2c_mm'] or 0, s['length_mm'] or 0))
    jobs.append((p['list_img'], IMG / 'thumb' / f'{ps}.jpg'))
    products.append(dict(slug=ps, name=p['name'], url=p['url'], collection=p['variants'][0].get('collection'),
                         kind_hint='pull' if any(s['c2c_mm'] for s in out_sizes) else 'knob',
                         finishes=sorted({v['finish'] or 'Default' for v in p['variants']}), sizes=out_sizes,
                         thumb=f'img/thumb/{ps}.jpg', description=p.get('description')))
for f, lst in by_finish.items():
    seen = set()
    for ps, sku, u in lst:
        if ps in seen: continue
        seen.add(ps); jobs.append((u, IMG / 'finish' / slug(f) / f'{sku}.jpg'))
        if len(seen) >= 8: break

with cf.ThreadPoolExecutor(6) as ex:
    res = list(ex.map(lambda j: get(*j), jobs))
miss = [str(j[1].relative_to(TK)) for j, r in zip(jobs, res) if not r]
(TK / 'products.json').write_text(json.dumps(dict(source=cat['source'], fetched=cat['fetched'], products=products), indent=1))
nsz = sum(len(p['sizes']) for p in products)
print(f'{len(products)} products ({sum(p["kind_hint"] == "pull" for p in products)} pull, {sum(p["kind_hint"] == "knob" for p in products)} knob), '
      f'{nsz} sizes, {len(by_finish)} finishes, {len(jobs)} photos ({len(miss)} failed)')
if miss: print('failed:', miss[:20])
