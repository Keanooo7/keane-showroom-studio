"""Every MSI Q quartz colour on msisurfaces.com/quartz-countertops/quartz-collections/ -> materials/msi_quartz/.

The collections page lists each colour once (name, product page, thumbnail). Each product page carries the full slab
photo (products/slab/large/<slug>.jpg, 3000 px, slab edge to edge) and the specs: primary / accent colours, style
("Marble - Dramatic Veins", "Concrete" ...), finishes, slab size class (Jumbo / Super Jumbo) and price range.
MSI gives no slab dimensions in inches, so the studio maps the photo's width to 126 in, like the Vadara slabs.

    python3 tools/fetch_msi_quartz.py     # photos already in materials/msi_quartz/img/ are not refetched
Writes materials/msi_quartz/quartz.json: name, slug, page, style, primary, accent, finishes, sizes, price, photo, file.
"""
import json, re, html, time, concurrent.futures as cf
from pathlib import Path
from urllib.request import Request, urlopen

R = Path(__file__).resolve().parent.parent
OUT = R / 'materials/msi_quartz'; IMG = OUT / 'img'; IMG.mkdir(parents=True, exist_ok=True)
SITE = 'https://www.msisurfaces.com'
LIST = SITE + '/quartz-countertops/quartz-collections/'
CDN = 'https://cdn.msisurfaces.com/images/quartz-countertops/products/'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'

def get(url, binary=False, tries=4):
    for i in range(tries):
        try:
            b = urlopen(Request(url, headers={'User-Agent': UA}), timeout=60).read()
            return b if binary else b.decode('utf-8', 'replace')
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 + 3 * i)

def text(s):
    s = re.sub(r'<script[\s\S]*?</script>|<style[\s\S]*?</style>', '', s)
    return [l.strip() for l in html.unescape(re.sub(r'<[^>]+>', '\n', s)).split('\n') if l.strip()]

s = get(LIST)
items = {}
for name, href, thumb in re.findall(r'product-title-inner">(.*?)</div>[\s\S]*?href="(/quartz-countertops/[^"]+)"[\s\S]*?data-src="([^"]+)"', s):
    slug = thumb.rsplit('/', 1)[1].rsplit('.', 1)[0]
    name = html.unescape(re.sub(r'<sup>.*?</sup>|<[^>]+>', '', name)).strip()
    items.setdefault(slug, dict(name=name, slug=slug, page=SITE + href))
items = list(items.values())

def after(lines, key):
    """Value lines under a spec heading, up to the next heading."""
    heads = {'Primary Color(s)', 'Accent Color(s)', 'Other Industry Names(aka)', 'Style', 'Available Finishes', 'Price Range',
             'Book Match', 'Variations', 'SIZE(S)', 'Applications'}
    if key not in lines: return []
    i = lines.index(key) + 1; out = []
    while i < len(lines) and lines[i] not in heads and not lines[i].startswith('SLABS') and len(out) < 4:
        out.append(lines[i]); i += 1
    return out

def fetch(d):
    p = get(d['page']); L = text(p)
    one = lambda k: (after(L, k) or [None])[0]
    d.update(style=one('Style'), primary=one('Primary Color(s)'), accent=one('Accent Color(s)'),
             finishes=[f.strip() for f in (one('Available Finishes') or '').split(',') if f.strip()],
             sizes=after(L, 'SIZE(S)'), price=one('Price Range'))
    m = re.search(r'https://cdn\.msisurfaces\.com/images/quartz-countertops/products/slab/large/[^"\'\s?)]+\.jpg', p)
    d['photo'] = m.group(0) if m else CDN + f'slab/large/{d["slug"]}.jpg'
    f = IMG / f'{d["slug"]}.jpg'
    if not f.exists():
        try: f.write_bytes(get(d['photo'], binary=True))
        except Exception as e: print(f'no slab photo for {d["name"]}: {e}'); return
    d['file'] = str(f.relative_to(R))
with cf.ThreadPoolExecutor(4) as ex: list(ex.map(fetch, items))
items.sort(key=lambda d: d['name'])
rows = ',\n'.join(json.dumps(d) for d in items)
(OUT / 'quartz.json').write_text(f'{{"source": {json.dumps(LIST)}, "quartz": [\n{rows}\n]}}\n')
st = {}
for d in items: st[d['style']] = st.get(d['style'], 0) + 1
print(f'{len(items)} colours, {sum("file" in d for d in items)} with a slab photo; styles:', st)
