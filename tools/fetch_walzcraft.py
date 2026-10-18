"""Every stained wood finish on walzcraft.com/wood-finish (category "Wood Stains") -> materials/walzcraft/.

The archive is a FacetWP listing, 16 finishes a page. Each finish is one stain on one wood species, shown on a photo of
WalzCraft's color block (12 x 7 in, 3/4 in thick, D17 reeded edge; walzcraft.com/product/non-inventory-color-blocks):
the block is photographed with its grain vertical, so the 7 in height spans the photo's width, reeded edge on the right.

    python3 tools/fetch_walzcraft.py     # photos already in materials/walzcraft/img/ are not refetched
Writes materials/walzcraft/finishes.json: id, stain, stain type code (W, WBW, SNW ...), species, shade, page, photo.
"""
import json, re, html, time, concurrent.futures as cf
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

R = Path(__file__).resolve().parent.parent
OUT = R / 'materials/walzcraft'; IMG = OUT / 'img'; IMG.mkdir(parents=True, exist_ok=True)
LIST = 'https://walzcraft.com/wood-finish/?fwp_cat=1-wood-stains'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'

def get(url, binary=False, tries=4):
    for i in range(tries):
        try:
            b = urlopen(Request(url, headers={'User-Agent': UA}), timeout=60).read()
            return b if binary else b.decode('utf-8', 'replace')
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 + 3 * i)

def page(n):
    s = get(f'{LIST}&fwp_paged={n}')
    pages = json.loads(re.search(r'FWP_JSON = (\{.*?\});\s*\n', s).group(1))['preload_data']['settings']['pager']['total_pages']
    out = []
    for m in re.finditer(r'<article id="post-(\d+)" class="([^"]*)">([\s\S]*?)</article>', s):
        pid, cls, body = m.groups()
        href = re.search(r'<a href="([^"]+)"', body).group(1)
        src = re.search(r'<img[^>]* src="([^"]+)"', body).group(1)
        stain = html.unescape(re.search(r'Stain: </span>([^<]+)', body).group(1)).strip()
        wood = html.unescape(re.search(r'Wood: </span>([^<]+)', body).group(1)).strip()
        st = re.match(r'(.*?)\s*\(([^)]+)\)$', stain)
        shade = re.search(r'\bshade-\d-(\w+)', cls)
        out.append(dict(id=int(pid), stain=st.group(1) if st else stain, type=st.group(2) if st else None, species=wood,
                        shade=shade.group(1) if shade else None, page=href, photo=src))
    return pages, out

pages, items = page(1)
for n in range(2, pages + 1):
    items += page(n)[1]
items = list({d['id']: d for d in items}.values())   # a finish can sit on two listing pages while the order shifts

def fetch(d):
    f = IMG / f'{d["id"]}.jpg'
    if not f.exists():
        f.write_bytes(get(quote(d['photo'], safe=':/%'), binary=True))
    d['file'] = str(f.relative_to(R))
with cf.ThreadPoolExecutor(6) as ex: list(ex.map(fetch, items))
items.sort(key=lambda d: (d['species'], d['stain'], d['type'] or ''))
rows = ',\n'.join(json.dumps(d) for d in items)
(OUT / 'finishes.json').write_text(f'{{"source": {json.dumps(LIST)}, "finishes": [\n{rows}\n]}}\n')
sp = {}
for d in items: sp[d['species']] = sp.get(d['species'], 0) + 1
print(f'{len(items)} finishes on {pages} pages, {len(sp)} species:', sp)
