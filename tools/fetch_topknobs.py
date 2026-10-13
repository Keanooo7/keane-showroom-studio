"""Scrape every product in topknobs.com/products/knobs-pulls.html -> materials/topknobs/catalog.json.

Each listing entry is a Magento configurable product (e.g. "Brixton Pull"). Its page carries a jsonConfig with the
Size and Finish options, one child SKU per combination, and per-SKU data: name, collection, the spec table (Center to
Center, Overall Length, Width, Diameter, Base Diameter, Projection, Base Metal ...) and the product photos.

    python3 tools/fetch_topknobs.py            # resumes: pages already in materials/topknobs/pages/ are not refetched
"""
import json, re, html, time, sys, concurrent.futures as cf
from pathlib import Path
from urllib.request import Request, urlopen

R = Path(__file__).resolve().parent.parent
OUT = R / 'materials/topknobs'; PAGES = OUT / 'pages'; PAGES.mkdir(parents=True, exist_ok=True)
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'
LIST = 'https://www.topknobs.com/products/knobs-pulls.html'

def get(url, tries=4):
    for i in range(tries):
        try:
            return urlopen(Request(url, headers={'User-Agent': UA}), timeout=60).read().decode('utf-8', 'replace')
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2 + 3 * i)

def listing():
    items, p = {}, 1
    while True:
        h = get(f'{LIST}?product_list_limit=60&p={p}')
        new = 0
        for m in re.finditer(r'<li class="item product product-item.*?<a href="([^"]+)"\s+class="product photo product-item-photo".*?'
                             r'<img class="product-image-photo[^"]*" src="([^"]+)" alt="([^"]*)".*?data-product-id="(\d+)"', h, re.S):
            url, img, name, pid = m.groups()
            if url not in items: items[url] = dict(url=url, name=html.unescape(name), pid=pid, list_img=html.unescape(img).split('?')[0]); new += 1
        total = int((re.findall(r'toolbar-number">(\d+)', h) or [0])[-1])
        print(f'listing page {p}: +{new} ({len(items)}/{total})', flush=True)
        if not new or len(items) >= total: break
        p += 1; time.sleep(0.5)
    return list(items.values())

def rows(attr_html):
    return {html.unescape(k).strip(): html.unescape(v).strip()
            for k, v in re.findall(r'data-th="(.*?)">(.*?)</td>', attr_html, re.S)}

def parse(h, item):
    cfg = None
    for m in re.finditer(r'"jsonConfig"\s*:\s*', h):   # related-product widgets carry empty ones; take the product's own
        try: c, _ = json.JSONDecoder().raw_decode(h, m.end())
        except Exception: continue
        if isinstance(c, dict) and c.get('attributes') and c.get('index'): cfg = c; break
    if cfg is None: return parse_simple(h, item)
    opt = {}   # attribute id -> {option id -> label}
    names = {}
    for aid, a in cfg['attributes'].items():
        names[aid] = a['label']; opt[aid] = {o['id']: o['label'] for o in a['options']}
    # per-child blobs: "<childid>":{"spConfig":{...},"addtional_lables":{...},...,"name":..,"sku":..,"desc":..,"attributes":"<table>"}
    dec, child = json.JSONDecoder(), {}
    for m in re.finditer(r'"(\d+)":\{"spConfig"', h):
        try: obj, _ = dec.raw_decode(h, m.start() + len(m.group(1)) + 3)
        except Exception: continue
        child.setdefault(m.group(1), obj)
    variants = []
    for cid, idx in cfg['index'].items():
        c = child.get(cid, {})
        sel = {names[a]: opt[a].get(o) for a, o in idx.items()}
        variants.append(dict(id=cid, sku=cfg['skus'].get(cid), name=c.get('name'), finish=sel.get('Finish'), size=sel.get('Size'),
                             options=sel, collection=(c.get('addtional_lables') or {}).get('collection'),
                             specs=rows(c.get('attributes', '')),
                             images=[im['full'].split('?')[0] for im in cfg['images'].get(cid, [])]))
    parent = rows(re.search(r'id="product-attribute-specs-table".*?</table>', h, re.S).group(0)) if 'id="product-attribute-specs-table"' in h else {}
    desc = re.search(r'<div class="product attribute description">\s*<div class="value"\s*>(.*?)</div>', h, re.S)
    return dict(item, options={names[a]: list(opt[a].values()) for a in opt}, parent_specs=parent,
                description=re.sub(r'<[^>]+>', ' ', html.unescape(desc.group(1))).strip() if desc else None, variants=variants)

def parse_simple(h, item):
    """A product without options: one SKU, spec table and gallery straight on the page."""
    t = re.search(r'id="product-attribute-specs-table".*?</table>', h, re.S)
    specs = rows(t.group(0)) if t else {}
    g = re.search(r'"data"\s*:\s*(\[\{"thumb".*?\}\])', h, re.S)
    imgs = [im['full'].split('?')[0] for im in json.loads(g.group(1))] if g else []
    sku = specs.get('Item Number') or item['name']
    fin = re.search(r'"finish_tk"\s*:\s*"([^"]+)"', h)
    return dict(item, options={}, parent_specs=specs, simple=True,
                variants=[dict(id=item['pid'], sku=sku, name=item['name'], finish=fin.group(1) if fin else None, size=None,
                               options={}, collection=None, specs=specs, images=imgs)] if specs or imgs else [],
                error=None if specs or imgs else 'no product config or spec table')

def page(item):
    f = PAGES / (re.sub(r'[^a-z0-9]+', '_', item['url'].split('/')[-1].lower().replace('.html', '')) + '.json')
    if f.exists(): return json.loads(f.read_text())
    d = parse(get(item['url']), item); f.write_text(json.dumps(d)); time.sleep(0.3)
    return d

if __name__ == '__main__':
    items = listing()
    out, done = [], 0
    with cf.ThreadPoolExecutor(4) as ex:
        for d in ex.map(page, items):
            out.append(d); done += 1
            if done % 25 == 0: print(f'{done}/{len(items)} products', flush=True)
    cat = dict(source=LIST, fetched=time.strftime('%Y-%m-%d'), products=out)
    (OUT / 'catalog.json').write_text(json.dumps(cat, indent=1))
    nv = sum(len(p['variants']) for p in out)
    print(f'{len(out)} products, {nv} SKUs, {sum(1 for p in out if p.get("error"))} without jsonConfig -> materials/topknobs/catalog.json')
