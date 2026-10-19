"""Cosentino Silestone and Dekton slab photos -> materials/cosentino/<brand>/.

cosentino.com sits behind a JavaScript challenge, but Cosentino's BIM colour libraries are plain downloads from its
asset server (linked from every colour page): assetstools.cosentino.com/api/v1/bynder/downloads/bim/<brand>.zip.
Each holds an Autodesk material library plus a zip of the textures it uses: one full slab photo per colour, 2000 px
wide ("<Name> - Tabla vN.jpg" for Dekton, "<Name>.jpg" for Silestone). The libraries date from June 2022 ("PN14 06.22
BIM colour libraries"), so colours launched since then are missing.

Dekton extras are skipped: bookmatch pieces (Aura 15 Book A2 ...), gloss / UV / relief maps, flat-light and "more
light" variants and a duplicate (Dekton-Aeris); of several versions of one slab the highest is kept.
Slab sizes: Silestone 128.74 x 62.60 in (cosentino.com colour pages, "FORMAT"), Dekton 3200 x 1440 mm. The photos
have the same aspect, so each is mapped at the slab's length and its height follows the photo.

    python3 tools/fetch_cosentino.py     # zips are cached in materials/cosentino/cache/ (not in git)
Writes materials/cosentino/colors.json: brand, name, file, source (the zip member it came from).
"""
import io, json, re, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen

R = Path(__file__).resolve().parent.parent
OUT = R / 'materials/cosentino'; CACHE = OUT / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
BIM = 'https://assetstools.cosentino.com/api/v1/bynder/downloads/bim/{}.zip'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'
SKIP = re.compile(r'\bBook\b|Mapa|Relieve|Luz Plana|mas luz|^Dekton-', re.I)

def get(url, tries=4):
    for i in range(tries):
        try: return urlopen(Request(url, headers={'User-Agent': UA}), timeout=300).read()
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 + 3 * i)

def textures(brand):
    """(member name, bytes) of every jpg in the texture zip inside the BIM zip."""
    z = CACHE / f'{brand}.zip'
    if not z.exists(): z.write_bytes(get(BIM.format(brand)))
    outer = zipfile.ZipFile(z)
    inner = zipfile.ZipFile(io.BytesIO(outer.read(next(n for n in outer.namelist() if n.lower().endswith('.zip')))))
    return [(n.rsplit('/', 1)[-1], inner.read(n)) for n in inner.namelist() if n.lower().endswith('.jpg')]

def version(stem):
    m = re.search(r'Tabla v(\d+)', stem); return int(m.group(1)) if m else 0

out = []
for brand in ('dekton', 'silestone'):
    best = {}
    for fn, data in textures(brand):
        stem = fn.rsplit('.', 1)[0]
        if brand == 'dekton':
            if SKIP.search(stem): continue
            name = re.split(r'\s+-\s+Tabla', stem)[0]
        else:
            name = stem
        name = re.sub(r'\s+', ' ', name.replace('_', ' ')).strip()
        if name not in best or version(stem) > version(best[name][0]): best[name] = (stem, fn, data)
    d = OUT / brand; d.mkdir(exist_ok=True)
    for name, (stem, fn, data) in sorted(best.items()):
        f = d / (re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') + '.jpg')
        f.write_bytes(data)
        out.append(dict(brand=brand, name=name, file=str(f.relative_to(R)), source=f'{brand}.zip: {fn}'))
    print(f'{brand}: {len(best)} colours')
rows = ',\n'.join(json.dumps(d) for d in out)
(OUT / 'colors.json').write_text(f'{{"source": {json.dumps(BIM.format("<brand>"))}, "colors": [\n{rows}\n]}}\n')
