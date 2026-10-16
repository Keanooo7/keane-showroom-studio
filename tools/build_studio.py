"""Fill viewer/studio_template.html with the GLB, a digest of moving parts, the solved photo cameras and the finish
catalog -> viewer/keane_showroom_studio.html.

Finish textures are written as viewer/textures/<id>.js (one JSONP call carrying a data: URL each). The page loads
them only when a finish is picked. Script tags work from file://, and data: URLs can go to WebGL without a server.
Top Knobs knobs & pulls (materials/topknobs/: products.json, geom/, finishes.json) go the same way: the catalog is
embedded, each product's shapes are viewer/hardware/<slug>.js and its chip picture viewer/hardware/thumbs/<slug>.jpg.
Products without a geometry file are left out with a warning.

Sources (downloaded by hand into materials/, see materials/manifest.json):
  Countertops/*.png       showroom slab photos (Vadara)        full slab, 126 x 63 in
  materials/vadara/*.jpg  vadara.uk/designs                    full slab, 126 x 63 in
  materials/msi/*.jpg     msisurfaces.com/granite-countertops  close-up, assumed 24 x 24 in
  materials/decore/*.jpg  decore.com DRS131 finishes           400 px chip, assumed ~10 in across
Sherwin-Williams paint colours are materials/sherwin_williams/colors.json (tools/fetch_sw_colors.py): hex values only.
MSI backsplash tile comes from materials/msi_tile/tiles.json, whose textures tools/msi_tiles.py has already written to
viewer/textures/ (true-scale repeats measured from the maker's photos); only the catalog goes into the page.

Paths are relative to the repo root, so this runs from any checkout:  python3 tools/build_studio.py
(TK_DATA=<dir> python3 tools/build_studio.py reads geom/ and finishes.json from <dir> instead, e.g. mock data.)
"""
import json, base64, io, math, os, re
from pathlib import Path
from PIL import Image, ImageOps, ImageStat

R = Path(__file__).resolve().parent.parent
A = json.load(open(R / 'cad_output/assembly.json'))
cams = json.load(open(R / 'sfm/cameras.json'))['images']
M = json.load(open(R / 'materials/manifest.json'))
TEX = R / 'viewer/textures'; TEX.mkdir(exist_ok=True)

parts = {}
for c in A['components']:
    d = {}
    sw = c.get('door_swing')
    if sw:
        d['swing'] = dict(sign=1 if sw['open_angle_deg'] > 0 else -1, max=sw['max_open_deg'])
    if c.get('drawer_slide'):
        d['travel_mm'] = c['drawer_slide']['travel_mm']
    if d:
        parts[c['id']] = d

views = []
for k in sorted(cams):
    c = cams[k]; ctr = c['camera_center']; vd = c['viewing_direction_world']
    fov = 2 * math.degrees(math.atan(c['height'] / 2 / c['f_px']))
    views.append(dict(id=k.split('.')[0], label='Photo ' + k[:2], center=ctr, target=[ctr[i] + 100 * vd[i] for i in range(3)],
                      fov=round(fov, 2)))

slug = lambda s: re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
s2l = lambda v: (v / 255) / 12.92 if v <= 10.31 else ((v / 255 + 0.055) / 1.055) ** 2.4

def jpg_uri(im, q):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

def write_tex(tid, im, q=86):
    (TEX / f'{tid}.js').write_text(f'window.__texData({json.dumps(tid)},{json.dumps(jpg_uri(im, q))});\n')

def mirror4(im):
    """2 x 2 mirrored tile: seamless under plain repeat (the path tracer has no mirrored wrap)."""
    w, h = im.size; out = Image.new('RGB', (2 * w, 2 * h))
    out.paste(im, (0, 0)); out.paste(ImageOps.mirror(im), (w, 0))
    out.paste(ImageOps.flip(im), (0, h)); out.paste(ImageOps.flip(ImageOps.mirror(im)), (w, h))
    return out

def seamless(im):
    """Offset-and-blend tile for random stone: crossfade with a half-period roll so the borders come from the
    image interior. No mirror axes, which read as Rorschach symmetry on granite."""
    import numpy as np
    a = np.asarray(im, dtype=np.float32); h, w = a.shape[:2]
    r = np.roll(np.roll(a, h // 2, 0), w // 2, 1)
    ramp = lambda n: np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / (0.22 * n), 0, 1)
    m = (np.outer(ramp(h), ramp(w)) ** 0.8)[..., None]
    return Image.fromarray((a * m + r * (1 - m)).round().astype(np.uint8))

def thumb(im, w, h):
    return jpg_uri(ImageOps.fit(im, (w, h), Image.Resampling.LANCZOS), 72)

counters, fronts = [], []
# ---- quartz: full slabs, 126 x 63 in
for d in M['vadara']:
    im = Image.open(R / d['file']).convert('RGB')
    w, h = im.size; m = max(4, round(0.006 * w)); im = im.crop((m, m, w - m, h - m))   # thin photo border
    if im.width > 2048: im = im.resize((2048, round(2048 * im.height / im.width)), Image.Resampling.LANCZOS)
    tid = 'vadara_' + slug(d['name']); write_tex(tid, im)
    counters.append(dict(id=tid, name=d['name'], code=d.get('code'), brand='Vadara quartz', kind='slab', group=d.get('collection') or 'Vadara',
                         size_mm=[3200.4, 1600.2], thumb=thumb(im.crop((im.width // 4, 0, im.width * 3 // 4, im.height)), 84, 56)))
# ---- granite: square close-ups, tiled
for d in M['msi']:
    im = Image.open(R / d['file']).convert('RGB')
    tid = 'msi_' + slug(d['name'])
    st = seamless(im)   # 2048 px at most, like the slabs: larger costs GPU memory and the path tracer resizes to 2048 anyway
    if max(st.size) > 2048: st = st.resize((round(2048 * st.width / max(st.size)), round(2048 * st.height / max(st.size))), Image.Resampling.LANCZOS)
    write_tex(tid, st, 86)
    lum = ImageStat.Stat(im.convert('L')).mean[0]   # tone bands for grouping, like a maker's colour tiers
    tone = 'Light & white' if lum >= 155 else 'Mid tones' if lum >= 80 else 'Dark & black'
    counters.append(dict(id=tid, name=d['name'], brand='MSI granite', kind='tile', group=tone, size_mm=[609.6, 609.6], thumb=thumb(im, 84, 56)))

# ---- Decore DRS131 (1-piece RTF/MDF): finish behaviour from the product name
def finish(name):
    n = name.lower()
    if 'supermatte' in n: return dict(rough=0.82, cc=0.0)
    if 'suedette' in n or 'velvet' in n: return dict(rough=0.68, cc=0.0)
    if n.startswith('satin'): return dict(rough=0.36, cc=0.3)
    return dict(rough=0.46, cc=0.15)
for d in M['decore']:
    im = Image.open(R / d['file']).convert('RGB')
    st = ImageStat.Stat(im.convert('L')); mean = ImageStat.Stat(im).mean
    lin = [s2l(v) for v in mean]; k = min(1, 0.85 / max(max(lin), 1e-6)); lin = [max(0.012, v * k) for v in lin]   # paint albedo tops out ~0.85
    textured = st.stddev[0] > 4
    n = d['name'].lower()
    family = ('Supermatte' if 'supermatte' in n else 'Soft-touch' if ('suedette' in n or 'velvet' in n) else 'Satin' if n.startswith('satin') else 'Smooth')
    e = dict(id='decore_' + slug(d['name']), name=d['name'], brand='Decore', group=d['group'], family=family, color=[round(v, 4) for v in lin], **finish(d['name']))
    if textured:
        strong = any(t in d['name'].lower() for t in ('taction', 'deep grain', 'groovz', 'portuna'))
        e.update(kind='tex', size_mm=[2 * 254, 2 * 457], emboss=0.55 if strong else 0.22, thumb=thumb(im, 56, 56))
        if d['group'] == 'woodgrain': e.update(rough=0.55 if not strong else 0.62, cc=0.08, family='Textured' if strong else 'Smooth woodgrain')
        write_tex(e['id'], mirror4(im), 88)
    else:
        e.update(kind='solid')
    fronts.append(e)

# ---- Sherwin-Williams paint (materials/sherwin_williams/colors.json, from tools/fetch_sw_colors.py): a painted
# five-piece door in each colour. The published hex is the colour; lightest first within each family.
SW = json.load(open(R / 'materials/sherwin_williams/colors.json'))['colors']
for c in sorted(SW, key=lambda c: -sum(int(c['hex'][i:i + 2], 16) for i in (1, 3, 5))):
    fronts.append(dict(id='sw_' + c['code'][3:], name=c['name'], code=c['code'], brand='Sherwin-Williams', group='sw', family=c['family'],
                       kind='paint', color=[round(s2l(int(c['hex'][i:i + 2], 16)), 4) for i in (1, 3, 5)]))

# ---- Top Knobs knobs & pulls (materials/topknobs/README.md): the page embeds the catalog; each product's geometry is
# viewer/hardware/<slug>.js (JSONP, loaded on pick) and its chip picture viewer/hardware/thumbs/<slug>.jpg.
# TK_DATA=<dir> reads geom/ and finishes.json from another folder (mock data for testing the page).
TK = R / 'materials/topknobs'; TKD = Path(os.environ.get('TK_DATA') or TK)
HWD = R / 'viewer/hardware'; (HWD / 'thumbs').mkdir(parents=True, exist_ok=True)

def rnd(o, nd=2):
    if isinstance(o, float): return round(o, nd) + 0.0
    if isinstance(o, list): return [rnd(v, nd) for v in o]
    if isinstance(o, dict): return {k: rnd(v, 4 if k == 'stretch' else nd) for k, v in o.items()}
    return o

def hardware():
    prods = json.load(open(TK / 'products.json'))['products'] if (TK / 'products.json').exists() else []
    fpath = TKD / 'finishes.json'
    F = json.load(open(fpath)) if fpath.exists() else {'finishes': {}, 'secondary': {}}
    if not fpath.exists(): print(f'warning: {fpath.relative_to(R) if fpath.is_relative_to(R) else fpath} missing, using a neutral metal for every finish')
    fin, out, wrote, missing = F['finishes'], [], set(), []
    for p in prods:
        gp, op = TKD / 'geom' / f'{p["slug"]}.json', TKD / 'geom/overrides' / f'{p["slug"]}.json'
        g = json.load(open(gp)) if gp.exists() else None
        ov = json.load(open(op)) if op.exists() else {}
        shapes = {k: v['shape'] for k, v in (g or {}).get('sizes', {}).items() if v.get('shape')}
        shapes.update({k: v['shape'] for k, v in ov.get('sizes', {}).items() if v.get('shape')})
        sizes = [s for s in p['sizes'] if s['key'] in shapes]
        if not sizes: missing.append(p['slug']); continue
        typ = ov.get('type') or (g or {}).get('type') or 'other'
        notes = ' '.join([(g or {}).get('notes') or '', ov.get('notes') or ''] + [(v.get('fit') or {}).get('notes') or '' for v in (g or {}).get('sizes', {}).values()])
        # Mounting: 'edge' tabs hang over the top edge of the front (origin on that edge); 'hang' pieces were traced from
        # the front with +Z = down the door (cup, drop, plate), so they stay horizontal even on doors (tools/hw_pulls.py).
        mount = 'edge' if 'edge tab' in notes or 'mode tab' in notes else 'hang' if '+Z = down' in notes else None
        data = {'type': typ, 'sizes': {s['key']: rnd(shapes[s['key']]) for s in sizes}}
        if mount: data['mount'] = mount
        (HWD / f'{p["slug"]}.js').write_text(f'window.__hwData({json.dumps(p["slug"])},{json.dumps(data, separators=(",", ":"))});\n')
        wrote.add(p['slug'])
        if p.get('thumb') and (TK / p['thumb']).exists():   # crop the white sweep so the part fills its chip
            im = Image.open(TK / p['thumb']).convert('RGBA'); bg = Image.new('RGBA', im.size, 'white'); bg.alpha_composite(im); im = bg.convert('RGB')
            box = ImageOps.invert(im.convert('L')).point(lambda v: 255 if v > 12 else 0).getbbox() or (0, 0, *im.size)
            cx, cy, half = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1]) * 0.56
            x0, y0, n = round(cx - half), round(cy - half), round(2 * half)
            sq = Image.new('RGB', (n, n), 'white'); sq.paste(im, (-x0, -y0))
            sq.resize((150, 150), Image.Resampling.LANCZOS).save(HWD / 'thumbs' / f'{p["slug"]}.jpg', 'JPEG', quality=80, optimize=True)
        # "Default" (SS46 / SS47) names no finish: take the longest finish label the product name spells out instead
        named = max((f for f in fin if f.lower() in p['name'].lower()), key=len, default=None)
        if 'Default' in p['finishes'] and named:
            sizes = [dict(z, skus={named if f == 'Default' else f: v for f, v in z['skus'].items()}) for z in sizes]
            p = dict(p, finishes=[named if f == 'Default' else f for f in p['finishes']])
        labels = [f for f in p['finishes'] if any(f in s['skus'] for s in sizes)]
        for f in labels:
            if f not in fin:   # no measured finish: neutral metal so the product still shows
                fin[f] = dict(id='tk_' + slug(f), label=f, family='Specialty', color=[0.5, 0.5, 0.5], roughness=0.3, metalness=1, swatch='#bcbcbc')
        url = (p.get('url') or '').replace('https://www.topknobs.com/', '')
        out.append(dict(slug=p['slug'], name=p['name'], collection=p.get('collection') or 'Other', type=typ, finishes=labels, url=url,
                        sizes=[dict(s, skus=[s['skus'].get(f) for f in labels]) for s in sizes]))
    for f in HWD.glob('*.js'):   # products whose geometry went away
        if f.stem not in wrote: f.unlink()
    for f in (HWD / 'thumbs').glob('*.jpg'):
        if f.stem not in wrote: f.unlink()
    if missing: print(f'warning: {len(missing)} of {len(prods)} Top Knobs products have no geometry yet and are left out: {", ".join(missing[:8])}{" ..." if len(missing) > 8 else ""}')
    used = {f for p in out for f in p['finishes']}
    used |= {t for f in used for t in fin[f].get('two_tone', [])}
    keep = ('id', 'label', 'family', 'color', 'roughness', 'metalness', 'clearcoat', 'clearcoat_roughness', 'anisotropy', 'swatch', 'two_tone')
    order = ['Polished', 'Brushed', 'Satin', 'Matte', 'Bronze', 'Brass', 'Black', 'Specialty']
    freq = {f: sum(f in p['finishes'] for p in out) for f in used}
    labs = sorted(used, key=lambda f: (order.index(fin[f].get('family')) if fin[f].get('family') in order else len(order), -freq[f], f))
    finishes = {f: {k: rnd(fin[f][k], 4) for k in keep if k in fin[f]} for f in labs}
    secondary = {k: {a: rnd(v[a], 4) for a in ('label', 'color', 'roughness', 'metalness', 'transmission', 'ior', 'thickness') if a in v}
                 for k, v in F.get('secondary', {}).items()}
    # Compact form, expanded by the page into products[].sizes[] = {key, label, c2c_mm, length_mm, width_mm, proj_mm,
    # diam_mm, skus}: finishes become indices into the finish list, dims an array, and the url and size key are left
    # out when they follow from the slug and label. Keeps 422 products under 200 kB.
    dims = ('c2c_mm', 'length_mm', 'width_mm', 'proj_mm', 'diam_mm')
    fi = {f: i for i, f in enumerate(labs)}
    prods = []
    for p in out:
        e = dict(slug=p['slug'], name=p['name'], collection=p['collection'], type=p['type'], f=[fi[f] for f in p['finishes']],
                 s=[dict(l=z['label'], d=[round(z[k], 2) if z.get(k) else 0 for k in dims], s=z['skus'],
                         **({} if z['key'] == slug(z['label']) else {'k': z['key']})) for z in p['sizes']])
        if p['url'] != f'collections/{p["slug"].replace("_", "-")}.html': e['url'] = p['url']
        prods.append(e)
    return dict(finishes=finishes, secondary=secondary, products=prods)

hw = hardware()

def tiles():
    """MSI backsplash tile (materials/msi_tile/tiles.json, made by tools/msi_tiles.py; its textures are already in
    viewer/textures/). Every usable SKU goes in. lay 'line': a rectangular tile laid in courses (running or stack bond),
    offered both horizontally and turned 90 deg for vertical; lay 'custom': squares, hexagons, penny rounds, arabesques,
    pickets, herringbone and other patterns, laid as photographed. Trim pieces (bullnose, quarter round) are not wall
    field tile and stay out. Compact keys: p = texture period [w, h] mm, cc = clearcoat, th = chip picture."""
    src = R / 'materials/msi_tile/tiles.json'
    if not src.exists(): print('warning: materials/msi_tile/tiles.json missing, no MSI tile'); return []
    T = [t for t in json.load(open(src))['tiles'] if t.get('usable') and t.get('tex')]
    have = [t for t in T if (TEX / f'{t["id"]}.js').exists() and (TEX / f'{t["id"]}_n.js').exists()]
    if len(have) < len(T): print(f'warning: {len(T) - len(have)} MSI tiles have no texture file and are left out: '
                                 + ', '.join(t['sku'] for t in T if t not in have))
    tones = [('White & cream', ('white', 'cream')), ('Gray', ('gray', 'grey', 'silver')), ('Beige & brown', ('beige', 'brown', 'taupe')),
             ('Blue', ('blue',)), ('Green', ('green',)), ('Black', ('black',)), ('Red & pink', ('red', 'pink'))]
    shapes = [('Hexagon', ('hexagon', 'hexo', 'hive', 'petal')), ('Penny round', ('penny',)), ('Octagon', ('octagon',)),
              ('Arabesque & scallop', ('arabesque', 'scalop', 'scallop', 'fish scale', 'teardrop')), ('Picket', ('picket', 'pkt')),
              ('Herringbone & weave', ('herringbone', 'basketweave', 'weave'))]
    pick = lambda text, table, other: next((k for k, ws in table if any(w in text for w in ws)), other)
    dup = {}
    for t in have: dup[t['product']] = dup.get(t['product'], 0) + 1
    out = []
    for t in have:
        L, S = t['tile_mm'] or (0, 0); bond = t['tex']['bond']
        lay = 'line' if bond in ('running', 'stack') and S and L / S >= 1.4 else 'custom'
        text = ' '.join([t['name'], t['product'], t['style']]).lower()
        if lay == 'line': grp = pick((t['color'] or '').split(',')[0].lower(), tones, 'Other')
        elif bond in ('running', 'stack') and S and L / S < 1.4 and 'hex' not in text: grp = 'Squares & checks'
        elif bond == 'herringbone': grp = 'Herringbone & weave'
        else: grp = pick(text, shapes, 'Squares & checks' if re.search(r'chex|check', text) else 'Other patterns')
        name = re.sub(r'\s*®\s*', ' ', t['product']).strip()
        if dup[t['product']] > 1: name += f' {t["size_label"]} {t["finish"]}'.rstrip()
        size = re.sub(r'(?i)\s*x\s*', ' × ', t['size_label'].replace('"', '').replace('”', '')).strip()
        out.append(dict(id=t['id'], name=name, code=t['sku'], family=' · '.join(v for v in (size + (' in' if re.search(r'\d$', size) else ''), t['finish']) if v),
                        coll=t['collection'] or '', lay=lay, group=grp, bond=bond, p=t['tex']['period_mm'],
                        rough=t['rough'], cc=t['clearcoat'], relief=t['relief'], th=t['thumb'],
                        url=t['url'].replace('https://www.msisurfaces.com/', '')))
    order = {g: i for i, g in enumerate([k for k, _ in tones] + ['Other'] + [k for k, _ in shapes] + ['Squares & checks', 'Other patterns'])}
    out.sort(key=lambda o: (o['lay'] != 'line', order.get(o['group'], 99), o['coll'], o['name']))
    return out

tile = tiles()

mats = A['materials']
digest = dict(parts=parts, views=views, photoPaint=mats['mat_cabinet_primary']['baseColor'],
              photoMetal=mats['mat_hardware_finish']['baseColor'])
catalog = dict(counters=counters, fronts=fronts, sources=M.get('sources', {}), hardware=hw, tiles=tile)
glb = base64.b64encode(open(R / 'export/kitchen_world.glb', 'rb').read()).decode()
safe = lambda o: json.dumps(o, separators=(',', ':')).replace('</', '<\\/')
html = (R / 'viewer/studio_template.html').read_text()
html = html.replace('__ASM__', safe(digest)).replace('__CATALOG__', safe(catalog)).replace('__GLB__', glb)
out = R / 'viewer/keane_showroom_studio.html'
out.write_text(html)
tex_bytes = sum(p.stat().st_size for p in TEX.glob('*.js'))
print(f'counters {len(counters)} (quartz {sum(c["kind"] == "slab" for c in counters)}, granite {sum(c["kind"] == "tile" for c in counters)}), '
      f'fronts {len(fronts)} (solid {sum(f["group"] == "solid" for f in fronts)}, woodgrain {sum(f["group"] == "woodgrain" for f in fronts)}, '
      f'textured {sum(f["kind"] == "tex" for f in fronts)}, Sherwin-Williams paint {sum(f["group"] == "sw" for f in fronts)})')
print(f'MSI tile {len(tile)} (horizontal / vertical {sum(t["lay"] == "line" for t in tile)}, custom {sum(t["lay"] == "custom" for t in tile)}), '
      f'{len(safe(tile)) / 1e3:.0f} kB in the page')
print(f'page {len(html) / 1e6:.2f} MB -> {out.relative_to(R)}; textures {tex_bytes / 1e6:.1f} MB in {TEX.relative_to(R)}/')
hw_bytes = sum(p.stat().st_size for p in HWD.rglob('*.*'))
print(f'hardware {len(hw["products"])} products, {sum(len(p["s"]) for p in hw["products"])} sizes, {len(hw["finishes"])} finishes; '
      f'catalog {len(safe(hw)) / 1e3:.0f} kB in the page, {hw_bytes / 1e6:.1f} MB in {HWD.relative_to(R)}/')
