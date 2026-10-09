"""Fill viewer/studio_template.html with the GLB, a digest of moving parts, the solved photo cameras and the finish
catalog -> viewer/keane_showroom_studio.html.

Finish textures are written as viewer/textures/<id>.js (one JSONP call carrying a data: URL each). The page loads
them only when a finish is picked. Script tags work from file://, and data: URLs can go to WebGL without a server.

Sources (downloaded by hand into materials/, see materials/manifest.json):
  Countertops/*.png       showroom slab photos (Vadara)        full slab, 126 x 63 in
  materials/vadara/*.jpg  vadara.uk/designs                    full slab, 126 x 63 in
  materials/msi/*.jpg     msisurfaces.com/granite-countertops  close-up, assumed 24 x 24 in
  materials/decore/*.jpg  decore.com DRS131 finishes           400 px chip, assumed ~10 in across

Paths are relative to the repo root, so this runs from any checkout:  python3 tools/build_studio.py
"""
import json, base64, io, math, re
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
    write_tex(tid, seamless(im), 86)
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

mats = A['materials']
digest = dict(parts=parts, views=views, photoPaint=mats['mat_cabinet_primary']['baseColor'],
              photoMetal=mats['mat_hardware_finish']['baseColor'])
catalog = dict(counters=counters, fronts=fronts, sources=M.get('sources', {}))
glb = base64.b64encode(open(R / 'export/kitchen_world.glb', 'rb').read()).decode()
safe = lambda o: json.dumps(o, separators=(',', ':')).replace('</', '<\\/')
html = (R / 'viewer/studio_template.html').read_text()
html = html.replace('__ASM__', safe(digest)).replace('__CATALOG__', safe(catalog)).replace('__GLB__', glb)
out = R / 'viewer/keane_showroom_studio.html'
out.write_text(html)
tex_bytes = sum(p.stat().st_size for p in TEX.glob('*.js'))
print(f'counters {len(counters)} (quartz {sum(c["kind"] == "slab" for c in counters)}, granite {sum(c["kind"] == "tile" for c in counters)}), '
      f'fronts {len(fronts)} (solid {sum(f["group"] == "solid" for f in fronts)}, woodgrain {sum(f["group"] == "woodgrain" for f in fronts)}, '
      f'textured {sum(f["kind"] == "tex" for f in fronts)})')
print(f'page {len(html) / 1e6:.2f} MB -> {out.relative_to(R)}; textures {tex_bytes / 1e6:.1f} MB in {TEX.relative_to(R)}/')
