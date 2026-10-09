"""Fill viewer/template.html with the GLB, a compact assembly digest and the fit-check overlays."""
import json, base64, io, re
from PIL import Image
R = '/home/claude/kitchen/'
A = json.load(open(R + 'cad_output/assembly.json')); V = json.load(open(R + 'artifacts/export/validation_report.json'))
cams = json.load(open(R + 'artifacts/sfm/cameras.json'))['images']; fit = json.load(open(R + 'artifacts/verify/alignment_scores.json'))
def size(s):
    m = re.match(r'(.+?)W x (.+?)H x (.+?)D', s); return ' × '.join(m.groups()) if m else s
comps = []
for i, c in enumerate(A['components']):
    d = dict(id=c['id'], label=c['label'], category=c['category'], layer=c['layer'], run=c.get('run'), order=i,
             size=size(c.get('nominal_size_frac', '')), obb=[c['obb']['width_mm'], c['obb']['height_mm'], c['obb']['depth_mm']],
             mat=c['material_default'], pivot=c['pivot']['type'], parent=c.get('parent'), children=len(c.get('children') or []))
    sw = c.get('door_swing')
    if sw:
        fc = sw.get('first_collision'); d['swing'] = dict(open=sw['open_angle_deg'], max=sw['max_open_deg'], side=sw['hinge_side'],
                                                          hit=', '.join(fc['parts']) if fc else None)
    if c.get('drawer_slide'): d['travel_mm'] = c['drawer_slide']['travel_mm']
    comps.append(d)
s = V['summary']
c2 = cams['02_corner_wide.jpg']; ctr = c2['camera_center']; vd = c2['viewing_direction_world']
digest = dict(components=comps,
    stats=dict(parts=s['parts'], watertight=s['parts'] if s['all_watertight'] else '?', overlap_mm3=int(s['max_intersection_mm3']),
               reveal_mm=s['front_front_edge_min_gap_mm'], scale_pct=1.24),
    cam02=dict(center=ctr, target=[ctr[k] + 100 * vd[k] for k in range(3)]),
    finishes=[
      dict(slot='mat_cabinet_primary', label='Cabinets', options=[
        dict(name='Showroom greige (from photos)', color=A['materials']['mat_cabinet_primary']['baseColor'], roughness=0.55, metallic=0),
        dict(name='Alabaster', color=[0.846, 0.823, 0.745], roughness=0.5, metallic=0),
        dict(name='Sage', color=[0.332, 0.376, 0.262], roughness=0.5, metallic=0),
        dict(name='Navy', color=[0.056, 0.076, 0.112], roughness=0.45, metallic=0)]),
      dict(slot='mat_hardware_finish', label='Pulls', options=[
        dict(name='Champagne bronze (from photos)', color=A['materials']['mat_hardware_finish']['baseColor'], roughness=0.35, metallic=1),
        dict(name='Matte black', color=[0.03, 0.03, 0.03], roughness=0.6, metallic=0.3),
        dict(name='Polished nickel', color=[0.80, 0.78, 0.74], roughness=0.15, metallic=1)]),
      dict(slot='mat_countertop_surface', label='Countertop', options=[
        dict(name='White quartz', color=A['materials']['mat_countertop_surface']['baseColor'], roughness=0.2, metallic=0),
        dict(name='Soapstone', color=[0.07, 0.075, 0.08], roughness=0.45, metallic=0),
        dict(name='Butcher block', color=[0.45, 0.24, 0.10], roughness=0.6, metallic=0)])])
fits = []
for k in ['01_sink_closeup.jpg', '02_corner_wide.jpg', '03_corner_high.jpg', '04_right_run.jpg', '05_left_run.jpg']:
    im = Image.open(R + 'artifacts/verify/overlay_' + k); im = im.resize((460, int(460 * im.height / im.width)))
    b = io.BytesIO(); im.save(b, 'JPEG', quality=70)
    fits.append(dict(name=k.replace('.jpg', ''), src='data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode(),
                     median=fit[k]['median_px_fullres'], w16=fit[k]['within_16px'],
                     note='portrait-mode blur lowers the edge score' if k.startswith('01') else ''))
glb = base64.b64encode(open(R + 'artifacts/export/kitchen_world.glb', 'rb').read()).decode()
html = open(R + 'viewer/template.html').read()
safe = lambda o: json.dumps(o, separators=(',', ':')).replace('</', '<\\/')
html = html.replace('__ASM__', safe(digest)).replace('__FITS__', safe(fits)).replace('__GLB__', glb)
open(R + 'viewer/keane_showroom_twin.html', 'w').write(html)
print('bytes', len(html))
