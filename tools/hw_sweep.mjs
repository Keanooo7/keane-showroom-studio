/* Headless check of the Top Knobs hardware in viewer/keane_showroom_studio.html -> materials/topknobs/qa/sweep.json.

Serves the repo over http, opens the studio in headless Chrome and, through the page's test hook (window.__studio.hw):
  - places every size of every product on all 24 mounts and compares the mounted bounds (hardware frame) with the spec
    table: overall length on X, projection on Y, width on Z (knobs: diameter / length x width), more than 3 % off is a
    failure. Dims a generator replaced on purpose (fit.spec_fix in geom/<slug>.json) are compared instead of the
    catalogue's, and the frames materials/topknobs/README.md documents for edge tabs and hanging rings are honoured;
  - checks each placed piece in its front's frame: back face on the surface it was set on, its footprint inside the
    recessed panel when set on it, no overhang past the front (unless the page already flags it as not fitting), finite
    vertices and normals, every mount placed;
  - picks every finish once and compares the live body / accent materials with finishes.json (two-tones included);
  - puts knobs on the doors and pulls on the drawers, checks each target got its own piece, then reloads the page from
    the written #hd / #hr link and checks the same build comes back;
  - collects console errors, page errors and failed requests (missing hardware/<slug>.js, thumbs, textures).
--shots writes the review screenshots to materials/topknobs/qa/shots/ instead (see SHOTS).

  node tools/hw_sweep.mjs [--only slug,slug] [--file] [--out path]   sweep (all products, or these)
  node tools/hw_sweep.mjs --shots [--only name,name]                  screenshots (all of SHOTS, or these names)
--file opens the page from file:// instead of the local server (the hardware/ scripts must load from disk too); --out
writes the report elsewhere than materials/topknobs/qa/sweep.json (relative to the repo).
Needs puppeteer-core (npm i --no-save puppeteer-core in the repo, or PUPPETEER_DIR=<path to node_modules/puppeteer-core>)
and Google Chrome (CHROME=<binary>, default the macOS app). Exit code 1 when anything failed.
*/
import fs from 'fs';
import http from 'http';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TK = path.join(ROOT, 'materials/topknobs');
const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const only = arg('--only') ? new Set(arg('--only').split(',')) : null;
const TOL = 0.03;
const require = createRequire(import.meta.url);
const puppeteer = (() => {
  for (const p of [process.env.PUPPETEER_DIR, 'puppeteer-core'].filter(Boolean)) { try { return require(p); } catch {} }
  console.error('puppeteer-core not found: npm i --no-save puppeteer-core, or set PUPPETEER_DIR'); process.exit(2);
})();
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// ---------- static server for the repo (the page loads hardware/, textures/ next to itself)
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png', '.glb': 'model/gltf-binary' };
const server = http.createServer((req, res) => {
  const f = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (req.url === '/favicon.ico') { res.writeHead(204); return res.end(); }   // the browser asks; the page has none
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const BASE = process.argv.includes('--file') ? 'file://' + path.join(ROOT, 'viewer/keane_showroom_studio.html')
  : `http://127.0.0.1:${server.address().port}/viewer/keane_showroom_studio.html`;

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', protocolTimeout: 1800000,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader'] });
const issues = [];   // console errors, page errors, failed requests
async function open(hash = '', w = 1400, h = 900) {
  const page = await browser.newPage();
  await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error') issues.push({ kind: 'console', text: m.text() }); });
  page.on('pageerror', e => issues.push({ kind: 'pageerror', text: e.message }));
  page.on('requestfailed', r => issues.push({ kind: 'requestfailed', text: r.url() }));
  page.on('response', r => { if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) issues.push({ kind: 'http ' + r.status(), text: r.url() }); });
  await page.goto(BASE + hash, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction(() => window.__studio && window.__studio.nodes, { timeout: 180000 });
  await page.waitForFunction(() => !location.hash || location.hash.includes('sc='), { timeout: 60000 });   // restoreBuild done
  await new Promise(r => setTimeout(r, 1500));
  return page;
}
const settle = ms => new Promise(r => setTimeout(r, ms));

// ---------- review screenshots
// Every shot is a build link: hd / hr = slug.size.finishId on doors / drawers, v = view (0 Corner, 2 Drawers).
const SHOTS = [
  ['default_corner', ''], ['default_drawers', '#v=2'],
  ['round_knob', 'hd=kent_knob.one_size.tk_flat_black&hr=kent_knob.one_size.tk_flat_black'],
  ['bar_pull', 'hd=kent_pull.6_5_16_inch_pull.tk_honey_bronze&hr=kent_pull.6_5_16_inch_pull.tk_honey_bronze'],
  ['appliance_pull', 'hd=princetonian_bar_pull_3_posts.15_1_16_inch_pull.tk_brushed_satin_nickel&hr=princetonian_bar_pull_3_posts.15_1_16_inch_pull.tk_brushed_satin_nickel'],
  ['cup_pull', 'hd=trevi_cup_pull.2_1_2_inch_pull.tk_tuscan_bronze&hr=trevi_cup_pull.2_1_2_inch_pull.tk_tuscan_bronze'],
  ['crystal_knob', 'hd=amber_crystal_knob.1_3_8_inch.tk_oil_rubbed_bronze&hr=amber_crystal_knob.1_3_8_inch.tk_oil_rubbed_bronze'],
  ['t_knob', 'hd=hopewell_t_handle.one_size.tk_polished_nickel&hr=hopewell_t_handle.one_size.tk_polished_nickel'],
  ['ring_pull', 'hd=newton_ring.one_size.tk_german_bronze&hr=newton_ring.one_size.tk_german_bronze'],
  ['knobs_doors_pulls_drawers', 'hd=kent_knob.one_size.tk_brushed_satin_nickel&hr=kent_pull.6_5_16_inch_pull.tk_brushed_satin_nickel'],
  ['two_tone', 'hd=bit_pull.5_1_16_inch_pull.tk_flat_black_and_honey_bronze&hr=bit_pull.5_1_16_inch_pull.tk_flat_black_and_honey_bronze'],
];
// camera 3/4 above and to the side of one placed piece, looking at it (runs in the page)
function close(mount) {
  const S = window.__studio, T = S.THREE, o = S.scene.getObjectByName(mount + '_tk');
  const box = new T.Box3().setFromObject(o), c = box.getCenter(new T.Vector3()), r = box.getSize(new T.Vector3()).length();
  const n = new T.Vector3(0, 1, 0).applyQuaternion(o.getWorldQuaternion(new T.Quaternion())), up = new T.Vector3(0, 1, 0);
  const side = new T.Vector3().crossVectors(up, n).normalize(), d = Math.max(2.4 * r, 0.12);
  S.camera.position.copy(c).addScaledVector(n, d).addScaledVector(up, 0.45 * d).addScaledVector(side, 0.4 * d);
  S.camera.fov = 30; S.camera.updateProjectionMatrix(); S.controls.target.copy(c); S.controls.update();
}
if (process.argv.includes('--shots')) {
  const dir = path.join(TK, 'qa/shots'); fs.mkdirSync(dir, { recursive: true });
  for (const [name, h] of SHOTS.filter(([n]) => !only || only.has(n))) {
    const views = name.startsWith('default') ? [['', '']] : [['_drawers', '&v=2'], ['_corner', '&v=0']];
    for (const [suf, v] of views) {
      const hash = h.startsWith('#') ? h : h || v ? '#' + (h + v).replace(/^&/, '') : '';
      const page = await open(hash);
      await settle(2500);
      await page.screenshot({ path: path.join(dir, name + suf + '.png') });
      const m = await page.evaluate(() => window.__studio.hw.measure());
      console.log(name + suf, [...new Set(m.map(x => `${x.target}:${x.slug}${x.placed ? '' : ' (NOT PLACED)'}`))].join(' '));
      if (!name.startsWith('default') && suf === '_drawers')   // close-ups: one drawer mount and one door mount
        for (const tgt of ['drawers', 'doors']) {
          const mt = m.find(x => x.target === tgt && x.fits && x.placed && x.slug !== 'bar'); if (!mt) continue;
          await page.evaluate(close, mt.mount); await settle(1500);
          await page.screenshot({ path: path.join(dir, `${name}_close_${tgt.slice(0, -1)}.png`), clip: { x: 28, y: 80, width: 910, height: 712 } });
        }
      await page.close();
    }
  }
  console.log(issues.length ? issues : 'no console errors'); await browser.close(); server.close(); process.exit(0);
}

// ---------- spec of a size in the hardware frame
const products = JSON.parse(fs.readFileSync(path.join(TK, 'products.json'))).products;
const geomOf = slug => { try { return JSON.parse(fs.readFileSync(path.join(TK, 'geom', slug + '.json'))); } catch { return null; } };
function expected(p, z, g) {
  const fit = g?.sizes?.[z.key]?.fit || {}, d = { ...z, ...(fit.spec_fix || {}) };
  const knob = !z.c2c_mm, notes = (g?.notes || '') + ' ' + Object.values(g?.sizes || {}).map(v => v.fit?.notes || '').join(' ');
  const L = knob ? d.diam_mm || d.length_mm : d.length_mm, W = knob ? d.width_mm || d.diam_mm || d.length_mm : d.width_mm, P = d.proj_mm;
  if (/edge tab|mode tab/.test(notes)) return { x: L, y: W, z: P, frame: 'edge tab: W on Y, P on Z' };
  if (/ring hangs towards \+Z/.test(notes)) return { x: W, y: P, z: L, frame: 'hanging ring: L on Z, W on X' };
  return { x: L, y: P, z: W };
}

const page = await open();
const cat = await page.evaluate(() => window.__studio.hw.catalog());
const inPage = new Set(cat.products.map(p => p.slug));
const fail = [], pass = [];
const bad = (slug, size, what) => fail.push({ slug, size, problem: what });
for (const p of products) if (!inPage.has(p.slug) && (!only || only.has(p.slug))) bad(p.slug, null, 'product missing from the page catalog');

// ---------- every product, every size, on every mount
let sizes = 0;
const t0 = Date.now();
for (const p of products) {
  if (only && !only.has(p.slug) || !inPage.has(p.slug)) continue;
  const g = geomOf(p.slug), pc = cat.products.find(q => q.slug === p.slug);
  if (!g) bad(p.slug, null, 'geom file missing');
  for (const z of p.sizes) {
    if (!pc.sizes.includes(z.key)) { bad(p.slug, z.key, 'size missing from the page catalog'); continue; }
    sizes++;
    let m;
    try { m = await page.evaluate(a => window.__studio.hw.set(a), { slug: p.slug, size: z.key, target: 'all' }); }
    catch (e) { bad(p.slug, z.key, 'set() threw: ' + e.message.slice(0, 200)); continue; }
    const probs = [], e = expected(p, z, g);
    const placed = m.filter(r => r.placed && r.slug === p.slug && r.size === z.key);
    if (placed.length !== m.length) probs.push(`placed on ${placed.length} of ${m.length} mounts`);
    const bb = placed[0]?.bbox_mm;
    if (bb) {
      [['x', 0], ['y', 1], ['z', 2]].forEach(([k, i]) => { if (e[k] && Math.abs(bb[i] / e[k] - 1) > TOL) probs.push(`${k} ${bb[i]} vs ${(+e[k]).toFixed(2)} (${((bb[i] / e[k] - 1) * 100).toFixed(1)}%)`); });
      if (placed.some(r => r.bbox_mm.some((v, i) => Math.abs(v - bb[i]) > 0.01))) probs.push('bounds differ between mounts');
    }
    for (const r of placed) {
      if (!r.finite) probs.push(`non-finite vertices on ${r.mount}`);
      if (Math.abs(r.back_mm) > 0.05) probs.push(`${r.back_mm > 0 ? 'floats' : 'sunk'} ${r.back_mm} mm on ${r.mount}`);
      if (!r.base_on_panel) probs.push(`footprint crosses the frame of ${r.mount} while set on its recessed panel`);
      if (r.over_mm > 0.5 && r.fits && !r.edge) probs.push(`overhangs ${r.mount} by ${r.over_mm} mm`);
    }
    const rec = { slug: p.slug, size: z.key, bbox_mm: bb, expected: e, fits: placed.filter(r => r.fits).length + '/' + placed.length };
    if (probs.length) fail.push({ ...rec, problem: [...new Set(probs)].join('; '), notes: g?.sizes?.[z.key]?.fit?.notes || '' }); else pass.push(rec);
  }
}
const sweepSecs = (Date.now() - t0) / 1000;

// ---------- every finish once (body and accent on both targets)
const fins = {};
for (const [label, f] of Object.entries(cat.finishes)) {
  const p = cat.products.find(q => q.finishes.includes(label));
  await page.evaluate(a => window.__studio.hw.set(a), { slug: p.slug, size: p.sizes[0], finish: label, target: 'all' });
  const mats = await page.evaluate(() => window.__studio.hw.mats()), want = f.two_tone ? f.two_tone.map(l => cat.finishes[l]) : [f, f];
  const probs = [];
  for (const t of ['door', 'drawer']) ['body', 'accent'].forEach((k, i) => {
    const got = mats[t][k], w = want[i];
    if (got.color.some((v, j) => Math.abs(v - w.color[j]) > 2e-3) || Math.abs(got.roughness - w.roughness) > 1e-3 || Math.abs(got.metalness - (w.metalness ?? 1)) > 1e-3)
      probs.push(`${t} ${k}: got ${JSON.stringify(got)} want color ${w.color} r ${w.roughness} m ${w.metalness}`);
  });
  fins[label] = { product: p.slug, ok: !probs.length, ...(probs.length ? { problem: probs.join('; ') } : {}) };
  if (probs.length) bad(p.slug, null, `finish ${label}: ${probs.join('; ')}`);
}

// ---------- doors / drawers split, and the link that restores it
const split = { doors: 'kent_knob', drawers: 'kent_pull' };
await page.evaluate(() => window.__studio.hw.set({ slug: 'kent_knob', target: 'doors' }));
const m1 = await page.evaluate(() => window.__studio.hw.set({ slug: 'kent_pull', size: '6_5_16_inch_pull', finish: 'Honey Bronze', target: 'drawers' }));
const hash = await page.evaluate(() => location.hash);
const sum = m => m.map(r => `${r.target}:${r.slug}:${r.size}:${r.finish}:${r.placed}`).sort().join('|');
split.targets_ok = m1.every(r => r.slug === (r.target === 'doors' ? 'kent_knob' : 'kent_pull') && r.placed);
split.hash = hash;
const page2 = await open(hash);
split.restored_ok = sum(await page2.evaluate(() => window.__studio.hw.measure())) === sum(m1);
await page2.close();
await page.evaluate(() => window.__studio.hw.set({ slug: 'bar', finish: 'metal_black' }));
split.bar_ok = (await page.evaluate(() => window.__studio.hw.measure())).every(r => r.slug === 'bar' && r.placed);
if (!split.targets_ok || !split.restored_ok || !split.bar_ok) bad(null, null, 'doors / drawers split: ' + JSON.stringify(split));

await browser.close(); server.close();
const out = { when: new Date().toISOString(), page: BASE.replace(/^http:\/\/[^/]+\//, ''), tolerance: TOL,
  products: products.filter(p => !only || only.has(p.slug)).length, sizes, sweep_secs: sweepSecs, pass: pass.length, fail: fail.length,
  failures: fail, issues: [...new Map(issues.map(i => [i.kind + i.text, i])).values()], finishes: fins, split, passed: pass };
fs.writeFileSync(arg('--out') ? path.resolve(ROOT, arg('--out')) : path.join(TK, 'qa/sweep.json'), JSON.stringify(out, null, 1));
console.log(`${out.products} products, ${sizes} sizes in ${sweepSecs.toFixed(0)} s: ${pass.length} pass, ${fail.length} fail; ` +
  `${Object.values(fins).filter(f => f.ok).length}/${Object.keys(fins).length} finishes ok; split ${split.targets_ok && split.restored_ok && split.bar_ok ? 'ok' : 'FAILED'}; ${out.issues.length} console/network issues`);
for (const f of fail.slice(0, 60)) console.log(' ', f.slug, f.size || '', f.problem);
process.exit(fail.length || out.issues.length ? 1 : 0);
