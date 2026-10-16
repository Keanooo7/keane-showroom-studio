"""Sherwin-Williams paint colours -> materials/sherwin_williams/colors.json.

The colour pages on sherwin-williams.com read one list from the shared colour service. It also carries archived colours
(color-family/archived-paint-colors/) and VS colours with no page on the site; both are left out, so the file holds the
colours a customer can still order. Each colour keeps its family (White, Neutral, Red ... from its page URL) and the
fan-deck strip and position where it has one.

    python3 tools/fetch_sw_colors.py
"""
import json, re
from pathlib import Path
from urllib.request import Request, urlopen

R = Path(__file__).resolve().parent.parent
OUT = R / 'materials/sherwin_williams'; OUT.mkdir(parents=True, exist_ok=True)
API = 'https://api.sherwin-williams.com/shared-color-service/colors?lng=en-US'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'

raw = json.loads(urlopen(Request(API, headers={'User-Agent': UA}), timeout=60).read())
colors = []
for c in raw:
    m = re.search(r'/color-family/([a-z]+)-paint-colors/', c.get('colorUrl') or '')
    if not c['identifier'].startswith('SW') or not m or m.group(1) == 'archived':
        continue
    colors.append(dict(code='SW ' + c['identifier'][2:], name=c['name'], hex='#' + c['hex'].lower(), family=m.group(1).title(),
                       strip=c['stripNumber'] or None, pos=c['stripPosition'] or None, url='https://www.sherwin-williams.com' + c['colorUrl']))
colors.sort(key=lambda c: c['code'])
rows = ',\n'.join(json.dumps(c) for c in colors)   # one colour per line: diffs stay readable when the range changes
(OUT / 'colors.json').write_text(f'{{"source": {json.dumps(API)}, "colors": [\n{rows}\n]}}\n')
fam = {}
for c in colors: fam[c['family']] = fam.get(c['family'], 0) + 1
print(f'{len(colors)} colours of {len(raw)} in the service ->', OUT.relative_to(R) / 'colors.json', fam)
