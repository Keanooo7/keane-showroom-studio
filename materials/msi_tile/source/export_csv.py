import json,sys,csv,os
src,outdir=sys.argv[1],sys.argv[2]
D=json.load(open(src)); P=D['products']; BASE="https://www.msisurfaces.com"
COLL_ORDER=["Subway Tile (main grid)","Coronado","Flamenco","Highland Park","Domino","Brickstone","Urbano","Hyde Studio","Village","Renzo","Stella","Marza"]
def norm(s): return s.strip().upper()
for p in P:
    skus=p['skus']; ld=norm(p.get('primary_sku',''))
    prim=next((s for s in skus if norm(s['sku'])==ld),None) or next((s for s in skus if ld and norm(s['sku']).startswith(ld)),None) or (skus[0] if skus else {})
    p['_prim']=prim
    p['_colls']=[c for c in COLL_ORDER if any(m[0]==c for m in p['memberships'])]
    p['_paths']=sorted({m[1] for m in p['memberships'] if m[1]})
SPEC=[("Primary Color(s)","primary_color"),("Material Type","material"),("Style","style"),("Available Finishes","available_finishes"),
      ("Collection","collection_spec"),("Shade Variations","shade_variations"),("Variations","variations"),("Tile Type","tile_type"),
      ("Chips Per Sheet","chips_per_sheet"),("Sheet Size","sheet_size"),("PEI RATING","pei_rating"),("RECTIFIED/NON-RECTIFIED","rectified"),
      ("Rectified/Pressed","rectified_pressed"),("Recommended Grout Color(s)","recommended_grout")]
APP=[]
for p in P:
    for g,v in p.get('applications',{}).items():
        for k in v:
            c=(g+" "+k).lower().replace(" ","_")
            if c not in APP: APP.append(c)
cols=["product_name","primary_sku","size","finish","item_description","collections","collection_subgroup","in_main_subway_grid"]+[k for _,k in SPEC]+\
     APP+["description","product_url","image_url","thumbnail_url","msi_product_id","sibling_skus_on_page"]
rows=[]
for p in P:
    a=p.get('applications',{}); pr=p['_prim']
    r=dict(product_name=p['name'],primary_sku=pr.get('sku',''),size=pr.get('size',''),finish=pr.get('finish',''),item_description=pr.get('item_name',''),
           collections="; ".join(c for c in p['_colls'] if c!="Subway Tile (main grid)"),collection_subgroup="; ".join(x for x in p['_paths'] if ">" in x),
           in_main_subway_grid="Yes" if "Subway Tile (main grid)" in p['_colls'] else "No",
           description=p['description'],product_url=BASE+p['path'],image_url=pr.get('image') or p.get('main_image',''),thumbnail_url=p['thumbnail'],
           msi_product_id=p['product_id'],sibling_skus_on_page="; ".join(s['sku'] for s in p['skus'] if s is not pr))
    for k,c in SPEC: r[c]=p['specs'].get(k,'')
    if not r['material']: r['material']=p.get('ld_material','')
    for g,v in a.items():
        for k,val in v.items(): r[(g+" "+k).lower().replace(" ","_")]=val
    rows.append(r)
rows.sort(key=lambda r:(r['collections']=="" ,r['collections'],r['product_name'].lower()))
with open(os.path.join(outdir,"msi_subway_tile_products.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,cols); w.writeheader(); w.writerows(rows)
# unique SKUs
home={}; 
for p in P:
    if p['_prim']: home.setdefault(p['_prim']['sku'],p)
S={}
for p in P:
    for s in p['skus']:
        e=S.setdefault(s['sku'],dict(sku=s['sku'],item_description=s['item_name'],size=s['size'],finish=s['finish'],image_url=s['image'],colls=set(),pages=[]))
        e['colls'].update(p['_colls']); e['pages'].append(BASE+p['path'])
srows=[]
for k,e in S.items():
    h=home.get(k)
    srows.append(dict(sku=k,item_description=e['item_description'],size=e['size'],finish=e['finish'],
        product_name=h['name'] if h else "",product_url=BASE+h['path'] if h else "",
        has_own_product_page="Yes" if h else "No (listed as a size/trim on sibling pages)",
        material=(h['specs'].get('Material Type','') or h.get('ld_material','')) if h else "",
        collections="; ".join(c for c in COLL_ORDER if c in e['colls'] and c!="Subway Tile (main grid)"),
        image_url=e['image_url'],listed_on_pages="; ".join(sorted(set(e['pages']))[:5])+(" ..." if len(set(e['pages']))>5 else "")))
srows.sort(key=lambda r:(r['collections']=="",r['collections'],r['item_description'].upper()))
with open(os.path.join(outdir,"msi_subway_tile_skus.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,list(srows[0])); w.writeheader(); w.writerows(srows)
for p in P: p.pop('_prim'); p['collections']=p.pop('_colls'); p.pop('_paths')
json.dump(D,open(os.path.join(outdir,"msi_subway_tile_full.json"),"w"),indent=1)
from collections import Counter
print("product pages:",len(rows),"| unique SKUs:",len(srows),"| SKUs w/o own page:",sum(1 for r in srows if not r['product_url']))
c=Counter(); 
for p in P:
    for x in p['collections']: c[x]+=1
for k in COLL_ORDER: print(f"  {k:26s} {c[k]}")
print("only in collections (not in main grid):",sum(1 for r in rows if r['in_main_subway_grid']=="No"))
