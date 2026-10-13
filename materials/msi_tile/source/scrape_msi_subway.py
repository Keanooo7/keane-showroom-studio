import re,sys,json,html,time,urllib.request,os,csv,threading
from concurrent.futures import ThreadPoolExecutor
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
BASE="https://www.msisurfaces.com"
MAIN="/backsplash-tile/subway-tile/"
COLS={"Coronado":"/products/natural-stone-collections/marble/coronado/","Flamenco":"/porcelain-collection/flamenco/",
"Highland Park":"/ceramic-collection/highland-park/","Domino":"/porcelain-mosaics/domino/","Brickstone":"/porcelain/brickstone/",
"Urbano":"/ceramic-collection/urbano/","Hyde Studio":"/backsplash-tile/hyde-studio/","Village":"/ceramic/village/",
"Renzo":"/ceramic-collection/renzo/","Stella":"/ceramic-collection/stella/","Marza":"/ceramic-collection/marza/"}
outdir=sys.argv[1]; cache=os.path.join(outdir,"_cache"); os.makedirs(cache,exist_ok=True)
lock=threading.Lock()
def T(s): return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',s))).strip()
def get(path):
    fn=os.path.join(cache,re.sub(r'[^A-Za-z0-9]+','_',path).strip('_')[:180]+".html")
    if os.path.exists(fn): return open(fn).read()
    err=None
    for a in range(4):
        try:
            r=urllib.request.urlopen(urllib.request.Request(BASE+path,headers={"User-Agent":UA}),timeout=40)
            h=r.read().decode("utf-8","replace"); open(fn,"w").write(h); time.sleep(0.3); return h
        except Exception as e: err=e; time.sleep(2+a*2)
    print("FAILED",path,err,file=sys.stderr); return ""
def is_product(h): return 'id="innerDataList"' in h or 'PRODUCT DETAILS &amp; SPECS' in h or 'PRODUCT DETAILS & SPECS' in h
def main_region(h):
    i=h.find('id="ProductContent"')
    if i<0: i=h.find('<h1')
    r=h[i:]; j=min([x for x in (r.find('id="loadmorescroll"'),r.find('id="MediaContent"')) if x>0] or [len(r)])
    return r[:j]
TILE=re.compile(r'<a href="(/[^"#?]+)" data-href="[^"]*"([^>]*)>(.*?)</a>',re.S)
def tiles(h):
    out=[]
    for m in TILE.finditer(main_region(h)):
        img=re.search(r'data-src="([^"]+)"',m[3]); lab=re.search(r'<label[^>]*>(.*?)</label>',m[3],re.S)
        name=T(lab[1]) if lab else T(m[3])
        pid=re.search(r'data-product-id="(\d+)"',m[2])
        out.append(dict(path=m[1],name=name,thumbnail=img[1] if img else "",product_id=pid[1] if pid else "",
                        is_coll_link='collection page' in m[2]))
    return out
def list_page(path):
    h=get(path); items=tiles(h)
    tot=re.search(r'id="hdnPageTtl"[^>]*value="(\d+)"',h); tot=int(tot[1]) if tot else 1
    for p in range(2,tot+1): items+=tiles(get(path+"?handler=CollectionProductPartial&page=%d"%p))
    shown=re.search(r'Showing\s+(\d+)\s+of\s+(\d+)',h)
    return h,items,(int(shown[2]) if shown else None)
products={}  # path -> dict
def add(t,collection,trail):
    p=products.setdefault(t['path'],dict(path=t['path'],grid_name=t['name'],thumbnail=t['thumbnail'],product_id=t['product_id'],memberships=[]))
    if not p['product_id'] and t['product_id']: p['product_id']=t['product_id']
    mem=(collection," > ".join(trail))
    if list(mem) not in p['memberships']: p['memberships'].append(list(mem))
log=[]
def crawl(path,collection,trail,seen,depth=0):
    if path in seen or depth>4: return
    seen.add(path)
    h,items,expected=list_page(path)
    log.append(dict(collection=collection,page=BASE+path,trail=" > ".join(trail),tiles=len(items),site_count=expected))
    for t in items:
        if t['product_id'] and not t['is_coll_link']: add(t,collection,trail); continue
        sub=get(t['path'])
        if is_product(sub): add(t,collection,trail)
        else: crawl(t['path'],collection,trail+[t['name']],seen,depth+1)
_,grid,exp=list_page(MAIN)
log.append(dict(collection="Subway Tile (main grid)",page=BASE+MAIN,trail="",tiles=len(grid),site_count=exp))
for t in grid: add(t,"Subway Tile (main grid)",[])
for name,path in COLS.items(): crawl(path,name,[name],set())
print("unique product pages:",len(products),file=sys.stderr)
# ---- detail pages
def parse(path):
    h=get(path); d={}
    m=re.search(r'<h1[^>]*>(.*?)</h1>',h,re.S); d['name']=T(m[1]) if m else ""
    specs={}
    for m in re.finditer(r"<div class='specs-heading'><span>(.*?)</span></div><div class='specs-label'>(.*?)</div>",h,re.S):
        specs[T(m[1])]=T(m[2])
    d['specs']=specs
    ld={}
    for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>',h,re.S):
        try:
            j=json.loads(m[1])
            if isinstance(j,dict) and j.get('@type')=='Product': ld=j
            if isinstance(j,dict) and j.get('@type')=='BreadcrumbList': d['breadcrumb']=" > ".join(x.get('name','') for x in j.get('itemListElement',[]))
        except Exception: pass
    d['ld_description']=ld.get('description',''); d['main_image']=ld.get('image','') if isinstance(ld.get('image',''),str) else (ld.get('image') or [''])[0]
    d['primary_sku']=ld.get('productID','')
    d['ld_material']=ld.get('material','')
    d['product_category']=ld.get('productCategory',''); d['category']=ld.get('category','')
    m=re.search(r'Description:</?[^>]*>(.*?)(?:LEARN MORE|Order Sample|Check Inventory)',h,re.S)
    m2=re.search(r'PRODUCT DETAILS &(?:amp;)? SPECS.*?<(?:p|span|div)[^>]*>(.*?)</(?:p|span|div)>\s*(?:</div>\s*)*(?:<div class="row product-details-content">|<!--)',h,re.S)
    m2=re.search(r'id="productDetailsAccordion".*?<label>(.*?)</label>',h,re.S)
    d['description']=T(m2[1]) if m2 else ""
    apps={}
    ai=h.find('class="application-info"')
    if ai>=0:
        blk=h[ai:ai+4000]; cur=None
        for m in re.finditer(r"<div class='specs-heading[^']*'><b>(.*?)</b></div>|<div class='specs-label'>(.*?):\s*<b>(.*?)</b></div>",blk,re.S):
            if m[1]: cur=T(m[1]); apps[cur]={}
            elif cur: apps[cur][T(m[2])]=T(m[3])
    d['applications']=apps
    skus=[]
    for part in re.split(r'(?=<div class="border-bottom size-spec")',h)[1:]:
        part=part[:6000]
        idm=re.search(r'ID#:\s*<b[^>]*>(.*?)</b>',part)
        if not idm: continue
        sz=re.search(r'id="ancTileSize"[^>]*>(.*?)</span>',part,re.S); nm=re.search(r'id="divItemDesc"[^>]*>(.*?)</div>',part,re.S)
        fn=re.search(r'Finish:\s*<b>(.*?)</b>',part,re.S); im=re.search(r'<a href="(https://cdn[^"]+)"',part)
        skus.append(dict(sku=T(idm[1]),item_name=T(nm[1]) if nm else "",size=T(sz[1]) if sz else "",finish=T(fn[1]) if fn else "",image=im[1] if im else ""))
    d['skus']=skus
    return path,d
with ThreadPoolExecutor(4) as ex:
    for path,d in ex.map(parse,list(products)): products[path].update(d)
json.dump(dict(scraped_at=time.strftime("%Y-%m-%d"),source=BASE+MAIN,crawl_log=log,products=list(products.values())),open(os.path.join(outdir,"msi_subway_tile.json"),"w"),indent=1)
print(json.dumps(log,indent=0),file=sys.stderr)
