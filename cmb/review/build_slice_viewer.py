"""Slice viewer for quality control: TWO large axial images per subject, left the reference, right the overlays of
several models; the mouse wheel (or arrow keys) moves through z, both images stay in sync (user request 2026-09-29).

Differences to cmb.review.build_qc (the scroll sheet): one slice at a time instead of a sheet, and ANY number of
prediction runs side by side as coloured outlines that can be switched on and off. Same conventions otherwise: image as
acquired (microbleeds dark), 1-98 % window inside the brain, slices at roughly the ACQUIRED spacing with every grid slice
in between folded into the shown one (no finding can hide between two slices), fixed post-processing (threshold 0.3,
>= 2 mm3, giant components removed) and optionally the CSF rule per model (``NAME=FOLDER:csf`` needs ``--synthseg``).
No server: one HTML file plus an ``img/`` folder next to it. Private cohorts: build into a private folder, never send.

    python -m cmb.review.build_slice_viewer --dataset VALDO --manifest dev/manifest_valdo_gitter_d.json \
        --run basis=ergebnisse/turnier/ens2-a03-aniso-e60-gd "basis+liquor=ergebnisse/turnier/ens2-a03-aniso-e60-gd:csf" \
              kubisch=ergebnisse/turnier/ens2-a03-aniso-e60-gk --synthseg "dev/synthseg/{id}_synthseg.nii.gz" \
        --out ergebnisse/qc/valdo_slices [--limit N]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from typing import Dict, List, Optional
import nibabel as nib, numpy as np
from PIL import Image
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
CONNECTIVITY_26 = np.ones((3, 3, 3), bool)
THRESHOLD, MIN_MM3, MAX_VOXELS, VOXEL_MM3 = 0.3, 2.0, 2100, 0.25
CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}
TILE_WIDTH, MAX_TILES, OUTLINE_PX = 640, 80, 2
import colorsys
COLOURS = [(255, 69, 58), (255, 159, 10), (10, 200, 255), (200, 80, 255), (48, 209, 88), (255, 105, 180)] + [
    tuple(int(255 * v) for v in colorsys.hsv_to_rgb(((i * 0.618033) % 1.0), 0.85 if i % 2 else 0.6, 1.0)) for i in range(40)]
REF_COLOUR = (255, 214, 10)


def post_process(probability: np.ndarray, seg: Optional[np.ndarray]) -> np.ndarray:
    labels, n = ndimage.label(probability > THRESHOLD, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    if seg is not None:
        for c, box in enumerate(ndimage.find_objects(labels), start=1):
            if keep[c]:
                vals, counts = np.unique(seg[box][labels[box] == c], return_counts=True)
                if int(vals[np.argmax(counts)]) in CSF_LABELS:
                    keep[c] = False
    return keep[labels]


def outline(mask: np.ndarray, size) -> np.ndarray:
    big = np.asarray(Image.fromarray(np.rot90(mask).astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return big & ~ndimage.binary_erosion(big, iterations=OUTLINE_PX)


def acquired_slice_mm(case: Dict) -> float:
    info = f"{case['dir']}/info.json"
    if os.path.exists(info):
        spacing = json.load(open(info)).get("quelle_spacing")
        if spacing:
            return float(spacing[2])
    return 1.0


def build_patient(case: Dict, runs: List[Dict], synthseg: Optional[str], out_dir: str) -> Dict:
    cid = case["id"]
    load = lambda p: np.asarray(nib.load(p).dataobj)
    inverted = load(f"{case['dir']}/{cid}_image.nii.gz").astype(np.float32)
    reference = load(f"{case['dir']}/{cid}_label.nii.gz") > 0
    seg = None
    if synthseg and any(r["csf"] for r in runs):
        seg_img = nib.load(synthseg.format(id=cid, sid=cid.split("-", 1)[1] if "-" in cid else cid))
        if seg_img.shape != inverted.shape:                     # private cohorts: SynthSeg ran on the native 1 mm image
            from nibabel.processing import resample_from_to
            grid_img = nib.load(f"{case['dir']}/{cid}_image.nii.gz")
            seg_img = resample_from_to(seg_img, (grid_img.shape, grid_img.affine), order=0)
        seg = np.asarray(seg_img.dataobj).astype(np.int32)
    preds = {}
    for r in runs:
        found = glob.glob(f"{r['folder']}/fold*/{cid}_pred_proba.nii.gz")
        preds[r["name"]] = None
        if found:
            prob = load(found[0]).astype(np.float32) / 255.0
            if prob.shape == inverted.shape:
                preds[r["name"]] = post_process(prob, seg if r["csf"] else None)
            else:
                print(f"  {cid}: {r['name']} on another grid ({prob.shape} vs {inverted.shape}) -- skipped", flush=True)
    brain = inverted > 0
    shown = np.where(brain, 1.0 - inverted, 0.0)
    lo, hi = np.percentile(shown[brain], [1, 98])
    grey = (np.clip((shown - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8); grey[~brain] = 0
    xs, ys = np.nonzero(brain.any(axis=2)); zs = np.nonzero(brain.any(axis=(0, 1)))[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    step = max(1, int(round(acquired_slice_mm(case))))
    while (zs.max() - zs.min() + 1) / step > MAX_TILES:
        step += 1
    centres = list(range(int(zs.min()) + step // 2, int(zs.max()) + 1, step))
    tile_w = TILE_WIDTH; tile_h = int(round(tile_w * (x1 - x0) / (y1 - y0)))
    sheet = Image.new("L", (tile_w, len(centres) * tile_h))
    layers = {"ref": np.zeros((len(centres) * tile_h, tile_w, 4), np.uint8)}
    for r in runs:
        layers[r["name"]] = np.zeros((len(centres) * tile_h, tile_w, 4), np.uint8)
    ref_labels, n_ref = ndimage.label(reference, structure=CONNECTIVITY_26)
    vox_ml = float(np.prod(nib.load(f"{case['dir']}/{cid}_image.nii.gz").header.get_zooms()[:3])) / 1000.0
    ref_ml = round(float(reference.sum()) * vox_ml, 3)
    stats = {}; hit = {}
    for r in runs:
        p = preds[r["name"]]
        if p is None:
            continue
        pl, n_pred = ndimage.label(p, structure=CONNECTIVITY_26)
        hr = np.zeros(n_ref + 1, bool); hp = np.zeros(n_pred + 1, bool)
        both = reference & p
        hr[np.unique(ref_labels[both])] = True; hp[np.unique(pl[both])] = True; hr[0] = hp[0] = False
        hit[r["name"]] = (pl, hr, hp)
        stats[r["name"]] = dict(tp=int(hr.sum()), fn=int(n_ref - hr.sum()), fp=int(n_pred - hp.sum()), ml=round(float(p.sum()) * vox_ml, 3))
    tiles = []
    for k, z in enumerate(centres):
        lo_z, hi_z = z - step // 2, z - step // 2 + step
        sheet.paste(Image.fromarray(np.rot90(grey[x0:x1, y0:y1, z])).resize((tile_w, tile_h), Image.BILINEAR), (0, k * tile_h))
        box = slice(k * tile_h, (k + 1) * tile_h)
        flags = dict(z=int(z), ref=0)
        ref_here = reference[x0:x1, y0:y1, lo_z:hi_z].any(axis=2)
        if ref_here.any():
            layers["ref"][box][outline(ref_here, (tile_w, tile_h))] = REF_COLOUR + (255,)
            flags["ref"] = int(len(np.unique(ref_labels[x0:x1, y0:y1, lo_z:hi_z])) - 1)
        for i, r in enumerate(runs):
            p = preds[r["name"]]
            if p is None:
                continue
            here = p[x0:x1, y0:y1, lo_z:hi_z].any(axis=2)
            if here.any():
                layers[r["name"]][box][outline(here, (tile_w, tile_h))] = r["colour"] + (255,)
            pl, hr, hp = hit[r["name"]]
            pids = np.unique(pl[x0:x1, y0:y1, lo_z:hi_z]); pids = pids[pids > 0]
            ids = np.unique(ref_labels[x0:x1, y0:y1, lo_z:hi_z]); ids = ids[ids > 0]
            flags[r["name"]] = dict(fp=int((~hp[pids]).sum()), missed=int((~hr[ids]).sum()), detected=int(hr[ids].sum()))
        tiles.append(flags)
    os.makedirs(f"{out_dir}/img", exist_ok=True)
    sheet.save(f"{out_dir}/img/{cid}.jpg", quality=88)
    for name, layer in layers.items():
        Image.fromarray(layer, "RGBA").save(f"{out_dir}/img/{cid}_{name}.png", optimize=True)
    return dict(id=cid, cohort=case.get("cohort", ""), n_ref=int(n_ref), ref_ml=ref_ml, tiles=tiles, tile=[tile_w, tile_h], step_mm=step, stats=stats)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"><title>CMB slice viewer __DATASET__</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif;display:grid;grid-template-columns:230px 1fr;height:100vh}
#list{overflow:auto;border-right:1px solid #333;padding:6px}
#list input{width:95%;margin-bottom:6px;background:#222;color:#ddd;border:1px solid #444;padding:4px}
#list div.c{padding:4px 6px;cursor:pointer;border-radius:4px;white-space:nowrap}#list div.c:hover{background:#2a2a2a}#list div.c.sel{background:#375}
#main{display:grid;grid-template-rows:auto 1fr auto;height:100vh}
#bar{padding:6px 10px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}
#bar label{cursor:pointer;margin-right:6px;white-space:nowrap}
details{display:inline-block;vertical-align:top}details summary{cursor:pointer;color:#9cf;user-select:none}#info{max-width:min(70vw,900px)}#infotext{padding:4px 0;font-size:13px;color:#ccc}#infotext div{margin:2px 0}#infotext b{color:#fff}#bar button{margin-right:6px}#toggles{display:flex;flex-wrap:wrap;gap:2px 4px;max-width:70vw}#bar .sw{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px}
#panels{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:6px;min-height:0}
.panel{position:relative;background:#000;overflow:hidden;display:flex;align-items:center;justify-content:center}
.panel .stack{position:relative;overflow:hidden}
.panel img{position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:auto}
.panel .cap{position:absolute;left:8px;top:6px;color:#fff;text-shadow:0 0 4px #000;font-size:13px;z-index:9}
#foot{padding:6px 10px;border-top:1px solid #333;display:flex;gap:16px;align-items:center}
#foot input[type=range]{flex:1}
#sel{display:none}
#nav{display:none}
@media (max-width:900px), (orientation:portrait){
 body{display:block;height:auto;overflow:auto}
 #list{display:none}
 #sel{display:block;width:100%;font-size:16px;padding:8px;background:#222;color:#ddd;border:1px solid #444;margin:4px 0}
 #main{display:block;height:auto}
 #bar{font-size:15px;gap:8px}#bar .hint{display:none}#bar label{font-size:15px;padding:6px 4px}#bar button{font-size:16px;padding:8px 12px}
 #panels{display:grid;grid-template-columns:1fr;gap:3px;padding:3px}
 .panel{height:auto;touch-action:none}
 #cnt{display:none}#flags{display:none}
 #foot{position:sticky;bottom:0;background:#111;flex-wrap:nowrap;font-size:13px;padding:3px 6px;gap:6px;align-items:center}
 #zlab{white-space:nowrap;font-size:12px}#foot input[type=range]{flex:1;min-width:60px}
 #nav{display:flex;gap:4px;width:auto}#nav button{font-size:16px;padding:4px 9px;line-height:1.2}
 #bar{padding:3px 6px;gap:6px}#bar label{font-size:13px;padding:2px 2px}#bar button{font-size:13px;padding:4px 8px}#title{font-size:14px}details{display:block}#info{max-width:100%}
 #sel{font-size:14px;padding:5px;margin:2px 0}
}
</style></head><body>
<div id="list"><input id="q" placeholder="search case"><div id="cases"></div></div>
<div id="main">
<select id="sel"></select>
<div id="bar"><b id="title"></b><span id="cnt"></span><span style="flex:1"></span>
<details id="tgd" open><summary>Models</summary><span id="toggles"></span></details>
<details id="info"><summary>What do the models mean?</summary><div id="infotext"></div></details>
<span class="hint">mouse wheel / arrows: slice · PgUp/PgDn: case · R: reference on/off</span></div>
<div id="panels">
 <div class="panel"><div class="stack" id="left"><div class="cap">Reference</div></div></div>
 <div class="panel"><div class="stack" id="right"><div class="cap">Models</div></div></div>
</div>
<div id="foot"><span id="zlab"></span><input type="range" id="z" min="0" max="0" value="0"><span id="flags"></span>
<div id="nav"><button id="prevc" title="previous case">◀</button><button id="zm" title="slice up">▲</button><button id="zp" title="slice down">▼</button><button id="nextc" title="next case">▶</button></div></div>
</div>
<script>
const DATA=__DATA__; const RUNS=__RUNS__; let cur=0, zi=0, showRef=true; const on={}; RUNS.forEach(r=>on[r.name]=!!r.on);
function setAll(v){RUNS.forEach(r=>{on[r.name]=v;});document.querySelectorAll('#toggles input').forEach(i=>i.checked=v);R.querySelectorAll('img.ov').forEach(im=>im.style.display=v?'':'none');place();}
const L=document.getElementById('left'), R=document.getElementById('right');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function layer(stack,src,cls){const im=document.createElement('img');im.src=src;im.className=cls;stack.appendChild(im);return im;}
function buildList(f){const box=document.getElementById('cases');box.innerHTML='';DATA.forEach((c,i)=>{if(f&&!c.id.includes(f))return;const d=document.createElement('div');d.className='c'+(i===cur?' sel':'');
 const s=RUNS.length&&c.stats[RUNS[0].name]?` · ${RUNS[0].name} TP ${c.stats[RUNS[0].name].tp} FP ${c.stats[RUNS[0].name].fp} FN ${c.stats[RUNS[0].name].fn}`:'';
 d.textContent=`${c.id} · Ref ${c.n_ref}${s}`;d.onclick=()=>{show(i);};box.appendChild(d);});}
function fit(){const c=DATA[cur];const [tw,th]=c.tile;const ph=document.querySelector('.panel').clientHeight,pw=document.querySelector('.panel').clientWidth;
 const mobile=window.innerWidth<=900||window.innerHeight>window.innerWidth;const avail=window.innerHeight-document.getElementById('bar').offsetHeight-document.getElementById('sel').offsetHeight-document.getElementById('foot').offsetHeight-14;const s=mobile?Math.min(pw/tw,(avail/2)/th):Math.min(pw/tw,ph/th);[L,R].forEach(st=>{st.style.width=(tw*s)+'px';st.style.height=(th*s)+'px';});
 document.querySelectorAll('.stack img').forEach(im=>{im.style.width=(tw*s)+'px';im.style.height=(c.tiles.length*th*s)+'px';});place();}
function place(){const c=DATA[cur];const [tw,th]=c.tile;const s=L.clientWidth/tw;document.querySelectorAll('.stack img').forEach(im=>{im.style.top=(-zi*th*s)+'px';});
 document.getElementById('zlab').textContent=(window.innerWidth<=900||window.innerHeight>window.innerWidth)?`${zi+1}/${c.tiles.length}`:`slice ${zi+1} / ${c.tiles.length} (z=${c.tiles[zi].z}, ${c.step_mm} mm)`;document.getElementById('z').value=zi;
 const t=c.tiles[zi];let f=`reference here: ${t.ref}`;RUNS.forEach(r=>{if(t[r.name]&&on[r.name])f+=` · ${r.name}: detected ${t[r.name].detected}, missed ${t[r.name].missed}, false positives ${t[r.name].fp}`;});document.getElementById('flags').textContent=f;}
let show=function(i){cur=i;const c=DATA[cur];L.innerHTML='<div class="cap">Reference (yellow)</div>';R.innerHTML='<div class="cap">Models</div>';
 layer(L,`img/${c.id}.jpg`,'base');const rl=layer(L,`img/${c.id}_ref.png`,'ref');rl.style.display=showRef?'':'none';
 layer(R,`img/${c.id}.jpg`,'base');RUNS.forEach(r=>{const im=layer(R,`img/${c.id}_${r.name}.png`,'ov');im.dataset.run=r.name;im.style.display=on[r.name]?'':'none';});
 zi=Math.min(zi,c.tiles.length-1);const first=c.tiles.findIndex(t=>t.ref>0);if(first>=0&&arguments[1]!==false)zi=first;
 document.getElementById('z').max=c.tiles.length-1;document.getElementById('title').textContent=`${c.id} (${c.cohort})`;
 let cnt=`reference ${c.n_ref}${c.ref_ml!==undefined?` (${c.ref_ml.toFixed(3)} ml)`:''}`;RUNS.forEach(r=>{const s=c.stats[r.name];if(s)cnt+=` · ${r.name}: TP ${s.tp} FP ${s.fp} FN ${s.fn}${s.ml!==undefined?` (${s.ml.toFixed(3)} ml)`:''}`;});document.getElementById('cnt').textContent=cnt;
 buildList(document.getElementById('q').value);setTimeout(fit,30);};
function step(d){const n=DATA[cur].tiles.length;zi=Math.max(0,Math.min(n-1,zi+d));place();}
document.getElementById('panels').addEventListener('wheel',e=>{e.preventDefault();step(e.deltaY>0?1:-1);},{passive:false});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowDown'||e.key==='ArrowRight')step(1);else if(e.key==='ArrowUp'||e.key==='ArrowLeft')step(-1);
 else if(e.key==='PageDown')show(Math.min(DATA.length-1,cur+1));else if(e.key==='PageUp')show(Math.max(0,cur-1));else if(e.key==='r'||e.key==='R'){showRef=!showRef;L.querySelector('.ref').style.display=showRef?'':'none';}});
document.getElementById('z').addEventListener('input',e=>{zi=+e.target.value;place();});
document.getElementById('q').addEventListener('input',e=>buildList(e.target.value));
const it=document.getElementById('infotext');it.innerHTML=RUNS.map(r=>`<div><span class="sw" style="background:rgb(${r.colour})"></span><b>${esc(r.name)}</b>: ${esc(r.info||'')}</div>`).join('');
const mobile0=window.innerWidth<=900||window.innerHeight>window.innerWidth;if(mobile0)document.getElementById('tgd').removeAttribute('open');
document.getElementById('tgd').addEventListener('toggle',()=>setTimeout(fit,30));document.getElementById('info').addEventListener('toggle',()=>setTimeout(fit,30));
const tg=document.getElementById('toggles');const ba=document.createElement('button');ba.textContent='alle an';ba.onclick=()=>setAll(true);const bo=document.createElement('button');bo.textContent='alle aus';bo.onclick=()=>setAll(false);tg.appendChild(ba);tg.appendChild(bo);
RUNS.forEach(r=>{const lab=document.createElement('label');lab.title=r.info||'';lab.innerHTML=`<input type="checkbox" ${r.on?'checked':''}> <span class="sw" style="background:rgb(${r.colour})"></span>${esc(r.name)} `;
 lab.querySelector('input').onchange=ev=>{on[r.name]=ev.target.checked;R.querySelectorAll('img.ov').forEach(im=>{if(im.dataset.run===r.name)im.style.display=on[r.name]?'':'none';});place();};tg.appendChild(lab);});
const sel=document.getElementById('sel');DATA.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c.id} · Ref ${c.n_ref}`;sel.appendChild(o);});sel.onchange=e=>show(+e.target.value);
document.getElementById('prevc').onclick=()=>show(Math.max(0,cur-1));document.getElementById('nextc').onclick=()=>show(Math.min(DATA.length-1,cur+1));
document.getElementById('zm').onclick=()=>step(-1);document.getElementById('zp').onclick=()=>step(1);
let ty=null,tacc=0;const pn=document.getElementById('panels');
pn.addEventListener('touchstart',e=>{ty=e.touches[0].clientY;tacc=0;},{passive:true});
pn.addEventListener('touchmove',e=>{if(ty===null)return;const dy=e.touches[0].clientY-ty;ty=e.touches[0].clientY;tacc+=dy;while(tacc>=18){step(-1);tacc-=18;}while(tacc<=-18){step(1);tacc+=18;}e.preventDefault();},{passive:false});
pn.addEventListener('touchend',()=>{ty=null;});
const _show=show;show=function(i,keep){_show(i,keep);sel.value=cur;};
window.addEventListener('resize',fit);const h0=decodeURIComponent(location.hash.slice(1));const i0=DATA.findIndex(c=>c.id===h0);show(i0>=0?i0:0);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER[:csf]")
    ap.add_argument("--synthseg", default=None); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh-html", action="store_true", help="only rewrite betrachter.html from data.json (after editing PAGE)")
    ap.add_argument("--on", nargs="*", default=None, help="runs switched ON at start (default: the first run only)")
    ap.add_argument("--single-file", type=int, default=0, metavar="N", help="also write betrachter_einzeldatei.html with the N cases with most lesions embedded (0 = off, -1 = all)")
    ap.add_argument("--single-width", type=int, default=480); ap.add_argument("--single-quality", type=int, default=72)
    ap.add_argument("--info", nargs="*", default=[], metavar="NAME=TEXT", help="one-line explanation per run, shown under 'What do the models mean?' and as tooltip")
    a = ap.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    runs = []
    for i, spec in enumerate(a.run):
        name, folder = spec.split("=", 1); csf = folder.endswith(":csf"); folder = folder[:-4] if csf else folder
        runs.append(dict(name=name, folder=absolute(folder), csf=csf, colour=COLOURS[i % len(COLOURS)]))
    cases = [c for m in a.manifest for c in json.load(open(absolute(m)))]
    if a.limit:
        cases = cases[:a.limit]
    out = absolute(a.out); os.makedirs(f"{out}/img", exist_ok=True)
    if a.refresh_html:
        data = json.load(open(f"{out}/data.json"))
    else:
        data = []
        for k, c in enumerate(cases, 1):
            data.append(build_patient(c, runs, absolute(a.synthseg) if a.synthseg else None, out))
            print(f"[{k}/{len(cases)}] {c['id']} {data[-1]['stats']}", flush=True)
        json.dump(data, open(f"{out}/data.json", "w"))
    on = set(a.on) if a.on is not None else {runs[0]["name"]}
    info = dict(i.split("=", 1) for i in a.info)
    runs_js = json.dumps([dict(name=r["name"], colour=list(r["colour"]), on=r["name"] in on, info=info.get(r["name"], "")) for r in runs])
    html = PAGE.replace("__DATASET__", a.dataset).replace("__DATA__", json.dumps(data)).replace("__RUNS__", runs_js)
    open(f"{out}/betrachter.html", "w").write(html)
    if a.single_file:
        import base64, io
        chosen = sorted(data, key=lambda c: -c["n_ref"])[:a.single_file] if a.single_file > 0 else data
        embedded, uris = [], {}
        for c in chosen:
            cid = c["id"]; w0, h0 = c["tile"]; scale = a.single_width / w0; w1, h1 = a.single_width, int(round(h0 * scale))
            img = Image.open(f"{out}/img/{cid}.jpg"); img = img.resize((w1, int(round(img.height * scale))), Image.BILINEAR)
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=a.single_quality); uris[f"img/{cid}.jpg"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            for name in ["ref"] + [r["name"] for r in runs]:
                png = Image.open(f"{out}/img/{cid}_{name}.png"); png = png.resize((w1, int(round(png.height * scale))), Image.NEAREST)
                buf = io.BytesIO(); png.save(buf, "PNG", optimize=True); uris[f"img/{cid}_{name}.png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            embedded.append(dict(c, tile=[w1, h1]))
        single = PAGE.replace("__DATASET__", a.dataset + " (single file)").replace("__DATA__", json.dumps(embedded)).replace("__RUNS__", runs_js)
        single = single.replace("const DATA=", "const URIS=" + json.dumps(uris) + ";const DATA=").replace("im.src=src;", "im.src=URIS[src]||src;")
        path = f"{out}/betrachter_einzeldatei.html"; open(path, "w").write(single)
        print(f"-> {path} ({len(embedded)} cases, {os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"-> {out}/betrachter.html ({len(data)} cases, {len(runs)} models)")


if __name__ == "__main__":
    main()
