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
def _model_colours(n=40, avoid=(0.50, 0.74)):
    """Golden-ratio hues for the model outlines, skipping the blue band reserved for the reference."""
    out, i = [], 0
    while len(out) < n:
        hue = (i * 0.618033) % 1.0
        if not avoid[0] <= hue <= avoid[1]:
            out.append(tuple(int(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.85 if i % 2 else 0.6, 1.0)))
        i += 1
    return out


COLOURS = _model_colours()
REF_COLOUR = (66, 150, 255)   # reference (ground truth) in blue; model colours avoid the blue band


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


VERDICT_LAYERS = [("TP", (0, 200, 0)), ("FP", (255, 205, 0)), ("FN", (235, 60, 60))]   # detected / false alarm / missed


def build_patient(case: Dict, runs: List[Dict], synthseg: Optional[str], out_dir: str, verdict: bool = False) -> Dict:
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
    if verdict:
        for name, _ in VERDICT_LAYERS:
            layers[name] = np.zeros((len(centres) * tile_h, tile_w, 4), np.uint8)
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
            if verdict and i == 0:                                   # one model: colour by verdict instead of by model
                sub_p = pl[x0:x1, y0:y1, lo_z:hi_z]; sub_r = ref_labels[x0:x1, y0:y1, lo_z:hi_z]
                masks = {"TP": (hp[sub_p] & (sub_p > 0)).any(axis=2), "FP": (~hp[sub_p] & (sub_p > 0)).any(axis=2),
                         "FN": (~hr[sub_r] & (sub_r > 0)).any(axis=2)}
                for name, colour in VERDICT_LAYERS:
                    if masks[name].any():
                        layers[name][box][outline(masks[name], (tile_w, tile_h))] = colour + (255,)
        tiles.append(flags)
    os.makedirs(f"{out_dir}/img", exist_ok=True)
    sheet.save(f"{out_dir}/img/{cid}.jpg", quality=88)
    for name, layer in layers.items():
        Image.fromarray(layer, "RGBA").save(f"{out_dir}/img/{cid}_{name}.png", optimize=True)
    return dict(id=cid, cohort=case.get("cohort", ""), n_ref=int(n_ref), ref_ml=ref_ml, tiles=tiles, tile=[tile_w, tile_h], step_mm=step, stats=stats)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"><title>CMB slice viewer __DATASET__</title>
<style>
body{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif;display:grid;grid-template-columns:var(--side,420px) 6px minmax(0,1fr);height:100vh}
body.nolist{grid-template-columns:0 6px 1fr}body.nolist #list{display:none}body.nolist #split{display:none}
#split{cursor:col-resize;background:#1c1c1c;border-right:1px solid #333}#split:hover{background:#2f6}
#list{overflow:auto;border-right:1px solid #333;padding:6px}
#list input{width:95%;margin-bottom:4px;background:#222;color:#ddd;border:1px solid #444;padding:5px;font-size:15px}#sum{font-size:14px;color:#aaa;line-height:1.35;padding:2px 8px 6px;border-bottom:1px solid #333;margin-bottom:4px;white-space:nowrap}#sum b{color:#ddd}
#list div.c{padding:5px 8px;cursor:pointer;border-radius:4px;white-space:nowrap;line-height:1.3}#list div.c b{font-weight:600}#list div.c:hover{background:#2a2a2a}#list div.c.sel{background:#375}
#list div.c .dot{display:inline-block;width:10px;height:10px;border-radius:50%;border:1px solid #555;margin-right:7px;vertical-align:-1px}#list div.c .dot.on{background:#3c3;border-color:#3c3}
#rev{font-size:13px;color:#aaa;line-height:1.3;padding:2px 8px 6px;border-bottom:1px solid #333;margin-bottom:4px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}#rev b{color:#ddd}#rev button{margin-right:2px}
#list .m{display:grid;grid-template-columns:max-content repeat(4,max-content);gap:0 10px;color:#bbb;font-size:14px;font-variant-numeric:tabular-nums}#list .m span:first-child{color:#ddd}#list .m b{color:#fff}#list .m span:nth-child(n+3){text-align:right}
#main{min-width:0;display:grid;grid-template-columns:minmax(0,1fr);grid-template-rows:auto 1fr auto;height:100vh}#bar,#foot,#panels{min-width:0}
#bar{padding:2px 8px;display:flex;gap:10px;align-items:center;flex-wrap:nowrap;white-space:nowrap;border-bottom:1px solid #333;font-size:14px;min-height:28px}#cnt{color:#ccc;flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis}#title{flex:0 0 auto}
#bar label{cursor:pointer;margin-right:4px;white-space:nowrap}
#info{position:relative;flex:0 0 auto}#info summary{cursor:pointer;color:#9cf;user-select:none;list-style:none;border:1px solid #556;border-radius:10px;padding:0 7px}#info summary::-webkit-details-marker{display:none}#infotext{position:absolute;right:0;top:24px;z-index:20;background:#222;border:1px solid #555;border-radius:4px;padding:8px 10px;width:max-content;max-width:min(80vw,760px);white-space:normal;font-size:13px;color:#ccc;box-shadow:0 4px 16px #000}#infotext div{margin:3px 0}#infotext b{color:#fff}#bar button{margin-right:2px;padding:1px 6px}#toggles{display:flex;flex-wrap:nowrap;gap:2px 4px;align-items:center;flex:0 0 auto}#bar .sw{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px}
#panels{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px;min-height:0}
.panel{position:relative;background:#000;overflow:hidden;display:flex;align-items:center;justify-content:center;min-width:0;min-height:0}
.panel .stack{position:relative;overflow:hidden;transform-origin:center center;will-change:transform;cursor:grab;flex:0 0 auto}
.panel img{position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:auto}
.panel .cap{position:absolute;left:8px;top:6px;color:#fff;text-shadow:0 0 4px #000;font-size:13px;z-index:9}
#foot{padding:2px 8px;border-top:1px solid #333;display:flex;gap:10px;align-items:center;white-space:nowrap;font-size:14px;min-height:28px}
#foot input[type=range]{flex:1 1 140px;min-width:80px;margin:0}#zlab{flex:0 0 auto}#flags{flex:2 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;color:#ccc}
#sel{display:none}
#nav{display:flex;gap:3px;flex:0 0 auto}#nav button{padding:1px 7px}
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
 #nav{display:flex;gap:3px;width:auto;flex:0 0 auto}#nav button{font-size:15px;padding:4px 7px;line-height:1.2}#zoomin,#zoomout{display:none}#foot input[type=range]{flex:1 1 90px;min-width:90px}#zlab{flex:0 0 auto}
 #bar{padding:3px 6px;gap:6px;flex-wrap:wrap;white-space:normal}#bar label{font-size:13px;padding:2px 2px}#bar button{font-size:13px;padding:4px 8px}#title{font-size:14px}#infotext{max-width:92vw}
 #sel{font-size:14px;padding:5px;margin:2px 0}
}
</style></head><body>
<div id="list"><input id="q" placeholder="search case"><div id="sum"></div><div id="rev"><span id="revcount"></span><button id="bRevAll" title="mark all cases as reviewed">all</button><button id="bRevNone" title="unmark all cases">none</button><button id="bRevSave" title="save the checklist as review_status.json (download, plus to the live server when running)">save</button><button id="bRevLoad" title="load a saved checklist">load</button><input type="file" id="revfile" accept=".json" style="display:none"></div><div id="cases"></div></div>
<div id="split" title="drag: resize the case list · double-click or L: hide / show it"></div>
<div id="main">
<select id="sel"></select>
<div id="bar"><button id="listtoggle" title="hide / show the case list (L)">&#9776;</button><b id="title"></b><span id="cnt"></span>
<span id="toggles"></span>
<details id="info"><summary title="keys, mouse, and what the models mean">?</summary><div id="infotext"></div></details></div>
<div id="panels">
 <div class="panel"><div class="stack" id="left"><div class="cap">Reference</div></div></div>
 <div class="panel"><div class="stack" id="right"><div class="cap">Models</div></div></div>
</div>
<div id="foot"><span id="zlab"></span><input type="range" id="z" min="0" max="0" value="0"><span id="flags"></span>
<div id="nav"><button id="prevc" title="previous case (↑)">▲</button><button id="zm" title="previous slice (←)">◀</button><button id="zp" title="next slice (→)">▶</button><button id="nextc" title="next case (↓)">▼</button><button id="zoomout" title="zoom out">−</button><button id="zoomin" title="zoom in">+</button><button id="zoomreset" title="reset zoom">1:1</button></div></div>
</div>
<script>
const DATA=__DATA__; const RUNS=__RUNS__; const VERDICT=__VERDICT__;
const LAYERS=VERDICT?[{name:'TP',colour:[0,200,0],on:true,info:'detected (true positive): a predicted component touches a reference lesion'},{name:'FP',colour:[255,205,0],on:true,info:'false alarm (false positive): a predicted component touches no reference lesion'},{name:'FN',colour:[235,60,60],on:true,info:'missed (false negative): a reference lesion that no predicted component touches'}]:RUNS;
let cur=0, zi=0, showRef=true; const on={}; LAYERS.forEach(r=>on[r.name]=!!r.on);
function setAll(v){LAYERS.forEach(r=>{on[r.name]=v;});document.querySelectorAll('#toggles input').forEach(i=>i.checked=v);R.querySelectorAll('img.ov').forEach(im=>im.style.display=v?'':'none');place();}
const L=document.getElementById('left'), R=document.getElementById('right');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function f1(s){const d=2*s.tp+s.fp+s.fn;return d?2*s.tp/d:null;}function fmt(v){return v===null?'–':v.toFixed(2);}
function summary(){const box=document.getElementById('sum');let h=`<b>${DATA.length} cases</b> · ${DATA.reduce((a,c)=>a+c.n_ref,0)} reference lesions`;RUNS.forEach(r=>{const cs=DATA.filter(c=>c.stats[r.name]);const vals=cs.map(c=>f1(c.stats[r.name])).filter(v=>v!==null);const pooled=cs.reduce((a,c)=>{const s=c.stats[r.name];a.tp+=s.tp;a.fp+=s.fp;a.fn+=s.fn;return a;},{tp:0,fp:0,fn:0});const mean=vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null;h+=`<br><span class="sw" style="background:rgb(${r.colour})"></span><b>${esc(r.name)}</b>: F1 mean ${fmt(mean)} · pooled ${fmt(f1(pooled))}`;});box.innerHTML=h;}
function layer(stack,src,cls){const im=document.createElement('img');im.src=src;im.className=cls;stack.appendChild(im);return im;}
function buildList(f){const box=document.getElementById('cases');box.innerHTML='';DATA.forEach((c,i)=>{if(f&&!c.id.includes(f))return;const d=document.createElement('div');d.className='c'+(i===cur?' sel':'');d.dataset.id=c.id;
 const st=RUNS.filter(r=>c.stats[r.name]).map(r=>{const s=c.stats[r.name];return `<span>${esc(r.name)}</span><span><b>F1 ${fmt(f1(s))}</b></span><span>TP ${s.tp}</span><span>FP ${s.fp}</span><span>FN ${s.fn}</span>`;}).join('');
 d.innerHTML=`<span class="dot${reviewed[c.id]?' on':''}" title="toggle reviewed (M)"></span><b>${esc(c.id)}</b> · Ref ${c.n_ref}${st?`<div class="m">${st}</div>`:''}`;d.onclick=ev=>{if(ev.target.classList.contains('dot')){toggleReviewed(c.id);return;}show(i);};box.appendChild(d);});}
function fit(){const c=DATA[cur];const [tw,th]=c.tile;const ph=document.querySelector('.panel').clientHeight,pw=document.querySelector('.panel').clientWidth;
 const mobile=window.innerWidth<=900||window.innerHeight>window.innerWidth;const avail=window.innerHeight-document.getElementById('bar').offsetHeight-document.getElementById('sel').offsetHeight-document.getElementById('foot').offsetHeight-14;const s=mobile?Math.min(pw/tw,(avail/2)/th):Math.min(pw/tw,ph/th);scale=s;[L,R].forEach(st=>{st.style.width=(tw*s)+'px';st.style.height=(th*s)+'px';});
 document.querySelectorAll('.stack img').forEach(im=>{im.style.width=(tw*s)+'px';im.style.height=(c.tiles.length*th*s)+'px';});place();}
let scale=1;function place(){const c=DATA[cur];const [tw,th]=c.tile;const s=scale;document.querySelectorAll('.stack img').forEach(im=>{im.style.top=(-zi*th*s)+'px';});
 document.getElementById('zlab').textContent=(window.innerWidth<=900||window.innerHeight>window.innerWidth)?`${zi+1}/${c.tiles.length}`:`slice ${zi+1} / ${c.tiles.length} (z=${c.tiles[zi].z}, ${c.step_mm} mm)`;document.getElementById('z').value=zi;
 const t=c.tiles[zi];let f=`reference here: ${t.ref}`;RUNS.forEach(r=>{if(t[r.name]&&on[r.name]!==false)f+=` · ${r.name}: detected ${t[r.name].detected}, missed ${t[r.name].missed}, false positives ${t[r.name].fp}`;});document.getElementById('flags').textContent=f;}
let show=function(i){cur=i;const c=DATA[cur];L.innerHTML='<div class="cap">Reference (blue)</div>';R.innerHTML='<div class="cap">'+(VERDICT?'our model, by verdict':'Models')+'</div>';
 layer(L,`img/${c.id}.jpg`,'base');const rl=layer(L,`img/${c.id}_ref.png`,'ref');rl.style.display=showRef?'':'none';
 layer(R,`img/${c.id}.jpg`,'base');LAYERS.forEach(r=>{const im=layer(R,`img/${c.id}_${r.name}.png`,'ov');im.dataset.run=r.name;im.style.display=on[r.name]?'':'none';});
 zi=Math.min(zi,c.tiles.length-1);const first=c.tiles.findIndex(t=>t.ref>0);if(first>=0&&arguments[1]!==false)zi=first;
 document.getElementById('z').max=c.tiles.length-1;document.getElementById('title').textContent=`${c.id} (${c.cohort})`;
 let cnt=`reference ${c.n_ref}${c.ref_ml!==undefined?` (${c.ref_ml.toFixed(3)} ml)`:''}`;RUNS.forEach(r=>{const s=c.stats[r.name];if(s)cnt+=` · ${r.name}: F1 ${fmt(f1(s))} (TP ${s.tp} FP ${s.fp} FN ${s.fn}${s.ml!==undefined?`, ${s.ml.toFixed(3)} ml`:''})`;});const ce=document.getElementById('cnt');ce.textContent=cnt;ce.title=cnt;
 buildList(document.getElementById('q').value);fit();setTimeout(fit,30);};
function step(d){const n=DATA[cur].tiles.length;zi=Math.max(0,Math.min(n-1,zi+d));place();}
let zoom=1,px=0,py=0;function applyZoom(){[L,R].forEach(st=>{st.style.transform=`translate(${px}px,${py}px) scale(${zoom})`;st.style.cursor=zoom>1?'grab':'default';});document.getElementById('zoomreset').textContent=zoom.toFixed(1)+'x';}
function clampPan(){const w=L.clientWidth,h=L.clientHeight;const mx=Math.max(0,(w*zoom-w)/2),my=Math.max(0,(h*zoom-h)/2);px=Math.max(-mx,Math.min(mx,px));py=Math.max(-my,Math.min(my,py));}
function zoomAt(f,cx,cy){const nz=Math.max(1,Math.min(12,zoom*f));if(cx!==undefined){const r=L.getBoundingClientRect();const ox=cx-(r.left+r.width/2),oy=cy-(r.top+r.height/2);px=ox-(ox-px)*(nz/zoom);py=oy-(oy-py)*(nz/zoom);}zoom=nz;if(zoom===1){px=0;py=0;}clampPan();applyZoom();}
function resetZoom(){zoom=1;px=0;py=0;applyZoom();}
document.getElementById('panels').addEventListener('wheel',e=>{e.preventDefault();const r=(e.currentTarget.querySelector('.stack')||L).getBoundingClientRect();zoomAt(e.deltaY<0?1.2:1/1.2,e.clientX,e.clientY);},{passive:false});
document.getElementById('zoomin').onclick=()=>zoomAt(1.5);document.getElementById('zoomout').onclick=()=>zoomAt(1/1.5);document.getElementById('zoomreset').onclick=resetZoom;
let drag=null;document.getElementById('panels').addEventListener('mousedown',e=>{if(zoom<=1||e.button!==0)return;drag={x:e.clientX-px,y:e.clientY-py};e.preventDefault();});
window.addEventListener('mousemove',e=>{if(!drag)return;px=e.clientX-drag.x;py=e.clientY-drag.y;clampPan();applyZoom();});window.addEventListener('mouseup',()=>{drag=null;});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowRight'){step(1);e.preventDefault();}else if(e.key==='ArrowLeft'){step(-1);e.preventDefault();}
 else if(e.key==='ArrowDown'||e.key==='PageDown'){show(Math.min(DATA.length-1,cur+1));e.preventDefault();}else if(e.key==='ArrowUp'||e.key==='PageUp'){show(Math.max(0,cur-1));e.preventDefault();}else if(e.key==='r'||e.key==='R'){showRef=!showRef;L.querySelector('.ref').style.display=showRef?'':'none';}});
document.getElementById('z').addEventListener('input',e=>{zi=+e.target.value;place();});
document.getElementById('q').addEventListener('input',e=>buildList(e.target.value));
const it=document.getElementById('infotext');it.innerHTML='<div><b>Keys</b>: ← → slice · ↑ ↓ (or PgUp / PgDn) previous / next case · R reference on/off · L case list</div><div><b>Mouse</b>: wheel / pinch zoom · drag pan · click a case in the list</div><div><b>Left</b>: reference lesions (blue). <b>Right</b>: '+(VERDICT?'our model only, coloured by verdict: <span class="sw" style="background:rgb(0,200,0)"></span>green = detected, <span class="sw" style="background:rgb(255,205,0)"></span>yellow = false alarm, <span class="sw" style="background:rgb(235,60,60)"></span>red = missed; switch them with the checkboxes.':'model outlines, switch them with the checkboxes.')+' F1 = 2·TP / (2·TP + FP + FN) per case; the list head shows the mean over cases and the pooled value.</div>'+RUNS.map(r=>`<div><span class="sw" style="background:rgb(${r.colour})"></span><b>${esc(r.name)}</b>: ${esc(r.info||'')}</div>`).join('');
document.addEventListener('click',e=>{const d=document.getElementById('info');if(d.open&&!d.contains(e.target))d.removeAttribute('open');});
const tg=document.getElementById('toggles');const ba=document.createElement('button');ba.textContent='all on';ba.onclick=()=>setAll(true);const bo=document.createElement('button');bo.textContent='all off';bo.onclick=()=>setAll(false);tg.appendChild(ba);tg.appendChild(bo);
LAYERS.forEach(r=>{const lab=document.createElement('label');lab.title=r.info||'';lab.innerHTML=`<input type="checkbox" ${r.on?'checked':''}> <span class="sw" style="background:rgb(${r.colour})"></span>${esc(r.name)} `;
 lab.querySelector('input').onchange=ev=>{on[r.name]=ev.target.checked;R.querySelectorAll('img.ov').forEach(im=>{if(im.dataset.run===r.name)im.style.display=on[r.name]?'':'none';});place();};tg.appendChild(lab);});
const sel=document.getElementById('sel');DATA.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c.id} · Ref ${c.n_ref}`;sel.appendChild(o);});sel.onchange=e=>show(+e.target.value);
document.getElementById('prevc').onclick=()=>show(Math.max(0,cur-1));document.getElementById('nextc').onclick=()=>show(Math.min(DATA.length-1,cur+1));
document.getElementById('zm').onclick=()=>step(-1);document.getElementById('zp').onclick=()=>step(1);
const pn=document.getElementById('panels');let t1=null,pinch=null;
function dist(e){const a=e.touches[0],b=e.touches[1];return Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY);}
pn.addEventListener('touchstart',e=>{if(e.touches.length===2){pinch={d:dist(e),z:zoom,cx:(e.touches[0].clientX+e.touches[1].clientX)/2,cy:(e.touches[0].clientY+e.touches[1].clientY)/2};t1=null;}else if(e.touches.length===1){t1={x:e.touches[0].clientX-px,y:e.touches[0].clientY-py};}},{passive:true});
pn.addEventListener('touchmove',e=>{if(pinch&&e.touches.length===2){const f=dist(e)/pinch.d;const target=Math.max(1,Math.min(12,pinch.z*f));zoomAt(target/zoom,pinch.cx,pinch.cy);e.preventDefault();}else if(t1&&e.touches.length===1&&zoom>1){px=e.touches[0].clientX-t1.x;py=e.touches[0].clientY-t1.y;clampPan();applyZoom();e.preventDefault();}},{passive:false});
let lastTap=0;pn.addEventListener('touchend',e=>{if(e.touches.length<2)pinch=null;if(e.touches.length===0){t1=null;const now=Date.now();if(now-lastTap<300&&zoom>1){resetZoom();}lastTap=now;}});
const _show=show;show=function(i,keep){_show(i,keep);sel.value=cur;resetZoom();try{history.replaceState(null,'','#'+encodeURIComponent(DATA[cur].id));}catch(e){}};
const _place=place;place=function(){_place();try{sessionStorage.setItem('qc_z_'+DATA[cur].id,zi);}catch(e){}};
// review checklist: mark cases as reviewed (dot, M key); survives reloads (localStorage per cohort); saveable as review_status.json
(function(){
 const key='qc_review_'+(DATA[0]&&DATA[0].cohort?DATA[0].cohort:location.pathname);
 let reviewed={};try{reviewed=JSON.parse(localStorage.getItem(key)||'{}');}catch(e){}
 window.reviewed=reviewed;
 const save0=()=>{try{localStorage.setItem(key,JSON.stringify(reviewed));}catch(e){}};
 const count=()=>DATA.filter(c=>reviewed[c.id]).length;
 const render=()=>{const el=document.getElementById('revcount');if(el)el.innerHTML=`reviewed <b>${count()}</b> / ${DATA.length}`;
  document.querySelectorAll('#cases div.c').forEach(d=>{const dot=d.querySelector('.dot');if(dot)dot.classList.toggle('on',!!reviewed[d.dataset.id]);});};
 window.toggleReviewed=id=>{if(reviewed[id])delete reviewed[id];else reviewed[id]=1;save0();render();};
 const setAll=v=>{DATA.forEach(c=>{if(v)reviewed[c.id]=1;else delete reviewed[c.id];});save0();render();};
 const payload=()=>({cohort:DATA[0]&&DATA[0].cohort||'',page:location.pathname,saved:new Date().toISOString(),total:DATA.length,reviewed:DATA.filter(c=>reviewed[c.id]).map(c=>c.id)});
 const note=t=>{const el=document.getElementById('revcount');el.innerHTML+=` <b>${t}</b>`;setTimeout(render,1500);};
 async function save(){
  const txt=JSON.stringify(payload(),null,1);
  try{const r=await fetch(location.pathname,{method:'POST',headers:{'Content-Type':'application/json'},body:txt});
   if(r.ok){note('saved to server');return;}}catch(e){}
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([txt],{type:'application/json'}));a.download='review_status.json';a.click();note('downloaded');}
 document.getElementById('bRevAll').onclick=()=>setAll(true);
 document.getElementById('bRevNone').onclick=()=>setAll(false);
 document.getElementById('bRevSave').onclick=save;
 document.getElementById('bRevLoad').onclick=()=>document.getElementById('revfile').click();
 document.getElementById('revfile').onchange=async e=>{const f=e.target.files[0];if(!f)return;
  try{const p=JSON.parse(await f.text());const seen={};(p.reviewed||[]).forEach(id=>seen[id]=1);DATA.forEach(c=>{if(seen[c.id])reviewed[c.id]=1;else delete reviewed[c.id];});save0();render();}
  catch(err){alert('could not read checklist: '+err.message);}e.target.value='';};
 document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='m'||e.key==='M'){toggleReviewed(DATA[cur].id);e.preventDefault();}});
 render();
})();
summary();window.addEventListener('resize',fit);const h0=decodeURIComponent(location.hash.slice(1));const i0=DATA.findIndex(c=>c.id===h0);show(i0>=0?i0:0);
try{const z0=sessionStorage.getItem('qc_z_'+DATA[cur].id);if(i0>=0&&z0!==null){zi=Math.max(0,Math.min(DATA[cur].tiles.length-1,+z0));place();}}catch(e){}
// live mode (python -m cmb.review.live_server): poll the page's Last-Modified and reload when it was rebuilt; inert on file://
if(/^https?:/.test(location.protocol)){let seen=null;setInterval(()=>fetch(location.pathname,{method:'HEAD',cache:'no-store'}).then(r=>{const v=(r.headers.get('Last-Modified')||'')+'|'+(r.headers.get('Content-Length')||'');if(seen!==null&&v!==seen)location.reload();seen=v;}).catch(()=>{}),1500);}

// case list: draggable width, collapsible (button, double-click on the divider, key L); remembered in this browser
(function(){const sp=document.getElementById('split');let w=null;try{w=localStorage.getItem('qc_side');if(localStorage.getItem('qc_nolist')==='1')document.body.classList.add('nolist');}catch(e){}
 if(w)document.body.style.setProperty('--side',w+'px');
 function toggle(){document.body.classList.toggle('nolist');try{localStorage.setItem('qc_nolist',document.body.classList.contains('nolist')?'1':'0');}catch(e){}}
 document.getElementById('listtoggle').onclick=toggle;sp.ondblclick=toggle;
 sp.onmousedown=e=>{e.preventDefault();const mv=ev=>{const px=Math.max(160,Math.min(700,ev.clientX));document.body.classList.remove('nolist');document.body.style.setProperty('--side',px+'px');};
  const up=()=>{document.removeEventListener('mousemove',mv);document.removeEventListener('mouseup',up);try{localStorage.setItem('qc_side',parseInt(getComputedStyle(document.body).getPropertyValue('--side'))||300);localStorage.setItem('qc_nolist','0');}catch(e){}};
  document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up);};
 document.addEventListener('keydown',e=>{if((e.key==='l'||e.key==='L')&&!/input|select|textarea/i.test(e.target.tagName)){toggle();e.preventDefault();}});})();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER[:csf]")
    ap.add_argument("--synthseg", default=None); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh-html", action="store_true", help="only rewrite quality_control.html from data.json (after editing PAGE)")
    ap.add_argument("--cohort", default=None, help="display name of the cohort (replaces the manifest's cohort field in the page, e.g. charite-swi)")
    ap.add_argument("--on", nargs="*", default=None, help="runs switched ON at start (default: the first run only)")
    ap.add_argument("--single-file", type=int, default=0, metavar="N", help="also write quality_control_single_file.html with the N cases with most lesions embedded (0 = off, -1 = all)")
    ap.add_argument("--single-width", type=int, default=480); ap.add_argument("--single-quality", type=int, default=72)
    ap.add_argument("--info", nargs="*", default=[], metavar="NAME=TEXT", help="one-line explanation per run, shown under 'What do the models mean?' and as tooltip")
    ap.add_argument("--verdict", action="store_true", help="ONE run: colour the right image by verdict (green detected, yellow false alarm, red missed) instead of by model")
    a = ap.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    runs = []
    for i, spec in enumerate(a.run):
        name, folder = spec.split("=", 1); csf = folder.endswith(":csf"); folder = folder[:-4] if csf else folder
        runs.append(dict(name=name, folder=absolute(folder), csf=csf, colour=COLOURS[i % len(COLOURS)]))
    if a.verdict and len(runs) != 1:
        raise SystemExit("--verdict needs exactly one --run")
    cases = [c for m in a.manifest for c in json.load(open(absolute(m)))]
    if a.limit:
        cases = cases[:a.limit]
    out = absolute(a.out); os.makedirs(f"{out}/img", exist_ok=True)
    json.dump([x for x in sys.argv[1:] if x != "--refresh-html"], open(f"{out}/build_args.json", "w"), indent=0)
    if a.refresh_html:
        data = json.load(open(f"{out}/data.json"))
        if a.cohort:
            for c in data:
                c["cohort"] = a.cohort
    else:
        data = []
        for k, c in enumerate(cases, 1):
            data.append(build_patient(c, runs, absolute(a.synthseg) if a.synthseg else None, out, verdict=a.verdict))
            if a.cohort:
                data[-1]["cohort"] = a.cohort
            print(f"[{k}/{len(cases)}] {c['id']} {data[-1]['stats']}", flush=True)
        json.dump(data, open(f"{out}/data.json", "w"))
    on = set(a.on) if a.on is not None else {runs[0]["name"]}
    info = dict(i.split("=", 1) for i in a.info)
    runs_js = json.dumps([dict(name=r["name"], colour=list(r["colour"]), on=r["name"] in on, info=info.get(r["name"], "")) for r in runs])
    html = PAGE.replace("__DATASET__", a.dataset).replace("__DATA__", json.dumps(data)).replace("__RUNS__", runs_js).replace("__VERDICT__", "true" if a.verdict else "false")
    open(f"{out}/quality_control.html", "w").write(html)
    if a.single_file:
        import base64, io
        chosen = sorted(data, key=lambda c: -c["n_ref"])[:a.single_file] if a.single_file > 0 else data
        embedded, uris = [], {}
        for c in chosen:
            cid = c["id"]; w0, h0 = c["tile"]; scale = a.single_width / w0; w1, h1 = a.single_width, int(round(h0 * scale))
            img = Image.open(f"{out}/img/{cid}.jpg"); img = img.resize((w1, int(round(img.height * scale))), Image.BILINEAR)
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=a.single_quality); uris[f"img/{cid}.jpg"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            for name in ["ref"] + [r["name"] for r in runs] + ([n for n, _ in VERDICT_LAYERS] if a.verdict else []):
                png = Image.open(f"{out}/img/{cid}_{name}.png"); png = png.resize((w1, int(round(png.height * scale))), Image.NEAREST)
                buf = io.BytesIO(); png.save(buf, "PNG", optimize=True); uris[f"img/{cid}_{name}.png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            embedded.append(dict(c, tile=[w1, h1]))
        single = PAGE.replace("__DATASET__", a.dataset + " (single file)").replace("__DATA__", json.dumps(embedded)).replace("__RUNS__", runs_js).replace("__VERDICT__", "true" if a.verdict else "false")
        single = single.replace("const DATA=", "const URIS=" + json.dumps(uris) + ";const DATA=").replace("im.src=src;", "im.src=URIS[src]||src;")
        path = f"{out}/quality_control_single_file.html"; open(path, "w").write(single)
        print(f"-> {path} ({len(embedded)} cases, {os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"-> {out}/quality_control.html ({len(data)} cases, {len(runs)} models)")


if __name__ == "__main__":
    main()
