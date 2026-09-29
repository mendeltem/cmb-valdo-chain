"""Build an offline review page for microbleed candidates.

Why
---
Every automatic detection that does not touch a reference microbleed counts as a false positive,
and every reference microbleed that is not found counts as a miss. Both judgements depend on the
reference annotation being right. A human expert can tell us, candidate by candidate,

* whether a "false positive" is really a mimic (vessel, calcification, sulcal CSF, artefact) or a
  microbleed the annotators overlooked, and
* whether a "missed" reference microbleed is convincing at all.

That turns error counts into knowledge about the errors and gives an estimate of how much of the
remaining gap is annotation noise.

How (design borrowed from the WMH editor ``7_qc_edit.py``)
----------------------------------------------------------
* one HTML file, no server, images stored as PNG sprites next to it -- patient data never
  leaves the machine;
* verdicts are kept in the browser (localStorage) while working and written as one JSON file,
  either straight into a chosen folder (File System Access API, Chrome / Edge) or as a download;
* keyboard first: one key per verdict, arrow keys to move on.

A microbleed is 2-5 mm, so instead of whole-slice montages each candidate gets a zoomed crop:
seven consecutive axial slices (a vessel continues through them, a microbleed stops), plus one
coronal and one sagittal cut. Images are shown as acquired (microbleeds dark).

    python -m cmb.review.build_review \
        --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d \
        --out ergebnisse/review/a03-aniso-e60-gd --dataset VALDO --model a03-aniso-e60-gd

Blinded mode (``--blind``, for numbers that go into a paper)
-----------------------------------------------------------
The open page tells the expert what each candidate is ("false positive", "missed", network
probability, red / yellow outlines). That is fine for exploring errors but biases a verdict that is
meant to measure the quality of the reference annotation. With ``--blind``

* candidates come in random order (``--seed``) under neutral ids (c001, c002, ...);
* the page shows neither the kind nor the probability, volume or radial symmetry, and no outlines --
  only a neutral circle (8 mm across) at the candidate's position, identical for every kind;
* ``--repeat N`` shows N randomly chosen candidates a second time (intra-rater agreement);
* the key (id -> kind, probability, reference lesions touched, repeat_of) is written NEXT TO the page
  folder as ``<out>.key.json`` and is never loaded by the page, so the folder can be handed out
  without it.

``--fixed-cell`` builds the candidates from the probability maps with the reported operating point
(threshold 0.3, >= 2 mm3, giant components removed, CSF rule where a SynthSeg map exists) instead of the
per-fold binary maps, so the reviewed errors are exactly the errors behind the reported F1.

    python -m cmb.review.build_review --blind --fixed-cell --repeat 20 \
        --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d \
        --out ergebnisse/review/a03-aniso-e60-gd-blind --dataset VALDO --model a03-aniso-e60-gd
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, ROOT)
from cmb.analysis.fp_location import load_synthseg                                               # noqa: E402
from cmb.analysis.missed_lesions import majority_label                                           # noqa: E402
from cmb.transfer.evaluate import MAX_VOXELS, MIN_MM3, THRESHOLD, VOXEL_MM3                      # noqa: E402

CONNECTIVITY_26 = np.ones((3, 3, 3), bool)
VOXEL_MM = (0.5, 0.5, 1.0)             # grid spacing (x, y, z)
MARKER_MM = 4.0                        # blinded mode: radius of the neutral circle
HALF_IN_PLANE = 24                     # voxels -> 24 mm wide crops
HALF_THROUGH_PLANE = 12                # slices -> 25 mm high coronal / sagittal cuts
AXIAL_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
TILE = 144                             # pixels per tile in the sprite

# SynthSeg label -> anatomical region (same table as code/gewebekarte.py)
CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}
REGION = {}
for _l in (2, 41, 3, 42):
    REGION[_l] = "lobar"
for _l in (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60):
    REGION[_l] = "deep"
for _l in (7, 8, 16, 46, 47):
    REGION[_l] = "infratentorial"
for _l in (17, 18, 53, 54):
    REGION[_l] = "mesiotemporal"
for _l in CSF_LABELS:
    REGION[_l] = "csf"


def crop(volume: np.ndarray, centre, half) -> np.ndarray:
    """Crop ``volume`` around ``centre`` with half-widths ``half``; zero-pad at the borders."""
    out = np.zeros([2 * h + 1 for h in half], volume.dtype)
    src, dst = [], []
    for c, h, n in zip(centre, half, volume.shape):
        lo, hi = c - h, c + h + 1
        start, stop = max(lo, 0), min(hi, n)
        if start >= stop:                      # the window lies completely outside the volume
            return out                         # (e.g. the slice "centre + 3" past the last slice)
        src.append(slice(start, stop))
        dst.append(slice(start - lo, stop - lo))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def to_tile(plane: np.ndarray, resample=Image.BILINEAR) -> Image.Image:
    """2D array (values 0..255) -> square tile. ``rot90`` puts anterior / superior at the top."""
    return Image.fromarray(np.rot90(plane).astype(np.uint8)).resize((TILE, TILE), resample)


def outline(mask_plane: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask after upscaling to tile size."""
    big = np.asarray(to_tile(mask_plane.astype(np.uint8) * 255, Image.NEAREST)) > 127
    return big & ~ndimage.binary_erosion(big, iterations=1)      # one pixel wide: the lesion itself stays visible


def render_candidate(image, prediction, reference, centre, window, path_stem: str, with_outlines: bool = True) -> None:
    """Write ``<stem>_img.png`` (grey) and, unless blinded, ``<stem>_ovl.png`` (transparent, outlines only)."""
    # Window from the neighbourhood of the candidate, not from the whole brain: a microbleed in the dark
    # basal ganglia and one in bright white matter both need to be readable without touching the controls.
    local = crop(image, centre, (HALF_IN_PLANE, HALF_IN_PLANE, 3))
    local = 1.0 - local[local > 0]
    if local.size > 50:
        # half local, half whole-brain: purely local windows turn homogeneous tissue into noise
        lo = 0.5 * (float(np.percentile(local, 1)) + window[0])
        hi = 0.5 * (float(np.percentile(local, 99.5)) + window[1])
    else:
        lo, hi = window
    planes = []                        # (image plane, prediction plane, reference plane)
    for dz in AXIAL_OFFSETS:
        c = (centre[0], centre[1], centre[2] + dz)
        half = (HALF_IN_PLANE, HALF_IN_PLANE, 0)
        planes.append(tuple(crop(v, c, half)[:, :, 0] for v in (image, prediction, reference)))
    half = (HALF_IN_PLANE, 0, HALF_THROUGH_PLANE)
    planes.append(tuple(crop(v, centre, half)[:, 0, :] for v in (image, prediction, reference)))   # coronal
    half = (0, HALF_IN_PLANE, HALF_THROUGH_PLANE)
    planes.append(tuple(crop(v, centre, half)[0, :, :] for v in (image, prediction, reference)))   # sagittal

    grey = Image.new("L", (TILE * len(planes), TILE))
    overlay = np.zeros((TILE, TILE * len(planes), 4), np.uint8)
    for i, (img, pred, ref) in enumerate(planes):
        # the grid stores the INVERTED image (microbleeds bright); show it as acquired
        shown = np.where(img > 0, 1.0 - img, 0.0)
        shown = np.clip((shown - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        grey.paste(to_tile(shown), (i * TILE, 0))
        block = overlay[:, i * TILE:(i + 1) * TILE]
        block[outline(ref > 0)] = (255, 220, 0, 255)       # reference: yellow
        block[outline(pred > 0)] = (255, 60, 60, 255)      # prediction: red (drawn last, on top)
    grey.save(path_stem + "_img.png", optimize=True)
    if with_outlines:                  # blinded pages must not carry the outlines at all: they give the kind away
        Image.fromarray(overlay, "RGBA").save(path_stem + "_ovl.png", optimize=True)


def region_of(component_mask: np.ndarray, seg: np.ndarray) -> str:
    labels = seg[component_mask]
    labels = labels[labels > 0]
    if len(labels) == 0:
        return "outside"
    return REGION.get(int(np.bincount(labels).argmax()), "outside")


def final_chain(probability: np.ndarray, seg: Optional[np.ndarray]) -> np.ndarray:
    """The reported operating point: threshold 0.3, >= 2 mm3, giant components removed, CSF rule (if ``seg`` is given).

    Same semantics as ``cmb.analysis.operating_point.score_case`` at the fixed cell.
    """
    labels, n = ndimage.label(probability > THRESHOLD, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    if seg is not None:
        for c, box in enumerate(ndimage.find_objects(labels), start=1):
            if keep[c] and majority_label(labels[box] == c, seg[box]) in CSF_LABELS:
                keep[c] = False
    return keep[labels]


def candidates_of_case(case_id: str, fold_dir: str, grid_dir: str, out_dir: str,
                       fixed_cell: bool = False, blind: bool = False) -> List[Dict]:
    load = lambda p: np.asarray(nib.load(p).dataobj)
    image = load(f"{grid_dir}/{case_id}/{case_id}_image.nii.gz").astype(np.float32)
    frst = load(f"{grid_dir}/{case_id}/{case_id}_frst.nii.gz").astype(np.float32)
    reference = load(f"{grid_dir}/{case_id}/{case_id}_label.nii.gz") > 0
    proba_image = nib.load(f"{fold_dir}/{case_id}_pred_proba.nii.gz")
    probability = np.asarray(proba_image.dataobj).astype(np.float32) / 255.0
    seg_path = f"{ROOT}/dev/synthseg/{case_id}_synthseg.nii.gz"
    has_seg = os.path.exists(seg_path)
    seg = load_synthseg(seg_path, proba_image) if has_seg else np.zeros(image.shape, np.int32)
    if fixed_cell:
        prediction = final_chain(probability, seg if has_seg else None)
    else:
        prediction = load(f"{fold_dir}/{case_id}_pred.nii.gz") > 0

    brain = image > 0
    shown = 1.0 - image[brain]
    window = (float(np.percentile(shown, 1)), float(np.percentile(shown, 99.5)))

    pred_lab, n_pred = ndimage.label(prediction, structure=CONNECTIVITY_26)
    ref_lab, n_ref = ndimage.label(reference, structure=CONNECTIVITY_26)
    rows = []

    def add(kind: str, labels: np.ndarray, index: int, box, reference_ids: List[int]) -> None:
        mask = labels[box] == index
        centre = [int(round(c)) + s.start for c, s in zip(ndimage.center_of_mass(mask), box)]
        number = len(rows)
        stem = f"{case_id}_{number:03d}"
        render_candidate(image, prediction, reference, centre, window, f"{out_dir}/sprites/{stem}", with_outlines=not blind)
        rows.append(dict(
            id=stem, case=case_id, kind=kind, sprite=f"sprites/{stem}",
            probability=round(float(probability[box][mask].max()), 3),
            volume_mm3=round(float(mask.sum() * np.prod(VOXEL_MM)), 1),
            frst=round(float(frst[box][mask].max()), 3),
            region=region_of(mask, seg[box]), centre_voxel=centre,
            reference_ids=[f"{case_id}#{r}" for r in reference_ids]))      # reference lesions this candidate stands for

    for index, box in enumerate(ndimage.find_objects(pred_lab), 1):
        touched = np.unique(ref_lab[box][pred_lab[box] == index])
        touched = [int(r) for r in touched if r > 0]
        add("detected" if touched else "false_positive", pred_lab, index, box, touched)
    for index, box in enumerate(ndimage.find_objects(ref_lab), 1):
        if not (prediction[box] & (ref_lab[box] == index)).any():
            add("missed", ref_lab, index, box, [index])
    return rows


def blind_rows(rows: List[Dict], out_dir: str, seed: int, repeats: int) -> List[Dict]:
    """Shuffle, add repeats, rename ids and sprites to neutral names. Returns the full rows (the key)."""
    rng = np.random.default_rng(seed)
    rows = [dict(r, repeat_of=None) for r in rows]
    for k in rng.choice(len(rows), size=min(repeats, len(rows)), replace=False):
        rows.append(dict(rows[int(k)], repeat_of=rows[int(k)]["id"]))
    order = rng.permutation(len(rows))
    old_to_new: Dict[str, str] = {}
    out: List[Dict] = []
    for position, k in enumerate(order, start=1):
        row = dict(rows[int(k)])
        new_id = f"c{position:03d}"
        if row["repeat_of"] is None:
            old_to_new[row["id"]] = new_id
        shutil.copyfile(f"{out_dir}/{row['sprite']}_img.png", f"{out_dir}/sprites/{new_id}_img.png")
        row.update(original_id=row["id"], id=new_id, sprite=f"sprites/{new_id}")
        out.append(row)
    for row in out:                                           # repeats point to the NEW id of their original
        if row["repeat_of"] is not None:
            row["repeat_of"] = old_to_new[row["repeat_of"]]
    for name in {r["original_id"] for r in out}:              # remove the sprites with the telling names
        os.remove(f"{out_dir}/sprites/{name}_img.png")
    return out


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CMB candidate review</title>
<style>
 body{margin:0;font:14px system-ui,sans-serif;background:#16181d;color:#e6e6e6;display:flex;height:100vh}
 #list{width:330px;overflow:auto;border-right:1px solid #333;background:#1d2027}
 #list div{padding:4px 8px;cursor:pointer;border-bottom:1px solid #262a33;display:flex;gap:6px;white-space:nowrap}
 #list div.sel{background:#2f4a73} #list .v1{color:#7fd67f} #list .v2{color:#ff8a80} #list .v3{color:#ffd54f}
 #main{flex:1;padding:12px;overflow:auto} canvas{background:#000;image-rendering:auto;max-width:100%}
 .bar{margin:8px 0;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
 button{background:#2a2f3a;color:#e6e6e6;border:1px solid #444;border-radius:4px;padding:5px 9px;cursor:pointer}
 button.on{background:#3d6fb6;border-color:#5b8fd8} select{background:#2a2f3a;color:#e6e6e6;border:1px solid #444;padding:4px}
 .k{opacity:.6;font-size:12px} h2{margin:0 0 4px 0;font-size:16px} .meta{opacity:.85}
 .fp{color:#ff8a80}.ms{color:#ffd54f}.tp{color:#7fd67f}
</style></head><body>
<div id="list"></div>
<div id="main">
 <h2 id="title"></h2><div class="meta" id="meta"></div>
 <div class="bar">
  <button id="bOverlay">outlines (d)</button>
  <span class="k">brightness [ ] &nbsp; contrast - + &nbsp; reset 0</span>
  <select id="filter"><option value="all">all candidates</option><option value="false_positive">false positives</option>
   <option value="missed">missed reference</option><option value="detected">detected</option><option value="open">not yet judged</option></select>
  <span id="progress" class="k"></span>
 </div>
 <canvas id="view"></canvas>
 <div class="k">seven consecutive axial slices (centre slice framed) &middot; coronal &middot; sagittal &nbsp;|&nbsp;
  <span id="legend"><span style="color:#ff5050">red</span> = network, <span style="color:#ffdc00">yellow</span> = reference annotation</span></div>
 <div class="bar" id="verdicts"></div>
 <div class="bar" id="reasons"></div>
 <div class="bar"><input id="note" placeholder="note (optional)" style="flex:1;background:#2a2f3a;color:#e6e6e6;border:1px solid #444;padding:6px"></div>
 <div class="bar"><button id="bFolder">target folder</button><button id="bSave">save verdicts (s)</button><span id="target" class="k"></span></div>
 <div class="k">keys: 1 microbleed &middot; 2 not a microbleed &middot; 3 unsure &middot; v vessel &middot; c calcification &middot; l sulcus/CSF &middot;
  a artefact &middot; e brain edge &middot; o other &middot; &larr;/&rarr; previous/next &middot; d outlines &middot; s save</div>
</div>
<script>
const DATASET = __DATASET__, MODEL = __MODEL__, CANDIDATES = __CANDIDATES__, BLIND = __BLIND__, MARKER = __MARKER__;
const KEY = 'cmb_review_' + (BLIND ? 'blind_' : '') + DATASET + '_' + MODEL, FILE = 'review_' + (BLIND ? 'blind_' : '') + DATASET + '_' + MODEL + '.json';
const VERDICTS = [['1','microbleed'],['2','not_microbleed'],['3','unsure']];
const REASONS = [['v','vessel'],['c','calcification'],['l','sulcus_csf'],['a','artefact'],['e','brain_edge'],['o','other']];
let review = JSON.parse(localStorage.getItem(KEY) || '{}');
let state = {i:0, overlay:true, brightness:1, contrast:1, filter:'all'}, folder = null;
const img = new Image(), ovl = new Image(), canvas = document.getElementById('view'), ctx = canvas.getContext('2d');
const SCALE = 1.6;
function visible(){ return CANDIDATES.filter(c => state.filter==='all' || (state.filter==='open' ? !(review[c.id]||{}).verdict : c.kind===state.filter)); }
function current(){ const v = visible(); return v[Math.min(state.i, v.length-1)]; }
function persist(){ localStorage.setItem(KEY, JSON.stringify(review)); }
function draw(){
  if (!img.complete || !img.naturalWidth) return;
  canvas.width = img.naturalWidth*SCALE; canvas.height = img.naturalHeight*SCALE;
  ctx.filter = 'brightness(' + state.brightness + ') contrast(' + state.contrast + ')';
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height); ctx.filter = 'none';
  if (!BLIND && state.overlay && ovl.complete && ovl.naturalWidth) ctx.drawImage(ovl, 0, 0, canvas.width, canvas.height);
  const t = img.naturalHeight*SCALE;
  if (BLIND && state.overlay){ ctx.strokeStyle = '#35d0ff'; ctx.lineWidth = 1.5;      // same neutral circle for every kind of candidate
    for (let k = 0; k < Math.round(img.naturalWidth/img.naturalHeight); k++){ ctx.beginPath(); ctx.arc(k*t + t/2, t/2, t*MARKER, 0, 2*Math.PI); ctx.stroke(); } } ctx.strokeStyle = '#4da3ff'; ctx.lineWidth = 2; ctx.strokeRect(3*t+1, 1, t-2, t-2);
  ctx.fillStyle = '#9ab'; ctx.font = '12px system-ui';
  ['-3','-2','-1','0','+1','+2','+3','coronal','sagittal'].forEach((s,k) => ctx.fillText(s, k*t+5, 14));
}
function show(){
  const c = current(); if (!c) { document.getElementById('title').textContent = 'nothing to show'; return; }
  if (BLIND){
    document.getElementById('title').textContent = 'candidate ' + c.id + ' · ' + c.case;
    document.getElementById('meta').textContent = 'region ' + c.region + ' | is the structure inside the circle a microbleed?';
    img.onload = draw; img.src = c.sprite + '_img.png';
  } else {
    const kind = {false_positive:['fp','false positive (network only)'], missed:['ms','missed (reference only)'], detected:['tp','detected (both)']}[c.kind];
    document.getElementById('title').innerHTML = c.case + ' &middot; <span class="' + kind[0] + '">' + kind[1] + '</span>';
    document.getElementById('meta').textContent = 'network probability ' + c.probability + ' | volume ' + c.volume_mm3 + ' mm3 | radial symmetry ' + c.frst + ' | region ' + c.region;
    img.onload = draw; ovl.onload = draw; img.src = c.sprite + '_img.png'; ovl.src = c.sprite + '_ovl.png';
  }
  const r = review[c.id] || {};
  document.querySelectorAll('#verdicts button').forEach(b => b.classList.toggle('on', b.dataset.v === r.verdict));
  document.querySelectorAll('#reasons button').forEach(b => b.classList.toggle('on', b.dataset.v === r.reason));
  document.getElementById('note').value = r.note || '';
  document.getElementById('bOverlay').classList.toggle('on', state.overlay);
  list(); const done = CANDIDATES.filter(x => (review[x.id]||{}).verdict).length;
  document.getElementById('progress').textContent = done + ' of ' + CANDIDATES.length + ' judged';
}
function list(){
  const el = document.getElementById('list'), v = visible(); el.innerHTML = '';
  v.forEach((c,k) => { const r = review[c.id] || {}, d = document.createElement('div');
    const mark = r.verdict ? {microbleed:'<span class="v1">&#10003;</span>', not_microbleed:'<span class="v2">&#10007;</span>', unsure:'<span class="v3">?</span>'}[r.verdict] : '&middot;';
    const cls = {false_positive:'fp', missed:'ms', detected:'tp'}[c.kind];
    d.innerHTML = BLIND ? (mark + ' ' + c.id + ' <span class="k">' + c.case + '</span>')
                        : (mark + ' <span class="' + cls + '">' + c.kind.replace('_',' ') + '</span> ' + c.case + ' <span class="k">p ' + c.probability + '</span>');
    if (k === Math.min(state.i, v.length-1)) d.className = 'sel'; d.onclick = () => { state.i = k; show(); }; el.appendChild(d); });
  const s = el.querySelector('.sel'); if (s) s.scrollIntoView({block:'nearest'});
}
function setField(field, value, advance){
  const c = current(); if (!c) return; const r = review[c.id] = review[c.id] || {};
  r[field] = (r[field] === value && field !== 'verdict') ? undefined : value; r.time = new Date().toISOString();
  if (field === 'verdict' && value !== 'not_microbleed') r.reason = undefined;
  persist(); if (advance) move(1); else show();
}
function move(step){ const n = visible().length; state.i = Math.max(0, Math.min(n-1, state.i + step)); show(); }
function payload(){ return {dataset:DATASET, model:MODEL, saved:new Date().toISOString(),
  candidates: CANDIDATES.filter(c => review[c.id] && review[c.id].verdict).map(c => Object.assign({id:c.id, case:c.case, kind:c.kind,
    probability:c.probability, region:c.region, centre_voxel:c.centre_voxel}, review[c.id]))}; }
async function save(){
  const text = JSON.stringify(payload(), null, 1);
  if (folder){ try { const h = await folder.getFileHandle(FILE, {create:true}), w = await h.createWritable(); await w.write(text); await w.close();
      document.getElementById('target').textContent = 'saved ' + FILE + ' at ' + new Date().toLocaleTimeString(); return; } catch(e){ alert('could not write: ' + e.message); } }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type:'application/json'})); a.download = FILE; a.click();
  document.getElementById('target').textContent = 'downloaded ' + FILE + ' -- copy it next to this page';
}
document.getElementById('bFolder').onclick = async () => { if (!window.showDirectoryPicker) { alert('This browser cannot pick a folder; saving will download the file instead.'); return; }
  try { folder = await window.showDirectoryPicker({id:'cmb_review', mode:'readwrite'}); document.getElementById('target').textContent = 'target folder: ' + folder.name; } catch(e){} };
document.getElementById('bSave').onclick = save;
document.getElementById('bOverlay').onclick = () => { state.overlay = !state.overlay; show(); };
document.getElementById('filter').onchange = e => { state.filter = e.target.value; state.i = 0; show(); };
document.getElementById('note').onchange = e => { const c = current(); if (!c) return; (review[c.id] = review[c.id] || {}).note = e.target.value; persist(); };
VERDICTS.forEach(([k,v]) => { const b = document.createElement('button'); b.textContent = k + '  ' + v.replace('_',' '); b.dataset.v = v; b.onclick = () => setField('verdict', v, v !== 'not_microbleed'); document.getElementById('verdicts').appendChild(b); });
REASONS.forEach(([k,v]) => { const b = document.createElement('button'); b.textContent = k + '  ' + v.replace('_',' / '); b.dataset.v = v; b.onclick = () => setField('reason', v, true); document.getElementById('reasons').appendChild(b); });
window.addEventListener('keydown', e => { if (e.target.id === 'note') return; const k = e.key;
  const v = VERDICTS.find(x => x[0] === k), r = REASONS.find(x => x[0] === k);
  if (v) setField('verdict', v[1], v[1] !== 'not_microbleed');
  else if (r) { const c = current(); if (c && (review[c.id]||{}).verdict !== 'not_microbleed') setField('verdict', 'not_microbleed', false); setField('reason', r[1], true); }
  else if (k === 'ArrowRight' || k === 'ArrowDown') move(1); else if (k === 'ArrowLeft' || k === 'ArrowUp') move(-1);
  else if (k === 'd') { state.overlay = !state.overlay; show(); } else if (k === 's') save();
  else if (k === ']') { state.brightness *= 1.1; draw(); } else if (k === '[') { state.brightness /= 1.1; draw(); }
  else if (k === '+' || k === '=') { state.contrast *= 1.1; draw(); } else if (k === '-') { state.contrast /= 1.1; draw(); }
  else if (k === '0') { state.brightness = state.contrast = 1; draw(); } else return; e.preventDefault(); });
if (BLIND){
  document.querySelectorAll('#filter option').forEach(o => { if (['false_positive','missed','detected'].includes(o.value)) o.remove(); });
  document.getElementById('legend').innerHTML = '<span style="color:#35d0ff">circle</span> = position of the candidate (8 mm across); kind of candidate and network output are hidden on purpose';
  document.getElementById('bOverlay').textContent = 'circle (d)';
}
show();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="folder with fold<k>/<id>_pred.nii.gz and _pred_proba.nii.gz")
    parser.add_argument("--grid", required=True, help="folder with <id>/<id>_{image,frst,label}.nii.gz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", nargs="*", help="only these cases (smoke test)")
    parser.add_argument("--fixed-cell", action="store_true",
                        help="candidates from the probability maps at the reported operating point (0.3, >= 2 mm3, CSF rule) instead of fold<k>/<id>_pred.nii.gz")
    parser.add_argument("--blind", action="store_true", help="random order, neutral ids, no kind / probability / outlines; key written to <out>.key.json")
    parser.add_argument("--seed", type=int, default=0, help="random order and choice of repeats in blinded mode")
    parser.add_argument("--repeat", type=int, default=0, help="blinded mode: show this many candidates twice (intra-rater agreement)")
    args = parser.parse_args()
    if args.repeat and not args.blind:
        raise SystemExit("--repeat only makes sense with --blind (in the open page a repeat is recognised at once)")
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    predictions, grid, out = absolute(args.predictions), absolute(args.grid), absolute(args.out)
    os.makedirs(f"{out}/sprites", exist_ok=True)

    rows: List[Dict] = []
    cases: List[str] = []                       # every case looked at, also those without any candidate (needed for the bootstrap over cases)
    for path in sorted(glob.glob(f"{predictions}/fold*/*_pred_proba.nii.gz")):
        case_id = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        if args.cases and case_id not in args.cases:
            continue
        cases.append(case_id)
        found = candidates_of_case(case_id, os.path.dirname(path), grid, out, fixed_cell=args.fixed_cell, blind=args.blind)
        rows += found
        print(f"{case_id}: {len(found)} candidates", flush=True)

    # Most informative first: confident false positives, then missed reference lesions (largest
    # first), then the detections that were right anyway.
    order = {"false_positive": 0, "missed": 1, "detected": 2}
    counts = {k: sum(r["kind"] == k for r in rows) for k in order}
    reference_hit = len({r_id for r in rows if r["kind"] == "detected" for r_id in r["reference_ids"]})
    info = dict(dataset=args.dataset, model=args.model, predictions=predictions, fixed_cell=args.fixed_cell, counts=counts,
                reference_lesions_detected=reference_hit, cases=sorted(cases))
    if args.blind:
        key_rows = blind_rows(rows, out, args.seed, args.repeat)
        shown = [dict(id=r["id"], case=r["case"], sprite=r["sprite"], region=r["region"]) for r in key_rows]     # nothing that tells the kind
        json.dump(dict(info, seed=args.seed, repeats=args.repeat, candidates=key_rows), open(f"{out}.key.json", "w"), indent=1)
        json.dump(dict(dataset=args.dataset, model=args.model, blind=True, candidates=len(shown)), open(f"{out}/summary.json", "w"), indent=1)
    else:
        rows.sort(key=lambda r: (order[r["kind"]], -r["probability"] if r["kind"] != "missed" else -r["volume_mm3"]))
        shown = rows
        json.dump(info, open(f"{out}/summary.json", "w"), indent=1)
    in_plane_mm = (2 * HALF_IN_PLANE + 1) * VOXEL_MM[0]
    page = (PAGE.replace("__DATASET__", json.dumps(args.dataset)).replace("__MODEL__", json.dumps(args.model))
                .replace("__BLIND__", json.dumps(bool(args.blind))).replace("__MARKER__", json.dumps(round(MARKER_MM / in_plane_mm, 4)))
                .replace("__CANDIDATES__", json.dumps(shown)))
    open(f"{out}/review.html", "w", encoding="utf-8").write(page)
    print(f"wrote {out}/review.html  {counts}  reference lesions detected {reference_hit}"
          + (f"  | blinded: {len(shown)} shown ({args.repeat} repeats), key -> {out}.key.json" if args.blind else ""))


if __name__ == "__main__":
    sys.exit(main())
